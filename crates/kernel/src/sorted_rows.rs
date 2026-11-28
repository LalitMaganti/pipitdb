//! `SortedRows`: rows held so a breaker can output them in another order, as
//! Perfetto's `core/exec/sorted_rows` holds them. Key values are laid out as
//! rows arrive, the output order is a list of row numbers sorted by them,
//! and rows are copied out once, in that order.

use crate::allocator::AllocError;
use crate::column::{ColumnView, DataType, StringRef};
use crate::containers::{Array, BitSlice, SmallVec};
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;
use crate::radix_sort::msd_radix_sort;
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::row_blocks::RowBlocks;
use crate::row_layout::{self, RowLayout, Word};
use crate::row_layout_sort::{SlotInWords, Sorted, sort_row_layout};
use crate::row_store::RowStore;
use crate::selection::Kept;

/// A column to sort by, and its direction.
#[derive(Clone, Copy)]
pub struct SortKey {
    pub column: u32,
    pub descending: bool,
}

/// A key's column, as the first batch has it.
#[derive(Clone, Copy)]
pub(crate) struct KeyColumn {
    /// Its number in each batch.
    column: u32,
    /// Its type, which every batch's must match.
    data_type: DataType,
}

impl KeyColumn {
    /// Its number if it's a string column, whose ranks among the rows'
    /// strings are laid out only once all are in.
    pub(crate) fn string(self) -> Option<u32> {
        (self.data_type == DataType::String).then_some(self.column)
    }
}

pub struct SortedRows {
    /// How each row's keys are laid out in `keys`, once the first batch says
    /// their types.
    layout: Option<RowLayout>,
    /// Each key's column, once the first batch says.
    columns: SmallVec<KeyColumn>,
    /// Each row's keys, until they're sorted and nothing is decoded from
    /// them, once the first batch says how long they are.
    keys: Option<RowBlocks>,
    /// The rows themselves, but for the columns `decoded`.
    rows: RowStore,
    /// A bit for each key column not held, as its first key holds its
    /// values.
    decoded: u64,
    /// The rows in output order: the first `order_len`.
    order: Sorted,
    /// How many of `order` there are.
    order_len: u32,
    /// How many rows of the order `serve` has output.
    served: u32,
}

impl SortedRows {
    pub(crate) fn new(env: &Env) -> Result<SortedRows, AllocError> {
        Ok(SortedRows {
            layout: None,
            columns: SmallVec::new(env, 1 << 6)?,
            keys: None,
            rows: RowStore::new(env)?,
            decoded: 0,
            order: Sorted::Held { reverse: false, count: 0 },
            order_len: 0,
            served: 0,
        })
    }

    pub(crate) fn len(&self) -> u32 {
        self.rows.len()
    }

    /// Holds the rows `batch` keeps and their values for `keys`, which must
    /// be the same for every batch. Fails if a key's type isn't the one it
    /// had in the first batch.
    pub(crate) fn append(
        &mut self,
        context: &mut Context,
        batch: &RowBatch,
        keys: &[SortKey],
    ) -> Result<(), Error> {
        self.store(context, batch, keys, None)
    }

    /// As `append`, with the rows' keys at `laid_out` as `lay_out_keys_from` lays
    /// them out, which saves doing so again.
    pub(crate) fn append_laid_out(
        &mut self,
        context: &mut Context,
        batch: &RowBatch,
        keys: &[SortKey],
        laid_out: &[u8],
    ) -> Result<(), Error> {
        self.store(context, batch, keys, Some(laid_out))
    }

    fn store(
        &mut self,
        context: &mut Context,
        batch: &RowBatch,
        keys: &[SortKey],
        laid_out: Option<&[u8]>,
    ) -> Result<(), Error> {
        let count = batch.selection().len();
        if count == 0 {
            return Ok(());
        }
        let stride = self.check_keys(context.env(), batch, keys)?.stride() as usize;
        if self.rows.len() == 0 {
            for (k, key) in keys.iter().enumerate() {
                let data_type = at!(self.columns, k).data_type;
                let decoded = self.decoded >> key.column.min(63) & 1 != 0;
                if !data_type.has_bytes() && !decoded && key.column < 63 {
                    self.rows.drop_column(context, key.column)?;
                    self.decoded |= 1 << key.column;
                }
            }
        }
        if stride > 0 {
            if self.keys.is_none() {
                self.keys = Some(RowBlocks::new(context, stride)?);
            }
            let (Some(rows), Some(layout)) = (&mut self.keys, &self.layout) else { failed!() };
            // Rows can't be added once their keys are sorted.
            let first = rows.len();
            check!(first == self.rows.len());
            rows.extend(context, count, |row, bytes| {
                let from = row - first;
                match laid_out {
                    Some(laid_out) => {
                        let at = from as usize * stride;
                        bytes.copy_from_slice(at!(laid_out, at..at + bytes.len()));
                    }
                    None => lay_out_keys_from(layout, batch, keys, from, bytes),
                }
            })?;
        }
        self.rows.append(context, batch)
    }

    /// How keys are laid out, fixed by the first batch's keys' types, which
    /// every batch's must match, as Perfetto checks.
    pub(crate) fn check_keys(
        &mut self,
        env: &Env,
        batch: &RowBatch,
        keys: &[SortKey],
    ) -> Result<&RowLayout, Error> {
        let data_type = |key: &SortKey| batch.column(key.column).data_type();
        if self.layout.is_none() {
            // Every key is laid out as nullable, as whether a column has nulls
            // can change from batch to batch.
            let mut columns = SmallVec::fixed(env, keys.len())?;
            for key in keys {
                let (column, data_type) = (key.column, data_type(key));
                self.columns.push(KeyColumn { column, data_type })?;
                let data_type = match data_type {
                    DataType::Int64 => row_layout::Type::Int64,
                    DataType::Float64 => row_layout::Type::Float64,
                    DataType::String => row_layout::Type::Uint32,
                };
                let descending = key.descending;
                columns.push(row_layout::Column { data_type, nullable: true, descending })?;
            }
            self.layout = Some(RowLayout::new(env, &columns)?);
        }
        if keys.iter().zip(self.columns.iter()).any(|(key, c)| data_type(key) != c.data_type) {
            return Err(Error::Unsupported);
        }
        Ok(self.layout())
    }

    /// How keys are laid out, once the first batch's are checked.
    pub(crate) fn layout(&self) -> &RowLayout {
        let Some(layout) = &self.layout else { failed!() };
        layout
    }

    /// Each key's column, once the first batch says.
    pub(crate) fn key_columns(&self) -> &[KeyColumn] {
        &self.columns
    }

    /// Holds the rows packed so far, so `string` can read every row.
    pub(crate) fn flush(&mut self, context: &mut Context) -> Result<(), Error> {
        self.rows.flush(context)
    }

    /// The string of the held row `row` in column `column`, a string column,
    /// if not null. Rows must be flushed first.
    pub(crate) fn string(&self, column: u32, row: u32) -> Option<&[u8]> {
        self.rows.string(column, row)
    }

    /// The keys of the held row `row`, laid out.
    pub(crate) fn keys_of(&self, row: u32) -> &[u8] {
        let Some(keys) = &self.keys else { failed!() };
        keys.get(row)
    }

    /// How many rows the order has.
    /// Keeps only the first `count` rows of the order.
    pub(crate) fn truncate_order(&mut self, count: u32) {
        self.order_len = self.order_len.min(count);
    }

    /// Resets the order to the order rows came in, then sorts it stably by
    /// the keys. Nulls sort as in SQLite: first ascending, last descending.
    /// Strings sort by their bytes.
    pub(crate) fn sort(&mut self, context: &mut Context) -> Result<(), Error> {
        let env = context.env();
        self.order = Sorted::Held { reverse: false, count: self.rows.len() };
        self.order_len = self.rows.len();
        let (Some(layout), Some(keys)) = (&self.layout, &mut self.keys) else { return Ok(()) };
        self.rows.flush(context)?;
        let stride = layout.stride() as usize;
        for (k, column) in self.columns.iter().enumerate() {
            let Some(column) = column.string() else { continue };
            let ranks = rank_strings(env, &self.rows, column)?;
            keys.for_each_block_mut(|first, bytes| {
                let get =
                    |i: u32| Some(*at!(ranks, (first + i) as usize)).filter(|&r| r != NULL_RANK);
                #[expect(clippy::cast_possible_truncation, reason = "a block's rows")]
                let count = (bytes.len() / stride) as u32;
                RowLayout::write(layout.slot(k), count, get, bytes);
            });
        }
        self.order = sort_row_layout(env, keys)?;
        if self.decoded == 0 {
            keys.free();
        }
        Ok(())
    }

    /// Copies up to a batch's worth of rows into `out`, from position `at` in
    /// the order, returning how many.
    /// Fills `out` with the next rows of the order, returning whether any
    /// are left after them.
    pub(crate) fn serve(
        &mut self,
        context: &mut Context,
        out: &mut RowBatch,
    ) -> Result<bool, Error> {
        self.served += self.gather(context, out, self.served)?;
        Ok(self.served < self.order_len)
    }

    pub(crate) fn gather(
        &mut self,
        context: &mut Context,
        out: &mut RowBatch,
        at: u32,
    ) -> Result<u32, Error> {
        let end = self.order_len.min(at.saturating_add(BATCH_ROWS_MAX));
        let start = at.min(end);
        let mut rows = [0; BATCH_ROWS_MAX as usize];
        for (row, i) in rows.iter_mut().zip(start..end) {
            *row = self.order.row(i);
        }
        let rows = at!(rows, ..(end - start) as usize);
        let (order, keys, columns) = (&self.order, &self.keys, &self.columns);
        let mut nullable = 0_u64;
        for column in bits(self.decoded) {
            nullable |= u64::from(self.rows.has_nulls(column)) << column;
        }
        let given = |context: &mut Context, column: u32| {
            let k = first_key(columns, column);
            let Some(layout) = &self.layout else { failed!() };
            let (slot, data_type) = (layout.slot(k), at!(columns, k).data_type);
            let float = data_type == DataType::Float64;
            let in_words = order.slot_in_words(slot, float);
            let range = start as usize..end as usize;
            let mut values = context.values_buffer(rows.len() * 8)?;
            if let Some((words, in_words)) = &in_words
                && in_words.decode_all(at!(words, range.clone()), values.as_mut_slice::<i64>())
            {
                return Ok(ColumnView::new(context, data_type, values, None)?);
            }
            let Some(keys) = keys else { failed!() };
            if in_words.is_none() && nullable >> column & 1 == 0 {
                let value_at = usize::from(slot.offset) + usize::from(slot.nullable);
                for (out, &row) in values.as_mut_slice::<i64>().iter_mut().zip(rows) {
                    let value = at!(keys.get(row), value_at..value_at + 8);
                    *out = row_layout::decode_value(slot, value, float);
                }
                return Ok(ColumnView::new(context, data_type, values, None)?);
            }
            let mut validity = context.small_buffer(BitSlice::<u8>::words_for(rows.len()))?;
            let (out, bits) = (values.as_mut_slice::<i64>(), validity.as_mut_slice::<u8>());
            bits.fill(0);
            let mut nulls = false;
            let mut put = |r: usize, value: Option<i64>| {
                *at_mut!(out, r) = value.unwrap_or(0);
                BitSlice::of_mut(&mut *bits).set_if(r, value.is_some());
                nulls |= value.is_none();
            };
            if let Some((words, in_words)) = &in_words {
                for (r, &word) in (0..).zip(at!(words, range)) {
                    put(r, in_words.decode(word));
                }
            } else {
                for (r, &row) in (0..).zip(rows) {
                    put(r, row_layout::decode(slot, keys.get(row), float));
                }
            }
            Ok(ColumnView::new(context, data_type, values, nulls.then_some(validity))?)
        };
        self.rows.gather(context, out, (rows, Kept::Prefix), at, &mut { given })
    }

    /// Copies the columns `RowStore::scatter` can into the order, once
    /// sorted, so gathering them only views them. Rows still in the order
    /// they're held are viewed where they are anyway.
    pub(crate) fn scatter(&mut self, context: &mut Context) -> Result<(), Error> {
        let Sorted::Words { words, packing, varying, fixed } = &mut self.order else {
            return Ok(());
        };
        let packing = *packing;
        let words = at_mut!(words, ..self.order_len as usize);
        // Decoded columns are decoded from the words before scatter uses
        // their bits, if their keys are only there, or else from the keys,
        // each block let go of as soon as scatter is past its rows.
        let mut decode = None;
        if let (Some(layout), true) = (&self.layout, self.decoded != 0) {
            let mut slots = SmallVec::fixed(context.env(), self.decoded.count_ones() as usize)?;
            for column in bits(self.decoded) {
                let k = first_key(&self.columns, column);
                let (slot, float) =
                    (layout.slot(k), at!(self.columns, k).data_type == DataType::Float64);
                let in_words =
                    fixed.as_ref().map(|f| SlotInWords::new(packing, varying, f, slot, float));
                slots.push((column, slot, float, in_words))?;
            }
            let keys = &mut self.keys;
            decode = Some(move |column: usize, word_or_row: u64| {
                let Some((_, slot, float, in_words)) =
                    slots.iter().find(|(c, ..)| *c as usize == column)
                else {
                    failed!()
                };
                if let Some(in_words) = in_words {
                    return in_words.decode(word_or_row).unwrap_or(0);
                }
                let Some(keys) = keys.as_mut() else { failed!() };
                #[expect(clippy::cast_possible_truncation, reason = "a row held")]
                let row = word_or_row as u32;
                if row.is_multiple_of(BATCH_ROWS_MAX) {
                    keys.let_go_below(row, false);
                }
                let at = usize::from(slot.offset) + usize::from(slot.nullable);
                row_layout::decode_value(*slot, at!(keys.get(row), at..at + 8), *float)
            });
        }
        let by_word = fixed.is_some();
        self.rows.scatter(context, words, |word| packing.row(word), (by_word, decode))?;
        if bits(self.decoded).all(|column| self.rows.is_scattered(column))
            && let Some(keys) = &mut self.keys
        {
            keys.free();
        }
        Ok(())
    }
}

/// The bits set in `mask`, lowest first.
fn bits(mut mask: u64) -> impl Iterator<Item = u32> {
    core::iter::from_fn(move || {
        let bit = (mask != 0).then(|| mask.trailing_zeros());
        mask &= mask.wrapping_sub(1);
        bit
    })
}

/// The first of `columns` that is `column`.
fn first_key(columns: &[KeyColumn], column: u32) -> usize {
    let Some(k) = columns.iter().position(|c| c.column == column) else { failed!() };
    k
}

/// A null row's rank.
const NULL_RANK: u32 = u32::MAX;

/// Each held row's rank among the strings of `column`, a string column, of
/// the rows held: equal strings rank equally. Null rows rank `NULL_RANK`.
fn rank_strings(env: &Env, rows: &RowStore, column: u32) -> Result<Array<u32>, Error> {
    // Each row's string, found once: sorting reads it again at every byte.
    let found = SmallVec::fixed_from(env, (0..rows.len()).map(|row| rows.string_ref(column, row)))?;
    let string = |row: &u32| at!(found, *row as usize).map(StringRef::get);
    #[expect(clippy::cast_possible_truncation, reason = "a row held")]
    let mut order = Array::from_fn(env, rows.len() as usize, |row| row as u32)?;
    let mut scratch = Array::<u32>::zeroed(env, rows.len() as usize)?;
    msd_radix_sort(env, &mut order, &mut scratch, |row| string(row).unwrap_or_default())?;
    // The sort is done with its scratch, so the ranks go there.
    let mut ranks = scratch;
    let (mut rank, mut last) = (0, None);
    for row in order.iter() {
        let string = string(row);
        let value = string.unwrap_or_default();
        rank += u32::from(last.is_some_and(|last| last != value));
        last = Some(value);
        *at_mut!(ranks, *row as usize) = if string.is_some() { rank } else { NULL_RANK };
    }
    Ok(ranks)
}

/// Lays out at `rows` the keys of the rows `batch` keeps, by `layout`, from
/// the `from`th, as many as `rows` holds.
pub(crate) fn lay_out_keys_from(
    layout: &RowLayout,
    batch: &RowBatch,
    keys: &[SortKey],
    from: u32,
    rows: &mut [u8],
) {
    #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
    let count = (rows.len() / layout.stride().max(1) as usize) as u32;
    let kept = batch.selection().kept();
    let row = |i: u32| kept.row(from + i);
    for (k, key) in keys.iter().enumerate() {
        let column = batch.column(key.column);
        let slot = layout.slot(k);
        // Laid out by `sort`, once every string is in and ranked.
        if column.data_type() == DataType::String {
            continue;
        }
        // Integers and floats as words, so one loop lays out both.
        let (values, float) = (column.values(), column.data_type() == DataType::Float64);
        let words = values.words();
        let get = |i| {
            let row = row(i);
            (!values.is_null(row)).then(|| Word { bits: *at!(words, row as usize), float })
        };
        RowLayout::write(slot, count, get, rows);
    }
}
