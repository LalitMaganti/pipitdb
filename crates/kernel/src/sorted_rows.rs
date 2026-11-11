//! `SortedRows`: rows held so a breaker can output them in another order, as
//! Perfetto's `core/exec/sorted_rows` holds them. Key values are laid out as
//! rows arrive, the output order is a list of row numbers sorted by them,
//! and rows are copied out once, in that order.

use crate::allocator::{AllocError, Allocator};
use crate::column::{ColumnView, DataType};
use crate::context::Context;
use crate::error::Error;
use crate::radix_sort::msd_radix_sort;
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::row_layout::{self, RowLayout, Word};
use crate::row_layout_sort::{SlotInWords, Sorted, sort_row_layout};
use crate::row_store::RowStore;
use crate::slow_vec::SlowVec;

/// A column to sort by, and its direction.
#[derive(Clone, Copy)]
pub struct SortKey {
    pub column: u32,
    pub descending: bool,
}

pub(crate) struct SortedRows {
    /// How each row's keys are laid out in `keys`, once the first batch says
    /// their types.
    layout: Option<RowLayout>,
    /// Each key's type, which every batch's must match.
    types: SlowVec<DataType>,
    /// By key, its column if it's a string key, whose ranks among the rows'
    /// strings are laid out only once all are in.
    strings: SlowVec<Option<u32>>,
    /// Each row's keys, one row after another, until they're sorted and
    /// nothing is decoded from them.
    keys: SlowVec<u8>,
    /// The rows themselves, but for the columns `decoded`.
    rows: RowStore,
    /// The key columns not held, as their keys hold their values: each
    /// one's number and its key's.
    decoded: SlowVec<(u32, usize)>,
    /// The rows in output order: the first `order_len`.
    order: Sorted,
    order_len: u32,
}

impl SortedRows {
    pub(crate) fn new(allocator: &dyn Allocator) -> Result<SortedRows, AllocError> {
        Ok(SortedRows {
            layout: None,
            types: SlowVec::new(allocator, 1 << 6)?,
            strings: SlowVec::new(allocator, 1 << 6)?,
            keys: SlowVec::new(allocator, 1 << 31)?,
            rows: RowStore::new(allocator)?,
            decoded: SlowVec::new(allocator, 1 << 6)?,
            order: Sorted::Held { reverse: false, count: 0 },
            order_len: 0,
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

    /// As `append`, with the rows' keys at `laid_out` as `lay_out_keys` lays
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
        let stride = self.check_keys(context.allocator(), batch, keys)?.stride() as usize;
        if self.rows.len() == 0 {
            for (k, key) in keys.iter().enumerate() {
                let data_type = *at!(self.types, k);
                let decoded = self.decoded.iter().any(|&(c, _)| c == key.column);
                if !data_type.has_bytes() && !decoded && key.column < 63 {
                    self.rows.drop_column(context, key.column, data_type)?;
                    self.decoded.push((key.column, k))?;
                }
            }
        }
        let at = self.keys.len();
        // Rows can't be added once their keys are sorted.
        check!(at == self.rows.len() as usize * stride);
        self.keys.extend_zeroed(count as usize * stride)?;
        let rows = at_mut!(self.keys, at..);
        match (laid_out, &self.layout) {
            (Some(laid_out), _) => rows.copy_from_slice(at!(laid_out, ..rows.len())),
            (None, Some(layout)) => lay_out_keys(layout, batch, keys, rows),
            (None, None) => failed!(),
        }
        self.rows.append(context, batch)
    }

    /// How keys are laid out, fixed by the first batch's keys' types, which
    /// every batch's must match, as Perfetto checks.
    pub(crate) fn check_keys(
        &mut self,
        allocator: &dyn Allocator,
        batch: &RowBatch,
        keys: &[SortKey],
    ) -> Result<&RowLayout, Error> {
        let data_type = |key: &SortKey| batch.column(key.column).data_type();
        if self.layout.is_none() {
            // Every key is laid out as nullable, as whether a column has nulls
            // can change from batch to batch.
            let mut columns = SlowVec::fixed(allocator, keys.len())?;
            for key in keys {
                self.types.push(data_type(key))?;
                let (data_type, string) = match data_type(key) {
                    DataType::Int64 => (row_layout::Type::Int64, None),
                    DataType::Float64 => (row_layout::Type::Float64, None),
                    DataType::String => (row_layout::Type::Uint32, Some(key.column)),
                };
                self.strings.push(string)?;
                let descending = key.descending;
                columns.push(row_layout::Column { data_type, nullable: true, descending })?;
            }
            self.layout = Some(RowLayout::new(allocator, &columns)?);
        }
        if keys.iter().zip(self.types.iter()).any(|(key, &t)| data_type(key) != t) {
            return Err(Error::Unsupported);
        }
        Ok(self.layout())
    }

    /// How keys are laid out, once the first batch's are checked.
    pub(crate) fn layout(&self) -> &RowLayout {
        let Some(layout) = &self.layout else { failed!() };
        layout
    }

    /// By key, its column if it's a string key, once the first batch says.
    pub(crate) fn string_columns(&self) -> &[Option<u32>] {
        &self.strings
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
        let stride = self.layout().stride() as usize;
        at!(self.keys, row as usize * stride..(row as usize + 1) * stride)
    }

    /// How many rows the order has.
    pub(crate) fn order_len(&self) -> u32 {
        self.order_len
    }

    /// Keeps only the first `count` rows of the order.
    pub(crate) fn truncate_order(&mut self, count: u32) {
        self.order_len = self.order_len.min(count);
    }

    /// Resets the order to the order rows came in, then sorts it stably by
    /// the keys. Nulls sort as in SQLite: first ascending, last descending.
    /// Strings sort by their bytes.
    pub(crate) fn sort(&mut self, context: &mut Context) -> Result<(), Error> {
        let allocator = context.allocator();
        self.order = Sorted::Held { reverse: false, count: self.rows.len() };
        self.order_len = self.rows.len();
        let Some(layout) = &self.layout else { return Ok(()) };
        self.rows.flush(context)?;
        for (k, column) in self.strings.iter().enumerate() {
            let Some(column) = *column else { continue };
            let ranks = rank_strings(allocator, &self.rows, column)?;
            let get = |row: u32| self.rows.string(column, row).map(|_| *at!(ranks, row as usize));
            RowLayout::write(layout.slot(k), self.rows.len(), get, &mut self.keys);
        }
        if layout.stride() > 0 {
            self.order =
                sort_row_layout(allocator, &mut self.keys, layout.stride(), self.order_len)?;
        }
        if self.decoded.is_empty() {
            self.keys.free();
        }
        Ok(())
    }

    /// Copies up to a batch's worth of rows into `out`, from position `at` in
    /// the order, returning how many.
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
        let (order, keys, decoded, types) = (&self.order, &self.keys, &self.decoded, &self.types);
        let mut nullable = 0_u64;
        for &(column, _) in decoded.iter() {
            nullable |= u64::from(self.rows.has_nulls(column)) << column;
        }
        let given = |context: &mut Context, column: u32| {
            let Some(&(_, k)) = decoded.iter().find(|&&(c, _)| c == column) else { failed!() };
            let Some(layout) = &self.layout else { failed!() };
            let (slot, data_type) = (layout.slot(k), *at!(types, k));
            let (stride, float) = (layout.stride() as usize, data_type == DataType::Float64);
            let in_words = order.slot_in_words(slot, float);
            let range = start as usize..end as usize;
            let mut values = context.values_buffer(rows.len() * 8)?;
            if let Some((words, in_words)) = &in_words
                && in_words.decode_all(at!(words, range.clone()), values.as_mut_slice::<i64>())
            {
                return Ok(ColumnView::new(context, data_type, values, None)?);
            }
            if in_words.is_none() && nullable >> column & 1 == 0 {
                let value_at = usize::from(slot.offset) + usize::from(slot.nullable);
                for (out, &row) in values.as_mut_slice::<i64>().iter_mut().zip(rows) {
                    let at = row as usize * stride + value_at;
                    *out = row_layout::decode_value(slot, at!(keys, at..at + 8), float);
                }
                return Ok(ColumnView::new(context, data_type, values, None)?);
            }
            let mut validity = context.small_buffer(rows.len().div_ceil(8))?;
            let (out, bits) = (values.as_mut_slice::<i64>(), validity.as_mut_slice::<u8>());
            bits.fill(0);
            let mut nulls = false;
            let mut put = |r: usize, value: Option<i64>| {
                *at_mut!(out, r) = value.unwrap_or(0);
                *at_mut!(bits, r / 8) |= u8::from(value.is_some()) << (r % 8);
                nulls |= value.is_none();
            };
            if let Some((words, in_words)) = &in_words {
                for (r, &word) in (0..).zip(at!(words, range)) {
                    put(r, in_words.decode(word));
                }
            } else {
                for (r, &row) in (0..).zip(rows) {
                    let at = row as usize * stride;
                    put(r, row_layout::decode(slot, at!(keys, at..at + stride), float));
                }
            }
            Ok(ColumnView::new(context, data_type, values, nulls.then_some(validity))?)
        };
        self.rows.gather(context, out, rows, at, given)
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
        // their bits, if their keys are only there.
        let mut decode = None;
        if let (Some(fixed), Some(layout), false) = (fixed, &self.layout, self.decoded.is_empty()) {
            let mut slots = SlowVec::fixed(context.allocator(), self.decoded.len())?;
            for &(column, k) in self.decoded.iter() {
                let float = *at!(self.types, k) == DataType::Float64;
                slots.push((
                    column,
                    SlotInWords::new(packing, varying, fixed, layout.slot(k), float),
                ))?;
            }
            decode = Some(move |column: usize, word: u64| {
                let Some((_, slot)) = slots.iter().find(|(c, _)| *c as usize == column) else {
                    failed!()
                };
                slot.decode(word).unwrap_or(0)
            });
        }
        self.rows.scatter(context, words, |word| packing.row(word), decode)
    }
}

/// Each held row's rank among the strings of `column`, a string column, of
/// the rows held: equal strings rank equally. Null rows' ranks are any.
fn rank_strings(
    allocator: &dyn Allocator,
    rows: &RowStore,
    column: u32,
) -> Result<SlowVec<u32>, Error> {
    let string = |row: &u32| rows.string(column, *row).unwrap_or_default();
    let mut order = SlowVec::fixed_from(allocator, 0..rows.len())?;
    let mut scratch = SlowVec::<u32>::zeroed(allocator, rows.len() as usize)?;
    msd_radix_sort(allocator, &mut order, &mut scratch, string)?;
    let mut ranks = SlowVec::<u32>::zeroed(allocator, rows.len() as usize)?;
    let mut rank = 0;
    for (i, row) in order.iter().enumerate() {
        if i > 0 && string(row) != string(at!(order, i - 1)) {
            rank += 1;
        }
        *at_mut!(ranks, *row as usize) = rank;
    }
    Ok(ranks)
}

/// Lays out at `rows` the keys of each row `batch` keeps, by `layout`.
pub(crate) fn lay_out_keys(
    layout: &RowLayout,
    batch: &RowBatch,
    keys: &[SortKey],
    rows: &mut [u8],
) {
    let count = batch.selection().len();
    let kept = batch.selection().kept();
    let row = |i: u32| kept.row(i);
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
