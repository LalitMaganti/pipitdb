//! `RowStore`: the rows a breaker holds on to, to hand back later in any
//! order, as Perfetto's `core/exec/row_store` holds them.

use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType, Form};
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::selection::Kept;
use crate::slow_vec::SlowVec;
use core::ops::Range;

/// The bits of a row's place that say which row of its batch it is.
const ROW_BITS: u32 = 11;
const _: () = assert!(BATCH_ROWS_MAX <= 1 << ROW_BITS);

/// The most batches a store holds, as a place is 32 bits: with full
/// batches, about 4 billion rows.
const BATCHES_MAX: usize = 1 << (32 - ROW_BITS);

/// Rows are numbered from 0, in the order they were appended. Every column
/// appended must be flat, as planning makes the columns a breaker holds.
///
/// Holding rows never costs much more than the rows themselves. A batch
/// which keeps at least half a batch's worth of rows, or whose columns all
/// borrow what they view, is held by sharing its buffers: nothing is
/// copied. The rows any other batch keeps are packed, a full batch at a
/// time, into buffers from the run's `Context`, so a few rows never keep
/// whole buffers alive.
pub(crate) struct RowStore {
    /// Each batch held, by its columns.
    batches: SlowVec<SlowVec<ColumnView>>,
    /// Where each row held is: its batch, shifted up `ROW_BITS`, and its row
    /// in it.
    places: SlowVec<u32>,
    /// By column, the rows being packed, until there are a batch's worth or a
    /// batch is shared after them.
    packing: SlowVec<Packing>,
    /// How many rows are being packed.
    packed: u32,
}

impl RowStore {
    pub(crate) fn new(allocator: &dyn Allocator) -> Result<RowStore, AllocError> {
        Ok(RowStore {
            batches: SlowVec::new(allocator, BATCHES_MAX)?,
            places: SlowVec::new(allocator, 1 << 31)?,
            packing: SlowVec::new(allocator, 1 << 6)?,
            packed: 0,
        })
    }

    /// How many rows are held.
    pub(crate) fn len(&self) -> u32 {
        #[expect(clippy::cast_possible_truncation, reason = "at most 2^31")]
        let len = self.places.len() as u32 + self.packed;
        len
    }

    /// Holds the rows `batch` keeps. Every batch must have the same columns.
    pub(crate) fn append(&mut self, context: &mut Context, batch: &RowBatch) -> Result<(), Error> {
        let count = batch.selection().len();
        if count == 0 {
            return Ok(());
        }
        check!((0..batch.column_count()).all(|c| matches!(batch.column(c).form(), Form::Flat)));
        if self.len().checked_add(count).is_none_or(|len| len > 1 << 31) {
            return Err(Error::OutOfMemory);
        }
        if shares(batch) {
            // Rows packed before come first.
            self.flush(context)?;
            return self.hold(context, batch);
        }
        let columns = (0..batch.column_count()).map(|c| batch.column(c));
        let kept = batch.selection().kept();
        let mut done = 0;
        while done < count {
            let take = (count - done).min(BATCH_ROWS_MAX - self.packed);
            if self.packing.is_empty() {
                for column in columns.clone() {
                    self.packing.push(Packing::new(context, column.data_type())?)?;
                }
            }
            for (packing, column) in self.packing.iter_mut().zip(columns.clone()) {
                packing.append(column, kept, done..done + take)?;
            }
            (self.packed, done) = (self.packed + take, done + take);
            if self.packed == BATCH_ROWS_MAX {
                self.flush(context)?;
            }
        }
        Ok(())
    }

    /// Holds `batch` as it is, sharing its buffers.
    fn hold(&mut self, context: &mut Context, batch: &RowBatch) -> Result<(), Error> {
        let mut columns = SlowVec::fixed(context.allocator(), batch.column_count() as usize)?;
        for c in 0..batch.column_count() {
            columns.push(batch.column(c).clone())?;
        }
        let Ok(held) = u32::try_from(self.batches.len()) else { failed!() };
        if self.batches.len() == BATCHES_MAX {
            return Err(Error::OutOfMemory);
        }
        self.batches.push(columns)?;
        let kept = batch.selection().kept();
        for i in 0..batch.selection().len() {
            self.places.push(held << ROW_BITS | kept.row(i))?;
        }
        Ok(())
    }

    /// Holds the rows packed so far as one batch, so every row held is in a
    /// batch.
    pub(crate) fn flush(&mut self, context: &mut Context) -> Result<(), Error> {
        if self.packed == 0 {
            return Ok(());
        }
        let mut batch = RowBatch::new();
        batch.reset(self.packed);
        for packing in self.packing.iter() {
            let Ok(()) = batch.push_column(packing.take(context)?) else { failed!() };
        }
        while self.packing.pop().is_some() {}
        self.packed = 0;
        self.hold(context, &batch)
    }

    /// The string of the held row `row` in column `column`, a string column,
    /// if not null. Rows packed must be flushed first.
    pub(crate) fn string(&self, column: u32, row: u32) -> Option<&[u8]> {
        check!(self.packed == 0);
        let (held, row) = self.find(row);
        let values = at!(held, column as usize).values();
        (!values.is_null(row)).then(|| values.strings().get(row as usize))
    }

    /// Fills `out` with up to a batch's worth of `rows`, in that order and
    /// repeats included, returning how many. Rows held one after another in
    /// one batch are viewed where they are held; any others are copied into
    /// buffers from `context`.
    pub(crate) fn gather(
        &mut self,
        context: &mut Context,
        out: &mut RowBatch,
        rows: &[u32],
    ) -> Result<u32, Error> {
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
        let count = rows.len().min(BATCH_ROWS_MAX as usize) as u32;
        let rows = at!(rows, ..count as usize);
        out.reset(count);
        let Some(&first) = rows.first() else { return Ok(0) };
        self.flush(context)?;
        let place = |row: u32| *at!(self.places, row as usize);
        let (held, start) = self.find(first);
        // Rows from `start` on in one batch, not running past its last.
        let run = start + count <= 1 << ROW_BITS
            && (0..count)
                .zip(rows)
                .all(|(r, &row)| row == first + r && place(row) == place(first) + r);
        if run {
            for column in held.iter() {
                let Ok(()) = out.push_column(column.slice(start, count)) else { failed!() };
            }
            return Ok(count);
        }
        for c in 0..held.len() {
            let data_type = at!(held, c).data_type();
            let source = |r: usize| {
                let (held, row) = self.find(*at!(rows, r));
                (at!(held, c), row)
            };
            let column = gather(context, data_type, count, source)?;
            let Ok(()) = out.push_column(column) else { failed!() };
        }
        Ok(count)
    }

    /// The columns of the batch held row `row` is in, and its row there.
    fn find(&self, row: u32) -> (&SlowVec<ColumnView>, u32) {
        let place = *at!(self.places, row as usize);
        (at!(self.batches, (place >> ROW_BITS) as usize), place & ((1 << ROW_BITS) - 1))
    }
}

/// Whether `batch` is held by sharing its buffers rather than packed: if it
/// keeps at least half a batch's worth of rows, or its columns all borrow
/// what they view.
fn shares(batch: &RowBatch) -> bool {
    batch.selection().len() * 2 >= BATCH_ROWS_MAX
        || (0..batch.column_count()).all(|c| batch.is_borrowed(c))
}

/// A column being packed: its values, up to a batch's, and nulls.
struct Packing {
    data_type: DataType,
    /// Each row's value: a word, or a string's view into `bytes`.
    values: Buffer,
    validity: Buffer,
    nulls: bool,
    /// The strings' bytes, for a string column.
    bytes: Option<SlowVec<u8>>,
    /// How many rows are packed.
    rows: u32,
}

impl Packing {
    fn new(context: &mut Context, data_type: DataType) -> Result<Packing, AllocError> {
        let values = context.values_buffer(BATCH_ROWS_MAX as usize * 8)?;
        let mut validity = context.small_buffer(BATCH_ROWS_MAX.div_ceil(8) as usize)?;
        validity.as_mut_slice::<u8>().fill(0);
        let bytes = if data_type.has_bytes() {
            Some(SlowVec::new(context.allocator(), 1 << 31)?)
        } else {
            None
        };
        Ok(Packing { data_type, values, validity, nulls: false, bytes, rows: 0 })
    }

    /// Packs `column`'s rows `rows`, after those packed.
    fn append(&mut self, column: &ColumnView, kept: Kept, rows: Range<u32>) -> Result<(), Error> {
        let values = column.values();
        for i in rows {
            let row = kept.row(i);
            let at = self.rows as usize;
            let valid = !values.is_null(row);
            *at_mut!(self.validity.as_mut_slice::<u8>(), at / 8) |= u8::from(valid) << (at % 8);
            self.nulls |= !valid;
            match &mut self.bytes {
                None => {
                    *at_mut!(self.values.as_mut_slice::<i64>(), at) =
                        *at!(values.words(), row as usize);
                }
                Some(bytes) => {
                    let string = values.strings().get(row as usize);
                    let view = view(bytes.len(), string.len())?;
                    bytes.extend_from_slice(string)?;
                    at_mut!(self.values.as_mut_slice::<u32>(), 2 * at..2 * at + 2)
                        .copy_from_slice(&view);
                }
            }
            self.rows += 1;
        }
        Ok(())
    }

    /// The rows packed, as a flat column.
    fn take(&self, context: &mut Context) -> Result<ColumnView, Error> {
        let values = self.values.clone();
        let validity = self.nulls.then(|| self.validity.clone());
        let column = match &self.bytes {
            None => ColumnView::new(context, self.data_type, values, validity)?,
            Some(packed) => {
                let mut bytes = context.bytes_buffer(packed.len())?;
                bytes.as_mut_slice::<u8>().copy_from_slice(packed);
                ColumnView::strings(context, values, bytes, validity)?
            }
        };
        Ok(column.slice(0, self.rows))
    }
}

/// A string column's view of a string of `len` bytes from `start`.
fn view(start: usize, len: usize) -> Result<[u32; 2], Error> {
    let (Ok(start), Ok(len)) = (u32::try_from(start), u32::try_from(len)) else {
        return Err(Error::OutOfMemory);
    };
    Ok([start, len])
}

/// A flat column of `count` rows of `data_type`, row `r` row `source(r).1`
/// of the flat column `source(r).0`.
fn gather<'c>(
    context: &mut Context,
    data_type: DataType,
    count: u32,
    source: impl Fn(usize) -> (&'c ColumnView, u32),
) -> Result<ColumnView, Error> {
    let rows = count as usize;
    let mut values = context.values_buffer(rows * 8)?;
    let mut validity = context.small_buffer(rows.div_ceil(8))?;
    let bits = validity.as_mut_slice::<u8>();
    bits.fill(0);
    let (mut nulls, mut total) = (false, 0);
    let strings = data_type.has_bytes();
    // Each row's validity, and its word or its string's length.
    for (r, out) in values.as_mut_slice::<i64>().iter_mut().enumerate() {
        let (column, row) = source(r);
        let values = column.values();
        let valid = !values.is_null(row);
        *at_mut!(bits, r / 8) |= u8::from(valid) << (r % 8);
        nulls |= !valid;
        if strings {
            total += values.strings().get(row as usize).len();
        } else {
            *out = *at!(values.words(), row as usize);
        }
    }
    let validity = nulls.then_some(validity);
    if !strings {
        return Ok(ColumnView::new(context, data_type, values, validity)?);
    }
    let mut bytes = context.bytes_buffer(total)?;
    let out = bytes.as_mut_slice::<u8>();
    let mut start = 0;
    let (views, _) = values.as_mut_slice::<u32>().as_chunks_mut::<2>();
    for (r, out_view) in views.iter_mut().enumerate() {
        let (column, row) = source(r);
        let string = column.values().strings().get(row as usize);
        at_mut!(out, start..start + string.len()).copy_from_slice(string);
        *out_view = view(start, string.len())?;
        start += string.len();
    }
    Ok(ColumnView::strings(context, values, bytes, validity)?)
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;

    /// The value row `row` of the batch from `base` holds, `None` if null.
    fn value(base: i64, row: u32) -> Option<i64> {
        (row % 7 != 3).then_some(base + i64::from(row))
    }

    /// A batch of a full batch's rows, an integer and a string column, of
    /// `value`s, keeping the rows `keep` says.
    fn batch(context: &mut Context, base: i64, keep: impl FnMut(u16) -> bool) -> RowBatch<'static> {
        let rows = BATCH_ROWS_MAX;
        let mut ints = context.values_buffer(rows as usize * 8).unwrap();
        let mut validity = context.small_buffer(rows.div_ceil(8) as usize).unwrap();
        validity.as_mut_slice::<u8>().fill(0);
        let mut text = String::new();
        let mut views = context.values_buffer(rows as usize * 8).unwrap();
        for row in 0..rows {
            let value = value(base, row);
            ints.as_mut_slice::<i64>()[row as usize] = value.unwrap_or(0);
            validity.as_mut_slice::<u8>()[row as usize / 8] |=
                u8::from(value.is_some()) << (row % 8);
            let string = format!("v{}", value.unwrap_or(0));
            let start = u32::try_from(text.len()).unwrap();
            views.as_mut_slice::<u32>()[2 * row as usize..2 * row as usize + 2]
                .copy_from_slice(&[start, u32::try_from(string.len()).unwrap()]);
            text.push_str(&string);
        }
        let mut bytes = context.bytes_buffer(text.len()).unwrap();
        bytes.as_mut_slice::<u8>().copy_from_slice(text.as_bytes());
        let ints = ColumnView::new(context, DataType::Int64, ints, Some(validity.clone())).unwrap();
        let strings = ColumnView::strings(context, views, bytes, Some(validity)).unwrap();
        let mut batch = RowBatch::new();
        batch.reset(rows);
        batch.push_column(ints).ok().unwrap();
        batch.push_column(strings).ok().unwrap();
        batch.selection_mut().retain(keep);
        batch
    }

    /// The rows of `out`, as `value`s and their strings.
    fn read(out: &RowBatch) -> Vec<(Option<i64>, Vec<u8>)> {
        let (ints, strings) = (out.column(0), out.column(1));
        (0..out.row_count())
            .map(|r| {
                let int = (!ints.is_null(r)).then(|| ints.int64s()[r as usize]);
                (int, strings.string_values().get(r as usize).to_vec())
            })
            .collect()
    }

    /// Which of a batch's rows to keep.
    type Keep = fn(u16) -> bool;

    #[test]
    fn hands_back_rows_shared_and_packed_in_any_order() {
        let mut context = Context::new(&Heap);
        let mut store = RowStore::new(&Heap).unwrap();
        // Shared, packed, shared after packed rows, and packed again.
        let keeps: [(i64, Keep); 4] = [
            (0, |_| true),
            (10_000, |row| row % 10 == 0),
            (20_000, |row| row < 1500),
            (30_000, |row| row % 30 == 1),
        ];
        let mut expected = Vec::new();
        let mut first = None;
        for (base, keep) in keeps {
            let batch = batch(&mut context, base, keep);
            first.get_or_insert_with(|| batch.column(0).clone());
            for row in (0..BATCH_ROWS_MAX).filter(|&row| keep(u16::try_from(row).unwrap())) {
                let value = value(base, row);
                expected.push((value, format!("v{}", value.unwrap_or(0)).into_bytes()));
            }
            store.append(&mut context, &batch).unwrap();
        }
        assert_eq!(store.len() as usize, expected.len());

        let mut out = RowBatch::new();
        let all: Vec<u32> = (0..store.len()).collect();
        let mut got = Vec::new();
        for chunk in all.chunks(BATCH_ROWS_MAX as usize) {
            assert_eq!(store.gather(&mut context, &mut out, chunk).unwrap() as usize, chunk.len());
            if chunk[0] == 0 {
                // The first batch's rows, in order, are viewed where held.
                assert!(out.column(0).shares_values(first.as_ref().unwrap()));
            }
            got.extend(read(&out));
        }
        assert_eq!(got, expected);

        let backwards: Vec<u32> = (0..store.len()).rev().step_by(3).collect();
        let mut got = Vec::new();
        for chunk in backwards.chunks(BATCH_ROWS_MAX as usize) {
            store.gather(&mut context, &mut out, chunk).unwrap();
            got.extend(read(&out));
        }
        let expected: Vec<_> = backwards.iter().map(|&r| expected[r as usize].clone()).collect();
        assert_eq!(got, expected);
    }
}
