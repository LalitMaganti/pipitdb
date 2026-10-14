//! `RowStore`: the rows a breaker holds, to hand back later in any order, as
//! Perfetto's does.

use crate::allocator::{AllocError, Allocator};
use crate::column::{ColumnView, DataType, Forms};
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::selection::Kept;
use crate::slow_vec::SlowVec;

/// The bits of a row's place that say which row of its batch it is.
const ROW_BITS: u32 = 11;
const _: () = assert!(BATCH_ROWS_MAX <= 1 << ROW_BITS);

/// The most batches a store holds: a place is 32 bits.
const BATCHES_MAX: usize = 1 << (32 - ROW_BITS);

/// Rows are numbered from 0 in the order they were held.
///
/// A batch whose rows are all kept is held as it is, its columns' buffers
/// shared. One that keeps fewer is copied, its kept rows only, so the store
/// doesn't hold on to rows it dropped. Columns that aren't flat are made so.
pub(crate) struct RowStore {
    /// Each batch held, by its columns, every row of them kept.
    batches: SlowVec<SlowVec<ColumnView>>,
    /// Where each row held is: its batch, shifted up `ROW_BITS`, and its row
    /// in it.
    places: SlowVec<u32>,
}

impl RowStore {
    pub(crate) fn new(allocator: &dyn Allocator) -> Result<RowStore, AllocError> {
        Ok(RowStore {
            batches: SlowVec::new(allocator, BATCHES_MAX)?,
            places: SlowVec::new(allocator, 1 << 31)?,
        })
    }

    /// How many rows are held.
    pub(crate) fn len(&self) -> usize {
        self.places.len()
    }

    /// Holds the rows `batch` keeps. Every batch must have the same columns.
    pub(crate) fn append(&mut self, context: &mut Context, batch: &RowBatch) -> Result<(), Error> {
        let allocator = context.allocator();
        let count = batch.column_count() as usize;
        let mut columns = SlowVec::fixed(allocator, count)?;
        let rows = match batch.selection().kept() {
            Kept::None => return Ok(()),
            Kept::All => {
                for i in 0..batch.column_count() {
                    columns.push(flat(context, batch.column(i))?)?;
                }
                batch.row_count()
            }
            Kept::Select(kept) => {
                for i in 0..batch.column_count() {
                    let column = &flat(context, batch.column(i))?;
                    let rows = kept.iter().map(|&row| (column, u32::from(row)));
                    columns
                        .push(gather(context, column.data_type(), rows)?)
                        .map_err(|_| AllocError)?;
                }
                u32::try_from(kept.len()).unwrap_or(0)
            }
        };
        self.hold(columns, rows)
    }

    fn hold(&mut self, columns: SlowVec<ColumnView>, rows: u32) -> Result<(), Error> {
        let Ok(batch) = u32::try_from(self.batches.len()) else { return Err(Error::OutOfMemory) };
        self.batches.push(columns)?;
        for row in 0..rows {
            self.places.push(batch << ROW_BITS | row)?;
        }
        Ok(())
    }

    /// Fills `out` with the rows `rows` says, by number, in that order:
    /// at most a batch's.
    pub(crate) fn view(
        &self,
        context: &mut Context,
        out: &mut RowBatch,
        rows: &[u32],
    ) -> Result<(), Error> {
        check!(rows.len() <= BATCH_ROWS_MAX as usize);
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
        out.reset(rows.len() as u32);
        let Some(first) = self.batches.first() else { return Ok(()) };
        for c in 0..first.len() {
            let data_type = at!(first, c).data_type();
            let at = rows.iter().map(|&row| {
                let place = *at!(self.places, row as usize);
                let columns = at!(self.batches, (place >> ROW_BITS) as usize);
                (at!(columns, c), place & ((1 << ROW_BITS) - 1))
            });
            let column = gather(context, data_type, at)?;
            let Ok(()) = out.push_column(column) else { failed!() };
        }
        Ok(())
    }
}

/// `column`, made flat if it isn't: it's in whatever form the first op to
/// read it takes.
fn flat(context: &mut Context, column: &ColumnView) -> Result<ColumnView, AllocError> {
    let mut column = column.clone();
    column.make_in(context, Forms::FLAT)?;
    Ok(column)
}

/// A flat column of `data_type` of the rows `rows` says, each a row of a
/// flat column, in that order: at most a batch's.
fn gather<'c>(
    context: &mut Context,
    data_type: DataType,
    rows: impl ExactSizeIterator<Item = (&'c ColumnView, u32)> + Clone,
) -> Result<ColumnView, Error> {
    let len = rows.len();
    let mut values = context.values_buffer(len * 8)?;
    let mut validity = context.small_buffer(len.div_ceil(8))?;
    validity.as_mut_slice::<u8>().fill(0);
    let mut any_null = false;
    let bits = validity.as_mut_slice::<u8>();
    for (i, (column, row)) in rows.clone().enumerate() {
        let valid = !column.is_null(row);
        *at_mut!(bits, i / 8) |= u8::from(valid) << (i % 8);
        any_null |= !valid;
    }
    let validity = any_null.then_some(validity);
    if !data_type.has_bytes() {
        for (out, (column, row)) in values.as_mut_slice::<i64>().iter_mut().zip(rows) {
            *out = *at!(words(column), row as usize);
        }
        return Ok(ColumnView::new(context, data_type, values, validity)?);
    }
    let total: usize =
        rows.clone().map(|(column, row)| column.string_values().get(row as usize).len()).sum();
    let mut bytes = context.bytes_buffer(total)?;
    let (mut start, out) = (0_usize, bytes.as_mut_slice::<u8>());
    let (views, _) = values.as_mut_slice::<u32>().as_chunks_mut::<2>();
    for (view, (column, row)) in views.iter_mut().zip(rows) {
        let value = column.string_values().get(row as usize);
        at_mut!(out, start..start + value.len()).copy_from_slice(value);
        let (Ok(at), Ok(len)) = (u32::try_from(start), u32::try_from(value.len())) else {
            return Err(Error::OutOfMemory);
        };
        *view = [at, len];
        start += value.len();
    }
    Ok(ColumnView::strings(context, values, bytes, validity)?)
}

/// A flat column's values as words: integers, or floats' bits.
fn words(column: &ColumnView) -> &[i64] {
    // SAFETY: any bits are an `i64`.
    let (before, words, after) = unsafe { column.value_bytes().align_to::<i64>() };
    check!(before.is_empty() && after.is_empty());
    words
}
