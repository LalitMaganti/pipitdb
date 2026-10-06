//! `RowBatch`: up to `BATCH_ROWS_MAX` rows, as columns of the same length.

use core::marker::PhantomData;
use core::mem::MaybeUninit;

use crate::column::ColumnView;
use crate::selection::Selection;

pub const BATCH_ROWS_MAX: u32 = 2048;
pub const BATCH_COLUMNS_MAX: u32 = 64;

/// The row count is set separately from the columns, so a batch can have
/// rows but no columns, e.g. for `COUNT(*)`. Its selection says which of the
/// rows are kept: whoever reads a batch reads only those. Its columns can
/// borrow what they view for `'a`, which costs nothing to add or drop; a
/// clone of one is its own, to keep.
pub struct RowBatch<'a> {
    row_count: u32,
    column_count: u32,
    // The first `column_count` are initialized.
    columns: [MaybeUninit<ColumnView>; BATCH_COLUMNS_MAX as usize],
    // A bit for each column that's a copy of a view it borrows, holding no
    // reference, so never dropped.
    borrowed: u64,
    selection: Selection,
    // What the columns may borrow, and for how long.
    borrows: PhantomData<&'a ColumnView>,
}

impl<'a> RowBatch<'a> {
    pub fn new() -> RowBatch<'a> {
        RowBatch {
            row_count: 0,
            column_count: 0,
            selection: Selection::all(0),
            columns: [const { MaybeUninit::uninit() }; _],
            borrowed: 0,
            borrows: PhantomData,
        }
    }

    /// Makes an empty batch at `batch`, in place: only the counts are written,
    /// so a big batch isn't built and then copied there.
    ///
    /// # Safety
    ///
    /// `batch` must be valid for writes of a `RowBatch`, and is then one.
    pub(crate) unsafe fn init(batch: *mut RowBatch<'a>) {
        // SAFETY: upheld by the caller. The column array needs no writing, as
        // none are counted.
        unsafe {
            (&raw mut (*batch).row_count).write(0);
            (&raw mut (*batch).column_count).write(0);
            (&raw mut (*batch).borrowed).write(0);
            Selection::init(&raw mut (*batch).selection, 0);
        }
    }

    /// Empties the batch so it can be refilled with `row_count` rows, all
    /// kept. Inlined, as every source and operator calls it for each batch.
    #[inline]
    pub fn reset(&mut self, row_count: u32) {
        check!(row_count <= BATCH_ROWS_MAX);
        self.drop_columns();
        self.row_count = row_count;
        self.selection.reset(row_count);
    }

    /// Which rows are kept.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    pub fn selection_mut(&mut self) -> &mut Selection {
        &mut self.selection
    }

    /// The columns, to read, and the selection, to narrow, at once.
    pub fn columns_and_selection(&mut self) -> (&[ColumnView], &mut Selection) {
        let count = self.column_count as usize;
        let columns = at!(self.columns, ..count);
        // SAFETY: the first `column_count` columns are initialized.
        let columns = unsafe { &*(core::ptr::from_ref(columns) as *const [ColumnView]) };
        (columns, &mut self.selection)
    }

    /// Fails, giving `column` back, if the batch has `BATCH_COLUMNS_MAX`
    /// columns.
    pub fn push_column(&mut self, column: ColumnView) -> Result<(), ColumnView> {
        check!(column.row_count() == self.row_count);
        if self.column_count == BATCH_COLUMNS_MAX {
            return Err(column);
        }
        at_mut!(self.columns, self.column_count as usize).write(column);
        self.column_count += 1;
        Ok(())
    }

    /// Adds a column that borrows each of `columns`, which outlive the
    /// batch, so nothing is counted to add or drop them. Fails a check if
    /// they don't fit.
    #[inline]
    pub fn push_borrowed(&mut self, columns: impl IntoIterator<Item = &'a ColumnView>) {
        let mut columns = columns.into_iter();
        let (start, rows) = (self.column_count as usize, self.row_count);
        let mut count = start;
        for (slot, column) in at_mut!(self.columns, start..).iter_mut().zip(&mut columns) {
            check!(column.row_count() == rows);
            // SAFETY: a copy of a view that lives for `'a`, longer than the
            // batch, holding no reference of its own: `borrowed` says so,
            // so it's never dropped, and the batch only lends it out.
            slot.write(unsafe { core::ptr::read(column) });
            count += 1;
        }
        check!(columns.next().is_none());
        #[expect(clippy::cast_possible_truncation, reason = "at most 64")]
        let count = count as u32;
        // The columns added are together, after those there were.
        let added = u64::MAX.checked_shr(64 - (count - self.column_count)).unwrap_or(0);
        self.borrowed |= added << self.column_count;
        self.column_count = count;
    }

    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    pub fn column_count(&self) -> u32 {
        self.column_count
    }

    pub fn column(&self, index: u32) -> &ColumnView {
        check!(index < self.column_count);
        // SAFETY: columns below `column_count` are initialized.
        unsafe { at!(self.columns, index as usize).assume_init_ref() }
    }

    /// Replaces column `index` with `column`, of as many rows.
    pub fn set_column(&mut self, index: u32, column: ColumnView) {
        check!(index < self.column_count && column.row_count() == self.row_count);
        let slot = at_mut!(self.columns, index as usize);
        let bit = 1 << index;
        if self.borrowed & bit == 0 {
            // SAFETY: as in `column`, and it's the batch's own, to drop.
            unsafe { slot.assume_init_drop() };
        }
        self.borrowed &= !bit;
        slot.write(column);
    }

    /// Drops the batch's own columns; borrowed ones are just forgotten.
    #[inline]
    fn drop_columns(&mut self) {
        let count = self.column_count;
        let all = u64::MAX.checked_shr(64 - count).unwrap_or(0);
        let mut owned = all & !self.borrowed;
        (self.column_count, self.borrowed) = (0, 0);
        while owned != 0 {
            let i = owned.trailing_zeros() as usize;
            owned &= owned - 1;
            // SAFETY: the first `count` columns are initialized, and this one
            // is the batch's own; the count is reset first, so it's dropped
            // once.
            unsafe { at_mut!(self.columns, i).assume_init_drop() };
        }
    }
}

impl Default for RowBatch<'_> {
    fn default() -> Self {
        RowBatch::new()
    }
}

impl Drop for RowBatch<'_> {
    fn drop(&mut self) {
        self.drop_columns();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::DataType;
    use crate::context::Context;

    #[test]
    fn holds_columns_of_the_same_length() {
        let mut values = Buffer::allocate(&Heap, 3 * 8).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(&[1, 2, 3]);
        let column =
            ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, None).unwrap();

        let mut batch = RowBatch::new();
        batch.reset(3);
        assert!(batch.push_column(column.clone()).is_ok());
        assert!(batch.push_column(column).is_ok());
        assert_eq!(batch.column(1).int64s(), [1, 2, 3]);

        batch.reset(0);
        assert_eq!(batch.column_count(), 0);
    }
}
