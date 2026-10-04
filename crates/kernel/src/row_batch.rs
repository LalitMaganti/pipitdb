//! `RowBatch`: up to `BATCH_ROWS_MAX` rows, as columns of the same length.

use core::mem::MaybeUninit;

use crate::column::ColumnView;
use crate::selection::Selection;

pub const BATCH_ROWS_MAX: u32 = 2048;
pub const BATCH_COLUMNS_MAX: u32 = 64;

/// The row count is set separately from the columns, so a batch can have
/// rows but no columns, e.g. for `COUNT(*)`. Its selection says which of the
/// rows are kept: whoever reads a batch reads only those.
pub struct RowBatch {
    row_count: u32,
    column_count: u32,
    // The first `column_count` are initialized.
    columns: [MaybeUninit<ColumnView>; BATCH_COLUMNS_MAX as usize],
    selection: Selection,
}

impl RowBatch {
    pub fn new() -> RowBatch {
        RowBatch {
            row_count: 0,
            column_count: 0,
            selection: Selection::all(0),
            columns: [const { MaybeUninit::uninit() }; _],
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

    pub fn columns_mut(&mut self) -> &mut [ColumnView] {
        let columns = at_mut!(self.columns, ..self.column_count as usize);
        // SAFETY: as in `column`.
        unsafe { &mut *(core::ptr::from_mut(columns) as *mut [ColumnView]) }
    }

    fn drop_columns(&mut self) {
        let count = self.column_count as usize;
        self.column_count = 0;
        let columns = at_mut!(self.columns, ..count).as_mut_ptr().cast::<ColumnView>();
        // SAFETY: the first `count` columns are initialized, and the count is
        // reset first, so each is dropped once.
        unsafe { core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(columns, count)) };
    }
}

impl Default for RowBatch {
    fn default() -> RowBatch {
        RowBatch::new()
    }
}

impl Drop for RowBatch {
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

    #[test]
    fn holds_columns_of_the_same_length() {
        let mut values = Buffer::allocate(Heap, 3 * 8).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(&[1, 2, 3]);
        let column = ColumnView::new(DataType::Int64, values, None);

        let mut batch = RowBatch::new();
        batch.reset(3);
        assert!(batch.push_column(column.clone()).is_ok());
        assert!(batch.push_column(column).is_ok());
        assert_eq!(batch.column(1).int64s(), [1, 2, 3]);

        batch.reset(0);
        assert_eq!(batch.column_count(), 0);
    }
}
