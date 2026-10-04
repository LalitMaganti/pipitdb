//! `Scan`: reads a row group's columns in batches.

use crate::column::ColumnView;
use crate::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};

/// Fills batches of up to `BATCH_ROWS_MAX` rows with slices of the selected
/// columns, in selection order. Nothing is copied.
pub struct Scan<'a> {
    columns: &'a [ColumnView],
    selection: &'a [u32],
    row_count: u32,
    next_row: u32,
}

impl<'a> Scan<'a> {
    /// Every column must have `row_count` rows. A selection can be empty, to
    /// count rows.
    pub fn new(columns: &'a [ColumnView], row_count: u32, selection: &'a [u32]) -> Scan<'a> {
        check!(columns.iter().all(|column| column.row_count() == row_count));
        check!(selection.len() <= BATCH_COLUMNS_MAX as usize);
        check!(selection.iter().all(|&index| (index as usize) < columns.len()));
        Scan { columns, selection, row_count, next_row: 0 }
    }

    /// Refills `batch` with the next rows. Returns false, leaving `batch`
    /// empty, once every row has been read.
    pub fn next_batch(&mut self, batch: &mut RowBatch) -> bool {
        let start = self.next_row;
        let count = (self.row_count - start).min(BATCH_ROWS_MAX);
        batch.reset(count);
        if count == 0 {
            return false;
        }
        for &index in self.selection {
            let column = at!(self.columns, index as usize).slice(start, count);
            if batch.push_column(column).is_err() {
                crate::check::check_failed(line!());
            }
        }
        self.next_row = start + count;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::DataType;

    fn column(values: impl Iterator<Item = i64>, row_count: u32) -> ColumnView {
        let mut buffer = Buffer::allocate(Heap, row_count as usize * 8).unwrap();
        for (slot, value) in buffer.as_mut_slice::<i64>().iter_mut().zip(values) {
            *slot = value;
        }
        ColumnView::new(DataType::Int64, buffer, None)
    }

    #[test]
    fn reads_selected_columns_in_batches() {
        let rows = 5000;
        let columns = [column(0.., rows), column((0..).map(|i| -i), rows)];
        let mut scan = Scan::new(&columns, rows, &[1, 0]);
        let mut batch = RowBatch::new();

        let mut start = 0;
        while scan.next_batch(&mut batch) {
            assert_eq!(batch.column_count(), 2);
            assert_eq!(batch.column(0).int64s()[0], -start);
            assert_eq!(batch.column(1).int64s()[0], start);
            start += i64::from(batch.row_count());
        }
        assert_eq!(start, 5000);
        assert_eq!(batch.row_count(), 0);
    }

    #[test]
    fn counts_rows_without_columns() {
        let columns = [column(0.., 3000)];
        let mut scan = Scan::new(&columns, 3000, &[]);
        let mut batch = RowBatch::new();
        assert!(scan.next_batch(&mut batch));
        assert_eq!((batch.row_count(), batch.column_count()), (2048, 0));
        assert!(scan.next_batch(&mut batch));
        assert_eq!(batch.row_count(), 952);
        assert!(!scan.next_batch(&mut batch));
    }
}
