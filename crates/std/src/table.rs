//! `Table`: columns in row groups, the shape of a Parquet file's footer.
//! `TableScan`: a source of a table's rows.

use pipit_kernel::allocator::{AllocError, Allocator};
use pipit_kernel::array::Array;
use pipit_kernel::column::{ColumnView, DataType};
use pipit_kernel::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::step::Source;

pub struct Table {
    types: Array<DataType>,
    row_groups: Array<RowGroup>,
}

/// Rows stored together: a column of each type in the table.
pub struct RowGroup {
    row_count: u32,
    columns: Array<ColumnView>,
}

impl Table {
    /// A table of the given row groups, each a column of each of `types`,
    /// with the same number of rows. The columns' buffers are shared.
    pub fn new<A: Allocator + Clone + 'static>(
        allocator: A,
        types: &[DataType],
        row_groups: &[&[ColumnView]],
    ) -> Result<Table, AllocError> {
        let row_groups = Array::new(allocator.clone(), row_groups.len(), |i| {
            let columns = row_groups[i];
            assert!(columns.len() == types.len(), "a row group has the wrong number of columns");
            let row_count = columns.first().map_or(0, ColumnView::row_count);
            for (column, &data_type) in columns.iter().zip(types) {
                assert!(column.data_type() == data_type, "a column has the wrong type");
                assert!(column.row_count() == row_count, "columns have different row counts");
            }
            let columns = Array::new(allocator.clone(), columns.len(), |i| Ok(columns[i].clone()))?;
            Ok(RowGroup { row_count, columns })
        })?;
        let types = Array::new(allocator, types.len(), |i| Ok(types[i]))?;
        Ok(Table { types, row_groups })
    }

    pub fn types(&self) -> &[DataType] {
        &self.types
    }

    pub fn row_groups(&self) -> &[RowGroup] {
        &self.row_groups
    }
}

impl RowGroup {
    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    pub fn columns(&self) -> &[ColumnView] {
        &self.columns
    }
}

/// Reads the selected columns of a table, in selection order, a row group at
/// a time, in batches of up to `BATCH_ROWS_MAX` rows. Nothing is copied.
pub struct TableScan<'t> {
    table: &'t Table,
    columns: &'t [u32],
}

impl<'t> TableScan<'t> {
    pub fn new(table: &'t Table, columns: &'t [u32]) -> TableScan<'t> {
        assert!(columns.len() <= BATCH_COLUMNS_MAX as usize, "too many columns");
        assert!(columns.iter().all(|&i| (i as usize) < table.types.len()), "no such column");
        TableScan { table, columns }
    }
}

/// Where a scan is: a row group, and a row in it.
pub struct ScanState {
    row_group: usize,
    row: u32,
}

impl Source for TableScan<'_> {
    type State = ScanState;

    fn new_state(&self) -> ScanState {
        ScanState { row_group: 0, row: 0 }
    }

    fn next(&self, batch: &mut RowBatch, at: &mut ScanState) -> bool {
        loop {
            let Some(row_group) = self.table.row_groups.get(at.row_group) else { return false };
            let rows = (row_group.row_count - at.row).min(BATCH_ROWS_MAX);
            if rows == 0 {
                *at = ScanState { row_group: at.row_group + 1, row: 0 };
                continue;
            }
            batch.reset(rows);
            for &i in self.columns {
                let column = row_group.columns[i as usize].slice(at.row, rows);
                assert!(batch.push_column(column).is_ok(), "checked by `new`");
            }
            at.row += rows;
            return true;
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec;
    use std::vec::Vec;

    use pipit_kernel::allocator::Heap;
    use pipit_kernel::buffer::Buffer;

    use super::*;

    fn int64s(values: impl Iterator<Item = i64>) -> ColumnView {
        let values: Vec<i64> = values.collect();
        let mut buffer = Buffer::allocate(Heap, values.len() * 8).unwrap();
        buffer.as_mut_slice::<i64>().copy_from_slice(&values);
        ColumnView::new(DataType::Int64, buffer, None)
    }

    /// Each batch's row count, and its first row.
    fn scan(table: &Table, columns: &[u32]) -> Vec<(u32, Vec<i64>)> {
        let scan = TableScan::new(table, columns);
        let mut state = scan.new_state();
        let mut batch = RowBatch::new();
        let mut batches = Vec::new();
        while scan.next(&mut batch, &mut state) {
            let first = (0..batch.column_count()).map(|i| batch.column(i).int64s()[0]);
            batches.push((batch.row_count(), first.collect()));
        }
        batches
    }

    #[test]
    fn reads_row_groups_in_batches() {
        let first = [int64s(0..3000), int64s((0..3000).map(|i| -i))];
        let empty = [int64s(0..0), int64s(0..0)];
        let second = [int64s(3000..3500), int64s((3000..3500).map(|i| -i))];
        let types = [DataType::Int64; 2];
        let table = Table::new(Heap, &types, &[&first, &empty, &second]).unwrap();

        let batches = scan(&table, &[1, 0]);
        let expected = [(2048, vec![0, 0]), (952, vec![-2048, 2048]), (500, vec![-3000, 3000])];
        assert_eq!(batches, expected);
    }

    #[test]
    fn counts_rows_without_columns() {
        let columns = [int64s(0..5000)];
        let table = Table::new(Heap, &[DataType::Int64], &[&columns]).unwrap();
        let counts: Vec<u32> = scan(&table, &[]).iter().map(|(rows, _)| *rows).collect();
        assert_eq!(counts, [2048, 2048, 904]);
    }

    #[test]
    #[should_panic(expected = "wrong type")]
    fn checks_column_types() {
        let columns = [int64s(0..1)];
        let _ = Table::new(Heap, &[DataType::Float64], &[&columns]);
    }
}
