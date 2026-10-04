//! `Table`: named columns in row groups, the shape of a Parquet file's
//! footer, which pipelines can scan.

use pipit_kernel::allocator::{AllocError, Allocator, DynAllocator};
use pipit_kernel::column::{ColumnView, DataType};
use pipit_kernel::row_batch::{BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::Scannable;
use pipit_kernel::vec::Vec;

use pipit_kernel::names::{Name, Names};

pub struct Table {
    names: Names,
    columns: Vec<(Name, DataType)>,
    row_groups: Vec<RowGroup>,
}

/// Rows stored together: a column of each of the table's types.
pub struct RowGroup {
    row_count: u32,
    columns: Vec<ColumnView>,
}

impl Table {
    /// A table with a column for each of `columns`' names and types, in
    /// `row_groups` of a column each, with the same number of rows. The
    /// columns' buffers are shared, not copied.
    pub fn new<A: Allocator + Clone + 'static>(
        allocator: A,
        columns: &[(&str, DataType)],
        row_groups: &[&[ColumnView]],
    ) -> Result<Table, AllocError> {
        check!(u32::try_from(columns.len()).is_ok());
        let mut groups = Vec::fixed(allocator.clone(), row_groups.len())?;
        for &views in row_groups {
            check!(views.len() == columns.len());
            let row_count = views.first().map_or(0, ColumnView::row_count);
            for (view, &(_, data_type)) in views.iter().zip(columns) {
                check!(view.data_type() == data_type && view.row_count() == row_count);
            }
            let views = Vec::fixed_from(allocator.clone(), views.iter().cloned())?;
            groups.push(RowGroup { row_count, columns: views })?;
        }
        let name_bytes = columns.iter().map(|(name, _)| name.len()).sum();
        let mut names = Names::fixed(allocator.clone(), name_bytes)?;
        let mut schema = Vec::fixed(allocator, columns.len())?;
        for &(name, data_type) in columns {
            schema.push((names.add(name)?, data_type))?;
        }
        Ok(Table { names, columns: schema, row_groups: groups })
    }

    /// The column called `name`, if any.
    #[expect(clippy::cast_possible_truncation, reason = "checked by `new`")]
    pub fn find_column(&self, name: &str) -> Option<u32> {
        let found = self.columns.iter().position(|&(column, _)| self.names.get(column) == name);
        found.map(|i| i as u32)
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

/// Where a scan of a table is: a row group, and a row in it.
pub struct ScanState {
    row_group: usize,
    row: u32,
}

/// Reads a row group at a time, in batches of up to `BATCH_ROWS_MAX` rows.
/// Nothing is copied.
impl Scannable for Table {
    type State = ScanState;

    #[expect(clippy::cast_possible_truncation, reason = "checked by `new`")]
    fn column_count(&self) -> u32 {
        self.columns.len() as u32
    }

    fn column_name(&self, column: u32) -> &str {
        self.names.get(at!(self.columns, column as usize).0)
    }

    fn column_type(&self, column: u32) -> DataType {
        at!(self.columns, column as usize).1
    }

    fn new_state(&self, _: &DynAllocator) -> Result<ScanState, AllocError> {
        Ok(ScanState { row_group: 0, row: 0 })
    }

    fn next(&self, columns: &[u32], batch: &mut RowBatch, at: &mut ScanState) -> bool {
        loop {
            let Some(row_group) = self.row_groups.get(at.row_group) else { return false };
            let rows = (row_group.row_count - at.row).min(BATCH_ROWS_MAX);
            if rows == 0 {
                *at = ScanState { row_group: at.row_group + 1, row: 0 };
                continue;
            }
            batch.reset(rows);
            for &i in columns {
                let column = at!(row_group.columns, i as usize).slice(at.row, rows);
                check!(batch.push_column(column).is_ok());
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
        let allocator = DynAllocator::new(Heap).unwrap();
        let mut state = table.new_state(&allocator).unwrap();
        let mut batch = RowBatch::new();
        let mut batches = Vec::new();
        while table.next(columns, &mut batch, &mut state) {
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
        let columns = [("a", DataType::Int64), ("b", DataType::Int64)];
        let table = Table::new(Heap, &columns, &[&first, &empty, &second]).unwrap();

        let expected = [(2048, vec![0, 0]), (952, vec![-2048, 2048]), (500, vec![-3000, 3000])];
        assert_eq!(scan(&table, &[1, 0]), expected);
    }

    #[test]
    fn counts_rows_without_columns() {
        let columns = [int64s(0..5000)];
        let table = Table::new(Heap, &[("a", DataType::Int64)], &[&columns]).unwrap();
        let counts: Vec<u32> = scan(&table, &[]).iter().map(|(rows, _)| *rows).collect();
        assert_eq!(counts, [2048, 2048, 904]);
    }

    #[test]
    #[should_panic(expected = "data_type")]
    fn checks_column_types() {
        let columns = [int64s(0..1)];
        let _ = Table::new(Heap, &[("a", DataType::Float64)], &[&columns]);
    }

    #[test]
    fn finds_columns_by_name() {
        let columns = [int64s(0..1), int64s(0..1)];
        let schema = [("ts", DataType::Int64), ("dur", DataType::Int64)];
        let table = Table::new(Heap, &schema, &[&columns]).unwrap();
        assert_eq!((table.find_column("dur"), table.find_column("name")), (Some(1), None));
        assert_eq!(table.column_name(0), "ts");
    }
}
