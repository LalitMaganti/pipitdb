//! `Table`: named columns in row groups, the shape of a Parquet file's
//! footer, which pipelines can scan.

use pipit_kernel::allocator::{AllocError, Allocator};
use pipit_kernel::column::{ColumnView, DataType};
use pipit_kernel::context::Context;
use pipit_kernel::error::Error;
use pipit_kernel::row_batch::{BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::Scannable;
use pipit_kernel::slow_vec::SlowVec;

use pipit_kernel::names::{Name, Names};

pub struct Table {
    names: Names,
    columns: SlowVec<(Name, DataType)>,
    row_groups: SlowVec<RowGroup>,
}

/// Rows stored together, at most a batch's: a column of each of the
/// table's types. A scan reads each as one batch.
pub struct RowGroup {
    row_count: u32,
    columns: SlowVec<ColumnView>,
}

impl Table {
    /// A table with a column for each of `columns`' names and types, in
    /// `row_groups` of a column each, with the same number of rows, at most
    /// `BATCH_ROWS_MAX`. The columns' buffers are shared, not copied.
    pub fn new(
        allocator: &dyn Allocator,
        columns: &[(&str, DataType)],
        row_groups: &[&[ColumnView]],
    ) -> Result<Table, AllocError> {
        check!(u32::try_from(columns.len()).is_ok());
        let mut groups = SlowVec::fixed(allocator, row_groups.len())?;
        for &views in row_groups {
            check!(views.len() == columns.len());
            let row_count = views.first().map_or(0, ColumnView::row_count);
            check!(row_count <= BATCH_ROWS_MAX);
            for (view, &(_, data_type)) in views.iter().zip(columns) {
                check!(view.data_type() == data_type && view.row_count() == row_count);
            }
            let views = SlowVec::fixed_from(allocator, views.iter().cloned())?;
            groups.push(RowGroup { row_count, columns: views })?;
        }
        let name_bytes = columns.iter().map(|(name, _)| name.len()).sum();
        let mut names = Names::fixed(allocator, name_bytes)?;
        let mut schema = SlowVec::fixed(allocator, columns.len())?;
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

/// Where a scan of a table is: the next row group.
pub struct ScanState {
    row_group: usize,
}

/// Reads each row group as one batch. Nothing is copied.
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

    fn new_state(&self, _: &mut Context) -> Result<ScanState, Error> {
        Ok(ScanState { row_group: 0 })
    }

    fn next(
        &self,
        columns: &[u32],
        _: &mut Context,
        at: &mut ScanState,
        batch: &mut RowBatch,
    ) -> Result<bool, Error> {
        loop {
            let Some(row_group) = self.row_groups.get(at.row_group) else { return Ok(false) };
            at.row_group += 1;
            // A batch with no rows would be dropped anyway.
            if row_group.row_count == 0 {
                continue;
            }
            batch.reset(row_group.row_count);
            for &i in columns {
                check!(batch.push_column(at!(row_group.columns, i as usize).clone()).is_ok());
            }
            return Ok(true);
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
        let mut buffer = Buffer::allocate(&Heap, values.len() * 8).unwrap();
        buffer.as_mut_slice::<i64>().copy_from_slice(&values);
        ColumnView::new(&mut Context::new(&Heap), DataType::Int64, buffer, None).unwrap()
    }

    /// Each batch's row count, and its first row.
    fn scan(table: &Table, columns: &[u32]) -> Vec<(u32, Vec<i64>)> {
        let mut context = Context::new(&Heap);
        let mut state = table.new_state(&mut context).unwrap();
        let mut batch = RowBatch::new();
        let mut batches = Vec::new();
        while table.next(columns, &mut context, &mut state, &mut batch).unwrap() {
            let first = (0..batch.column_count()).map(|i| batch.column(i).int64s()[0]);
            batches.push((batch.row_count(), first.collect()));
        }
        batches
    }

    #[test]
    fn reads_each_row_group_as_a_batch() {
        let first = [int64s(0..2048), int64s((0..2048).map(|i| -i))];
        let empty = [int64s(0..0), int64s(0..0)];
        let second = [int64s(2048..2548), int64s((2048..2548).map(|i| -i))];
        let columns = [("a", DataType::Int64), ("b", DataType::Int64)];
        let table = Table::new(&Heap, &columns, &[&first, &empty, &second]).unwrap();

        let expected = [(2048, vec![0, 0]), (500, vec![-2048, 2048])];
        assert_eq!(scan(&table, &[1, 0]), expected);
    }

    #[test]
    fn counts_rows_without_columns() {
        let (first, second) = ([int64s(0..2048)], [int64s(0..904)]);
        let table = Table::new(&Heap, &[("a", DataType::Int64)], &[&first, &second]).unwrap();
        let counts: Vec<u32> = scan(&table, &[]).iter().map(|(rows, _)| *rows).collect();
        assert_eq!(counts, [2048, 904]);
    }

    #[test]
    #[should_panic(expected = "BATCH_ROWS_MAX")]
    fn checks_row_groups_fit_in_a_batch() {
        let columns = [int64s(0..2049)];
        let _ = Table::new(&Heap, &[("a", DataType::Int64)], &[&columns]);
    }

    #[test]
    #[should_panic(expected = "data_type")]
    fn checks_column_types() {
        let columns = [int64s(0..1)];
        let _ = Table::new(&Heap, &[("a", DataType::Float64)], &[&columns]);
    }

    #[test]
    fn finds_columns_by_name() {
        let columns = [int64s(0..1), int64s(0..1)];
        let schema = [("ts", DataType::Int64), ("dur", DataType::Int64)];
        let table = Table::new(&Heap, &schema, &[&columns]).unwrap();
        assert_eq!((table.find_column("dur"), table.find_column("name")), (Some(1), None));
        assert_eq!(table.column_name(0), "ts");
    }
}
