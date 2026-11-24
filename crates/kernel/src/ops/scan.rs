//! Scanning a table's rows.

use crate::column::Forms;
use crate::containers::SmallVec;
use crate::error::Error;
use crate::plan::{ColumnId, DynOp, ScanColumn, ScanOp};
use crate::query::{Builder, Columns, Relation};
use crate::scannable::DynScannable;

/// Scans tables.
pub trait Scan<'e> {
    /// The rows of `table`, of each of its columns, in order.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if the query would have more columns than a
    /// plan can.
    fn scan(&mut self, table: &'e DynScannable<'e>) -> Result<(Relation, Columns), Error>;
}

impl<'e> Scan<'e> for Builder<'e> {
    fn scan(&mut self, table: &'e DynScannable<'e>) -> Result<(Relation, Columns), Error> {
        let count = table.column_count();
        let mut reads = SmallVec::fixed(self.env, count as usize)?;
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_COLUMNS_MAX`")]
        let first = self.plan.columns.len() as ColumnId;
        for column in 0..count {
            let data_type = table.column_type(column);
            let binding = self.add_column(data_type, table.column_nullable(column))?.0;
            reads.push(ScanColumn { column, binding, forms: Forms::FLAT })?;
        }
        let node = self.add(DynOp::new(self.env, ScanOp::new(table, reads))?, [])?;
        Ok((node, Columns { first, count }))
    }
}
