//! `Values`: a source of one row of given values, such as an aggregate's,
//! known without reading any rows.

use crate::column::{ColumnView, DataType};
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::RowBatch;
use crate::slow_vec::SlowVec;
use crate::step::Source;

/// Outputs one row: each column's `Int64`.
pub struct Values {
    pub values: SlowVec<i64>,
}

impl Source for Values {
    /// Whether the row was output.
    type State = bool;

    fn new_state(&self, _: &mut Context) -> Result<bool, Error> {
        Ok(false)
    }

    fn next(
        &self,
        context: &mut Context,
        done: &mut bool,
        batch: &mut RowBatch,
    ) -> Result<bool, Error> {
        if core::mem::replace(done, true) {
            return Ok(false);
        }
        batch.reset(1);
        for &value in self.values.iter() {
            let mut values = context.values_buffer(8)?;
            *at_mut!(values.as_mut_slice::<i64>(), 0) = value;
            let column = ColumnView::new(context, DataType::Int64, values, None)?;
            let Ok(()) = batch.push_column(column) else { failed!() };
        }
        Ok(true)
    }
}
