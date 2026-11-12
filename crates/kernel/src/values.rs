//! `ValuesOp`: one row of given values, such as an aggregate's, known
//! without reading any rows.

use crate::column::{ColumnView, DataType};
use crate::context::Context;
use crate::error::Error;
use crate::lower::{LowerError, Lowering};
use crate::optimize::Needed;
use crate::output_properties::{DeriveContext, OutputProperties};
use crate::plan::{ColumnId, Op, PlanNode};
use crate::row_batch::RowBatch;
use crate::slow_vec::SlowVec;
use crate::step::{DynSource, Source};

/// Outputs one row: each column's `Int64`, or null.
pub struct ValuesOp {
    pub values: SlowVec<(ColumnId, Option<i64>)>,
}

impl<'c> Op<'c> for ValuesOp {
    fn derive(&self, _: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        let mut out = OutputProperties::NONE;
        for &(column, _) in self.values.iter() {
            out.push_column(column)?;
        }
        Ok(out)
    }

    fn lower(&self, _: &PlanNode<'c>, lowering: &mut Lowering<'_, 'c>) -> Result<(), LowerError> {
        let allocator = lowering.allocator();
        let values = SlowVec::fixed_from(allocator, self.values.iter().map(|&(_, value)| value))?;
        lowering.set_source(DynSource::new(allocator, Values { values })?);
        Ok(())
    }

    /// Reads none: it makes them.
    fn reads(&self, _: &mut Needed) {}
}

/// The source of a `ValuesOp`'s row.
struct Values {
    values: SlowVec<Option<i64>>,
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
            *at_mut!(values.as_mut_slice::<i64>(), 0) = value.unwrap_or(0);
            // A null's validity bit is unset.
            let validity = if value.is_some() {
                None
            } else {
                let mut validity = context.small_buffer(1)?;
                *at_mut!(validity.as_mut_slice::<u8>(), 0) = 0;
                Some(validity)
            };
            let Ok(()) =
                batch.push_column(ColumnView::new(context, DataType::Int64, values, validity)?)
            else {
                failed!();
            };
        }
        Ok(true)
    }
}
