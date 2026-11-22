//! `CrossJoinOp`: each row of its first child paired with each row of its
//! second, the build side, which is held whole first.

use crate::allocator::{AllocError, Allocator};
use crate::context::Context;
use crate::error::Error;
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{Arrangements, Column, DeriveContext, OutputProperties};
use crate::plan::{ColumnId, Given, Op, PlanNode};
use crate::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use crate::row_store::RowStore;
use crate::slow_vec::SlowVec;
use crate::step::{DynOperator, DynSource, Operator, Progress, Step};

/// Outputs the probe side's columns, then `build_columns`.
pub struct CrossJoinOp {
    /// The build side's columns it outputs.
    pub build_columns: SlowVec<ColumnId>,
}

impl<'c> Op<'c> for CrossJoinOp {
    /// Keeps the build columns needed above.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        self.build_columns.retain(|&column| needed.is_needed(column));
        self.reads(needed);
        Pruned::Keep
    }

    /// The build columns, which it copies.
    fn reads(&self, reads: &mut Needed) {
        for &column in self.build_columns.iter() {
            reads.need(column);
        }
    }

    /// A condition above can be applied to the side whose column it reads.
    fn moves_condition(&self, column: ColumnId) -> Option<u32> {
        Some(u32::from(self.build_columns.contains(&column)))
    }

    /// The probe side's columns, as they are.
    fn passes_through(&self, column: ColumnId) -> bool {
        !self.build_columns.contains(&column)
    }

    /// Nothing of its own: a condition above only drops rows it outputs.
    fn give_condition(&mut self, _: &dyn Allocator, _: ColumnId) -> Result<Given, AllocError> {
        Ok(Given::Passed(None))
    }

    /// The probe side's columns, then the build columns, whose rows come
    /// out by build row, in no order.
    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        let (mut out, build) = (*c.input(0), c.input(1));
        for &id in self.build_columns.iter() {
            let (data_type, nullable) = (build.type_of(id), build.nullable_of(id));
            out.push_column(Column { id, data_type, nullable })?;
        }
        out.arrangements = Arrangements::NONE;
        out.drop_rows();
        out.scan = None;
        Ok(out)
    }

    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        let allocator = lowering.allocator();
        let (build, rows) = lowering.lower_side(node, 1)?;
        let mut build_columns = [0; BATCH_COLUMNS_MAX as usize];
        for (position, &column) in build_columns.iter_mut().zip(self.build_columns.iter()) {
            let Some(at) = rows.position_of(column) else { failed!() };
            *position = at;
        }
        lowering.lower_input(node, 0)?;
        let count = self.build_columns.len();
        let join = CrossJoin { build, build_columns, count };
        lowering.add_operator(Step::Operator(DynOperator::new(allocator, join)?))
    }
}

/// Outputs each input batch paired with each build row in turn.
struct CrossJoin<'c> {
    /// The build side.
    build: DynSource<'c>,
    /// Where the build columns are in the build side's batches, the first
    /// `count`.
    build_columns: [u32; BATCH_COLUMNS_MAX as usize],
    count: usize,
}

/// What a run of a `CrossJoin` keeps.
struct Held {
    /// The build columns of every build row, once drained.
    rows: Option<RowStore>,
    /// The build row to pair the input batch with next.
    next: u32,
    /// For `gather`, a batch of one build row's number.
    repeated: SlowVec<u32>,
}

impl Operator for CrossJoin<'_> {
    type State = Held;

    fn new_state(&self, context: &mut Context) -> Result<Held, Error> {
        // Not an array, which would be copied into place on each run.
        let repeated = SlowVec::zeroed(context.allocator(), BATCH_ROWS_MAX as usize)?;
        Ok(Held { rows: None, next: 0, repeated })
    }

    fn execute(
        &self,
        context: &mut Context,
        held: &mut Held,
        input: &RowBatch,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        let rows = match &mut held.rows {
            Some(rows) => rows,
            rows => {
                rows.insert(drain(context, &self.build, at!(self.build_columns, ..self.count))?)
            }
        };
        // No probe row can be joined.
        if rows.len() == 0 {
            return Ok(Progress::Done);
        }
        output.copy_from(input);
        let repeated = at_mut!(held.repeated, ..input.row_count() as usize);
        repeated.fill(held.next);
        let mut build = RowBatch::new();
        rows.gather(context, &mut build, repeated, 0, &mut |_, _| failed!())?;
        for c in 0..build.column_count() {
            let Ok(()) = output.push_column(build.column(c).clone()) else { failed!() };
        }
        held.next += 1;
        if held.next < rows.len() {
            return Ok(Progress::MoreOutput);
        }
        held.next = 0;
        Ok(Progress::NeedInput)
    }

    fn finish(&self, _: &mut Context, _: &mut Held, _: &mut RowBatch) -> Result<Progress, Error> {
        Ok(Progress::NeedInput)
    }
}

/// Holds `columns` of every row `build` makes.
fn drain(context: &mut Context, build: &DynSource<'_>, columns: &[u32]) -> Result<RowStore, Error> {
    let mut rows = RowStore::new(context.allocator())?;
    let mut run = build.start(context)?;
    let mut batch = RowBatch::new();
    while run.next(context, &mut batch)? {
        let mut held = RowBatch::new();
        held.reset(batch.row_count());
        held.push_borrowed(columns.iter().map(|&c| batch.column(c)));
        held.selection_mut().clone_from(batch.selection());
        rows.append(context, &held)?;
    }
    rows.flush(context)?;
    Ok(rows)
}
