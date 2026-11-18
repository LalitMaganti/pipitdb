//! `JoinOp`: an inner join of its two children on a key each, as Perfetto's
//! joins on ids are. Each row of the first child, the probe side, is joined
//! to the row of the second, the build side, whose key equals its own, if
//! there is one: N:1. The build side is read whole first, as a pipeline of
//! its own.
//!
//! So far a build key must be a sequence, such as a table's id: a key's
//! row is then the key less the first, so nothing is looked up. Any other
//! fails as `Unsupported`.

use crate::allocator::{AllocError, Allocator};
use crate::column::{DataType, Form, Forms};
use crate::context::Context;
use crate::error::Error;
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{DeriveContext, OutputProperties};
use crate::plan::{ColumnId, Given, Op, PlanNode};
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::row_store::RowStore;
use crate::slow_vec::SlowVec;
use crate::step::{DynOperator, DynSource, Operator, Progress, Step};

/// Outputs the probe side's columns, then `build_columns`.
pub struct JoinOp {
    /// The probe side's column it joins on, an `Int64`.
    pub probe_key: ColumnId,
    /// The build side's column it joins on, an `Int64`.
    pub build_key: ColumnId,
    /// The build side's columns it outputs.
    pub build_columns: SlowVec<ColumnId>,
}

impl JoinOp {
    fn is_build_column(&self, column: ColumnId) -> bool {
        column == self.build_key || self.build_columns.contains(&column)
    }
}

impl<'c> Op<'c> for JoinOp {
    /// Keeps the build columns needed.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        self.build_columns.retain(|&column| needed.is_needed(column));
        self.reads(needed);
        Pruned::Keep
    }

    /// Its keys, and the build columns it outputs, which it copies.
    fn reads(&self, reads: &mut Needed) {
        reads.need(self.probe_key);
        reads.need(self.build_key);
        for &column in self.build_columns.iter() {
            reads.need(column);
        }
    }

    /// Sequences of its keys, whose rows it finds by their numbers.
    fn accepts(&self, column: ColumnId) -> Forms {
        if column == self.probe_key || column == self.build_key {
            Forms::FLAT | Forms::SEQUENCE
        } else {
            Forms::FLAT
        }
    }

    /// The probe side's columns, as they are. The build side's it outputs
    /// are copies.
    fn passes_through(&self, column: ColumnId) -> bool {
        !self.is_build_column(column)
    }

    /// Nothing of its own: it keeps the probe side's rows whose key a build
    /// row has, and a condition above on either side's columns only drops
    /// rows it would output.
    fn give_condition(&mut self, _: &dyn Allocator, _: ColumnId) -> Result<Given, AllocError> {
        Ok(Given::Passed(None))
    }

    /// The probe side's columns and order, less some rows, then the build
    /// columns.
    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        for (input, key) in [(0, self.probe_key), (1, self.build_key)] {
            if c.input(input).type_of(key) != DataType::Int64 {
                return Err(LowerError::WrongType(key));
            }
        }
        let mut out = *c.input(0);
        out.drop_rows();
        // Its rows are kept by the build side's keys, not by the probe
        // scan's values alone.
        out.scan = None;
        for &column in self.build_columns.iter() {
            out.push_column(c.column_of(column))?;
        }
        Ok(out)
    }

    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 2);
        let (build, rows) = lowering.lower_side(node, 1)?;
        let position = |column| {
            let Some(position) = rows.position_of(column) else { failed!() };
            position
        };
        let build_columns = self.build_columns.iter().map(|&column| position(column));
        let build_columns = SlowVec::fixed_from(lowering.allocator(), build_columns)?;
        let build_key = position(self.build_key);
        lowering.lower_input(node, 0)?;
        let probe_key = lowering.query_position(self.probe_key);
        let join = Join { build, probe_key, build_key, build_columns };
        lowering.add_operator(Step::Operator(DynOperator::new(lowering.allocator(), join)?))
    }
}

/// Joins each input batch's rows to the build side's, which it drains on
/// its first input: an empty probe side never reads it. Each key's build
/// row is the key less the build side's first.
struct Join<'c> {
    /// The build side.
    build: DynSource<'c>,
    /// Where the keys are in each side's batches.
    probe_key: u32,
    build_key: u32,
    /// Where the build columns output are in the build side's batches.
    build_columns: SlowVec<u32>,
}

/// The build side's rows, once drained.
struct Built {
    /// Whether the build side is drained.
    drained: bool,
    /// The build columns output, of every build row, in order.
    rows: RowStore,
    /// The key of the first build row, and how many there are.
    first: i64,
    count: u32,
    /// By row of the batch being joined, the build row it's joined to.
    matched: SlowVec<u32>,
}

impl Join<'_> {
    /// Holds every row of the build side, whose keys must make one sequence.
    fn drain(&self, context: &mut Context, built: &mut Built) -> Result<(), Error> {
        built.drained = true;
        let mut run = self.build.start(context)?;
        let mut batch = RowBatch::new();
        while run.next(context, &mut batch)? {
            let Form::Sequence(first) = batch.column(self.build_key).form() else {
                return Err(Error::Unsupported);
            };
            let selection = batch.selection();
            let next = built.first.checked_add(i64::from(built.count));
            let follows = built.count == 0 || next == Some(first);
            if !selection.is_prefix() || !follows {
                return Err(Error::Unsupported);
            }
            if built.count == 0 {
                built.first = first;
            }
            built.count = built.count.checked_add(selection.len()).ok_or(Error::OutOfMemory)?;
            if self.build_columns.is_empty() {
                continue;
            }
            let mut held = RowBatch::new();
            held.reset(batch.row_count());
            held.push_borrowed(self.build_columns.iter().map(|&c| batch.column(c)));
            held.selection_mut().clone_from(selection);
            built.rows.append(context, &held)?;
        }
        built.rows.flush(context)
    }
}

impl Operator for Join<'_> {
    type State = Built;

    fn new_state(&self, context: &mut Context) -> Result<Built, Error> {
        let allocator = context.allocator();
        let matched =
            SlowVec::fixed_from(allocator, core::iter::repeat_n(0, BATCH_ROWS_MAX as usize))?;
        let rows = RowStore::new(allocator)?;
        Ok(Built { drained: false, rows, first: 0, count: 0, matched })
    }

    fn execute(
        &self,
        context: &mut Context,
        built: &mut Built,
        input: &RowBatch,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        if !built.drained {
            self.drain(context, built)?;
        }
        // No probe row can be joined.
        if built.count == 0 {
            return Ok(Progress::Done);
        }
        let key = input.column(self.probe_key);
        let (first, count) = (built.first, u64::from(built.count));
        let rows = input.row_count() as usize;
        let matched = at_mut!(built.matched, ..rows);
        // Rows not joined are gathered too, as any build row: those kept
        // are written below, so only those dropped already need one.
        if !input.selection().is_prefix() {
            matched.fill(0);
        }
        // Whether the probe row `row`, of key `value`, is joined, writing
        // its build row: without branching, as misses are at random.
        let mut join = |row: u16, value: i64| {
            let build_row = value.wrapping_sub(first).cast_unsigned();
            let joined = build_row < count;
            #[expect(clippy::cast_possible_truncation, reason = "below `count` if joined")]
            let build_row = if joined { build_row as u32 } else { 0 };
            *at_mut!(matched, usize::from(row)) = build_row;
            joined
        };
        output.copy_from(input);
        let selection = output.selection_mut();
        // A loop for each form, so none tests it each row.
        match key.form() {
            Form::Sequence(start) => {
                selection.retain(|row| join(row, start.wrapping_add(i64::from(row))));
            }
            Form::Flat => {
                let values = key.int64s();
                match key.validity() {
                    None => selection.retain(|row| join(row, *at!(values, usize::from(row)))),
                    Some(validity) => selection.retain(|row| {
                        // SAFETY: `row` is one of the batch's, which are the key's.
                        let valid = unsafe { validity.is_valid_unchecked(row.into()) };
                        join(row, *at!(values, usize::from(row))) & valid
                    }),
                }
            }
            _ => failed!(),
        }
        if output.selection().is_empty() || self.build_columns.is_empty() {
            return Ok(Progress::NeedInput);
        }
        let mut gathered = RowBatch::new();
        built.rows.gather(context, &mut gathered, matched, 0, |_, _| failed!())?;
        for c in 0..gathered.column_count() {
            let Ok(()) = output.push_column(gathered.column(c).clone()) else { failed!() };
        }
        Ok(Progress::NeedInput)
    }

    fn finish(&self, _: &mut Context, _: &mut Built, _: &mut RowBatch) -> Result<Progress, Error> {
        Ok(Progress::NeedInput)
    }
}
