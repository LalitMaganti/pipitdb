//! `JoinOp`: a join of its two children on a key each, as Perfetto's joins
//! on ids are. Each row of one child, the probe side, is joined to the row
//! of the other, the build side, whose key equals its own, if there is one:
//! N:1. For an inner join planning chooses which is which. The build side is
//! read whole first, as a pipeline of its own. `JoinKind` says what's output.
//!
//! So far a build key must be a sequence, such as a table's id: a key's
//! row is then the key less the first, so nothing is looked up. Any other
//! fails as `Unsupported`.

use crate::allocator::{AllocError, Allocator};
use crate::column::{ColumnView, DataType, Form, Forms};
use crate::context::Context;
use crate::eight;
use crate::error::Error;
use crate::estimate::Estimate;
use crate::filter::{self, Range};
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{DeriveContext, OutputProperties};
use crate::plan::{ColumnId, Given, Op, PlanNode};
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::row_store::RowStore;
use crate::slow_vec::SlowVec;
use crate::step::{DynOperator, DynSource, Operator, Progress, Step};

/// What a join outputs of its probe side's rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JoinKind {
    /// Those whose key a build row has, joined to it.
    Inner,
    /// Every one, joined to the build row with its key, or else to nulls.
    Left,
    /// Those whose key a build row has, alone: `key IN (build keys)`.
    Semi,
    /// Those whose key no build row has, alone, with `NOT IN`'s nulls: a
    /// null key is kept only if the build side is empty, and a null build
    /// key would keep none.
    Anti,
}

/// Outputs the probe side's columns, then the build side's.
pub struct JoinOp {
    /// What it outputs.
    kind: JoinKind,
    /// Each child's column it joins on, an `Int64`.
    keys: [ColumnId; 2],
    /// Each child's columns it outputs: the build side's, none, for a semi
    /// or anti join.
    columns: [SlowVec<ColumnId>; 2],
    /// Which child is the build side: the second, unless planning chooses
    /// the first.
    build: usize,
}

impl JoinOp {
    /// Joins its children on `keys`, outputting `columns` of each, as
    /// `kind` says: for a semi or anti join, the first's. Only an inner
    /// join may build the first.
    pub fn new(kind: JoinKind, keys: [ColumnId; 2], columns: [SlowVec<ColumnId>; 2]) -> JoinOp {
        let outputs_second = matches!(kind, JoinKind::Inner | JoinKind::Left);
        check!(outputs_second || at!(columns, 1).is_empty());
        JoinOp { kind, keys, columns, build: 1 }
    }

    fn probe(&self) -> usize {
        1 - self.build
    }

    fn is_build_column(&self, column: ColumnId) -> bool {
        column == *at!(self.keys, self.build) || at!(self.columns, self.build).contains(&column)
    }
}

impl<'c> Op<'c> for JoinOp {
    /// Keeps the columns needed.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        for columns in &mut self.columns {
            columns.retain(|&column| needed.is_needed(column));
        }
        self.reads(needed);
        Pruned::Keep
    }

    /// Its keys, and the build columns it outputs, which it copies.
    fn reads(&self, reads: &mut Needed) {
        for &key in &self.keys {
            reads.need(key);
        }
        for &column in at!(self.columns, self.build).iter() {
            reads.need(column);
        }
    }

    /// Sequences of its keys, whose rows it finds by their numbers.
    fn accepts(&self, column: ColumnId) -> Forms {
        if self.keys.contains(&column) { Forms::FLAT | Forms::SEQUENCE } else { Forms::FLAT }
    }

    /// The probe side's columns, as they are. The build side's it outputs
    /// are copies.
    fn passes_through(&self, column: ColumnId) -> bool {
        !self.is_build_column(column)
    }

    /// Nothing of its own: it keeps the probe side's rows whose key a build
    /// row has, and a condition above on either side's columns only drops
    /// rows it would output.
    ///
    /// Not a left join's build columns: skipping build rows there would make
    /// a row that has a match one that hasn't, with nulls that pass a test
    /// such as `IS NULL`.
    fn give_condition(&mut self, _: &dyn Allocator, column: ColumnId) -> Result<Given, AllocError> {
        if self.kind == JoinKind::Left && self.is_build_column(column) {
            return Ok(Given::Stop);
        }
        Ok(Given::Passed(None))
    }

    /// Builds a side whose key is unique, as an N:1 join must: one whose key
    /// is a sequence first, as it's held without being copied, then the
    /// smaller. With no such side, the second, which fails until 1:N joins.
    fn choose(&mut self, inputs: &[Estimate]) {
        // Only an inner join's sides are the same either way round.
        if self.kind != JoinKind::Inner {
            return;
        }
        let &[first, second] = inputs else { failed!() };
        // A better key, then fewer rows, with unknown as many as can be.
        let rank = |input: &Estimate, key| (input.key(key), u64::MAX - input.rows.estimated);
        let [left, right] = self.keys;
        self.build = usize::from(rank(&first, left) <= rank(&second, right));
    }

    /// At most the probe side's rows, each joined to one build row, and its
    /// keys, less any sequence, as rows are dropped.
    fn estimate(&self, inputs: &[Estimate]) -> Option<Estimate> {
        // Not the build side's keys: one build row may be joined to many.
        let mut out = at!(inputs, self.probe()).after_filters();
        match self.kind {
            // A left join keeps every row, in order.
            JoinKind::Left => {}
            JoinKind::Inner => out.demote_ids(),
            JoinKind::Semi | JoinKind::Anti => {
                out.demote_ids();
                out.keep_some();
            }
        }
        Some(out)
    }

    /// The probe side's columns and order, less some rows, then the build
    /// columns.
    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        for (input, &key) in (0..).zip(&self.keys) {
            if c.input(input).type_of(key) != DataType::Int64 {
                return Err(LowerError::WrongType(key));
            }
        }
        #[expect(clippy::cast_possible_truncation, reason = "0 or 1")]
        let mut out = *c.input(self.probe() as u32);
        if self.kind != JoinKind::Left {
            out.drop_rows();
        }
        // Its rows are kept by the build side's keys, not by the probe
        // scan's values alone.
        out.scan = None;
        for &column in at!(self.columns, self.build).iter() {
            let mut column = c.column_of(column);
            column.nullable |= self.kind == JoinKind::Left;
            out.push_column(column)?;
        }
        Ok(out)
    }

    #[expect(clippy::cast_possible_truncation, reason = "0 or 1")]
    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 2);
        let (build, rows) = lowering.lower_side(node, self.build as u32)?;
        let position = |column| {
            let Some(position) = rows.position_of(column) else { failed!() };
            position
        };
        let build_columns = at!(self.columns, self.build).iter().map(|&column| position(column));
        let build_columns = SlowVec::fixed_from(lowering.allocator(), build_columns)?;
        let build_key = position(*at!(self.keys, self.build));
        let types = at!(self.columns, self.build).iter().map(|&column| rows.type_of(column));
        let build_types = SlowVec::fixed_from(lowering.allocator(), types)?;
        lowering.lower_input(node, self.probe() as u32)?;
        let probe_key = lowering.query_position(*at!(self.keys, self.probe()));
        let (kind, allocator) = (self.kind, lowering.allocator());
        let join = Join { kind, build, probe_key, build_key, build_columns, build_types };
        lowering.add_operator(Step::Operator(DynOperator::new(allocator, join)?))
    }
}

/// Joins each input batch's rows to the build side's, which it drains on
/// its first input: an empty probe side never reads it. Each key's build
/// row is the key less the build side's first.
struct Join<'c> {
    /// What it outputs.
    kind: JoinKind,
    /// The build side.
    build: DynSource<'c>,
    /// Where the keys are in each side's batches.
    probe_key: u32,
    build_key: u32,
    /// Where the build columns output are in the build side's batches, and
    /// their types.
    build_columns: SlowVec<u32>,
    build_types: SlowVec<DataType>,
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
    /// By row of the batch being joined, the build row it's joined to, and
    /// for a left join, a bit for whether it has none.
    matched: SlowVec<u32>,
    missed: SlowVec<u8>,
}

impl Join<'_> {
    /// Adds a column of `rows` nulls of each build column's type to
    /// `output`, for a left join with no build rows.
    fn push_nulls(
        &self,
        context: &mut Context,
        rows: u32,
        output: &mut RowBatch,
    ) -> Result<(), Error> {
        for &data_type in self.build_types.iter() {
            let mut values = context.values_buffer(rows as usize * 8)?;
            values.as_mut_slice::<u64>().fill(0);
            let mut validity = context.small_buffer((rows as usize).div_ceil(8))?;
            validity.as_mut_slice::<u8>().fill(0);
            let column = if data_type.has_bytes() {
                let bytes = context.bytes_buffer(0)?;
                ColumnView::strings(context, values, bytes, Some(validity))?
            } else {
                ColumnView::new(context, data_type, values, Some(validity))?
            };
            let Ok(()) = output.push_column(column) else { failed!() };
        }
        Ok(())
    }

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

impl Built {
    /// Writes each of the first `rows` rows' build row, by `key`, to
    /// `matched`, and any for a row not joined, as every row is gathered: in
    /// a loop of its own, which needs no branch.
    fn find(&mut self, key: &ColumnView, rows: usize) {
        let (first, count) = (self.first, u64::from(self.count));
        let matched = at_mut!(self.matched, ..rows);
        let build_row = |value: i64| {
            let row = value.wrapping_sub(first).cast_unsigned();
            #[expect(clippy::cast_possible_truncation, reason = "below `count` if kept")]
            let row = if row < count { row as u32 } else { 0 };
            row
        };
        match key.form() {
            Form::Sequence(start) => {
                for (row, out) in (0..).zip(matched.iter_mut()) {
                    *out = build_row(start.wrapping_add(row));
                }
            }
            Form::Flat => {
                for (out, &value) in matched.iter_mut().zip(key.int64s()) {
                    *out = build_row(value);
                }
            }
            _ => failed!(),
        }
    }

    /// Writes a bit to `missed` for each of the first `rows` rows, set if
    /// it has no build row: its key is outside the build side's, or null.
    fn find_misses(&mut self, key: &ColumnView, rows: usize) {
        let (first, count) = (self.first, u64::from(self.count));
        let missed = at_mut!(self.missed, ..rows.div_ceil(8));
        missed.fill(0);
        let (start, values, validity) = match key.form() {
            Form::Sequence(start) => (start, None, None),
            _ => (0, Some(key.int64s()), key.validity()),
        };
        // Eight at a time where it can, as a filter tests them.
        let mut done = 0;
        if let (true, Some(values), None) = (eight::EIGHT_AT_A_TIME, values, validity) {
            let span = count - 1;
            for (at, missed) in (0..rows - rows % 8).step_by(8).zip(missed.iter_mut()) {
                // SAFETY: the eight from `at` are rows of the batch, which
                // are the key's.
                let (low, high) = unsafe { eight::in_range(values, at, first, span) };
                *missed = !(low | high << 4);
            }
            done = rows - rows % 8;
        }
        #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "a row")]
        for row in done..rows {
            let value = values.map_or(start.wrapping_add(row as i64), |v| *at!(v, row));
            // SAFETY: `row` is one of the batch's, which are the key's.
            let valid = validity.is_none_or(|v| unsafe { v.is_valid_unchecked(row as u32) });
            let hit = valid && value.wrapping_sub(first).cast_unsigned() < count;
            *at_mut!(missed, row / 8) |= u8::from(!hit) << (row % 8);
        }
    }
}

impl Operator for Join<'_> {
    type State = Built;

    fn new_state(&self, context: &mut Context) -> Result<Built, Error> {
        let allocator = context.allocator();
        let matched =
            SlowVec::fixed_from(allocator, core::iter::repeat_n(0, BATCH_ROWS_MAX as usize))?;
        let rows = RowStore::new(allocator)?;
        let missed =
            SlowVec::fixed_from(allocator, core::iter::repeat_n(0, BATCH_ROWS_MAX as usize / 8))?;
        Ok(Built { drained: false, rows, first: 0, count: 0, matched, missed })
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
        output.copy_from(input);
        // With no build row, an anti join keeps every row, as `x NOT IN ()`
        // is true even of a null `x`, a left join joins each to nulls, and
        // the others keep none.
        if built.count == 0 {
            match self.kind {
                JoinKind::Anti => {}
                JoinKind::Left => self.push_nulls(context, input.row_count(), output)?,
                JoinKind::Inner | JoinKind::Semi => {
                    output.reset(0);
                    return Ok(Progress::Done);
                }
            }
            return Ok(Progress::NeedInput);
        }
        let key = input.column(self.probe_key);
        let (first, count) = (built.first, built.count);
        // The probe rows joined are those whose key is a build row's, as a
        // filter on the key would keep them; an anti join keeps the rest, of
        // the keys that aren't null, and a left join every one.
        let last = first.wrapping_add(i64::from(count - 1));
        let kept = match self.kind {
            JoinKind::Inner | JoinKind::Semi => Some(Range::between(first, last)),
            JoinKind::Anti => Some(Range::between(last.wrapping_add(1), first.wrapping_sub(1))),
            JoinKind::Left => None,
        };
        if let Some(kept) = kept {
            filter::keep_within(key, kept, output.selection_mut(), None);
        }
        if output.selection().is_empty() || self.build_columns.is_empty() {
            return Ok(Progress::NeedInput);
        }
        let rows = input.row_count() as usize;
        built.find(key, rows);
        let mut gathered = RowBatch::new();
        if self.kind == JoinKind::Left {
            built.find_misses(key, rows);
            let (matched, missed) =
                (at!(built.matched, ..rows), at!(built.missed, ..rows.div_ceil(8)));
            built.rows.gather_or_null(context, &mut gathered, matched, missed)?;
        } else {
            let matched = at!(built.matched, ..rows);
            built.rows.gather(context, &mut gathered, matched, 0, |_, _| failed!())?;
        }
        for c in 0..gathered.column_count() {
            let Ok(()) = output.push_column(gathered.column(c).clone()) else { failed!() };
        }
        Ok(Progress::NeedInput)
    }

    fn finish(&self, _: &mut Context, _: &mut Built, _: &mut RowBatch) -> Result<Progress, Error> {
        Ok(Progress::NeedInput)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;
    use crate::estimate::RowEstimate;
    use crate::scannable::Key;

    /// The child `JoinOp` builds, given its children's rows and keys.
    fn built(first: (Option<u64>, Key), second: (Option<u64>, Key)) -> usize {
        let estimate = |(rows, key): (Option<u64>, Key), column| {
            let rows = rows.map_or(RowEstimate::UNKNOWN, RowEstimate::exactly);
            let mut estimate = Estimate::of(rows);
            estimate.add_key(column, key);
            estimate
        };
        let none = || SlowVec::fixed(&Heap, 0).unwrap();
        let mut join = JoinOp::new(JoinKind::Inner, [0, 1], [none(), none()]);
        join.choose(&[estimate(first, 0), estimate(second, 1)]);
        join.build
    }

    #[test]
    fn builds_the_side_with_the_better_key_then_fewer_rows() {
        let (few, many) = (Some(10), Some(1_000_000));
        // A sequence, even if bigger, then a unique key, then fewer rows.
        assert_eq!(built((many, Key::Id), (few, Key::Unique)), 0);
        assert_eq!(built((many, Key::Unique), (few, Key::None)), 0);
        assert_eq!(built((few, Key::Unique), (many, Key::Unique)), 0);
        assert_eq!(built((many, Key::Unique), (few, Key::Unique)), 1);
        // As written when nothing tells them apart.
        assert_eq!(built((None, Key::None), (None, Key::None)), 1);
        assert_eq!(built((few, Key::Id), (few, Key::Id)), 1);
    }
}
