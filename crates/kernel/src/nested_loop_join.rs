//! `NestedLoopJoinOp`: a join of its two children on any condition. The
//! build side, its second child, is held whole first, as `JoinOp`'s is.
//! Without keys equal on both sides, the condition is tested on every pair
//! of rows; with them, only on the pairs a hash table of the build side's
//! keys finds.

use crate::allocator::{AllocError, Allocator};
use crate::column::DataType;
use crate::context::Context;
use crate::error::Error;
use crate::expression::{
    Compute, ComputeState, EXPR_NODES_MAX, Expr, ExprOp, ExprStep, check_expr_type, constant,
    mark_expr_columns, type_expr,
};
use crate::filter::{self, Comparison, Value};
use crate::group_table::{GroupTable, KEYS_MAX, NONE};
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{Arrangements, Column, DeriveContext, OutputProperties};
use crate::plan::{ColumnId, DynOp, Given, Op, PlanNode};
use crate::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use crate::row_store::RowStore;
use crate::selection::{Kept, Selection};
use crate::slow_vec::SlowVec;
use crate::step::{DynOperator, DynSource, Operator, Progress, Step, Transform};

/// Which probe rows a join outputs, and with what.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JoinKind {
    /// Each pair of rows the condition is true for.
    Inner,
    /// As `Inner`, and each probe row it's true for with no build row, with
    /// nulls for the build side's columns.
    Left,
}

/// Outputs the probe side's columns, then `build_columns`.
pub struct NestedLoopJoinOp {
    /// Which rows it outputs.
    pub kind: JoinKind,
    /// Which pairs of rows are joined, an `Int64` true where it isn't 0 or
    /// null: every pair if none, a cross join.
    pub condition: Option<Expr>,
    /// The build side's columns it holds: those it outputs, and those the
    /// condition reads.
    pub build_columns: SlowVec<ColumnId>,
}

impl NestedLoopJoinOp {
    fn is_build_column(&self, column: ColumnId) -> bool {
        self.build_columns.contains(&column)
    }

    /// `probe`'s columns, then the build columns of `build`: a pair's.
    fn pairs(
        &self,
        probe: &OutputProperties,
        build: &OutputProperties,
    ) -> Result<OutputProperties, LowerError> {
        let mut pairs = *probe;
        for &id in self.build_columns.iter() {
            let nullable = self.kind == JoinKind::Left || build.nullable_of(id);
            pairs.push_column(Column { id, data_type: build.type_of(id), nullable })?;
        }
        Ok(pairs)
    }
}

/// Whether `condition` reads `column`.
fn reads_column(condition: Option<&Expr>, column: ColumnId) -> bool {
    condition.is_some_and(|expr| {
        expr.nodes.iter().any(|node| node.op == ExprOp::Column && node.a == column)
    })
}

impl<'c> Op<'c> for NestedLoopJoinOp {
    /// Keeps the build columns the condition reads, and those needed above
    /// that it outputs.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        let condition = self.condition.as_ref();
        self.build_columns
            .retain(|&column| needed.is_needed(column) || reads_column(condition, column));
        self.reads(needed);
        Pruned::Keep
    }

    /// Its condition's columns, and the build columns, which it copies.
    fn reads(&self, reads: &mut Needed) {
        if let Some(condition) = &self.condition {
            mark_expr_columns(condition, reads);
        }
        for &column in self.build_columns.iter() {
            reads.need(column);
        }
    }

    /// A condition on a probe-side column can be applied to the probe side,
    /// and for an inner join, one on a build-side column to the build side.
    /// A left join outputs a probe row whatever its build rows are.
    fn moves_condition(&self, column: ColumnId) -> Option<u32> {
        if !self.is_build_column(column) {
            return Some(0);
        }
        (self.kind == JoinKind::Inner).then_some(1)
    }

    /// A left join under an op that drops every row with nulls for the
    /// build columns, such as a filter on `b.x > 1`, keeps no unmatched
    /// probe row, so is an inner join, whose conditions can move.
    fn simplify_under(&mut self, above: &DynOp<'c>) -> bool {
        let rejected = above.rejects_nulls(&|column| self.is_build_column(column));
        if self.kind != JoinKind::Left || !rejected {
            return false;
        }
        self.kind = JoinKind::Inner;
        true
    }

    /// The probe side's columns, as they are. The build side's it outputs
    /// are copies.
    fn passes_through(&self, column: ColumnId) -> bool {
        !self.is_build_column(column)
    }

    /// Nothing of its own. A condition above only drops rows it would
    /// output, but for one on a left join's build column: skipping build
    /// rows by it could leave a probe row with none, whose nulls it may be
    /// true for.
    fn give_condition(&mut self, _: &dyn Allocator, column: ColumnId) -> Result<Given, AllocError> {
        if self.kind == JoinKind::Left && self.is_build_column(column) {
            return Ok(Given::Stop);
        }
        Ok(Given::Passed(None))
    }

    /// The probe side's columns, less some rows, then the build columns of
    /// an inner or left join, whose rows come out by build row, in no order.
    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        let pairs = self.pairs(c.input(0), c.input(1))?;
        if let Some(condition) = &self.condition {
            let data_type = check_expr_type(condition, &|column| pairs.type_of(column));
            if data_type != Ok(DataType::Int64) {
                // Named by the first column it reads.
                let column = condition.nodes.iter().find(|node| node.op == ExprOp::Column);
                return Err(LowerError::WrongType(column.map_or(0, |node| node.a)));
            }
        }
        let mut out = pairs;
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
        check!(node.children.len() == 2);
        let allocator = lowering.allocator();
        let (build, rows) = lowering.lower_side(node, 1)?;
        let position = |column| {
            let Some(position) = rows.position_of(column) else { failed!() };
            position
        };
        let mut build_columns = [0; BATCH_COLUMNS_MAX as usize];
        let mut build_types = [DataType::Int64; BATCH_COLUMNS_MAX as usize];
        for (i, &column) in self.build_columns.iter().enumerate() {
            *at_mut!(build_columns, i) = position(column);
            *at_mut!(build_types, i) = rows.type_of(column);
        }
        let build_count = self.build_columns.len();
        lowering.lower_input(node, 0)?;
        let mut condition = None;
        let mut keys = None;
        if let Some(expr) = &self.condition {
            let pairs = self.pairs(lowering.query_rows(), rows)?;
            let Ok(program) = type_expr(allocator, &pairs, expr) else { failed!() };
            let programs = SlowVec::fixed_from(allocator, [program].into_iter())?;
            condition = Some(Compute { programs });
            keys = self.keys(lowering.query_rows(), rows, expr);
            // A condition that's only its keys' equalities needs no test once
            // they're found equal: each key takes three nodes, and each `AND`
            // between them one.
            if keys.as_ref().is_some_and(|keys| expr.nodes.len() == keys.probe.len() * 4 - 1) {
                condition = None;
            }
        }
        let join = Join {
            kind: self.kind,
            build,
            build_columns,
            build_types,
            build_count,
            condition,
            keys,
        };
        lowering.add_operator(Step::Operator(DynOperator::new(allocator, join)?))
    }
}

impl NestedLoopJoinOp {
    /// The keys to hash the build side by, if `condition` requires a probe
    /// column equal to a build column of a type a hash table keys: the
    /// `=`s it `AND`s at its top. A row with a null key, equal to none, is
    /// paired with none.
    fn keys(
        &self,
        probe: &OutputProperties,
        build: &OutputProperties,
        condition: &Expr,
    ) -> Option<Keys> {
        let nodes = &condition.nodes;
        let mut keys = Keys { probe: [0; KEYS_MAX], build: [0; KEYS_MAX], count: 0, strings: 0 };
        // The nodes still to look at, from the root.
        let mut pending = [0; EXPR_NODES_MAX];
        let mut count = 1;
        *at_mut!(pending, 0) = nodes.len() - 1;
        while count > 0 && keys.count < KEYS_MAX {
            count -= 1;
            let node = at!(nodes, *at!(pending, count));
            // Only `AND` and `=` read two nodes: other nodes' `a` and `b` can be
            // anything, such as a column's id.
            let operands = || (at!(nodes, node.a as usize), at!(nodes, node.b as usize));
            match node.op {
                ExprOp::And => {
                    *at_mut!(pending, count) = node.a as usize;
                    *at_mut!(pending, count + 1) = node.b as usize;
                    count += 2;
                }
                ExprOp::Eq
                    if operands().0.op == ExprOp::Column && operands().1.op == ExprOp::Column =>
                {
                    let (a, b) = operands();
                    let (p, q) = if self.is_build_column(a.a) { (b.a, a.a) } else { (a.a, b.a) };
                    let Some(index) = self.build_columns.iter().position(|&c| c == q) else {
                        continue;
                    };
                    if self.is_build_column(p) || probe.position_of(p).is_none() {
                        continue;
                    }
                    let data_type = probe.type_of(p);
                    if data_type != build.type_of(q) || data_type == DataType::Float64 {
                        continue;
                    }
                    if data_type == DataType::String {
                        keys.strings |= 1 << keys.count;
                    }
                    *at_mut!(keys.probe, keys.count) = probe.position_of(p).unwrap_or(0);
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "at most a batch's columns"
                    )]
                    let index = index as u32;
                    *at_mut!(keys.build, keys.count) = index;
                    keys.count += 1;
                }
                _ => {}
            }
        }
        (keys.count > 0).then_some(keys)
    }
}

/// The columns a join hashes its build side by, which its condition
/// requires equal on both sides.
struct Keys {
    /// Where each of the first `count` is in the probe side's batches.
    probe: [u32; KEYS_MAX],
    /// Which of the build columns each is.
    build: [u32; KEYS_MAX],
    /// How many there are.
    count: usize,
    /// A bit for each key that's a string.
    strings: u64,
}

/// Tests its condition on each input batch's rows paired with build rows,
/// a round at a time, outputting the pairs it's true for. Without keys, a
/// round pairs every row with one build row, in turn. With keys, it pairs
/// each row with its next build row of the same keys, found by hashing. The
/// build side is drained on the first input: an empty probe side never
/// reads it.
struct Join<'c> {
    /// Which rows it outputs.
    kind: JoinKind,
    /// The build side.
    build: DynSource<'c>,
    /// Where the first `build_count` build columns are in the build side's
    /// batches, and their types.
    build_columns: [u32; BATCH_COLUMNS_MAX as usize],
    build_types: [DataType; BATCH_COLUMNS_MAX as usize],
    build_count: usize,
    /// Computes the condition, as a column after a pair's.
    condition: Option<Compute>,
    /// The keys it hashes by, if any.
    keys: Option<Keys>,
}

/// What a run of a `Join` keeps.
struct Held {
    /// Whether the build side is drained.
    drained: bool,
    /// The build columns of every build row, in order, and how many rows.
    rows: RowStore,
    count: u32,
    /// Whether the input batch has had its first round.
    started: bool,
    /// Without keys, the build row of the next round.
    next: u32,
    /// With keys, each build row's group of keys, which holds its last build
    /// row plus one; and each build row's one before it of the group, or
    /// `NONE`; and whether any group has more than one.
    table: Option<GroupTable>,
    before: SlowVec<u32>,
    repeats: bool,
    /// With keys, each input row's build row of the next round, or `NONE`.
    next_of: SlowVec<u32>,
    /// A bit for each row of the batch being joined, set once a build row
    /// matched it.
    matched: [u64; BATCH_ROWS_MAX as usize / 64],
    /// For an inner or left join, a null column of each build column's
    /// type, a batch long: an unmatched row's.
    nulls: RowBatch<'static>,
    /// Each input row's build row in this round, for `gather`: always a
    /// build row's number, as every row is gathered.
    paired: SlowVec<u32>,
    /// What computing the condition keeps.
    condition: Option<ComputeState>,
}

/// Group `group`'s last build row, which its payload holds plus one, or
/// `NONE`.
fn last_of(table: &GroupTable, group: u32) -> u32 {
    u32::try_from(table.payload(group)).map_or(NONE, |last| last.wrapping_sub(1))
}

fn is_matched(matched: &[u64], row: u16) -> bool {
    at!(matched, usize::from(row / 64)) >> (row % 64) & 1 != 0
}

/// The kept rows of `selection`, by index, or empty if they're a prefix,
/// as `GroupTable::find_or_create` takes them.
fn selected(selection: &Selection) -> &[u16] {
    match selection.kept() {
        Kept::Select(rows) => rows,
        Kept::Prefix | Kept::None => &[],
    }
}

impl Join<'_> {
    /// Holds every row of the build side, and with keys, finds each one's
    /// group.
    fn drain(&self, context: &mut Context, held: &mut Held) -> Result<(), Error> {
        held.drained = true;
        let mut run = self.build.start(context)?;
        let mut batch = RowBatch::new();
        while run.next(context, &mut batch)? {
            let selection = batch.selection();
            let first = held.count;
            held.count = held.count.checked_add(selection.len()).ok_or(Error::OutOfMemory)?;
            if self.build_count == 0 || selection.is_empty() {
                continue;
            }
            if let (Some(keys), Some(table)) = (&self.keys, &mut held.table) {
                let mut columns = [batch.column(0); KEYS_MAX];
                for (column, &k) in columns.iter_mut().zip(at!(keys.build, ..keys.count)) {
                    *column = batch.column(*at!(self.build_columns, k as usize));
                }
                let columns = at!(columns, ..keys.count);
                // In `next_of`, unused until probing.
                let groups = at_mut!(held.next_of, ..selection.len() as usize);
                table.find_or_create(context, columns, selected(selection), groups)?;
                let nullable = columns.iter().any(|column| column.validity().is_some());
                for (i, (row, &group)) in (first..).zip(groups.iter()).enumerate() {
                    #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
                    let at = selection.kept().row(i as u32);
                    if nullable && columns.iter().any(|column| column.is_null(at)) {
                        held.before.push(NONE)?;
                        continue;
                    }
                    let last = last_of(table, group);
                    held.repeats |= last != NONE;
                    held.before.push(last)?;
                    table.set_payload(group, i64::from(row) + 1);
                }
            }
            let mut columns = RowBatch::new();
            columns.reset(batch.row_count());
            let build_columns = at!(self.build_columns, ..self.build_count);
            columns.push_borrowed(build_columns.iter().map(|&c| batch.column(c)));
            columns.selection_mut().clone_from(selection);
            held.rows.append(context, &columns)?;
        }
        if let Some(table) = &mut held.table {
            table.freeze();
        }
        held.rows.flush(context)
    }

    /// Starts joining `input`: no row matched, and with keys, each row's
    /// first build row of its keys.
    fn start(&self, context: &mut Context, held: &mut Held, input: &RowBatch) -> Result<(), Error> {
        held.started = true;
        held.next = 0;
        held.matched.fill(0);
        let (Some(keys), Some(table)) = (&self.keys, &mut held.table) else { return Ok(()) };
        let selection = input.selection();
        let mut columns = [input.column(0); KEYS_MAX];
        for (column, &position) in columns.iter_mut().zip(at!(keys.probe, ..keys.count)) {
            *column = input.column(position);
        }
        let columns = at!(columns, ..keys.count);
        // Each kept row's group, by its place among them, then its first
        // build row, by its number: from the last, as a row's number is at
        // least its place, so no group is written over before it's read.
        let groups = at_mut!(held.next_of, ..selection.len() as usize);
        table.find_or_create(context, columns, selected(selection), groups)?;
        let nullable = columns.iter().any(|column| column.validity().is_some());
        for i in (0..selection.len()).rev() {
            let (group, row) = (*at!(held.next_of, i as usize), selection.kept().row(i));
            let null = nullable && columns.iter().any(|column| column.is_null(row));
            let last = if group == NONE || null { NONE } else { last_of(table, group) };
            *at_mut!(held.next_of, row as usize) = last;
        }
        Ok(())
    }

    /// Narrows `pairs`, a copy of the input batch, to its rows with a build
    /// row this round, writing each one's to `paired`, and readies the next
    /// round. Returns `None` if none has one, else whether any has a build
    /// row after this round's.
    fn next_round(&self, held: &mut Held, pairs: &mut RowBatch) -> Option<bool> {
        if self.keys.is_none() {
            if held.next == held.count {
                return None;
            }
            at_mut!(held.paired, ..pairs.row_count() as usize).fill(held.next);
            held.next += 1;
            return (!pairs.selection().is_empty()).then_some(held.next < held.count);
        }
        // Rows left out keep the build rows they had, which every row has,
        // as they're gathered too.
        let (paired, next_of, before) = (&mut held.paired, &mut held.next_of, &held.before);
        let repeats = held.repeats;
        let mut more = false;
        pairs.selection_mut().retain(|row| {
            let build = *at!(next_of, usize::from(row));
            if build == NONE {
                return false;
            }
            // Without repeats, no build row has one before it.
            let next = if repeats { *at!(before, build as usize) } else { NONE };
            *at_mut!(paired, usize::from(row)) = build;
            *at_mut!(next_of, usize::from(row)) = next;
            more |= next != NONE;
            true
        });
        (!pairs.selection().is_empty()).then_some(more)
    }

    /// Adds to `pairs` the build columns of each row's build row this
    /// round, and keeps the rows the condition is true for.
    fn pair(
        &self,
        context: &mut Context,
        held: &mut Held,
        pairs: &mut RowBatch,
    ) -> Result<(), Error> {
        if self.build_count > 0 {
            let paired = at!(held.paired, ..pairs.row_count() as usize);
            let mut build = RowBatch::new();
            held.rows.gather_held(context, &mut build, paired)?;
            for c in 0..build.column_count() {
                let Ok(()) = pairs.push_column(build.column(c).clone()) else { failed!() };
            }
        }
        if let (Some(condition), Some(state)) = (&self.condition, &mut held.condition) {
            condition.process(context, state, pairs)?;
            let (columns, selection) = pairs.columns_and_selection();
            let Some(truth) = columns.last() else { failed!() };
            filter::compare(truth, Comparison::NotEqual, Value::Int64(0), selection, None);
        }
        Ok(())
    }

    /// Outputs the input's rows of this kind once every round is done: a
    /// left join's unmatched rows, and an inner join's none.
    fn finish_input(&self, held: &mut Held, input: &RowBatch, output: &mut RowBatch) {
        held.started = false;
        output.copy_from(input);
        let matched = &held.matched;
        let left = self.kind == JoinKind::Left;
        output.selection_mut().retain(|row| left && !is_matched(matched, row));
        for c in 0..held.nulls.column_count() {
            let null = held.nulls.column(c).slice(0, input.row_count());
            let Ok(()) = output.push_column(null) else { failed!() };
        }
    }
}

impl Operator for Join<'_> {
    type State = Held;

    fn new_state(&self, context: &mut Context) -> Result<Held, Error> {
        let allocator = context.allocator();
        let rows = RowStore::new(allocator)?;
        let mut nulls = RowBatch::new();
        nulls.reset(BATCH_ROWS_MAX);
        for &data_type in at!(self.build_types, ..self.build_count) {
            let step = ExprStep {
                op: ExprOp::Null,
                data_type,
                a: 0,
                b: 0,
                c: 0,
                int_value: 0,
                double_value: 0.0,
            };
            let Some(null) = constant(context, &step)? else { failed!() };
            let Ok(()) = nulls.push_column(null) else { failed!() };
        }
        let condition = match &self.condition {
            Some(condition) => Some(condition.new_state(context)?),
            None => None,
        };
        // A word a group after its keys, for its last build row.
        let table = match &self.keys {
            Some(keys) => Some(GroupTable::new(allocator, keys.count, keys.strings, 0, 1)?),
            None => None,
        };
        Ok(Held {
            drained: false,
            rows,
            count: 0,
            started: false,
            next: 0,
            table,
            before: SlowVec::new(allocator, 1 << 31)?,
            repeats: false,
            // Not arrays, which would be copied into place on each run.
            next_of: SlowVec::zeroed(allocator, BATCH_ROWS_MAX as usize)?,
            matched: [0; BATCH_ROWS_MAX as usize / 64],
            nulls,
            paired: SlowVec::zeroed(allocator, BATCH_ROWS_MAX as usize)?,
            condition,
        })
    }

    fn execute(
        &self,
        context: &mut Context,
        held: &mut Held,
        input: &RowBatch,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        if !held.drained {
            self.drain(context, held)?;
        }
        // No probe row can be joined.
        if held.count == 0 && self.kind == JoinKind::Inner {
            return Ok(Progress::Done);
        }
        if !held.started {
            self.start(context, held, input)?;
        }
        let probe_columns = input.column_count();
        loop {
            let mut pairs = RowBatch::new();
            pairs.copy_from(input);
            let Some(more) = self.next_round(held, &mut pairs) else { break };
            self.pair(context, held, &mut pairs)?;
            let selection = pairs.selection();
            for i in 0..selection.len() {
                let kept = selection.kept().row(i);
                *at_mut!(held.matched, kept as usize / 64) |= 1 << (kept % 64);
            }
            if selection.is_empty() {
                continue;
            }
            output.reset(input.row_count());
            #[expect(clippy::cast_possible_truncation, reason = "at most a batch's columns")]
            let build_columns = self.build_count as u32;
            for c in 0..probe_columns + build_columns {
                let Ok(()) = output.push_column(pairs.column(c).clone()) else { failed!() };
            }
            output.selection_mut().clone_from(pairs.selection());
            // An inner join has nothing to add once no row has a build row
            // left: as when each has one, N:1.
            if self.kind == JoinKind::Inner && !more {
                held.started = false;
                return Ok(Progress::NeedInput);
            }
            return Ok(Progress::MoreOutput);
        }
        self.finish_input(held, input, output);
        Ok(Progress::NeedInput)
    }

    fn finish(&self, _: &mut Context, _: &mut Held, _: &mut RowBatch) -> Result<Progress, Error> {
        Ok(Progress::NeedInput)
    }
}
