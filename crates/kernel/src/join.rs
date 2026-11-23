//! `JoinOp`: a join of its two children on any condition. The build side,
//! the child planning chooses, runs as a pipeline of its own and is held
//! whole first.
//! Without keys the condition requires equal on both sides, it's tested on
//! every pair of rows: a nested loop. With them, the build side is hashed by
//! them, and only the pairs of equal keys are tested. With one integer key,
//! unique and of values close enough, a key's build row is found by its
//! value, as a perfect hash finds it: from an array, or with no array if
//! each key is its row's number above the first, as a table's ids are.

use crate::allocator::{AllocError, Allocator};
use crate::column::{ColumnView, DataType, Form, Forms};
use crate::context::Context;
use crate::error::Error;
use crate::estimate::Estimate;
use crate::expression::{
    Compute, ComputeState, EXPR_NODES_MAX, Expr, ExprOp, ExprStep, Exprs, check_expr_type,
    columns_of, conjuncts, constant, mark_expr_columns, type_expr,
};
use crate::filter::{self, Comparison, Range, Value};
use crate::group_table::{GroupTable, NONE};
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{Arrangements, Column, DeriveContext, OutputProperties};
use crate::plan::{ColumnId, Given, Op, PlanNode};
use crate::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};

use crate::row_store::RowStore;
use crate::scannable::Key;
use crate::selection::{Kept, Selection};
use crate::slow_vec::SlowVec;
use crate::step::{DynOperator, DynSource, Operator, Progress, Step, Transform};

/// The most columns a batch has.
const COLUMNS: usize = BATCH_COLUMNS_MAX as usize;

/// Which probe rows a join outputs, and with what.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JoinKind {
    /// Each pair of rows the condition is true for.
    Inner,
    /// As `Inner`, and each probe row it's true for with no build row, with
    /// nulls for the build side's columns.
    Left,
}

/// Outputs the probe side's columns, then the build side's it outputs.
pub struct JoinOp {
    /// Which rows it outputs.
    kind: JoinKind,
    /// Which pairs of rows are joined, an `Int64` true where it isn't 0 or
    /// null: every pair if none, a cross join.
    condition: Option<Expr>,
    /// Each child's columns it outputs.
    columns: [SlowVec<ColumnId>; 2],
    /// Which child is the build side: the second, unless planning chooses
    /// the first.
    build: u32,
}

impl JoinOp {
    /// A join of `kind` on `condition`, outputting `columns` of each child.
    pub fn new(kind: JoinKind, condition: Option<Expr>, columns: [SlowVec<ColumnId>; 2]) -> JoinOp {
        JoinOp { kind, condition, columns, build: 1 }
    }

    fn probe(&self) -> u32 {
        1 - self.build
    }

    /// The build side's columns it outputs.
    /// Which child `column`, of one, is from.
    fn child_of(&self, column: ColumnId) -> u32 {
        if self.is_build_column(column) { self.build } else { self.probe() }
    }

    /// Whether it outputs the probe rows no build row matched: a left join
    /// that builds its second child.
    fn keeps_probe(&self) -> bool {
        self.kind == JoinKind::Left && self.build == 1
    }

    /// Whether it outputs the build rows no probe row matched: a left join
    /// that builds its first child.
    fn keeps_build(&self) -> bool {
        self.kind == JoinKind::Left && self.build == 0
    }

    fn build_columns(&self) -> &[ColumnId] {
        at!(self.columns, self.build as usize)
    }

    fn is_build_column(&self, column: ColumnId) -> bool {
        #[expect(clippy::manual_contains, reason = "`contains` unrolls for long slices")]
        self.build_columns().iter().any(|&c| c == column)
    }

    /// The best key either side has among the equalities at the top of its
    /// condition: what decides which side to build.
    fn side_keys(&self, exprs: &Exprs, inputs: &[Estimate]) -> [Key; 2] {
        let mut keys = [Key::None; 2];
        let Some(condition) = self.condition else { return keys };
        let nodes = exprs.nodes(condition);
        conjuncts(nodes, condition.root, &mut |conjunct| {
            let node = at!(nodes, conjunct as usize);
            let column = |i: u32| nodes.get(i as usize).filter(|n| n.op == ExprOp::Column);
            let (Some(a), Some(b)) = (column(node.a), column(node.b)) else { return };
            if node.op != ExprOp::Eq {
                return;
            }
            for (side, key) in keys.iter_mut().enumerate() {
                for column in [a.a, b.a] {
                    if at!(self.columns, side).iter().any(|&c| c == column) {
                        *key = (*key).max(at!(inputs, side).key(column));
                    }
                }
            }
        });
        keys
    }

    /// The build columns, then if `tested`, the other columns of `build` the
    /// condition reads: what it holds of each build row, and how many.
    fn held(
        &self,
        exprs: &Exprs,
        build: &OutputProperties,
        tested: bool,
    ) -> ([ColumnId; COLUMNS], usize) {
        let mut held = [0; COLUMNS];
        let mut count = 0;
        let mut add = |column| {
            let new = build.position_of(column).is_some()
                && !at!(held, ..count).iter().any(|&c| c == column);
            if new && count < COLUMNS {
                *at_mut!(held, count) = column;
                count += 1;
            }
        };
        for &column in self.build_columns() {
            add(column);
        }
        if let Some(condition) = self.condition.filter(|_| tested) {
            columns_of(exprs.nodes(condition), condition.root, &mut add);
        }
        (held, count)
    }

    /// `probe`'s columns, then the `held` columns of `build`: a pair's.
    fn pairs(
        &self,
        probe: &OutputProperties,
        build: &OutputProperties,
        held: &[ColumnId],
    ) -> Result<OutputProperties, LowerError> {
        let mut pairs = *probe;
        for &id in held {
            let nullable = self.keeps_probe() || build.nullable_of(id);
            pairs.push_column(Column { id, data_type: build.type_of(id), nullable })?;
        }
        Ok(pairs)
    }
}

impl<'c> Op<'c> for JoinOp {
    /// Keeps the columns needed above.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        for columns in &mut self.columns {
            columns.retain(|&column| needed.is_needed(column));
        }
        self.reads(needed);
        Pruned::Keep
    }

    /// Its condition's columns, and the build columns, which it copies.
    fn reads(&self, reads: &mut Needed) {
        if let Some(condition) = self.condition {
            mark_expr_columns(condition, reads);
        }
        for columns in &self.columns {
            for &column in columns.iter() {
                reads.need(column);
            }
        }
    }

    /// Sequences of what can be its keys, as a table's ids come: a build
    /// side's it holds no hash table for.
    fn accepts(&self, exprs: &Exprs, column: ColumnId) -> Forms {
        let key = self.condition.is_some_and(|expr| {
            let nodes = exprs.nodes(expr);
            let is_column = |i: u32| {
                nodes.get(i as usize).is_some_and(|n| n.op == ExprOp::Column && n.a == column)
            };
            nodes
                .iter()
                .any(|node| node.op == ExprOp::Eq && (is_column(node.a) || is_column(node.b)))
        });
        if key { Forms::FLAT | Forms::SEQUENCE } else { Forms::FLAT }
    }

    /// A condition on a child's column can be applied to that child, but
    /// for a left join's second: the join outputs each row of its first
    /// whatever rows of its second it has.
    fn moves_condition(&self, column: ColumnId) -> Option<u32> {
        let child = self.child_of(column);
        (self.kind == JoinKind::Inner || child == 0).then_some(child)
    }

    /// Builds the side whose key is best, an id first, as it's never
    /// hashed, then a unique key, as each probe row then has one build row;
    /// then the side with fewer rows, by planning's estimates. A left join
    /// that builds its first child outputs its unmatched rows at the end.
    fn choose(&mut self, exprs: &Exprs, inputs: &[Estimate]) {
        let &[first, second] = inputs else { failed!() };
        let [left, right] = self.side_keys(exprs, inputs);
        // A better key, then fewer rows; as written if nothing tells them apart.
        let rank = |key, input: &Estimate| (key, u64::MAX - input.rows.estimated);
        self.build = u32::from(rank(left, &first) <= rank(right, &second));
    }

    /// As many rows as the probe side, or fewer, and its keys but for ids,
    /// as rows may be dropped. Not the build side's: one build row may be
    /// joined to many.
    fn estimate(&self, _: &Exprs, inputs: &[Estimate]) -> Option<Estimate> {
        let mut out = at!(inputs, self.probe() as usize).after_filters();
        out.demote_ids();
        Some(out)
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
    fn give_condition(
        &mut self,
        _: &Exprs,
        _: &dyn Allocator,
        column: ColumnId,
    ) -> Result<Given, AllocError> {
        if self.kind == JoinKind::Left && self.child_of(column) == 1 {
            return Ok(Given::Stop);
        }
        Ok(Given::Passed(None))
    }

    /// The probe side's columns, less some rows, then the build columns of
    /// an inner or left join, whose rows come out by build row, in no order.
    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        let (probe, build) = (self.probe(), self.build);
        let (held, count) = self.held(c.exprs(), c.input(build), true);
        let pairs = self.pairs(c.input(probe), c.input(build), at!(held, ..count))?;
        if let Some(condition) = self.condition {
            let nodes = c.exprs().nodes(condition);
            let data_type = check_expr_type(nodes, &|column| pairs.type_of(column));
            if data_type != Ok(DataType::Int64) {
                // Named by the first column it reads.
                let column = nodes.iter().find(|node| node.op == ExprOp::Column);
                return Err(LowerError::WrongType(column.map_or(0, |node| node.a)));
            }
        }
        let mut out = *c.input(probe);
        if self.keeps_build() {
            out.make_nullable();
        }
        let mut out = self.pairs(&out, c.input(build), self.build_columns())?;
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
        let (build, rows) = lowering.lower_side(node, self.build)?;
        lowering.lower_input(node, self.probe())?;
        let probe = lowering.query_rows();
        let mut condition = None;
        let mut keys = None;
        let mut tested = false;
        let exprs = lowering.exprs();
        if let Some(expr) = self.condition {
            let nodes = exprs.nodes(expr);
            keys = Self::keys(probe, rows, nodes);
            // A condition that's only its keys' equalities needs no test once
            // they're found equal: each key takes three nodes, and each `AND`
            // between them one.
            tested = keys.as_ref().is_none_or(|keys| nodes.len() != keys.count * 4 - 1);
            if tested {
                let (held, count) = self.held(exprs, rows, true);
                let pairs = self.pairs(probe, rows, at!(held, ..count))?;
                let values = lowering.expr_values();
                let Ok(program) = type_expr(allocator, &pairs, nodes, values, true) else {
                    failed!()
                };
                let mut programs = SlowVec::fixed(allocator, 1)?;
                programs.push(program)?;
                condition = Some(Compute { programs });
            }
        }
        let (held, count) = self.held(exprs, rows, tested);
        let held = at!(held, ..count);
        let position = |&column: &ColumnId| {
            let Some(position) = rows.position_of(column) else { failed!() };
            position
        };
        let mut build_columns = [0; BATCH_COLUMNS_MAX as usize];
        for (at, column) in build_columns.iter_mut().zip(held) {
            *at = position(column);
        }
        let mut build_types = [DataType::Int64; BATCH_COLUMNS_MAX as usize];
        for (data_type, &column) in build_types.iter_mut().zip(self.build_columns()) {
            *data_type = rows.type_of(column);
        }
        let (held, outputs) = (held.len(), self.build_columns().len());
        let mut probe_types = SlowVec::fixed(allocator, probe.column_count() as usize)?;
        if self.keeps_build() {
            probe_types.extend_within(probe.types());
        }
        let join = Join {
            kind: self.kind,
            build,
            build_columns,
            held,
            build_types,
            outputs,
            condition,
            keys,
            keeps_build: self.keeps_build(),
            probe_types,
        };
        lowering.add_operator(Step::Operator(DynOperator::new(allocator, join)?))
    }
}

impl JoinOp {
    /// The keys to hash the build side by, if `condition` requires a probe
    /// column equal to a build column of a type a hash table keys: the
    /// `=`s it `AND`s at its top. A row with a null key, equal to none, is
    /// paired with none.
    fn keys(
        probe: &OutputProperties,
        build: &OutputProperties,
        nodes: &[ExprStep],
    ) -> Option<Keys> {
        let mut keys =
            Keys { probe: [0; JOIN_KEYS_MAX], build: [0; JOIN_KEYS_MAX], count: 0, strings: 0 };
        // The nodes still to look at, from the root.
        let mut pending = [0; EXPR_NODES_MAX];
        let mut count = 1;
        *at_mut!(pending, 0) = nodes.len() - 1;
        while count > 0 && keys.count < JOIN_KEYS_MAX {
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
                    let (p, q) =
                        if probe.position_of(a.a).is_some() { (a.a, b.a) } else { (b.a, a.a) };
                    let (Some(at), Some(index)) = (probe.position_of(p), build.position_of(q))
                    else {
                        continue;
                    };
                    let data_type = probe.type_of(p);
                    if data_type != build.type_of(q) || data_type == DataType::Float64 {
                        continue;
                    }
                    if data_type == DataType::String {
                        keys.strings |= 1 << keys.count;
                    }
                    *at_mut!(keys.probe, keys.count) = at;
                    *at_mut!(keys.build, keys.count) = index;
                    keys.count += 1;
                }
                _ => {}
            }
        }
        (keys.count > 0).then_some(keys)
    }
}

/// The most keys a join hashes by. Equalities past them are tested, as the
/// rest of its condition is.
const JOIN_KEYS_MAX: usize = 4;

/// The columns a join hashes its build side by, which its condition
/// requires equal on both sides.
struct Keys {
    /// Where each of the first `count` is in the probe side's batches.
    probe: [u32; JOIN_KEYS_MAX],
    /// Where each is in the build side's batches.
    build: [u32; JOIN_KEYS_MAX],
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
    /// Where the first `held` build columns are in the build side's
    /// batches: those it outputs, then those only its condition reads; and
    /// the types of the first `outputs`, those it outputs.
    build_columns: [u32; BATCH_COLUMNS_MAX as usize],
    held: usize,
    build_types: [DataType; BATCH_COLUMNS_MAX as usize],
    outputs: usize,
    /// Computes the condition, as a column after a pair's.
    condition: Option<Compute>,
    /// The keys it hashes by, if any.
    keys: Option<Keys>,
    /// Whether it outputs the build rows no probe row matched, at the end,
    /// with nulls of `probe_types` for the probe side's columns.
    keeps_build: bool,
    probe_types: SlowVec<DataType>,
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
    /// Without keys, the build row of the next round; with keys, how many
    /// rounds there have been.
    next: u32,
    /// With keys, each build row's group of keys, which holds its last build
    /// row plus one; and each build row's one before it of the group, or
    /// `NONE`; and whether any group has more than one.
    table: Option<GroupTable>,
    before: SlowVec<u32>,
    repeats: bool,
    /// With one integer key, unique, of values close enough, how to find
    /// a key's build row by its value, in place of `table`.
    direct: Option<Direct>,
    /// With one integer key, the first build row's key while each build
    /// row's has been its number above it, as a table's ids are: then none
    /// is hashed, as `direct` finds them all.
    sequence: Sequence,
    /// With keys, each input row's build row of the next round, or `NONE`.
    next_of: SlowVec<u32>,
    /// A bit for each row of the batch being joined, set once a build row
    /// matched it.
    matched: [u64; BATCH_ROWS_MAX as usize / 64],
    /// For an inner or left join, a null column of each build column's
    /// type, a batch long: an unmatched row's.
    nulls: SlowVec<ColumnView>,
    /// Each input row's build row in this round, for `gather`: always a
    /// build row's number, as every row is gathered.
    paired: SlowVec<u32>,
    /// What computing the condition keeps.
    condition: Option<ComputeState>,
    /// When it keeps the build rows: a bit for each, set once a probe row
    /// matched it; the next it hasn't looked at for output at the end; and
    /// a null column of each probe column's type, a batch long.
    build_matched: SlowVec<u64>,
    unmatched: u32,
    probe_nulls: SlowVec<ColumnView>,
}

/// The most a key can be above the smallest for `Direct` to find build
/// rows by their keys, as DuckDB's perfect hash join's: an array of 4 MB.
const DIRECT_SPAN_MAX: u64 = 1 << 20;

/// Finds the build row of a key by its value, as a perfect hash does: by
/// how far it is above the smallest key.
struct Direct {
    /// The smallest key, and how far above it the largest is.
    min: i64,
    span: u64,
    /// The build row of each key, by how far it is above `min`, or `NONE`.
    /// Empty if each key's build row is that, as a table's ids' are.
    rows: SlowVec<u32>,
}

impl Direct {
    /// For the build rows of `table`'s groups, of one integer key each,
    /// if their keys are close enough.
    fn of(
        allocator: &dyn Allocator,
        table: &GroupTable,
        count: u32,
    ) -> Result<Option<Direct>, AllocError> {
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let groups = table.len() as u32;
        let keys = (0..groups).map(|group| table.key(group, 0));
        let (Some(min), Some(max)) = (keys.clone().min(), keys.max()) else { return Ok(None) };
        let span = max.wrapping_sub(min).cast_unsigned();
        if max < min || span >= DIRECT_SPAN_MAX {
            return Ok(None);
        }
        let above = |group| table.key(group, 0).wrapping_sub(min).cast_unsigned();
        // Each build row its key's number above the first: none is a null
        // key's, which has no group.
        let each = (0..groups).all(|group| u64::from(last_of(table, group)) == above(group));
        if groups == count && each {
            return Ok(Some(Direct { min, span, rows: SlowVec::fixed(allocator, 0)? }));
        }
        #[expect(clippy::cast_possible_truncation, reason = "below `DIRECT_SPAN_MAX`")]
        let mut rows =
            SlowVec::fixed_from(allocator, core::iter::repeat_n(NONE, span as usize + 1))?;
        for group in 0..groups {
            #[expect(clippy::cast_possible_truncation, reason = "below `DIRECT_SPAN_MAX`")]
            let at = above(group) as usize;
            *at_mut!(rows, at) = last_of(table, group);
        }
        Ok(Some(Direct { min, span, rows }))
    }
}

/// Whether a join's build rows' keys have each been their number above the
/// first's.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sequence {
    /// No build row yet.
    Empty,
    /// So far, from this first key.
    From(i64),
    /// Not, or not one integer key.
    Broken,
}

impl Direct {
    /// Narrows `pairs` to its rows whose key, at `key`, has a build row,
    /// writing each one's to `paired`, of the `count` build rows. Every row
    /// is gathered, so one without a build row takes the first's.
    fn pair(
        &self,
        context: &mut Context,
        key: u32,
        count: u32,
        paired: &mut [u32],
        pairs: &mut RowBatch,
    ) -> Result<(), Error> {
        let mut key = pairs.column(key).clone();
        key.make_in(context, Forms::FLAT)?;
        let (min, span) = (self.min, self.span);
        filter::keep_within(&key, Range { lo: min, span }, pairs.selection_mut(), None);
        // A bound of 32 bits, so the loop works in 32 bits.
        #[expect(clippy::cast_possible_truncation, reason = "below `DIRECT_SPAN_MAX`")]
        let keys = span as u32 + 1;
        let row_of = |value: i64| {
            let above = value.wrapping_sub(min).cast_unsigned();
            #[expect(clippy::cast_possible_truncation, reason = "below `keys`")]
            let row = if above < u64::from(keys) { above as u32 } else { 0 };
            row
        };
        for (paired, &value) in paired.iter_mut().zip(key.int64s()) {
            *paired = row_of(value);
        }
        if self.rows.is_empty() {
            return Ok(());
        }
        let rows = &self.rows;
        for paired in paired.iter_mut() {
            *paired = *at!(rows, *paired as usize);
        }
        pairs.selection_mut().retain(|row| *at!(paired, usize::from(row)) != NONE);
        for paired in paired.iter_mut() {
            *paired = (*paired).min(count - 1);
        }
        Ok(())
    }
}

/// Writes to `groups` the group in `table` of the keys at `positions` of
/// each kept row of `batch`, by its place among them, or `NONE` if a key is
/// null, as a null is equal to none. Out of line, as both sides take it.
#[inline(never)]
fn find_groups(
    context: &mut Context,
    table: &mut GroupTable,
    batch: &RowBatch,
    positions: &[u32],
    groups: &mut [u32],
) -> Result<(), Error> {
    // Flat, or a dictionary's, as a group table takes them.
    let mut copies = [const { None }; JOIN_KEYS_MAX];
    for (copy, &position) in copies.iter_mut().zip(positions) {
        let column = batch.column(position);
        if !column.is_in(Forms::FLAT | Forms::DICTIONARY) {
            let mut flat = column.clone();
            flat.make_in(context, Forms::FLAT)?;
            *copy = Some(flat);
        }
    }
    let mut columns = [batch.column(0); JOIN_KEYS_MAX];
    for ((column, copy), &position) in columns.iter_mut().zip(&copies).zip(positions) {
        *column = copy.as_ref().unwrap_or_else(|| batch.column(position));
    }
    let columns = at!(columns, ..positions.len());
    let selection = batch.selection();
    table.find_or_create(context, columns, selected(selection), groups)?;
    if columns.iter().any(|column| column.validity().is_some()) {
        for (i, group) in groups.iter_mut().enumerate() {
            #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
            let row = selection.kept().row(i as u32);
            if columns.iter().any(|column| column.is_null(row)) {
                *group = NONE;
            }
        }
    }
    Ok(())
}

/// The first of `key`'s first `count` values if each is one above the one
/// before, as a table's ids are, flat or as a sequence, and `sequence` isn't
/// broken already.
fn consecutive(key: &ColumnView, count: u32, sequence: Sequence) -> Option<i64> {
    match key.form() {
        _ if sequence == Sequence::Broken => None,
        Form::Sequence(start) => Some(start),
        Form::Flat if key.validity().is_none() => {
            let values = at!(key.int64s(), ..count as usize);
            let start = *values.first()?;
            let mut expected = start;
            let mut all = true;
            for &value in values {
                all &= value == expected;
                expected = expected.wrapping_add(1);
            }
            all.then_some(start)
        }
        _ => None,
    }
}

/// A null column of each of `types`, a batch long.
fn null_columns(context: &mut Context, types: &[DataType]) -> Result<SlowVec<ColumnView>, Error> {
    let mut nulls = SlowVec::fixed(context.allocator(), types.len())?;
    for &data_type in types {
        let step = ExprStep { data_type, ..ExprStep::new(ExprOp::Null, 0, 0, 0) };
        let Some(null) = constant(context, &step)? else { failed!() };
        nulls.push(null)?;
    }
    Ok(nulls)
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
    /// Whether it outputs the probe rows no build row matched.
    fn keeps_probe(&self) -> bool {
        self.kind == JoinKind::Left && !self.keeps_build
    }

    /// Holds every row of the build side, and with keys, finds each one's
    /// group.
    fn drain(&self, context: &mut Context, held: &mut Held) -> Result<(), Error> {
        held.drained = true;
        let mut run = self.build.start(context)?;
        let mut batch = RowBatch::new();
        let build_columns = at!(self.build_columns, ..self.held);
        while run.next(context, &mut batch)? {
            let selection = batch.selection();
            let first = held.count;
            held.count = held.count.checked_add(selection.len()).ok_or(Error::OutOfMemory)?;
            if selection.is_empty() {
                continue;
            }
            if let Some(keys) = &self.keys {
                Self::hash(context, held, keys, &batch, first)?;
            }
            if build_columns.is_empty() {
                continue;
            }
            let mut columns = RowBatch::new();
            columns.reset(batch.row_count());
            let selection = batch.selection();
            if build_columns.iter().all(|&c| matches!(batch.column(c).form(), Form::Flat)) {
                columns.push_borrowed(build_columns.iter().map(|&c| batch.column(c)));
            } else {
                // A row store holds flat columns, and keys can come as
                // sequences.
                for &c in build_columns {
                    let mut column = batch.column(c).clone();
                    column.make_in(context, Forms::FLAT)?;
                    let Ok(()) = columns.push_column(column) else { failed!() };
                }
            }
            columns.selection_mut().clone_from(selection);
            held.rows.append(context, &columns)?;
        }
        if let Some(table) = &mut held.table {
            table.freeze();
        }
        // One integer key, each in one build row, is found by its value.
        if let Sequence::From(min) = held.sequence {
            let span = u64::from(held.count - 1);
            held.direct = Some(Direct { min, span, rows: SlowVec::fixed(context.allocator(), 0)? });
            held.table = None;
        } else if let (true, false, Some(table)) = (self.one_integer(), held.repeats, &held.table) {
            held.direct = Direct::of(context.allocator(), table, held.count)?;
            if held.direct.is_some() {
                held.table = None;
            }
        }
        if self.keeps_build {
            let words = held.count.div_ceil(64) as usize;
            held.build_matched = SlowVec::zeroed(context.allocator(), words)?;
        }
        held.rows.flush(context)
    }

    /// Whether it joins on one integer key.
    fn one_integer(&self) -> bool {
        self.keys.as_ref().is_some_and(|keys| keys.count == 1 && keys.strings == 0)
    }

    /// Finds the groups of `batch`'s keys, its rows numbered from `first`,
    /// in the hash table, working in `next_of`, unused until probing:
    /// unless its one integer key continues a sequence, which needs none.
    fn hash(
        context: &mut Context,
        held: &mut Held,
        keys: &Keys,
        batch: &RowBatch,
        first: u32,
    ) -> Result<(), Error> {
        let selection = batch.selection();
        let key = batch.column(*at!(keys.build, 0));
        if let (Some(start), true) =
            (consecutive(key, selection.len(), held.sequence), selection.is_prefix())
        {
            let follows = match held.sequence {
                Sequence::Empty => Some(start),
                Sequence::From(min) => {
                    (min.checked_add(i64::from(first)) == Some(start)).then_some(min)
                }
                Sequence::Broken => None,
            };
            if let Some(min) = follows {
                held.sequence = Sequence::From(min);
                return Ok(());
            }
        }
        let Some(table) = &mut held.table else { failed!() };
        if let Sequence::From(min) = held.sequence {
            // The rows before, held as a sequence, are hashed now.
            let mut row = 0;
            while row < first {
                let count = (first - row).min(BATCH_ROWS_MAX);
                let mut keys = ColumnView::sequence(context, min + i64::from(row), count)?;
                keys.make_in(context, Forms::FLAT)?;
                let groups = at_mut!(held.next_of, ..count as usize);
                table.find_or_create(context, &[&keys], &[], groups)?;
                for &group in groups.iter() {
                    held.before.push(NONE)?;
                    table.set_payload(group, i64::from(row) + 1);
                    row += 1;
                }
            }
        }
        held.sequence = Sequence::Broken;
        let groups = at_mut!(held.next_of, ..selection.len() as usize);
        find_groups(context, table, batch, at!(keys.build, ..keys.count), groups)?;
        for (row, &group) in (first..).zip(groups.iter()) {
            if group == NONE {
                held.before.push(NONE)?;
                continue;
            }
            let last = last_of(table, group);
            held.repeats |= last != NONE;
            held.before.push(last)?;
            table.set_payload(group, i64::from(row) + 1);
        }
        Ok(())
    }

    /// Starts joining `input`: no row matched, and with keys, each row's
    /// first build row of its keys.
    fn start(&self, context: &mut Context, held: &mut Held, input: &RowBatch) -> Result<(), Error> {
        held.started = true;
        held.next = 0;
        held.matched.fill(0);
        let Some(keys) = &self.keys else { return Ok(()) };
        let selection = input.selection();
        // `next_round` finds them by their keys' values.
        if held.direct.is_some() {
            return Ok(());
        }
        let Some(table) = &mut held.table else { failed!() };
        // Each kept row's group, by its place among them, then its first
        // build row, by its number: from the last, as a row's number is at
        // least its place, so no group is written over before it's read.
        let next_of = &mut *held.next_of;
        let groups = at_mut!(next_of, ..selection.len() as usize);
        find_groups(context, table, input, at!(keys.probe, ..keys.count), groups)?;
        for i in (0..selection.len()).rev() {
            let (group, row) = (*at!(next_of, i as usize), selection.kept().row(i));
            let last = if group == NONE { NONE } else { last_of(table, group) };
            *at_mut!(next_of, row as usize) = last;
        }
        Ok(())
    }

    /// Narrows `pairs`, a copy of the input batch, to its rows with a build
    /// row this round, writing each one's to `paired`, and readies the next
    /// round. Returns `None` if none has one, else whether any has a build
    /// row after this round's.
    fn next_round(
        &self,
        context: &mut Context,
        held: &mut Held,
        pairs: &mut RowBatch,
    ) -> Result<Option<bool>, Error> {
        if self.keys.is_none() {
            if held.next == held.count {
                return Ok(None);
            }
            at_mut!(held.paired, ..pairs.row_count() as usize).fill(held.next);
            held.next += 1;
            return Ok((!pairs.selection().is_empty()).then_some(held.next < held.count));
        }
        if let (Some(direct), Some(keys)) = (&held.direct, &self.keys) {
            // One round: each row's build row found by its key's value.
            if held.next > 0 {
                return Ok(None);
            }
            held.next = 1;
            let paired = at_mut!(held.paired, ..pairs.row_count() as usize);
            direct.pair(context, *at!(keys.probe, 0), held.count, paired, pairs)?;
            return Ok((!pairs.selection().is_empty()).then_some(false));
        }
        let (paired, next_of, before) = (&mut held.paired, &mut held.next_of, &held.before);
        if !held.repeats {
            // One round, the first: a row without a build row takes the
            // first's, as every row is gathered.
            if held.next > 0 {
                return Ok(None);
            }
            held.next = 1;
            pairs.selection_mut().retain(|row| {
                let build = *at!(next_of, usize::from(row));
                *at_mut!(paired, usize::from(row)) = if build == NONE { 0 } else { build };
                build != NONE
            });
            return Ok((!pairs.selection().is_empty()).then_some(false));
        }
        // Rows left out keep the build rows they had, which every row has,
        // as they're gathered too.
        let mut more = false;
        pairs.selection_mut().retain(|row| {
            let build = *at!(next_of, usize::from(row));
            if build == NONE {
                return false;
            }
            let next = *at!(before, build as usize);
            *at_mut!(paired, usize::from(row)) = build;
            *at_mut!(next_of, usize::from(row)) = next;
            more |= next != NONE;
            true
        });
        Ok((!pairs.selection().is_empty()).then_some(more))
    }

    /// Adds to `pairs` the build columns of each row's build row this
    /// round, and keeps the rows the condition is true for.
    fn pair(
        &self,
        context: &mut Context,
        held: &mut Held,
        pairs: &mut RowBatch,
    ) -> Result<(), Error> {
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's columns")]
        let outputs = pairs.column_count() + self.outputs as u32;
        if self.held > 0 {
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
        // Only the condition read the columns after the output's.
        while pairs.column_count() > outputs {
            pairs.pop_column();
        }
        Ok(())
    }

    /// Outputs the input's rows of this kind once every round is done: a
    /// left join's unmatched rows, and an inner join's none.
    fn finish_input(&self, held: &mut Held, input: &RowBatch, output: &mut RowBatch) {
        held.started = false;
        output.copy_from(input);
        let matched = &held.matched;
        let left = self.keeps_probe();
        output.selection_mut().retain(|row| left && !is_matched(matched, row));
        for null in held.nulls.iter() {
            let null = null.slice(0, input.row_count());
            let Ok(()) = output.push_column(null) else { failed!() };
        }
    }
}

impl Operator for Join<'_> {
    type State = Held;

    fn new_state(&self, context: &mut Context) -> Result<Held, Error> {
        let allocator = context.allocator();
        let rows = RowStore::new(allocator)?;
        let nulls = null_columns(context, at!(self.build_types, ..self.outputs))?;
        let probe_nulls = null_columns(context, &self.probe_types)?;
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
            direct: None,
            // Not arrays, which would be copied into place on each run.
            sequence: if self.one_integer() { Sequence::Empty } else { Sequence::Broken },
            next_of: SlowVec::zeroed(allocator, BATCH_ROWS_MAX as usize)?,
            matched: [0; BATCH_ROWS_MAX as usize / 64],
            nulls,
            paired: SlowVec::zeroed(allocator, BATCH_ROWS_MAX as usize)?,
            condition,
            build_matched: SlowVec::zeroed(allocator, 0)?,
            unmatched: 0,
            probe_nulls,
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
        if held.count == 0 && !self.keeps_probe() {
            return Ok(Progress::Done);
        }
        if !held.started {
            self.start(context, held, input)?;
        }
        loop {
            output.copy_from(input);
            let Some(more) = self.next_round(context, held, output)? else { break };
            self.pair(context, held, output)?;
            let selection = output.selection();
            // Only a left join outputs rows by whether they matched.
            for i in (0..selection.len()).filter(|_| self.keeps_probe()) {
                let kept = selection.kept().row(i);
                *at_mut!(held.matched, kept as usize / 64) |= 1 << (kept % 64);
            }
            for i in (0..selection.len()).filter(|_| self.keeps_build) {
                let build = *at!(held.paired, selection.kept().row(i) as usize);
                *at_mut!(held.build_matched, build as usize / 64) |= 1 << (build % 64);
            }
            if selection.is_empty() {
                continue;
            }
            // A join that doesn't keep probe rows has nothing to add once no
            // row has a build row left: as when each has one, N:1.
            if !self.keeps_probe() && !more {
                held.started = false;
                return Ok(Progress::NeedInput);
            }
            return Ok(Progress::MoreOutput);
        }
        self.finish_input(held, input, output);
        Ok(Progress::NeedInput)
    }

    /// Outputs the build rows no probe row matched, if it keeps them, with
    /// nulls for the probe side's columns, a batch at a time.
    fn finish(
        &self,
        context: &mut Context,
        held: &mut Held,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        if !self.keeps_build {
            return Ok(Progress::NeedInput);
        }
        if !held.drained {
            self.drain(context, held)?;
        }
        let mut count = 0;
        while held.unmatched < held.count && count < BATCH_ROWS_MAX {
            let row = held.unmatched;
            held.unmatched += 1;
            if *at!(held.build_matched, row as usize / 64) >> (row % 64) & 1 == 0 {
                *at_mut!(held.paired, count as usize) = row;
                count += 1;
            }
        }
        if count == 0 {
            return Ok(Progress::NeedInput);
        }
        output.reset(count);
        for null in held.probe_nulls.iter() {
            let Ok(()) = output.push_column(null.slice(0, count)) else { failed!() };
        }
        let mut build = RowBatch::new();
        held.rows.gather_held(context, &mut build, at!(held.paired, ..count as usize))?;
        for c in 0..self.outputs {
            #[expect(clippy::cast_possible_truncation, reason = "at most a batch's columns")]
            let column = build.column(c as u32).clone();
            let Ok(()) = output.push_column(column) else { failed!() };
        }
        let more = held.unmatched < held.count;
        Ok(if more { Progress::MoreOutput } else { Progress::NeedInput })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;
    use crate::estimate::RowEstimate;
    use crate::expression::ExprBuilder;

    /// The child a join of `kind` on `a = b` builds, given its children's
    /// rows and keys.
    fn built(kind: JoinKind, first: (u64, Key), second: (u64, Key)) -> u32 {
        let mut exprs = Exprs::new(&Heap).unwrap();
        let mut nodes = ExprBuilder::new(&Heap).unwrap();
        let (a, b) = (nodes.column(0).unwrap(), nodes.column(1).unwrap());
        nodes.node(ExprOp::Eq, a, b).unwrap();
        let condition = Some(exprs.add(nodes.nodes()).unwrap());
        let estimate = |(rows, key): (u64, Key), column| {
            let mut estimate = Estimate::of(RowEstimate::exactly(rows));
            estimate.add_key(column, key);
            estimate
        };
        let one = |column| SlowVec::fixed_from(&Heap, [column].into_iter()).unwrap();
        let mut join = JoinOp::new(kind, condition, [one(0), one(1)]);
        join.choose(&exprs, &[estimate(first, 0), estimate(second, 1)]);
        join.build
    }

    #[test]
    fn builds_the_side_with_the_better_key_then_fewer_rows() {
        let (few, many) = (10, 1_000_000);
        for kind in [JoinKind::Inner, JoinKind::Left] {
            // An id, even if bigger, then a unique key, then fewer rows.
            assert_eq!(built(kind, (many, Key::Id), (few, Key::Unique)), 0);
            assert_eq!(built(kind, (many, Key::Unique), (few, Key::None)), 0);
            assert_eq!(built(kind, (few, Key::None), (many, Key::None)), 0);
            assert_eq!(built(kind, (many, Key::None), (few, Key::None)), 1);
            // As written when nothing tells them apart.
            assert_eq!(built(kind, (few, Key::Id), (few, Key::Id)), 1);
        }
    }
}
