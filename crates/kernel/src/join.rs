//! `JoinOp`: a join of its two children on any condition. The build side,
//! the child planning chooses, runs as a pipeline of its own and is held
//! whole first.
//! Without keys the condition requires equal on both sides, it's tested on
//! every pair of rows: a nested loop. With them, the build side is hashed by
//! them, and only the pairs of equal keys are tested. With one integer key,
//! unique and of values close enough, a key's build row is found by its
//! value, as a perfect hash finds it: from an array, or with no array if
//! each key is its row's number above the first, as a table's ids are.

use crate::allocator::AllocError;
use crate::column::{ColumnView, DataType, Form, Forms};
use crate::containers::{Array, BitSlice, Bitmap, SmallVec};
use crate::context::Context;
use crate::error::Error;
use crate::estimate::Estimate;
use crate::expression::{
    Compute, ComputeState, Expr, ExprOp, ExprStep, Exprs, check_expr_type, columns_of, constant,
    mark_expr_columns, type_expr,
};
use crate::filter::{self, Comparison, Value};
use crate::group_table::{GroupTable, NONE};
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{Arrangements, Column, DeriveContext, OutputProperties};
use crate::plan::{ColumnId, Conjuncts, Given, Op, PlanNode};
use crate::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};

use crate::env::Env;
use crate::row_store::RowStore;
use crate::scannable::Key;
use crate::selection::{Kept, Selection};
use crate::step::{DynOperator, DynSource, Operator, Progress, Step, Transform};

mod direct;

use direct::{Direct, Sequence, consecutive};

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
    /// Which pairs of rows are joined: those every one of these conditions,
    /// an `Int64`, is true for, not 0 or null. With none, every pair: a
    /// cross join.
    conjuncts: Conjuncts,
    /// Each child's columns, at most a batch's, and a bit for each it
    /// outputs, which pruning clears: planning finds which child a column
    /// comes from by them all.
    columns: [SmallVec<ColumnId>; 2],
    outputs: [u64; 2],
    /// Which child is the build side: the second, unless planning chooses
    /// the first.
    build: u32,
}

impl JoinOp {
    /// A join of `kind` on `condition`, outputting `columns` of each child.
    pub fn new(kind: JoinKind, conjuncts: Conjuncts, columns: [SmallVec<ColumnId>; 2]) -> JoinOp {
        let all = |columns: &SmallVec<ColumnId>| {
            check!(columns.len() <= COLUMNS);
            #[expect(clippy::cast_possible_truncation, reason = "at most `COLUMNS`")]
            let len = columns.len() as u32;
            u64::MAX.checked_shr(64 - len).unwrap_or(0)
        };
        let outputs = [all(at!(columns, 0)), all(at!(columns, 1))];
        JoinOp { kind, conjuncts, columns, outputs, build: 1 }
    }

    pub(crate) fn kind(&self) -> JoinKind {
        self.kind
    }

    pub(crate) fn conjuncts(&self) -> &[Expr] {
        &self.conjuncts
    }

    /// Child `child`'s columns, all of them.
    pub(crate) fn all_columns(&self, child: usize) -> &[ColumnId] {
        at!(self.columns, child)
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

    /// The build side's columns it outputs, and how many.
    fn build_columns(&self) -> ([ColumnId; COLUMNS], usize) {
        let (columns, outputs) = (self.build as usize, *at!(self.outputs, self.build as usize));
        let mut out = [0; COLUMNS];
        let mut count = 0;
        for (i, &column) in at!(self.columns, columns).iter().enumerate() {
            if outputs >> i & 1 != 0 {
                *at_mut!(out, count) = column;
                count += 1;
            }
        }
        (out, count)
    }

    fn is_build_column(&self, column: ColumnId) -> bool {
        at!(self.columns, self.build as usize).iter().any(|&c| c == column)
    }

    /// The best key either side has among the equalities at the top of its
    /// condition: what decides which side to build.
    fn side_keys(&self, exprs: &Exprs, inputs: &[Estimate]) -> [Key; 2] {
        let mut keys = [Key::None; 2];
        for &conjunct in self.conjuncts.iter() {
            let Some(columns) = equality(exprs.nodes(conjunct)) else { continue };
            for (side, key) in keys.iter_mut().enumerate() {
                for column in columns {
                    if at!(self.columns, side).iter().any(|&c| c == column) {
                        *key = (*key).max(at!(inputs, side).key(column));
                    }
                }
            }
        }
        keys
    }

    /// The build columns, then if `tested`, the other columns of `build` the
    /// condition reads: what it holds of each build row, and how many.
    fn held(
        &self,
        exprs: &Exprs,
        build: &OutputProperties,
        tested: &[Expr],
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
        let (outputs, output_count) = self.build_columns();
        for &column in at!(outputs, ..output_count) {
            add(column);
        }
        for &condition in tested {
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
    fn as_join(&self) -> Option<&JoinOp> {
        Some(self)
    }

    /// Keeps the columns needed above.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        for (columns, outputs) in self.columns.iter().zip(&mut self.outputs) {
            for (i, &column) in columns.iter().enumerate() {
                if !needed.is_needed(column) {
                    *outputs &= !(1 << i);
                }
            }
        }
        self.reads(needed);
        Pruned::Keep
    }

    /// Its condition's columns, and the build columns, which it copies.
    fn reads(&self, reads: &mut Needed) {
        for &condition in self.conjuncts.iter() {
            mark_expr_columns(condition, reads);
        }
        for (columns, outputs) in self.columns.iter().zip(&self.outputs) {
            for (i, &column) in columns.iter().enumerate() {
                if outputs >> i & 1 != 0 {
                    reads.need(column);
                }
            }
        }
    }

    /// Sequences of what can be its keys, as a table's ids come: a build
    /// side's it holds no hash table for.
    fn accepts(&self, exprs: &Exprs, column: ColumnId) -> Forms {
        let key = self.conjuncts.iter().any(|&conjunct| {
            equality(exprs.nodes(conjunct)).is_some_and(|columns| columns.contains(&column))
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
        _: &Env,
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
        let (held, count) = self.held(c.exprs(), c.input(build), &self.conjuncts);
        let pairs = self.pairs(c.input(probe), c.input(build), at!(held, ..count))?;
        for &condition in self.conjuncts.iter() {
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
        let (outputs, count) = self.build_columns();
        let mut out = self.pairs(&out, c.input(build), at!(outputs, ..count))?;
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
        let env = lowering.env();
        let (build, rows) = lowering.lower_side(node, self.build)?;
        lowering.lower_input(node, self.probe())?;
        let probe = lowering.query_rows();
        let exprs = lowering.exprs();
        // The conjuncts it hashes by need no test once their keys are found
        // equal; the rest are computed, a column each, and tested.
        let mut tested = SmallVec::fixed(env, self.conjuncts.len())?;
        let keys = Self::keys(exprs, probe, rows, &self.conjuncts, &mut tested);
        let mut condition = None;
        if !tested.is_empty() {
            let (held, count) = self.held(exprs, rows, &tested);
            let pairs = self.pairs(probe, rows, at!(held, ..count))?;
            let mut programs = SmallVec::fixed(env, tested.len())?;
            for &conjunct in tested.iter() {
                let (nodes, values) = (exprs.nodes(conjunct), lowering.expr_values());
                let Ok(program) = type_expr(env, &pairs, nodes, values, true) else { failed!() };
                programs.push(program)?;
            }
            condition = Some(Compute { programs });
        }
        let (held, count) = self.held(exprs, rows, &tested);
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
        let (outputs, count) = self.build_columns();
        for (data_type, &column) in build_types.iter_mut().zip(at!(outputs, ..count)) {
            *data_type = rows.type_of(column);
        }
        let (held, outputs) = (held.len(), count);
        let mut probe_types = SmallVec::fixed(env, probe.column_count() as usize)?;
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
        lowering.add_operator(Step::Operator(DynOperator::new(env, join)?))
    }
}

impl JoinOp {
    /// The keys to hash the build side by, if any of `conjuncts` requires a
    /// probe column equal to a build column of a type a hash table keys; and
    /// in `tested`, the rest of them. A row with a null key, equal to none,
    /// is paired with none.
    fn keys(
        exprs: &Exprs,
        probe: &OutputProperties,
        build: &OutputProperties,
        conjuncts: &[Expr],
        tested: &mut SmallVec<Expr>,
    ) -> Option<Keys> {
        let mut keys =
            Keys { probe: [0; JOIN_KEYS_MAX], build: [0; JOIN_KEYS_MAX], count: 0, strings: 0 };
        for &conjunct in conjuncts {
            let key = equality(exprs.nodes(conjunct)).and_then(|[a, b]| {
                let (p, q) = if probe.position_of(a).is_some() { (a, b) } else { (b, a) };
                let (at, index) = (probe.position_of(p)?, build.position_of(q)?);
                let data_type = probe.type_of(p);
                let hashed = data_type == build.type_of(q) && data_type != DataType::Float64;
                (hashed && keys.count < JOIN_KEYS_MAX).then_some((at, index, data_type))
            });
            let Some((at, index, data_type)) = key else {
                tested.extend_within(core::iter::once(conjunct));
                continue;
            };
            if data_type == DataType::String {
                keys.strings |= 1 << keys.count;
            }
            *at_mut!(keys.probe, keys.count) = at;
            *at_mut!(keys.build, keys.count) = index;
            keys.count += 1;
        }
        (keys.count > 0).then_some(keys)
    }
}

/// The columns `nodes`, a conjunct, requires equal, if it's an equality of
/// two columns.
pub(crate) fn equality(nodes: &[ExprStep]) -> Option<[ColumnId; 2]> {
    let root = nodes.last()?;
    let column = |i: u32| nodes.get(i as usize).filter(|n| n.op == ExprOp::Column).map(|n| n.a);
    (root.op == ExprOp::Eq).then_some([column(root.a)?, column(root.b)?])
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
    probe_types: SmallVec<DataType>,
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
    before: SmallVec<u32>,
    repeats: bool,
    /// With one integer key, unique, of values close enough, how to find
    /// a key's build row by its value, in place of `table`.
    direct: Option<Direct>,
    /// With one integer key, the first build row's key while each build
    /// row's has been its number above it, as a table's ids are: then none
    /// is hashed, as `direct` finds them all.
    sequence: Sequence,
    /// With keys, each input row's build row of the next round, or `NONE`.
    next_of: Array<u32>,
    /// A bit for each row of the batch being joined, set once a build row
    /// matched it.
    matched: [u64; BATCH_ROWS_MAX as usize / 64],
    /// For an inner or left join, a null column of each build column's
    /// type, a batch long: an unmatched row's.
    nulls: SmallVec<ColumnView>,
    /// Each input row's build row in this round, for `gather`: always a
    /// build row's number, as every row is gathered.
    paired: Array<u32>,
    /// What computing the condition keeps.
    condition: Option<ComputeState>,
    /// When it keeps the build rows: a bit for each, set once a probe row
    /// matched it; the next it hasn't looked at for output at the end; and
    /// a null column of each probe column's type, a batch long.
    build_matched: Bitmap,
    unmatched: u32,
    probe_nulls: SmallVec<ColumnView>,
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

/// A null column of each of `types`, a batch long.
fn null_columns(context: &mut Context, types: &[DataType]) -> Result<SmallVec<ColumnView>, Error> {
    let mut nulls = SmallVec::fixed(context.env(), types.len())?;
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
    BitSlice::of(matched).get(usize::from(row))
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
            held.direct = Some(Direct::sequence(min, held.count));
            held.table = None;
        } else if let (true, false, Some(table)) = (self.one_integer(), held.repeats, &held.table) {
            held.direct = Direct::of(context.env(), table, held.count)?;
            if held.direct.is_some() {
                held.table = None;
            }
        }
        if self.keeps_build {
            held.build_matched = Bitmap::new(context.env(), held.count as usize)?;
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
            let first = pairs.column_count() as usize;
            condition.process(context, state, pairs)?;
            let (columns, selection) = pairs.columns_and_selection();
            for truth in at!(columns, first..) {
                filter::compare(truth, Comparison::NotEqual, Value::Int64(0), selection, None);
            }
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
        let env = context.env();
        let rows = RowStore::new(env)?;
        let nulls = null_columns(context, at!(self.build_types, ..self.outputs))?;
        let probe_nulls = null_columns(context, &self.probe_types)?;
        let condition = match &self.condition {
            Some(condition) => Some(condition.new_state(context)?),
            None => None,
        };
        // A word a group after its keys, for its last build row.
        let table = match &self.keys {
            Some(keys) => Some(GroupTable::new(env, keys.count, keys.strings, 0, 1)?),
            None => None,
        };
        Ok(Held {
            drained: false,
            rows,
            count: 0,
            started: false,
            next: 0,
            table,
            before: SmallVec::new(env, 1 << 31)?,
            repeats: false,
            direct: None,
            // Not arrays, which would be copied into place on each run.
            sequence: if self.one_integer() { Sequence::Empty } else { Sequence::Broken },
            next_of: Array::zeroed(env, BATCH_ROWS_MAX as usize)?,
            matched: [0; BATCH_ROWS_MAX as usize / 64],
            nulls,
            paired: Array::zeroed(env, BATCH_ROWS_MAX as usize)?,
            condition,
            build_matched: Bitmap::new(env, 0)?,
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
                BitSlice::of_mut(&mut held.matched).set(kept as usize);
            }
            for i in (0..selection.len()).filter(|_| self.keeps_build) {
                let build = *at!(held.paired, selection.kept().row(i) as usize);
                held.build_matched.set(build as usize);
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
            if !held.build_matched.get(row as usize) {
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

    use crate::estimate::RowEstimate;
    use crate::expression::ExprBuilder;

    /// The child a join of `kind` on `a = b` builds, given its children's
    /// rows and keys.
    fn built(kind: JoinKind, first: (u64, Key), second: (u64, Key)) -> u32 {
        let mut exprs = Exprs::new(Env::current()).unwrap();
        let mut nodes = ExprBuilder::new(Env::current()).unwrap();
        let (a, b) = (nodes.column(0).unwrap(), nodes.column(1).unwrap());
        nodes.node(ExprOp::Eq, a, b).unwrap();
        let condition = Some(exprs.add(nodes.nodes()).unwrap());
        let estimate = |(rows, key): (u64, Key), column| {
            let mut estimate = Estimate::of(RowEstimate::exactly(rows));
            estimate.add_key(column, key);
            estimate
        };
        let one = |column| SmallVec::fixed_from(Env::current(), [column].into_iter()).unwrap();
        let conjuncts = SmallVec::fixed_from(Env::current(), condition.into_iter()).unwrap();
        let mut join = JoinOp::new(kind, conjuncts, [one(0), one(1)]);
        join.choose(&exprs, &[estimate(first, 0), estimate(second, 1)]);
        join.build
    }

    #[test]
    fn builds_by_keys_nothing_above_reads() {
        // Joined on 0 = 1, of which only 2 and 3 are read above: pruned,
        // the join still knows its first side's key is an id.
        let mut exprs = Exprs::new(Env::current()).unwrap();
        let mut nodes = ExprBuilder::new(Env::current()).unwrap();
        let (a, b) = (nodes.column(0).unwrap(), nodes.column(1).unwrap());
        nodes.node(ExprOp::Eq, a, b).unwrap();
        let condition = exprs.add(nodes.nodes()).unwrap();
        let conjuncts = SmallVec::fixed_from(Env::current(), [condition].into_iter()).unwrap();
        let two = |a, b| SmallVec::fixed_from(Env::current(), [a, b].into_iter()).unwrap();
        let mut join = JoinOp::new(JoinKind::Inner, conjuncts, [two(0, 2), two(1, 3)]);
        let mut needed = Needed::none(Env::current(), 4, &exprs).unwrap();
        needed.need(2);
        needed.need(3);
        join.prune(&mut needed);
        let mut first = Estimate::of(RowEstimate::exactly(1_000_000));
        first.add_key(0, Key::Id);
        join.choose(&exprs, &[first, Estimate::of(RowEstimate::exactly(10))]);
        assert_eq!(join.build, 0);
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
