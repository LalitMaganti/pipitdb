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
use crate::buffer::Buffer;
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
use crate::pipeline::{SinkId, with_built};
use crate::row_store::RowStore;
use crate::scannable::Key;
use crate::selection::{Kept, Selection};
use crate::step::{DynOperator, Operator, Progress, Sink, Step, Transform};

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
    /// Each child's columns, and a bit for each it outputs, which pruning
    /// clears: planning finds which child a column comes from by them all.
    /// Only those it outputs need fit in a batch.
    columns: [SmallVec<ColumnId>; 2],
    outputs: [Bitmap; 2],
    /// Which child is the build side: the second, unless planning chooses
    /// the first.
    build: u32,
}

impl JoinOp {
    /// A join of `kind` on `condition`, outputting `columns` of each child.
    pub fn new(
        env: &Env,
        kind: JoinKind,
        conjuncts: Conjuncts,
        columns: [SmallVec<ColumnId>; 2],
    ) -> Result<JoinOp, AllocError> {
        let all = |columns: &SmallVec<ColumnId>| {
            let mut outputs = Bitmap::new(env, columns.len())?;
            for i in 0..columns.len() {
                outputs.set(i);
            }
            Ok::<_, AllocError>(outputs)
        };
        let outputs = [all(at!(columns, 0))?, all(at!(columns, 1))?];
        Ok(JoinOp { kind, conjuncts, columns, outputs, build: 1 })
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

    /// The build side's columns it outputs, and how many: no more than a
    /// batch holds.
    fn build_columns(&self) -> Result<([ColumnId; COLUMNS], usize), LowerError> {
        let outputs = at!(self.outputs, self.build as usize);
        let mut out = [0; COLUMNS];
        let mut count = 0;
        for (i, &column) in at!(self.columns, self.build as usize).iter().enumerate() {
            if outputs.get(i) {
                *out.get_mut(count).ok_or(LowerError::TooManyColumns)? = column;
                count += 1;
            }
        }
        Ok((out, count))
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
    ) -> Result<([ColumnId; COLUMNS], usize), LowerError> {
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
        let (outputs, output_count) = self.build_columns()?;
        for &column in at!(outputs, ..output_count) {
            add(column);
        }
        for &condition in tested {
            columns_of(exprs.nodes(condition), condition.root, &mut add);
        }
        Ok((held, count))
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
                    outputs.clear(i);
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
                if outputs.get(i) {
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

    /// Builds a side whose key is an id, as it's never hashed; otherwise the
    /// side whose hash table is smaller by planning's estimates: its rows,
    /// each a word for every column it outputs and one for its entry. Of
    /// sides of one size, one whose key is unique, as each probe row then
    /// has one build row. A left join that builds its first child outputs
    /// its unmatched rows at the end.
    fn choose(&mut self, exprs: &Exprs, inputs: &[Estimate]) {
        let &[first, second] = inputs else { failed!() };
        let keys = self.side_keys(exprs, inputs);
        let rank = |side: usize, input: &Estimate| {
            let key = *at!(keys, side);
            let outputs = at!(self.outputs, side).count_ones();
            let bytes = input.rows.estimated.saturating_mul(u64::from(outputs) + 1);
            (key == Key::Id, u64::MAX - bytes, key)
        };
        // As written if nothing tells them apart.
        self.build = u32::from(rank(0, &first) <= rank(1, &second));
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
        let (held, count) = self.held(c.exprs(), c.input(build), &self.conjuncts)?;
        let pairs = self.pairs(c.input(probe), c.input(build), at!(held, ..count))?;
        for &condition in self.conjuncts.iter() {
            let nodes = c.exprs().nodes(condition);
            let data_type = check_expr_type(c.env(), nodes, &|column| pairs.type_of(column));
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
        let (outputs, count) = self.build_columns()?;
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
            let (held, count) = self.held(exprs, rows, &tested)?;
            let pairs = self.pairs(probe, rows, at!(held, ..count))?;
            let mut programs = SmallVec::fixed(env, tested.len())?;
            for &conjunct in tested.iter() {
                let (nodes, values) = (exprs.nodes(conjunct), lowering.expr_values());
                let Ok(program) = type_expr(env, &pairs, nodes, values, true) else { failed!() };
                programs.push(program)?;
            }
            condition = Some(Compute { programs });
        }
        let (held, count) = self.held(exprs, rows, &tested)?;
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
        let (outputs, count) = self.build_columns()?;
        for (data_type, &column) in build_types.iter_mut().zip(at!(outputs, ..count)) {
            *data_type = rows.type_of(column);
        }
        let (held, outputs) = (held.len(), count);
        let mut probe_types = SmallVec::fixed(env, probe.column_count() as usize)?;
        if self.keeps_build() {
            probe_types.extend_within(probe.types());
        }
        let build = lowering.add_sink(build, HashBuild { build_columns, held, keys })?;
        let join = Join {
            kind: self.kind,
            build,
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
/// rest of its condition is. Finding groups holds as many columns on the
/// stack each batch, so it's no more than joins need: some get more keys
/// than written, as the planner adds the equalities between relations.
const JOIN_KEYS_MAX: usize = 16;

/// The columns a join hashes its build side by, which its condition
/// requires equal on both sides.
#[derive(Clone, Copy)]
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
/// each row with its next build row of the same keys, found by hashing. Its
/// `HashBuild` is built on the first input: an empty probe side never reads
/// the build side.
struct Join {
    /// Which rows it outputs.
    kind: JoinKind,
    /// What holds the build side.
    build: SinkId<HashBuild>,
    /// How many build columns it holds.
    held: usize,
    /// The types of the build columns it holds.
    build_types: [DataType; BATCH_COLUMNS_MAX as usize],
    /// How many of the build columns it holds it outputs.
    outputs: usize,
    /// Computes the condition, as a column after a pair's.
    condition: Option<Compute>,
    /// The keys it hashes by, if any.
    keys: Option<Keys>,
    /// Whether it outputs the build rows no probe row matched, at the end,
    /// with nulls of `probe_types` for the probe side's columns.
    keeps_build: bool,
    /// The types of the probe side's columns.
    probe_types: SmallVec<DataType>,
}

/// Holds every row of a join's build side, and with keys, finds each one's
/// group in a hash table, for the join to probe.
struct HashBuild {
    /// Where the first `held` build columns are in the build side's
    /// batches: those the join outputs, then those only its condition reads.
    build_columns: [u32; BATCH_COLUMNS_MAX as usize],
    /// How many build columns it holds.
    held: usize,
    /// The keys it hashes by, if any.
    keys: Option<Keys>,
}

/// What a run of a `HashBuild`'s pipeline builds.
struct Building {
    /// The build side so far, until it's combined.
    built: Option<Built>,
    /// Each batch's groups, as hashing finds them: a batch of `u32`s.
    groups: Buffer,
}

/// What draining the build side makes.
struct Built {
    /// The build columns of every build row, in order.
    rows: RowStore,
    /// How many build rows there are.
    count: u32,
    /// With keys, each build row's group of keys, which holds its last build
    /// row plus one.
    table: Option<GroupTable>,
    /// With keys, each build row's one before it of its group, or `NONE`.
    before: SmallVec<u32>,
    /// With keys, whether any group has more than one build row.
    repeats: bool,
    /// With one integer key, unique, of values close enough, how to find
    /// a key's build row by its value, in place of `table`.
    direct: Option<Direct>,
    /// With one integer key, the first build row's key while each build
    /// row's has been its number above it, as a table's ids are: then none
    /// is hashed, as `direct` finds them all.
    sequence: Sequence,
}

/// What probing keeps, from one input batch to the next.
struct Probing {
    /// Whether it's read the build side, once built.
    read_build: bool,
    /// Whether the input batch has had its first round.
    started: bool,
    /// Without keys, the build row of the next round; with keys, how many
    /// rounds there have been.
    next: u32,
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
    /// When it keeps the build rows, a bit for each, set once a probe row
    /// matched it.
    build_matched: Bitmap,
    /// When it keeps the build rows, the next it hasn't looked at for
    /// output at the end.
    unmatched: u32,
    /// When it keeps the build rows, a null column of each probe column's
    /// type, a batch long.
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

impl HashBuild {
    /// Whether it joins on one integer key.
    fn one_integer(&self) -> bool {
        self.keys.as_ref().is_some_and(|keys| keys.count == 1 && keys.strings == 0)
    }

    /// Finds the groups of `batch`'s keys, its rows numbered from `first`,
    /// in the hash table, working in `next_of`: unless its one integer key
    /// continues a sequence, which needs none.
    fn hash(
        context: &mut Context,
        built: &mut Built,
        next_of: &mut [u32],
        keys: &Keys,
        batch: &RowBatch,
        first: u32,
    ) -> Result<(), Error> {
        let selection = batch.selection();
        let key = batch.column(*at!(keys.build, 0));
        if let (Some(start), true) =
            (consecutive(key, selection.len(), built.sequence), selection.is_prefix())
        {
            let follows = match built.sequence {
                Sequence::Empty => Some(start),
                Sequence::From(min) => {
                    (min.checked_add(i64::from(first)) == Some(start)).then_some(min)
                }
                Sequence::Broken => None,
            };
            if let Some(min) = follows {
                built.sequence = Sequence::From(min);
                return Ok(());
            }
        }
        let Some(table) = &mut built.table else { failed!() };
        if let Sequence::From(min) = built.sequence {
            // The rows before, held as a sequence, are hashed now.
            let mut row = 0;
            while row < first {
                let count = (first - row).min(BATCH_ROWS_MAX);
                let mut keys = ColumnView::sequence(context, min + i64::from(row), count)?;
                keys.make_in(context, Forms::FLAT)?;
                let groups = at_mut!(next_of, ..count as usize);
                table.find_or_create(context, &[&keys], &[], groups)?;
                for &group in groups.iter() {
                    built.before.push(NONE)?;
                    table.set_payload(group, i64::from(row) + 1);
                    row += 1;
                }
            }
        }
        built.sequence = Sequence::Broken;
        let groups = at_mut!(next_of, ..selection.len() as usize);
        find_groups(context, table, batch, at!(keys.build, ..keys.count), groups)?;
        for (row, &group) in (first..).zip(groups.iter()) {
            if group == NONE {
                built.before.push(NONE)?;
                continue;
            }
            let last = last_of(table, group);
            built.repeats |= last != NONE;
            built.before.push(last)?;
            table.set_payload(group, i64::from(row) + 1);
        }
        Ok(())
    }
}

impl Sink for HashBuild {
    type State = Option<Built>;
    type Local = Building;

    fn new_state(&self, _: &mut Context) -> Result<Option<Built>, Error> {
        Ok(None)
    }

    fn new_local(&self, context: &mut Context) -> Result<Building, Error> {
        let env = context.env();
        // A word a group after its keys, for its last build row.
        let table = match &self.keys {
            Some(keys) => Some(GroupTable::new(env, keys.count, keys.strings, 0, 1)?),
            None => None,
        };
        let built = Built {
            rows: RowStore::new(env)?,
            count: 0,
            table,
            before: SmallVec::new(env, 1 << 31)?,
            repeats: false,
            direct: None,
            sequence: if self.one_integer() { Sequence::Empty } else { Sequence::Broken },
        };
        let groups = context.indices_buffer(BATCH_ROWS_MAX as usize * size_of::<u32>())?;
        Ok(Building { built: Some(built), groups })
    }

    fn consume(
        &self,
        context: &mut Context,
        building: &mut Building,
        batch: &RowBatch,
    ) -> Result<(), Error> {
        let Some(built) = &mut building.built else { failed!() };
        let selection = batch.selection();
        let first = built.count;
        built.count = built.count.checked_add(selection.len()).ok_or(Error::OutOfMemory)?;
        if selection.is_empty() {
            return Ok(());
        }
        if let Some(keys) = &self.keys {
            let groups = building.groups.as_mut_slice::<u32>();
            Self::hash(context, built, groups, keys, batch, first)?;
        }
        let build_columns = at!(self.build_columns, ..self.held);
        if build_columns.is_empty() {
            return Ok(());
        }
        let mut columns = RowBatch::new();
        columns.reset(batch.row_count());
        if build_columns.iter().all(|&c| matches!(batch.column(c).form(), Form::Flat)) {
            columns.push_borrowed(build_columns.iter().map(|&c| batch.column(c)));
        } else {
            // A row store holds flat columns, and keys can come as sequences.
            for &c in build_columns {
                let mut column = batch.column(c).clone();
                column.make_in(context, Forms::FLAT)?;
                let Ok(()) = columns.push_column(column) else { failed!() };
            }
        }
        columns.selection_mut().clone_from(selection);
        built.rows.append(context, &columns)
    }

    /// Takes what the one run built: merging runs' comes with the executor
    /// that runs more than one.
    fn combine(
        &self,
        _: &mut Context,
        state: &mut Option<Built>,
        building: &mut Building,
    ) -> Result<(), Error> {
        check!(state.is_none());
        *state = building.built.take();
        Ok(())
    }

    fn finalize(&self, context: &mut Context, state: &mut Option<Built>) -> Result<(), Error> {
        let Some(built) = state else { failed!() };
        if let Some(table) = &mut built.table {
            table.freeze();
        }
        // One integer key, each in one build row, is found by its value.
        if let Sequence::From(min) = built.sequence {
            built.direct = Some(Direct::sequence(min, built.count));
            built.table = None;
        } else if let (true, false, Some(table)) = (self.one_integer(), built.repeats, &built.table)
        {
            built.direct = Direct::of(context.env(), table, built.count)?;
            if built.direct.is_some() {
                built.table = None;
            }
        }
        built.rows.flush(context)
    }
}

impl Join {
    /// Whether it outputs the probe rows no build row matched.
    fn keeps_probe(&self) -> bool {
        self.kind == JoinKind::Left && !self.keeps_build
    }

    /// Starts joining `input`: no row matched, and with keys, each row's
    /// first build row of its keys.
    fn start(
        &self,
        context: &mut Context,
        built: &mut Built,
        probing: &mut Probing,
        input: &RowBatch,
    ) -> Result<(), Error> {
        probing.started = true;
        probing.next = 0;
        probing.matched.fill(0);
        let Some(keys) = &self.keys else { return Ok(()) };
        let selection = input.selection();
        // `next_round` finds them by their keys' values.
        if built.direct.is_some() {
            return Ok(());
        }
        let Some(table) = &mut built.table else { failed!() };
        // Each kept row's group, by its place among them, then its first
        // build row, by its number: from the last, as a row's number is at
        // least its place, so no group is written over before it's read.
        let next_of = &mut *probing.next_of;
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
        built: &Built,
        probing: &mut Probing,
        pairs: &mut RowBatch,
    ) -> Result<Option<bool>, Error> {
        if self.keys.is_none() {
            if probing.next == built.count {
                return Ok(None);
            }
            at_mut!(probing.paired, ..pairs.row_count() as usize).fill(probing.next);
            probing.next += 1;
            return Ok((!pairs.selection().is_empty()).then_some(probing.next < built.count));
        }
        if let (Some(direct), Some(keys)) = (&built.direct, &self.keys) {
            // One round: each row's build row found by its key's value.
            if probing.next > 0 {
                return Ok(None);
            }
            probing.next = 1;
            let paired = at_mut!(probing.paired, ..pairs.row_count() as usize);
            direct.pair(context, *at!(keys.probe, 0), built.count, paired, pairs)?;
            return Ok((!pairs.selection().is_empty()).then_some(false));
        }
        let (paired, next_of, before) = (&mut probing.paired, &mut probing.next_of, &built.before);
        if !built.repeats {
            // One round, the first: a row without a build row takes the
            // first's, as every row is gathered.
            if probing.next > 0 {
                return Ok(None);
            }
            probing.next = 1;
            pairs.selection_mut().retain(|row| {
                let build_row = *at!(next_of, usize::from(row));
                *at_mut!(paired, usize::from(row)) = if build_row == NONE { 0 } else { build_row };
                build_row != NONE
            });
            return Ok((!pairs.selection().is_empty()).then_some(false));
        }
        // Rows left out keep the build rows they had, which every row has,
        // as they're gathered too.
        let mut more = false;
        pairs.selection_mut().retain(|row| {
            let build_row = *at!(next_of, usize::from(row));
            if build_row == NONE {
                return false;
            }
            let next = *at!(before, build_row as usize);
            *at_mut!(paired, usize::from(row)) = build_row;
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
        built: &mut Built,
        probing: &mut Probing,
        pairs: &mut RowBatch,
    ) -> Result<(), Error> {
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's columns")]
        let outputs = pairs.column_count() + self.outputs as u32;
        if self.held > 0 {
            let paired = at!(probing.paired, ..pairs.row_count() as usize);
            let mut gathered = RowBatch::new();
            built.rows.gather_held(context, &mut gathered, paired)?;
            for c in 0..gathered.column_count() {
                let Ok(()) = pairs.push_column(gathered.column(c).clone()) else { failed!() };
            }
        }
        if let (Some(condition), Some(state)) = (&self.condition, &mut probing.condition) {
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
    fn finish_input(&self, probing: &mut Probing, input: &RowBatch, output: &mut RowBatch) {
        probing.started = false;
        output.copy_from(input);
        let matched = &probing.matched;
        let left = self.keeps_probe();
        output.selection_mut().retain(|row| left && !is_matched(matched, row));
        for null in probing.nulls.iter() {
            let null = null.slice(0, input.row_count());
            let Ok(()) = output.push_column(null) else { failed!() };
        }
    }
}

impl Join {
    /// Calls `f` with the build side, built the first time, and `probing`,
    /// which the first time readies for the build side's rows.
    fn with_build<R>(
        &self,
        context: &mut Context,
        probing: &mut Probing,
        f: impl FnOnce(&mut Context, &mut Built, &mut Probing) -> Result<R, Error>,
    ) -> Result<R, Error> {
        with_built(context, self.build, |context, _, built| {
            let Some(built) = built else { failed!() };
            if !probing.read_build {
                probing.read_build = true;
                if self.keeps_build {
                    probing.build_matched = Bitmap::new(context.env(), built.count as usize)?;
                }
            }
            f(context, built, probing)
        })?
    }
}

impl Operator for Join {
    type State = Probing;

    fn new_state(&self, context: &mut Context) -> Result<Probing, Error> {
        let env = context.env();
        let nulls = null_columns(context, at!(self.build_types, ..self.outputs))?;
        let probe_nulls = null_columns(context, &self.probe_types)?;
        let condition = match &self.condition {
            Some(condition) => Some(condition.new_state(context)?),
            None => None,
        };
        Ok(Probing {
            read_build: false,
            started: false,
            next: 0,
            // Not arrays, which would be copied into place on each run.
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
        probing: &mut Probing,
        input: &RowBatch,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        self.with_build(context, probing, |context, built, probing| {
            self.probe(context, built, probing, input, output)
        })
    }

    /// Outputs the build rows no probe row matched, if it keeps them, with
    /// nulls for the probe side's columns, a batch at a time.
    fn finish(
        &self,
        context: &mut Context,
        probing: &mut Probing,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        if !self.keeps_build {
            return Ok(Progress::NeedInput);
        }
        self.with_build(context, probing, |context, built, probing| {
            self.unmatched(context, built, probing, output)
        })
    }
}

impl Join {
    /// Joins `input` with the build side, as `execute`.
    fn probe(
        &self,
        context: &mut Context,
        built: &mut Built,
        probing: &mut Probing,
        input: &RowBatch,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        // No probe row can be joined.
        if built.count == 0 && !self.keeps_probe() {
            return Ok(Progress::Done);
        }
        if !probing.started {
            self.start(context, built, probing, input)?;
        }
        loop {
            output.copy_from(input);
            let Some(more) = self.next_round(context, built, probing, output)? else { break };
            self.pair(context, built, probing, output)?;
            let selection = output.selection();
            // Only a left join outputs rows by whether they matched.
            for i in (0..selection.len()).filter(|_| self.keeps_probe()) {
                let kept = selection.kept().row(i);
                BitSlice::of_mut(&mut probing.matched).set(kept as usize);
            }
            for i in (0..selection.len()).filter(|_| self.keeps_build) {
                let build_row = *at!(probing.paired, selection.kept().row(i) as usize);
                probing.build_matched.set(build_row as usize);
            }
            if selection.is_empty() {
                continue;
            }
            // A join that doesn't keep probe rows has nothing to add once no
            // row has a build row left: as when each has one, N:1.
            if !self.keeps_probe() && !more {
                probing.started = false;
                return Ok(Progress::NeedInput);
            }
            return Ok(Progress::MoreOutput);
        }
        self.finish_input(probing, input, output);
        Ok(Progress::NeedInput)
    }

    /// The build rows no probe row matched, a batch at a time, as `finish`.
    fn unmatched(
        &self,
        context: &mut Context,
        built: &mut Built,
        probing: &mut Probing,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        let mut count = 0;
        while probing.unmatched < built.count && count < BATCH_ROWS_MAX {
            let row = probing.unmatched;
            probing.unmatched += 1;
            if !probing.build_matched.get(row as usize) {
                *at_mut!(probing.paired, count as usize) = row;
                count += 1;
            }
        }
        if count == 0 {
            return Ok(Progress::NeedInput);
        }
        output.reset(count);
        for null in probing.probe_nulls.iter() {
            let Ok(()) = output.push_column(null.slice(0, count)) else { failed!() };
        }
        let mut gathered = RowBatch::new();
        built.rows.gather_held(context, &mut gathered, at!(probing.paired, ..count as usize))?;
        for c in 0..self.outputs {
            #[expect(clippy::cast_possible_truncation, reason = "at most a batch's columns")]
            let column = gathered.column(c as u32).clone();
            let Ok(()) = output.push_column(column) else { failed!() };
        }
        let more = probing.unmatched < built.count;
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
        let mut join = JoinOp::new(Env::current(), kind, conjuncts, [one(0), one(1)]).unwrap();
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
        let mut join =
            JoinOp::new(Env::current(), JoinKind::Inner, conjuncts, [two(0, 2), two(1, 3)])
                .unwrap();
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
    fn has_sides_wider_than_a_batch_until_pruned() {
        let env = Env::current();
        let wide = SmallVec::fixed_from(env, 0..100).unwrap();
        let narrow = SmallVec::fixed_from(env, [100].into_iter()).unwrap();
        let none = SmallVec::fixed(env, 0).unwrap();
        let mut join = JoinOp::new(env, JoinKind::Inner, none, [wide, narrow]).unwrap();
        let exprs = Exprs::new(env).unwrap();
        let mut needed = Needed::none(env, 101, &exprs).unwrap();
        needed.need(99);
        join.prune(&mut needed);
        let mut reads = Needed::none(env, 101, &exprs).unwrap();
        join.reads(&mut reads);
        assert!(reads.is_needed(99) && !reads.is_needed(0) && !reads.is_needed(100));
    }

    #[test]
    fn builds_an_id_then_the_smaller_side_then_a_unique_key() {
        let (few, many) = (10, 1_000_000);
        for kind in [JoinKind::Inner, JoinKind::Left] {
            // An id, even if bigger, then fewer rows, then a unique key.
            assert_eq!(built(kind, (many, Key::Id), (few, Key::Unique)), 0);
            assert_eq!(built(kind, (many, Key::Unique), (few, Key::None)), 1);
            assert_eq!(built(kind, (few, Key::None), (many, Key::Unique)), 0);
            assert_eq!(built(kind, (few, Key::None), (few, Key::Unique)), 1);
            assert_eq!(built(kind, (many, Key::None), (few, Key::None)), 1);
            // As written when nothing tells them apart.
            assert_eq!(built(kind, (few, Key::Id), (few, Key::Id)), 1);
        }
    }
}
