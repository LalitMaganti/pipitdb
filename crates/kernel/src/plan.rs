//! `LogicalPlan`: what a query does, as Perfetto's: a tree of operations over
//! columns named by id, independent of how batches lay them out. Frontends
//! build one; `plan_nodes` plans it, asking each operation what it needs of
//! its inputs and what it outputs, then `lower` turns it into a pipeline,
//! asking each to lower itself, so extensions can add operations.

#![allow(unsafe_code)]

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::allocator::AllocError;
use crate::column::{DataType, Forms};
use crate::condition::{CONDITIONS_MAX, Condition, Cutoff};
use crate::containers::{Box, ErasedBox, InlineVec, SmallVec};
use crate::env::Env;
use crate::erase::{value_mut_of, value_of};
use crate::estimate::{Estimate, RowEstimate};
use crate::expression::{
    Compute, EXPR_NODES_MAX, Expr, ExprOp, Exprs, columns_of, conjuncts, mark_expr_columns,
    type_expr,
};
use crate::function::DynFunction;
use crate::join::JoinOp;
use crate::lower::{LowerError, Lowering};
use crate::names::Names;
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{ARRANGEMENT_KEYS_MAX, DeriveContext, InputNeeds, OutputProperties};
use crate::predicate::{Filter, Predicate, compared_with};
use crate::query::{Column, Named};
use crate::row_batch::BATCH_COLUMNS_MAX;
use crate::scannable::{DynScannable, Key, Summary};
use crate::statement::Statement;
use crate::step::{DynTransform, Step};
use crate::string_pool::Id;

/// The most columns a plan can have.
pub const PLAN_COLUMNS_MAX: usize = 1 << 10;
/// The most nodes a plan can have.
pub const PLAN_NODES_MAX: usize = 1 << 16;
/// The most bytes a plan's names take.
pub const PLAN_NAME_BYTES_MAX: usize = 1 << 14;

/// Stable within a plan. Lowering gives each a position in batches.
pub type ColumnId = u32;

pub type PlanNodeId = u32;

/// A key rows are sorted by.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SortKey {
    pub column: ColumnId,
    pub descending: bool,
}

pub struct ColumnSchema {
    /// What type its values are.
    pub data_type: DataType,
    /// Whether it can hold a null.
    pub nullable: bool,
}

/// An operation in a plan that reads what it borrows for `'c`, and knows how
/// to lower itself.
pub trait Op<'c> {
    /// Drops the columns this makes that aren't in `needed`, and marks the
    /// ones it reads. By default, it drops none, and marks what `reads` does.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        self.reads(needed);
        Pruned::Keep
    }

    /// Marks the columns whose values this reads. By default, all of them.
    fn reads(&self, reads: &mut Needed) {
        reads.need_all();
    }

    /// Of the columns this reads, the forms it takes `column` in. By default,
    /// flat only. Accept a form only where it gains, not just where it can be
    /// read: a dictionary pays where testing an entry costs much more than
    /// looking a row's entry up, as comparing strings, hashing into a group
    /// or adding to a set do, and not where it costs about as much, as
    /// comparing an integer does (ClickBench's integer filters were 11-63%
    /// slower through dictionaries).
    fn accepts(&self, exprs: &Exprs, column: ColumnId) -> Forms {
        let _ = (exprs, column);
        Forms::FLAT
    }

    /// Whether this op hands `column` on to the ops above it, as a filter
    /// does. False for an aggregate's keys, which come out one row per
    /// group. When false, ops above don't affect which forms, such as
    /// dictionary, the column can be loaded in. True by default.
    fn passes_through(&self, column: ColumnId) -> bool {
        let _ = column;
        true
    }

    /// Tells this, if it makes `column`, the forms the plan allows it in,
    /// and returns those of them it will make it in. By default, flat only.
    fn allow(&mut self, column: ColumnId, allowed: Forms) -> Forms {
        let _ = (column, allowed);
        Forms::FLAT
    }

    /// If this passes on its child's rows as they are but for leaving some
    /// out, as a filter does: a copy of the part of its condition that reads
    /// only `column`, for what makes the rows to skip by. By default, it
    /// passes on other rows, so nothing below can use its condition.
    fn give_condition(
        &mut self,
        exprs: &Exprs,
        env: &Env,
        column: ColumnId,
    ) -> Result<Given, AllocError> {
        let _ = (exprs, env, column);
        Ok(Given::Stop)
    }

    /// Which input, if any, a condition from above on `column` can be
    /// applied to instead, with the same result. For example, a condition on
    /// a column one side of a join makes can be applied to that side. By
    /// default, none.
    fn moves_condition(&self, column: ColumnId) -> Option<u32> {
        let _ = column;
        None
    }

    /// For an op that drops rows by a condition, as a filter does: splits off
    /// the parts of the condition that `below`, its child, can apply to one
    /// of its inputs instead (see `moves_condition`), returning them as new
    /// filters, one per input, and keeping the rest. By default, nothing is
    /// split off.
    fn split_below(
        &mut self,
        exprs: &Exprs,
        env: &Env,
        below: &DynOp<'c>,
    ) -> Result<Moved<'c>, AllocError> {
        let _ = (exprs, env, below);
        Ok(Moved::none())
    }

    /// Whether this makes `column` and takes conditions on it. By default,
    /// no.
    fn takes_condition(&self, column: ColumnId) -> bool {
        let _ = column;
        false
    }

    /// Takes `condition`, from an op above, on the rows it makes of
    /// `column`: pushdown. Only called if `takes_condition` says so.
    fn take_condition(
        &mut self,
        exprs: &Exprs,
        env: &Env,
        column: ColumnId,
        condition: Conjuncts,
    ) -> Result<(), AllocError> {
        let _ = (exprs, env, column, condition);
        failed!()
    }

    /// The op that loads `columns`, which this makes lazy, each in one of
    /// the forms given with it, to put above it. Only called for columns
    /// `allow` said it would make lazy.
    fn materialize(
        &self,
        env: &Env,
        columns: SmallVec<(ColumnId, Forms)>,
    ) -> Result<DynOp<'c>, AllocError> {
        let _ = (env, columns);
        failed!()
    }

    /// What's known of the rows this makes without reading them, and of
    /// `column`'s values if given, as `Scannable::summary`. By default,
    /// nothing.
    fn summary(&self, column: Option<ColumnId>) -> Option<Summary> {
        let _ = column;
        None
    }

    /// Whether this now makes its rows from what `child`, its one child,
    /// says of its own in `summary`, without reading them. By default, it
    /// can't.
    fn answer(&mut self, env: &Env, child: &DynOp<'c>) -> Result<bool, AllocError> {
        let _ = (env, child);
        Ok(false)
    }

    /// The estimate of the rows this will make, worked out from its inputs'
    /// estimates, or `None` for the default. The default is at most as many
    /// rows as the first input, with that input's keys that this passes
    /// through, none of them a sequence any more, since this may drop rows.
    fn estimate(&self, exprs: &Exprs, inputs: &[Estimate]) -> Option<Estimate> {
        let _ = (exprs, inputs);
        None
    }

    /// This, if it's a join: what the join planner reorders. By default,
    /// it isn't.
    fn as_join(&self) -> Option<&JoinOp> {
        None
    }

    /// Makes the choices that depend on its inputs' estimates, such as which
    /// side of a join to build. By default, there are none.
    fn choose(&mut self, exprs: &Exprs, inputs: &[Estimate]) {
        let _ = (exprs, inputs);
    }

    /// What this needs of the rows of input `input`. By default, nothing.
    fn needs(&self, input: u32) -> InputNeeds<'_> {
        let _ = input;
        InputNeeds::NONE
    }

    /// What is known of this node's output, given its inputs' needs are met.
    fn derive(&self, derive: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError>;

    /// Adds what runs `node`, which holds this, to `lowering`: first its
    /// inputs, through `Lowering::lower_input`, then the source or steps that
    /// make what `derive` said. It decides nothing about row order.
    fn lower(&self, node: &PlanNode<'c>, lowering: &mut Lowering<'_, 'c>)
    -> Result<(), LowerError>;
}

/// The filters split off an op's condition, one for each input of the op
/// below, and whether nothing of the condition is left.
pub struct Moved<'c> {
    pub filters: [Option<DynOp<'c>>; 2],
    pub emptied: bool,
}

impl<'c> Moved<'c> {
    /// Nothing split off.
    pub fn none() -> Moved<'c> {
        Moved { filters: [None, None], emptied: false }
    }
}

/// What an op gives up of its condition on a column.
pub enum Given {
    /// It passes on other rows than its child's: nothing below can apply a
    /// condition of it, or of anything above it.
    Stop,
    /// It passes on its child's rows: a copy of the part of its condition
    /// that reads only the column, if any, and any above may give theirs
    /// too.
    Passed(Option<Conjuncts>),
}

/// The parts of a condition, all of which must be true: the top-level
/// conjuncts of an expression, or of several.
pub type Conjuncts = SmallVec<Expr>;

type GiveCondition = unsafe fn(NonNull<()>, &Exprs, &Env, ColumnId) -> Result<Given, AllocError>;

type TakeCondition =
    unsafe fn(NonNull<()>, &Exprs, &Env, ColumnId, Conjuncts) -> Result<(), AllocError>;

type Answer<'c> = unsafe fn(NonNull<()>, &Env, &DynOp<'c>) -> Result<bool, AllocError>;

type SplitBelow<'c> =
    unsafe fn(NonNull<()>, &Exprs, &Env, &DynOp<'c>) -> Result<Moved<'c>, AllocError>;

type Materialize<'c> =
    unsafe fn(NonNull<()>, &Env, SmallVec<(ColumnId, Forms)>) -> Result<DynOp<'c>, AllocError>;

/// An `Op` of any type that lives for `'c`, owned in memory from an
/// `Env`, and the functions that know its type.
pub struct DynOp<'c> {
    op: ErasedBox,
    functions: &'c OpFunctions<'c>,
}

/// An op type's functions: one table for each type, which its ops share.
struct OpFunctions<'c> {
    prune: unsafe fn(NonNull<()>, &mut Needed) -> Pruned,
    reads: unsafe fn(NonNull<()>, &mut Needed),
    accepts: unsafe fn(NonNull<()>, &Exprs, ColumnId) -> Forms,
    passes_through: unsafe fn(NonNull<()>, ColumnId) -> bool,
    allow: unsafe fn(NonNull<()>, ColumnId, Forms) -> Forms,
    give_condition: GiveCondition,
    takes_condition: unsafe fn(NonNull<()>, ColumnId) -> bool,
    take_condition: TakeCondition,
    materialize: Materialize<'c>,
    summary: unsafe fn(NonNull<()>, Option<ColumnId>) -> Option<Summary>,
    answer: Answer<'c>,
    moves_condition: unsafe fn(NonNull<()>, ColumnId) -> Option<u32>,
    split_below: SplitBelow<'c>,
    estimate: unsafe fn(NonNull<()>, &Exprs, &[Estimate]) -> Option<Estimate>,
    choose: unsafe fn(NonNull<()>, &Exprs, &[Estimate]),
    as_join: unsafe fn(NonNull<()>) -> Option<NonNull<JoinOp>>,
    // Takes the `DynOp`, not its op's pointer, as the needs borrow from it.
    needs: for<'n> unsafe fn(&'n DynOp<'c>, u32) -> InputNeeds<'n>,
    derive: for<'p> unsafe fn(
        NonNull<()>,
        &mut DeriveContext<'p, 'c>,
    ) -> Result<OutputProperties, LowerError>,
    lower: for<'l> unsafe fn(
        NonNull<()>,
        &PlanNode<'c>,
        &mut Lowering<'l, 'c>,
    ) -> Result<(), LowerError>,
}

/// Holds `T`'s table.
struct Functions<'c, T>(PhantomData<(&'c (), T)>);

impl<'c, T: Op<'c> + 'c> Functions<'c, T> {
    const TABLE: OpFunctions<'c> = OpFunctions {
        // SAFETY: only called with this op, which the caller has mutably.
        prune: |op, needed| unsafe { value_mut_of::<T>(op).prune(needed) },
        // SAFETY: only called with this op.
        reads: |op, reads| unsafe { value_of::<T>(op).reads(reads) },
        // SAFETY: as for `reads`.
        accepts: |op, exprs, column| unsafe { value_of::<T>(op).accepts(exprs, column) },
        // SAFETY: as for `reads`.
        passes_through: |op, column| unsafe { value_of::<T>(op).passes_through(column) },
        // SAFETY: as for `prune`.
        allow: |op, column, allowed| unsafe { value_mut_of::<T>(op).allow(column, allowed) },
        // SAFETY: as for `prune`.
        give_condition: |op, exprs, env, column| unsafe {
            value_mut_of::<T>(op).give_condition(exprs, env, column)
        },
        // SAFETY: as for `reads`.
        takes_condition: |op, column| unsafe { value_of::<T>(op).takes_condition(column) },
        // SAFETY: as for `prune`.
        take_condition: |op, exprs, env, column, condition| unsafe {
            value_mut_of::<T>(op).take_condition(exprs, env, column, condition)
        },
        // SAFETY: as for `reads`.
        materialize: |op, env, columns| unsafe { value_of::<T>(op).materialize(env, columns) },
        // SAFETY: as for `reads`.
        summary: |op, column| unsafe { value_of::<T>(op).summary(column) },
        // SAFETY: as for `prune`.
        answer: |op, env, child| unsafe { value_mut_of::<T>(op).answer(env, child) },
        // SAFETY: as for `reads`.
        moves_condition: |op, column| unsafe { value_of::<T>(op).moves_condition(column) },
        // SAFETY: as for `prune`.
        split_below: |op, exprs, env, below| unsafe {
            value_mut_of::<T>(op).split_below(exprs, env, below)
        },
        // SAFETY: as for `reads`.
        estimate: |op, exprs, inputs| unsafe { value_of::<T>(op).estimate(exprs, inputs) },
        // SAFETY: as for `prune`.
        choose: |op, exprs, inputs| unsafe { value_mut_of::<T>(op).choose(exprs, inputs) },
        // SAFETY: as for `reads`.
        as_join: |op| unsafe { value_of::<T>(op).as_join().map(NonNull::from) },
        // SAFETY: only called with this op, which outlives what it returns.
        needs: |op, input| unsafe { value_of::<T>(op.op.as_ptr()).needs(input) },
        // SAFETY: as for `reads`.
        derive: |op, derive| unsafe { value_of::<T>(op).derive(derive) },
        // SAFETY: as for `reads`.
        lower: |op, node, lowering| unsafe { value_of::<T>(op).lower(node, lowering) },
    };
}

impl<'c> DynOp<'c> {
    pub fn new<T: Op<'c> + 'c>(env: &Env, op: T) -> Result<DynOp<'c>, AllocError> {
        let op = Box::new(env, op)?.erase();
        Ok(DynOp { op, functions: &Functions::<'c, T>::TABLE })
    }

    pub(crate) fn prune(&mut self, needed: &mut Needed) -> Pruned {
        // SAFETY: the function matches the op's type, which `self` holds
        // mutably.
        unsafe { (self.functions.prune)(self.op.as_ptr(), needed) }
    }

    pub(crate) fn reads(&self, reads: &mut Needed) {
        // SAFETY: the function matches the op's type.
        unsafe { (self.functions.reads)(self.op.as_ptr(), reads) }
    }

    pub(crate) fn accepts(&self, exprs: &Exprs, column: ColumnId) -> Forms {
        // SAFETY: the function matches the op's type.
        unsafe { (self.functions.accepts)(self.op.as_ptr(), exprs, column) }
    }

    pub(crate) fn passes_through(&self, column: ColumnId) -> bool {
        // SAFETY: the function matches the op's type.
        unsafe { (self.functions.passes_through)(self.op.as_ptr(), column) }
    }

    pub(crate) fn allow(&mut self, column: ColumnId, allowed: Forms) -> Forms {
        // SAFETY: the function matches the op's type, which `self` holds
        // mutably.
        unsafe { (self.functions.allow)(self.op.as_ptr(), column, allowed) }
    }

    pub(crate) fn give_condition(
        &mut self,
        exprs: &Exprs,
        env: &Env,
        column: ColumnId,
    ) -> Result<Given, AllocError> {
        // SAFETY: the function matches the op's type, which `self` holds
        // mutably.
        unsafe { (self.functions.give_condition)(self.op.as_ptr(), exprs, env, column) }
    }

    pub(crate) fn takes_condition(&self, column: ColumnId) -> bool {
        // SAFETY: the function matches the op's type.
        unsafe { (self.functions.takes_condition)(self.op.as_ptr(), column) }
    }

    pub(crate) fn take_condition(
        &mut self,
        exprs: &Exprs,
        env: &Env,
        column: ColumnId,
        condition: Conjuncts,
    ) -> Result<(), AllocError> {
        let op = self.op.as_ptr();
        // SAFETY: the function matches the op's type, which `self` holds
        // mutably.
        unsafe { (self.functions.take_condition)(op, exprs, env, column, condition) }
    }

    pub(crate) fn materialize(
        &self,
        env: &Env,
        columns: SmallVec<(ColumnId, Forms)>,
    ) -> Result<DynOp<'c>, AllocError> {
        // SAFETY: the function matches the op's type.
        unsafe { (self.functions.materialize)(self.op.as_ptr(), env, columns) }
    }

    pub(crate) fn moves_condition(&self, column: ColumnId) -> Option<u32> {
        // SAFETY: the function matches the op's type.
        unsafe { (self.functions.moves_condition)(self.op.as_ptr(), column) }
    }

    pub(crate) fn split_below(
        &mut self,
        exprs: &Exprs,
        env: &Env,
        below: &DynOp<'c>,
    ) -> Result<Moved<'c>, AllocError> {
        // SAFETY: the function matches the op's type, which `self` holds
        // mutably.
        unsafe { (self.functions.split_below)(self.op.as_ptr(), exprs, env, below) }
    }

    pub fn summary(&self, column: Option<ColumnId>) -> Option<Summary> {
        // SAFETY: the function matches the op's type.
        unsafe { (self.functions.summary)(self.op.as_ptr(), column) }
    }

    /// The op's estimate, or the default estimate if it gives none.
    pub(crate) fn estimate(&self, exprs: &Exprs, inputs: &[Estimate]) -> Estimate {
        // SAFETY: the function matches the op's type.
        let estimate = unsafe { (self.functions.estimate)(self.op.as_ptr(), exprs, inputs) };
        if let Some(estimate) = estimate {
            return estimate;
        }
        let Some(input) = inputs.first() else { return Estimate::UNKNOWN };
        let mut out = Estimate::of(input.rows);
        for &(column, key) in input.known_keys() {
            if self.passes_through(column) {
                out.add_key(column, key.min(Key::Unique));
            }
        }
        out
    }

    pub(crate) fn choose(&mut self, exprs: &Exprs, inputs: &[Estimate]) {
        // SAFETY: the function matches the op's type, which `self` holds
        // mutably.
        unsafe { (self.functions.choose)(self.op.as_ptr(), exprs, inputs) }
    }

    pub(crate) fn as_join(&self) -> Option<&JoinOp> {
        // SAFETY: the function matches the op's type, and the join it
        // returns is the op, which lives as long as `self`.
        unsafe { (self.functions.as_join)(self.op.as_ptr()).map(|join| join.as_ref()) }
    }

    pub(crate) fn answer(&mut self, env: &Env, child: &DynOp<'c>) -> Result<bool, AllocError> {
        // SAFETY: the function matches the op's type.
        unsafe { (self.functions.answer)(self.op.as_ptr(), env, child) }
    }

    pub(crate) fn needs(&self, input: u32) -> InputNeeds<'_> {
        // SAFETY: the function matches the op's type.
        unsafe { (self.functions.needs)(self, input) }
    }

    pub(crate) fn derive(
        &self,
        derive: &mut DeriveContext<'_, 'c>,
    ) -> Result<OutputProperties, LowerError> {
        // SAFETY: the function matches the op's type.
        unsafe { (self.functions.derive)(self.op.as_ptr(), derive) }
    }

    pub(crate) fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        // SAFETY: the function matches the op's type.
        unsafe { (self.functions.lower)(self.op.as_ptr(), node, lowering) }
    }
}

/// An operation, and the nodes whose rows it reads.
pub struct PlanNode<'c> {
    pub op: DynOp<'c>,
    pub children: Children,
}

/// The most nodes a node reads, as a join does.
pub const CHILDREN_MAX: usize = 2;

/// The nodes a node reads, in order.
pub type Children = InlineVec<PlanNodeId, CHILDREN_MAX>;

/// Borrows what it reads, such as a catalog's tables, for `'c`.
pub struct LogicalPlan<'c> {
    /// The columns' names.
    pub names: Names,
    /// Indexed by `ColumnId`.
    pub columns: SmallVec<ColumnSchema>,
    /// Indexed by `PlanNodeId`.
    pub nodes: SmallVec<PlanNode<'c>>,
    /// The node whose rows are the plan's rows.
    pub root: PlanNodeId,
    /// The columns of the result, in order.
    pub output: SmallVec<Named>,
    /// The nodes of its expressions.
    pub exprs: Exprs,
    /// What its runs read through their context, such as its literals.
    statement: Statement,
}

impl<'c> LogicalPlan<'c> {
    pub fn new(env: &Env) -> Result<LogicalPlan<'c>, AllocError> {
        Ok(LogicalPlan {
            names: Names::new(env, PLAN_NAME_BYTES_MAX)?,
            columns: SmallVec::new(env, PLAN_COLUMNS_MAX)?,
            nodes: SmallVec::new(env, PLAN_NODES_MAX)?,
            root: 0,
            output: SmallVec::new(env, PLAN_COLUMNS_MAX)?,
            exprs: Exprs::new(env)?,
            statement: Statement::default(),
        })
    }

    /// Adds `literal` to its literals, allocating from `env`, which
    /// must outlive the plan, and returns its id there.
    pub fn add_literal(&mut self, env: &Env, literal: &[u8]) -> Result<Id, AllocError> {
        self.statement.add_literal(env, literal)
    }

    /// The literal whose id is `id`.
    pub fn literal(&self, id: Id) -> &[u8] {
        let literal = self.statement.literals().and_then(|literals| literals.get(id));
        let Some(literal) = literal else { failed!() };
        literal
    }

    /// Binds `function` to it, allocating from `env`, which must
    /// outlive the plan, and returns its id, or `None` if it has the most it
    /// can.
    pub fn add_function(
        &mut self,
        env: &Env,
        function: DynFunction,
    ) -> Result<Option<u32>, AllocError> {
        self.statement.add_function(env, function)
    }

    /// What its runs read, which lowering gives the physical plan.
    pub(crate) fn take_statement(&mut self) -> Statement {
        core::mem::take(&mut self.statement)
    }

    /// A new column, and a copy of `name`, which it can be output as.
    pub fn add_column(
        &mut self,
        name: &str,
        data_type: DataType,
        nullable: bool,
    ) -> Result<Named, AllocError> {
        let name = self.names.add(name)?;
        Ok(Named { name, column: Column(self.add_unnamed_column(data_type, nullable)?) })
    }

    /// A new column, which has a name only if it's output.
    #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_COLUMNS_MAX`")]
    pub fn add_unnamed_column(
        &mut self,
        data_type: DataType,
        nullable: bool,
    ) -> Result<ColumnId, AllocError> {
        let id = self.columns.len() as ColumnId;
        self.columns.push(ColumnSchema { data_type, nullable })?;
        Ok(id)
    }

    /// Adds a node and makes it the root, which holds while a plan is built
    /// bottom up: each node added is the topmost so far.
    #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
    pub fn add_node(
        &mut self,
        op: DynOp<'c>,
        children: &[PlanNodeId],
    ) -> Result<PlanNodeId, AllocError> {
        let id = self.nodes.len() as PlanNodeId;
        self.nodes.push(PlanNode { op, children: Children::from_slice(children)? })?;
        self.root = id;
        Ok(id)
    }

    /// The id of `node`, one of this plan's nodes.
    pub fn id_of(&self, node: &PlanNode<'c>) -> PlanNodeId {
        let offset = core::ptr::from_ref(node).addr() - self.nodes.as_ptr().addr();
        let id = offset / size_of::<PlanNode<'c>>();
        check!(id < self.nodes.len());
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
        let id = id as PlanNodeId;
        id
    }
}

/// Reads the rows of a scannable, binding the columns in `columns`, maybe
/// skipping some failing `conditions`.
pub struct ScanOp<'c> {
    /// What it reads.
    scannable: &'c DynScannable<'c>,
    /// The scannable's columns it reads, and what each is bound to.
    columns: SmallVec<ScanColumn>,
    /// Conditions the plan pushed down from filters above, if any, which the
    /// scannable may use to skip: the filters still test every row.
    conditions: Option<SmallVec<Condition>>,
}

impl<'c> ScanOp<'c> {
    pub fn new(scannable: &'c DynScannable<'c>, columns: SmallVec<ScanColumn>) -> ScanOp<'c> {
        ScanOp { scannable, columns, conditions: None }
    }
}

/// A column a scan reads, and what it's bound to in the plan.
#[derive(Clone, Copy)]
pub struct ScanColumn {
    /// Which of the scannable's columns.
    pub column: u32,
    /// What it's bound to.
    pub binding: ColumnId,
    /// The forms it's read in, which the plan chooses.
    pub forms: Forms,
}

impl<'c> Op<'c> for ScanOp<'c> {
    /// Keeps the needed columns, and at least one: a batch with no columns
    /// has no rows.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        let any = self.columns.iter().any(|column| needed.is_needed(column.binding));
        let mut i = 0;
        self.columns.retain(|column| {
            let keep = needed.is_needed(column.binding) || (!any && i == 0);
            i += 1;
            keep
        });
        Pruned::Keep
    }

    /// Reads none: it makes them.
    fn reads(&self, _: &mut Needed) {}

    fn allow(&mut self, column: ColumnId, allowed: Forms) -> Forms {
        let found = self.columns.iter_mut().find(|c| c.binding == column);
        // Chosen once: a column chosen already is loaded already.
        let Some(scanned) = found.filter(|scanned| scanned.forms == Forms::FLAT) else {
            return Forms::FLAT;
        };
        scanned.forms = allowed & self.scannable.forms(scanned.column);
        scanned.forms
    }

    /// Those on its columns.
    fn takes_condition(&self, column: ColumnId) -> bool {
        self.columns.iter().any(|scanned| scanned.binding == column)
    }

    fn take_condition(
        &mut self,
        exprs: &Exprs,
        env: &Env,
        column: ColumnId,
        condition: Conjuncts,
    ) -> Result<(), AllocError> {
        let Some(scanned) = self.columns.iter().find(|c| c.binding == column) else {
            failed!();
        };
        let data_type = self.scannable.column_type(scanned.column);
        let column = |_| (scanned.column, data_type);
        for conjunct in condition.iter() {
            // Only what a predicate tests as leaves says which values it
            // keeps, and an `IN` list doesn't say.
            if exprs.nodes(*conjunct).iter().any(|node| node.op == ExprOp::In) {
                continue;
            }
            let mut computed = SmallVec::new(env, EXPR_NODES_MAX)?;
            let predicate = Predicate::from_expr(
                env,
                exprs,
                core::slice::from_ref(conjunct),
                None,
                &column,
                0,
                &mut computed,
            )?;
            if !computed.is_empty() {
                continue;
            }
            let condition = Condition::new(scanned.column, &predicate);
            let conditions = match &mut self.conditions {
                Some(conditions) => conditions,
                None => self.conditions.insert(SmallVec::new(env, CONDITIONS_MAX)?),
            };
            conditions.push(condition).map_err(|_| AllocError)?;
        }
        Ok(())
    }

    fn materialize(
        &self,
        env: &Env,
        columns: SmallVec<(ColumnId, Forms)>,
    ) -> Result<DynOp<'c>, AllocError> {
        DynOp::new(env, MaterializeOp { scannable: self.scannable, columns })
    }

    /// Its scannable's.
    fn summary(&self, column: Option<ColumnId>) -> Option<Summary> {
        let scanned = match column {
            Some(column) => Some(self.columns.iter().find(|c| c.binding == column)?.column),
            None => None,
        };
        self.scannable.summary(scanned)
    }

    /// The table's row count, which of the columns read are keys, and the
    /// bounds of those it knows the bounds of. If the scan took pushed-down
    /// conditions, it may skip rows, so no key is known to be a sequence. It doesn't reduce its row count for those
    /// conditions: the filters they were copied from already do.
    fn estimate(&self, _: &Exprs, _: &[Estimate]) -> Option<Estimate> {
        let summary = self.scannable.summary(None);
        let rows = summary.and_then(|summary| u64::try_from(summary.rows).ok());
        let mut out = Estimate::of(rows.map_or(RowEstimate::UNKNOWN, RowEstimate::exactly));
        for scanned in self.columns.iter() {
            out.add_key(scanned.binding, self.scannable.column_key(scanned.column));
            let summary = self.scannable.summary(Some(scanned.column));
            if let Some(bounds) = summary.and_then(|summary| summary.bounds) {
                out.add_bounds(scanned.binding, bounds);
            }
        }
        if self.conditions.as_ref().is_some_and(|conditions| !conditions.is_empty()) {
            out.demote_ids();
        }
        Some(out)
    }

    /// Its columns, in order, and each order the scannable says its rows
    /// are in, as far as the columns it sorts by are read: a sort by one
    /// that isn't says nothing of the order by those after it.
    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        let mut out = OutputProperties::NONE;
        for column in self.columns.iter() {
            out.push_column(c.column_of(column.binding))?;
        }
        out.scan = Some(c.node());
        let read = |&(column, descending): &(u32, bool)| {
            let read = self.columns.iter().find(|read| read.column == column);
            read.map(|read| SortKey { column: read.binding, descending })
        };
        for order in self.scannable.sorted_by() {
            let mut sorted = [SortKey { column: 0, descending: false }; ARRANGEMENT_KEYS_MAX];
            let mut len = 0;
            for (slot, key) in sorted.iter_mut().zip(order.iter().map_while(read)) {
                (*slot, len) = (key, len + 1);
            }
            if len > 0 {
                out.arrangements.push(&[], at!(sorted, ..len));
            }
        }
        Ok(out)
    }

    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        let read = self.columns.iter().map(|column| (column.column, column.forms));
        let conditions = self.conditions.as_deref().unwrap_or(&[]);
        let cutoff = lowering.query_cutoff(node).map(|(key, slot)| {
            let Some(scanned) = self.columns.iter().find(|c| c.binding == key.column) else {
                failed!()
            };
            Cutoff { slot, column: scanned.column, descending: key.descending }
        });
        let source = self.scannable.scan(lowering.env(), read, conditions, cutoff)?;
        lowering.set_source(source);
        Ok(())
    }
}

/// Loads `columns` of its one child's batches, which a scan of `scannable`
/// made lazy, for the rows each keeps, each in one of the forms with it.
pub struct MaterializeOp<'c> {
    scannable: &'c DynScannable<'c>,
    columns: SmallVec<(ColumnId, Forms)>,
}

impl<'c> Op<'c> for MaterializeOp<'c> {
    /// Keeps loading the needed columns, if any.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        self.columns.retain(|&(column, _)| needed.is_needed(column));
        if self.columns.is_empty() { Pruned::Child(0) } else { Pruned::Keep }
    }

    /// Reads none: it loads them.
    fn reads(&self, _: &mut Needed) {}

    /// Its input's: it loads columns where they are.
    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        Ok(*c.input(0))
    }

    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        lowering.lower_input(node, 0)?;
        let positions = self.columns.iter().map(|&(column, _)| lowering.query_position(column));
        let positions = SmallVec::fixed_from(lowering.env(), positions)?;
        let forms = self.columns.iter().map(|&(_, forms)| forms);
        let forms = SmallVec::fixed_from(lowering.env(), forms)?;
        let transform = self.scannable.materialize(lowering.env(), positions, forms)?;
        lowering.add_operator(Step::Transform(transform))
    }
}

/// Keeps the rows of its one child that each of `conjuncts` is true for, as
/// `WHERE` does: neither false nor null. Their columns are `ColumnId`s.
pub struct FilterOp {
    pub conjuncts: Conjuncts,
}

impl FilterOp {
    /// The filter of `condition`, of `exprs`: its top-level conjuncts.
    pub fn new(env: &Env, exprs: &Exprs, condition: Expr) -> Result<FilterOp, AllocError> {
        Ok(FilterOp { conjuncts: conjuncts_of(env, exprs, condition)? })
    }
}

/// The top-level conjuncts of `condition`, of `exprs`: what its `AND`s join.
pub fn conjuncts_of(env: &Env, exprs: &Exprs, condition: Expr) -> Result<Conjuncts, AllocError> {
    let mut count = 0;
    conjuncts(exprs.nodes(condition), condition.root, &mut |_| count += 1);
    let mut parts = SmallVec::fixed(env, count)?;
    conjuncts(exprs.nodes(condition), condition.root, &mut |root| {
        parts.extend_within(core::iter::once(Expr { root, ..condition }));
    });
    Ok(parts)
}

/// The conjuncts of `conjuncts` that `keep` is true for, or `None` if none.
/// Out of line: each caller's `keep` differs, and inlining one copy each
/// costs more than calling it.
#[inline(never)]
fn kept(
    env: &Env,
    conjuncts: &[Expr],
    keep: &dyn Fn(Expr) -> bool,
) -> Result<Option<Conjuncts>, AllocError> {
    let count = conjuncts.iter().filter(|&&conjunct| keep(conjunct)).count();
    if count == 0 {
        return Ok(None);
    }
    let kept = conjuncts.iter().copied().filter(|&conjunct| keep(conjunct));
    let mut parts = SmallVec::fixed(env, count)?;
    parts.extend_within(kept);
    Ok(Some(parts))
}

impl<'c> Op<'c> for FilterOp {
    /// Its conditions' columns.
    fn reads(&self, reads: &mut Needed) {
        for &conjunct in self.conjuncts.iter() {
            mark_expr_columns(conjunct, reads);
        }
    }

    /// Constants, which it tests once; dictionaries of columns it only
    /// compares with strings, whose entries it tests once each; and
    /// sequences of columns it only compares with integers, whose rows it
    /// keeps by their numbers: see `Op::accepts`.
    fn accepts(&self, exprs: &Exprs, column: ColumnId) -> Forms {
        match compared_with(exprs, &self.conjuncts, column) {
            Some(DataType::String) => Forms::FLAT | Forms::CONSTANT | Forms::DICTIONARY,
            Some(DataType::Int64) => Forms::FLAT | Forms::CONSTANT | Forms::SEQUENCE,
            _ => Forms::FLAT | Forms::CONSTANT,
        }
    }

    /// Splits off the conjuncts that `below` can apply to an input instead:
    /// those whose columns are all in one of its inputs.
    fn split_below(
        &mut self,
        exprs: &Exprs,
        env: &Env,
        below: &DynOp<'c>,
    ) -> Result<Moved<'c>, AllocError> {
        // Where `conjunct` goes: 0 to stay, or 1 more than the input all its
        // columns are in.
        let part = |conjunct: Expr| {
            // `None` before any column, then the input all are in, if one.
            let mut place = None;
            columns_of(exprs.nodes(conjunct), conjunct.root, &mut |column| {
                let input = below.moves_condition(column).filter(|&input| input < 2);
                place = Some(match place {
                    Some(placed) if placed != input => None,
                    _ => input,
                });
            });
            match place {
                Some(Some(input)) => 1 + input,
                _ => 0,
            }
        };
        // Each conjunct's place, found once.
        let mut places = SmallVec::fixed(env, self.conjuncts.len())?;
        places.extend_within(self.conjuncts.iter().map(|&conjunct| part(conjunct)));
        let place = |conjunct: Expr| {
            let at = self.conjuncts.iter().position(|&c| c == conjunct).unwrap_or(0);
            *at!(places, at)
        };
        let moves = |to| kept(env, &self.conjuncts, &|conjunct| place(conjunct) == to);
        let (first, second) = (moves(1)?, moves(2)?);
        if first.is_none() && second.is_none() {
            return Ok(Moved::none());
        }
        let stay = moves(0)?;
        let filter = |part: Option<Conjuncts>| {
            part.map(|conjuncts| DynOp::new(env, FilterOp { conjuncts })).transpose()
        };
        let filters = [filter(first)?, filter(second)?];
        let emptied = stay.is_none();
        if let Some(stay) = stay {
            self.conjuncts = stay;
        }
        Ok(Moved { filters, emptied })
    }

    /// Its conjuncts that read only `column`.
    fn give_condition(
        &mut self,
        exprs: &Exprs,
        env: &Env,
        column: ColumnId,
    ) -> Result<Given, AllocError> {
        let reads_only = |conjunct: Expr| {
            let (mut read, mut other) = (false, false);
            columns_of(exprs.nodes(conjunct), conjunct.root, &mut |read_column| {
                read |= read_column == column;
                other |= read_column != column;
            });
            read && !other
        };
        Ok(Given::Passed(kept(env, &self.conjuncts, &reads_only)?))
    }

    /// Its input's estimate, less the rows its conjuncts are expected to
    /// drop: for a column of known bounds compared with an integer, the
    /// share of its bounds the comparison keeps; for the rest, by the rules
    /// Perfetto's dataframe planner uses for its filters.
    fn estimate(&self, exprs: &Exprs, inputs: &[Estimate]) -> Option<Estimate> {
        let mut out = *inputs.first()?;
        out.demote_ids();
        for &conjunct in self.conjuncts.iter() {
            let nodes = exprs.nodes(conjunct);
            let node = at!(nodes, conjunct.root as usize);
            let column = |i: u32| nodes.get(i as usize).filter(|n| n.op == ExprOp::Column);
            let constant = |i: u32| nodes.get(i as usize).is_some_and(|n| n.op == ExprOp::Constant);
            let integer = |i: u32| {
                let node = nodes.get(i as usize)?;
                (node.op == ExprOp::Constant && node.data_type == DataType::Int64)
                    .then_some(node.int_value)
            };
            // A column compared with an integer, the comparison read with the
            // column on its left.
            let compared = match (column(node.a), column(node.b), node.op.comparison()) {
                (Some(c), None, Some(op)) => integer(node.b).map(|v| (c.a, op, v)),
                (None, Some(c), Some(op)) => integer(node.a).map(|v| (c.a, op.flipped(), v)),
                _ => None,
            };
            if let Some((c, op, v)) = compared
                && out.span(c).is_some()
                && let Some((min, max)) = op.range(v)
            {
                out.keep_within(c, min, max);
                continue;
            }
            match node.op {
                ExprOp::Eq => match (column(node.a), column(node.b)) {
                    (Some(c), None) if constant(node.b) => out.keep_equal(c.a),
                    (None, Some(c)) if constant(node.a) => out.keep_equal(c.a),
                    _ => out.keep_some(),
                },
                ExprOp::In => match column(node.a) {
                    Some(c) => out.keep_in(c.a, u64::from(node.c)),
                    None => out.keep_some(),
                },
                _ => out.keep_some(),
            }
        }
        Some(out)
    }

    /// Its input's, less some rows.
    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        let mut out = *c.input(0);
        out.drop_rows();
        Ok(out)
    }

    /// A filter of a predicate, which tests what it can of the conditions
    /// itself, first computing the rest.
    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        lowering.lower_input(node, 0)?;
        let (env, exprs, values) = (lowering.env(), lowering.exprs(), lowering.expr_values());
        let rows = lowering.query_rows();
        let column = |column| {
            let Some(position) = rows.position_of(column) else { failed!() };
            (position, rows.type_of(column))
        };
        let mut computed = SmallVec::new(env, EXPR_NODES_MAX)?;
        let at = rows.column_count();
        let conjuncts = &self.conjuncts;
        let predicate = Predicate::from_expr(
            env,
            exprs,
            conjuncts,
            values.clone(),
            &column,
            at,
            &mut computed,
        )?;
        if at as usize + computed.len() > BATCH_COLUMNS_MAX as usize {
            return Err(LowerError::TooManyColumns);
        }
        let mut compute = None;
        if !computed.is_empty() {
            let mut programs = SmallVec::fixed(env, computed.len())?;
            for &part in computed.iter() {
                let nodes = exprs.nodes(part);
                let program = match type_expr(env, rows, nodes, values.clone(), true) {
                    Ok(program) => program,
                    Err(crate::error::Error::OutOfMemory) => return Err(LowerError::OutOfMemory),
                    // Named by the first column it reads.
                    Err(_) => {
                        let mut first = None;
                        columns_of(nodes, part.root, &mut |column| _ = first.get_or_insert(column));
                        return Err(LowerError::WrongType(first.unwrap_or(0)));
                    }
                };
                programs.push(program)?;
            }
            compute = Some(Compute { programs });
        }
        let filter = Filter { compute, predicate };
        lowering.add_operator(Step::Transform(DynTransform::new(env, filter)?))
    }
}
