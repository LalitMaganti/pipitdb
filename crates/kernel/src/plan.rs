//! `LogicalPlan`: what a query does, as Perfetto's: a tree of operations over
//! columns named by id, independent of how batches lay them out. Frontends
//! build one; `lower` turns it into a pipeline, asking each operation to
//! lower itself, so extensions can add operations.

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::boxed::{Box, ErasedBox};
use crate::column::{DataType, Forms};
use crate::condition::{CONDITIONS_MAX, Condition};
use crate::erase::{value_mut_of, value_of};
use crate::lower::{LowerError, Lowering};
use crate::names::{Name, Names};
use crate::optimize::{Needed, Pruned};
use crate::predicate::{Filter, Predicate};
use crate::scannable::DynScannable;
use crate::slow_vec::SlowVec;
use crate::step::{DynTransform, Step};

/// The most columns a plan can have.
pub const PLAN_COLUMNS_MAX: usize = 1 << 10;
/// The most nodes a plan can have.
pub const PLAN_NODES_MAX: usize = 1 << 6;
/// The most bytes a plan's names take.
pub const PLAN_NAME_BYTES_MAX: usize = 1 << 14;

/// Stable within a plan. Lowering gives each a position in batches.
pub type ColumnId = u32;

pub type PlanNodeId = u32;

pub struct ColumnSchema {
    pub name: Name,
    pub data_type: DataType,
}

/// A name in a scope or result. Several names can refer to one column.
#[derive(Clone, Copy)]
pub struct NamedColumn {
    pub name: Name,
    pub id: ColumnId,
}

/// An operation in a plan that reads what it borrows for `'c`, and knows how
/// to lower itself.
pub trait Op<'c> {
    /// Adds what runs `node`, which holds this, to `lowering`: first its
    /// children, through `Lowering::lower`, then its own source or steps,
    /// defining the columns it makes.
    fn lower(&self, node: &PlanNode<'c>, lowering: &mut Lowering<'_, 'c>)
    -> Result<(), LowerError>;

    /// Drops the columns this makes that aren't in `needed`, and marks the
    /// ones it reads. By default, it can't say, so everything is kept.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        needed.need_all();
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
    fn accepts(&self, column: ColumnId) -> Forms {
        let _ = column;
        Forms::FLAT
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
        allocator: &dyn Allocator,
        column: ColumnId,
    ) -> Result<Given, AllocError> {
        let _ = (allocator, column);
        Ok(Given::Stop)
    }

    /// Whether this makes `column` and takes conditions on it. By default,
    /// no.
    fn takes_condition(&self, column: ColumnId) -> bool {
        let _ = column;
        false
    }

    /// Takes `condition`, from an op above, on the rows it makes of
    /// `column`: pushdown. If `may_apply`, as nothing else reads the column,
    /// returns whether it applies it, so the op can remove it. Only called
    /// if `takes_condition` says so.
    fn take_condition(
        &mut self,
        allocator: &dyn Allocator,
        column: ColumnId,
        condition: Predicate,
        may_apply: bool,
    ) -> Result<bool, AllocError> {
        let _ = (allocator, column, condition, may_apply);
        crate::check::check_failed(line!())
    }

    /// Stops testing its conjuncts that read only `column`, as the scan
    /// below now applies them. Returns whether it tests nothing now, so the
    /// plan can remove it. Only called once it's given them.
    fn remove_condition(
        &mut self,
        allocator: &dyn Allocator,
        column: ColumnId,
    ) -> Result<bool, AllocError> {
        let _ = (allocator, column);
        crate::check::check_failed(line!())
    }

    /// The op that loads `columns`, which this makes lazy, each in one of
    /// the forms given with it, to put above it. Only called for columns
    /// `allow` said it would make lazy.
    fn materialize(
        &self,
        allocator: &dyn Allocator,
        columns: SlowVec<(ColumnId, Forms)>,
    ) -> Result<DynOp<'c>, AllocError> {
        let _ = (allocator, columns);
        crate::check::check_failed(line!())
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
    Passed(Option<Predicate>),
}

type GiveCondition = unsafe fn(NonNull<()>, &dyn Allocator, ColumnId) -> Result<Given, AllocError>;

type TakeCondition =
    unsafe fn(NonNull<()>, &dyn Allocator, ColumnId, Predicate, bool) -> Result<bool, AllocError>;

type RemoveCondition = unsafe fn(NonNull<()>, &dyn Allocator, ColumnId) -> Result<bool, AllocError>;

type Materialize<'c> = unsafe fn(
    NonNull<()>,
    &dyn Allocator,
    SlowVec<(ColumnId, Forms)>,
) -> Result<DynOp<'c>, AllocError>;

/// An `Op` of any type that lives for `'c`, owned in memory from an
/// allocator, and the function that knows its type.
pub struct DynOp<'c> {
    op: ErasedBox,
    lower: for<'l> unsafe fn(
        NonNull<()>,
        &PlanNode<'c>,
        &mut Lowering<'l, 'c>,
    ) -> Result<(), LowerError>,
    prune: unsafe fn(NonNull<()>, &mut Needed) -> Pruned,
    reads: unsafe fn(NonNull<()>, &mut Needed),
    accepts: unsafe fn(NonNull<()>, ColumnId) -> Forms,
    allow: unsafe fn(NonNull<()>, ColumnId, Forms) -> Forms,
    give_condition: GiveCondition,
    takes_condition: unsafe fn(NonNull<()>, ColumnId) -> bool,
    take_condition: TakeCondition,
    remove_condition: RemoveCondition,
    materialize: Materialize<'c>,
    lifetime: PhantomData<&'c ()>,
}

impl<'c> DynOp<'c> {
    pub fn new<T: Op<'c> + 'c>(allocator: &dyn Allocator, op: T) -> Result<DynOp<'c>, AllocError> {
        Ok(DynOp {
            op: Box::new(allocator, op)?.erase(),
            // SAFETY: only called with this op.
            lower: |op, node, lowering| unsafe { value_of::<T>(op).lower(node, lowering) },
            // SAFETY: as above, and the caller has the op mutably.
            prune: |op, needed| unsafe { value_mut_of::<T>(op).prune(needed) },
            // SAFETY: as for `lower`.
            reads: |op, reads| unsafe { value_of::<T>(op).reads(reads) },
            // SAFETY: as for `lower`.
            accepts: |op, column| unsafe { value_of::<T>(op).accepts(column) },
            // SAFETY: as for `prune`.
            allow: |op, column, allowed| unsafe { value_mut_of::<T>(op).allow(column, allowed) },
            // SAFETY: as for `prune`.
            give_condition: |op, allocator, column| unsafe {
                value_mut_of::<T>(op).give_condition(allocator, column)
            },
            // SAFETY: as for `lower`.
            takes_condition: |op, column| unsafe { value_of::<T>(op).takes_condition(column) },
            // SAFETY: as for `prune`.
            take_condition: |op, allocator, column, condition, may_apply| unsafe {
                value_mut_of::<T>(op).take_condition(allocator, column, condition, may_apply)
            },
            // SAFETY: as for `prune`.
            remove_condition: |op, allocator, column| unsafe {
                value_mut_of::<T>(op).remove_condition(allocator, column)
            },
            // SAFETY: as for `lower`.
            materialize: |op, allocator, columns| unsafe {
                value_of::<T>(op).materialize(allocator, columns)
            },
            lifetime: PhantomData,
        })
    }

    pub(crate) fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        // SAFETY: the function matches the op's type.
        unsafe { (self.lower)(self.op.as_ptr(), node, lowering) }
    }

    pub(crate) fn prune(&mut self, needed: &mut Needed) -> Pruned {
        // SAFETY: the function matches the op's type, which `self` holds
        // mutably.
        unsafe { (self.prune)(self.op.as_ptr(), needed) }
    }

    pub(crate) fn reads(&self, reads: &mut Needed) {
        // SAFETY: the function matches the op's type.
        unsafe { (self.reads)(self.op.as_ptr(), reads) }
    }

    pub(crate) fn accepts(&self, column: ColumnId) -> Forms {
        // SAFETY: the function matches the op's type.
        unsafe { (self.accepts)(self.op.as_ptr(), column) }
    }

    pub(crate) fn allow(&mut self, column: ColumnId, allowed: Forms) -> Forms {
        // SAFETY: the function matches the op's type, which `self` holds
        // mutably.
        unsafe { (self.allow)(self.op.as_ptr(), column, allowed) }
    }

    pub(crate) fn give_condition(
        &mut self,
        allocator: &dyn Allocator,
        column: ColumnId,
    ) -> Result<Given, AllocError> {
        // SAFETY: the function matches the op's type, which `self` holds
        // mutably.
        unsafe { (self.give_condition)(self.op.as_ptr(), allocator, column) }
    }

    pub(crate) fn takes_condition(&self, column: ColumnId) -> bool {
        // SAFETY: the function matches the op's type.
        unsafe { (self.takes_condition)(self.op.as_ptr(), column) }
    }

    pub(crate) fn take_condition(
        &mut self,
        allocator: &dyn Allocator,
        column: ColumnId,
        condition: Predicate,
        may_apply: bool,
    ) -> Result<bool, AllocError> {
        // SAFETY: the function matches the op's type, which `self` holds
        // mutably.
        unsafe { (self.take_condition)(self.op.as_ptr(), allocator, column, condition, may_apply) }
    }

    pub(crate) fn remove_condition(
        &mut self,
        allocator: &dyn Allocator,
        column: ColumnId,
    ) -> Result<bool, AllocError> {
        // SAFETY: the function matches the op's type, which `self` holds
        // mutably.
        unsafe { (self.remove_condition)(self.op.as_ptr(), allocator, column) }
    }

    pub(crate) fn materialize(
        &self,
        allocator: &dyn Allocator,
        columns: SlowVec<(ColumnId, Forms)>,
    ) -> Result<DynOp<'c>, AllocError> {
        // SAFETY: the function matches the op's type.
        unsafe { (self.materialize)(self.op.as_ptr(), allocator, columns) }
    }
}

/// An operation, and the nodes whose rows it reads.
pub struct PlanNode<'c> {
    pub op: DynOp<'c>,
    pub children: SlowVec<PlanNodeId>,
}

/// Borrows what it reads, such as a catalog's tables, for `'c`.
pub struct LogicalPlan<'c> {
    /// The columns' names.
    pub names: Names,
    /// Indexed by `ColumnId`.
    pub columns: SlowVec<ColumnSchema>,
    /// Indexed by `PlanNodeId`.
    pub nodes: SlowVec<PlanNode<'c>>,
    /// The node whose rows are the plan's rows.
    pub root: PlanNodeId,
    /// The columns of the result, in order.
    pub output: SlowVec<NamedColumn>,
}

impl<'c> LogicalPlan<'c> {
    pub fn new(allocator: &dyn Allocator) -> Result<LogicalPlan<'c>, AllocError> {
        Ok(LogicalPlan {
            names: Names::new(allocator, PLAN_NAME_BYTES_MAX)?,
            columns: SlowVec::new(allocator, PLAN_COLUMNS_MAX)?,
            nodes: SlowVec::new(allocator, PLAN_NODES_MAX)?,
            root: 0,
            output: SlowVec::new(allocator, PLAN_COLUMNS_MAX)?,
        })
    }

    /// A new column, with a copy of `name`.
    #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_COLUMNS_MAX`")]
    pub fn add_column(
        &mut self,
        name: &str,
        data_type: DataType,
    ) -> Result<NamedColumn, AllocError> {
        let id = self.columns.len() as ColumnId;
        let name = self.names.add(name)?;
        self.columns.push(ColumnSchema { name, data_type })?;
        Ok(NamedColumn { name, id })
    }

    /// Adds a node and makes it the root, which holds while a plan is built
    /// bottom up: each node added is the topmost so far.
    #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
    pub fn add_node(
        &mut self,
        op: DynOp<'c>,
        children: SlowVec<PlanNodeId>,
    ) -> Result<PlanNodeId, AllocError> {
        let id = self.nodes.len() as PlanNodeId;
        self.nodes.push(PlanNode { op, children })?;
        self.root = id;
        Ok(id)
    }
}

/// Reads the rows of a scannable, binding the columns in `columns`, maybe
/// skipping some failing `conditions`.
pub struct ScanOp<'c> {
    scannable: &'c DynScannable<'c>,
    columns: SlowVec<ScanColumn>,
    /// Conditions the plan pushed down from filters above, if any, which the
    /// scannable may use to skip: the filters still test every row.
    conditions: Option<SlowVec<Condition>>,
}

impl<'c> ScanOp<'c> {
    pub fn new(scannable: &'c DynScannable<'c>, columns: SlowVec<ScanColumn>) -> ScanOp<'c> {
        ScanOp { scannable, columns, conditions: None }
    }
}

/// A column a scan reads, and what it's bound to in the plan.
#[derive(Clone, Copy)]
pub struct ScanColumn {
    /// Which of the scannable's columns.
    pub column: u32,
    /// What it's bound to.
    pub binding: NamedColumn,
    /// The forms it's read in, which the plan chooses.
    pub forms: Forms,
}

impl<'c> Op<'c> for ScanOp<'c> {
    fn lower(&self, _: &PlanNode<'c>, lowering: &mut Lowering<'_, 'c>) -> Result<(), LowerError> {
        for column in self.columns.iter() {
            lowering.define(column.binding.id)?;
        }
        let read = self.columns.iter().map(|column| (column.column, column.forms));
        let conditions = self.conditions.as_deref().unwrap_or(&[]);
        lowering.set_source(self.scannable.scan(lowering.allocator(), read, conditions)?);
        Ok(())
    }

    /// Keeps the needed columns, and at least one: a batch with no columns
    /// has no rows.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        let any = self.columns.iter().any(|column| needed.is_needed(column.binding.id));
        let mut i = 0;
        self.columns.retain(|column| {
            let keep = needed.is_needed(column.binding.id) || (!any && i == 0);
            i += 1;
            keep
        });
        Pruned::Keep
    }

    /// Reads none: it makes them.
    fn reads(&self, _: &mut Needed) {}

    /// Those on its columns.
    fn takes_condition(&self, column: ColumnId) -> bool {
        self.columns.iter().any(|scanned| scanned.binding.id == column)
    }

    /// Applies it, if it may, where the scannable says it does.
    fn take_condition(
        &mut self,
        allocator: &dyn Allocator,
        column: ColumnId,
        condition: Predicate,
        may_apply: bool,
    ) -> Result<bool, AllocError> {
        let Some(scanned) = self.columns.iter().find(|c| c.binding.id == column) else {
            crate::check::check_failed(line!());
        };
        let mut condition = Condition::new(scanned.column, condition.renumbered(allocator, |_| 0)?);
        let applied = may_apply && self.scannable.applies(&condition);
        if applied {
            condition.set_applied();
        }
        let conditions = match &mut self.conditions {
            Some(conditions) => conditions,
            None => self.conditions.insert(SlowVec::new(allocator, CONDITIONS_MAX)?),
        };
        conditions.push(condition).map_err(|_| AllocError)?;
        Ok(applied)
    }

    fn allow(&mut self, column: ColumnId, allowed: Forms) -> Forms {
        let found = self.columns.iter_mut().find(|c| c.binding.id == column);
        // Chosen once: a column chosen already is loaded already.
        let Some(scanned) = found.filter(|scanned| scanned.forms == Forms::FLAT) else {
            return Forms::FLAT;
        };
        scanned.forms = allowed & self.scannable.forms(scanned.column);
        scanned.forms
    }

    fn materialize(
        &self,
        allocator: &dyn Allocator,
        columns: SlowVec<(ColumnId, Forms)>,
    ) -> Result<DynOp<'c>, AllocError> {
        DynOp::new(allocator, MaterializeOp { scannable: self.scannable, columns })
    }
}

/// Loads `columns` of its one child's batches, which a scan of `scannable`
/// made lazy, for the rows each keeps, each in one of the forms with it.
pub struct MaterializeOp<'c> {
    scannable: &'c DynScannable<'c>,
    columns: SlowVec<(ColumnId, Forms)>,
}

impl<'c> Op<'c> for MaterializeOp<'c> {
    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        lowering.lower(*at!(node.children, 0))?;
        let positions = self.columns.iter().map(|&(column, _)| lowering.position(column));
        let positions = SlowVec::fixed_from(lowering.allocator(), positions)?;
        let forms = self.columns.iter().map(|&(_, forms)| forms);
        let forms = SlowVec::fixed_from(lowering.allocator(), forms)?;
        let transform = self.scannable.materialize(lowering.allocator(), positions, forms)?;
        lowering.add_step(Step::Transform(transform))
    }

    /// Keeps loading the needed columns, if any.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        self.columns.retain(|&(column, _)| needed.is_needed(column));
        if self.columns.is_empty() { Pruned::Child(0) } else { Pruned::Keep }
    }

    /// Reads none: it loads them.
    fn reads(&self, _: &mut Needed) {}
}

/// Keeps the rows of its one child that `predicate` is true for, as `WHERE`
/// does. The predicate reads columns by `ColumnId`, and each comparison's
/// value has its column's type.
pub struct FilterOp {
    pub predicate: Predicate,
}

impl<'c> Op<'c> for FilterOp {
    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        lowering.lower(*at!(node.children, 0))?;
        let allocator = lowering.allocator();
        let predicate = self.predicate.renumbered(allocator, |id| lowering.position(id))?;
        lowering.add_step(Step::Transform(DynTransform::new(allocator, Filter { predicate })?))
    }

    /// Makes no columns.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        self.reads(needed);
        Pruned::Keep
    }

    /// Its predicate's columns.
    fn reads(&self, reads: &mut Needed) {
        for column in self.predicate.columns() {
            reads.need(column);
        }
    }

    /// Constants, which it tests once, and dictionaries of columns it only
    /// compares with strings, whose entries it tests once each: see
    /// `Op::accepts`.
    fn accepts(&self, column: ColumnId) -> Forms {
        self.predicate.accepts(column)
    }

    /// A copy of the conjuncts of its predicate that read only `column`,
    /// unless the others read it too: then it's tested here anyway.
    fn give_condition(
        &mut self,
        allocator: &dyn Allocator,
        column: ColumnId,
    ) -> Result<Given, AllocError> {
        let rest = self.predicate.without_conjuncts_on(allocator, column)?;
        if rest.is_some_and(|rest| rest.columns().any(|read| read == column)) {
            return Ok(Given::Passed(None));
        }
        Ok(Given::Passed(self.predicate.conjuncts_on(allocator, column)?))
    }

    /// Keeps only its conjuncts that read more than `column`, if any.
    fn remove_condition(
        &mut self,
        allocator: &dyn Allocator,
        column: ColumnId,
    ) -> Result<bool, AllocError> {
        let Some(rest) = self.predicate.without_conjuncts_on(allocator, column)? else {
            return Ok(true);
        };
        self.predicate = rest;
        Ok(false)
    }
}
