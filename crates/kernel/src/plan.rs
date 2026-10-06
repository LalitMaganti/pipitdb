//! `LogicalPlan`: what a query does, as Perfetto's: a tree of operations over
//! columns named by id, independent of how batches lay them out. Frontends
//! build one; `lower` turns it into a pipeline, asking each operation to
//! lower itself, so extensions can add operations.

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::boxed::{Box, ErasedBox};
use crate::column::{DataType, Forms};
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
    /// flat only.
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

    /// A condition every row this passes on meets, as a filter's: what's
    /// below it can skip what can't meet it. By default, none.
    fn condition(&self) -> Option<&Predicate> {
        None
    }

    /// Tells this, if it makes rows, conditions every row the plan keeps of
    /// them meets, so it can skip row groups whose bounds rule one out. By
    /// default, it skips none.
    fn skip_row_groups(
        &mut self,
        allocator: &dyn Allocator,
        conditions: &[&Predicate],
    ) -> Result<(), AllocError> {
        let _ = (allocator, conditions);
        Ok(())
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

type SkipRowGroups =
    unsafe fn(NonNull<()>, &dyn Allocator, &[&Predicate]) -> Result<(), AllocError>;

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
    condition: unsafe fn(NonNull<()>) -> Option<NonNull<Predicate>>,
    skip_row_groups: SkipRowGroups,
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
            // SAFETY: as for `lower`.
            condition: |op| unsafe { value_of::<T>(op).condition().map(NonNull::from) },
            // SAFETY: as for `prune`.
            skip_row_groups: |op, allocator, conditions| unsafe {
                value_mut_of::<T>(op).skip_row_groups(allocator, conditions)
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

    pub(crate) fn condition(&self) -> Option<&Predicate> {
        // SAFETY: the function matches the op's type, and the condition is
        // the op's, borrowed with it.
        unsafe { (self.condition)(self.op.as_ptr()).map(|condition| condition.as_ref()) }
    }

    pub(crate) fn skip_row_groups(
        &mut self,
        allocator: &dyn Allocator,
        conditions: &[&Predicate],
    ) -> Result<(), AllocError> {
        // SAFETY: the function matches the op's type, which `self` holds
        // mutably.
        unsafe { (self.skip_row_groups)(self.op.as_ptr(), allocator, conditions) }
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

/// Reads the rows of a scannable, binding the columns in `columns`: those
/// of `row_groups`, or all if `None`.
pub struct ScanOp<'c> {
    scannable: &'c DynScannable<'c>,
    columns: SlowVec<ScanColumn>,
    row_groups: Option<SlowVec<u32>>,
}

impl<'c> ScanOp<'c> {
    pub fn new(scannable: &'c DynScannable<'c>, columns: SlowVec<ScanColumn>) -> ScanOp<'c> {
        ScanOp { scannable, columns, row_groups: None }
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
        let row_groups = match &self.row_groups {
            Some(row_groups) => {
                Some(SlowVec::fixed_from(lowering.allocator(), row_groups.iter().copied())?)
            }
            None => None,
        };
        lowering.set_source(self.scannable.scan(lowering.allocator(), read, row_groups)?);
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

    fn allow(&mut self, column: ColumnId, allowed: Forms) -> Forms {
        let found = self.columns.iter_mut().find(|c| c.binding.id == column);
        // Chosen once: a column chosen already is loaded already.
        let Some(scanned) = found.filter(|scanned| scanned.forms == Forms::FLAT) else {
            return Forms::FLAT;
        };
        scanned.forms = allowed & self.scannable.forms(scanned.column);
        scanned.forms
    }

    /// Skips the row groups whose bounds, the scannable's, rule a condition
    /// out, if any: worked out from statistics, before any row is read.
    fn skip_row_groups(
        &mut self,
        allocator: &dyn Allocator,
        conditions: &[&Predicate],
    ) -> Result<(), AllocError> {
        let bounds = |group: u32, id: ColumnId| {
            let scanned = self.columns.iter().find(|scanned| scanned.binding.id == id)?;
            self.scannable.bounds(group, scanned.column)
        };
        let count = self.scannable.row_group_count();
        let mut row_groups = SlowVec::fixed(allocator, count as usize)?;
        for group in 0..count {
            if conditions.iter().all(|condition| condition.may_hold(|id| bounds(group, id))) {
                row_groups.push(group)?;
            }
        }
        if row_groups.len() < count as usize {
            self.row_groups = Some(row_groups);
        }
        Ok(())
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

    fn condition(&self) -> Option<&Predicate> {
        Some(&self.predicate)
    }

    /// Flat, and constants, which it tests once. Not dictionaries yet.
    fn accepts(&self, _: ColumnId) -> Forms {
        Forms::FLAT | Forms::CONSTANT
    }
}
