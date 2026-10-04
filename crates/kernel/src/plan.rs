//! `LogicalPlan`: what a query does, as Perfetto's: a tree of operations over
//! columns named by id, independent of how batches lay them out. Frontends
//! build one; `lower` turns it into a pipeline, asking each operation to
//! lower itself, so extensions can add operations.

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::boxed::{Box, ErasedBox};
use crate::column::DataType;
use crate::erase::value_of;
use crate::lower::Lowering;
use crate::names::{Name, Names};
use crate::scannable::DynScannable;
use crate::vec::Vec;

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
    -> Result<(), AllocError>;
}

/// An `Op` of any type that lives for `'c`, owned in memory from an
/// allocator, and the function that knows its type.
pub struct DynOp<'c> {
    op: ErasedBox,
    lower: for<'l> unsafe fn(
        NonNull<()>,
        &PlanNode<'c>,
        &mut Lowering<'l, 'c>,
    ) -> Result<(), AllocError>,
    lifetime: PhantomData<&'c ()>,
}

impl<'c> DynOp<'c> {
    pub fn new<A: Allocator + Clone + 'static, T: Op<'c> + 'c>(
        allocator: A,
        op: T,
    ) -> Result<DynOp<'c>, AllocError> {
        Ok(DynOp {
            op: Box::new(allocator, op)?.erase(),
            // SAFETY: only called with this op.
            lower: |op, node, lowering| unsafe { value_of::<T>(op).lower(node, lowering) },
            lifetime: PhantomData,
        })
    }

    pub(crate) fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), AllocError> {
        // SAFETY: the function matches the op's type.
        unsafe { (self.lower)(self.op.as_ptr(), node, lowering) }
    }
}

/// An operation, and the nodes whose rows it reads.
pub struct PlanNode<'c> {
    pub op: DynOp<'c>,
    pub children: Vec<PlanNodeId>,
}

/// Borrows what it reads, such as a catalog's tables, for `'c`.
pub struct LogicalPlan<'c> {
    pub names: Names,
    /// Indexed by `ColumnId`.
    pub columns: Vec<ColumnSchema>,
    pub nodes: Vec<PlanNode<'c>>,
    /// The node whose rows are the plan's rows.
    pub root: PlanNodeId,
    /// The columns of the result, in order.
    pub output: Vec<NamedColumn>,
}

impl<'c> LogicalPlan<'c> {
    pub fn new<A: Allocator + Clone + 'static>(
        allocator: A,
    ) -> Result<LogicalPlan<'c>, AllocError> {
        Ok(LogicalPlan {
            names: Names::new(allocator.clone(), PLAN_NAME_BYTES_MAX)?,
            columns: Vec::new(allocator.clone(), PLAN_COLUMNS_MAX)?,
            nodes: Vec::new(allocator.clone(), PLAN_NODES_MAX)?,
            root: 0,
            output: Vec::new(allocator, PLAN_COLUMNS_MAX)?,
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
        children: Vec<PlanNodeId>,
    ) -> Result<PlanNodeId, AllocError> {
        let id = self.nodes.len() as PlanNodeId;
        self.nodes.push(PlanNode { op, children })?;
        self.root = id;
        Ok(id)
    }
}

/// Reads all the rows of a scannable. `columns` binds each of its columns,
/// in its order.
pub struct ScanOp<'c> {
    pub scannable: &'c DynScannable<'c>,
    pub columns: Vec<NamedColumn>,
}

impl<'c> Op<'c> for ScanOp<'c> {
    fn lower(&self, _: &PlanNode<'c>, lowering: &mut Lowering<'_, 'c>) -> Result<(), AllocError> {
        check!(self.columns.len() == self.scannable.column_count() as usize);
        let all = Vec::fixed_from(lowering.allocator(), 0..self.scannable.column_count())?;
        lowering.set_source(self.scannable.scan(lowering.allocator(), all)?);
        for column in self.columns.iter() {
            lowering.define(column.id);
        }
        Ok(())
    }
}
