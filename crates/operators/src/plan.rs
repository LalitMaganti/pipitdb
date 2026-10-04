//! `LogicalPlan`: what a query does, as Perfetto's: a tree of operations over
//! columns named by id, independent of how batches lay them out. Frontends
//! compile to it, finding tables in a `Catalog`, and `lower` turns it into a
//! pipeline.

use pipit_kernel::allocator::{AllocError, Allocator};
use pipit_kernel::column::DataType;
use pipit_kernel::vec::Vec;

use crate::names::{Name, Names};
use crate::table::Table;

/// The most columns a plan can have.
pub const PLAN_COLUMNS_MAX: usize = 1 << 10;
/// The most nodes a plan can have.
pub const PLAN_NODES_MAX: usize = 1 << 6;
/// The most bytes a plan's names take.
pub const PLAN_NAME_BYTES_MAX: usize = 1 << 14;

/// The tables a frontend can read, provided by the embedder.
pub trait Catalog {
    /// The table registered as `name`, if any.
    fn find_table(&self, name: &str) -> Option<&Table>;
}

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

pub enum Op<'t> {
    /// Reads all the rows of a table. `columns` binds each of the table's
    /// columns, in the table's order.
    Scan { table: &'t Table, columns: Vec<NamedColumn> },
}

/// An operation, and the nodes whose rows it reads.
pub struct PlanNode<'t> {
    pub op: Op<'t>,
    pub children: Vec<PlanNodeId>,
}

/// Borrows the tables it reads, for `'t`.
pub struct LogicalPlan<'t> {
    pub names: Names,
    /// Indexed by `ColumnId`.
    pub columns: Vec<ColumnSchema>,
    pub nodes: Vec<PlanNode<'t>>,
    /// The node whose rows are the plan's rows.
    pub root: PlanNodeId,
    /// The columns of the result, in order.
    pub output: Vec<NamedColumn>,
}

impl<'t> LogicalPlan<'t> {
    pub fn new<A: Allocator + Clone + 'static>(
        allocator: A,
    ) -> Result<LogicalPlan<'t>, AllocError> {
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
        op: Op<'t>,
        children: Vec<PlanNodeId>,
    ) -> Result<PlanNodeId, AllocError> {
        let id = self.nodes.len() as PlanNodeId;
        self.nodes.push(PlanNode { op, children })?;
        self.root = id;
        Ok(id)
    }
}
