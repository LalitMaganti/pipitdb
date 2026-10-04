//! `optimize`: passes that make a `LogicalPlan` cheaper to run without
//! changing its rows. Each op takes part through `Op`, so an extension's ops
//! are optimized too.

use crate::allocator::{AllocError, Allocator};
use crate::plan::{ColumnId, LogicalPlan, PLAN_NODES_MAX, PlanNodeId};
use crate::slow_vec::SlowVec;

/// Runs every pass over `plan`, with scratch memory from `allocator`.
pub fn optimize(allocator: &dyn Allocator, plan: &mut LogicalPlan<'_>) -> Result<(), AllocError> {
    prune_columns(allocator, plan)
}

/// The columns something later in a plan uses, by id.
pub struct Needed {
    columns: SlowVec<bool>,
}

impl Needed {
    pub fn is_needed(&self, column: ColumnId) -> bool {
        *at!(self.columns, column as usize)
    }

    /// Marks `column` as used, such as by an op that reads it.
    pub fn need(&mut self, column: ColumnId) {
        *at_mut!(self.columns, column as usize) = true;
    }

    /// Marks every column as used, for an op that can't say what it reads.
    pub fn need_all(&mut self) {
        self.columns.fill(true);
    }
}

/// What a node becomes once pruned.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pruned {
    Keep,
    /// Its work isn't needed: its `n`th child takes its place.
    Child(u32),
}

/// Drops columns that nothing in the plan uses, so sources read less and ops
/// skip work nobody reads, as Perfetto's `PruneColumns`. From the output down,
/// each op drops what it makes that isn't needed, and marks what it reads.
pub fn prune_columns(
    allocator: &dyn Allocator,
    plan: &mut LogicalPlan<'_>,
) -> Result<(), AllocError> {
    let unneeded = core::iter::repeat_n(false, plan.columns.len());
    let mut needed = Needed { columns: SlowVec::fixed_from(allocator, unneeded)? };
    for column in plan.output.iter() {
        needed.need(column.id);
    }
    // Nodes left to prune, and where each is referred to from: its parent's
    // children, or the root if `None`.
    let mut pending = SlowVec::new(allocator, PLAN_NODES_MAX)?;
    pending.push((plan.root, None::<(PlanNodeId, usize)>))?;
    while let Some((mut id, from)) = pending.pop() {
        loop {
            let node = at_mut!(plan.nodes, id as usize);
            match node.op.prune(&mut needed) {
                Pruned::Keep => break,
                Pruned::Child(child) => id = *at!(node.children, child as usize),
            }
        }
        match from {
            None => plan.root = id,
            Some((parent, slot)) => {
                *at_mut!(at_mut!(plan.nodes, parent as usize).children, slot) = id;
            }
        }
        for (slot, &child) in at!(plan.nodes, id as usize).children.iter().enumerate() {
            pending.push((child, Some((id, slot))))?;
        }
    }
    Ok(())
}
