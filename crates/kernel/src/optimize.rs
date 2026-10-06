//! `optimize`: passes that make a `LogicalPlan` cheaper to run without
//! changing its rows. Each op takes part through `Op`, so an extension's ops
//! are optimized too.

use crate::allocator::{AllocError, Allocator};
use crate::column::Forms;
use crate::plan::{ColumnId, LogicalPlan, PLAN_COLUMNS_MAX, PLAN_NODES_MAX, PlanNode, PlanNodeId};
use crate::slow_vec::SlowVec;

/// Runs every pass over `plan`, with scratch memory from `allocator`.
pub fn optimize(allocator: &dyn Allocator, plan: &mut LogicalPlan<'_>) -> Result<(), AllocError> {
    prune_columns(allocator, plan)?;
    choose_forms(allocator, plan)
}

/// The columns something later in a plan uses, by id.
pub struct Needed {
    columns: SlowVec<bool>,
}

impl Needed {
    /// None of `count` columns.
    fn none(allocator: &dyn Allocator, count: usize) -> Result<Needed, AllocError> {
        Ok(Needed { columns: SlowVec::fixed_from(allocator, core::iter::repeat_n(false, count))? })
    }

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
    /// It stays.
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
    let mut needed = Needed::none(allocator, plan.columns.len())?;
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

/// Where a lazy column is loaded.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Load {
    /// Just below the node with this id, from its child in this slot.
    Below(PlanNodeId, usize),
    /// Above the root, for the plan's output.
    AtRoot,
    /// Nowhere: nothing reads it.
    Never,
}

/// Where `column` is first read above `maker`, the node with that id: below a
/// node, above the root, or nowhere; and whether a node is passed first.
fn first_read(
    parents: &[Option<(PlanNodeId, usize)>],
    reads: &[Needed],
    output: &Needed,
    maker: usize,
    column: ColumnId,
) -> (Load, bool) {
    let (mut at, mut above_parent) = (*at!(parents, maker), false);
    loop {
        match at {
            None if output.is_needed(column) => return (Load::AtRoot, above_parent),
            None => return (Load::Never, above_parent),
            Some((parent, slot)) if at!(reads, parent as usize).is_needed(column) => {
                return (Load::Below(parent, slot), above_parent);
            }
            Some((parent, _)) => (at, above_parent) = (*at!(parents, parent as usize), true),
        }
    }
}

/// The forms the plan's output columns may come in: whoever takes the
/// output can make them flat.
const OUTPUT_FORMS: Forms = Forms::FLAT.union(Forms::CONSTANT).union(Forms::DICTIONARY);

/// Chooses the forms each column is made in. A column may be made in any
/// form the op that first reads it accepts, as ops between pass a column
/// they don't read on as it is; its maker is told which, and says which it
/// will make. Lazy is allowed too where a column is read later than it's
/// made, so its values are read only for the rows left then, or never read:
/// a lazy column is loaded just below the op that first reads it, unless
/// that op accepts it lazy, or above the root for the plan's output. A
/// column read by the op just above its maker isn't made lazy, as nothing
/// between could drop rows, and one nothing reads is never loaded. Each op
/// takes part through `Op`: what it reads and accepts, what it will make,
/// and what loads it.
pub fn choose_forms(
    allocator: &dyn Allocator,
    plan: &mut LogicalPlan<'_>,
) -> Result<(), AllocError> {
    let (nodes, columns) = (plan.nodes.len(), plan.columns.len());
    // Each node's parent, and which of its children it is.
    let orphans = core::iter::repeat_n(None::<(PlanNodeId, usize)>, nodes);
    let mut parents = SlowVec::fixed_from(allocator, orphans)?;
    for (id, node) in (0..).zip(plan.nodes.iter()) {
        for (slot, &child) in node.children.iter().enumerate() {
            *at_mut!(parents, child as usize) = Some((id, slot));
        }
    }
    let mut reads = SlowVec::fixed(allocator, nodes)?;
    for node in plan.nodes.iter() {
        let mut read = Needed::none(allocator, columns)?;
        node.op.reads(&mut read);
        reads.push(read).map_err(|_| AllocError)?;
    }
    let mut output = Needed::none(allocator, columns)?;
    for column in plan.output.iter() {
        output.need(column.id);
    }
    // Each column made lazy: what made it, and where it's loaded.
    let mut lazy = SlowVec::new(allocator, PLAN_COLUMNS_MAX)?;
    for maker in 0..nodes {
        for column in (0..).take(columns) {
            let (load, above_parent) = first_read(&parents, &reads, &output, maker, column);
            let accepted = match load {
                Load::Below(reader, _) => at!(plan.nodes, reader as usize).op.accepts(column),
                Load::AtRoot => OUTPUT_FORMS,
                Load::Never => Forms::FLAT,
            };
            // A column nothing reads may be lazy too, and isn't loaded.
            let worth = above_parent || load == Load::Never;
            let allowed = if worth { accepted | Forms::LAZY } else { accepted };
            let made = at_mut!(plan.nodes, maker).op.allow(column, allowed);
            if made.contains(Forms::LAZY) && !accepted.contains(Forms::LAZY) && load != Load::Never
            {
                lazy.push((maker, load, column, accepted)).map_err(|_| AllocError)?;
            }
        }
    }
    // One op loads the columns one node made lazy that are loaded at one
    // place.
    for (i, &(maker, load, ..)) in lazy.iter().enumerate() {
        let same = |&&(m, l, ..): &&(usize, Load, ColumnId, Forms)| (m, l) == (maker, load);
        if at!(lazy, ..i).iter().any(|entry| same(&entry)) {
            continue;
        }
        let mut loaded = SlowVec::fixed(allocator, at!(lazy, i..).iter().filter(same).count())?;
        for &(_, _, column, forms) in at!(lazy, i..).iter().filter(same) {
            loaded.push((column, forms)).map_err(|_| AllocError)?;
        }
        let op = at!(plan.nodes, maker).op.materialize(allocator, loaded)?;
        let below = match load {
            Load::Below(parent, slot) => *at!(at!(plan.nodes, parent as usize).children, slot),
            Load::AtRoot | Load::Never => plan.root,
        };
        let children = SlowVec::fixed_from(allocator, core::iter::once(below))?;
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
        let id = plan.nodes.len() as PlanNodeId;
        plan.nodes.push(PlanNode { op, children }).map_err(|_| AllocError)?;
        match load {
            Load::Below(parent, slot) => {
                *at_mut!(at_mut!(plan.nodes, parent as usize).children, slot) = id;
            }
            Load::AtRoot | Load::Never => plan.root = id,
        }
    }
    Ok(())
}
