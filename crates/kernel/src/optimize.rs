//! `optimize`: passes that make a `LogicalPlan` cheaper to run without
//! changing its rows. Each op takes part through `Op`, so an extension's ops
//! are optimized too.

use crate::allocator::{AllocError, Allocator};
use crate::column::Forms;
use crate::plan::{
    ColumnId, Given, LogicalPlan, PLAN_COLUMNS_MAX, PLAN_NODES_MAX, PlanNode, PlanNodeId,
};
use crate::slow_vec::SlowVec;

/// Runs every pass over `plan`, with scratch memory from `allocator`.
pub fn optimize(allocator: &dyn Allocator, plan: &mut LogicalPlan<'_>) -> Result<(), AllocError> {
    prune_columns(allocator, plan)?;
    answer_from_summaries(allocator, plan)?;
    push_down_conditions(allocator, plan)?;
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
        // Last first, so children are pruned first to last, as Perfetto does.
        for (slot, &child) in at!(plan.nodes, id as usize).children.iter().enumerate().rev() {
            pending.push((child, Some((id, slot))))?;
        }
    }
    remove_unreachable_nodes(plan);
    Ok(())
}

/// Removes the nodes pruning or answering left unread, so nothing after it
/// can reach them. Inputs come before what reads them, which removing nodes
/// keeps true.
fn remove_unreachable_nodes(plan: &mut LogicalPlan<'_>) {
    let mut reachable = [false; PLAN_NODES_MAX];
    *at_mut!(reachable, plan.root as usize) = true;
    for id in (0..plan.nodes.len()).rev() {
        if *at!(reachable, id) {
            for &child in at!(plan.nodes, id).children.iter() {
                check!((child as usize) < id);
                *at_mut!(reachable, child as usize) = true;
            }
        }
    }
    let mut new_ids = [0; PLAN_NODES_MAX];
    let mut kept: PlanNodeId = 0;
    for (id, node) in plan.nodes.iter_mut().enumerate() {
        if !*at!(reachable, id) {
            continue;
        }
        for child in node.children.iter_mut() {
            *child = *at!(new_ids, *child as usize);
        }
        *at_mut!(new_ids, id) = kept;
        kept += 1;
    }
    let mut id = 0;
    plan.nodes.retain(|_| {
        id += 1;
        *at!(reachable, id - 1)
    });
    plan.root = *at!(new_ids, plan.root as usize);
}

/// Has each op that can make its rows from what its one child knows of its
/// own without reading them, such as an aggregate from a file's statistics,
/// do so. The child is then left unread, and removed.
pub fn answer_from_summaries(
    allocator: &dyn Allocator,
    plan: &mut LogicalPlan<'_>,
) -> Result<(), AllocError> {
    for id in 0..plan.nodes.len() {
        // Children are added before their parents.
        let (below, rest) = plan.nodes.split_at_mut(id);
        let node = at_mut!(rest, 0);
        let &[child] = &node.children[..] else { continue };
        if node.op.answer(allocator, &at!(below, child as usize).op)? {
            node.children = SlowVec::fixed(allocator, 0)?;
        }
    }
    remove_unreachable_nodes(plan);
    Ok(())
}

/// Each node's parent, and which of its children it is; `None` for the
/// root.
fn parents(
    allocator: &dyn Allocator,
    plan: &LogicalPlan<'_>,
) -> Result<SlowVec<Option<(PlanNodeId, usize)>>, AllocError> {
    let orphans = core::iter::repeat_n(None, plan.nodes.len());
    let mut parents = SlowVec::fixed_from(allocator, orphans)?;
    for (id, node) in (0..).zip(plan.nodes.iter()) {
        for (slot, &child) in node.children.iter().enumerate() {
            *at_mut!(parents, child as usize) = Some((id, slot));
        }
    }
    Ok(parents)
}

/// Pushes conditions down: for each column a node makes and takes
/// conditions on, copies of the parts of filters' predicates above it that
/// read only that column, up to the first op that passes on other rows, so
/// it can skip rows failing them. The filters still test every row.
pub fn push_down_conditions(
    allocator: &dyn Allocator,
    plan: &mut LogicalPlan<'_>,
) -> Result<(), AllocError> {
    let parents = parents(allocator, plan)?;
    for maker in 0..plan.nodes.len() {
        for column in (0..).take(plan.columns.len()) {
            if !at!(plan.nodes, maker).op.takes_condition(column) {
                continue;
            }
            let mut at = *at!(parents, maker);
            while let Some((parent, _)) = at {
                let parent_op = &mut at_mut!(plan.nodes, parent as usize).op;
                let given = parent_op.give_condition(allocator, column)?;
                let Given::Passed(condition) = given else { break };
                if let Some(condition) = condition {
                    at_mut!(plan.nodes, maker).op.take_condition(allocator, column, condition)?;
                }
                at = *at!(parents, parent as usize);
            }
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

/// Walks up from `maker`, the node with that id, until an op doesn't pass
/// `column` through. Returns where `column` is first read: below a node,
/// above the root, or nowhere; whether that's higher than `maker`'s parent;
/// and the forms accepted by every op on the way that reads it.
fn first_read(
    nodes: &[PlanNode<'_>],
    parents: &[Option<(PlanNodeId, usize)>],
    reads: &[Needed],
    output: &Needed,
    maker: usize,
    column: ColumnId,
) -> (Load, bool, Forms) {
    let (mut at, mut above_parent) = (*at!(parents, maker), false);
    let (mut first, mut accepted) = (None, OUTPUT_FORMS | Forms::LAZY);
    while let Some((parent, slot)) = at {
        let op = &at!(nodes, parent as usize).op;
        if at!(reads, parent as usize).is_needed(column) {
            accepted = accepted & op.accepts(column);
            first.get_or_insert((Load::Below(parent, slot), above_parent));
        }
        if !op.passes_through(column) {
            return match first {
                Some((load, above_parent)) => (load, above_parent, accepted),
                None => (Load::Never, above_parent, Forms::FLAT),
            };
        }
        (at, above_parent) = (*at!(parents, parent as usize), true);
    }
    if output.is_needed(column) {
        accepted = accepted & OUTPUT_FORMS;
        first.get_or_insert((Load::AtRoot, above_parent));
    }
    match first {
        Some((load, above_parent)) => (load, above_parent, accepted),
        None => (Load::Never, above_parent, Forms::FLAT),
    }
}

/// The forms the plan's output columns may come in: whoever takes the
/// output can make them flat.
const OUTPUT_FORMS: Forms = Forms::FLAT.union(Forms::CONSTANT).union(Forms::DICTIONARY);

/// Chooses the forms each column is made in. A column may be made in any
/// form every op that reads it accepts, as each passes it on as it is; its
/// maker is told which, and says which it
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
    let parents = parents(allocator, plan)?;
    let mut reads = SlowVec::fixed(allocator, nodes)?;
    for node in plan.nodes.iter() {
        let mut read = Needed::none(allocator, columns)?;
        node.op.reads(&mut read);
        reads.push(read)?;
    }
    let mut output = Needed::none(allocator, columns)?;
    for column in plan.output.iter() {
        output.need(column.id);
    }
    // Each column made lazy: what made it, and where it's loaded.
    let mut lazy = SlowVec::new(allocator, PLAN_COLUMNS_MAX)?;
    for maker in 0..nodes {
        for column in (0..).take(columns) {
            let (load, above_parent, accepted) =
                first_read(&plan.nodes, &parents, &reads, &output, maker, column);
            // A column nothing reads may be lazy too, and isn't loaded.
            let worth = above_parent || load == Load::Never;
            let allowed = if worth { accepted | Forms::LAZY } else { accepted };
            let made = at_mut!(plan.nodes, maker).op.allow(column, allowed);
            if made.contains(Forms::LAZY) && !accepted.contains(Forms::LAZY) && load != Load::Never
            {
                lazy.push((maker, load, column, accepted))?;
            }
        }
    }
    // One op loads the columns one node made lazy that are loaded at one
    // place.
    for i in 0..lazy.len() {
        let (maker, load, ..) = *at!(lazy, i);
        let same = |&&(m, l, ..): &&(usize, Load, ColumnId, Forms)| (m, l) == (maker, load);
        if at!(lazy, ..i).iter().any(|entry| same(&entry)) {
            continue;
        }
        let mut loaded = SlowVec::fixed(allocator, at!(lazy, i..).iter().filter(same).count())?;
        for &(_, _, column, forms) in at!(lazy, i..).iter().filter(same) {
            loaded.push((column, forms))?;
        }
        let op = at!(plan.nodes, maker).op.materialize(allocator, loaded)?;
        let below = match load {
            Load::Below(parent, slot) => *at!(at!(plan.nodes, parent as usize).children, slot),
            Load::AtRoot | Load::Never => plan.root,
        };
        let children = SlowVec::fixed_from(allocator, core::iter::once(below))?;
        // Just before what reads it, so inputs still come before their readers.
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
        let id = match load {
            Load::Below(parent, _) => parent,
            Load::AtRoot | Load::Never => plan.nodes.len() as PlanNodeId,
        };
        insert_node(plan, id, PlanNode { op, children })?;
        for (maker, load, ..) in lazy.iter_mut() {
            *maker += usize::from(*maker >= id as usize);
            if let Load::Below(parent, _) = load {
                *parent += PlanNodeId::from(*parent >= id);
            }
        }
        match load {
            Load::Below(parent, slot) => {
                *at_mut!(at_mut!(plan.nodes, parent as usize + 1).children, slot) = id;
            }
            Load::AtRoot | Load::Never => plan.root = id,
        }
    }
    Ok(())
}

/// Adds `node` to `plan` as node `id`, moving the nodes from `id` on up one.
fn insert_node<'c>(
    plan: &mut LogicalPlan<'c>,
    id: PlanNodeId,
    node: PlanNode<'c>,
) -> Result<(), AllocError> {
    plan.nodes.push(node)?;
    at_mut!(plan.nodes, id as usize..).rotate_right(1);
    let moved = |node: PlanNodeId| if node >= id { node + 1 } else { node };
    for (at, node) in plan.nodes.iter_mut().enumerate() {
        if at != id as usize {
            for child in node.children.iter_mut() {
                *child = moved(*child);
            }
        }
    }
    plan.root = moved(plan.root);
    Ok(())
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use super::*;
    use crate::allocator::Heap;
    use crate::column::DataType;
    use crate::lower::{LowerError, Lowering};
    use crate::output_properties::{DeriveContext, OutputProperties};
    use crate::plan::{DynOp, LogicalPlan, Op, PlanNode};

    /// Makes column 0, and records the forms it's allowed in.
    struct Maker<'a>(&'a Cell<Forms>);

    /// Reads column 0 in the forms `accepts`, and passes it through if
    /// `passes_through`.
    struct Reader {
        accepts: Forms,
        passes_through: bool,
    }

    impl<'c> Op<'c> for Maker<'_> {
        fn allow(&mut self, column: ColumnId, allowed: Forms) -> Forms {
            if column == 0 {
                self.0.set(allowed);
            }
            Forms::FLAT
        }

        fn derive(&self, _: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
            failed!()
        }

        fn lower(&self, _: &PlanNode<'c>, _: &mut Lowering<'_, 'c>) -> Result<(), LowerError> {
            failed!()
        }
    }

    impl<'c> Op<'c> for Reader {
        fn accepts(&self, _: ColumnId) -> Forms {
            self.accepts
        }

        fn passes_through(&self, _: ColumnId) -> bool {
            self.passes_through
        }

        fn derive(&self, _: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
            failed!()
        }

        fn lower(&self, _: &PlanNode<'c>, _: &mut Lowering<'_, 'c>) -> Result<(), LowerError> {
            failed!()
        }
    }

    /// Plans a maker, then a reader that takes dictionaries, then one that
    /// only takes flat columns, and returns the forms the maker is allowed.
    fn allowed(passes_through: bool) -> Forms {
        let allowed = Cell::new(Forms::FLAT);
        let mut plan = LogicalPlan::new(&Heap).unwrap();
        let column = plan.add_column("a", DataType::String).unwrap();
        let mut node = plan
            .add_node(
                DynOp::new(&Heap, Maker(&allowed)).unwrap(),
                SlowVec::fixed(&Heap, 0).unwrap(),
            )
            .unwrap();
        let readers = [
            Reader { accepts: Forms::FLAT | Forms::DICTIONARY, passes_through },
            Reader { accepts: Forms::FLAT, passes_through: true },
        ];
        for reader in readers {
            let children = SlowVec::fixed_from(&Heap, [node].into_iter()).unwrap();
            node = plan.add_node(DynOp::new(&Heap, reader).unwrap(), children).unwrap();
        }
        assert!(plan.output.push(column).is_ok());
        choose_forms(&Heap, &mut plan).unwrap();
        allowed.get()
    }

    #[test]
    fn forms_ignore_readers_above_an_op_that_does_not_pass_the_column_through() {
        assert!(allowed(false).contains(Forms::DICTIONARY));
        assert!(!allowed(true).contains(Forms::DICTIONARY));
    }
}
