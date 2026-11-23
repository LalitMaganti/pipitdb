//! `optimize`: passes that make a `LogicalPlan` cheaper to run without
//! changing its rows. Each op takes part through `Op`, so an extension's ops
//! are optimized too.

use crate::allocator::AllocError;
use crate::column::Forms;
use crate::estimate::Estimate;
use crate::expression::Exprs;
use crate::global::Global;
use crate::join_planner::plan_regions;
use crate::plan::{
    Children, ColumnId, Given, LogicalPlan, PLAN_COLUMNS_MAX, PLAN_NODES_MAX, PlanNode, PlanNodeId,
};
use crate::slow_vec::SlowVec;

/// Runs every pass over `plan`, with scratch memory from `allocator`.
pub fn optimize(allocator: &Global, plan: &mut LogicalPlan<'_>) -> Result<(), AllocError> {
    prune_columns(allocator, plan)?;
    answer_from_summaries(allocator, plan)?;
    place_filters(allocator, plan)?;
    push_down_conditions(allocator, plan)?;
    // Runs after pushdown, which can make scans skip rows, and before
    // forms, which depend on what is chosen here. Joins planned in a new
    // order choose again, and keep only the columns needed.
    let estimates = choose_by_estimates(allocator, plan)?;
    if plan_regions(allocator, plan, &estimates)? {
        prune_columns(allocator, plan)?;
        choose_by_estimates(allocator, plan)?;
    }
    choose_forms(allocator, plan)
}

/// The columns something later in a plan uses, by id.
pub struct Needed<'e> {
    columns: SlowVec<bool>,
    /// The plan's expressions, which ops' columns are read in.
    exprs: &'e Exprs,
}

impl<'e> Needed<'e> {
    /// None of `count` columns, of a plan whose expressions are `exprs`.
    pub(crate) fn none(
        allocator: &Global,
        count: usize,
        exprs: &'e Exprs,
    ) -> Result<Needed<'e>, AllocError> {
        let columns = SlowVec::fixed_from(allocator, core::iter::repeat_n(false, count))?;
        Ok(Needed { columns, exprs })
    }

    /// The plan's expressions.
    pub fn exprs(&self) -> &'e Exprs {
        self.exprs
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
pub fn prune_columns(allocator: &Global, plan: &mut LogicalPlan<'_>) -> Result<(), AllocError> {
    let mut needed = Needed::none(allocator, plan.columns.len(), &plan.exprs)?;
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
/// can reach them, and orders the rest so inputs come before what reads
/// them: a pass may rebuild part of a plan in another order.
fn remove_unreachable_nodes(plan: &mut LogicalPlan<'_>) {
    // Each node's new id: the order a walk from the root finishes them, then
    // the unreachable ones'.
    let mut new_ids = [PlanNodeId::MAX; PLAN_NODES_MAX];
    let mut pending = [(0, false); 2 * PLAN_NODES_MAX];
    let (mut count, mut kept) = (1, 0);
    *at_mut!(pending, 0) = (plan.root, false);
    while count > 0 {
        count -= 1;
        let (id, children_done) = *at!(pending, count);
        if *at!(new_ids, id as usize) != PlanNodeId::MAX {
            continue;
        }
        if children_done {
            *at_mut!(new_ids, id as usize) = kept;
            kept += 1;
            continue;
        }
        *at_mut!(pending, count) = (id, true);
        count += 1;
        for &child in at!(plan.nodes, id as usize).children.iter() {
            *at_mut!(pending, count) = (child, false);
            count += 1;
        }
    }
    let unreachable = at_mut!(new_ids, ..plan.nodes.len()).iter_mut();
    for (id, next) in unreachable.filter(|id| **id == PlanNodeId::MAX).zip(kept..) {
        *id = next;
    }
    for node in plan.nodes.iter_mut() {
        for child in node.children.iter_mut() {
            *child = *at!(new_ids, *child as usize);
        }
    }
    plan.root = *at!(new_ids, plan.root as usize);
    // Moves each node to its new id along the cycles of the order.
    for i in 0..plan.nodes.len() {
        while *at!(new_ids, i) as usize != i {
            let to = *at!(new_ids, i) as usize;
            plan.nodes.swap(i, to);
            new_ids.swap(i, to);
        }
    }
    while plan.nodes.len() > kept as usize {
        plan.nodes.pop();
    }
}

/// Has each op that can make its rows from what its one child knows of its
/// own without reading them, such as an aggregate from a file's statistics,
/// do so. The child is then left unread, and removed.
pub fn answer_from_summaries(
    allocator: &Global,
    plan: &mut LogicalPlan<'_>,
) -> Result<(), AllocError> {
    for id in 0..plan.nodes.len() {
        // Children are added before their parents.
        let (below, rest) = plan.nodes.split_at_mut(id);
        let node = at_mut!(rest, 0);
        let &[child] = &node.children[..] else { continue };
        if node.op.answer(allocator, &at!(below, child as usize).op)? {
            node.children = Children::new();
        }
    }
    remove_unreachable_nodes(plan);
    Ok(())
}

/// Moves each filter's conditions as far down the plan as possible, so rows
/// are dropped before work is done on them. For each condition, the op below
/// the filter says which of its inputs, if any, it can go to
/// (`Op::moves_condition`), and the condition becomes a filter on that input.
/// This repeats until nothing moves. A filter left with no conditions is
/// removed.
pub fn place_filters(allocator: &Global, plan: &mut LogicalPlan<'_>) -> Result<(), AllocError> {
    // Filters only move down, so this ends.
    while move_one_filter(allocator, plan)? {}
    remove_unreachable_nodes(plan);
    Ok(())
}

/// Moves the conditions of the first filter that can move any. Returns
/// whether one did.
fn move_one_filter(allocator: &Global, plan: &mut LogicalPlan<'_>) -> Result<bool, AllocError> {
    for id in 0..plan.nodes.len() {
        let &[child] = &at!(plan.nodes, id).children[..] else { continue };
        // Children are added before their parents.
        let (below, rest) = plan.nodes.split_at_mut(id);
        let below = &at!(below, child as usize).op;
        let moved = at_mut!(rest, 0).op.split_below(&plan.exprs, allocator, below)?;
        if moved.filters.iter().all(Option::is_none) {
            continue;
        }
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
        let mut filter = id as PlanNodeId;
        let mut under = child;
        for (input, op) in moved.filters.into_iter().enumerate() {
            let Some(op) = op else { continue };
            let grandchild = *at!(at!(plan.nodes, under as usize).children, input);
            let children = Children::from_slice(&[grandchild])?;
            // Inserted just before the op it goes under, so every node still
            // comes after its inputs.
            insert_node(plan, under, PlanNode { op, children })?;
            *at_mut!(at_mut!(plan.nodes, under as usize + 1).children, input) = under;
            (under, filter) = (under + 1, filter + 1);
        }
        if moved.emptied {
            bypass(plan, filter, under);
            // Detached, so nothing reaches it and it isn't moved again.
            at_mut!(plan.nodes, filter as usize).children = Children::new();
        }
        return Ok(true);
    }
    Ok(false)
}

/// Makes every node that reads node `id` read `child` instead, leaving `id`
/// unreachable.
fn bypass(plan: &mut LogicalPlan<'_>, id: PlanNodeId, child: PlanNodeId) {
    for node in plan.nodes.iter_mut() {
        for c in node.children.iter_mut() {
            if *c == id {
                *c = child;
            }
        }
    }
    if plan.root == id {
        plan.root = child;
    }
}

/// Works out each node's estimate from its children's, from the scans up,
/// and lets each op make the choices that depend on them, such as which
/// side of a join to build. Returns the estimates.
pub fn choose_by_estimates(
    allocator: &Global,
    plan: &mut LogicalPlan<'_>,
) -> Result<SlowVec<Estimate>, AllocError> {
    let mut estimates = SlowVec::fixed(allocator, plan.nodes.len())?;
    let mut inputs = SlowVec::new(allocator, PLAN_NODES_MAX)?;
    let exprs = &plan.exprs;
    for node in plan.nodes.iter_mut() {
        inputs.retain(|_| false);
        for &child in node.children.iter() {
            // Children are added before their parents.
            inputs.push(*at!(estimates, child as usize))?;
        }
        node.op.choose(exprs, &inputs);
        estimates.push(node.op.estimate(exprs, &inputs))?;
    }
    Ok(estimates)
}

/// Each node's parent, and which of its children it is; `None` for the
/// root.
fn parents(
    allocator: &Global,
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
    allocator: &Global,
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
                let given = parent_op.give_condition(&plan.exprs, allocator, column)?;
                let Given::Passed(condition) = given else { break };
                if let Some(condition) = condition {
                    let maker = &mut at_mut!(plan.nodes, maker).op;
                    maker.take_condition(&plan.exprs, allocator, column, condition)?;
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
    exprs: &Exprs,
    nodes: &[PlanNode<'_>],
    parents: &[Option<(PlanNodeId, usize)>],
    reads: &[Needed<'_>],
    output: &Needed<'_>,
    maker: usize,
    column: ColumnId,
) -> (Load, bool, Forms) {
    let (mut at, mut above_parent) = (*at!(parents, maker), false);
    let (mut first, mut accepted) = (None, ANY_FORM);
    while let Some((parent, slot)) = at {
        let op = &at!(nodes, parent as usize).op;
        if at!(reads, parent as usize).is_needed(column) {
            accepted = accepted & op.accepts(exprs, column);
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

/// Every form, which what reads a column narrows.
const ANY_FORM: Forms = OUTPUT_FORMS.union(Forms::LAZY).union(Forms::SEQUENCE);

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
pub fn choose_forms(allocator: &Global, plan: &mut LogicalPlan<'_>) -> Result<(), AllocError> {
    let (nodes, columns) = (plan.nodes.len(), plan.columns.len());
    let parents = parents(allocator, plan)?;
    let mut reads = SlowVec::fixed(allocator, nodes)?;
    for node in plan.nodes.iter() {
        let mut read = Needed::none(allocator, columns, &plan.exprs)?;
        node.op.reads(&mut read);
        reads.push(read)?;
    }
    let mut output = Needed::none(allocator, columns, &plan.exprs)?;
    for column in plan.output.iter() {
        output.need(column.id);
    }
    // Each column made lazy: what made it, and where it's loaded.
    let mut lazy = SlowVec::new(allocator, PLAN_COLUMNS_MAX)?;
    for maker in 0..nodes {
        for column in (0..).take(columns) {
            let (load, above_parent, accepted) =
                first_read(&plan.exprs, &plan.nodes, &parents, &reads, &output, maker, column);
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
    // They read the plan's expressions, which go on, but not the nodes,
    // which change.
    drop((reads, output));
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
        let children = Children::from_slice(&[below])?;
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

    use crate::column::DataType;
    use crate::estimate::Estimate;
    use crate::expression::{Expr, ExprBuilder, ExprOp, Exprs};
    use crate::lower::{LowerError, Lowering};
    use crate::output_properties::{DeriveContext, OutputProperties};
    use crate::plan::{Conjuncts, DynOp, Given, LogicalPlan, Op, PlanNode};

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
        fn accepts(&self, _: &Exprs, _: ColumnId) -> Forms {
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

    /// Plans a maker, then a reader that takes `accepts`, then one that
    /// only takes flat columns, and returns the forms the maker is allowed.
    fn allowed(accepts: Forms, passes_through: bool) -> Forms {
        let global = Global::current();
        let allowed = Cell::new(Forms::FLAT);
        let mut plan = LogicalPlan::new(global).unwrap();
        let column = plan.add_column("a", DataType::String, true).unwrap();
        let mut node = plan.add_node(DynOp::new(global, Maker(&allowed)).unwrap(), &[]).unwrap();
        let readers = [
            Reader { accepts, passes_through },
            Reader { accepts: Forms::FLAT, passes_through: true },
        ];
        for reader in readers {
            let children = [node];
            node = plan.add_node(DynOp::new(global, reader).unwrap(), &children).unwrap();
        }
        assert!(plan.output.push(column).is_ok());
        choose_forms(global, &mut plan).unwrap();
        allowed.get()
    }

    #[test]
    fn forms_ignore_readers_above_an_op_that_does_not_pass_the_column_through() {
        let dictionaries = Forms::FLAT | Forms::DICTIONARY;
        assert!(allowed(dictionaries, false).contains(Forms::DICTIONARY));
        assert!(!allowed(dictionaries, true).contains(Forms::DICTIONARY));
    }

    #[test]
    fn forms_are_any_their_readers_take() {
        let sequences = Forms::FLAT | Forms::SEQUENCE;
        assert!(allowed(sequences, false).contains(Forms::SEQUENCE));
        // The output is never a sequence.
        assert!(!allowed(sequences, true).contains(Forms::SEQUENCE));
    }

    /// Takes conditions on column 0, and records that it took one.
    struct Taker<'a>(&'a Cell<bool>);

    /// Gives the ops below it its condition, `IS NULL` on column 0.
    struct Giver(Expr);

    /// Records, when it chooses, whether `taken` was set by then.
    struct Chooser<'a> {
        taken: &'a Cell<bool>,
        seen: &'a Cell<Option<bool>>,
    }

    impl<'c> Op<'c> for Taker<'_> {
        fn takes_condition(&self, column: ColumnId) -> bool {
            column == 0
        }

        fn take_condition(
            &mut self,
            _: &Exprs,
            _: &Global,
            _: ColumnId,
            _: Conjuncts,
        ) -> Result<(), AllocError> {
            self.0.set(true);
            Ok(())
        }

        fn derive(&self, _: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
            failed!()
        }

        fn lower(&self, _: &PlanNode<'c>, _: &mut Lowering<'_, 'c>) -> Result<(), LowerError> {
            failed!()
        }
    }

    impl<'c> Op<'c> for Giver {
        fn give_condition(
            &mut self,
            _: &Exprs,
            allocator: &Global,
            _: ColumnId,
        ) -> Result<Given, AllocError> {
            Ok(Given::Passed(Some(SlowVec::fixed_from(allocator, [self.0].into_iter())?)))
        }

        fn derive(&self, _: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
            failed!()
        }

        fn lower(&self, _: &PlanNode<'c>, _: &mut Lowering<'_, 'c>) -> Result<(), LowerError> {
            failed!()
        }
    }

    impl<'c> Op<'c> for Chooser<'_> {
        fn choose(&mut self, _: &Exprs, _: &[Estimate]) {
            self.seen.set(Some(self.taken.get()));
        }

        fn derive(&self, _: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
            failed!()
        }

        fn lower(&self, _: &PlanNode<'c>, _: &mut Lowering<'_, 'c>) -> Result<(), LowerError> {
            failed!()
        }
    }

    #[test]
    fn orders_reachable_nodes_inputs_first() {
        // Node 1 reads node 3, after it, and nothing reads nodes 0 and 2.
        let mut plan = LogicalPlan::new(Global::current()).unwrap();
        let reader = || {
            DynOp::new(Global::current(), Reader { accepts: Forms::FLAT, passes_through: true })
                .unwrap()
        };
        for children in [&[][..], &[0], &[], &[]] {
            plan.add_node(reader(), children).unwrap();
        }
        at_mut!(plan.nodes, 1).children = Children::from_slice(&[3]).unwrap();
        plan.root = 1;
        remove_unreachable_nodes(&mut plan);
        assert_eq!(plan.nodes.len(), 2);
        assert_eq!((plan.root, &at!(plan.nodes, 1).children[..]), (1, &[0][..]));
    }

    #[test]
    fn chooses_once_conditions_are_pushed_down() {
        let (taken, seen) = (Cell::new(false), Cell::new(None));
        let mut plan = LogicalPlan::new(Global::current()).unwrap();
        let column = plan.add_column("a", DataType::Int64, true).unwrap();
        let mut null = ExprBuilder::new(Global::current()).unwrap();
        let a = null.column(0).unwrap();
        null.node(ExprOp::IsNull, a, 0).unwrap();
        let null = plan.exprs.add(null.nodes()).unwrap();
        let mut node =
            plan.add_node(DynOp::new(Global::current(), Taker(&taken)).unwrap(), &[]).unwrap();
        for op in [
            DynOp::new(Global::current(), Giver(null)).unwrap(),
            DynOp::new(Global::current(), Chooser { taken: &taken, seen: &seen }).unwrap(),
        ] {
            node = plan.add_node(op, &[node]).unwrap();
        }
        assert!(plan.output.push(column).is_ok());
        optimize(Global::current(), &mut plan).unwrap();
        assert_eq!(seen.get(), Some(true));
    }
}
