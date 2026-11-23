//! The planner's first and last stages: finding each region of joins in a
//! plan, and rebuilding it in the order planned.

use crate::allocator::AllocError;
use crate::error::Error;
use crate::estimate::Estimate;
use crate::expression::{Expr, Exprs};
use crate::global::Global;
use crate::join::{JoinKind as Kind, JoinOp, equality};
use crate::plan::{ColumnId, DynOp, LogicalPlan, PLAN_COLUMNS_MAX, PLAN_NODES_MAX, PlanNodeId};
use crate::scannable::Key;
use crate::slow_vec::SlowVec;

use super::{
    EquivalenceSet, JoinEdge, JoinKind, JoinNode, JoinTree, QueryGraph, Relation, RelationSet,
};

/// No relation, or no node.
const NONE: u32 = u32::MAX;

/// The most nodes a region's join tree has: a leaf for each relation, and a
/// join for each but one.
const TREE_MAX: usize = 2 * crate::plan::PLAN_NODES_MAX;

/// An equality between two columns, of different relations, that a join in
/// a region requires.
#[derive(Clone, Copy)]
struct Equality {
    /// The conjunct that requires it.
    conjunct: Expr,
    /// Its columns, and their relations.
    columns: [ColumnId; 2],
    relations: [u32; 2],
}

/// Plans each region of joins in `plan`, given each node's estimate, and
/// rebuilds those it finds a plan for in their new order. Returns whether
/// any changed, which leaves the plan to prune again.
pub(crate) fn plan_regions(
    global: &Global,
    plan: &mut LogicalPlan<'_>,
    estimates: &[Estimate],
) -> Result<bool, AllocError> {
    let count = plan.nodes.len();
    // Each node's parent, `NONE` for the root and unreachable nodes.
    let mut parents = [NONE; PLAN_NODES_MAX];
    let mut pending = [0; PLAN_NODES_MAX];
    let mut waiting = 1;
    *at_mut!(pending, 0) = plan.root;
    while waiting > 0 {
        waiting -= 1;
        let id = *at!(pending, waiting);
        for &child in at!(plan.nodes, id as usize).children.iter() {
            *at_mut!(parents, child as usize) = id;
            *at_mut!(pending, waiting) = child;
            waiting += 1;
        }
    }
    let mut changed = false;
    for id in 0..count {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
        let id = id as PlanNodeId;
        let parent = *at!(parents, id as usize);
        let is_plannable = |id| plannable(&plan.exprs, &at!(plan.nodes, id as usize).op);
        // A region is planned from its root, which no plannable join reads.
        let reachable = id == plan.root || parent != NONE;
        let root = is_plannable(id) && (parent == NONE || !is_plannable(parent));
        if reachable && root {
            changed |= plan_region(global, plan, estimates, id, parent)?;
        }
    }
    // Pruning next orders the nodes again, inputs first.
    Ok(changed)
}

/// Whether `op` is a join the planner can reorder: an inner join on
/// equalities between columns.
fn plannable(exprs: &Exprs, op: &DynOp<'_>) -> bool {
    let Some(join) = op.as_join() else { return false };
    let conjuncts = join.conjuncts();
    join.kind() == Kind::Inner
        && !conjuncts.is_empty()
        && conjuncts.iter().all(|&conjunct| equality(exprs.nodes(conjunct)).is_some())
}

/// Plans the region whose root is join `root`, and rebuilds it if planning
/// finds a plan. Returns whether it did.
fn plan_region(
    global: &Global,
    plan: &mut LogicalPlan<'_>,
    estimates: &[Estimate],
    root: PlanNodeId,
    parent: PlanNodeId,
) -> Result<bool, AllocError> {
    // Its joins, and its relations: the children of its joins that aren't
    // joins of it.
    let mut joins = SlowVec::new(global, crate::plan::PLAN_NODES_MAX)?;
    let mut leaves = SlowVec::new(global, crate::plan::PLAN_NODES_MAX)?;
    let mut pending = SlowVec::new(global, crate::plan::PLAN_NODES_MAX)?;
    pending.push(root)?;
    while let Some(id) = pending.pop() {
        joins.push(id)?;
        for &child in at!(plan.nodes, id as usize).children.iter() {
            let child_op = &at!(plan.nodes, child as usize).op;
            if plannable(&plan.exprs, child_op) {
                pending.push(child)?;
            } else {
                leaves.push(child)?;
            }
        }
    }
    if leaves.len() <= 2 {
        return Ok(false);
    }
    // Each column's relation, by the joins' lists of each child's columns.
    let mut relation_of = [NONE; PLAN_COLUMNS_MAX];
    for &id in joins.iter() {
        let node = at!(plan.nodes, id as usize);
        let Some(join) = node.op.as_join() else { failed!() };
        for (side, &child) in node.children.iter().enumerate() {
            let Some(relation) = leaves.iter().position(|&leaf| leaf == child) else { continue };
            for &column in join.all_columns(side) {
                #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
                let relation = relation as u32;
                *at_mut!(relation_of, column as usize) = relation;
            }
        }
    }
    // The equalities the region's joins require.
    let mut equalities = SlowVec::new(global, crate::expression::EXPR_NODES_MAX)?;
    let mut unplaced = false;
    for &id in joins.iter() {
        let Some(join) = at!(plan.nodes, id as usize).op.as_join() else { failed!() };
        for &conjunct in join.conjuncts() {
            let Some(columns) = equality(plan.exprs.nodes(conjunct)) else { failed!() };
            let relations = columns.map(|column| *at!(relation_of, column as usize));
            // A column of no relation, or both of one, isn't a join's.
            let [a, b] = relations;
            unplaced |= a == NONE || b == NONE || a == b;
            equalities.push(Equality { conjunct, columns, relations })?;
        }
    }
    if unplaced {
        return Ok(false);
    }
    let graph = graph(global, estimates, &leaves, &equalities)?;
    let tree = match super::plan(global, &graph) {
        Ok(Some(tree)) => tree,
        Ok(None) | Err(Error::Unsupported) => return Ok(false),
        Err(_) => return Err(AllocError),
    };
    let new_root = rebuild(global, plan, &joins, &leaves, &equalities, &relation_of, &tree)?;
    // The new root takes the old one's place: its first join's.
    match parent {
        NONE => plan.root = new_root,
        parent => {
            let children = &mut at_mut!(plan.nodes, parent as usize).children;
            for child in children.iter_mut().filter(|child| **child == root) {
                *child = new_root;
            }
        }
    }
    Ok(true)
}

/// The query graph of a region's relations, whose estimates are those of
/// `leaves`, joined by `equalities`.
fn graph(
    global: &Global,
    estimates: &[Estimate],
    leaves: &[PlanNodeId],
    equalities: &[Equality],
) -> Result<QueryGraph, AllocError> {
    let mut graph = QueryGraph::new(global)?;
    for &leaf in leaves {
        let estimate = at!(estimates, leaf as usize);
        // A filter lowers a relation's estimate, but not the most it can
        // have: its rows before filters.
        let rows = estimate.rows;
        graph.relations.push(Relation {
            rows: rows.estimated,
            unfiltered_rows: rows.max,
            max_rows: rows.max,
        })?;
    }
    // Each column's equivalence set: columns joined by equality, directly or
    // through others, hold the same values.
    let mut set_of = [NONE; PLAN_COLUMNS_MAX];
    for equality in equalities {
        let [a, b] = equality.columns.map(|column| *at!(set_of, column as usize));
        let set = match (a, b) {
            (NONE, NONE) => {
                #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
                let set = graph.equivalences.len() as u32;
                graph.equivalences.push(EquivalenceSet { domain: u64::MAX })?;
                set
            }
            (NONE, set) | (set, NONE) => set,
            (a, b) => {
                // Merges `b` into `a`.
                for set in set_of.iter_mut().filter(|set| **set == b) {
                    *set = a;
                }
                a
            }
        };
        for &column in &equality.columns {
            *at_mut!(set_of, column as usize) = set;
        }
    }
    // A set's domain: the rows of a relation the set's column is unique in,
    // before filters, or else the fewest rows of its relations.
    for equality in equalities {
        for (&column, &relation) in equality.columns.iter().zip(&equality.relations) {
            let estimate = at!(estimates, *at!(leaves, relation as usize) as usize);
            let rows = estimate.rows;
            let bound = if estimate.key(column) >= Key::Unique { rows.max } else { rows.estimated };
            let set = at_mut!(graph.equivalences, *at!(set_of, column as usize) as usize);
            set.domain = set.domain.min(bound.max(1));
        }
    }
    for equality in equalities {
        let (mut left, mut right) = (RelationSet::new(global)?, RelationSet::new(global)?);
        let ([a, b], [column, _]) = (equality.relations, equality.columns);
        left.insert(a)?;
        right.insert(b)?;
        let equivalence = *at!(set_of, column as usize);
        graph.edges.push(JoinEdge { left, right, kind: JoinKind::Inner, equivalence })?;
    }
    Ok(graph)
}

/// Rebuilds a region as the join tree `tree`, in the slots of its joins,
/// `joins`, and returns its root. Each join of the tree requires the
/// equalities its two sides bring together, and outputs all their
/// relations' columns, for pruning to trim.
fn rebuild(
    global: &Global,
    plan: &mut LogicalPlan<'_>,
    joins: &[PlanNodeId],
    leaves: &[PlanNodeId],
    equalities: &[Equality],
    relation_of: &[u32],
    tree: &JoinTree,
) -> Result<PlanNodeId, AllocError> {
    let nodes = &tree.nodes;
    check!(!nodes.is_empty() && nodes.len() < TREE_MAX);
    // Each tree node's plan node, and the relations under it, a bit each: a
    // region has at most `PLAN_NODES_MAX` relations.
    let mut built = [(NONE, 0_u64); TREE_MAX];
    let mut slots = joins.iter().copied();
    // The tree's nodes, each after its sides, from the root, which is last.
    let mut pending = [(0, false); 2 * TREE_MAX];
    #[expect(clippy::cast_possible_truncation, reason = "at most a tree's nodes")]
    let mut count = {
        *at_mut!(pending, 0) = (nodes.len() as u32 - 1, false);
        1
    };
    while count > 0 {
        count -= 1;
        let (i, sides_built) = *at!(pending, count);
        let node = at!(nodes, i as usize);
        let (left, right) = match node.kind {
            JoinNode::Relation(relation) => {
                *at_mut!(built, i as usize) = (*at!(leaves, relation as usize), 1 << relation);
                continue;
            }
            JoinNode::Join { left, right } => (left, right),
        };
        if !sides_built {
            for next in [(i, true), (left, false), (right, false)] {
                *at_mut!(pending, count) = next;
                count += 1;
            }
            continue;
        }
        let ((left, under_left), (right, under_right)) =
            (*at!(built, left as usize), *at!(built, right as usize));
        // The equalities whose relations it joins, one on each side: only
        // the lowest join that has both splits them so.
        let mut conjuncts = SlowVec::fixed(global, equalities.len())?;
        for equality in equalities {
            let [a, b] = equality.relations.map(|relation| 1_u64 << relation);
            if (a & under_left != 0 && b & under_right != 0)
                || (a & under_right != 0 && b & under_left != 0)
            {
                conjuncts.extend_within(core::iter::once(equality.conjunct));
            }
        }
        let columns = [
            columns_of(global, relation_of, under_left)?,
            columns_of(global, relation_of, under_right)?,
        ];
        let op = JoinOp::new(Kind::Inner, conjuncts, columns);
        let Some(slot) = slots.next() else { failed!() };
        let plan_node = at_mut!(plan.nodes, slot as usize);
        plan_node.op = DynOp::new(global, op)?;
        plan_node.children = crate::plan::Children::from_slice(&[left, right])?;
        *at_mut!(built, i as usize) = (slot, under_left | under_right);
    }
    Ok(at!(built, nodes.len() - 1).0)
}

/// The columns of the relations of `under`, a bit each.
fn columns_of(
    global: &Global,
    relation_of: &[u32],
    under: u64,
) -> Result<SlowVec<ColumnId>, AllocError> {
    let of = |&relation: &u32| relation != NONE && under >> relation & 1 != 0;
    let mut columns = SlowVec::fixed(global, relation_of.iter().filter(|r| of(r)).count())?;
    #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_COLUMNS_MAX`")]
    columns.extend_within(
        (0..relation_of.len()).filter(|&c| of(at!(relation_of, c))).map(|c| c as ColumnId),
    );
    Ok(columns)
}
