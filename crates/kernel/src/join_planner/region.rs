//! The planner's first and last stages: finding each region of joins in a
//! plan, and rebuilding it in the order planned.

use crate::allocator::AllocError;
use crate::containers::{SmallVec, SpillVec};
use crate::env::Env;
use crate::error::Error;
use crate::estimate::Estimate;
use crate::expression::{Expr, Exprs};
use crate::join::{JoinKind as Kind, JoinOp, equality};
use crate::plan::{ColumnId, DynOp, LogicalPlan, PLAN_COLUMNS_MAX, PLAN_NODES_MAX, PlanNodeId};
use crate::scannable::Key;

use super::{EquivalenceSet, JoinEdge, JoinNode, JoinTree, QueryGraph, Relation, RelationSet};

/// No relation, or no node.
const NONE: u32 = u32::MAX;

/// The most relations a region the planner reorders has, a bit each of a
/// word. A bigger one keeps the order it's written in.
const RELATIONS_MAX: usize = 64;

/// The most nodes a region's join tree has: a leaf for each relation, and a
/// join for each but one.
const TREE_MAX: usize = 2 * RELATIONS_MAX;

/// How many nodes' parents finding regions holds in place, as most plans
/// have no more.
const NODES_INLINE: usize = 64;

/// An equality between two columns, of different relations, that a join in
/// a region requires.
#[derive(Clone, Copy)]
struct Equality {
    /// The conjunct that requires it.
    conjunct: Expr,
    /// Its columns.
    columns: [ColumnId; 2],
    /// The relation each of `columns` is of.
    relations: [u32; 2],
}

/// Plans each region of joins in `plan`, given each node's estimate, and
/// rebuilds those it finds a plan for in their new order. Returns whether
/// any changed, which leaves the plan to prune again.
pub(crate) fn plan_regions(
    env: &Env,
    plan: &mut LogicalPlan<'_>,
    estimates: &[Estimate],
) -> Result<bool, AllocError> {
    let count = plan.nodes.len();
    // Each node's parent, `NONE` for the root. Every node is reachable, as
    // the passes before remove those that aren't.
    let mut parents = SpillVec::<u32, NODES_INLINE>::filled(env, PLAN_NODES_MAX, count, NONE)?;
    for (id, node) in (0..).zip(plan.nodes.iter()) {
        for &child in node.children.iter() {
            *at_mut!(parents, child as usize) = id;
        }
    }
    let mut changed = false;
    for id in 0..count {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
        let id = id as PlanNodeId;
        let parent = *at!(parents, id as usize);
        let is_plannable = |id| plannable(&plan.exprs, &at!(plan.nodes, id as usize).op);
        // A region is planned from its root, which no plannable join reads.
        if is_plannable(id) && (parent == NONE || !is_plannable(parent)) {
            changed |= plan_region(env, plan, estimates, id, parent)?;
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
    env: &Env,
    plan: &mut LogicalPlan<'_>,
    estimates: &[Estimate],
    root: PlanNodeId,
    parent: PlanNodeId,
) -> Result<bool, AllocError> {
    // Its joins, and its relations: the children of its joins that aren't
    // joins of it.
    let mut joins = SmallVec::new(env, crate::plan::PLAN_NODES_MAX)?;
    let mut leaves = SmallVec::new(env, crate::plan::PLAN_NODES_MAX)?;
    let mut pending = SmallVec::new(env, crate::plan::PLAN_NODES_MAX)?;
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
    if leaves.len() <= 2 || leaves.len() > RELATIONS_MAX {
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
    let mut equalities = SmallVec::new(env, crate::expression::EXPR_NODES_MAX)?;
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
    let graph = graph(env, estimates, &leaves, &equalities)?;
    let tree = match super::plan(env, &graph) {
        Ok(Some(tree)) => tree,
        Ok(None) | Err(Error::Unsupported) => return Ok(false),
        Err(_) => return Err(AllocError),
    };
    let new_root = rebuild(env, plan, &joins, &leaves, &equalities, &relation_of, &tree)?;
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
    env: &Env,
    estimates: &[Estimate],
    leaves: &[PlanNodeId],
    equalities: &[Equality],
) -> Result<QueryGraph, AllocError> {
    let mut graph = QueryGraph::new(env)?;
    for &leaf in leaves {
        let estimate = at!(estimates, leaf as usize);
        // A filter lowers a relation's estimate, but not the most it can
        // have: its rows before filters.
        let rows = estimate.rows;
        graph.relations.push(Relation { rows: rows.estimated, max_rows: rows.max })?;
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
    // A set's domain: the fewest values any of its columns can take. A
    // column of known bounds takes at most those within them, and at most
    // its relation's rows before filters, as DuckDB has it; a unique column
    // takes those rows; any other column is assumed to take as many values
    // as its relation has rows.
    for equality in equalities {
        for (&column, &relation) in equality.columns.iter().zip(&equality.relations) {
            let estimate = at!(estimates, *at!(leaves, relation as usize) as usize);
            let rows = estimate.rows;
            let span = estimate.span(column);
            let bound = match span {
                Some(span) => span.min(rows.max),
                None if estimate.key(column) >= Key::Unique => rows.max,
                None => rows.estimated,
            };
            let set = at_mut!(graph.equivalences, *at!(set_of, column as usize) as usize);
            set.domain = set.domain.min(bound.max(1));
        }
    }
    for equality in equalities {
        let (mut left, mut right) = (RelationSet::new(env)?, RelationSet::new(env)?);
        let ([a, b], [column, _]) = (equality.relations, equality.columns);
        left.insert(a)?;
        right.insert(b)?;
        let equivalence = *at!(set_of, column as usize);
        graph.edges.push(JoinEdge { left, right, equivalence })?;
    }
    Ok(graph)
}

/// Rebuilds a region as the join tree `tree`, in the slots of its joins,
/// `joins`, and returns its root. Each join of the tree requires the
/// equalities its two sides bring together, and outputs all their
/// relations' columns, for pruning to trim.
fn rebuild(
    env: &Env,
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
    // region has at most `RELATIONS_MAX` relations.
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
        let mut conjuncts = SmallVec::fixed(env, equalities.len())?;
        for equality in equalities {
            let [a, b] = equality.relations.map(|relation| 1_u64 << relation);
            if (a & under_left != 0 && b & under_right != 0)
                || (a & under_right != 0 && b & under_left != 0)
            {
                conjuncts.extend_within(core::iter::once(equality.conjunct));
            }
        }
        let columns =
            [columns_of(env, relation_of, under_left)?, columns_of(env, relation_of, under_right)?];
        let op = JoinOp::new(env, Kind::Inner, conjuncts, columns)?;
        let Some(slot) = slots.next() else { failed!() };
        let plan_node = at_mut!(plan.nodes, slot as usize);
        plan_node.op = DynOp::new(env, op)?;
        plan_node.children = crate::plan::Children::from_slice(&[left, right])?;
        *at_mut!(built, i as usize) = (slot, under_left | under_right);
    }
    Ok(at!(built, nodes.len() - 1).0)
}

/// The columns of the relations of `under`, a bit each.
fn columns_of(
    env: &Env,
    relation_of: &[u32],
    under: u64,
) -> Result<SmallVec<ColumnId>, AllocError> {
    let of = |&relation: &u32| relation != NONE && under >> relation & 1 != 0;
    let mut columns = SmallVec::fixed(env, relation_of.iter().filter(|r| of(r)).count())?;
    #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_COLUMNS_MAX`")]
    columns.extend_within(
        (0..relation_of.len()).filter(|&c| of(at!(relation_of, c))).map(|c| c as ColumnId),
    );
    Ok(columns)
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::String;

    use super::*;
    use crate::column::{Bounds, DataType, Forms};
    use crate::context::Context;
    use crate::expression::{ExprBuilder, ExprOp};
    use crate::plan::{ScanColumn, ScanOp, conjuncts_of};
    use crate::row_batch::RowBatch;
    use crate::scannable::{DynScannable, Scannable, Summary};

    /// A table of `rows` rows and two columns, which it says only through
    /// `summary`: each column's values lie in its `bounds`, if known.
    struct Summarized {
        rows: i64,
        bounds: [Option<Bounds>; 2],
    }

    impl Scannable for Summarized {
        type State<'s>
            = ()
        where
            Self: 's;
        type Loader = ();

        fn column_count(&self) -> u32 {
            2
        }

        fn column_name(&self, _: u32) -> &'static str {
            "a"
        }

        fn column_type(&self, _: u32) -> DataType {
            DataType::Int64
        }

        fn open(&self, _: &mut Context, _: &[(u32, Forms)]) -> Result<(), Error> {
            Ok(())
        }

        fn next(&self, _: &mut Context, (): &mut (), _: &mut RowBatch) -> Result<bool, Error> {
            Ok(false)
        }

        fn summary(&self, column: Option<u32>) -> Option<Summary> {
            let bounds = column.and_then(|column| *at!(self.bounds, column as usize));
            Some(Summary { rows: self.rows, bounds })
        }
    }

    /// The order the planner joins `tables` in, joined along a chain, each
    /// table's second column to the next's first, as nested tables by their
    /// rows.
    fn planned(tables: &[DynScannable<'_>]) -> String {
        let env = Env::current();
        let mut plan = LogicalPlan::new(env).unwrap();
        let mut root: Option<(PlanNodeId, SmallVec<ColumnId>)> = None;
        let mut previous = 0;
        for table in tables {
            let a = plan.add_column("a", DataType::Int64, false).unwrap();
            let b = plan.add_column("b", DataType::Int64, false).unwrap();
            let scanned = [a, b].into_iter().zip(0..2).map(|(binding, column)| ScanColumn {
                column,
                binding,
                forms: Forms::FLAT,
            });
            let columns = SmallVec::fixed_from(env, scanned).unwrap();
            let scan = DynOp::new(env, ScanOp::new(table, columns)).unwrap();
            let scan = plan.add_node(scan, &[]).unwrap();
            plan.output.push(a).unwrap();
            let Some((left, left_columns)) = root.take() else {
                root = Some((scan, SmallVec::fixed_from(env, [a.id, b.id].into_iter()).unwrap()));
                previous = b.id;
                continue;
            };
            let mut nodes = ExprBuilder::new(env).unwrap();
            let (x, y) = (nodes.column(previous).unwrap(), nodes.column(a.id).unwrap());
            nodes.node(ExprOp::Eq, x, y).unwrap();
            let condition = plan.exprs.add(nodes.nodes()).unwrap();
            let conjuncts = conjuncts_of(env, &plan.exprs, condition).unwrap();
            let mut all = SmallVec::new(env, PLAN_COLUMNS_MAX).unwrap();
            for &column in left_columns.iter().chain(&[a.id, b.id]) {
                all.push(column).unwrap();
            }
            let right = SmallVec::fixed_from(env, [a.id, b.id].into_iter()).unwrap();
            let join = JoinOp::new(env, Kind::Inner, conjuncts, [left_columns, right]).unwrap();
            let join = plan.add_node(DynOp::new(env, join).unwrap(), &[left, scan]).unwrap();
            root = Some((join, all));
            previous = b.id;
        }
        crate::optimize::optimize(env, &mut plan).unwrap();
        shape(&plan, plan.root)
    }

    /// The joins under `node` as nested tables, each by its rows.
    fn shape(plan: &LogicalPlan<'_>, node: PlanNodeId) -> String {
        let node = at!(plan.nodes, node as usize);
        if node.children.is_empty() {
            return format!("{}", node.op.summary(None).unwrap().rows);
        }
        let mut sides = node.children.iter().map(|&child| shape(plan, child));
        let (Some(a), Some(b)) = (sides.next(), sides.next()) else { failed!() };
        // In either order: each join chooses its build side later.
        if a < b { format!("({a} {b})") } else { format!("({b} {a})") }
    }

    #[test]
    fn bounds_a_domain_by_the_values_its_columns_can_take() {
        // A chain 1000 - 500 - 2000: the first join's key has 5 values, so
        // joining the 1000 and the 500 first makes 100,000 rows.
        let table =
            |rows, bounds| DynScannable::new(Env::current(), Summarized { rows, bounds }).unwrap();
        let few = Some(Bounds { min: 1, max: 5 });
        let order = |known: bool| {
            let few = few.filter(|_| known);
            planned(&[table(1000, [None, few]), table(500, [few, None]), table(2000, [None, None])])
        };
        // Without bounds, a key is taken to have as many values as the
        // smaller side has rows.
        assert_eq!(order(false), "((1000 500) 2000)");
        assert_eq!(order(true), "((2000 500) 1000)");
    }
}
