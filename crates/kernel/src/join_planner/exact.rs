//! Exact search, by `DPhyp` (Moerkotte and Neumann, "Dynamic Programming
//! Strikes Back", 2008). It joins each connected set of relations to each
//! connected set beside it, smallest first, and keeps the cheapest valid
//! tree of each set, so the tree of all of them is the optimum. It first
//! counts what it would visit, and plans only if that fits `VISITS_MAX`.

use core::iter::repeat_n;

use super::estimate::Estimator;
use super::graph::{LIST_MAX, QueryGraph};
use super::tree::{JoinNode, JoinTree};
use crate::error::Error;
use crate::global::Global;
use crate::slow_vec::SlowVec;

/// The most units it plans: a set of them is the bits of a `u64`.
const UNITS_MAX: usize = 64;

/// The most connected sets of relations, and pairs of them, it visits.
const VISITS_MAX: usize = 1 << 11;

/// The most relations of a graph that it plans without counting, as even a
/// clique's 2^n sets and 3^n / 2 pairs are few.
const SMALL_MAX: usize = 10;

/// An edge between sets of units: relations, or a window's subtrees.
#[derive(Clone, Copy)]
pub(super) struct Link {
    /// The units of its left side, then of its right, as bits.
    pub(super) sides: [u64; 2],
    /// Its index in the graph.
    pub(super) edge: u32,
}

/// The cheapest tree found for a set of units, in a hash table.
#[derive(Clone, Copy)]
struct Plan {
    /// The units, or 0 for an empty slot.
    set: u64,
    /// Its cost.
    cost: f64,
    /// Its rows.
    rows: f64,
    /// The units on its left side.
    split: u64,
}

/// An empty slot.
const EMPTY: Plan = Plan { set: 0, cost: f64::INFINITY, rows: 0.0, split: 0 };

/// Whether the operator can run a join of two sets of units.
type CanRun<'r> = &'r mut dyn FnMut(u64, u64) -> bool;

/// The search's state.
#[repr(C)]
struct Exact<'g, 'e, 'l, 'r> {
    /// The graph being planned.
    graph: &'g QueryGraph,
    /// Estimates the rows of the joins it tries.
    estimator: &'e mut Estimator<'g>,
    /// The edges between the units.
    links: &'l [Link],
    /// Which joins of units can run.
    can_run: CanRun<'r>,
    /// Whether an edge has a side of more than one unit.
    hyper: bool,
    /// The units that are one relation of at most one row, which may be
    /// joined to anything.
    single: u64,
    /// All the units.
    full: u64,
    /// While counting, how many more visits there may be.
    visits: usize,
    /// The cheapest tree of each connected set, by `find`.
    plans: SlowVec<Plan>,
    /// For each unit, those an edge of one unit a side joins it to.
    adjacent: [u64; UNITS_MAX],
}

/// The optimum tree of `graph`, if it has at most `UNITS_MAX` relations
/// and exact search would visit fewer than `VISITS_MAX` sets and pairs, and
/// it has a valid tree.
pub(super) fn plan<'g>(
    global: &Global,
    graph: &'g QueryGraph,
    estimator: &mut Estimator<'g>,
) -> Result<Option<JoinTree>, Error> {
    let count = graph.relations.len();
    if count > UNITS_MAX {
        return Ok(None);
    }
    // Every relation is below 64, so in the first word.
    let mut links = SlowVec::new(global, LIST_MAX)?;
    #[expect(clippy::cast_possible_truncation, reason = "at most `LIST_MAX`")]
    for (edge, found) in (0..graph.edges.len() as u32).zip(graph.edges.iter()) {
        links.push(Link { sides: [found.left.word(0), found.right.word(0)], edge })?;
    }
    let mut rows = [0.0; UNITS_MAX];
    for (rows, relation) in rows.iter_mut().zip(graph.relations.iter()) {
        #[expect(clippy::cast_precision_loss, reason = "an estimate")]
        let estimate = relation.rows as f64;
        *rows = estimate;
    }
    let mut can_run = |left, right| {
        graph.executable.is_none_or(|executable| executable(graph, &[left], &[right]))
    };
    let single = (0..)
        .zip(graph.relations.iter())
        .fold(0, |single, (i, relation)| single | u64::from(relation.max_rows <= 1) << i);
    let mut exact = Exact::new(global, graph, estimator, &links, &mut can_run, count, single);
    // A graph of n relations has fewer than 2^n connected sets. Each set
    // the search visits but the relations is made by a pair it counted.
    let small = (count <= SMALL_MAX).then_some(1 << count);
    let Some(sets) = small.or_else(|| Some(exact.count(count)? + count)) else { return Ok(None) };
    exact.table(global, at!(rows, ..count), sets)?;
    // It visits no more than were counted, or than `SMALL_MAX` relations have.
    exact.enumerate(count, true);
    exact.tree(global, count)
}

impl<'g, 'e, 'l, 'r> Exact<'g, 'e, 'l, 'r> {
    /// A search of `units` units, joined by `links`, of which those in
    /// `single` are one relation of at most one row.
    fn new(
        global: &Global,
        graph: &'g QueryGraph,
        estimator: &'e mut Estimator<'g>,
        links: &'l [Link],
        can_run: CanRun<'r>,
        units: usize,
        single: u64,
    ) -> Exact<'g, 'e, 'l, 'r> {
        let (mut adjacent, mut hyper) = ([0; UNITS_MAX], false);
        for link in links {
            let [left, right] = link.sides;
            if left.is_power_of_two() && right.is_power_of_two() {
                *at_mut!(adjacent, left.trailing_zeros() as usize) |= right;
                *at_mut!(adjacent, right.trailing_zeros() as usize) |= left;
            } else {
                hyper = true;
            }
        }
        let full = u64::MAX >> (UNITS_MAX - units);
        let plans = SlowVec::none(global);
        Exact { graph, estimator, links, can_run, hyper, single, full, visits: 0, plans, adjacent }
    }

    /// Makes room for `sets` sets, and a tree of each unit of `rows` rows.
    fn table(&mut self, global: &Global, rows: &[f64], sets: usize) -> Result<(), Error> {
        // Half full at most, so that a set is found in a few probes.
        let len = (2 * sets).next_power_of_two();
        self.plans = SlowVec::fixed(global, len)?;
        self.plans.extend_within(repeat_n(EMPTY, len));
        for (i, &rows) in rows.iter().enumerate() {
            let (set, slot) = (1 << i, self.find(1 << i));
            *at_mut!(self.plans, slot) = Plan { set, cost: 0.0, rows, split: 0 };
        }
        Ok(())
    }

    /// The cheapest tree of all `units` units, if they have one.
    fn tree(&self, global: &Global, units: usize) -> Result<Option<JoinTree>, Error> {
        if self.get(self.full).is_none() {
            return Ok(None);
        }
        let mut tree = JoinTree::new(global, units)?;
        self.build(&mut tree, self.full);
        Ok(Some(tree))
    }

    /// How many connected sets of the first `relations` relations, and pairs
    /// of them, the search visits, if fewer than `VISITS_MAX`. With
    /// hyperedges, it counts some sets that aren't connected and pairs that
    /// can't be joined.
    #[inline(never)]
    fn count(&mut self, relations: usize) -> Option<usize> {
        self.visits = VISITS_MAX;
        self.enumerate(relations, false);
        (self.visits > 0).then(|| VISITS_MAX - self.visits)
    }

    /// Joins each pair of connected sets of the first `relations` relations
    /// if `solve`, or counts them.
    fn enumerate(&mut self, relations: usize, solve: bool) {
        for relation in (0..relations).rev() {
            let set = 1 << relation;
            let plan = if solve { self.get(set).unwrap_or(EMPTY) } else { EMPTY };
            self.complements(plan, set, solve);
            self.subgraphs(set, set | (set - 1), solve);
        }
    }

    /// The units an edge joins to all of `set`, outside `set` and
    /// `excluded`: for a side of more than one unit, its lowest.
    #[inline(never)]
    fn neighbors(&self, set: u64, excluded: u64) -> u64 {
        let outside = self.full & !(set | excluded);
        // A unit of one row is beside every unit, and every unit beside it.
        let single = if set & self.single != 0 { outside } else { self.single };
        let mut near = (self.adjacent(set) | single) & outside;
        for link in self.links.iter().filter(|_| self.hyper) {
            let [left, right] = link.sides;
            if left & !set == 0 && right & !outside == 0 {
                near |= right.isolate_lowest_one();
            }
            if right & !set == 0 && left & !outside == 0 {
                near |= left.isolate_lowest_one();
            }
        }
        near
    }

    /// The units an edge of one unit a side joins to some of `set`.
    fn adjacent(&self, set: u64) -> u64 {
        let (mut near, mut rest) = (0, set);
        while rest != 0 {
            near |= at!(self.adjacent, rest.trailing_zeros() as usize);
            rest &= rest - 1;
        }
        near
    }

    /// Joins each connected set that grows from `set` outside `excluded` to
    /// the sets beside it, or counts those sets and pairs.
    fn subgraphs(&mut self, set: u64, excluded: u64, solve: bool) {
        let near = self.neighbors(set, excluded);
        for grow in [false, true] {
            let mut more = 0_u64;
            loop {
                more = more.wrapping_sub(near) & near;
                if more == 0 || !solve && self.visits == 0 {
                    break;
                }
                if !solve && !grow {
                    self.visits -= 1;
                }
                if grow {
                    self.subgraphs(set | more, excluded | near, solve);
                } else if !solve {
                    self.complements(EMPTY, set | more, false);
                } else if let Some(plan) = self.get(set | more) {
                    self.complements(plan, set | more, true);
                }
            }
        }
    }

    /// Joins the connected set `left`, of plan `plan`, to each connected set
    /// beside it above its lowest unit, so that each pair is joined once, or
    /// counts them.
    fn complements(&mut self, plan: Plan, left: u64, solve: bool) {
        let excluded = left | (left.isolate_lowest_one() - 1);
        let near = self.neighbors(left, excluded);
        let mut rest = near;
        while rest != 0 {
            let unit = 1 << rest.ilog2();
            rest ^= unit;
            self.grow(plan, unit, excluded | near & (unit | (unit - 1)), solve);
        }
    }

    /// Joins the set of plan `left` to `right` and to each connected set
    /// that grows from `right` outside `excluded`, or counts them.
    fn grow(&mut self, left: Plan, right: u64, excluded: u64, solve: bool) {
        if !solve {
            if self.visits == 0 {
                return;
            }
            self.visits -= 1;
        } else if let Some(plan) = self.get(right) {
            self.pair(left, plan);
        }
        let near = self.neighbors(right, excluded);
        let mut more = 0_u64;
        loop {
            more = more.wrapping_sub(near) & near;
            if more == 0 {
                break;
            }
            self.grow(left, right | more, excluded | near, solve);
        }
    }

    /// The slot of `set` in `plans`, or the empty slot it would take.
    #[inline(never)]
    fn find(&self, set: u64) -> usize {
        let mask = self.plans.len() - 1;
        let hash = (set.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as u32;
        let mut i = hash as usize & mask;
        for _ in 0..=mask {
            let found = at!(self.plans, i).set;
            if found == 0 || found == set {
                return i;
            }
            i = (i + 1) & mask;
        }
        // The table has room for every set it's given.
        failed!()
    }

    /// Whether `set` is one unit of at most one row, which may be joined to
    /// anything.
    fn lone(&self, set: u64) -> bool {
        set.is_power_of_two() && set & self.single != 0
    }

    /// The cheapest tree of `set`, if it has one.
    fn get(&self, set: u64) -> Option<Plan> {
        Some(*at!(self.plans, self.find(set))).filter(|plan| plan.set == set)
    }

    /// Makes the join of plans `l` and `r` the cheapest tree of both, if it
    /// can be joined and is cheaper.
    #[inline(never)]
    fn pair(&mut self, l: Plan, r: Plan) {
        let (left, right) = (l.set, r.set);
        let set = left | right;
        let i = self.find(set);
        let found = *at!(self.plans, i);
        // A set's rows are estimated from its first valid split.
        let fresh = found.set != set;
        // Without hyperedges, every edge between the sides applies.
        let lone = self.lone(left) || self.lone(right);
        let valid = if fresh || self.hyper {
            self.joins([set, left, right], fresh.then_some([l.rows, r.rows]), lone)
        } else {
            self.adjacent(left) & right != 0 || lone
        };
        if !valid || !(self.can_run)(left, right) {
            return;
        }
        let rows = if fresh { self.estimator.finish() } else { found.rows };
        let cost = l.cost + r.cost + rows.max(1.0);
        if fresh || cost < found.cost {
            *at_mut!(self.plans, i) = Plan { set, cost, rows, split: left };
        }
    }

    /// Whether units `left` and `right`, which together are `set`, can be
    /// joined: a link joins them, and every one that they bring together
    /// applies, as stage 4 of `docs/join-planner.md` has it. If given `rows`,
    /// it starts the estimator on the join of sides of those rows, and
    /// applies the links.
    fn joins(&mut self, [set, left, right]: [u64; 3], rows: Option<[f64; 2]>, lone: bool) -> bool {
        if let Some([left, right]) = rows {
            self.estimator.start(left, right);
        }
        let mut any = lone;
        for link in self.links {
            let [l, r] = link.sides;
            let both = l | r;
            if both & !set != 0 || both & !left == 0 || both & !right == 0 {
                continue;
            }
            let forward = l & !left == 0 && r & !right == 0;
            if !(forward || l & !right == 0 && r & !left == 0) {
                return false;
            }
            if rows.is_some() {
                self.estimator.apply(at!(self.graph.edges, link.edge as usize), forward);
            }
            any = true;
        }
        any
    }

    /// Adds the cheapest tree of `set` to `tree`, and returns its root.
    #[inline(never)]
    fn build(&self, tree: &mut JoinTree, set: u64) -> u32 {
        let plan = *at!(self.plans, self.find(set));
        if set.is_power_of_two() {
            return tree.add(JoinNode::Relation(set.trailing_zeros()), plan.rows);
        }
        let left = self.build(tree, plan.split);
        let right = self.build(tree, set ^ plan.split);
        tree.add(JoinNode::Join { left, right }, plan.rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::global::Global;
    use crate::join_planner::{EquivalenceSet, JoinEdge, JoinKind, Relation, RelationSet};

    /// A graph of relations of `rows`, each a pair of rows and the most rows,
    /// and inner joins between pairs of relations on domains of `edges`.
    fn graph(rows: &[(u64, u64)], edges: &[(u32, u32, u64)]) -> QueryGraph {
        let mut graph = QueryGraph::new(Global::current()).unwrap();
        for &(rows, max_rows) in rows {
            graph.relations.push(Relation { rows, unfiltered_rows: rows, max_rows }).unwrap();
        }
        for (equivalence, &(left, right, domain)) in (0..).zip(edges) {
            graph.equivalences.push(EquivalenceSet { domain }).unwrap();
            let (mut left_set, mut right_set) = (
                RelationSet::new(Global::current()).unwrap(),
                RelationSet::new(Global::current()).unwrap(),
            );
            left_set.insert(left).unwrap();
            right_set.insert(right).unwrap();
            let (left, right, kind) = (left_set, right_set, JoinKind::Inner);
            graph.edges.push(JoinEdge { left, right, kind, equivalence }).unwrap();
        }
        graph
    }

    #[test]
    fn one_row_relations_join_anything() {
        // Two joined relations, and one of a row with no edge to them.
        let mut graph = graph(&[(100, 100), (10, 10), (1, 1)], &[(0, 1, 10)]);
        let mut estimator = Estimator::new(Global::current(), &graph).unwrap();
        let tree = plan(Global::current(), &graph, &mut estimator).unwrap();
        assert_eq!(tree.map(|tree| tree.nodes.len()), Some(5));
        // Without its bound of a row, it needs a cross product.
        at_mut!(graph.relations, 2).max_rows = 2;
        let mut estimator = Estimator::new(Global::current(), &graph).unwrap();
        assert!(plan(Global::current(), &graph, &mut estimator).unwrap().is_none());
    }
}
