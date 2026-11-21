//! Join ordering by greedy operator ordering (GOO, Fegaras 1998): while
//! more than one tree is left, join the two trees whose join makes the fewest
//! rows.
//!
//! Each join reads a window of the graph's edges, so that dense graphs stay
//! bounded by `WORK_MAX`. Once that work is spent, each join takes the first
//! pair of trees that can be joined.
//!
//! Most functions are `#[inline(never)]`: inlined into one another, they were
//! larger.

use super::{Estimator, JoinTree, QueryGraph, RelationSet, as_written};
use crate::allocator::Allocator;
use crate::error::Error;
use crate::slow_vec::SlowVec;

/// No node, tree or edge.
const NONE: u32 = u32::MAX;

/// How many pairs of trees each join prices by all the edges between them.
/// They are the pairs whose single edges make the fewest rows. A pair's rows
/// can be far fewer than any one of its edges says, so this has to be more
/// than a few.
const TOP: usize = 16;

/// The most work planning does before it takes the first pair it can: edges
/// and hyperedges read. It's about 0.8 s at most.
const WORK_MAX: usize = 1 << 27;

/// A node of the plan. The graph's relations come first, then the joins.
#[derive(Clone, Copy)]
struct Node {
    /// The left child, or `NONE` for a relation.
    left: u32,
    /// The right child, or `NONE` for a relation.
    right: u32,
    /// The rows it makes.
    rows: f64,
    /// For a relation, the root of the tree it's in.
    place: u32,
}

/// What the planner reads of the graph's edge of the same index.
#[derive(Clone, Copy)]
struct Edge {
    /// The first relation of its left side, then of its right.
    first: [u32; 2],
    /// Whether its left side, then its right, has more than one relation.
    more: [bool; 2],
    /// For a hyperedge in the list that starts at `Planner::hypers`, the next
    /// one, or `NONE`.
    next: u32,
    /// How many nodes there were when joining its two trees was found to
    /// leave a hyperedge that could never apply. While both trees are older
    /// than that, they still can't be joined.
    dead: u32,
}

/// The plan as it is built, and the memory for building it.
struct Planner<'a> {
    /// The graph being planned.
    graph: &'a QueryGraph,
    /// Estimates the rows of the joins it considers.
    estimator: Estimator<'a>,
    /// The work left: see `WORK_MAX`.
    work: usize,
    /// How many words a set of relations takes.
    words: usize,
    /// The plan's nodes.
    nodes: SlowVec<Node>,
    /// Each node's relations, `words` words a node.
    sets: SlowVec<u64>,
    /// The graph's edges.
    edges: SlowVec<Edge>,
    /// The first of the hyperedges, linked through `Edge::next`, or `NONE`.
    /// A hyperedge with the same sides as the one linked before it is left
    /// out, as it can't change whether a join is valid.
    hypers: u32,
}

/// Plans a graph of three relations or more: GOO, as above.
pub fn plan(allocator: &dyn Allocator, graph: &QueryGraph) -> Result<JoinTree, Error> {
    let mut planner = Planner::new(allocator, graph)?;
    if !planner.greedy() {
        return as_written(allocator, graph);
    }
    planner.tree(allocator)
}

/// The first relation of `set`, and whether it has more than one.
#[inline(never)]
fn first(set: &RelationSet) -> (u32, bool) {
    let (mut first, mut count) = (NONE, 0);
    for i in 0..set.words() {
        let word = set.word(i);
        if first == NONE && word != 0 {
            #[expect(clippy::cast_possible_truncation, reason = "at most `LIST_MAX`")]
            let relation = i as u32 * 64 + word.trailing_zeros();
            first = relation;
        }
        count += word.count_ones();
    }
    (first, count > 1)
}

/// Whether `side` has any relation of `set`, and whether all its relations
/// are in `set`.
fn meets(side: &RelationSet, set: &[u64]) -> (bool, bool) {
    let (mut some, mut all) = (false, true);
    for (i, set) in set.iter().enumerate() {
        let side = side.word(i);
        some |= side & set != 0;
        all &= side & !set == 0;
    }
    (some, all)
}

impl<'a> Planner<'a> {
    fn new(allocator: &dyn Allocator, graph: &'a QueryGraph) -> Result<Planner<'a>, Error> {
        let count = graph.relations.len();
        let words = count.div_ceil(64);
        #[expect(clippy::cast_possible_truncation, reason = "at most `2 * LIST_MAX`")]
        let nodes = (0..2 * count as u32 - 1).map(|node| {
            #[expect(clippy::cast_precision_loss, reason = "an estimate")]
            let rows = graph.relations.get(node as usize).map_or(0.0, |found| found.rows as f64);
            Node { left: NONE, right: NONE, rows, place: node }
        });
        let estimator = Estimator::new(allocator, graph)?;
        let nodes = SlowVec::fixed_from(allocator, nodes)?;
        let sets = SlowVec::zeroed(allocator, (2 * count - 1) * words)?;
        let edges = SlowVec::fixed(allocator, graph.edges.len())?;
        let (work, hypers) = (WORK_MAX, NONE);
        let mut planner = Planner { graph, estimator, work, words, nodes, sets, edges, hypers };
        for relation in 0..count {
            *at_mut!(planner.sets, relation * words + relation / 64) |= 1 << (relation % 64);
        }
        // Each hyperedge is linked in front of those before it, unless it has
        // the same sides as the last one linked.
        #[expect(clippy::cast_possible_truncation, reason = "at most `LIST_MAX`")]
        for (e, edge) in (0..graph.edges.len() as u32).zip(graph.edges.iter()) {
            let ((left, more_left), (right, more_right)) = (first(&edge.left), first(&edge.right));
            let (first, more, next) = ([left, right], [more_left, more_right], planner.hypers);
            let last = graph.edges.get(planner.hypers as usize);
            let same = |a: &RelationSet, b: &RelationSet| a.first == b.first && *a.rest == *b.rest;
            if (more_left || more_right)
                && !last.is_some_and(|last| {
                    same(&last.left, &edge.left) && same(&last.right, &edge.right)
                })
            {
                planner.hypers = e;
            }
            planner.edges.push(Edge { first, more, next, dead: 0 })?;
        }
        Ok(planner)
    }

    /// Sets the place of each relation of node `node` to `value`.
    #[inline(never)]
    fn mark(&mut self, node: u32, value: u32) {
        for i in 0..self.words {
            let mut word = *at!(self.sets, node as usize * self.words + i);
            while word != 0 {
                at_mut!(self.nodes, i * 64 + word.trailing_zeros() as usize).place = value;
                word &= word - 1;
            }
        }
    }

    /// Side `right` of edge `e`: 0 for its left side, 1 for its right.
    #[inline(never)]
    fn side(&self, e: usize, right: usize) -> &RelationSet {
        let edge = at!(self.graph.edges, e);
        if right == 0 { &edge.left } else { &edge.right }
    }

    /// Whether side `right` of edge `e` has any relation of `set`, and
    /// whether all its relations are in `set`. A side of one relation is
    /// answered from that relation alone.
    fn meets(&self, e: usize, right: usize, set: &[u64]) -> (bool, bool) {
        let edge = at!(self.edges, e);
        if *at!(edge.more, right) {
            return meets(self.side(e, right), set);
        }
        let relation = *at!(edge.first, right) as usize;
        let has = at!(set, relation / 64) >> (relation % 64) & 1 != 0;
        (has, has)
    }

    /// The relations of node `node`.
    #[inline(never)]
    fn set(&self, node: u32) -> &[u64] {
        at!(self.sets, node as usize * self.words..(node as usize + 1) * self.words)
    }

    /// The rows node `node` makes.
    #[inline(never)]
    fn rows(&self, node: u32) -> f64 {
        at!(self.nodes, node as usize).rows
    }

    /// Makes node `slot` a join of nodes `left` and `right`, which makes
    /// `rows` rows.
    #[inline(never)]
    fn join(&mut self, slot: u32, left: u32, right: u32, rows: f64) {
        *at_mut!(self.nodes, slot as usize) = Node { left, right, rows, place: NONE };
        for i in 0..self.words {
            let word = at!(self.set(left), i) | at!(self.set(right), i);
            *at_mut!(self.sets, slot as usize * self.words + i) = word;
        }
    }

    /// The trees that hold all of each side of edge `e`. A side whose
    /// relations are in more than one tree has `NONE`.
    #[inline(never)]
    fn trees(&self, e: usize) -> [u32; 2] {
        let edge = at!(self.edges, e);
        let mut trees = [NONE; 2];
        for (right, tree) in trees.iter_mut().enumerate() {
            let owner = at!(self.nodes, *at!(edge.first, right) as usize).place;
            if self.meets(e, right, self.set(owner)).1 {
                *tree = owner;
            }
        }
        trees
    }

    /// Whether joining trees `a` and `b` would still let every hyperedge
    /// apply. A join that brings together relations of both sides of a
    /// hyperedge must apply it, unless one of the trees already has.
    /// Otherwise the hyperedge could never apply above it.
    fn valid(&mut self, a: u32, b: u32) -> bool {
        let (a, b) = (self.set(a), self.set(b));
        let (mut e, mut work, mut fits) = (self.hypers as usize, self.work, true);
        while fits && let Some(edge) = self.edges.get(e) {
            // The right side first: it's often one relation, in neither tree.
            work = work.saturating_sub(1);
            let ((right_a, right_in_a), (right_b, right_in_b)) =
                (self.meets(e, 1, a), self.meets(e, 1, b));
            if right_a || right_b {
                // Reading a side of more than one relation costs a step a word.
                work = work.saturating_sub(self.words);
                let ((left_a, left_in_a), (left_b, left_in_b)) =
                    (self.meets(e, 0, a), self.meets(e, 0, b));
                let applies = left_in_a && right_in_b || left_in_b && right_in_a;
                fits &= applies || !(left_a || left_b) || left_a && right_a || left_b && right_b;
            }
            e = edge.next as usize;
        }
        self.work = work;
        fits
    }

    /// The work that reading an edge takes: 4 steps, and with hyperedges a
    /// step more for each word of a set.
    fn read_cost(&self) -> usize {
        if self.hypers == NONE { 4 } else { 4 + self.words }
    }

    /// Joins trees until one is left. Each join reads a window of the edges,
    /// starting where the last one stopped, and reads on if it finds no pair
    /// of trees in it. Of the pairs it finds, the `TOP` whose single edges
    /// make the fewest rows are priced by all their edges, and the cheapest
    /// that can be joined is. Returns false if it's stuck in a connected
    /// graph.
    #[inline(never)]
    fn greedy(&mut self) -> bool {
        let (graph, count) = (self.graph, self.edges.len());
        let relations = graph.relations.len();
        // A quarter of the work, spread over the joins, for a read of each
        // edge in the window and the rows of the pairs found.
        let wide = (WORK_MAX / (4 * self.read_cost() * relations)).clamp(1, count);
        let mut at = 0;
        #[expect(clippy::cast_possible_truncation, reason = "at most `2 * LIST_MAX`")]
        for made in relations as u32..2 * relations as u32 - 1 {
            // Once the work is spent, the first pair that can be joined is.
            let window = if self.work == 0 { 1 } else { wide };
            let (mut best, mut a, mut b, mut spanning) = (f64::INFINITY, NONE, NONE, false);
            while a == NONE {
                // The edges whose own rows are fewest, fewest first.
                let mut top = [(f64::INFINITY, 0); TOP];
                let mut read = 0;
                while read < count && (read < window || at!(top, 0).0 == f64::INFINITY) {
                    let (e, edge) = (at, at!(graph.edges, at));
                    (at, read) = (if at + 1 == count { 0 } else { at + 1 }, read + 1);
                    let [left, right] = self.trees(e);
                    if left == right && left != NONE {
                        continue;
                    }
                    spanning = true;
                    if left == NONE || right == NONE || left.max(right) < at!(self.edges, e).dead {
                        continue;
                    }
                    self.estimator.start(self.rows(left), self.rows(right));
                    self.estimator.apply(edge, true);
                    let (rows, mut i) = (self.estimator.finish(), TOP - 1);
                    if rows < at!(top, i).0 {
                        *at_mut!(top, i) = (rows, e);
                    }
                    while i > 0 && at!(top, i).0 < at!(top, i - 1).0 {
                        top.swap(i, i - 1);
                        i -= 1;
                    }
                }
                self.work = self.work.saturating_sub(self.read_cost() * read);
                if at!(top, 0).0 == f64::INFINITY {
                    break;
                }
                // Of their pairs, the one whose edges together make the fewest
                // rows, of those that can be joined.
                for &(rows, e) in &top {
                    if rows == f64::INFINITY {
                        break;
                    }
                    let [left, right] = self.trees(e);
                    if self.hypers != NONE && !self.valid(left, right) {
                        at_mut!(self.edges, e).dead = made;
                        continue;
                    }
                    let rows = self.rows_of(left, right, e, window);
                    if rows < best {
                        (best, a, b) = (rows, left, right);
                    }
                }
            }
            if a == NONE {
                if spanning {
                    return false;
                }
                // The graph isn't connected: join the first relation's tree
                // to another tree.
                a = at!(self.nodes, 0).place;
                let relations = at!(self.nodes, ..relations).iter();
                b = relations.map(|relation| relation.place).find(|&b| b != a).unwrap_or(a);
                best = self.rows_of(a, b, 0, 0);
            }
            self.join(made, a, b, best);
            self.mark(made, made);
        }
        true
    }

    /// The rows that joining trees `a` and `b` makes, by the edges between
    /// them among the `count` edges from edge `start` on.
    #[inline(never)]
    fn rows_of(&mut self, a: u32, b: u32, start: usize, count: usize) -> f64 {
        let graph = self.graph;
        self.work = self.work.saturating_sub(count);
        self.estimator.start(self.rows(a), self.rows(b));
        let mut e = start;
        for _ in 0..count {
            let [left, right] = self.trees(e);
            if left == a && right == b || left == b && right == a {
                self.estimator.apply(at!(graph.edges, e), left == a);
            }
            e = if e + 1 == graph.edges.len() { 0 } else { e + 1 };
        }
        self.estimator.finish()
    }

    /// The plan as a `JoinTree`, whose nodes are the planner's, in the same
    /// order.
    fn tree(&self, allocator: &dyn Allocator) -> Result<JoinTree, Error> {
        let mut tree = JoinTree::new(allocator, self.graph.relations.len())?;
        // The graph's relations are its first nodes.
        #[expect(clippy::cast_possible_truncation, reason = "at most `2 * LIST_MAX`")]
        for (index, node) in (0..self.nodes.len() as u32).zip(self.nodes.iter()) {
            match node.right {
                NONE => tree.relation(index)?,
                right => tree.join(node.left, right)?,
            };
        }
        Ok(tree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;
    use crate::join_order::{EquivalenceSet, JoinEdge, JoinKind, Relation, cost, enumerate};

    /// A cycle of `count` relations, with a left join to the last, which
    /// needs all the others first.
    fn cycle(count: u32) -> QueryGraph {
        let mut graph = QueryGraph::new(&Heap).unwrap();
        let set = |relations: &mut dyn Iterator<Item = u32>| {
            let mut set = RelationSet::new(&Heap).unwrap();
            relations.for_each(|relation| set.insert(relation).unwrap());
            set
        };
        for relation in 0..count {
            let rows = 10 + u64::from(relation * 7919 % 1000);
            graph.relations.push(Relation { rows, unfiltered_rows: rows }).unwrap();
            graph.equivalences.push(EquivalenceSet { domain: rows }).unwrap();
            let (left, right) = match relation {
                0 => continue,
                last if last + 1 == count => (set(&mut (0..last)), set(&mut [last].into_iter())),
                _ => (set(&mut [relation - 1].into_iter()), set(&mut [relation].into_iter())),
            };
            let kind = if relation + 1 == count { JoinKind::Left } else { JoinKind::Inner };
            graph.edges.push(JoinEdge { left, right, kind, equivalence: relation }).unwrap();
        }
        let (left, right) = (set(&mut [0].into_iter()), set(&mut [count - 2].into_iter()));
        graph.edges.push(JoinEdge { left, right, kind: JoinKind::Inner, equivalence: 0 }).unwrap();
        graph
    }

    #[test]
    fn plans_are_valid_and_no_worse_than_written() {
        for count in [3, 10, 70] {
            let graph = cycle(count);
            let planned = cost(&Heap, &graph, &enumerate(&Heap, &graph).unwrap()).unwrap();
            let written = cost(&Heap, &graph, &as_written(&Heap, &graph).unwrap()).unwrap();
            assert!(planned.valid.is_ok() && planned.cost <= written.cost);
        }
    }

    #[test]
    fn one_or_two_relations_need_no_planning() {
        let mut graph = QueryGraph::new(&Heap).unwrap();
        graph.relations.push(Relation { rows: 10, unfiltered_rows: 10 }).unwrap();
        assert_eq!(enumerate(&Heap, &graph).unwrap().nodes.len(), 1);
        graph.relations.push(Relation { rows: 20, unfiltered_rows: 20 }).unwrap();
        let tree = enumerate(&Heap, &graph).unwrap();
        assert_eq!(cost(&Heap, &graph, &tree).unwrap().valid, Ok(()));
    }

    #[test]
    fn disconnected_graphs_take_cross_products() {
        for count in [5, 70] {
            let mut graph = cycle(count);
            graph.relations.push(Relation { rows: 3, unfiltered_rows: 3 }).unwrap();
            let planned = cost(&Heap, &graph, &enumerate(&Heap, &graph).unwrap()).unwrap();
            assert!(planned.valid.is_ok());
        }
    }
}
