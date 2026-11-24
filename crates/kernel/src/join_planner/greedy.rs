//! Greedy operator ordering (Fegaras, "A New Heuristic for Optimizing Large
//! Queries", 1998): starting with one tree per relation, it joins the valid
//! pair of trees whose join makes the fewest rows, until one tree is left.

use super::estimate::Estimator;
use super::graph::{QueryGraph, Sides};
use super::tree::{JoinNode, JoinTree};
use crate::containers::{Array, BitSlice, SmallVec};
use crate::env::Env;
use crate::error::Error;

/// No node, tree or edge.
const NONE: u32 = u32::MAX;

/// How many pairs of trees each join prices by all the edges between them.
/// They are the pairs whose single edges make the fewest rows. A pair's rows
/// can be far fewer than any one of its edges says, so this has to be more
/// than a few.
const TOP: usize = 16;

/// A node of the forest. The graph's relations come first, then the joins.
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

/// What it reads of the graph's edge of the same index.
#[derive(Clone, Copy)]
struct Edge {
    /// Its sides' first relations, and whether they have more.
    sides: Sides,
    /// For a hyperedge in the list that starts at `Greedy::hypers`, the next
    /// one, or `NONE`.
    next: u32,
    /// How many nodes there were when joining its two trees was found to be
    /// invalid. While both trees are older than that, they still can't be
    /// joined.
    dead: u32,
}

/// The forest as it's joined, and what it reads of the graph.
struct Greedy<'g, 'e> {
    /// The graph being planned.
    graph: &'g QueryGraph,
    /// Estimates the rows of the joins it tries.
    estimator: &'e mut Estimator<'g>,
    /// How many words a set of relations takes.
    words: usize,
    /// The forest's nodes.
    nodes: SmallVec<Node>,
    /// Each node's relations, `words` words a node.
    sets: Array<u64>,
    /// The graph's edges.
    edges: SmallVec<Edge>,
    /// The first of the hyperedges, linked through `Edge::next`, or `NONE`.
    /// A hyperedge with the same sides as the one linked before it is left
    /// out, as it can't change whether a join is valid.
    hypers: u32,
}

/// Plans `graph` greedily, drawing on `work`: once it's spent, the first
/// valid pair is joined. `None` if no valid pair is left in a connected
/// graph, which hyperedges can cause.
pub(super) fn plan<'g>(
    env: &Env,
    graph: &'g QueryGraph,
    estimator: &mut Estimator<'g>,
    work: &mut usize,
) -> Result<Option<JoinTree>, Error> {
    let count = graph.relations.len();
    let words = BitSlice::<u64>::words_for(count);
    let mut greedy = Greedy {
        graph,
        estimator,
        words,
        nodes: SmallVec::fixed(env, 2 * count - 1)?,
        sets: Array::zeroed(env, (2 * count - 1) * words)?,
        edges: SmallVec::fixed(env, graph.edges.len())?,
        hypers: NONE,
    };
    #[expect(clippy::cast_possible_truncation, reason = "at most `2 * LIST_MAX`")]
    for node in 0..2 * count as u32 - 1 {
        #[expect(clippy::cast_precision_loss, reason = "an estimate")]
        let rows = graph.relations.get(node as usize).map_or(0.0, |found| found.rows as f64);
        greedy.nodes.push(Node { left: NONE, right: NONE, rows, place: node })?;
    }
    for relation in 0..count {
        BitSlice::of_mut(at_mut!(greedy.sets, relation * words..)).set(relation);
    }
    // Each hyperedge is linked in front of those before it, unless it has
    // the same sides as the last one linked.
    #[expect(clippy::cast_possible_truncation, reason = "at most `LIST_MAX`")]
    for (e, edge) in (0..graph.edges.len() as u32).zip(graph.edges.iter()) {
        let (sides, next) = (Sides::of(edge), greedy.hypers);
        let last = graph.edges.get(greedy.hypers as usize);
        if sides.more.contains(&true)
            && !last.is_some_and(|last| last.left == edge.left && last.right == edge.right)
        {
            greedy.hypers = e;
        }
        greedy.edges.push(Edge { sides, next, dead: 0 })?;
    }
    if !greedy.join_all(work) {
        return Ok(None);
    }
    let mut tree = JoinTree::new(env, count)?;
    #[expect(clippy::cast_possible_truncation, reason = "at most `2 * LIST_MAX`")]
    for (index, node) in (0..greedy.nodes.len() as u32).zip(greedy.nodes.iter()) {
        match node.right {
            NONE => tree.add(JoinNode::Relation(index), node.rows),
            right => tree.add(JoinNode::Join { left: node.left, right }, node.rows),
        };
    }
    Ok(Some(tree))
}

impl Greedy<'_, '_> {
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

    /// Whether side `right` of edge `e` has any relation of `set`, and
    /// whether all its relations are in `set`.
    fn meets(&self, e: usize, right: usize, set: &[u64]) -> (bool, bool) {
        at!(self.edges, e).sides.meets(at!(self.graph.edges, e), right, set)
    }

    /// The relations of node `node`.
    #[inline(never)]
    fn set(&self, node: u32) -> &[u64] {
        at!(self.sets, node as usize * self.words..(node as usize + 1) * self.words)
    }

    /// The rows node `node` makes.
    fn rows(&self, node: u32) -> f64 {
        at!(self.nodes, node as usize).rows
    }

    /// Makes node `slot` a join of nodes `left` and `right`, which makes
    /// `rows` rows.
    #[inline(never)]
    fn join(&mut self, slot: u32, left: u32, right: u32, rows: f64) {
        *at_mut!(self.nodes, slot as usize) = Node { left, right, rows, place: NONE };
        super::graph::union(&mut self.sets, self.words, slot, [left, right]);
    }

    /// The trees that hold all of each side of edge `e`. A side whose
    /// relations are in more than one tree has `NONE`.
    #[inline(never)]
    fn trees(&self, e: usize) -> [u32; 2] {
        let edge = at!(self.edges, e);
        let mut trees = [NONE; 2];
        for (right, tree) in trees.iter_mut().enumerate() {
            let owner = at!(self.nodes, *at!(edge.sides.first, right) as usize).place;
            if self.meets(e, right, self.set(owner)).1 {
                *tree = owner;
            }
        }
        trees
    }

    /// Whether joining trees `a` and `b` is valid, drawing on `work`. A join
    /// that brings together relations of both sides of a hyperedge must
    /// apply it, unless one of the trees already has, or the hyperedge could
    /// never apply above it. And the operator must be able to run it.
    fn valid(&self, a: u32, b: u32, work: &mut usize) -> bool {
        let (a, b) = (self.set(a), self.set(b));
        if let Some(executable) = self.graph.executable
            && !executable(self.graph, a, b)
        {
            return false;
        }
        let (mut e, mut fits) = (self.hypers as usize, true);
        while fits && let Some(edge) = self.edges.get(e) {
            // The right side first: it's often one relation, in neither tree.
            *work = work.saturating_sub(1);
            let ((right_a, right_in_a), (right_b, right_in_b)) =
                (self.meets(e, 1, a), self.meets(e, 1, b));
            if right_a || right_b {
                // Reading a side of more than one relation costs a step a word.
                *work = work.saturating_sub(self.words);
                let ((left_a, left_in_a), (left_b, left_in_b)) =
                    (self.meets(e, 0, a), self.meets(e, 0, b));
                let applies = left_in_a && right_in_b || left_in_b && right_in_a;
                fits &= applies || !(left_a || left_b) || left_a && right_a || left_b && right_b;
            }
            e = edge.next as usize;
        }
        fits
    }

    /// The work that reading an edge takes: 4 steps, and with hyperedges a
    /// step more for each word of a set.
    fn read_cost(&self) -> usize {
        if self.hypers == NONE { 4 } else { 4 + self.words }
    }

    /// Joins trees until one is left, drawing on `work`. Each join reads a
    /// window of the edges, starting where the last one stopped, and reads
    /// on if it finds no pair of trees in it. Of the pairs it finds, the
    /// `TOP` whose single edges make the fewest rows are priced by all their
    /// edges, and the cheapest that can be joined is. Returns false if it's
    /// stuck in a connected graph.
    #[inline(never)]
    fn join_all(&mut self, work: &mut usize) -> bool {
        let (graph, count) = (self.graph, self.edges.len());
        let relations = graph.relations.len();
        // A quarter of the work, spread over the joins, for a read of each
        // edge in the window and the rows of the pairs found.
        let wide = (*work / (4 * self.read_cost() * relations)).clamp(1, count.max(1));
        let mut at = 0;
        #[expect(clippy::cast_possible_truncation, reason = "at most `2 * LIST_MAX`")]
        for made in relations as u32..2 * relations as u32 - 1 {
            // Once the work is spent, the first pair that can be joined is.
            let window = if *work == 0 { 1 } else { wide };
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
                    self.estimator.apply(edge);
                    let (rows, mut i) = (self.estimator.finish(), TOP - 1);
                    if rows < at!(top, i).0 {
                        *at_mut!(top, i) = (rows, e);
                    }
                    while i > 0 && at!(top, i).0 < at!(top, i - 1).0 {
                        top.swap(i, i - 1);
                        i -= 1;
                    }
                }
                *work = work.saturating_sub(self.read_cost() * read);
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
                    if !self.valid(left, right, work) {
                        at_mut!(self.edges, e).dead = made;
                        continue;
                    }
                    let rows = self.rows_of(left, right, [e, window], work);
                    if rows < best {
                        (best, a, b) = (rows, left, right);
                    }
                }
            }
            (best, a, b) = self.single((best, a, b), work);
            if a == NONE {
                if spanning {
                    return false;
                }
                // The graph isn't connected: join the two trees whose cross
                // product is smallest, those of fewest rows.
                a = self.smallest(NONE, work);
                b = self.smallest(a, work);
                best = self.rows_of(a, b, [0, 0], work);
            }
            self.join(made, a, b, best);
            self.mark(made, made);
        }
        true
    }

    /// The cheaper of `best`, a join's rows and its trees, and the join of a
    /// lone relation of at most one row to the tree of fewest rows, whatever
    /// edges there are, drawing on `work`.
    #[inline(never)]
    fn single(&mut self, mut best: (f64, u32, u32), work: &mut usize) -> (f64, u32, u32) {
        for (single, relation) in (0..).zip(self.graph.relations.iter()) {
            if relation.max_rows > 1
                || *work == 0
                || at!(self.nodes, single as usize).place != single
            {
                continue;
            }
            let tree = self.smallest(single, work);
            if !self.valid(single, tree, work) {
                continue;
            }
            self.estimator.start(self.rows(single), self.rows(tree));
            let rows = self.estimator.finish();
            if rows < best.0 {
                best = (rows, single, tree);
            }
        }
        best
    }

    /// The tree of fewest rows other than `except`, drawing on `work`.
    fn smallest(&self, except: u32, work: &mut usize) -> u32 {
        let relations = self.graph.relations.len();
        *work = work.saturating_sub(relations);
        let mut found = NONE;
        for relation in at!(self.nodes, ..relations) {
            let tree = relation.place;
            if tree != except && (found == NONE || self.rows(tree) < self.rows(found)) {
                found = tree;
            }
        }
        found
    }

    /// The rows that joining trees `a` and `b` makes, by the edges between
    /// them among the `count` edges from edge `start` on, drawing on `work`.
    #[inline(never)]
    fn rows_of(&mut self, a: u32, b: u32, [start, count]: [usize; 2], work: &mut usize) -> f64 {
        let graph = self.graph;
        *work = work.saturating_sub(count);
        self.estimator.start(self.rows(a), self.rows(b));
        let mut e = start;
        for _ in 0..count {
            let [left, right] = self.trees(e);
            if left == a && right == b || left == b && right == a {
                self.estimator.apply(at!(graph.edges, e));
            }
            e = if e + 1 == graph.edges.len() { 0 } else { e + 1 };
        }
        self.estimator.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Env;
    use crate::join_planner::{EquivalenceSet, JoinEdge, JoinNode, Relation, RelationSet};

    /// A graph of relations of `rows`, each a pair of rows and the most rows,
    /// and inner joins between pairs of relations on domains of `edges`.
    fn graph(rows: &[(u64, u64)], edges: &[(u32, u32, u64)]) -> QueryGraph {
        let mut graph = QueryGraph::new(Env::current()).unwrap();
        for &(rows, max_rows) in rows {
            graph.relations.push(Relation { rows, max_rows }).unwrap();
        }
        for (equivalence, &(left, right, domain)) in (0..).zip(edges) {
            graph.equivalences.push(EquivalenceSet { domain }).unwrap();
            let (mut left_set, mut right_set) = (
                RelationSet::new(Env::current()).unwrap(),
                RelationSet::new(Env::current()).unwrap(),
            );
            left_set.insert(left).unwrap();
            right_set.insert(right).unwrap();
            let (left, right) = (left_set, right_set);
            graph.edges.push(JoinEdge { left, right, equivalence }).unwrap();
        }
        graph
    }

    /// The first join greedy ordering makes in `graph`.
    fn first_join(graph: &QueryGraph) -> JoinNode {
        let mut estimator = Estimator::new(&graph.equivalences, &mut []);
        let mut work = 1 << 20;
        let tree = plan(Env::current(), graph, &mut estimator, &mut work).unwrap().unwrap();
        at!(tree.nodes, graph.relations.len()).kind
    }

    #[test]
    fn cross_products_join_the_smallest_trees() {
        let graph = graph(&[(100, 100), (5, 5), (7, 7)], &[]);
        assert_eq!(first_join(&graph), JoinNode::Join { left: 1, right: 2 });
    }

    #[test]
    fn one_row_relations_join_the_smallest_tree() {
        // The edge's join makes 1000 rows, and the one-row relation's 10.
        let graph = graph(&[(1000, 1000), (10, 10), (1, 1)], &[(0, 1, 10)]);
        assert_eq!(first_join(&graph), JoinNode::Join { left: 2, right: 1 });
    }
}
