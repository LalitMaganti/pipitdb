//! Windows: bottom up, the subtrees of a plan under each join, up to
//! `WINDOW` of them, are joined again in the cheapest valid way, which exact
//! search's solver finds.

use super::estimate::Estimator;
use super::exact::{self, Link};
use super::graph::{LIST_MAX, QueryGraph};
use super::tree::{JoinNode, JoinTree, Node};
use crate::containers::{Array, BitSlice, SmallVec};
use crate::env::Env;
use crate::error::Error;

/// How many subtrees a window joins.
const WINDOW: usize = 8;

/// The plan as it's improved, and what it reads of the graph.
struct Windows<'a, 'g, 'e, 't> {
    /// Where the windows' searches get their memory.
    env: &'a Env,
    /// The graph being planned.
    graph: &'g QueryGraph,
    /// Estimates the rows of the joins it tries.
    estimator: &'e mut Estimator<'g>,
    /// The plan.
    tree: &'t mut JoinTree,
    /// How many words a set of relations takes.
    words: usize,
    /// Each node's relations, `words` words a node.
    sets: Array<u64>,
    /// For each relation of the window, the subtree it's in.
    label: Array<u8>,
    /// The edges between the window's subtrees.
    links: SmallVec<Link>,
}

/// Improves `tree`, a plan of `graph` whose joins come after their
/// children, with windows, drawing on `work`. Once it's spent, windows stop.
pub(super) fn improve<'g>(
    env: &Env,
    graph: &'g QueryGraph,
    estimator: &mut Estimator<'g>,
    tree: &mut JoinTree,
    work: &mut usize,
) -> Result<(), Error> {
    let count = graph.relations.len();
    let words = BitSlice::<u64>::words_for(count);
    let mut sets = Array::zeroed(env, (tree.nodes.len() + 2) * words)?;
    for (node, found) in tree.nodes.iter().enumerate() {
        match found.kind {
            JoinNode::Relation(relation) => {
                let relation = relation as usize;
                BitSlice::of_mut(at_mut!(sets, node * words..)).set(relation);
            }
            JoinNode::Join { left, right } => {
                #[expect(clippy::cast_possible_truncation, reason = "at most `2 * LIST_MAX`")]
                super::graph::union(&mut sets, words, node as u32, [left, right]);
            }
        }
    }
    let mut windows = Windows {
        env,
        graph,
        estimator,
        tree,
        words,
        sets,
        label: Array::zeroed(env, count)?,
        links: SmallVec::new(env, LIST_MAX)?,
    };
    // Joins come after their children, so each window's subtrees have
    // already been improved.
    #[expect(clippy::cast_possible_truncation, reason = "at most `2 * LIST_MAX`")]
    for root in count as u32..2 * count as u32 - 1 {
        if *work == 0 {
            break;
        }
        windows.refine(root, work)?;
    }
    Ok(())
}

impl Windows<'_, '_, '_, '_> {
    /// The relations of node `node`.
    fn set(&self, node: u32) -> &[u64] {
        at!(self.sets, node as usize * self.words..(node as usize + 1) * self.words)
    }

    /// The children of node `node`, if it's a join.
    fn children(&self, node: u32) -> Option<(u32, u32)> {
        match at!(self.tree.nodes, node as usize).kind {
            JoinNode::Join { left, right } => Some((left, right)),
            JoinNode::Relation(_) => None,
        }
    }

    /// Replaces the joins under `root` by the cheapest tree of up to
    /// `WINDOW` of its subtrees, its units, drawing on `work`. The units are
    /// `root`'s children at first. Then the unit that makes the most rows and
    /// is a join is replaced by its children, until there are `WINDOW`.
    #[inline(never)]
    fn refine(&mut self, root: u32, work: &mut usize) -> Result<(), Error> {
        let Some((left, right)) = self.children(root) else { return Ok(()) };
        // The joins the units are replaced from, which take the new joins.
        let (mut units, mut slots) = ([left; WINDOW], [root; WINDOW]);
        *at_mut!(units, 1) = right;
        let mut k = 2;
        while k < WINDOW {
            // The unit that makes the most rows and is a join, if any.
            let (mut found, mut rows) = (None, -1.0);
            for (i, &unit) in at!(units, ..k).iter().enumerate() {
                if let Some(children) = self.children(unit)
                    && at!(self.tree.nodes, unit as usize).rows > rows
                {
                    (found, rows) = (Some((i, children)), at!(self.tree.nodes, unit as usize).rows);
                }
            }
            let Some((i, (left, right))) = found else { break };
            *at_mut!(slots, k - 1) = *at!(units, i);
            (*at_mut!(units, i), *at_mut!(units, k)) = (left, right);
            k += 1;
        }
        if k < 3 {
            return Ok(());
        }
        let units = at!(units, ..k);
        self.links(root, units, work)?;
        // Every split of every set, about 3^k, each reading every link.
        let splits = (2..k).fold(9, |splits, _| splits * 3);
        *work = work.saturating_sub(splits * (self.links.len() + 1));
        let mut rows = [0.0; WINDOW];
        for (rows, &unit) in rows.iter_mut().zip(units) {
            *rows = at!(self.tree.nodes, unit as usize).rows;
        }
        let (graph, words) = (self.graph, self.words);
        let Some((sets, sides)) = self.sets.split_at_mut_checked(self.tree.nodes.len() * words)
        else {
            failed!()
        };
        let sets = &*sets;
        let mut can_run = |left, right| can_run(graph, sets, words, sides, units, [left, right]);
        // The units that are one relation of at most one row.
        let single = (0..).zip(units).fold(0, |single, (i, &unit)| {
            match at!(self.tree.nodes, unit as usize).kind {
                JoinNode::Relation(relation)
                    if at!(graph.relations, relation as usize).max_rows <= 1 =>
                {
                    single | 1 << i
                }
                _ => single,
            }
        });
        let (rows, links) = (at!(rows, ..k), &self.links);
        let solved =
            exact::solve(self.env, graph, self.estimator, links, rows, single, &mut can_run)?;
        let Some(solved) = solved else { return Ok(()) };
        // Its nodes come after their children and its root last, so the new
        // root goes in `root`, the first slot, which is taken last.
        let mut slots = at!(slots, ..k - 1).iter().rev();
        let mut nodes = [0; 2 * WINDOW];
        for (i, node) in solved.nodes.iter().enumerate() {
            *at_mut!(nodes, i) = match node.kind {
                JoinNode::Relation(unit) => *at!(units, unit as usize),
                JoinNode::Join { left, right } => {
                    let Some(&slot) = slots.next() else { failed!() };
                    let children = [left, right].map(|child| *at!(nodes, child as usize));
                    self.join(slot, children, node.rows);
                    slot
                }
            };
        }
        Ok(())
    }

    /// Labels the relations of `units` and finds the edges between them, of
    /// the edges whose relations are all under `root`, drawing on `work`.
    #[inline(never)]
    fn links(&mut self, root: u32, units: &[u32], work: &mut usize) -> Result<(), Error> {
        #[expect(clippy::cast_possible_truncation, reason = "at most `WINDOW`")]
        for (i, &unit) in units.iter().enumerate() {
            for w in 0..self.words {
                let mut bits = *at!(self.sets, unit as usize * self.words + w);
                while bits != 0 {
                    *at_mut!(self.label, w * 64 + bits.trailing_zeros() as usize) = i as u8;
                    bits &= bits - 1;
                }
            }
        }
        *work = work.saturating_sub((4 + self.words) * self.graph.edges.len());
        self.links.clear();
        #[expect(clippy::cast_possible_truncation, reason = "at most `LIST_MAX`")]
        'edges: for (edge, found) in (0..self.graph.edges.len() as u32).zip(self.graph.edges.iter())
        {
            let mut units = [0_u64; 2];
            for (relations, side) in [&found.left, &found.right].into_iter().zip(&mut units) {
                if !relations.meets(self.set(root)).1 {
                    continue 'edges;
                }
                for w in 0..relations.words() {
                    let mut bits = relations.word(w);
                    while bits != 0 {
                        *side |= 1 << at!(self.label, w * 64 + bits.trailing_zeros() as usize);
                        bits &= bits - 1;
                    }
                }
            }
            if !(at!(units, 0) | at!(units, 1)).is_power_of_two() {
                self.links.push(Link { sides: units, edge })?;
            }
        }
        Ok(())
    }

    /// Makes node `slot` a join of nodes `left` and `right`, of `rows` rows.
    fn join(&mut self, slot: u32, [left, right]: [u32; 2], rows: f64) {
        let kind = JoinNode::Join { left, right };
        *at_mut!(self.tree.nodes, slot as usize) = Node { kind, rows };
        super::graph::union(&mut self.sets, self.words, slot, [left, right]);
    }
}

/// Whether the operator can run a join of the window's `units` in `left` and
/// `right`, as bits. `sets` has each node's relations in `words` words, and
/// `sides` takes the two sides' relations.
fn can_run(
    graph: &QueryGraph,
    sets: &[u64],
    words: usize,
    sides: &mut [u64],
    units: &[u32],
    [left, right]: [u64; 2],
) -> bool {
    let Some(executable) = graph.executable else { return true };
    for (side, set) in [left, right].into_iter().enumerate() {
        for w in 0..words {
            let word = (0..units.len())
                .filter(|i| set >> i & 1 != 0)
                .fold(0, |word, i| word | at!(sets, *at!(units, i) as usize * words + w));
            *at_mut!(sides, side * words + w) = word;
        }
    }
    executable(graph, at!(sides, ..words), at!(sides, words..))
}

#[cfg(test)]
mod tests {
    use crate::env::Env;
    use crate::join_planner::{
        EquivalenceSet, JoinEdge, JoinKind, JoinNode, QueryGraph, Relation, RelationSet, plan,
    };

    /// A graph of relations of `rows`, each a pair of rows and the most rows,
    /// and inner joins between pairs of relations on domains of `edges`.
    fn graph(rows: &[(u64, u64)], edges: &[(u32, u32, u64)]) -> QueryGraph {
        let mut graph = QueryGraph::new(Env::current()).unwrap();
        for &(rows, max_rows) in rows {
            graph.relations.push(Relation { rows, unfiltered_rows: rows, max_rows }).unwrap();
        }
        for (equivalence, &(left, right, domain)) in (0..).zip(edges) {
            graph.equivalences.push(EquivalenceSet { domain }).unwrap();
            let (mut left_set, mut right_set) = (
                RelationSet::new(Env::current()).unwrap(),
                RelationSet::new(Env::current()).unwrap(),
            );
            left_set.insert(left).unwrap();
            right_set.insert(right).unwrap();
            let (left, right, kind) = (left_set, right_set, JoinKind::Inner);
            graph.edges.push(JoinEdge { left, right, kind, equivalence }).unwrap();
        }
        graph
    }

    #[test]
    fn windows_keep_one_row_relations() {
        // A chain too big for exact search, and a relation of a row with no
        // edge to it.
        let mut rows: alloc::vec::Vec<(u64, u64)> =
            (0..70).map(|i| (10 + i * 7 % 100, 10 + i * 7 % 100)).collect();
        rows.push((1, 1));
        let edges: alloc::vec::Vec<(u32, u32, u64)> = (1..70).map(|i| (i - 1, i, 10)).collect();
        let graph = graph(&rows, &edges);
        let tree = plan(Env::current(), &graph).unwrap().unwrap();
        let mut seen = [false; 71];
        for node in tree.nodes.iter() {
            if let JoinNode::Relation(relation) = node.kind {
                assert!(!core::mem::replace(at_mut!(seen, relation as usize), true));
            }
        }
        assert!(seen.iter().all(|&seen| seen));
    }
}
