//! Join ordering. A query graph holds the relations of a join region and the
//! joins between them. An enumerator picks a join tree for the graph: the
//! order to join the relations in. A tree's cost is `C_out`, the rows of all
//! its joins added up. Each join counts as at least one row, because even an
//! empty join costs something.
//!
//! A join's rows are estimated as DuckDB does, from equivalence sets: the
//! product of its inputs' rows, divided by the domain of each equivalence set
//! it joins on, once per set. The relations of an equivalence set are each
//! joined to every other one, so a set of relations makes the same rows
//! whichever order it's joined in, apart from rounding and the cap at
//! `ROWS_MAX`.

use crate::allocator::{AllocError, Allocator};
use crate::error::Error;
use crate::slow_vec::SlowVec;

mod planner;

/// The most relations, edges or nodes a graph or tree holds.
const LIST_MAX: usize = 1 << 24;

/// The most words a `RelationSet` grows to, enough for `LIST_MAX` relations.
const WORDS_MAX: usize = LIST_MAX / 64;

/// The most rows an estimate reaches.
#[expect(clippy::cast_precision_loss, reason = "a limit, not a count")]
pub const ROWS_MAX: f64 = u64::MAX as f64;

/// The least share of its left rows an anti join is estimated to keep, since
/// no estimate can be sure that every row matches.
const ANTI_KEPT_MIN: f64 = 0.1;

/// Picks a valid join tree for `graph`, with memory from `allocator`.
pub type Enumerator = fn(&dyn Allocator, &QueryGraph) -> Result<JoinTree, Error>;

/// A set of relations, by index. The first 64 relations take one word,
/// and the set grows a word at a time past them.
pub struct RelationSet {
    /// Relations 0 to 63.
    first: u64,
    /// Relations 64 and up, 64 to a word. Words past the end are empty.
    rest: SlowVec<u64>,
}

impl RelationSet {
    /// An empty set. It allocates nothing until it holds a relation past 63.
    pub fn new(allocator: &dyn Allocator) -> Result<RelationSet, AllocError> {
        Ok(RelationSet { first: 0, rest: SlowVec::new(allocator, WORDS_MAX)? })
    }

    /// Word `i`, which holds relations `64 * i` to `64 * i + 63`.
    #[inline]
    pub fn word(&self, i: usize) -> u64 {
        match i {
            0 => self.first,
            _ => self.rest.get(i - 1).copied().unwrap_or(0),
        }
    }

    /// How many words it has. All words past these are empty.
    #[inline]
    pub fn words(&self) -> usize {
        1 + self.rest.len()
    }

    pub fn insert(&mut self, relation: u32) -> Result<(), AllocError> {
        let word = relation as usize / 64;
        if word >= self.words() {
            self.rest.extend_zeroed(word + 1 - self.words())?;
        }
        *self.word_mut(word) |= 1 << (relation % 64);
        Ok(())
    }

    pub fn contains(&self, relation: u32) -> bool {
        self.word(relation as usize / 64) & (1 << (relation % 64)) != 0
    }

    /// Adds every relation of `other`.
    pub fn union_with(&mut self, other: &RelationSet) -> Result<(), AllocError> {
        if other.words() > self.words() {
            self.rest.extend_zeroed(other.words() - self.words())?;
        }
        for i in 0..other.words() {
            *self.word_mut(i) |= other.word(i);
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        (0..self.words()).all(|i| self.word(i) == 0)
    }

    pub fn len(&self) -> u32 {
        (0..self.words()).map(|i| self.word(i).count_ones()).sum()
    }

    pub fn is_subset(&self, of: &RelationSet) -> bool {
        (0..self.words()).all(|i| self.word(i) & !of.word(i) == 0)
    }

    /// Whether every relation of `self` is in `a` or `b`.
    pub fn is_subset_of_union(&self, a: &RelationSet, b: &RelationSet) -> bool {
        (0..self.words()).all(|i| self.word(i) & !(a.word(i) | b.word(i)) == 0)
    }

    pub fn overlaps(&self, other: &RelationSet) -> bool {
        (0..self.words().min(other.words())).any(|i| self.word(i) & other.word(i) != 0)
    }

    /// The relations, in order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.words()).flat_map(move |i| {
            let mut word = self.word(i);
            core::iter::from_fn(move || {
                let bit = word.trailing_zeros();
                (word != 0).then(|| {
                    word &= word - 1;
                    #[expect(clippy::cast_possible_truncation, reason = "at most `LIST_MAX`")]
                    let relation = i as u32 * 64 + bit;
                    relation
                })
            })
        })
    }

    fn word_mut(&mut self, i: usize) -> &mut u64 {
        match i {
            0 => &mut self.first,
            _ => at_mut!(self.rest, i - 1),
        }
    }
}

/// A relation to join: a table, or a subquery the join region can't look
/// into.
#[derive(Clone, Copy, Debug)]
pub struct Relation {
    /// Its estimated rows after its filters.
    pub rows: u64,
    /// Its rows before its filters.
    pub unfiltered_rows: u64,
}

/// Columns that are joined by equality, so that they all hold the same
/// values.
#[derive(Clone, Copy, Debug)]
pub struct EquivalenceSet {
    /// How many distinct values they can take. It's the largest known
    /// distinct count of the columns, or else the fewest rows of their
    /// relations. For a unique key, it's that relation's unfiltered rows.
    pub domain: u64,
}

/// Which rows a join returns.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JoinKind {
    /// The rows that match.
    Inner,
    /// The rows that match, and the left side's rows that don't, with nulls.
    Left,
    /// The left side's rows that match.
    Semi,
    /// The left side's rows that don't match.
    Anti,
}

/// A join between two sets of relations. It's an edge of the graph, or a
/// hyperedge if either side has more than one relation.
pub struct JoinEdge {
    /// The relations that must be on one side of a join for this edge to
    /// apply there. They are the relations its condition reads and, for a
    /// non-inner join, the left of its total eligibility set (TES): the side
    /// whose rows it keeps.
    pub left: RelationSet,
    /// The same for the other side.
    pub right: RelationSet,
    /// Which rows the join returns.
    pub kind: JoinKind,
    /// The equivalence set it joins on, by index.
    pub equivalence: u32,
}

/// The relations of a join region and the joins between them.
pub struct QueryGraph {
    pub relations: SlowVec<Relation>,
    pub edges: SlowVec<JoinEdge>,
    pub equivalences: SlowVec<EquivalenceSet>,
}

impl QueryGraph {
    pub fn new(allocator: &dyn Allocator) -> Result<QueryGraph, AllocError> {
        Ok(QueryGraph {
            relations: SlowVec::new(allocator, LIST_MAX)?,
            edges: SlowVec::new(allocator, LIST_MAX)?,
            equivalences: SlowVec::new(allocator, LIST_MAX)?,
        })
    }

    /// Whether the edges join every relation to every other, possibly
    /// through others. If not, a plan needs cross products.
    pub fn is_connected(&self, allocator: &dyn Allocator) -> Result<bool, AllocError> {
        let Some(last) = self.relations.len().checked_sub(1) else { return Ok(true) };
        let mut reached = RelationSet::new(allocator)?;
        reached.insert(0)?;
        let mut grew = true;
        while grew {
            grew = false;
            for edge in self.edges.iter() {
                let touches = edge.left.overlaps(&reached) || edge.right.overlaps(&reached);
                if touches && !(edge.left.is_subset(&reached) && edge.right.is_subset(&reached)) {
                    reached.union_with(&edge.left)?;
                    reached.union_with(&edge.right)?;
                    grew = true;
                }
            }
        }
        #[expect(clippy::cast_possible_truncation, reason = "at most `LIST_MAX`")]
        let count = last as u32 + 1;
        Ok(reached.len() == count)
    }
}

/// A node of a join tree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JoinNode {
    /// A relation, by index.
    Relation(u32),
    /// A join of two earlier nodes, by index.
    Join { left: u32, right: u32 },
}

/// A join tree. Each join comes after its children, and the root is last.
pub struct JoinTree {
    pub nodes: SlowVec<JoinNode>,
}

impl JoinTree {
    /// An empty tree with room for a join of `relations` relations.
    pub fn new(allocator: &dyn Allocator, relations: usize) -> Result<JoinTree, AllocError> {
        Ok(JoinTree { nodes: SlowVec::fixed(allocator, (2 * relations).saturating_sub(1))? })
    }

    /// Adds a leaf for `relation`, and returns its node.
    pub fn relation(&mut self, relation: u32) -> Result<u32, AllocError> {
        self.push(JoinNode::Relation(relation))
    }

    /// Adds a join of nodes `left` and `right`, and returns its node.
    pub fn join(&mut self, left: u32, right: u32) -> Result<u32, AllocError> {
        self.push(JoinNode::Join { left, right })
    }

    fn push(&mut self, node: JoinNode) -> Result<u32, AllocError> {
        #[expect(clippy::cast_possible_truncation, reason = "at most `LIST_MAX`")]
        let index = self.nodes.len() as u32;
        self.nodes.push(node)?;
        Ok(index)
    }
}

/// The joins between two sets of relations.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Joined {
    /// The rows they make, at most `ROWS_MAX`.
    pub rows: f64,
    /// Whether any edge joins them. If not, they're a cross product.
    pub any: bool,
}

/// Estimates the rows of joins, with memory it reuses from one join to the
/// next. `join` finds every edge between two sets of relations itself. An
/// enumerator that already knows which edges apply passes them to `start`,
/// `apply` and `finish` instead.
pub struct Estimator<'a> {
    /// The graph whose joins it estimates.
    graph: &'a QueryGraph,
    /// For each equivalence set, the last join (`stamp`) whose rows were
    /// divided by its domain, so that a join divides by it once.
    stamps: SlowVec<u64>,
    /// The join being estimated, numbered from 1.
    stamp: u64,
    /// The join's rows so far.
    rows: f64,
    /// The rows of its left and right sides.
    sides: [f64; 2],
    /// If a non-inner edge applies, its kind and the rows of the side whose
    /// rows it keeps.
    kept: Option<(JoinKind, f64)>,
}

impl<'a> Estimator<'a> {
    pub fn new(allocator: &dyn Allocator, graph: &'a QueryGraph) -> Result<Estimator<'a>, Error> {
        let stamps = SlowVec::zeroed(allocator, graph.equivalences.len())?;
        Ok(Estimator { graph, stamps, stamp: 0, rows: 0.0, sides: [0.0; 2], kept: None })
    }

    /// Starts estimating a join of sides of `left_rows` and `right_rows`.
    /// Until an edge applies, it's a cross product.
    #[inline(never)]
    pub fn start(&mut self, left_rows: f64, right_rows: f64) {
        self.stamp += 1;
        (self.rows, self.sides, self.kept) =
            (left_rows * right_rows, [left_rows, right_rows], None);
    }

    /// Applies `edge` to the join. `forward` is whether the edge's `left` is
    /// on the join's left side.
    pub fn apply(&mut self, edge: &JoinEdge, forward: bool) {
        if edge.kind != JoinKind::Inner {
            let [left, right] = self.sides;
            self.kept = Some((edge.kind, if forward { left } else { right }));
        }
        let stamp = at_mut!(self.stamps, edge.equivalence as usize);
        if *stamp != self.stamp {
            *stamp = self.stamp;
            let domain = at!(self.graph.equivalences, edge.equivalence as usize).domain;
            #[expect(clippy::cast_precision_loss, reason = "an estimate")]
            let domain = domain.max(1) as f64;
            self.rows /= domain;
        }
    }

    /// The join's rows by the edges applied, at most `ROWS_MAX`.
    pub fn finish(&self) -> f64 {
        let rows = match self.kept {
            None => self.rows,
            Some((JoinKind::Left, kept)) => self.rows.max(kept),
            Some((JoinKind::Semi, kept)) => self.rows.min(kept),
            Some((_, kept)) => kept * (1.0 - (self.rows / kept).min(1.0)).max(ANTI_KEPT_MIN),
        };
        rows.min(ROWS_MAX)
    }

    /// The rows that joining `left`, of `left_rows` rows, to `right`, of
    /// `right_rows`, makes, by every edge between them. The order of the
    /// sides doesn't matter: a non-inner edge keeps the rows of whichever
    /// side holds its `left`.
    pub fn join(
        &mut self,
        left: &RelationSet,
        left_rows: f64,
        right: &RelationSet,
        right_rows: f64,
    ) -> Joined {
        self.start(left_rows, right_rows);
        let mut any = false;
        let graph = self.graph;
        for edge in graph.edges.iter() {
            let forward = edge.left.is_subset(left) && edge.right.is_subset(right);
            if forward || edge.left.is_subset(right) && edge.right.is_subset(left) {
                any = true;
                self.apply(edge, forward);
            }
        }
        Joined { rows: self.finish(), any }
    }
}

/// Why a tree isn't a valid plan for its graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Invalid {
    /// It has no nodes, a node refers to one that isn't before it, a node
    /// isn't used exactly once, or the root isn't last.
    Shape,
    /// A relation is missing, repeated, or not in the graph.
    Relations,
    /// This edge's relations first meet at a join that can't apply it,
    /// because the join's sides don't split the edge's sides.
    Edge(u32),
    /// A join has no edge, though the graph is connected.
    CrossProduct,
}

/// A tree's cost, and whether it's valid.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Costed {
    /// `C_out`: the rows of every join, each at least 1, added up, at most
    /// `ROWS_MAX`.
    pub cost: f64,
    /// The rows the root makes.
    pub rows: f64,
    /// Whether the tree is valid, or why not.
    pub valid: Result<(), Invalid>,
}

/// The cost of `tree` for `graph`, and whether it's valid. A valid tree has
/// every relation once, applies every edge where its relations meet, and has
/// no cross product unless the graph isn't connected. A tree whose shape
/// isn't valid has no cost.
pub fn cost(
    allocator: &dyn Allocator,
    graph: &QueryGraph,
    tree: &JoinTree,
) -> Result<Costed, Error> {
    let invalid = |why| Costed { cost: ROWS_MAX, rows: ROWS_MAX, valid: Err(why) };
    let connected = graph.is_connected(allocator)?;
    let mut estimator = Estimator::new(allocator, graph)?;
    // Each node's relations and rows. A node's relations move to its parent.
    let mut sets = SlowVec::fixed(allocator, tree.nodes.len())?;
    let mut rows = SlowVec::fixed(allocator, tree.nodes.len())?;
    let mut used = RelationSet::new(allocator)?;
    let mut seen = RelationSet::new(allocator)?;
    let mut total = 0.0_f64;
    let (mut valid, mut root) = (Ok(()), 0);
    for (index, node) in (0_u32..).zip(tree.nodes.iter()) {
        root = index;
        let (set, count) = match *node {
            JoinNode::Relation(relation) => {
                let Some(found) = graph.relations.get(relation as usize) else {
                    return Ok(invalid(Invalid::Relations));
                };
                if seen.contains(relation) {
                    return Ok(invalid(Invalid::Relations));
                }
                seen.insert(relation)?;
                let mut set = RelationSet::new(allocator)?;
                set.insert(relation)?;
                #[expect(clippy::cast_precision_loss, reason = "an estimate")]
                let rows = found.rows as f64;
                (set, rows)
            }
            JoinNode::Join { left, right } => {
                if left >= index || right >= index || left == right {
                    return Ok(invalid(Invalid::Shape));
                }
                for child in [left, right] {
                    if used.contains(child) {
                        return Ok(invalid(Invalid::Shape));
                    }
                    used.insert(child)?;
                }
                let mut set =
                    core::mem::replace(at_mut!(sets, left as usize), RelationSet::new(allocator)?);
                let other = at!(sets, right as usize);
                let joined = estimator.join(
                    &set,
                    *at!(rows, left as usize),
                    other,
                    *at!(rows, right as usize),
                );
                if valid.is_ok() {
                    valid = check_join(graph, &set, other);
                    if valid.is_ok() && !joined.any && connected {
                        valid = Err(Invalid::CrossProduct);
                    }
                }
                set.union_with(other)?;
                *at_mut!(sets, right as usize) = RelationSet::new(allocator)?;
                total = (total + joined.rows.max(1.0)).min(ROWS_MAX);
                (set, joined.rows)
            }
        };
        sets.push(set)?;
        rows.push(count)?;
    }
    let Some(&root_rows) = rows.last() else { return Ok(invalid(Invalid::Shape)) };
    if used.len() as usize + 1 != tree.nodes.len() || used.contains(root) {
        return Ok(invalid(Invalid::Shape));
    }
    if seen.len() as usize != graph.relations.len() {
        return Ok(invalid(Invalid::Relations));
    }
    Ok(Costed { cost: total, rows: root_rows, valid })
}

/// Whether joining `left` and `right` applies every edge whose relations
/// they bring together, which needs one side of the edge in each.
fn check_join(graph: &QueryGraph, left: &RelationSet, right: &RelationSet) -> Result<(), Invalid> {
    for (index, edge) in (0_u32..).zip(graph.edges.iter()) {
        let meets = edge.left.is_subset_of_union(left, right)
            && edge.right.is_subset_of_union(left, right)
            && !(edge.left.is_subset(left) && edge.right.is_subset(left))
            && !(edge.left.is_subset(right) && edge.right.is_subset(right));
        let applies = (edge.left.is_subset(left) && edge.right.is_subset(right))
            || (edge.left.is_subset(right) && edge.right.is_subset(left));
        if meets && !applies {
            return Err(Invalid::Edge(index));
        }
    }
    Ok(())
}

/// The relations joined left-deep in the order they're written. It's the
/// baseline that enumerators are measured against.
pub fn as_written(allocator: &dyn Allocator, graph: &QueryGraph) -> Result<JoinTree, Error> {
    let count = graph.relations.len();
    let mut tree = JoinTree::new(allocator, count)?;
    let mut root = tree.relation(0)?;
    #[expect(clippy::cast_possible_truncation, reason = "at most `LIST_MAX`")]
    for relation in 1..count as u32 {
        let leaf = tree.relation(relation)?;
        root = tree.join(root, leaf)?;
    }
    Ok(tree)
}

/// Picks the order to join a graph's relations in. One or two relations
/// can be joined only one way, so they skip planning.
pub fn enumerate(allocator: &dyn Allocator, graph: &QueryGraph) -> Result<JoinTree, Error> {
    if graph.relations.len() <= 2 {
        return as_written(allocator, graph);
    }
    planner::plan(allocator, graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;

    fn set(relations: &[u32]) -> RelationSet {
        let mut set = RelationSet::new(&Heap).unwrap();
        for &relation in relations {
            set.insert(relation).unwrap();
        }
        set
    }

    /// A fact of 1000 rows, joined to a dimension of 10 on its key, and
    /// left joined to another of 100, which then joins the first dimension.
    fn star() -> QueryGraph {
        let mut graph = QueryGraph::new(&Heap).unwrap();
        for rows in [1000, 10, 100] {
            graph.relations.push(Relation { rows, unfiltered_rows: rows }).unwrap();
        }
        for domain in [10, 100] {
            graph.equivalences.push(EquivalenceSet { domain }).unwrap();
        }
        let edge = |left, right, kind, equivalence| JoinEdge {
            left: set(left),
            right: set(right),
            kind,
            equivalence,
        };
        graph.edges.push(edge(&[0], &[1], JoinKind::Inner, 0)).unwrap();
        graph.edges.push(edge(&[0, 1], &[2], JoinKind::Left, 1)).unwrap();
        graph.edges.push(edge(&[1], &[2], JoinKind::Inner, 1)).unwrap();
        graph
    }

    #[test]
    fn sets_grow_past_a_word() {
        let mut a = set(&[1, 70, 200]);
        assert_eq!(a.iter().collect::<alloc::vec::Vec<_>>(), [1, 70, 200]);
        assert!(set(&[70]).is_subset(&a) && !a.is_subset(&set(&[1, 70])));
        assert!(set(&[3]).is_subset_of_union(&a, &set(&[3])));
        a.union_with(&set(&[500])).unwrap();
        assert_eq!(a.len(), 4);
        assert!(a.contains(500) && !a.contains(499));
    }

    #[test]
    fn costs_and_checks_trees() {
        let graph = star();
        let tree = as_written(&Heap, &graph).unwrap();
        let costed = cost(&Heap, &graph, &tree).unwrap();
        // 1000 * 10 / 10, then a left join keeping all 1000.
        assert_eq!(costed, Costed { cost: 2000.0, rows: 1000.0, valid: Ok(()) });

        // The left join before the inner one breaks its TES.
        let mut early = JoinTree::new(&Heap, 3).unwrap();
        let (dimension, kept) = (early.relation(1).unwrap(), early.relation(2).unwrap());
        let joined = early.join(dimension, kept).unwrap();
        let fact = early.relation(0).unwrap();
        early.join(joined, fact).unwrap();
        assert_eq!(cost(&Heap, &graph, &early).unwrap().valid, Err(Invalid::Edge(1)));

        let mut cross = JoinTree::new(&Heap, 3).unwrap();
        let (fact, kept) = (cross.relation(0).unwrap(), cross.relation(2).unwrap());
        let joined = cross.join(fact, kept).unwrap();
        let dimension = cross.relation(1).unwrap();
        cross.join(joined, dimension).unwrap();
        assert_eq!(cost(&Heap, &graph, &cross).unwrap().valid, Err(Invalid::CrossProduct));
    }

    #[test]
    fn estimates_saturate() {
        let mut graph = QueryGraph::new(&Heap).unwrap();
        graph.equivalences.push(EquivalenceSet { domain: 2 }).unwrap();
        let mut estimator = Estimator::new(&Heap, &graph).unwrap();
        let joined = estimator.join(&set(&[0]), ROWS_MAX, &set(&[1]), ROWS_MAX);
        assert_eq!(joined, Joined { rows: ROWS_MAX, any: false });
    }
}
