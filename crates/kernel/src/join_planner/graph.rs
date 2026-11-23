//! The query graph: a join region's relations, and the joins between them as
//! edges.

use crate::allocator::AllocError;
use crate::containers::{BitSlice, SmallVec, Word};
use crate::env::Env;

/// The most relations, edges or equivalence sets a graph holds.
pub(super) const LIST_MAX: usize = 1 << 24;

/// The most words a `RelationSet` grows to, enough for `LIST_MAX` relations.
const WORDS_MAX: usize = LIST_MAX / 64;

/// A set of relations, by index. The first 64 relations take one word,
/// and the set grows a word at a time past them.
pub struct RelationSet {
    /// Relations 0 to 63.
    first: u64,
    /// Relations 64 and up, 64 to a word. Words past the end are empty.
    rest: SmallVec<u64>,
}

impl RelationSet {
    /// An empty set. It allocates nothing until it holds a relation past 63.
    pub fn new(env: &Env) -> Result<RelationSet, AllocError> {
        Ok(RelationSet { first: 0, rest: SmallVec::new(env, WORDS_MAX)? })
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
        let bits = self.word_mut(word);
        *bits = bits.or(relation as usize % 64, true);
        Ok(())
    }

    pub fn contains(&self, relation: u32) -> bool {
        self.word(relation as usize / 64).bit(relation as usize % 64)
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

    /// How many relations it has.
    pub fn count(&self) -> u32 {
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

    /// Whether it has any relation of `set`, given as words, and whether all
    /// its relations are in `set`.
    pub fn meets(&self, set: &[u64]) -> (bool, bool) {
        let (mut some, mut all) = (false, true);
        for (i, set) in set.iter().enumerate() {
            let word = self.word(i);
            some |= word & set != 0;
            all &= word & !set == 0;
        }
        (some, all)
    }

    fn word_mut(&mut self, i: usize) -> &mut u64 {
        match i {
            0 => &mut self.first,
            _ => at_mut!(self.rest, i - 1),
        }
    }
}

impl PartialEq for RelationSet {
    fn eq(&self, other: &RelationSet) -> bool {
        (0..self.words().max(other.words())).all(|i| self.word(i) == other.word(i))
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
    /// The most rows it can have, guaranteed, unlike `rows`. A relation of
    /// at most one row may be cross-joined to anything.
    pub max_rows: u64,
}

/// Columns that are joined by equality, so that they all hold the same
/// values.
#[derive(Clone, Copy, Debug)]
pub struct EquivalenceSet {
    /// How many distinct values they can take. For a unique key, it's that
    /// relation's unfiltered rows. Without distinct counts, it's the fewest
    /// rows of their relations.
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
    /// non-inner join, the left of its total eligibility set: the side whose
    /// rows it keeps.
    pub left: RelationSet,
    /// The same for the other side.
    pub right: RelationSet,
    /// Which rows the join returns.
    pub kind: JoinKind,
    /// The equivalence set it joins on, by index.
    pub equivalence: u32,
}

/// Sets node `slot`'s relations in `sets`, `words` words a node, to those
/// of nodes `left` and `right`.
#[inline(never)]
pub(super) fn union(sets: &mut [u64], words: usize, slot: u32, [left, right]: [u32; 2]) {
    let (left, right, slot) =
        (left as usize * words, right as usize * words, slot as usize * words);
    for i in 0..words {
        let word = *at!(sets, left + i) | *at!(sets, right + i);
        *at_mut!(sets, slot + i) = word;
    }
}

/// What a planner keeps of an edge to tell quickly which sets hold its sides.
#[derive(Clone, Copy)]
pub(super) struct Sides {
    /// The first relation of its left side, then of its right.
    pub(super) first: [u32; 2],
    /// Whether its left side, then its right, has more than one relation.
    pub(super) more: [bool; 2],
}

impl Sides {
    #[inline(never)]
    pub(super) fn of(edge: &JoinEdge) -> Sides {
        let mut sides = Sides { first: [u32::MAX; 2], more: [false; 2] };
        for (i, side) in [&edge.left, &edge.right].into_iter().enumerate() {
            let mut count = 0;
            for w in (0..side.words()).rev() {
                let word = side.word(w);
                if word != 0 {
                    #[expect(clippy::cast_possible_truncation, reason = "at most `LIST_MAX`")]
                    let relation = w as u32 * 64 + word.trailing_zeros();
                    *at_mut!(sides.first, i) = relation;
                }
                count += word.count_ones();
            }
            *at_mut!(sides.more, i) = count > 1;
        }
        sides
    }

    /// Whether side `right` of `edge`, whose sides these are, has any
    /// relation of `set`, and whether all its relations are in `set`. A side
    /// of one relation is answered from that relation alone.
    pub(super) fn meets(self, edge: &JoinEdge, right: usize, set: &[u64]) -> (bool, bool) {
        if *at!(self.more, right) {
            return if right == 0 { &edge.left } else { &edge.right }.meets(set);
        }
        let relation = *at!(self.first, right) as usize;
        let has = BitSlice::of(set).get(relation);
        (has, has)
    }
}

/// Whether the join operator can run a join of two sets of relations, each
/// given as the words of a `RelationSet`. The optimizer passes it in, for
/// joins the operator can't run yet.
pub type Executable = fn(&QueryGraph, &[u64], &[u64]) -> bool;

/// The relations of a join region and the joins between them.
pub struct QueryGraph {
    /// The relations, by index.
    pub relations: SmallVec<Relation>,
    /// The joins between them.
    pub edges: SmallVec<JoinEdge>,
    /// The equivalence sets the joins join on, by index.
    pub equivalences: SmallVec<EquivalenceSet>,
    /// Which joins can run, if not all of them.
    pub executable: Option<Executable>,
}

impl QueryGraph {
    /// An empty graph whose joins can all run.
    pub fn new(env: &Env) -> Result<QueryGraph, AllocError> {
        Ok(QueryGraph {
            relations: SmallVec::new(env, LIST_MAX)?,
            edges: SmallVec::new(env, LIST_MAX)?,
            equivalences: SmallVec::new(env, LIST_MAX)?,
            executable: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Env;

    fn set(relations: &[u32]) -> RelationSet {
        let mut set = RelationSet::new(Env::current()).unwrap();
        for &relation in relations {
            set.insert(relation).unwrap();
        }
        set
    }

    #[test]
    fn sets_grow_past_a_word() {
        let mut a = set(&[1, 70, 200]);
        assert_eq!((a.words(), a.word(1), a.word(2)), (4, 1 << 6, 0));
        assert!(set(&[70]).is_subset(&a) && !a.is_subset(&set(&[1, 70])));
        assert!(set(&[3]).is_subset_of_union(&a, &set(&[3])));
        assert!(a.overlaps(&set(&[200])) && !a.overlaps(&set(&[2])));
        a.union_with(&set(&[500])).unwrap();
        assert_eq!(a.count(), 4);
        assert!(a.contains(500) && !a.contains(499));
        let b = set(&[1, 70]);
        assert!(b.meets(&[2, 0]) == (true, false) && b.meets(&[2, 1 << 6]) == (true, true));
        assert!(set(&[3]) == set(&[3]) && set(&[3]) != set(&[3, 64]));
    }
}
