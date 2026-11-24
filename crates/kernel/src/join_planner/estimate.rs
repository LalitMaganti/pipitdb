//! Estimates a join's rows from its inputs' rows and the domains of the
//! equivalence sets it joins on. `docs/join-planner.md` gives each rule's
//! source.

use super::graph::{EquivalenceSet, JoinEdge, JoinKind, QueryGraph, Relation, RelationSet};

/// The most rows an estimate reaches.
#[expect(clippy::cast_precision_loss, reason = "a limit, not a count")]
pub const ROWS_MAX: f64 = u64::MAX as f64;

/// The most pairs of relations a join's estimate tracks for composite keys.
const PAIRS_MAX: usize = 8;

/// How many times its largest domain a composite key's cap can be: past it,
/// the join is taken to be between broad tables, not on a key.
const CAP_RATIO_MAX: f64 = 8.0;

/// The domains a join divided by, of edges between one pair of relations.
#[derive(Clone, Copy)]
struct Pair {
    /// The relations, the lesser first.
    relations: [u32; 2],
    /// The product of the domains.
    product: f64,
    /// The largest of them.
    largest: f64,
    /// How many there are.
    count: u32,
}

/// A non-inner edge that applies to the join being estimated.
#[derive(Clone, Copy)]
struct Kept {
    /// Which rows it returns.
    kind: JoinKind,
    /// The rows of the side whose rows it keeps.
    rows: f64,
    /// The share of those rows that match: the other side's distinct key
    /// values over the domain, at most 1.
    matched: f64,
}

/// Estimates the rows of joins, with memory it's given and reuses from one
/// join to the next. A join is estimated by `start`, then `apply` for each
/// edge it applies, then `finish`.
pub struct Estimator<'a> {
    /// The equivalence sets of the graph whose joins it estimates.
    equivalences: &'a [EquivalenceSet],
    /// The graph's relations.
    relations: &'a [Relation],
    /// The first `pairs_len` are the pairs of relations the join's edges
    /// joined, by the domains each divided by.
    pairs: [Pair; PAIRS_MAX],
    /// How many pairs there are.
    pairs_len: usize,
    /// The first 64 equivalence sets the join has divided by, as bits, so
    /// that it divides by each once.
    divided: u64,
    /// For each later equivalence set, the last join (`stamp`) that divided
    /// by it.
    stamps: &'a mut [u64],
    /// The join being estimated, numbered from 1.
    stamp: u64,
    /// The join's rows as an inner join, by the edges applied so far.
    rows: f64,
    /// The rows of its left and right sides.
    sides: [f64; 2],
    /// The non-inner edge that applies, if any.
    kept: Option<Kept>,
}

impl<'a> Estimator<'a> {
    /// How many stamps an estimator of joins on `equivalences` equivalence
    /// sets needs.
    pub fn stamps(equivalences: usize) -> usize {
        equivalences.saturating_sub(64)
    }

    /// Estimates joins of `graph`, with `stamps`, as many as
    /// `Estimator::stamps` says.
    pub fn new(graph: &'a QueryGraph, stamps: &'a mut [u64]) -> Estimator<'a> {
        stamps.fill(0);
        let (divided, stamp, rows, sides, kept) = (0, 0, 0.0, [0.0; 2], None);
        let (equivalences, relations) = (&*graph.equivalences, &*graph.relations);
        let pair = Pair { relations: [0; 2], product: 1.0, largest: 0.0, count: 0 };
        let (pairs, pairs_len) = ([pair; PAIRS_MAX], 0);
        Estimator {
            equivalences,
            relations,
            pairs,
            pairs_len,
            divided,
            stamps,
            stamp,
            rows,
            sides,
            kept,
        }
    }

    /// Starts estimating a join of sides of `left_rows` and `right_rows`.
    /// Until an edge applies, it's a cross product: left × right.
    #[inline(never)]
    pub fn start(&mut self, left_rows: f64, right_rows: f64) {
        (self.divided, self.stamp, self.pairs_len) = (0, self.stamp + 1, 0);
        (self.rows, self.sides, self.kept) =
            (left_rows * right_rows, [left_rows, right_rows], None);
    }

    /// Applies `edge` to the join. `forward` is whether the edge's `left` is
    /// on the join's left side.
    pub fn apply(&mut self, edge: &JoinEdge, forward: bool) {
        let domain = at!(self.equivalences, edge.equivalence as usize).domain;
        #[expect(clippy::cast_precision_loss, reason = "an estimate")]
        let domain = domain.max(1) as f64;
        if edge.kind != JoinKind::Inner {
            let [left, right] = self.sides;
            let (rows, other) = if forward { (left, right) } else { (right, left) };
            // The other side has at most as many distinct key values as rows.
            let matched = other.min(domain) / domain;
            self.kept = Some(Kept { kind: edge.kind, rows, matched });
        }
        let set = edge.equivalence as usize;
        let fresh = if let Some(at) = set.checked_sub(64) {
            let stamp = at_mut!(self.stamps, at);
            core::mem::replace(stamp, self.stamp) != self.stamp
        } else {
            let fresh = self.divided & 1 << set == 0;
            self.divided |= 1 << set;
            fresh
        };
        if fresh {
            self.rows /= domain;
            self.add_to_pair(edge, domain);
        }
    }

    /// Records that the join divided by `domain` for `edge`, if it's between
    /// two single relations.
    fn add_to_pair(&mut self, edge: &JoinEdge, domain: f64) {
        let single = |set: &RelationSet| {
            (set.count() == 1 && set.word(0) != 0).then(|| set.word(0).trailing_zeros())
        };
        let (Some(a), Some(b)) = (single(&edge.left), single(&edge.right)) else { return };
        let relations = [a.min(b), a.max(b)];
        let pairs = at_mut!(self.pairs, ..self.pairs_len);
        if let Some(pair) = pairs.iter_mut().find(|pair| pair.relations == relations) {
            (pair.product, pair.largest) = (pair.product * domain, pair.largest.max(domain));
            pair.count += 1;
        } else if self.pairs_len < PAIRS_MAX {
            *at_mut!(self.pairs, self.pairs_len) =
                Pair { relations, product: domain, largest: domain, count: 1 };
            self.pairs_len += 1;
        }
    }

    /// The join's rows by the edges applied, at most `ROWS_MAX`.
    pub fn finish(&self) -> f64 {
        // Equalities between the same two relations are a composite key,
        // whose values aren't independent: they divide once, by the smaller
        // relation's rows, as DuckDB's `ApplyJoinPairCap` has it, unless that's
        // far more than any one of them could be.
        let mut inner = self.rows;
        for pair in at!(self.pairs, ..self.pairs_len).iter().filter(|pair| pair.count > 1) {
            let [a, b] = pair.relations.map(|r| at!(self.relations, r as usize).unfiltered_rows);
            #[expect(clippy::cast_precision_loss, reason = "an estimate")]
            let cap = a.min(b).max(1) as f64;
            if cap <= pair.largest * CAP_RATIO_MAX {
                inner = inner * pair.product / cap;
            }
        }
        let rows = match self.kept {
            None => inner,
            Some(Kept { kind: JoinKind::Left, rows, .. }) => inner.max(rows),
            Some(Kept { kind: JoinKind::Semi, rows, matched }) => rows * matched,
            Some(Kept { rows, matched, .. }) => rows * (1.0 - matched),
        };
        rows.min(ROWS_MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Env;
    use crate::join_planner::{QueryGraph, RelationSet};

    /// A graph with equivalence sets of domains 10 and 100, and an edge of
    /// each kind on the first.
    fn graph() -> QueryGraph {
        let mut graph = QueryGraph::new(Env::current()).unwrap();
        for domain in [10, 100] {
            graph.equivalences.push(EquivalenceSet { domain }).unwrap();
        }
        for kind in [JoinKind::Inner, JoinKind::Left, JoinKind::Semi, JoinKind::Anti] {
            let (left, right) = (
                RelationSet::new(Env::current()).unwrap(),
                RelationSet::new(Env::current()).unwrap(),
            );
            graph.edges.push(JoinEdge { left, right, kind, equivalence: 0 }).unwrap();
        }
        let (left, right) =
            (RelationSet::new(Env::current()).unwrap(), RelationSet::new(Env::current()).unwrap());
        graph.edges.push(JoinEdge { left, right, kind: JoinKind::Inner, equivalence: 1 }).unwrap();
        graph
    }

    /// The rows of joining 1000 rows to 4, by edges `edges` of `graph()`.
    fn rows(edges: &[usize], forward: bool) -> f64 {
        let graph = graph();
        let mut estimator = Estimator::new(&graph, &mut []);
        estimator.start(1000.0, 4.0);
        for &edge in edges {
            estimator.apply(at!(graph.edges, edge), forward);
        }
        estimator.finish()
    }

    #[test]
    fn inner_joins_divide_by_each_domain_once() {
        assert!((rows(&[], true) - 4000.0).abs() < 1e-9);
        assert!((rows(&[0, 0], true) - 400.0).abs() < 1e-9);
        assert!((rows(&[0, 4], true) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn outer_semi_and_anti_joins_keep_a_side() {
        // A left join keeps all 1000 left rows, more than the 400 that match.
        assert!((rows(&[1], true) - 1000.0).abs() < 1e-9);
        // The 4 right rows hold at most 4 of the 10 values: 40% match.
        assert!((rows(&[2], true) - 400.0).abs() < 1e-9);
        assert!((rows(&[3], true) - 600.0).abs() < 1e-9);
        // Keeping the right side's 4 rows: the 1000 left rows hold every value.
        assert!((rows(&[2], false) - 4.0).abs() < 1e-9);
        assert!(rows(&[3], false).abs() < 1e-9);
    }

    /// The rows of joining relations of 6000 and 800 rows on two keys, of
    /// domains `domains`.
    fn composite(domains: [u64; 2]) -> f64 {
        let env = Env::current();
        let mut graph = QueryGraph::new(env).unwrap();
        for rows in [6000, 800] {
            let relation = Relation { rows, unfiltered_rows: rows, max_rows: rows };
            graph.relations.push(relation).unwrap();
        }
        for (equivalence, domain) in (0..).zip(domains) {
            graph.equivalences.push(EquivalenceSet { domain }).unwrap();
            let (mut left, mut right) =
                (RelationSet::new(env).unwrap(), RelationSet::new(env).unwrap());
            left.insert(0).unwrap();
            right.insert(1).unwrap();
            graph.edges.push(JoinEdge { left, right, kind: JoinKind::Inner, equivalence }).unwrap();
        }
        let mut estimator = Estimator::new(&graph, &mut []);
        estimator.start(6000.0, 800.0);
        for edge in graph.edges.iter() {
            estimator.apply(edge, true);
        }
        estimator.finish()
    }

    #[test]
    fn composite_keys_divide_by_the_smaller_relation() {
        // As a key, the 800 rows' pairs of values: each of the 6000 matches
        // one, not one in 200 × 10 = 2000 of them.
        assert!((composite([200, 10]) - 6000.0).abs() < 1e-9);
        // Past 8 times the larger domain, the keys are taken as independent.
        assert!((composite([50, 2]) - 6000.0 * 800.0 / 100.0).abs() < 1e-9);
    }

    #[test]
    fn estimates_saturate() {
        let graph = graph();
        let mut estimator = Estimator::new(&graph, &mut []);
        estimator.start(ROWS_MAX, ROWS_MAX);
        assert!(estimator.finish() <= ROWS_MAX);
    }
}
