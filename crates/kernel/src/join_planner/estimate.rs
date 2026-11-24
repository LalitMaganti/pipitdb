//! Estimates a join's rows from its inputs' rows and the domains of the
//! equivalence sets it joins on. `docs/join-planner.md` gives each rule's
//! source.

use super::graph::{EquivalenceSet, JoinEdge, QueryGraph, Relation, RelationSet};

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
    /// The join's rows by the edges applied so far.
    rows: f64,
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
        let (divided, stamp, rows) = (0, 0, 0.0);
        let (equivalences, relations) = (&*graph.equivalences, &*graph.relations);
        let pair = Pair { relations: [0; 2], product: 1.0, largest: 0.0, count: 0 };
        let (pairs, pairs_len) = ([pair; PAIRS_MAX], 0);
        Estimator { equivalences, relations, pairs, pairs_len, divided, stamps, stamp, rows }
    }

    /// Starts estimating a join of sides of `left_rows` and `right_rows`.
    /// Until an edge applies, it's a cross product: left × right.
    #[inline(never)]
    pub fn start(&mut self, left_rows: f64, right_rows: f64) {
        (self.divided, self.stamp, self.pairs_len) = (0, self.stamp + 1, 0);
        self.rows = left_rows * right_rows;
    }

    /// Applies `edge` to the join.
    pub fn apply(&mut self, edge: &JoinEdge) {
        let domain = at!(self.equivalences, edge.equivalence as usize).domain;
        #[expect(clippy::cast_precision_loss, reason = "an estimate")]
        let domain = domain.max(1) as f64;
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
            let alone = set.word(0).is_power_of_two() && (1..set.words()).all(|i| set.word(i) == 0);
            alone.then(|| set.word(0).trailing_zeros())
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
        let mut rows = self.rows;
        for pair in at!(self.pairs, ..self.pairs_len).iter().filter(|pair| pair.count > 1) {
            let [a, b] = pair.relations.map(|r| at!(self.relations, r as usize).max_rows);
            #[expect(clippy::cast_precision_loss, reason = "an estimate")]
            let cap = a.min(b).max(1) as f64;
            if cap <= pair.largest * CAP_RATIO_MAX {
                rows = rows * pair.product / cap;
            }
        }
        rows.min(ROWS_MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Env;
    use crate::join_planner::{QueryGraph, RelationSet};

    /// A graph with equivalence sets of domains 10 and 100, and an edge on
    /// each.
    fn graph() -> QueryGraph {
        let mut graph = QueryGraph::new(Env::current()).unwrap();
        for domain in [10, 100] {
            graph.equivalences.push(EquivalenceSet { domain }).unwrap();
        }
        for equivalence in [0, 1] {
            let (left, right) = (
                RelationSet::new(Env::current()).unwrap(),
                RelationSet::new(Env::current()).unwrap(),
            );
            graph.edges.push(JoinEdge { left, right, equivalence }).unwrap();
        }
        graph
    }

    /// The rows of joining 1000 rows to 4, by edges `edges` of `graph()`.
    fn rows(edges: &[usize]) -> f64 {
        let graph = graph();
        let mut estimator = Estimator::new(&graph, &mut []);
        estimator.start(1000.0, 4.0);
        for &edge in edges {
            estimator.apply(at!(graph.edges, edge));
        }
        estimator.finish()
    }

    #[test]
    fn inner_joins_divide_by_each_domain_once() {
        assert!((rows(&[]) - 4000.0).abs() < 1e-9);
        assert!((rows(&[0, 0]) - 400.0).abs() < 1e-9);
        assert!((rows(&[0, 1]) - 4.0).abs() < 1e-9);
    }

    /// The rows of joining relations of 6000 and 800 rows on two keys, of
    /// domains `domains`.
    fn composite(domains: [u64; 2]) -> f64 {
        let env = Env::current();
        let mut graph = QueryGraph::new(env).unwrap();
        for rows in [6000, 800] {
            let relation = Relation { rows, max_rows: rows };
            graph.relations.push(relation).unwrap();
        }
        for (equivalence, domain) in (0..).zip(domains) {
            graph.equivalences.push(EquivalenceSet { domain }).unwrap();
            let (mut left, mut right) =
                (RelationSet::new(env).unwrap(), RelationSet::new(env).unwrap());
            left.insert(0).unwrap();
            right.insert(1).unwrap();
            graph.edges.push(JoinEdge { left, right, equivalence }).unwrap();
        }
        let mut estimator = Estimator::new(&graph, &mut []);
        estimator.start(6000.0, 800.0);
        for edge in graph.edges.iter() {
            estimator.apply(edge);
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
