//! Estimates a join's rows from its inputs' rows and the domains of the
//! equivalence sets it joins on. `docs/join-planner.md` gives each rule's
//! source.

use super::graph::{EquivalenceSet, JoinEdge};

/// The most rows an estimate reaches.
#[expect(clippy::cast_precision_loss, reason = "a limit, not a count")]
pub const ROWS_MAX: f64 = u64::MAX as f64;

/// Estimates the rows of joins, with memory it's given and reuses from one
/// join to the next. A join is estimated by `start`, then `apply` for each
/// edge it applies, then `finish`.
pub struct Estimator<'a> {
    /// The equivalence sets of the graph whose joins it estimates.
    equivalences: &'a [EquivalenceSet],
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

    /// Estimates joins on `equivalences`, with `stamps`, as many as
    /// `Estimator::stamps` says.
    pub fn new(equivalences: &'a [EquivalenceSet], stamps: &'a mut [u64]) -> Estimator<'a> {
        stamps.fill(0);
        Estimator { equivalences, divided: 0, stamps, stamp: 0, rows: 0.0 }
    }

    /// Starts estimating a join of sides of `left_rows` and `right_rows`.
    /// Until an edge applies, it's a cross product: left × right.
    #[inline(never)]
    pub fn start(&mut self, left_rows: f64, right_rows: f64) {
        (self.divided, self.stamp) = (0, self.stamp + 1);
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
        }
    }

    /// The join's rows by the edges applied, at most `ROWS_MAX`.
    pub fn finish(&self) -> f64 {
        self.rows.min(ROWS_MAX)
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
        let mut estimator = Estimator::new(&graph.equivalences, &mut []);
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

    #[test]
    fn estimates_saturate() {
        let graph = graph();
        let mut estimator = Estimator::new(&graph.equivalences, &mut []);
        estimator.start(ROWS_MAX, ROWS_MAX);
        assert!(estimator.finish() <= ROWS_MAX);
    }
}
