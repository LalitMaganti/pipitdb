//! Estimates a join's rows from its inputs' rows and the domains of the
//! equivalence sets it joins on. `docs/join-planner.md` gives each rule's
//! source.

use super::graph::{JoinEdge, JoinKind, QueryGraph};
use crate::containers::Array;
use crate::env::Env;
use crate::error::Error;

/// The most rows an estimate reaches.
#[expect(clippy::cast_precision_loss, reason = "a limit, not a count")]
pub const ROWS_MAX: f64 = u64::MAX as f64;

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

/// Estimates the rows of joins, with memory it reuses from one join to the
/// next. A join is estimated by `start`, then `apply` for each edge it
/// applies, then `finish`.
pub struct Estimator<'a> {
    /// The graph whose joins it estimates.
    graph: &'a QueryGraph,
    /// The first 64 equivalence sets the join has divided by, as bits, so
    /// that it divides by each once.
    divided: u64,
    /// For each later equivalence set, the last join (`stamp`) that divided
    /// by it.
    stamps: Array<u64>,
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
    pub fn new(env: &Env, graph: &'a QueryGraph) -> Result<Estimator<'a>, Error> {
        let stamps = Array::zeroed(env, graph.equivalences.len().saturating_sub(64))?;
        let (divided, stamp, rows, sides, kept) = (0, 0, 0.0, [0.0; 2], None);
        Ok(Estimator { graph, divided, stamps, stamp, rows, sides, kept })
    }

    /// Starts estimating a join of sides of `left_rows` and `right_rows`.
    /// Until an edge applies, it's a cross product: left × right.
    #[inline(never)]
    pub fn start(&mut self, left_rows: f64, right_rows: f64) {
        (self.divided, self.stamp) = (0, self.stamp + 1);
        (self.rows, self.sides, self.kept) =
            (left_rows * right_rows, [left_rows, right_rows], None);
    }

    /// Applies `edge` to the join. `forward` is whether the edge's `left` is
    /// on the join's left side.
    pub fn apply(&mut self, edge: &JoinEdge, forward: bool) {
        let domain = at!(self.graph.equivalences, edge.equivalence as usize).domain;
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
        }
    }

    /// The join's rows by the edges applied, at most `ROWS_MAX`.
    pub fn finish(&self) -> f64 {
        let rows = match self.kept {
            None => self.rows,
            Some(Kept { kind: JoinKind::Left, rows, .. }) => self.rows.max(rows),
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
    use crate::join_planner::{EquivalenceSet, RelationSet};

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
        let mut estimator = Estimator::new(Env::current(), &graph).unwrap();
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

    #[test]
    fn estimates_saturate() {
        let graph = graph();
        let mut estimator = Estimator::new(Env::current(), &graph).unwrap();
        estimator.start(ROWS_MAX, ROWS_MAX);
        assert!(estimator.finish() <= ROWS_MAX);
    }
}
