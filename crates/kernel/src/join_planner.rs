//! The join planner: it chooses the order a join region's relations are
//! joined in. `docs/join-planner.md` describes its stages.

mod estimate;
mod exact;
mod graph;
mod tree;

pub use estimate::{Estimator, ROWS_MAX};
pub use graph::{
    EquivalenceSet, Executable, JoinEdge, JoinKind, QueryGraph, Relation, RelationSet,
};
pub use tree::{JoinNode, JoinTree, Node};

use crate::error::Error;
use crate::global::Global;

/// Plans the joins of `graph` exactly, if exact search can afford it. `None`
/// keeps the plan as written: for one or two relations, which have one plan,
/// or for a graph exact search can't afford, for now.
pub fn plan(global: &Global, graph: &QueryGraph) -> Result<Option<JoinTree>, Error> {
    if graph.relations.len() <= 2 {
        return Ok(None);
    }
    let mut estimator = Estimator::new(global, graph)?;
    exact::plan(global, graph, &mut estimator)
}
