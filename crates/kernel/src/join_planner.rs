//! The join planner: it chooses the order a join region's relations are
//! joined in. `docs/join-planner.md` describes its stages.

mod estimate;
mod exact;
mod graph;
mod greedy;
mod tree;

pub use estimate::{Estimator, ROWS_MAX};
pub use graph::{
    EquivalenceSet, Executable, JoinEdge, JoinKind, QueryGraph, Relation, RelationSet,
};
pub use tree::{JoinNode, JoinTree, Node};

use crate::error::Error;
use crate::global::Global;

/// The most work greedy ordering does, in edges read. The slowest plan, of
/// 1,000 relations, takes about 1 s; a soft target.
const WORK_MAX: usize = 1 << 27;

/// Plans the joins of `graph`: exactly if exact search can afford it, or
/// else greedily. `None` keeps the plan as written: for one or two
/// relations, which have one plan, or if greedy ordering is stuck, which
/// hyperedges can cause.
pub fn plan(global: &Global, graph: &QueryGraph) -> Result<Option<JoinTree>, Error> {
    if graph.relations.len() <= 2 {
        return Ok(None);
    }
    let mut estimator = Estimator::new(global, graph)?;
    if let Some(tree) = exact::plan(global, graph, &mut estimator)? {
        return Ok(Some(tree));
    }
    let mut work = WORK_MAX;
    greedy::plan(global, graph, &mut estimator, &mut work)
}
