//! The join planner: it chooses the order a join region's relations are
//! joined in. `docs/join-planner.md` describes its stages.

mod estimate;
mod exact;
mod graph;
mod greedy;
mod region;
mod tree;
mod windows;

pub use estimate::{Estimator, ROWS_MAX};
pub use graph::{
    EquivalenceSet, Executable, JoinEdge, JoinKind, QueryGraph, Relation, RelationSet,
};
pub(crate) use region::plan_regions;
pub use tree::{JoinNode, JoinTree, Node};

use crate::env::Env;
use crate::error::Error;

/// The most work greedy ordering and windows do, in edges read. The slowest
/// plan, of 1,000 relations, takes about 1 s; a soft target.
const WORK_MAX: usize = 1 << 27;

/// What a join's hash table costs for each row of its smaller side, where
/// each row the join makes costs 1. Building a row costs 8 to 15 times what
/// probing one does on x86-64 and Apple CPUs alike, as
/// `crates/bench/build_cost.py` measures; plans of TPC-H and TPC-DS are the
/// same for any value from 1 to 64.
pub const BUILD_COST: f64 = 10.0;

/// Plans the joins of `graph`: exactly if exact search can afford it, or
/// else greedily, improved by windows. `None` keeps the plan as written: for
/// one or two relations, which have one plan, or if greedy ordering is
/// stuck, which hyperedges can cause.
pub fn plan(env: &Env, graph: &QueryGraph) -> Result<Option<JoinTree>, Error> {
    if graph.relations.len() <= 2 {
        return Ok(None);
    }
    let mut estimator = Estimator::new(env, graph)?;
    if let Some(tree) = exact::plan(env, graph, &mut estimator)? {
        return Ok(Some(tree));
    }
    let mut work = WORK_MAX;
    let Some(mut tree) = greedy::plan(env, graph, &mut estimator, &mut work)? else {
        return Ok(None);
    };
    windows::improve(env, graph, &mut estimator, &mut tree, &mut work)?;
    Ok(Some(tree))
}
