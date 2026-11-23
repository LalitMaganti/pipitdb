//! The join planner: it chooses the order a join region's relations are
//! joined in. `docs/join-planner.md` describes its stages.

mod estimate;
mod graph;
mod tree;

pub use estimate::{Estimator, ROWS_MAX};
pub use graph::{
    EquivalenceSet, Executable, JoinEdge, JoinKind, QueryGraph, Relation, RelationSet,
};
pub use tree::{JoinNode, JoinTree, Node};
