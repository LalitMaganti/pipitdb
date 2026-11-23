//! The join planner: it chooses the order a join region's relations are
//! joined in. `docs/join-planner.md` describes its stages.

mod graph;
mod tree;

pub use graph::{
    EquivalenceSet, Executable, JoinEdge, JoinKind, QueryGraph, Relation, RelationSet,
};
pub use tree::{JoinNode, JoinTree, Node};
