//! The join tree: the planner's answer, the order to join a graph's
//! relations in.

use crate::allocator::AllocError;
use crate::containers::SmallVec;
use crate::env::Env;

/// A node of a join tree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JoinNode {
    /// A relation, by index.
    Relation(u32),
    /// A join of two other nodes, by index.
    Join { left: u32, right: u32 },
}

/// A node of a join tree, and its rows.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Node {
    /// What it is.
    pub kind: JoinNode,
    /// Its estimated rows, which choosing a join's build side needs.
    pub rows: f64,
}

/// A join tree. Its nodes can be in any order, but the root is last.
pub struct JoinTree {
    /// The nodes.
    pub nodes: SmallVec<Node>,
}

impl JoinTree {
    /// An empty tree with room for a join of `relations` relations.
    pub fn new(env: &Env, relations: usize) -> Result<JoinTree, AllocError> {
        Ok(JoinTree { nodes: SmallVec::fixed(env, (2 * relations).saturating_sub(1))? })
    }

    /// Adds a node to a tree that has room for it, and returns it.
    pub(super) fn add(&mut self, kind: JoinNode, rows: f64) -> u32 {
        match self.push(kind, rows) {
            Ok(index) => index,
            Err(_) => failed!(),
        }
    }

    fn push(&mut self, kind: JoinNode, rows: f64) -> Result<u32, AllocError> {
        #[expect(clippy::cast_possible_truncation, reason = "at most a `SmallVec`'s length")]
        let index = self.nodes.len() as u32;
        self.nodes.push(Node { kind, rows })?;
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Env;

    #[test]
    fn trees_number_their_nodes() {
        let mut tree = JoinTree::new(Env::current(), 2).unwrap();
        let (a, b) = (tree.add(JoinNode::Relation(1), 10.0), tree.add(JoinNode::Relation(0), 20.0));
        assert_eq!(tree.add(JoinNode::Join { left: a, right: b }, 5.0), 2);
        assert_eq!(at!(tree.nodes, 2).kind, JoinNode::Join { left: 0, right: 1 });
        assert_eq!(at!(tree.nodes, 2).rows.to_bits(), 5.0_f64.to_bits());
    }
}
