//! The join tree: the planner's answer, the order to join a graph's
//! relations in.

use crate::allocator::AllocError;
use crate::env::Env;
use crate::small_vec::SmallVec;

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

    /// Adds a leaf for `relation`, of `rows` rows, and returns its node.
    pub fn relation(&mut self, relation: u32, rows: f64) -> Result<u32, AllocError> {
        self.push(JoinNode::Relation(relation), rows)
    }

    /// Adds a join of nodes `left` and `right`, of `rows` rows, and returns
    /// its node.
    pub fn join(&mut self, left: u32, right: u32, rows: f64) -> Result<u32, AllocError> {
        self.push(JoinNode::Join { left, right }, rows)
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
        let (a, b) = (tree.relation(1, 10.0).unwrap(), tree.relation(0, 20.0).unwrap());
        assert_eq!(tree.join(a, b, 5.0).unwrap(), 2);
        assert_eq!(at!(tree.nodes, 2).kind, JoinNode::Join { left: 0, right: 1 });
        assert_eq!(at!(tree.nodes, 2).rows.to_bits(), 5.0_f64.to_bits());
    }
}
