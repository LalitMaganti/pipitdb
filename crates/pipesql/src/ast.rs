//! The syntax tree: every node is 8 bytes, in one array.
//!
//! A node's children sit next to each other, so a node only stores the
//! index of its first child. Leaves store their token's span instead. See
//! <https://jhwlr.io/super-flat-ast/>.

#![allow(unsafe_code)]

use crate::error::{ErrorCode, Span};
use crate::settings::build_setting;
use core::ptr::NonNull;

use pipit_kernel::allocator::AllocError;
use pipit_kernel::buffer::Primitive;
use pipit_kernel::containers::{Array, SmallVec};
use pipit_kernel::env::Env;

/// Leaves come first, so `tag < Unary` tells them apart.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Tag {
    /// A column or function name. A leaf.
    Name,
    /// An integer literal. A leaf.
    Integer,
    /// A float literal. A leaf.
    Float,
    /// A string literal. A leaf.
    String,
    /// `*`, as in `count(*)`. A leaf.
    Star,
    /// `operator` applied to the child at `first_child`.
    Unary,
    /// `operator` applied to the children at `first_child` and `first_child + 1`.
    Binary,
    /// A function call: `child_count` children from `first_child`, the name and
    /// then the arguments.
    Call,
    /// `CASE`: `child_count` children from `first_child`: its subject, then
    /// each `WHEN`'s condition and `THEN`'s value, then its `ELSE`'s value.
    /// A subject or `ELSE` it hasn't is an empty node.
    Case,
    /// `child_count` items from `first_child`, as in `SELECT a, b`.
    List,
    /// An expression and a name for it, a `Name`, from `first_child`: `a AS
    /// b` in `SELECT` and `RENAME`, and `b = a` in `SET`.
    Named,
    /// What rows are sorted by: the expression at `first_child`, and whether
    /// it descends.
    SortKey,
    /// A stage of a query: one child per item of its rule, from
    /// `first_child`. Its rule is `rule()`, an id in the parser's registry.
    Stage,
    /// A query: `child_count` stages from `first_child`.
    Query,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Operator {
    /// `-a`.
    Negate,
    /// `NOT a`.
    Not,
    /// `a OR b`.
    Or,
    /// `a AND b`.
    And,
    /// `a = b`.
    Equal,
    /// `a != b` or `a <> b`.
    NotEqual,
    /// `a < b`.
    Less,
    /// `a <= b`.
    LessEqual,
    /// `a > b`.
    Greater,
    /// `a >= b`.
    GreaterEqual,
    /// `a + b`.
    Add,
    /// `a - b`.
    Subtract,
    /// `a * b`.
    Multiply,
    /// `a / b`.
    Divide,
    /// `a % b`.
    Modulo,
    /// `a LIKE b`.
    Like,
    /// `a IS NULL`.
    IsNull,
    /// `a IS NOT NULL`.
    IsNotNull,
    /// `a IN (1, 2)`: its second child is a `List` of literals.
    In,
    /// `DISTINCT a`, as an aggregate's argument.
    Distinct,
}

/// `data` is the operator for `Unary` and `Binary`, the child count for `Call`,
/// and the span's length for leaves. `payload` is the first child's index, or
/// the span's start.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct Node {
    tag: u8,
    data: [u8; 3],
    payload: u32,
}

const _: () = assert!(size_of::<Node>() == 8);

// SAFETY: plain integers, so any bit pattern is a valid `Node`. `tag` and
// `operator` check their values before converting.
unsafe impl Primitive for Node {}

/// The size of each block of memory holding the tree's nodes. Set at build
/// time with `PIPIT_AST_BLOCK_BYTES`.
pub const BLOCK_BYTES: usize = build_setting(option_env!("PIPIT_AST_BLOCK_BYTES"), 64 * 1024);

/// How many nodes a block holds.
pub const BLOCK_NODES: usize = BLOCK_BYTES / size_of::<Node>();

/// How many blocks a tree can use. Set at build time with
/// `PIPIT_AST_BLOCK_SLOTS`.
pub const BLOCK_SLOTS: usize = build_setting(option_env!("PIPIT_AST_BLOCK_SLOTS"), 256);

const _: () =
    assert!(BLOCK_BYTES.is_power_of_two(), "PIPIT_AST_BLOCK_BYTES must be a power of two");
const _: () = assert!(BLOCK_NODES >= 2, "PIPIT_AST_BLOCK_BYTES is too small");
const _: () = assert!(BLOCK_SLOTS >= 1, "PIPIT_AST_BLOCK_SLOTS must be at least 1");
const _: () = assert!(
    BLOCK_SLOTS <= u32::MAX as usize / BLOCK_NODES,
    "a tree can't have more than u32::MAX nodes"
);
const BLOCK_SHIFT: u32 = BLOCK_NODES.trailing_zeros();
const BLOCK_MASK: usize = BLOCK_NODES - 1;

/// The longest token a leaf can hold, and the most children a `Call` can
/// have, as both are stored in 24 bits.
pub const DATA_MAX: u32 = (1 << 24) - 1;

impl Node {
    pub(crate) fn leaf(tag: Tag, span: Span) -> Node {
        check!(span.len <= DATA_MAX);
        let [a, b, c, _] = span.len.to_le_bytes();
        Node { tag: tag as u8, data: [a, b, c], payload: span.start }
    }

    /// A placeholder, for array slots not yet filled, and what a `Case` has
    /// for a subject or `ELSE` it hasn't: a name of no letters.
    pub(crate) const fn empty() -> Node {
        Node { tag: 0, data: [0; 3], payload: 0 }
    }

    /// Whether this is `empty()`'s.
    pub fn is_empty(self) -> bool {
        self.tag() == Tag::Name && self.span().len == 0
    }

    pub(crate) fn stage(rule: u16, first_child: u32) -> Node {
        let [a, b] = rule.to_le_bytes();
        Node { tag: Tag::Stage as u8, data: [a, b, 0], payload: first_child }
    }

    pub(crate) fn list(tag: Tag, child_count: u32, first_child: u32) -> Node {
        check!(child_count <= DATA_MAX);
        let [a, b, c, _] = child_count.to_le_bytes();
        Node { tag: tag as u8, data: [a, b, c], payload: first_child }
    }

    pub(crate) fn sort_key(descending: bool, first_child: u32) -> Node {
        Node { tag: Tag::SortKey as u8, data: [u8::from(descending), 0, 0], payload: first_child }
    }

    pub(crate) fn operation(tag: Tag, operator: Operator, first_child: u32) -> Node {
        Node { tag: tag as u8, data: [operator as u8, 0, 0], payload: first_child }
    }

    pub fn tag(self) -> Tag {
        check!(self.tag <= Tag::Query as u8);
        // SAFETY: `Tag` is a `u8` and the value is in range, checked above.
        unsafe { core::mem::transmute::<u8, Tag>(self.tag) }
    }

    pub fn operator(self) -> Operator {
        check!(matches!(self.tag(), Tag::Unary | Tag::Binary));
        check!(self.data[0] <= Operator::Distinct as u8);
        // SAFETY: `Operator` is a `u8` and the value is in range, checked above.
        unsafe { core::mem::transmute::<u8, Operator>(self.data[0]) }
    }

    pub fn is_leaf(self) -> bool {
        self.tag() < Tag::Unary
    }

    pub fn first_child(self) -> u32 {
        check!(!self.is_leaf());
        self.payload
    }

    /// For `Call`, `Case`, `List` and `Query`. A `Stage` has one child per
    /// item of its rule.
    pub fn child_count(self) -> u32 {
        check!(matches!(self.tag(), Tag::Call | Tag::Case | Tag::List | Tag::Query));
        self.data_u24()
    }

    /// For `SortKey`: whether it sorts descending.
    pub fn descending(self) -> bool {
        check!(self.tag() == Tag::SortKey);
        self.data[0] != 0
    }

    /// For `Stage`: its rule's id in the registry it was parsed with.
    pub fn rule(self) -> u16 {
        check!(self.tag() == Tag::Stage);
        let [a, b, _] = self.data;
        u16::from_le_bytes([a, b])
    }

    pub fn span(self) -> Span {
        check!(self.is_leaf());
        Span { start: self.payload, len: self.data_u24() }
    }

    fn data_u24(self) -> u32 {
        let [a, b, c] = self.data;
        u32::from_le_bytes([a, b, c, 0])
    }
}

/// A parsed tree. The root is the last node.
pub struct Ast {
    nodes: Nodes,
}

impl Ast {
    pub(crate) fn new(nodes: Nodes) -> Ast {
        check!(nodes.len() > 0);
        Ast { nodes }
    }

    pub fn root(&self) -> u32 {
        self.nodes.len() - 1
    }

    pub fn node(&self, index: u32) -> Node {
        self.nodes.get(index)
    }

    pub fn node_count(&self) -> u32 {
        self.nodes.len()
    }
}

impl core::fmt::Debug for Ast {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries((0..self.nodes.len()).map(|i| self.nodes.get(i))).finish()
    }
}

/// The tree's nodes, in equal-sized blocks. Only nodes below `count` have
/// been written.
pub(crate) struct Nodes {
    /// The first block, held directly, so reading it is the fast path.
    first: Array<Node>,
    /// The blocks after the first.
    rest: SmallVec<Array<Node>>,
    /// The block being filled. Always a whole block, so writes through it
    /// stay in bounds even if `grow` fails.
    current: NonNull<Node>,
    /// How many nodes there are.
    count: u32,
}

impl Nodes {
    pub(crate) fn new(env: &Env) -> Result<Nodes, AllocError> {
        // SAFETY: nodes are read only below `count`, once written.
        let first = unsafe { Array::uninit(env, BLOCK_NODES)? };
        let current = first.as_non_null();
        let rest = SmallVec::new(env, (BLOCK_SLOTS - 1).max(1).next_power_of_two())?;
        Ok(Nodes { first, rest, current, count: 0 })
    }

    pub(crate) fn len(&self) -> u32 {
        self.count
    }

    pub(crate) fn push(&mut self, node: Node) {
        // SAFETY: the offset is masked to within the block.
        unsafe { self.current.add(self.count as usize & BLOCK_MASK).write(node) };
        self.count += 1;
    }

    /// Whether the block being filled just became full, so `grow` must come
    /// before the next `push`.
    pub(crate) fn needs_block(&self) -> bool {
        self.count as usize & BLOCK_MASK == 0
    }

    /// Allocates the next block, from `env`, which the first came from.
    #[cold]
    pub(crate) fn grow(&mut self, env: &Env) -> Result<(), ErrorCode> {
        if self.rest.len() == BLOCK_SLOTS - 1 {
            return Err(ErrorCode::QueryTooLarge);
        }
        // SAFETY: nodes are read only below `count`, once written.
        let block =
            unsafe { Array::uninit(env, BLOCK_NODES) }.map_err(|_| ErrorCode::OutOfMemory)?;
        let current = block.as_non_null();
        self.rest.push(block).map_err(|_| ErrorCode::OutOfMemory)?;
        self.current = current;
        Ok(())
    }

    pub(crate) fn get(&self, index: u32) -> Node {
        check!(index < self.count);
        let index = index as usize;
        let block = match index >> BLOCK_SHIFT {
            0 => &self.first,
            block => at!(self.rest, block - 1),
        };
        // SAFETY: nodes below `count` have been written.
        unsafe { block.as_non_null().add(index & BLOCK_MASK).read() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nodes_span_blocks() {
        let env = Env::current();
        let mut nodes = Nodes::new(env).unwrap();
        let count = u32::try_from(2 * BLOCK_NODES + 1).unwrap();
        for i in 0..count {
            nodes.push(Node::leaf(Tag::Name, Span { start: i, len: 1 }));
            if nodes.needs_block() {
                nodes.grow(env).unwrap();
            }
        }
        for i in 0..count {
            assert_eq!(nodes.get(i).span().start, i);
        }
    }
}
