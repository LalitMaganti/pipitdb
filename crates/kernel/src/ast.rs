//! The syntax tree: every node is 8 bytes, in one array.
//!
//! A node's children sit next to each other, so a node only stores the
//! index of its first child. Leaves store their token's span instead. See
//! <https://jhwlr.io/super-flat-ast/>.

use crate::buffer::{Buffer, Primitive};
use crate::error::Span;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Tag {
    /// A column or function name. A leaf.
    Name,
    Integer,
    Float,
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
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Operator {
    Negate,
    Not,
    Or,
    And,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Add,
    Subtract,
    Multiply,
    Divide,
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

/// The longest token a leaf can hold, and the most children a `Call` can
/// have, as both are stored in 24 bits.
pub const DATA_MAX: u32 = (1 << 24) - 1;

impl Node {
    pub(crate) fn leaf(tag: Tag, span: Span) -> Node {
        check!(span.len <= DATA_MAX);
        let [a, b, c, _] = span.len.to_le_bytes();
        Node { tag: tag as u8, data: [a, b, c], payload: span.start }
    }

    pub(crate) fn list(tag: Tag, child_count: u32, first_child: u32) -> Node {
        check!(child_count <= DATA_MAX);
        let [a, b, c, _] = child_count.to_le_bytes();
        Node { tag: tag as u8, data: [a, b, c], payload: first_child }
    }

    pub(crate) fn operation(tag: Tag, operator: Operator, first_child: u32) -> Node {
        Node { tag: tag as u8, data: [operator as u8, 0, 0], payload: first_child }
    }

    pub fn tag(self) -> Tag {
        check!(self.tag <= Tag::Call as u8);
        // SAFETY: `Tag` is a `u8` and the value is in range, checked above.
        unsafe { core::mem::transmute::<u8, Tag>(self.tag) }
    }

    pub fn operator(self) -> Operator {
        check!(matches!(self.tag(), Tag::Unary | Tag::Binary));
        check!(self.data[0] <= Operator::Divide as u8);
        // SAFETY: `Operator` is a `u8` and the value is in range, checked above.
        unsafe { core::mem::transmute::<u8, Operator>(self.data[0]) }
    }

    pub fn is_leaf(self) -> bool {
        !matches!(self.tag(), Tag::Unary | Tag::Binary | Tag::Call)
    }

    pub fn first_child(self) -> u32 {
        check!(!self.is_leaf());
        self.payload
    }

    pub fn child_count(self) -> u32 {
        check!(self.tag() == Tag::Call);
        self.data_u24()
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
    nodes: Buffer,
    node_count: u32,
}

impl Ast {
    pub(crate) fn new(nodes: Buffer, node_count: u32) -> Ast {
        check!(node_count > 0);
        check!(node_count as usize <= nodes.as_slice::<Node>().len());
        Ast { nodes, node_count }
    }

    pub fn root(&self) -> u32 {
        self.node_count - 1
    }

    pub fn node(&self, index: u32) -> Node {
        check!(index < self.node_count);
        *at!(self.nodes.as_slice::<Node>(), index as usize)
    }

    pub fn node_count(&self) -> u32 {
        self.node_count
    }
}

impl core::fmt::Debug for Ast {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let nodes = at!(self.nodes.as_slice::<Node>(), ..self.node_count as usize);
        f.debug_list().entries(nodes).finish()
    }
}
