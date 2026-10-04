//! The syntax tree: every node is 8 bytes, in one array.
//!
//! A node's children sit next to each other, so a node only stores the
//! index of its first child. Leaves store their token's span instead. See
//! <https://jhwlr.io/super-flat-ast/>.

use crate::error::{ErrorCode, Span};
use crate::settings::build_setting;
use pipit_kernel::buffer::{Buffer, Primitive};

/// Leaves come first, so `tag < Unary` tells them apart.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
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
    /// `child_count` items from `first_child`, as in `SELECT a, b`.
    List,
    /// A stage of a query: one child per item of its rule, from
    /// `first_child`. Its rule is `rule()`, an id in the parser's registry.
    Stage,
    /// A query: `child_count` stages from `first_child`.
    Query,
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

    /// A placeholder, for array slots not yet filled.
    pub(crate) const fn empty() -> Node {
        Node { tag: 0, data: [0; 3], payload: 0 }
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
        check!(self.data[0] <= Operator::Divide as u8);
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

    /// For `Call`, `List` and `Query`. A `Stage` has one child per item of
    /// its rule.
    pub fn child_count(self) -> u32 {
        check!(matches!(self.tag(), Tag::Call | Tag::List | Tag::Query));
        self.data_u24()
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

/// The tree's nodes, in equal-sized blocks. Block 0 is held directly, so
/// reading it is the fast path. Later blocks are `Buffer`s kept in `rest`, a
/// slot table allocated with the second block. Only nodes below `count`, and
/// slots below `rest_count`, have been written.
pub(crate) struct Nodes {
    first: Buffer,
    rest: Option<Buffer>,
    rest_count: usize,
    /// The block being filled. Always a whole block, so writes through it
    /// stay in bounds even if `grow` fails.
    current: *mut Node,
    count: u32,
}

impl Nodes {
    /// `first` must come from `Buffer::allocate_uninit` with `BLOCK_BYTES`.
    pub(crate) fn new(mut first: Buffer) -> Nodes {
        check!(first.size_bytes() == BLOCK_BYTES);
        let current = first.as_mut_ptr::<Node>();
        Nodes { first, rest: None, rest_count: 0, current, count: 0 }
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

    /// Allocates the next block, and the slot table with the second block,
    /// from the allocator the first block came from.
    #[cold]
    pub(crate) fn grow(&mut self) -> Result<(), ErrorCode> {
        if self.rest_count == BLOCK_SLOTS - 1 {
            return Err(ErrorCode::QueryTooLarge);
        }
        if self.rest.is_none() {
            let size_bytes = (BLOCK_SLOTS - 1) * size_of::<Buffer>();
            // SAFETY: slots are read only once written.
            let table = unsafe { self.first.allocate_uninit_like(size_bytes) };
            self.rest = Some(table.map_err(|_| ErrorCode::OutOfMemory)?);
        }
        // SAFETY: nodes are read only once written.
        let mut block = unsafe { self.first.allocate_uninit_like(BLOCK_BYTES) }
            .map_err(|_| ErrorCode::OutOfMemory)?;
        self.current = block.as_mut_ptr::<Node>();
        let Some(table) = &self.rest else { pipit_kernel::check::check_failed(line!()) };
        // SAFETY: `rest_count` is below the table's length, this slot hasn't
        // been written, and the table is only reachable through `self`.
        unsafe { slots(table).cast_mut().add(self.rest_count).write(block) };
        self.rest_count += 1;
        Ok(())
    }

    pub(crate) fn get(&self, index: u32) -> Node {
        check!(index < self.count);
        let index = index as usize;
        if index < BLOCK_NODES {
            // SAFETY: nodes below `count` have been written.
            return unsafe { self.first.as_ptr::<Node>().add(index).read() };
        }
        let slot = (index >> BLOCK_SHIFT) - 1;
        check!(slot < self.rest_count);
        let Some(table) = &self.rest else { pipit_kernel::check::check_failed(line!()) };
        // SAFETY: slots below `rest_count`, and nodes below `count`, have been
        // written.
        unsafe {
            let block = &*slots(table).add(slot);
            block.as_ptr::<Node>().add(index & BLOCK_MASK).read()
        }
    }
}

impl Drop for Nodes {
    fn drop(&mut self) {
        let Some(table) = &self.rest else { return };
        let blocks = core::ptr::slice_from_raw_parts_mut(slots(table).cast_mut(), self.rest_count);
        // SAFETY: the first `rest_count` slots hold blocks, dropped once here.
        unsafe { core::ptr::drop_in_place(blocks) };
    }
}

/// The slot table's slots, which hold `Buffer`s. Read through `i64`, which is
/// aligned at least as `Buffer` is.
fn slots(table: &Buffer) -> *const Buffer {
    const { assert!(align_of::<Buffer>() <= align_of::<i64>()) };
    table.as_ptr::<i64>().cast()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipit_kernel::allocator::Heap;

    #[test]
    fn nodes_span_blocks() {
        // SAFETY: nodes are read only once written.
        let first = unsafe { Buffer::allocate_uninit(&Heap, BLOCK_BYTES) }.unwrap();
        let mut nodes = Nodes::new(first);
        let count = u32::try_from(2 * BLOCK_NODES + 1).unwrap();
        for i in 0..count {
            nodes.push(Node::leaf(Tag::Name, Span { start: i, len: 1 }));
            if nodes.needs_block() {
                nodes.grow().unwrap();
            }
        }
        for i in 0..count {
            assert_eq!(nodes.get(i).span().start, i);
        }
    }
}
