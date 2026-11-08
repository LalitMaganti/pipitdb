//! `Predicate`: a condition on a batch's rows, such as `WHERE`'s, which
//! narrows its selection to the rows the condition is true for.
//!
//! Each node can narrow a selection to the rows it's true for, or to those
//! it's false for. A null row is neither, as in SQL, so `NOT` swaps the two
//! and every node keeps SQL's three-valued logic without building booleans.

use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::column::{ColumnView, Form};
use crate::context::Context;
use crate::error::Error;
use crate::filter::{self, Comparison, Entries, Value};
use crate::flat_hash_map::FlatHashMap;
use crate::murmur_hash;
use crate::row_batch::RowBatch;
use crate::selection::Selection;
use crate::slow_vec::SlowVec;
use crate::step::Transform;

/// The most nodes a predicate can have.
pub const PREDICATE_NODES_MAX: usize = 1 << 6;

/// A node of a predicate. Children come before their parents.
#[derive(Clone, Copy, Debug)]
pub enum Node {
    Leaf(Leaf),
    And(u32, u32),
    Or(u32, u32),
    Not(u32),
}

/// A condition on one column, which a filter tests.
#[derive(Clone, Copy, Debug)]
pub enum Leaf {
    /// The column at `column` in a batch, compared with `value`.
    Compare {
        column: u32,
        comparison: Comparison,
        value: Value,
    },
    /// The string column at `column` in a batch, compared byte by byte with
    /// bytes `start..start + len` of the predicate's strings.
    CompareString {
        column: u32,
        comparison: Comparison,
        start: u32,
        len: u32,
    },
    /// The string column at `column` in a batch, matched with the `LIKE`
    /// pattern in bytes `start..start + len` of the predicate's strings.
    Like {
        column: u32,
        start: u32,
        len: u32,
    },
    IsNull {
        column: u32,
    },
    /// The integer column at `column` in a batch, one of values
    /// `start..start + len` of the predicate's values.
    In {
        column: u32,
        start: u32,
        len: u32,
    },
    /// The string column at `column` in a batch, one of the strings that
    /// values `start..start + len` of the predicate's values place in its
    /// strings: each its start in the high 32 bits, and length in the low.
    InStrings {
        column: u32,
        start: u32,
        len: u32,
    },
}

/// Its root is its last node.
pub struct Predicate {
    nodes: SlowVec<Node>,
    /// The values of its `IN` lists, then the bytes of the strings its
    /// conditions compare with, if it has any: shared by its copies.
    literals: Option<Buffer>,
    /// How many values `literals` starts with.
    value_count: usize,
    /// For each node, if any is an `IN` list of more than
    /// `filter::IN_SCAN_MAX` values, the set of them.
    sets: Option<SlowVec<Option<Set>>>,
    depth: u32,
}

/// An `IN` list of more than `filter::IN_SCAN_MAX` values, kept in a hash
/// map, as Perfetto keeps one, and made once, as Perfetto casts a list once
/// for each run of a query. Integers and strings share one map type, so its
/// code is compiled once.
type Set = FlatHashMap<filter::Listed, ()>;

impl Predicate {
    /// A predicate of `nodes`, the last of which is its root.
    pub fn new(nodes: SlowVec<Node>) -> Predicate {
        Predicate::checked(nodes, None, 0)
    }

    /// A predicate of `nodes`, whose string comparisons compare with bytes
    /// of `strings`, and whose `IN` lists are of `values`. Lists of more than
    /// `filter::IN_SCAN_MAX` values are put in sets, from `allocator`.
    pub fn with_literals(
        allocator: &dyn Allocator,
        nodes: SlowVec<Node>,
        strings: &[u8],
        values: &[i64],
    ) -> Result<Predicate, AllocError> {
        let mut literals = None;
        if !strings.is_empty() || !values.is_empty() {
            // Whole words, as the values are read as them.
            let bytes = (size_of_val(values) + strings.len()).next_multiple_of(size_of::<i64>());
            let mut buffer = Buffer::allocate(allocator, bytes)?;
            at_mut!(buffer.as_mut_slice::<i64>(), ..values.len()).copy_from_slice(values);
            let at = size_of_val(values);
            at_mut!(buffer.as_mut_slice::<u8>(), at..at + strings.len()).copy_from_slice(strings);
            literals = Some(buffer);
        }
        Predicate::sharing(allocator, nodes, literals, values.len())
    }

    /// A predicate of `nodes`, and of `literals`, starting with
    /// `value_count` values, which may be another's.
    fn sharing(
        allocator: &dyn Allocator,
        nodes: SlowVec<Node>,
        literals: Option<Buffer>,
        value_count: usize,
    ) -> Result<Predicate, AllocError> {
        let mut predicate = Predicate::checked(nodes, literals, value_count);
        let long = |node: &Node| match *node {
            Node::Leaf(Leaf::In { len, .. } | Leaf::InStrings { len, .. }) => {
                len as usize > filter::IN_SCAN_MAX
            }
            _ => false,
        };
        if predicate.nodes.iter().any(long) {
            let mut sets = SlowVec::fixed(allocator, predicate.nodes.len())?;
            for &node in predicate.nodes.iter() {
                sets.push(predicate.set_of(allocator, node)?)?;
            }
            predicate.sets = Some(sets);
        }
        Ok(predicate)
    }

    /// The set of node `node`'s values, if it's an `IN` list of more than
    /// `filter::IN_SCAN_MAX` of them.
    fn set_of(&self, allocator: &dyn Allocator, node: Node) -> Result<Option<Set>, AllocError> {
        let (strings, start, len) = match node {
            Node::Leaf(Leaf::In { start, len, .. }) => (false, start, len),
            Node::Leaf(Leaf::InStrings { start, len, .. }) => (true, start, len),
            _ => return Ok(None),
        };
        if len as usize <= filter::IN_SCAN_MAX {
            return Ok(None);
        }
        let mut set = FlatHashMap::for_len(allocator, len as usize)?;
        // A value listed twice is put in twice, which is harmless.
        for &value in at!(self.values(), start as usize..(start + len) as usize) {
            let hash = if strings {
                murmur_hash::bytes(self.placed_string(value))
            } else {
                murmur_hash::mix(value.cast_unsigned())
            };
            set.insert(allocator, filter::Listed { hash, value }, ())?;
        }
        Ok(Some(set))
    }

    /// The string of an `IN` list that `packed` places in its strings.
    fn placed_string(&self, packed: i64) -> &[u8] {
        let (start, len) = placed(packed);
        at!(self.strings(), start as usize..(start + len) as usize)
    }

    /// The values of its `IN` lists.
    fn values(&self) -> &[i64] {
        self.literals.as_ref().map_or(&[], |literals| at!(literals.as_slice(), ..self.value_count))
    }

    /// The bytes of the strings its conditions compare with.
    fn strings(&self) -> &[u8] {
        let values = self.value_count * size_of::<i64>();
        self.literals.as_ref().map_or(&[], |literals| at!(literals.as_slice(), values..))
    }

    /// A predicate of `nodes`, checked to be one, and of `literals`, but
    /// without sets.
    fn checked(nodes: SlowVec<Node>, literals: Option<Buffer>, value_count: usize) -> Predicate {
        check!(!nodes.is_empty() && nodes.len() <= PREDICATE_NODES_MAX);
        let literal_bytes = literals.as_ref().map_or(0, Buffer::size_bytes);
        check!(value_count * size_of::<i64>() <= literal_bytes);
        let bytes = literal_bytes - value_count * size_of::<i64>();
        let within = |start: u32, len: u32, end: usize| {
            (start as usize).checked_add(len as usize).is_some_and(|last| last <= end)
        };
        check!(nodes.iter().all(|node| match *node {
            Node::Leaf(Leaf::CompareString { start, len, .. } | Leaf::Like { start, len, .. }) => {
                within(start, len, bytes)
            }
            // Each value's string is checked to be in `strings` where it's read.
            Node::Leaf(Leaf::In { start, len, .. } | Leaf::InStrings { start, len, .. }) => {
                len != 0 && within(start, len, value_count)
            }
            _ => true,
        }));
        // Each node's depth: how many selections it holds at once, at most.
        let mut depths = [0_u32; PREDICATE_NODES_MAX];
        for (i, node) in (0..).zip(nodes.iter()) {
            let depth = |child: u32| *at!(depths, child as usize);
            *at_mut!(depths, i as usize) = match *node {
                Node::Leaf(_) => 0,
                // One while its children run, in case it needs the rows left.
                Node::And(a, b) | Node::Or(a, b) => {
                    check!(a < i && b < i);
                    1 + depth(a).max(depth(b))
                }
                Node::Not(a) => {
                    check!(a < i);
                    depth(a)
                }
            };
        }
        let depth = *at!(depths, nodes.len() - 1);
        Predicate { nodes, literals, value_count, sets: None, depth }
    }

    /// Its nodes, the last its root.
    pub(crate) fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// How many scratch selections `select` needs.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// The conjuncts, the parts of its top-level `AND`s, that read only
    /// `column`, joined by `AND`, or `None` if there are none.
    pub fn conjuncts_on(
        &self,
        allocator: &dyn Allocator,
        column: u32,
    ) -> Result<Option<Predicate>, AllocError> {
        let mut nodes = SlowVec::new(allocator, PREDICATE_NODES_MAX)?;
        #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX`")]
        let root = self.copy_on(self.nodes.len() as u32 - 1, column, &mut nodes, None)?;
        if root.is_none() {
            return Ok(None);
        }
        let literals = self.literals.clone();
        Ok(Some(Predicate::sharing(allocator, nodes, literals, self.value_count)?))
    }

    /// Copies node `node`'s conjuncts that read only `column` to `nodes`,
    /// each joined to `root`, those so far, by `AND`: the new root.
    fn copy_on(
        &self,
        node: u32,
        column: u32,
        nodes: &mut SlowVec<Node>,
        root: Option<u32>,
    ) -> Result<Option<u32>, AllocError> {
        if let Node::And(a, b) = *at!(self.nodes, node as usize) {
            let root = self.copy_on(a, column, nodes, root)?;
            return self.copy_on(b, column, nodes, root);
        }
        if !self.reads_only(node, column) {
            return Ok(root);
        }
        let copied = self.copy(node, nodes)?;
        Ok(Some(match root {
            Some(root) => push(nodes, Node::And(root, copied))?,
            None => copied,
        }))
    }

    /// Whether node `node` reads only `column`.
    fn reads_only(&self, node: u32, column: u32) -> bool {
        match *at!(self.nodes, node as usize) {
            Node::Leaf(leaf) => leaf.column() == column,
            Node::And(a, b) | Node::Or(a, b) => {
                self.reads_only(a, column) && self.reads_only(b, column)
            }
            Node::Not(a) => self.reads_only(a, column),
        }
    }

    /// Copies node `node` and those under it to `nodes`, returning where it
    /// is there.
    fn copy(&self, node: u32, nodes: &mut SlowVec<Node>) -> Result<u32, AllocError> {
        let copied = match *at!(self.nodes, node as usize) {
            Node::Leaf(leaf) => Node::Leaf(leaf),
            Node::And(a, b) => Node::And(self.copy(a, nodes)?, self.copy(b, nodes)?),
            Node::Or(a, b) => Node::Or(self.copy(a, nodes)?, self.copy(b, nodes)?),
            Node::Not(a) => Node::Not(self.copy(a, nodes)?),
        };
        push(nodes, copied)
    }

    /// Whether every condition it has on `column` compares it with a string,
    /// and it has one.
    pub fn compares_only_strings(&self, column: u32) -> bool {
        let leaves = self.nodes.iter().filter_map(|node| match *node {
            Node::Leaf(leaf) => Some(leaf),
            Node::And(..) | Node::Or(..) | Node::Not(_) => None,
        });
        let mut on_column = leaves.filter(|leaf| leaf.column() == column).peekable();
        on_column.peek().is_some()
            && on_column.all(|leaf| {
                matches!(
                    leaf,
                    Leaf::CompareString { .. } | Leaf::Like { .. } | Leaf::InStrings { .. }
                )
            })
    }

    /// The columns its comparisons and `IS NULL`s read, maybe more than once.
    pub fn columns(&self) -> impl Iterator<Item = u32> + '_ {
        self.nodes.iter().filter_map(|node| match *node {
            Node::Leaf(leaf) => Some(leaf.column()),
            Node::And(..) | Node::Or(..) | Node::Not(_) => None,
        })
    }

    /// A copy that reads column `renumber(c)` wherever this reads `c`, such
    /// as a plan's column ids turned into positions in batches.
    pub fn renumbered(
        &self,
        allocator: &dyn Allocator,
        renumber: impl Fn(u32) -> u32,
    ) -> Result<Predicate, AllocError> {
        let nodes = self.nodes.iter().map(|&node| match node {
            Node::Leaf(Leaf::Compare { column, comparison, value }) => {
                Node::Leaf(Leaf::Compare { column: renumber(column), comparison, value })
            }
            Node::Leaf(Leaf::CompareString { column, comparison, start, len }) => {
                let column = renumber(column);
                Node::Leaf(Leaf::CompareString { column, comparison, start, len })
            }
            Node::Leaf(Leaf::Like { column, start, len }) => {
                Node::Leaf(Leaf::Like { column: renumber(column), start, len })
            }
            Node::Leaf(Leaf::IsNull { column }) => {
                Node::Leaf(Leaf::IsNull { column: renumber(column) })
            }
            Node::Leaf(Leaf::In { column, start, len }) => {
                Node::Leaf(Leaf::In { column: renumber(column), start, len })
            }
            Node::Leaf(Leaf::InStrings { column, start, len }) => {
                Node::Leaf(Leaf::InStrings { column: renumber(column), start, len })
            }
            node => node,
        });
        let nodes = SlowVec::fixed_from(allocator, nodes)?;
        Predicate::sharing(allocator, nodes, self.literals.clone(), self.value_count)
    }

    /// Where each node keeps the dictionary entries it found, for `select`.
    pub fn new_entries(&self, allocator: &dyn Allocator) -> Result<SlowVec<Entries>, AllocError> {
        SlowVec::fixed_from(allocator, (0..self.nodes.len()).map(|_| Entries::default()))
    }

    /// Narrows `batch`'s selection to the rows this is true for, working in
    /// `scratch`, which holds at least `depth` selections. Its leaves keep
    /// the entries of dictionary columns they find in `entries`, made by
    /// `new_entries`, allocating from `allocator`.
    #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX` nodes")]
    #[inline]
    pub fn select(
        &self,
        allocator: &dyn Allocator,
        scratch: &mut [Selection],
        entries: &mut [Entries],
        batch: &mut RowBatch,
    ) -> Result<(), AllocError> {
        check!(scratch.len() >= self.depth as usize && entries.len() == self.nodes.len());
        let (columns, selection) = batch.columns_and_selection();
        let (strings, values) = (self.strings(), self.values());
        let sets = self.sets.as_deref().unwrap_or(&[]);
        let mut inputs = Inputs { columns, strings, values, sets, entries, allocator };
        let root = self.nodes.len() as u32 - 1;
        self.narrow(&mut inputs, scratch, root, true, selection)
    }

    /// Narrows `selection` to the rows node `node` is `want` for. A node that
    /// needs a selection takes the first of `scratch`, and its children work
    /// in the rest.
    fn narrow(
        &self,
        inputs: &mut Inputs<'_>,
        scratch: &mut [Selection],
        node: u32,
        want: bool,
        selection: &mut Selection,
    ) -> Result<(), AllocError> {
        match *at!(self.nodes, node as usize) {
            Node::Leaf(leaf) => leaf.narrow(inputs, node, None, want, selection)?,
            Node::Not(child) => self.narrow(inputs, scratch, child, !want, selection)?,
            // True for both, or false for both: each narrows what the other left.
            Node::And(a, b) if want => self.both(inputs, scratch, a, b, true, selection)?,
            Node::Or(a, b) if !want => self.both(inputs, scratch, a, b, false, selection)?,
            // False for either, or true for either: the rows `a` is `want` for,
            // and those `b` is among the rest.
            Node::And(a, b) | Node::Or(a, b) => {
                let Some((rest, scratch)) = scratch.split_first_mut() else {
                    failed!();
                };
                if let Node::Leaf(leaf) = *at!(self.nodes, a as usize) {
                    // In one pass.
                    leaf.narrow(inputs, a, Some(rest), want, selection)?;
                } else {
                    rest.clone_from(selection);
                    self.narrow(inputs, scratch, a, want, selection)?;
                    rest.subtract(selection);
                }
                self.narrow(inputs, scratch, b, want, rest)?;
                selection.union(rest);
            }
        }
        Ok(())
    }

    fn both(
        &self,
        inputs: &mut Inputs<'_>,
        scratch: &mut [Selection],
        a: u32,
        b: u32,
        want: bool,
        selection: &mut Selection,
    ) -> Result<(), AllocError> {
        self.narrow(inputs, scratch, a, want, selection)?;
        self.narrow(inputs, scratch, b, want, selection)
    }
}

/// What a predicate's nodes narrow a batch's selection with.
struct Inputs<'b> {
    /// The batch's columns.
    columns: &'b [ColumnView],
    /// The bytes its string comparisons compare with.
    strings: &'b [u8],
    /// The values of its `IN` lists.
    values: &'b [i64],
    /// For each node, the set of its `IN` list's values, if it's long.
    sets: &'b [Option<Set>],
    /// Where each node keeps the dictionary entries it found.
    entries: &'b mut [Entries],
    /// What the entries' bits are allocated from.
    allocator: &'b dyn Allocator,
}

impl Leaf {
    /// The column it reads.
    fn column(self) -> u32 {
        match self {
            Leaf::Compare { column, .. }
            | Leaf::CompareString { column, .. }
            | Leaf::Like { column, .. }
            | Leaf::IsNull { column }
            | Leaf::In { column, .. }
            | Leaf::InStrings { column, .. } => column,
        }
    }

    /// Narrows `selection` to the rows this, node `node`, is `want` for,
    /// writing the rows it drops to `dropped`, if given. A dictionary
    /// column's entries, given only to string comparisons, are each tested
    /// once, and the rows kept by their entry's.
    fn narrow(
        self,
        inputs: &mut Inputs<'_>,
        node: u32,
        dropped: Option<&mut Selection>,
        want: bool,
        selection: &mut Selection,
    ) -> Result<(), AllocError> {
        match self {
            Leaf::Compare { column, comparison, value } => {
                let comparison = if want { comparison } else { negated(comparison) };
                let column = at!(inputs.columns, column as usize);
                filter::compare(column, comparison, value, selection, dropped);
            }
            Leaf::CompareString { column, comparison, start, len } => {
                let comparison = if want { comparison } else { negated(comparison) };
                let string = at!(inputs.strings, start as usize..(start + len) as usize);
                let column = at!(inputs.columns, column as usize);
                if !matches!(column.form(), Form::Dictionary(_)) {
                    filter::compare_string(column, comparison, string, selection, dropped);
                    return Ok(());
                }
                let entries = at_mut!(inputs.entries, node as usize);
                let entries = entries.find(inputs.allocator, column, want, &|values, bits| {
                    filter::compare_string_bits(values, comparison, string, bits);
                })?;
                filter::keep_entries(column, entries, selection, dropped);
            }
            Leaf::Like { column, start, len } => {
                let pattern = at!(inputs.strings, start as usize..(start + len) as usize);
                let column = at!(inputs.columns, column as usize);
                if !matches!(column.form(), Form::Dictionary(_)) {
                    filter::like(column, pattern, want, selection, dropped);
                    return Ok(());
                }
                let entries = at_mut!(inputs.entries, node as usize);
                let entries = entries.find(inputs.allocator, column, want, &|values, bits| {
                    filter::like_bits(values, pattern, want, bits);
                })?;
                filter::keep_entries(column, entries, selection, dropped);
            }
            Leaf::IsNull { column } => {
                filter::is_null(at!(inputs.columns, column as usize), want, selection, dropped);
            }
            Leaf::In { column, start, len } => {
                let values = at!(inputs.values, start as usize..(start + len) as usize);
                let set = inputs.sets.get(node as usize).and_then(Option::as_ref);
                let column = at!(inputs.columns, column as usize);
                filter::is_in(column, values, set, want, selection, dropped);
            }
            Leaf::InStrings { column, start, len } => {
                let values = at!(inputs.values, start as usize..(start + len) as usize);
                in_strings(inputs, node, column, values, dropped, want, selection)?;
            }
        }
        Ok(())
    }
}

/// Narrows `selection` to the rows node `node`, `column IN (...)` of the
/// strings `values` place, is `want` for, as `Leaf::narrow` does.
fn in_strings(
    inputs: &mut Inputs<'_>,
    node: u32,
    column: u32,
    values: &[i64],
    dropped: Option<&mut Selection>,
    want: bool,
    selection: &mut Selection,
) -> Result<(), AllocError> {
    let strings = inputs.strings;
    let string = move |packed: i64| {
        let (start, len) = placed(packed);
        at!(strings, start as usize..(start + len) as usize)
    };
    let scanned = |cell: &[u8]| values.iter().any(|&packed| string(packed) == cell);
    let found = |set: &Set, cell: &[u8]| {
        let equal = |held: &filter::Listed| string(held.value) == cell;
        set.find_by(murmur_hash::bytes(cell), equal).is_some()
    };
    let looked_up;
    let listed: &dyn Fn(&[u8]) -> bool = match inputs.sets.get(node as usize) {
        Some(Some(set)) => {
            looked_up = move |cell: &[u8]| found(set, cell);
            &looked_up
        }
        _ => &scanned,
    };
    let column = at!(inputs.columns, column as usize);
    if !matches!(column.form(), Form::Dictionary(_)) {
        filter::is_in_strings(column, listed, want, selection, dropped);
        return Ok(());
    }
    let entries = at_mut!(inputs.entries, node as usize);
    let entries = entries.find(inputs.allocator, column, want, &|values, bits| {
        filter::in_strings_bits(values, listed, want, bits);
    })?;
    filter::keep_entries(column, entries, selection, dropped);
    Ok(())
}

/// A string of an `IN` list, at `start` in a predicate's strings and `len`
/// bytes long, as `Leaf::InStrings` places it.
pub fn place(start: u32, len: u32) -> i64 {
    (u64::from(start) << 32 | u64::from(len)).cast_signed()
}

/// Where a string of an `IN` list is in a predicate's strings: its start and
/// length, from `packed`, as `place` packs them.
pub fn placed(packed: i64) -> (u32, u32) {
    let packed = packed.cast_unsigned();
    #[expect(clippy::cast_possible_truncation, reason = "the two halves")]
    ((packed >> 32) as u32, packed as u32)
}

/// The comparison true for exactly the non-null rows `comparison` is false
/// for. Floats are ordered totally, so this holds for them too.
fn negated(comparison: Comparison) -> Comparison {
    match comparison {
        Comparison::Equal => Comparison::NotEqual,
        Comparison::NotEqual => Comparison::Equal,
        Comparison::Less => Comparison::GreaterEqual,
        Comparison::LessEqual => Comparison::Greater,
        Comparison::Greater => Comparison::LessEqual,
        Comparison::GreaterEqual => Comparison::Less,
    }
}

/// Adds `node` to `nodes`, returning where it is.
fn push(nodes: &mut SlowVec<Node>, node: Node) -> Result<u32, AllocError> {
    #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX`")]
    let at = nodes.len() as u32;
    nodes.push(node)?;
    Ok(at)
}

/// Narrows each batch's selection to the rows `predicate` is true for. The
/// predicate reads columns by their position in batches.
pub struct Filter {
    pub predicate: Predicate,
}

impl Transform for Filter {
    /// Where each node keeps the dictionary entries it found.
    type State = SlowVec<Entries>;

    fn new_state(&self, context: &mut Context) -> Result<SlowVec<Entries>, Error> {
        context.reserve_selections(self.predicate.depth() as usize)?;
        Ok(self.predicate.new_entries(context.allocator())?)
    }

    fn process(
        &self,
        context: &mut Context,
        entries: &mut SlowVec<Entries>,
        batch: &mut RowBatch,
    ) -> Result<(), Error> {
        let allocator = context.allocator();
        Ok(self.predicate.select(allocator, context.selections(), entries, batch)?)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec as StdVec;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::{DataType, Forms};
    use crate::context::Context;
    use crate::selection::Kept;

    /// `a` and `b`, with `None` for null.
    const A: [Option<i64>; 6] = [Some(1), None, Some(3), None, Some(5), Some(3)];
    const B: [Option<i64>; 6] = [None, Some(2), None, Some(4), Some(5), Some(1)];

    fn column(cells: [Option<i64>; 6]) -> ColumnView {
        let mut values = Buffer::allocate(&Heap, 48).unwrap();
        let mut validity = Buffer::allocate(&Heap, 1).unwrap();
        for (row, cell) in cells.iter().enumerate() {
            values.as_mut_slice::<i64>()[row] = cell.unwrap_or(0);
            validity.as_mut_slice::<u8>()[0] |= u8::from(cell.is_some()) << row;
        }
        ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, Some(validity)).unwrap()
    }

    /// A predicate to evaluate both ways: by `Predicate`, and by hand.
    enum Expr {
        Greater(u32, i64),
        IsNull(u32),
        And(&'static Expr, &'static Expr),
        Or(&'static Expr, &'static Expr),
        Not(&'static Expr),
    }

    /// SQL's three-valued logic, row by row: `None` is null.
    fn reference(expr: &Expr, row: usize) -> Option<bool> {
        let cell = |column: u32| if column == 0 { A[row] } else { B[row] };
        match *expr {
            Expr::Greater(column, value) => cell(column).map(|cell| cell > value),
            Expr::IsNull(column) => Some(cell(column).is_none()),
            Expr::And(a, b) => match (reference(a, row), reference(b, row)) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            },
            Expr::Or(a, b) => match (reference(a, row), reference(b, row)) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            },
            Expr::Not(a) => reference(a, row).map(|value| !value),
        }
    }

    /// Adds `expr`'s nodes to `nodes`, children first, returning its own.
    fn build(expr: &Expr, nodes: &mut SlowVec<Node>) -> u32 {
        let node = match *expr {
            Expr::Greater(column, value) => Node::Leaf(Leaf::Compare {
                column,
                comparison: Comparison::Greater,
                value: Value::Int64(value),
            }),
            Expr::IsNull(column) => Node::Leaf(Leaf::IsNull { column }),
            Expr::And(a, b) => Node::And(build(a, nodes), build(b, nodes)),
            Expr::Or(a, b) => Node::Or(build(a, nodes), build(b, nodes)),
            Expr::Not(a) => Node::Not(build(a, nodes)),
        };
        assert!(nodes.push(node).is_ok());
        u32::try_from(nodes.len() - 1).unwrap()
    }

    /// The rows `expr` keeps of `a` and `b`.
    fn predicate_of(expr: &Expr) -> Predicate {
        let mut nodes = SlowVec::new(&Heap, PREDICATE_NODES_MAX).unwrap();
        build(expr, &mut nodes);
        Predicate::new(nodes)
    }

    fn kept(expr: &Expr) -> StdVec<usize> {
        kept_by(&predicate_of(expr))
    }

    /// The rows of `a` and `b` `predicate` keeps.
    fn kept_by(predicate: &Predicate) -> StdVec<usize> {
        let mut scratch: StdVec<Selection> =
            (0..predicate.depth()).map(|_| Selection::all(0)).collect();
        let mut entries = predicate.new_entries(&Heap).unwrap();
        let mut batch = RowBatch::new();
        batch.reset(6);
        assert!(batch.push_column(column(A)).is_ok() && batch.push_column(column(B)).is_ok());
        predicate.select(&Heap, &mut scratch, &mut entries, &mut batch).unwrap();
        match batch.selection().kept() {
            Kept::Prefix => (0..6).collect(),
            Kept::None => StdVec::new(),
            Kept::Select(rows) => rows.iter().map(|&row| usize::from(row)).collect(),
        }
    }

    #[test]
    fn copies_the_conjuncts_on_a_column() {
        const A2: Expr = Expr::Greater(0, 2);
        const A4_OR_NULL: Expr = Expr::Or(&Expr::Greater(0, 4), &Expr::IsNull(0));
        const EITHER: Expr = Expr::Or(&A2, &Expr::Greater(1, 2));
        // `(a > 2 AND b IS NULL) AND (a > 4 OR a IS NULL) AND (a > 2 OR b > 2)`.
        const ALL: Expr =
            Expr::And(&Expr::And(&Expr::And(&A2, &Expr::IsNull(1)), &A4_OR_NULL), &EITHER);
        let predicate = predicate_of(&ALL);
        let on_a = predicate.conjuncts_on(&Heap, 0).unwrap().unwrap();
        assert!(on_a.columns().all(|column| column == 0));
        assert_eq!(kept_by(&on_a), kept(&Expr::And(&A2, &A4_OR_NULL)));
        // None reads only a column it doesn't read.
        assert!(predicate.conjuncts_on(&Heap, 2).unwrap().is_none());
        // One conjunct on the column is all of it.
        let all = predicate_of(&A4_OR_NULL).conjuncts_on(&Heap, 0).unwrap().unwrap();
        assert_eq!(kept_by(&all), kept(&A4_OR_NULL));
    }

    #[test]
    fn keeps_what_sql_does_with_nulls() {
        const A2: Expr = Expr::Greater(0, 2);
        const B2: Expr = Expr::Greater(1, 2);
        const BOTH: Expr = Expr::And(&A2, &B2);
        const EITHER: Expr = Expr::Or(&A2, &B2);
        const NULL_OR_B: Expr = Expr::Or(&Expr::IsNull(0), &B2);
        let exprs: [&Expr; 9] = [
            &A2,
            &Expr::Not(&A2),
            &BOTH,
            &EITHER,
            &Expr::Not(&BOTH),
            &Expr::Not(&EITHER),
            &NULL_OR_B,
            &Expr::Not(&NULL_OR_B),
            &Expr::Not(&Expr::Not(&Expr::And(&EITHER, &Expr::Not(&BOTH)))),
        ];
        // One scratch selection per `AND` or `OR` nested in another.
        let depth = |expr: &Expr| {
            let mut nodes = SlowVec::new(&Heap, PREDICATE_NODES_MAX).unwrap();
            build(expr, &mut nodes);
            Predicate::new(nodes).depth()
        };
        assert_eq!([&A2, &BOTH, exprs[8]].map(depth), [0, 1, 2]);
        for expr in exprs {
            let expected: StdVec<usize> =
                (0..6).filter(|&row| reference(expr, row) == Some(true)).collect();
            assert_eq!(kept(expr), expected);
        }
    }

    #[test]
    fn keeps_rows_in_a_list() {
        // `a IN (3, 5)`, `b IN (2)`, and either.
        let a_in = Leaf::In { column: 0, start: 0, len: 2 };
        let b_in = Leaf::In { column: 1, start: 2, len: 1 };
        let kept = |nodes: &[Node]| {
            // Then 100 to 119, for a list long enough for a set.
            let values: StdVec<i64> = [3, 5, 2, 7, 9, 11].into_iter().chain(100..120).collect();
            kept_by(
                &Predicate::with_literals(
                    &Heap,
                    SlowVec::fixed_from(&Heap, nodes.iter().copied()).unwrap(),
                    &[],
                    &values,
                )
                .unwrap(),
            )
        };
        assert_eq!(kept(&[Node::Leaf(a_in)]), [2, 4, 5]);
        // A list of more than four values.
        let long = Leaf::In { column: 0, start: 0, len: 6 };
        assert_eq!(kept(&[Node::Leaf(long), Node::Not(0)]), [0]);
        // One of more than `IN_SCAN_MAX` values, looked up in a set.
        let set = Leaf::In { column: 0, start: 0, len: 26 };
        assert_eq!(kept(&[Node::Leaf(set)]), [2, 4, 5]);
        assert_eq!(kept(&[Node::Leaf(set), Node::Not(0)]), [0]);
        // A null is in no list, and not out of one either.
        assert_eq!(kept(&[Node::Leaf(a_in), Node::Not(0)]), [0]);
        assert_eq!(kept(&[Node::Leaf(a_in), Node::Leaf(b_in), Node::Or(0, 1)]), [1, 2, 4, 5]);
    }

    /// `values` as a string column, `None` null.
    fn strings(values: &[Option<&str>]) -> ColumnView {
        let bytes: StdVec<u8> = values.iter().flatten().flat_map(|v| v.bytes()).collect();
        let mut views = Buffer::allocate(&Heap, values.len() * 8).unwrap();
        let mut validity = Buffer::allocate(&Heap, 1).unwrap();
        let mut start = 0;
        for (i, value) in values.iter().enumerate() {
            let len = u32::try_from(value.map_or(0, str::len)).unwrap();
            views.as_mut_slice::<u32>()[2 * i..2 * i + 2].copy_from_slice(&[start, len]);
            validity.as_mut_slice::<u8>()[0] |= u8::from(value.is_some()) << i;
            start += len;
        }
        let mut stored = Buffer::allocate(&Heap, bytes.len().max(1)).unwrap();
        stored.as_mut_slice::<u8>()[..bytes.len()].copy_from_slice(&bytes);
        ColumnView::strings(&mut Context::new(&Heap), views, stored, Some(validity)).unwrap()
    }

    /// `values` with `indices` into them, as a dictionary column.
    fn indexed(values: &ColumnView, indices: &[u32]) -> ColumnView {
        let mut buffer = Buffer::allocate(&Heap, indices.len() * 4).unwrap();
        buffer.as_mut_slice::<u32>().copy_from_slice(indices);
        ColumnView::dictionary(&mut Context::new(&Heap), values, buffer).unwrap()
    }

    /// `x <comparison> 'b'`, or `NOT` that, over the first column.
    fn compare_b(comparison: Comparison, not: bool) -> Predicate {
        let leaf = Leaf::CompareString { column: 0, comparison, start: 0, len: 1 };
        let nodes = [Node::Leaf(leaf), Node::Not(0)];
        let nodes = SlowVec::fixed_from(&Heap, nodes[..=usize::from(not)].iter().copied());
        Predicate::with_literals(&Heap, nodes.unwrap(), b"b", &[]).unwrap()
    }

    /// The rows `predicate` keeps of a batch of `column`, its entries kept in
    /// `entries` between batches.
    fn kept_of(predicate: &Predicate, entries: &mut [Entries], column: &ColumnView) -> StdVec<u16> {
        let mut scratch: StdVec<Selection> =
            (0..predicate.depth()).map(|_| Selection::all(0)).collect();
        let mut batch = RowBatch::new();
        batch.reset(column.row_count());
        assert!(batch.push_column(column.clone()).is_ok());
        predicate.select(&Heap, &mut scratch, entries, &mut batch).unwrap();
        match batch.selection().kept() {
            Kept::Prefix => (0..u16::try_from(column.row_count()).unwrap()).collect(),
            Kept::None => StdVec::new(),
            Kept::Select(rows) => rows.to_vec(),
        }
    }

    #[test]
    fn filters_string_dictionaries_as_their_strings() {
        // Entries out of order, one null, used in any order.
        let entries = strings(&[Some("c"), None, Some("a"), Some("b"), Some("")]);
        // Three groups of eight rows, and three after them.
        let indices = [[3, 1, 0, 2, 4, 3, 0, 1].as_slice(); 3].concat();
        let indices = [indices.as_slice(), &[2, 3, 0]].concat();
        let dictionary = indexed(&entries, &indices);
        let mut flat = dictionary.clone();
        flat.make_in(&mut Context::new(&Heap), Forms::FLAT).unwrap();
        for comparison in [
            Comparison::Equal,
            Comparison::NotEqual,
            Comparison::Less,
            Comparison::LessEqual,
            Comparison::Greater,
            Comparison::GreaterEqual,
        ] {
            for not in [false, true] {
                let predicate = compare_b(comparison, not);
                let mut entries = predicate.new_entries(&Heap).unwrap();
                let expected = kept_of(&predicate, &mut entries, &flat);
                assert_eq!(kept_of(&predicate, &mut entries, &dictionary), expected);
            }
        }
        // `x LIKE '_'` keeps entries 0, 2 and 3, and `NOT x LIKE '_'` entry 4:
        // neither keeps the null, entry 1.
        let rows_of = |kept: &[u32]| -> StdVec<u16> {
            (0..).zip(&indices).filter(|(_, e)| kept.contains(e)).map(|(row, _)| row).collect()
        };
        for (not, expected) in [(false, rows_of(&[0, 2, 3])), (true, rows_of(&[4]))] {
            let nodes = [Node::Leaf(Leaf::Like { column: 0, start: 0, len: 1 }), Node::Not(0)];
            let nodes = SlowVec::fixed_from(&Heap, nodes[..=usize::from(not)].iter().copied());
            let predicate = Predicate::with_literals(&Heap, nodes.unwrap(), b"_", &[]).unwrap();
            let mut entries = predicate.new_entries(&Heap).unwrap();
            assert_eq!(kept_of(&predicate, &mut entries, &flat), expected);
            assert_eq!(kept_of(&predicate, &mut entries, &dictionary), expected);
        }
        // `x IN ('b', 'c')` keeps entries 0 and 3, and `NOT` that entries 2 and
        // 4, whether the list is scanned, or long enough for a set: 'b' and
        // 'c' then 20 strings of two bytes, and 'b' again.
        let strings: StdVec<u8> =
            b"bc".iter().copied().chain((0..20).flat_map(|i| [b'x', i])).collect();
        let places: StdVec<i64> = [place(0, 1), place(1, 1)]
            .into_iter()
            .chain((0..20).map(|i| place(2 + 2 * i, 2)))
            .chain([place(0, 1)])
            .collect();
        for (len, not, expected) in [
            (2, false, rows_of(&[0, 3])),
            (2, true, rows_of(&[2, 4])),
            (23, false, rows_of(&[0, 3])),
            (23, true, rows_of(&[2, 4])),
        ] {
            let nodes = [Node::Leaf(Leaf::InStrings { column: 0, start: 0, len }), Node::Not(0)];
            let nodes = SlowVec::fixed_from(&Heap, nodes[..=usize::from(not)].iter().copied());
            let predicate =
                Predicate::with_literals(&Heap, nodes.unwrap(), &strings, &places).unwrap();
            let mut entries = predicate.new_entries(&Heap).unwrap();
            assert_eq!(kept_of(&predicate, &mut entries, &flat), expected);
            assert_eq!(kept_of(&predicate, &mut entries, &dictionary), expected);
        }
    }

    #[test]
    fn keeps_all_or_none_when_every_entry_or_none_matches() {
        let dictionary = indexed(&strings(&[Some("a"), Some("c")]), &[1, 0, 1]);
        let kept = |comparison| {
            let predicate = compare_b(comparison, false);
            let mut entries = predicate.new_entries(&Heap).unwrap();
            kept_of(&predicate, &mut entries, &dictionary)
        };
        assert_eq!(kept(Comparison::Equal), []);
        assert_eq!(kept(Comparison::NotEqual), [0, 1, 2]);
    }

    #[test]
    fn keeps_a_dictionarys_entries_while_batches_share_it() {
        let predicate = compare_b(Comparison::Greater, false);
        let mut entries = predicate.new_entries(&Heap).unwrap();
        let values = strings(&[Some("a"), Some("c"), None, Some("d")]);
        let first = indexed(&values, &[0, 1, 2, 3]);
        let second = indexed(&values, &[3, 3, 0, 1]);
        // Another chunk's batch: its dictionary is its own.
        let other = indexed(&strings(&[Some("z"), Some("b")]), &[1, 0, 0, 1]);
        assert!(first.shares_values(&second) && !first.shares_values(&other));
        assert_eq!(kept_of(&predicate, &mut entries, &first), [1, 3]);
        // The second batch reuses the first's entries' bits.
        assert_eq!(kept_of(&predicate, &mut entries, &second), [0, 1, 3]);
        assert_eq!(kept_of(&predicate, &mut entries, &other), [1, 2]);
        assert_eq!(kept_of(&predicate, &mut entries, &first), [1, 3]);
    }

    #[test]
    fn takes_dictionaries_only_of_columns_it_compares_with_strings() {
        let leaves = [
            Leaf::CompareString { column: 0, comparison: Comparison::Equal, start: 0, len: 1 },
            Leaf::Compare { column: 1, comparison: Comparison::Equal, value: Value::Int64(1) },
            Leaf::CompareString { column: 2, comparison: Comparison::Equal, start: 0, len: 1 },
            Leaf::IsNull { column: 2 },
        ];
        let nodes = [
            Node::Leaf(leaves[0]),
            Node::Leaf(leaves[1]),
            Node::Leaf(leaves[2]),
            Node::Leaf(leaves[3]),
            Node::And(0, 1),
            Node::And(2, 3),
            Node::And(4, 5),
        ];
        let nodes = SlowVec::fixed_from(&Heap, nodes.into_iter()).unwrap();
        let predicate = Predicate::with_literals(&Heap, nodes, b"b", &[]).unwrap();
        // A string column, an integer one, one also tested for nulls, and one
        // it doesn't read.
        let only: StdVec<bool> = (0..4).map(|c| predicate.compares_only_strings(c)).collect();
        assert_eq!(only, [true, false, false, false]);
    }
}
