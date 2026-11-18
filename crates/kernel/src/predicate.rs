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
use crate::estimate::Estimate;
use crate::filter::{self, Comparison, Entries, Value};
use crate::flat_hash_map::FlatHashMap;
use crate::murmur_hash;
use crate::row_batch::RowBatch;
use crate::selection::Selection;
use crate::slow_vec::SlowVec;
use crate::step::Transform;
use crate::string_pool::{Id, StringPool};

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
    /// `string`, one of the statement's literals.
    CompareString {
        column: u32,
        comparison: Comparison,
        string: Id,
    },
    /// The string column at `column` in a batch, matched with the `LIKE`
    /// pattern `pattern`, one of the statement's literals.
    Like {
        column: u32,
        pattern: Id,
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
    /// The string column at `column` in a batch, one of the strings whose
    /// ids in the statement's literals are values `start..start + len` of the
    /// predicate's values.
    InStrings {
        column: u32,
        start: u32,
        len: u32,
    },
}

/// Its root is its last node.
pub struct Predicate {
    nodes: SlowVec<Node>,
    /// The values of its `IN` lists, if it has any: shared by its copies.
    values: Option<Buffer>,
    depth: u32,
}

/// An `IN` list of more than `filter::IN_SCAN_MAX` values, kept in a hash
/// map, as Perfetto keeps one, and made once for each run of a query, as
/// Perfetto casts a list. Integers and strings share one map type, so its
/// code is compiled once.
pub type Set = FlatHashMap<filter::Listed, ()>;

impl Predicate {
    /// A predicate of `nodes`, the last of which is its root.
    pub fn new(nodes: SlowVec<Node>) -> Predicate {
        Predicate::checked(nodes, None)
    }

    /// A predicate of `nodes`, whose `IN` lists are of `values`.
    pub fn with_values(
        allocator: &dyn Allocator,
        nodes: SlowVec<Node>,
        values: &[i64],
    ) -> Result<Predicate, AllocError> {
        let mut buffer = None;
        if !values.is_empty() {
            let mut held = Buffer::allocate(allocator, size_of_val(values))?;
            held.as_mut_slice::<i64>().copy_from_slice(values);
            buffer = Some(held);
        }
        Ok(Predicate::checked(nodes, buffer))
    }

    /// For each node, the set of its `IN` list's values if it has more than
    /// `filter::IN_SCAN_MAX` of them, for `select`: none if no list does.
    /// String lists' values are ids in `literals`.
    pub fn new_sets(
        &self,
        allocator: &dyn Allocator,
        literals: Option<&StringPool>,
    ) -> Result<SlowVec<Option<Set>>, AllocError> {
        let long = |node: &Node| match *node {
            Node::Leaf(Leaf::In { len, .. } | Leaf::InStrings { len, .. }) => {
                len as usize > filter::IN_SCAN_MAX
            }
            _ => false,
        };
        if !self.nodes.iter().any(long) {
            return SlowVec::fixed(allocator, 0);
        }
        let mut sets = SlowVec::fixed(allocator, self.nodes.len())?;
        for &node in self.nodes.iter() {
            sets.push(self.set_of(allocator, literals, node)?)?;
        }
        Ok(sets)
    }

    /// The set of node `node`'s values, if it's an `IN` list of more than
    /// `filter::IN_SCAN_MAX` of them.
    fn set_of(
        &self,
        allocator: &dyn Allocator,
        literals: Option<&StringPool>,
        node: Node,
    ) -> Result<Option<Set>, AllocError> {
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
                murmur_hash::bytes(literal(literals, value))
            } else {
                murmur_hash::mix(value.cast_unsigned())
            };
            set.insert(allocator, filter::Listed { hash, value }, ())?;
        }
        Ok(Some(set))
    }

    /// The values of its `IN` lists.
    fn values(&self) -> &[i64] {
        self.values.as_ref().map_or(&[], Buffer::as_slice)
    }

    /// A predicate of `nodes`, checked to be one, and of `values`.
    fn checked(nodes: SlowVec<Node>, values: Option<Buffer>) -> Predicate {
        check!(!nodes.is_empty() && nodes.len() <= PREDICATE_NODES_MAX);
        let value_count =
            values.as_ref().map_or(0, |values| values.size_bytes() / size_of::<i64>());
        // Each string's id is checked to be one of the literals where it's read.
        check!(nodes.iter().all(|node| match *node {
            Node::Leaf(Leaf::In { start, len, .. } | Leaf::InStrings { start, len, .. }) => {
                len != 0
                    && (start as usize)
                        .checked_add(len as usize)
                        .is_some_and(|end| end <= value_count)
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
        Predicate { nodes, values, depth }
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
        let goes = |node| self.reads_only(node, column);
        #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX`")]
        let root = self.copy_where(self.nodes.len() as u32 - 1, &goes, &mut nodes, None)?;
        if root.is_none() {
            return Ok(None);
        }
        Ok(Some(Predicate::checked(nodes, self.values.clone())))
    }

    /// Splits its top-level conjuncts by where they can go. A conjunct whose
    /// columns `place` puts all in the same input, 0 or 1, goes to that
    /// input; any other stays. Returns the conjuncts that stay, then those
    /// for input 0, then for input 1, each joined by `AND`, or `None` if
    /// there are none.
    pub fn split(
        &self,
        allocator: &dyn Allocator,
        place: impl Fn(u32) -> Option<u32>,
    ) -> Result<[Option<Predicate>; 3], AllocError> {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX`")]
        let root = self.nodes.len() as u32 - 1;
        // In the order returned: those that stay, then those for each input.
        let mut parts = [const { None }; 3];
        for (part, input) in parts.iter_mut().zip([None, Some(0), Some(1)]) {
            let mut nodes = SlowVec::new(allocator, PREDICATE_NODES_MAX)?;
            let goes = |node| self.place_of(node, &place) == input;
            if self.copy_where(root, &goes, &mut nodes, None)?.is_some() {
                *part = Some(Predicate::checked(nodes, self.values.clone()));
            }
        }
        Ok(parts)
    }

    /// The input `place` puts all the columns node `node` reads in, if
    /// they're all in the same one.
    fn place_of(&self, node: u32, place: &impl Fn(u32) -> Option<u32>) -> Option<u32> {
        match *at!(self.nodes, node as usize) {
            Node::Leaf(leaf) => place(leaf.column()).filter(|&input| input < 2),
            Node::And(a, b) | Node::Or(a, b) => {
                let input = self.place_of(a, place)?;
                (self.place_of(b, place) == Some(input)).then_some(input)
            }
            Node::Not(a) => self.place_of(a, place),
        }
    }

    /// Copies the conjuncts under node `node` that `goes` accepts into
    /// `nodes`, joining each to `root`, those copied so far, by `AND`.
    /// Returns the new root.
    fn copy_where(
        &self,
        node: u32,
        goes: &dyn Fn(u32) -> bool,
        nodes: &mut SlowVec<Node>,
        root: Option<u32>,
    ) -> Result<Option<u32>, AllocError> {
        if let Node::And(a, b) = *at!(self.nodes, node as usize) {
            let root = self.copy_where(a, goes, nodes, root)?;
            return self.copy_where(b, goes, nodes, root);
        }
        if !goes(node) {
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
        self.only(column, |leaf| {
            matches!(leaf, Leaf::CompareString { .. } | Leaf::Like { .. } | Leaf::InStrings { .. })
        })
    }

    /// Whether every condition it has on `column` compares it with an
    /// integer or tests it for null, and it has one.
    pub fn compares_only_integers(&self, column: u32) -> bool {
        self.only(column, |leaf| {
            matches!(leaf, Leaf::Compare { value: Value::Int64(_), .. } | Leaf::IsNull { .. })
        })
    }

    /// Whether every condition it has on `column` is one `is`, and it has
    /// one.
    fn only(&self, column: u32, is: impl Fn(&Leaf) -> bool) -> bool {
        let leaves = self.nodes.iter().filter_map(|node| match *node {
            Node::Leaf(leaf) => Some(leaf),
            Node::And(..) | Node::Or(..) | Node::Not(_) => None,
        });
        let mut on_column = leaves.filter(|leaf| leaf.column() == column).peekable();
        on_column.peek().is_some() && on_column.all(|leaf| is(&leaf))
    }

    /// The columns its comparisons and `IS NULL`s read, maybe more than once.
    pub fn columns(&self) -> impl Iterator<Item = u32> + '_ {
        self.nodes.iter().filter_map(|node| match *node {
            Node::Leaf(leaf) => Some(leaf.column()),
            Node::And(..) | Node::Or(..) | Node::Not(_) => None,
        })
    }

    /// Applies each of its top-level conjuncts to `estimate`, using the rules
    /// Perfetto's dataframe planner uses for its filters.
    pub fn estimate(&self, estimate: &mut Estimate) {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX`")]
        self.estimate_from(self.nodes.len() as u32 - 1, estimate);
    }

    fn estimate_from(&self, node: u32, estimate: &mut Estimate) {
        match *at!(self.nodes, node as usize) {
            Node::And(a, b) => {
                self.estimate_from(a, estimate);
                self.estimate_from(b, estimate);
            }
            Node::Leaf(
                Leaf::Compare { column, comparison: Comparison::Equal, .. }
                | Leaf::CompareString { column, comparison: Comparison::Equal, .. },
            ) => estimate.keep_equal(column),
            Node::Leaf(Leaf::In { column, len, .. } | Leaf::InStrings { column, len, .. }) => {
                estimate.keep_in(column, u64::from(len));
            }
            Node::Leaf(_) | Node::Or(..) | Node::Not(_) => estimate.keep_some(),
        }
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
            Node::Leaf(Leaf::CompareString { column, comparison, string }) => {
                Node::Leaf(Leaf::CompareString { column: renumber(column), comparison, string })
            }
            Node::Leaf(Leaf::Like { column, pattern }) => {
                Node::Leaf(Leaf::Like { column: renumber(column), pattern })
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
        Ok(Predicate::checked(nodes, self.values.clone()))
    }

    /// Where each node keeps the dictionary entries it found, for `select`.
    pub fn new_entries(&self, allocator: &dyn Allocator) -> Result<SlowVec<Entries>, AllocError> {
        SlowVec::fixed_from(allocator, (0..self.nodes.len()).map(|_| Entries::default()))
    }

    /// Narrows `batch`'s selection to the rows this is true for, working in
    /// `scratch`, which holds at least `depth` selections. Its strings are
    /// ids in `literals`, and its long `IN` lists in `sets`, made by
    /// `new_sets`. Its leaves keep the entries of dictionary columns they find
    /// in `entries`, made by `new_entries`, allocating from `allocator`.
    #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX` nodes")]
    #[inline]
    pub fn select(
        &self,
        allocator: &dyn Allocator,
        literals: Option<&StringPool>,
        sets: &[Option<Set>],
        scratch: &mut [Selection],
        entries: &mut [Entries],
        batch: &mut RowBatch,
    ) -> Result<(), AllocError> {
        check!(scratch.len() >= self.depth as usize && entries.len() == self.nodes.len());
        let (columns, selection) = batch.columns_and_selection();
        let values = self.values();
        let mut inputs = Inputs { columns, literals, values, sets, entries, allocator };
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
    /// The statement's literals, which its strings are ids in.
    literals: Option<&'b StringPool>,
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
            Leaf::CompareString { column, comparison, string } => {
                let comparison = if want { comparison } else { negated(comparison) };
                let string = literal(inputs.literals, i64::from(string.raw_id()));
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
            Leaf::Like { column, pattern } => {
                let pattern = literal(inputs.literals, i64::from(pattern.raw_id()));
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
/// strings whose ids are `values`, is `want` for, as `Leaf::narrow` does.
fn in_strings(
    inputs: &mut Inputs<'_>,
    node: u32,
    column: u32,
    values: &[i64],
    dropped: Option<&mut Selection>,
    want: bool,
    selection: &mut Selection,
) -> Result<(), AllocError> {
    let literals = inputs.literals;
    let string = move |id: i64| literal(literals, id);
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

/// The literal whose id in `literals` is `id`, which must be one.
fn literal(literals: Option<&StringPool>, id: i64) -> &[u8] {
    let Ok(id) = u32::try_from(id) else { failed!() };
    let Some(string) = literals.and_then(|literals| literals.get(Id::raw(id))) else { failed!() };
    string
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

/// What a run of a `Filter` keeps between batches.
pub struct FilterState {
    /// The sets of its predicate's long `IN` lists.
    sets: SlowVec<Option<Set>>,
    /// Where each node keeps the dictionary entries it found.
    entries: SlowVec<Entries>,
}

impl Transform for Filter {
    type State = FilterState;

    fn new_state(&self, context: &mut Context) -> Result<FilterState, Error> {
        context.reserve_selections(self.predicate.depth() as usize)?;
        let allocator = context.allocator();
        let sets = self.predicate.new_sets(allocator, context.literals())?;
        Ok(FilterState { sets, entries: self.predicate.new_entries(allocator)? })
    }

    fn process(
        &self,
        context: &mut Context,
        state: &mut FilterState,
        batch: &mut RowBatch,
    ) -> Result<(), Error> {
        let (allocator, literals) = (context.allocator(), context.literals());
        let FilterState { sets, entries } = state;
        Ok(self.predicate.select(
            allocator,
            literals,
            sets,
            context.selections(),
            entries,
            batch,
        )?)
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
        let sets = predicate.new_sets(&Heap, None).unwrap();
        let mut batch = RowBatch::new();
        batch.reset(6);
        assert!(batch.push_column(column(A)).is_ok() && batch.push_column(column(B)).is_ok());
        predicate.select(&Heap, None, &sets, &mut scratch, &mut entries, &mut batch).unwrap();
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
            let nodes = SlowVec::fixed_from(&Heap, nodes.iter().copied()).unwrap();
            kept_by(&Predicate::with_values(&Heap, nodes, &values).unwrap())
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

    /// A statement's literals, holding `strings`.
    fn literals(strings: &[&[u8]]) -> StringPool {
        let mut literals = StringPool::new(&Heap).unwrap();
        for &string in strings {
            literals.intern_string(&Heap, Some(string)).unwrap();
        }
        literals
    }

    /// The id of `string`, one of `literals`.
    fn id(literals: &mut StringPool, string: &[u8]) -> Id {
        literals.intern_string(&Heap, Some(string)).unwrap()
    }

    /// `x <comparison> 'b'`, or `NOT` that, over the first column, and its
    /// literals.
    fn compare_b(comparison: Comparison, not: bool) -> (Predicate, StringPool) {
        let mut literals = literals(&[]);
        let string = id(&mut literals, b"b");
        let leaf = Leaf::CompareString { column: 0, comparison, string };
        let nodes = [Node::Leaf(leaf), Node::Not(0)];
        let nodes = SlowVec::fixed_from(&Heap, nodes[..=usize::from(not)].iter().copied());
        (Predicate::new(nodes.unwrap()), literals)
    }

    /// The rows `predicate`, with `literals`, keeps of a batch of `column`,
    /// its entries kept in `entries` between batches.
    fn kept_of(
        (predicate, literals): (&Predicate, &StringPool),
        entries: &mut [Entries],
        column: &ColumnView,
    ) -> StdVec<u16> {
        let mut scratch: StdVec<Selection> =
            (0..predicate.depth()).map(|_| Selection::all(0)).collect();
        let sets = predicate.new_sets(&Heap, Some(literals)).unwrap();
        let mut batch = RowBatch::new();
        batch.reset(column.row_count());
        assert!(batch.push_column(column.clone()).is_ok());
        let literals = Some(literals);
        predicate.select(&Heap, literals, &sets, &mut scratch, entries, &mut batch).unwrap();
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
                let (predicate, literals) = compare_b(comparison, not);
                let mut entries = predicate.new_entries(&Heap).unwrap();
                let expected = kept_of((&predicate, &literals), &mut entries, &flat);
                assert_eq!(kept_of((&predicate, &literals), &mut entries, &dictionary), expected);
            }
        }
        // `x LIKE '_'` keeps entries 0, 2 and 3, and `NOT x LIKE '_'` entry 4:
        // neither keeps the null, entry 1.
        let rows_of = |kept: &[u32]| -> StdVec<u16> {
            (0..).zip(&indices).filter(|(_, e)| kept.contains(e)).map(|(row, _)| row).collect()
        };
        for (not, expected) in [(false, rows_of(&[0, 2, 3])), (true, rows_of(&[4]))] {
            let mut literals = literals(&[]);
            let pattern = id(&mut literals, b"_");
            let nodes = [Node::Leaf(Leaf::Like { column: 0, pattern }), Node::Not(0)];
            let nodes = SlowVec::fixed_from(&Heap, nodes[..=usize::from(not)].iter().copied());
            let predicate = Predicate::new(nodes.unwrap());
            let mut entries = predicate.new_entries(&Heap).unwrap();
            assert_eq!(kept_of((&predicate, &literals), &mut entries, &flat), expected);
            assert_eq!(kept_of((&predicate, &literals), &mut entries, &dictionary), expected);
        }
        // `x IN ('b', 'c')` keeps entries 0 and 3, and `NOT` that entries 2 and
        // 4, whether the list is scanned, or long enough for a set: 'b' and
        // 'c' then 20 strings of two bytes, and 'b' again.
        let mut literals = literals(&[]);
        let strings: StdVec<StdVec<u8>> = [b"b".to_vec(), b"c".to_vec()]
            .into_iter()
            .chain((0..20).map(|i| [b'x', i].to_vec()))
            .chain([b"b".to_vec()])
            .collect();
        let ids: StdVec<i64> =
            strings.iter().map(|string| i64::from(id(&mut literals, string).raw_id())).collect();
        for (len, not, expected) in [
            (2, false, rows_of(&[0, 3])),
            (2, true, rows_of(&[2, 4])),
            (23, false, rows_of(&[0, 3])),
            (23, true, rows_of(&[2, 4])),
        ] {
            let nodes = [Node::Leaf(Leaf::InStrings { column: 0, start: 0, len }), Node::Not(0)];
            let nodes = SlowVec::fixed_from(&Heap, nodes[..=usize::from(not)].iter().copied());
            let predicate = Predicate::with_values(&Heap, nodes.unwrap(), &ids).unwrap();
            let mut entries = predicate.new_entries(&Heap).unwrap();
            assert_eq!(kept_of((&predicate, &literals), &mut entries, &flat), expected);
            assert_eq!(kept_of((&predicate, &literals), &mut entries, &dictionary), expected);
        }
    }

    #[test]
    fn keeps_all_or_none_when_every_entry_or_none_matches() {
        let dictionary = indexed(&strings(&[Some("a"), Some("c")]), &[1, 0, 1]);
        let kept = |comparison| {
            let (predicate, literals) = compare_b(comparison, false);
            let mut entries = predicate.new_entries(&Heap).unwrap();
            kept_of((&predicate, &literals), &mut entries, &dictionary)
        };
        assert_eq!(kept(Comparison::Equal), []);
        assert_eq!(kept(Comparison::NotEqual), [0, 1, 2]);
    }

    #[test]
    fn keeps_a_dictionarys_entries_while_batches_share_it() {
        let (predicate, literals) = compare_b(Comparison::Greater, false);
        let predicate = (&predicate, &literals);
        let mut entries = predicate.0.new_entries(&Heap).unwrap();
        let values = strings(&[Some("a"), Some("c"), None, Some("d")]);
        let first = indexed(&values, &[0, 1, 2, 3]);
        let second = indexed(&values, &[3, 3, 0, 1]);
        // Another chunk's batch: its dictionary is its own.
        let other = indexed(&strings(&[Some("z"), Some("b")]), &[1, 0, 0, 1]);
        assert!(first.shares_values(&second) && !first.shares_values(&other));
        assert_eq!(kept_of(predicate, &mut entries, &first), [1, 3]);
        // The second batch reuses the first's entries' bits.
        assert_eq!(kept_of(predicate, &mut entries, &second), [0, 1, 3]);
        assert_eq!(kept_of(predicate, &mut entries, &other), [1, 2]);
        assert_eq!(kept_of(predicate, &mut entries, &first), [1, 3]);
    }

    #[test]
    fn takes_dictionaries_only_of_columns_it_compares_with_strings() {
        let string = id(&mut literals(&[]), b"b");
        let leaves = [
            Leaf::CompareString { column: 0, comparison: Comparison::Equal, string },
            Leaf::Compare { column: 1, comparison: Comparison::Equal, value: Value::Int64(1) },
            Leaf::CompareString { column: 2, comparison: Comparison::Equal, string },
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
        let predicate = Predicate::new(nodes);
        // A string column, an integer one, one also tested for nulls, and one
        // it doesn't read.
        let only: StdVec<bool> = (0..4).map(|c| predicate.compares_only_strings(c)).collect();
        assert_eq!(only, [true, false, false, false]);
    }
}
