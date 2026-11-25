//! `Predicate`: a condition on a batch's rows, such as `WHERE`'s, which
//! narrows its selection to the rows the condition is true for.
//!
//! Each node can narrow a selection to the rows it's true for, or to those
//! it's false for. A null row is neither, as in SQL, so `NOT` swaps the two
//! and every node keeps SQL's three-valued logic without building booleans.

use crate::allocator::AllocError;
use crate::buffer::Buffer;
use crate::column::DataType;
use crate::column::{ColumnView, Form, Forms};
use crate::containers::{SmallVec, SpillVec};
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;
use crate::expression::{Compute, ComputeState, EXPR_NODES_MAX, Expr, ExprOp, ExprStep, Exprs};
use crate::filter::{self, Comparison, Entries, Value};
use crate::flat_hash_map::FlatHashMap;
use crate::murmur_hash;
use crate::plan::ColumnId;
use crate::row_batch::RowBatch;
use crate::selection::Selection;
use crate::step::Transform;
use crate::string_pool::{Id, StringPool};

/// The most nodes a predicate can have.
pub const PREDICATE_NODES_MAX: usize = EXPR_NODES_MAX;

/// How many nodes' depths finding a predicate's depth holds in place.
const DEPTHS_INLINE: usize = 256;

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
    Compare { column: u32, comparison: Comparison, value: Value },
    /// The string column at `column` in a batch, compared byte by byte with
    /// `string`, one of the statement's literals.
    CompareString { column: u32, comparison: Comparison, string: Id },
    /// The string column at `column` in a batch, matched with the `LIKE`
    /// pattern `pattern`, one of the statement's literals.
    Like { column: u32, pattern: Id },
    /// The column at `column` in a batch, null.
    IsNull { column: u32 },
    /// The integer column at `column` in a batch, one of values
    /// `start..start + len` of the predicate's values.
    In { column: u32, start: u32, len: u32 },
    /// The string column at `column` in a batch, one of the strings whose
    /// ids in the statement's literals are values `start..start + len` of the
    /// predicate's values.
    InStrings { column: u32, start: u32, len: u32 },
}

/// Its root is its last node.
pub struct Predicate {
    /// Its leaves, and the `And`, `Or` and `Not` nodes over them.
    nodes: SmallVec<Node>,
    /// The values of its `IN` lists, if it has any: shared by its copies.
    values: Option<Buffer>,
    /// How many scratch selections testing it holds at once, at most.
    depth: u32,
}

/// An `IN` list of more than `filter::IN_SCAN_MAX` values, kept in a hash
/// map, as Perfetto keeps one, and made once for each run of a query, as
/// Perfetto casts a list. Integers and strings share one map type, so its
/// code is compiled once.
pub type Set = FlatHashMap<filter::Listed, ()>;

impl Predicate {
    /// A predicate of `nodes`, the last of which is its root.
    pub fn new(env: &Env, nodes: SmallVec<Node>) -> Result<Predicate, AllocError> {
        Predicate::checked(env, nodes, None)
    }

    /// A predicate of `nodes`, whose `IN` lists are of `values`.
    pub fn with_values(
        env: &Env,
        nodes: SmallVec<Node>,
        values: &[i64],
    ) -> Result<Predicate, AllocError> {
        let mut buffer = None;
        if !values.is_empty() {
            let mut held = Buffer::allocate(env, size_of_val(values))?;
            held.as_mut_slice::<i64>().copy_from_slice(values);
            buffer = Some(held);
        }
        Predicate::checked(env, nodes, buffer)
    }

    /// A predicate of one leaf, whose `IN` lists are of `values`.
    pub(crate) fn of_leaf(
        env: &Env,
        leaf: Leaf,
        values: Option<Buffer>,
    ) -> Result<Predicate, AllocError> {
        let nodes = SmallVec::fixed_from(env, [Node::Leaf(leaf)].into_iter())?;
        Predicate::checked(env, nodes, values)
    }

    /// For each node, the set of its `IN` list's values if it has more than
    /// `filter::IN_SCAN_MAX` of them, for `select`: none if no list does.
    /// String lists' values are ids in `literals`.
    pub fn new_sets(
        &self,
        env: &Env,
        literals: Option<&StringPool>,
    ) -> Result<SmallVec<Option<Set>>, AllocError> {
        let long = |node: &Node| match *node {
            Node::Leaf(Leaf::In { len, .. } | Leaf::InStrings { len, .. }) => {
                len as usize > filter::IN_SCAN_MAX
            }
            _ => false,
        };
        if !self.nodes.iter().any(long) {
            return SmallVec::fixed(env, 0);
        }
        let mut sets = SmallVec::fixed(env, self.nodes.len())?;
        for &node in self.nodes.iter() {
            sets.push(self.set_of(env, literals, node)?)?;
        }
        Ok(sets)
    }

    /// The set of node `node`'s values, if it's an `IN` list of more than
    /// `filter::IN_SCAN_MAX` of them.
    fn set_of(
        &self,
        env: &Env,
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
        let mut set = FlatHashMap::for_len(env, len as usize)?;
        // A value listed twice is put in twice, which is harmless.
        for &value in at!(self.values(), start as usize..(start + len) as usize) {
            let hash = if strings {
                murmur_hash::bytes(literal(literals, value))
            } else {
                murmur_hash::mix(value.cast_unsigned())
            };
            set.insert(env, filter::Listed { hash, value }, ())?;
        }
        Ok(Some(set))
    }

    /// The values of its `IN` lists.
    fn values(&self) -> &[i64] {
        self.values.as_ref().map_or(&[], Buffer::as_slice)
    }

    /// A predicate of `nodes`, checked to be one, and of `values`.
    fn checked(
        env: &Env,
        nodes: SmallVec<Node>,
        values: Option<Buffer>,
    ) -> Result<Predicate, AllocError> {
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
        // Each node's depth, wanted true and wanted false: how many
        // selections `narrow` holds at once for it, at most. An `And` wanted
        // true, or an `Or` wanted false, narrows in place; the other way, it
        // holds one while its children run, for the rows left.
        let len = nodes.len();
        let mut held =
            SpillVec::<[u32; 2], DEPTHS_INLINE>::filled(env, PREDICATE_NODES_MAX, len, [0; 2])?;
        let depths: &mut [[u32; 2]] = &mut held;
        for (i, node) in (0..).zip(nodes.iter()) {
            let depth = |child: u32| *at!(depths, child as usize);
            *at_mut!(depths, i as usize) = match *node {
                Node::Leaf(_) => [0, 0],
                Node::And(a, b) | Node::Or(a, b) => {
                    check!(a < i && b < i);
                    let [a, b] = [depth(a), depth(b)];
                    // Its children are narrowed for what it's wanted for.
                    let [wanted_true, wanted_false] = [a[0].max(b[0]), a[1].max(b[1])];
                    if matches!(node, Node::And(..)) {
                        [wanted_true, wanted_false + 1]
                    } else {
                        [wanted_true + 1, wanted_false]
                    }
                }
                Node::Not(a) => {
                    check!(a < i);
                    let [true_, false_] = depth(a);
                    [false_, true_]
                }
            };
        }
        let depth = at!(depths, nodes.len() - 1)[0];
        Ok(Predicate { nodes, values, depth })
    }

    /// Its nodes, the last its root.
    pub(crate) fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// How many scratch selections `select` needs.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Where each node keeps the dictionary entries it found, for `select`.
    pub fn new_entries(&self, env: &Env) -> Result<SmallVec<Entries>, AllocError> {
        SmallVec::fixed_from(env, (0..self.nodes.len()).map(|_| Entries::default()))
    }

    /// Narrows `batch`'s selection to the rows this is true for, working in
    /// `scratch`, which holds at least `depth` selections. Its strings are
    /// ids in `literals`, and its long `IN` lists in `sets`, made by
    /// `new_sets`. Its leaves keep the entries of dictionary columns they find
    /// in `entries`, made by `new_entries`, allocating from `env`.
    #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX` nodes")]
    #[inline]
    pub fn select(
        &self,
        env: &Env,
        literals: Option<&StringPool>,
        sets: &[Option<Set>],
        scratch: &mut [Selection],
        entries: &mut [Entries],
        batch: &mut RowBatch,
    ) -> Result<(), AllocError> {
        check!(scratch.len() >= self.depth as usize && entries.len() == self.nodes.len());
        let (columns, selection) = batch.columns_and_selection();
        let values = self.values();
        let mut inputs = Inputs { columns, literals, values, sets, entries, env };
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
        let (mut a, mut b) = (a, b);
        loop {
            self.narrow(inputs, scratch, a, want, selection)?;
            // The rest of a chain, such as a filter's conjuncts, in a loop
            // rather than recursing.
            match *at!(self.nodes, b as usize) {
                Node::And(next, rest) if want => (a, b) = (next, rest),
                Node::Or(next, rest) if !want => (a, b) = (next, rest),
                _ => return self.narrow(inputs, scratch, b, want, selection),
            }
        }
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
    env: &'b Env,
}

impl Leaf {
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
                let comparison = if want { comparison } else { comparison.negated() };
                let column = at!(inputs.columns, column as usize);
                filter::compare(column, comparison, value, selection, dropped);
            }
            Leaf::CompareString { column, comparison, string } => {
                let comparison = if want { comparison } else { comparison.negated() };
                let string = literal(inputs.literals, i64::from(string.raw_id()));
                let column = at!(inputs.columns, column as usize);
                if !matches!(column.form(), Form::Dictionary(_)) {
                    filter::compare_string(column, comparison, string, selection, dropped);
                    return Ok(());
                }
                let entries = at_mut!(inputs.entries, node as usize);
                let entries = entries.find(inputs.env, column, want, &|values, bits| {
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
                let entries = entries.find(inputs.env, column, want, &|values, bits| {
                    filter::like_bits(values, pattern, want, bits);
                })?;
                filter::keep_entries(column, entries, selection, dropped);
            }
            Leaf::IsNull { column } => {
                let column = at!(inputs.columns, column as usize);
                if !matches!(column.form(), Form::Dictionary(_)) {
                    filter::is_null(column, want, selection, dropped);
                    return Ok(());
                }
                let entries = at_mut!(inputs.entries, node as usize);
                let entries = entries.find(inputs.env, column, want, &|values, bits| {
                    filter::null_bits(values, want, bits);
                })?;
                filter::keep_entries(column, entries, selection, dropped);
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
    let entries = entries.find(inputs.env, column, want, &|values, bits| {
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

/// The forms a filter of `conjuncts` takes `column` in: dictionaries if it
/// only compares the column with strings, whose entries it tests once each;
/// sequences if only with integers, whose rows it keeps by their numbers; and
/// any form if it only tests it for null, which reads neither.
pub fn tested_forms(exprs: &Exprs, conjuncts: &[Expr], column: ColumnId) -> Forms {
    // `None` before any test but for null, then the kind, if one.
    let mut found = None;
    let mut nulls = false;
    let mut test = |tested| {
        found = Some(match found {
            Some(before) if before != tested => None,
            _ => tested,
        });
    };
    for &conjunct in conjuncts {
        let nodes = exprs.nodes(conjunct);
        let reads = |i: u32| {
            let node = at!(nodes, i as usize);
            node.op == ExprOp::Column && node.a == column
        };
        let constant = |i: u32| {
            let node = at!(nodes, i as usize);
            (node.op == ExprOp::Constant).then_some(node.data_type)
        };
        // A condition that's the column itself tests it as an integer.
        if reads(conjunct.root) {
            test(Some(DataType::Int64));
        }
        for node in nodes {
            for (first, count) in node.operands() {
                for read in (first..first + count).filter(|&read| reads(read)) {
                    test(match node.op {
                        ExprOp::IsNull | ExprOp::IsNotNull => {
                            nulls = true;
                            continue;
                        }
                        ExprOp::And | ExprOp::Or | ExprOp::Not => Some(DataType::Int64),
                        ExprOp::Eq
                        | ExprOp::Ne
                        | ExprOp::Lt
                        | ExprOp::Le
                        | ExprOp::Gt
                        | ExprOp::Ge => constant(if read == node.a { node.b } else { node.a }),
                        ExprOp::Like => Some(DataType::String),
                        ExprOp::In if node.data_type == DataType::String => Some(DataType::String),
                        _ => None,
                    });
                }
            }
        }
    }
    match found {
        None if nulls => Forms::FLAT | Forms::CONSTANT | Forms::DICTIONARY | Forms::SEQUENCE,
        Some(Some(DataType::String)) => Forms::FLAT | Forms::CONSTANT | Forms::DICTIONARY,
        Some(Some(DataType::Int64)) => Forms::FLAT | Forms::CONSTANT | Forms::SEQUENCE,
        _ => Forms::FLAT | Forms::CONSTANT,
    }
}

impl Predicate {
    /// The predicate a filter of `conjuncts`, all true, tests, reading each
    /// column `c` at `column(c)`, its position and type, and its `IN` lists
    /// from `values`. A node it can't test as a leaf, such as `a < b`, it
    /// reads as a column from position `computed_at` on, and adds to
    /// `computed`: the filter computes those first, as 0, 1 or null, and
    /// keeps the rows where they aren't 0, as SQL does.
    pub fn from_expr(
        env: &Env,
        exprs: &Exprs,
        conjuncts: &[Expr],
        values: Option<Buffer>,
        column: &dyn Fn(ColumnId) -> (u32, DataType),
        computed_at: u32,
        computed: &mut SmallVec<Expr>,
    ) -> Result<Predicate, AllocError> {
        let mut nodes = SmallVec::new(env, PREDICATE_NODES_MAX)?;
        let mut root = None;
        // From the last, joined from the right, so `narrow` tests them in
        // order in a loop.
        for &conjunct in conjuncts.iter().rev() {
            let start = conjunct.start;
            let expr = exprs.nodes(conjunct);
            let mut from =
                FromExpr { expr, start, column, computed_at, computed, nodes: &mut nodes };
            let added = from.add(conjunct.root)?;
            root = Some(match root {
                Some(root) => push(&mut nodes, Node::And(added, root))?,
                None => added,
            });
        }
        check!(root.is_some());
        Predicate::checked(env, nodes, values)
    }
}

/// Turns an expression's nodes, from `start` of its plan's, into a
/// predicate's, as `Predicate::from_expr`.
struct FromExpr<'e> {
    expr: &'e [ExprStep],
    start: u32,
    column: &'e dyn Fn(ColumnId) -> (u32, DataType),
    computed_at: u32,
    computed: &'e mut SmallVec<Expr>,
    nodes: &'e mut SmallVec<Node>,
}

impl FromExpr<'_> {
    /// Adds node `node`'s predicate nodes, and returns its own.
    fn add(&mut self, node: u32) -> Result<u32, AllocError> {
        let step = *at!(self.expr, node as usize);
        let added = match step.op {
            ExprOp::And | ExprOp::Or => {
                let (a, b) = (self.add(step.a)?, self.add(step.b)?);
                if step.op == ExprOp::And { Node::And(a, b) } else { Node::Or(a, b) }
            }
            ExprOp::Not => Node::Not(self.add(step.a)?),
            _ => {
                if let Some(leaf) = self.leaf(&step) {
                    // `IS NOT NULL` is `NOT` of its leaf, `IS NULL`.
                    if step.op != ExprOp::IsNotNull {
                        return push(self.nodes, Node::Leaf(leaf));
                    }
                    Node::Not(push(self.nodes, Node::Leaf(leaf))?)
                } else {
                    #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
                    let at = self.computed_at + self.computed.len() as u32;
                    self.computed.push(Expr { start: self.start, root: node })?;
                    let (comparison, value) = (Comparison::NotEqual, Value::Int64(0));
                    Node::Leaf(Leaf::Compare { column: at, comparison, value })
                }
            }
        };
        push(self.nodes, added)
    }

    /// The leaf testing `step`, if a predicate can: an integer column itself,
    /// a column compared with a constant, or `LIKE`, `IS NULL`, `IS NOT
    /// NULL`'s null test or `IN` of a column.
    fn leaf(&self, step: &ExprStep) -> Option<Leaf> {
        let node = |i: u32| *at!(self.expr, i as usize);
        let column = |i: u32| {
            let read = node(i);
            (read.op == ExprOp::Column).then(|| (self.column)(read.a))
        };
        match step.op {
            ExprOp::Column => {
                let (column, data_type) = (self.column)(step.a);
                let value = Value::Int64(0);
                (data_type == DataType::Int64).then_some(Leaf::Compare {
                    column,
                    comparison: Comparison::NotEqual,
                    value,
                })
            }
            ExprOp::Like => {
                let (column, _) = column(step.a)?;
                Some(Leaf::Like { column, pattern: Id::raw(step.b) })
            }
            ExprOp::IsNull | ExprOp::IsNotNull => Some(Leaf::IsNull { column: column(step.a)?.0 }),
            ExprOp::In => {
                let (column, data_type) = column(step.a)?;
                let (start, len) = (step.b, step.c);
                match data_type {
                    _ if data_type != step.data_type => None,
                    DataType::Int64 => Some(Leaf::In { column, start, len }),
                    DataType::String => Some(Leaf::InStrings { column, start, len }),
                    DataType::Float64 => None,
                }
            }
            ExprOp::Eq | ExprOp::Ne | ExprOp::Lt | ExprOp::Le | ExprOp::Gt | ExprOp::Ge => {
                let Some(comparison) = step.op.comparison() else { failed!() };
                // A constant on the left is as one on the right the other way.
                let ((column, data_type), constant, comparison) =
                    match (column(step.a), column(step.b)) {
                        (Some(read), None) => (read, node(step.b), comparison),
                        (None, Some(read)) => (read, node(step.a), comparison.flipped()),
                        _ => return None,
                    };
                if constant.op != ExprOp::Constant {
                    return None;
                }
                #[expect(clippy::cast_precision_loss, reason = "typing checked it's exact")]
                let value = match (data_type, constant.data_type) {
                    (DataType::String, DataType::String) => {
                        let string = Id::raw(constant.a);
                        return Some(Leaf::CompareString { column, comparison, string });
                    }
                    (DataType::Int64, DataType::Int64) => Value::Int64(constant.int_value),
                    (DataType::Float64, DataType::Int64) => {
                        Value::Float64(constant.int_value as f64)
                    }
                    _ => return None,
                };
                Some(Leaf::Compare { column, comparison, value })
            }
            _ => None,
        }
    }
}

/// Adds `node` to `nodes`, returning where it is.
fn push(nodes: &mut SmallVec<Node>, node: Node) -> Result<u32, AllocError> {
    #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX`")]
    let at = nodes.len() as u32;
    nodes.push(node)?;
    Ok(at)
}

/// Narrows each batch's selection to the rows `predicate` is true for. The
/// predicate reads columns by their position in batches.
pub struct Filter {
    /// Computes what `predicate` reads after a batch's own columns, which
    /// are dropped once it's tested.
    pub compute: Option<Compute>,
    /// What a row must be true for to be kept.
    pub predicate: Predicate,
}

/// What a run of a `Filter` keeps between batches.
pub struct FilterState {
    /// The sets of its predicate's long `IN` lists.
    sets: SmallVec<Option<Set>>,
    /// Where each node keeps the dictionary entries it found.
    entries: SmallVec<Entries>,
    /// What computing keeps.
    compute: Option<ComputeState>,
}

impl Transform for Filter {
    const NAME: &'static str = "Filter";
    type State = FilterState;

    fn new_state(&self, context: &mut Context) -> Result<FilterState, Error> {
        context.reserve_selections(self.predicate.depth() as usize)?;
        let env = context.env();
        let sets = self.predicate.new_sets(env, context.literals())?;
        let entries = self.predicate.new_entries(env)?;
        let compute = match &self.compute {
            Some(compute) => Some(compute.new_state(context)?),
            None => None,
        };
        Ok(FilterState { sets, entries, compute })
    }

    fn process(
        &self,
        context: &mut Context,
        state: &mut FilterState,
        batch: &mut RowBatch,
    ) -> Result<(), Error> {
        let FilterState { sets, entries, compute } = state;
        let computed = match (&self.compute, compute) {
            (Some(compute), Some(state)) => {
                compute.process(context, state, batch)?;
                compute.programs.len()
            }
            _ => 0,
        };
        let (env, literals) = (context.env(), context.literals());
        self.predicate.select(env, literals, sets, context.selections(), entries, batch)?;
        for _ in 0..computed {
            batch.pop_column();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec as StdVec;

    use super::*;

    use crate::buffer::Buffer;
    use crate::column::{DataType, Forms};
    use crate::context::Context;
    use crate::expression::ExprBuilder;
    use crate::selection::Kept;

    /// `a` and `b`, with `None` for null.
    const A: [Option<i64>; 6] = [Some(1), None, Some(3), None, Some(5), Some(3)];
    const B: [Option<i64>; 6] = [None, Some(2), None, Some(4), Some(5), Some(1)];

    fn column(cells: [Option<i64>; 6]) -> ColumnView {
        ColumnView::from_int64s(&mut Context::new(Env::current(), None), cells.into_iter()).unwrap()
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
    fn build(expr: &Expr, nodes: &mut SmallVec<Node>) -> u32 {
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
        let env = Env::current();
        let mut nodes = SmallVec::new(env, PREDICATE_NODES_MAX).unwrap();
        build(expr, &mut nodes);
        Predicate::new(env, nodes).unwrap()
    }

    fn kept(expr: &Expr) -> StdVec<usize> {
        kept_by(&predicate_of(expr))
    }

    /// The rows of `a` and `b` `predicate` keeps.
    fn kept_by(predicate: &Predicate) -> StdVec<usize> {
        let env = Env::current();
        let mut scratch: StdVec<Selection> =
            (0..predicate.depth()).map(|_| Selection::all(0)).collect();
        let mut entries = predicate.new_entries(env).unwrap();
        let sets = predicate.new_sets(env, None).unwrap();
        let mut batch = RowBatch::new();
        batch.reset(6);
        assert!(batch.push_column(column(A)).is_ok() && batch.push_column(column(B)).is_ok());
        predicate.select(env, None, &sets, &mut scratch, &mut entries, &mut batch).unwrap();
        match batch.selection().kept() {
            Kept::Prefix => (0..6).collect(),
            Kept::None => StdVec::new(),
            Kept::Select(rows) => rows.iter().map(|&row| usize::from(row)).collect(),
        }
    }

    /// `expr` as a kernel expression, added to `exprs`, as a plan holds a
    /// condition.
    fn kernel(exprs: &mut Exprs, expr: &Expr) -> crate::expression::Expr {
        fn add(expr: &Expr, nodes: &mut ExprBuilder) -> u32 {
            let mut push = |op, a, b, value| nodes.push(ExprStep::new(op, a, b, value)).unwrap();
            match *expr {
                Expr::Greater(column, value) => {
                    let column = push(ExprOp::Column, column, 0, 0);
                    let value = push(ExprOp::Constant, 0, 0, value);
                    push(ExprOp::Gt, column, value, 0)
                }
                Expr::IsNull(column) => {
                    let column = push(ExprOp::Column, column, 0, 0);
                    push(ExprOp::IsNull, column, 0, 0)
                }
                Expr::And(a, b) | Expr::Or(a, b) => {
                    let (a, b) = (add(a, nodes), add(b, nodes));
                    let op = if matches!(expr, Expr::And(..)) { ExprOp::And } else { ExprOp::Or };
                    nodes.push(ExprStep::new(op, a, b, 0)).unwrap()
                }
                Expr::Not(a) => {
                    let a = add(a, nodes);
                    nodes.push(ExprStep::new(ExprOp::Not, a, 0, 0)).unwrap()
                }
            }
        }
        let mut nodes = ExprBuilder::new(Env::current()).unwrap();
        add(expr, &mut nodes);
        exprs.add(nodes.nodes()).unwrap()
    }

    /// The predicate testing `conjuncts`, of `exprs`, of columns at their
    /// ids' positions.
    fn from_kernel(exprs: &Exprs, conjuncts: &[crate::expression::Expr]) -> Predicate {
        let env = Env::current();
        let mut computed = SmallVec::new(env, 1).unwrap();
        let column = |column| (column, DataType::Int64);
        let predicate =
            Predicate::from_expr(env, exprs, conjuncts, None, &column, 2, &mut computed).unwrap();
        assert!(computed.is_empty());
        predicate
    }

    #[test]
    fn tests_what_a_filter_gives_a_column() {
        use crate::plan::{FilterOp, Given, Op};
        const A2: Expr = Expr::Greater(0, 2);
        const A4_OR_NULL: Expr = Expr::Or(&Expr::Greater(0, 4), &Expr::IsNull(0));
        const EITHER: Expr = Expr::Or(&A2, &Expr::Greater(1, 2));
        // `(a > 2 AND b IS NULL) AND (a > 4 OR a IS NULL) AND (a > 2 OR b > 2)`.
        const ALL: Expr =
            Expr::And(&Expr::And(&Expr::And(&A2, &Expr::IsNull(1)), &A4_OR_NULL), &EITHER);
        let env = Env::current();
        let mut exprs = Exprs::new(env).unwrap();
        let all = kernel(&mut exprs, &ALL);
        assert_eq!(kept_by(&from_kernel(&exprs, &[all])), kept(&ALL));
        let mut given = |expr: &Expr, column| {
            let condition = kernel(&mut exprs, expr);
            let mut filter = FilterOp::new(env, &exprs, condition).unwrap();
            let given = filter.give_condition(&exprs, env, column).unwrap();
            let Given::Passed(condition) = given else {
                panic!("a filter passes on its child's rows")
            };
            condition.map(|condition| kept_by(&from_kernel(&exprs, &condition)))
        };
        assert_eq!(given(&ALL, 0), Some(kept(&Expr::And(&A2, &A4_OR_NULL))));
        // None reads only a column it doesn't read.
        assert_eq!(given(&ALL, 2), None);
        // One conjunct on the column is all of it.
        assert_eq!(given(&A4_OR_NULL, 0), Some(kept(&A4_OR_NULL)));
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
        // A scratch selection for each `OR` wanted true, or `AND` wanted
        // false, nested in another: an `AND` wanted true narrows in place.
        let depth = |expr: &Expr| {
            let env = Env::current();
            let mut nodes = SmallVec::new(env, PREDICATE_NODES_MAX).unwrap();
            build(expr, &mut nodes);
            Predicate::new(env, nodes).unwrap().depth()
        };
        assert_eq!([&A2, &BOTH, &EITHER, exprs[4], exprs[8]].map(depth), [0, 0, 1, 1, 1]);
        for expr in exprs {
            let expected: StdVec<usize> =
                (0..6).filter(|&row| reference(expr, row) == Some(true)).collect();
            assert_eq!(kept(expr), expected);
        }
    }

    #[test]
    fn keeps_rows_in_a_list() {
        let env = Env::current();
        // `a IN (3, 5)`, `b IN (2)`, and either.
        let a_in = Leaf::In { column: 0, start: 0, len: 2 };
        let b_in = Leaf::In { column: 1, start: 2, len: 1 };
        let kept = |nodes: &[Node]| {
            // Then 100 to 119, for a list long enough for a set.
            let values: StdVec<i64> = [3, 5, 2, 7, 9, 11].into_iter().chain(100..120).collect();
            let nodes = SmallVec::fixed_from(env, nodes.iter().copied()).unwrap();
            kept_by(&Predicate::with_values(env, nodes, &values).unwrap())
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
        let env = Env::current();
        let bytes: StdVec<u8> = values.iter().flatten().flat_map(|v| v.bytes()).collect();
        let mut views = Buffer::allocate(env, values.len() * 8).unwrap();
        let mut validity = Buffer::allocate(env, 1).unwrap();
        let mut start = 0;
        for (i, value) in values.iter().enumerate() {
            let len = u32::try_from(value.map_or(0, str::len)).unwrap();
            views.as_mut_slice::<u32>()[2 * i..2 * i + 2].copy_from_slice(&[start, len]);
            validity.as_mut_slice::<u8>()[0] |= u8::from(value.is_some()) << i;
            start += len;
        }
        let mut stored = Buffer::allocate(env, bytes.len().max(1)).unwrap();
        stored.as_mut_slice::<u8>()[..bytes.len()].copy_from_slice(&bytes);
        ColumnView::strings(&mut Context::new(env, None), views, stored, Some(validity)).unwrap()
    }

    /// `values` with `indices` into them, as a dictionary column.
    fn indexed(values: &ColumnView, indices: &[u32]) -> ColumnView {
        let env = Env::current();
        let mut buffer = Buffer::allocate(env, indices.len() * 4).unwrap();
        buffer.as_mut_slice::<u32>().copy_from_slice(indices);
        ColumnView::dictionary(&mut Context::new(env, None), values, buffer).unwrap()
    }

    /// A statement's literals, holding `strings`.
    fn literals(strings: &[&[u8]]) -> StringPool {
        let env = Env::current();
        let mut literals = StringPool::new(env).unwrap();
        for &string in strings {
            literals.intern_string(env, Some(string)).unwrap();
        }
        literals
    }

    /// The id of `string`, one of `literals`.
    fn id(literals: &mut StringPool, string: &[u8]) -> Id {
        let env = Env::current();
        literals.intern_string(env, Some(string)).unwrap()
    }

    /// `x <comparison> 'b'`, or `NOT` that, over the first column, and its
    /// literals.
    fn compare_b(comparison: Comparison, not: bool) -> (Predicate, StringPool) {
        let env = Env::current();
        let mut literals = literals(&[]);
        let string = id(&mut literals, b"b");
        let leaf = Leaf::CompareString { column: 0, comparison, string };
        let nodes = [Node::Leaf(leaf), Node::Not(0)];
        let nodes = SmallVec::fixed_from(env, nodes[..=usize::from(not)].iter().copied());
        (Predicate::new(env, nodes.unwrap()).unwrap(), literals)
    }

    /// The rows `predicate`, with `literals`, keeps of a batch of `column`,
    /// its entries kept in `entries` between batches.
    fn kept_of(
        (predicate, literals): (&Predicate, &StringPool),
        entries: &mut [Entries],
        column: &ColumnView,
    ) -> StdVec<u16> {
        let env = Env::current();
        let mut scratch: StdVec<Selection> =
            (0..predicate.depth()).map(|_| Selection::all(0)).collect();
        let sets = predicate.new_sets(env, Some(literals)).unwrap();
        let mut batch = RowBatch::new();
        batch.reset(column.row_count());
        assert!(batch.push_column(column.clone()).is_ok());
        let literals = Some(literals);
        predicate.select(env, literals, &sets, &mut scratch, entries, &mut batch).unwrap();
        match batch.selection().kept() {
            Kept::Prefix => (0..u16::try_from(column.row_count()).unwrap()).collect(),
            Kept::None => StdVec::new(),
            Kept::Select(rows) => rows.to_vec(),
        }
    }

    #[test]
    fn filters_string_dictionaries_as_their_strings() {
        let env = Env::current();
        // Entries out of order, one null, used in any order.
        let entries = strings(&[Some("c"), None, Some("a"), Some("b"), Some("")]);
        // Three groups of eight rows, and three after them.
        let indices = [[3, 1, 0, 2, 4, 3, 0, 1].as_slice(); 3].concat();
        let indices = [indices.as_slice(), &[2, 3, 0]].concat();
        let dictionary = indexed(&entries, &indices);
        let mut flat = dictionary.clone();
        flat.make_in(&mut Context::new(env, None), Forms::FLAT).unwrap();
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
                let mut entries = predicate.new_entries(env).unwrap();
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
            let nodes = SmallVec::fixed_from(env, nodes[..=usize::from(not)].iter().copied());
            let predicate = Predicate::new(env, nodes.unwrap()).unwrap();
            let mut entries = predicate.new_entries(env).unwrap();
            assert_eq!(kept_of((&predicate, &literals), &mut entries, &flat), expected);
            assert_eq!(kept_of((&predicate, &literals), &mut entries, &dictionary), expected);
        }
        // `x IS NULL` keeps entry 1, and `x IS NOT NULL` the others.
        for (not, expected) in [(false, rows_of(&[1])), (true, rows_of(&[0, 2, 3, 4]))] {
            let nodes = [Node::Leaf(Leaf::IsNull { column: 0 }), Node::Not(0)];
            let nodes = SmallVec::fixed_from(env, nodes[..=usize::from(not)].iter().copied());
            let predicate = Predicate::new(env, nodes.unwrap()).unwrap();
            let (mut entries, literals) = (predicate.new_entries(env).unwrap(), literals(&[]));
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
            let nodes = SmallVec::fixed_from(env, nodes[..=usize::from(not)].iter().copied());
            let predicate = Predicate::with_values(env, nodes.unwrap(), &ids).unwrap();
            let mut entries = predicate.new_entries(env).unwrap();
            assert_eq!(kept_of((&predicate, &literals), &mut entries, &flat), expected);
            assert_eq!(kept_of((&predicate, &literals), &mut entries, &dictionary), expected);
        }
    }

    #[test]
    fn keeps_all_or_none_when_every_entry_or_none_matches() {
        let env = Env::current();
        let dictionary = indexed(&strings(&[Some("a"), Some("c")]), &[1, 0, 1]);
        let kept = |comparison| {
            let (predicate, literals) = compare_b(comparison, false);
            let mut entries = predicate.new_entries(env).unwrap();
            kept_of((&predicate, &literals), &mut entries, &dictionary)
        };
        assert_eq!(kept(Comparison::Equal), []);
        assert_eq!(kept(Comparison::NotEqual), [0, 1, 2]);
    }

    #[test]
    fn keeps_a_dictionarys_entries_while_batches_share_it() {
        let env = Env::current();
        let (predicate, literals) = compare_b(Comparison::Greater, false);
        let predicate = (&predicate, &literals);
        let mut entries = predicate.0.new_entries(env).unwrap();
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
        let env = Env::current();
        let step = |op, data_type, a, b| ExprStep {
            op,
            data_type,
            a,
            b,
            c: 0,
            int_value: 1,
            double_value: 0.0,
        };
        let (int, string) = (DataType::Int64, DataType::String);
        // `s = 'b' AND n = 1 AND (t = 'b' AND t IS NULL)`, of columns 0, 1 and 2.
        let nodes = [
            step(ExprOp::Column, int, 0, 0),
            step(ExprOp::Constant, string, 0, 0),
            step(ExprOp::Eq, int, 0, 1),
            step(ExprOp::Column, int, 1, 0),
            step(ExprOp::Constant, int, 0, 0),
            step(ExprOp::Eq, int, 3, 4),
            step(ExprOp::Column, int, 2, 0),
            step(ExprOp::Constant, string, 0, 0),
            step(ExprOp::Eq, int, 6, 7),
            step(ExprOp::Column, int, 2, 0),
            step(ExprOp::IsNull, int, 9, 0),
            step(ExprOp::And, int, 8, 10),
            step(ExprOp::And, int, 2, 5),
            step(ExprOp::And, int, 12, 11),
        ];
        let mut exprs = Exprs::new(env).unwrap();
        let expr = exprs.add(&nodes).unwrap();
        // A string column, an integer one, one also tested for nulls, which
        // doesn't count, and one it doesn't read.
        let forms: StdVec<_> = (0..4).map(|column| tested_forms(&exprs, &[expr], column)).collect();
        let (strings, ints) = (Forms::DICTIONARY | Forms::CONSTANT, Forms::SEQUENCE);
        assert!(forms.iter().map(|&forms| forms.contains(strings)).eq([true, false, true, false]));
        assert!(forms.iter().map(|&forms| forms.contains(ints)).eq([false, true, false, false]));
    }
}
