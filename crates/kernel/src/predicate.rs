//! `Predicate`: a condition on a batch's rows, such as `WHERE`'s, which
//! narrows its selection to the rows the condition is true for.
//!
//! Each node can narrow a selection to the rows it's true for, or to those
//! it's false for. A null row is neither, as in SQL, so `NOT` swaps the two
//! and every node keeps SQL's three-valued logic without building booleans.

use crate::allocator::{AllocError, Allocator};
use crate::column::{Bounds, ColumnView, Form};
use crate::context::Context;
use crate::error::Error;
use crate::filter::{self, Comparison, Value};
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
    IsNull {
        column: u32,
    },
}

/// Its root is its last node.
pub struct Predicate {
    nodes: SlowVec<Node>,
    /// The bytes of the strings its comparisons compare with, if any.
    strings: Option<SlowVec<u8>>,
    depth: u32,
}

impl Predicate {
    /// A predicate of `nodes`, the last of which is its root.
    pub fn new(nodes: SlowVec<Node>) -> Predicate {
        Predicate::with_strings(nodes, None)
    }

    /// A predicate of `nodes`, whose string comparisons compare with bytes
    /// of `strings`.
    pub fn with_strings(nodes: SlowVec<Node>, strings: Option<SlowVec<u8>>) -> Predicate {
        check!(!nodes.is_empty() && nodes.len() <= PREDICATE_NODES_MAX);
        let bytes = strings.as_ref().map_or(0, |strings| strings.len());
        check!(nodes.iter().all(|node| match *node {
            Node::Leaf(Leaf::CompareString { start, len, .. }) => {
                (start as usize).checked_add(len as usize).is_some_and(|end| end <= bytes)
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
        Predicate { nodes, strings, depth }
    }

    /// How many scratch selections `select` needs.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Values of `column` every row this is true for has, from comparisons
    /// of it with integers that must all hold, as in `a > 1 AND a < 5`: a
    /// row whose value is outside them isn't kept. Others keep all values.
    pub fn kept(&self, column: u32) -> Bounds {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX`")]
        self.kept_by(self.nodes.len() as u32 - 1, column)
    }

    fn kept_by(&self, node: u32, column: u32) -> Bounds {
        match *at!(self.nodes, node as usize) {
            Node::Leaf(Leaf::Compare { column: c, comparison, value: Value::Int64(value) })
                if c == column =>
            {
                // Saturating, `a < i64::MIN` gives `i64::MIN`: more than it
                // keeps, which is still true of every row kept.
                match comparison {
                    Comparison::Equal => Bounds { min: value, max: value },
                    Comparison::NotEqual => Bounds::ALL,
                    Comparison::Less => Bounds { max: value.saturating_sub(1), ..Bounds::ALL },
                    Comparison::LessEqual => Bounds { max: value, ..Bounds::ALL },
                    Comparison::Greater => Bounds { min: value.saturating_add(1), ..Bounds::ALL },
                    Comparison::GreaterEqual => Bounds { min: value, ..Bounds::ALL },
                }
            }
            Node::And(a, b) => self.kept_by(a, column).intersect(self.kept_by(b, column)),
            Node::Leaf(_) | Node::Or(..) | Node::Not(_) => Bounds::ALL,
        }
    }

    /// The columns its comparisons and `IS NULL`s read, maybe more than once.
    pub fn columns(&self) -> impl Iterator<Item = u32> + '_ {
        self.nodes.iter().filter_map(|node| match *node {
            Node::Leaf(
                Leaf::Compare { column, .. }
                | Leaf::CompareString { column, .. }
                | Leaf::IsNull { column },
            ) => Some(column),
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
            Node::Leaf(Leaf::IsNull { column }) => {
                Node::Leaf(Leaf::IsNull { column: renumber(column) })
            }
            node => node,
        });
        let nodes = SlowVec::fixed_from(allocator, nodes)?;
        let strings = match &self.strings {
            Some(strings) => Some(SlowVec::fixed_from(allocator, strings.iter().copied())?),
            None => None,
        };
        Ok(Predicate { nodes, strings, depth: self.depth })
    }

    /// Narrows `batch`'s selection to the rows this is true for, working in
    /// `scratch`, which holds at least `depth` selections.
    #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX` nodes")]
    pub fn select(&self, scratch: &mut [Selection], batch: &mut RowBatch) {
        check!(scratch.len() >= self.depth as usize);
        let (columns, selection) = batch.columns_and_selection();
        let root = self.nodes.len() as u32 - 1;
        self.narrow(scratch, root, true, columns, selection);
    }

    /// Narrows `selection` to the rows node `node` is `want` for. A node that
    /// needs a selection takes the first of `scratch`, and its children work
    /// in the rest.
    fn narrow(
        &self,
        scratch: &mut [Selection],
        node: u32,
        want: bool,
        columns: &[ColumnView],
        selection: &mut Selection,
    ) {
        match *at!(self.nodes, node as usize) {
            Node::Leaf(leaf) => leaf.narrow(self.strings(), None, want, columns, selection),
            Node::Not(child) => self.narrow(scratch, child, !want, columns, selection),
            // True for both, or false for both: each narrows what the other left.
            Node::And(a, b) if want => self.both(scratch, a, b, true, columns, selection),
            Node::Or(a, b) if !want => self.both(scratch, a, b, false, columns, selection),
            // False for either, or true for either: the rows `a` is `want` for,
            // and those `b` is among the rest.
            Node::And(a, b) | Node::Or(a, b) => {
                let Some((rest, scratch)) = scratch.split_first_mut() else {
                    crate::check::check_failed(line!());
                };
                if let Node::Leaf(leaf) = *at!(self.nodes, a as usize) {
                    // In one pass.
                    leaf.narrow(self.strings(), Some(rest), want, columns, selection);
                } else {
                    rest.clone_from(selection);
                    self.narrow(scratch, a, want, columns, selection);
                    rest.subtract(selection);
                }
                self.narrow(scratch, b, want, columns, rest);
                selection.union(rest);
            }
        }
    }

    /// The bytes its string comparisons compare with.
    fn strings(&self) -> &[u8] {
        self.strings.as_deref().unwrap_or(&[])
    }

    fn both(
        &self,
        scratch: &mut [Selection],
        a: u32,
        b: u32,
        want: bool,
        columns: &[ColumnView],
        selection: &mut Selection,
    ) {
        self.narrow(scratch, a, want, columns, selection);
        self.narrow(scratch, b, want, columns, selection);
    }
}

impl Leaf {
    /// Narrows `selection` to the rows this is `want` for, writing the rows it
    /// drops to `dropped`, if given.
    fn narrow(
        self,
        strings: &[u8],
        dropped: Option<&mut Selection>,
        want: bool,
        columns: &[ColumnView],
        selection: &mut Selection,
    ) {
        match self {
            Leaf::Compare { column, comparison, value } => {
                let comparison = if want { comparison } else { negated(comparison) };
                let column = at!(columns, column as usize);
                filter::compare(column, comparison, value, selection, dropped);
            }
            Leaf::CompareString { column, comparison, start, len } => {
                let comparison = if want { comparison } else { negated(comparison) };
                let string = at!(strings, start as usize..(start + len) as usize);
                filter::compare_string(
                    at!(columns, column as usize),
                    comparison,
                    string,
                    selection,
                    dropped,
                );
            }
            Leaf::IsNull { column } => {
                filter::is_null(at!(columns, column as usize), want, selection, dropped);
            }
        }
    }
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

/// Narrows each batch's selection to the rows `predicate` is true for. The
/// predicate reads columns by their position in batches.
pub struct Filter {
    pub predicate: Predicate,
}

impl Transform for Filter {
    type State = ();

    fn new_state(&self, context: &mut Context) -> Result<(), Error> {
        Ok(context.reserve_selections(self.predicate.depth() as usize)?)
    }

    fn process(
        &self,
        context: &mut Context,
        (): &mut (),
        batch: &mut RowBatch,
    ) -> Result<(), Error> {
        // Filters test flat and constant columns, the forms `FilterOp`
        // accepts, so a scan never gives them a dictionary.
        //
        // TODO: filter a dictionary column by testing each of its entries
        // once, with the flat column's test on its values, into a bit for
        // each, and then keeping the rows whose index's bit is set. The bits
        // are kept between batches, as a chunk's batches share its
        // dictionary, so a dictionary of a few strings over millions of rows
        // costs a few tests. This is what DuckDB does.
        let mut columns = self.predicate.columns();
        check!(
            !columns.any(|position| matches!(batch.column(position).form(), Form::Dictionary(_)))
        );
        self.predicate.select(context.selections(), batch);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec as StdVec;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::DataType;
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

    fn kept(expr: &Expr) -> StdVec<usize> {
        let mut nodes = SlowVec::new(&Heap, PREDICATE_NODES_MAX).unwrap();
        build(expr, &mut nodes);
        let predicate = Predicate::new(nodes);
        let mut scratch: StdVec<Selection> =
            (0..predicate.depth()).map(|_| Selection::all(0)).collect();
        let mut batch = RowBatch::new();
        batch.reset(6);
        assert!(batch.push_column(column(A)).is_ok() && batch.push_column(column(B)).is_ok());
        predicate.select(&mut scratch, &mut batch);
        match batch.selection().kept() {
            Kept::All => (0..6).collect(),
            Kept::None => StdVec::new(),
            Kept::Select(rows) => rows.iter().map(|&row| usize::from(row)).collect(),
        }
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
    fn keeps_the_values_comparisons_that_must_hold_allow() {
        const A2: Expr = Expr::Greater(0, 2);
        const B2: Expr = Expr::Greater(1, 2);
        const NARROWER: Expr = Expr::And(&A2, &Expr::Not(&Expr::IsNull(0)));
        let kept = |expr: &Expr, column| {
            let mut nodes = SlowVec::new(&Heap, PREDICATE_NODES_MAX).unwrap();
            build(expr, &mut nodes);
            Predicate::new(nodes).kept(column)
        };
        assert_eq!(kept(&A2, 0), Bounds { min: 3, ..Bounds::ALL });
        assert_eq!(kept(&A2, 1), Bounds::ALL);
        assert_eq!(kept(&Expr::And(&NARROWER, &Expr::Greater(0, 4)), 0).min, 5);
        assert_eq!(kept(&Expr::And(&A2, &B2), 1), Bounds { min: 3, ..Bounds::ALL });
        // Either side of an `OR` could hold, and a `NOT` keeps the others.
        assert_eq!(kept(&Expr::Or(&A2, &A2), 0), Bounds::ALL);
        assert_eq!(kept(&Expr::Not(&A2), 0), Bounds::ALL);

        let compared = |comparison, value| {
            let leaf = Leaf::Compare { column: 0, comparison, value: Value::Int64(value) };
            let nodes = SlowVec::fixed_from(&Heap, [Node::Leaf(leaf)].into_iter()).unwrap();
            Predicate::new(nodes).kept(0)
        };
        let (min, max) = (i64::MIN, i64::MAX);
        assert_eq!(compared(Comparison::Equal, 7), Bounds { min: 7, max: 7 });
        assert_eq!(compared(Comparison::NotEqual, 7), Bounds::ALL);
        assert_eq!(compared(Comparison::Less, 7), Bounds { min, max: 6 });
        assert_eq!(compared(Comparison::LessEqual, 7), Bounds { min, max: 7 });
        assert_eq!(compared(Comparison::GreaterEqual, 7), Bounds { min: 7, max });
        // Keeping more than `a < i64::MIN` does is still right.
        assert_eq!(compared(Comparison::Less, min), Bounds { min, max: min });
        assert_eq!(compared(Comparison::Greater, max), Bounds { min: max, max });
    }

    #[test]
    #[should_panic(expected = "Dictionary")]
    fn filters_check_they_get_no_dictionary() {
        let mut nodes = SlowVec::new(&Heap, PREDICATE_NODES_MAX).unwrap();
        build(&Expr::Greater(0, 2), &mut nodes);
        let filter = Filter { predicate: Predicate::new(nodes) };
        let mut context = Context::new(&Heap);
        filter.new_state(&mut context).unwrap();
        let mut indices = crate::buffer::Buffer::allocate(&Heap, 6 * 4).unwrap();
        indices.as_mut_slice::<u32>().copy_from_slice(&[0, 1, 2, 3, 4, 5]);
        let mut batch = RowBatch::new();
        batch.reset(6);
        let dictionary =
            ColumnView::dictionary(&mut Context::new(&Heap), &column(A), indices).unwrap();
        assert!(batch.push_column(dictionary).is_ok());
        let _ = filter.process(&mut context, &mut (), &mut batch);
    }
}
