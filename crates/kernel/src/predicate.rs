//! `Predicate`: a condition on a batch's rows, such as `WHERE`'s, which
//! narrows its selection to the rows the condition is true for.
//!
//! Each node can narrow a selection to the rows it's true for, or to those
//! it's false for. A null row is neither, as in SQL, so `NOT` swaps the two
//! and every node keeps SQL's three-valued logic without building booleans.

use crate::allocator::{AllocError, Allocator};
use crate::column::ColumnView;
use crate::filter::{self, Comparison, Value};
use crate::row_batch::RowBatch;
use crate::selection::Selection;
use crate::vec::Vec;

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
    IsNull {
        column: u32,
    },
}

/// Its root is its last node.
pub struct Predicate {
    nodes: Vec<Node>,
}

/// Selections a predicate's `AND`s and `OR`s work in, one for each node, made
/// once for a run so no batch allocates.
pub struct Scratch {
    selections: Vec<Selection>,
}

impl Predicate {
    /// A predicate of `nodes`, the last of which is its root.
    pub fn new(nodes: Vec<Node>) -> Predicate {
        check!(!nodes.is_empty());
        for (i, node) in (0..).zip(nodes.iter()) {
            match *node {
                Node::Leaf(_) => {}
                Node::And(a, b) | Node::Or(a, b) => check!(a < i && b < i),
                Node::Not(a) => check!(a < i),
            }
        }
        Predicate { nodes }
    }

    pub fn scratch<A: Allocator + Clone + 'static>(
        &self,
        allocator: A,
    ) -> Result<Scratch, AllocError> {
        let selections = (0..self.nodes.len()).map(|_| Selection::all(0));
        Ok(Scratch { selections: Vec::fixed_from(allocator, selections)? })
    }

    /// Narrows `batch`'s selection to the rows this is true for.
    #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX` nodes")]
    pub fn select(&self, batch: &mut RowBatch, scratch: &mut Scratch) {
        check!(scratch.selections.len() == self.nodes.len());
        let (columns, selection) = batch.columns_and_selection();
        let root = self.nodes.len() as u32 - 1;
        self.narrow(root, true, columns, selection, &mut scratch.selections);
    }

    /// Narrows `selection` to the rows node `node` is `want` for. Children come
    /// before parents, so a node's scratch is past all its children's.
    fn narrow(
        &self,
        node: u32,
        want: bool,
        columns: &[ColumnView],
        selection: &mut Selection,
        scratch: &mut [Selection],
    ) {
        let (below, mine) = scratch.split_at_mut(node as usize);
        let Some(mine) = mine.first_mut() else { crate::check::check_failed(line!()) };
        match *at!(self.nodes, node as usize) {
            Node::Leaf(leaf) => leaf.narrow(want, columns, selection, None),
            Node::Not(child) => self.narrow(child, !want, columns, selection, below),
            // True for both, or false for both: each narrows what the other left.
            Node::And(a, b) if want => self.both(a, b, true, columns, selection, below),
            Node::Or(a, b) if !want => self.both(a, b, false, columns, selection, below),
            // False for either, or true for either: the rows `a` is `want` for,
            // and those `b` is among the rest.
            Node::And(a, b) | Node::Or(a, b) => {
                let rest = mine;
                if let Node::Leaf(leaf) = *at!(self.nodes, a as usize) {
                    // In one pass.
                    leaf.narrow(want, columns, selection, Some(rest));
                } else {
                    rest.clone_from(selection);
                    self.narrow(a, want, columns, selection, below);
                    rest.subtract(selection);
                }
                self.narrow(b, want, columns, rest, below);
                selection.union(rest);
            }
        }
    }

    fn both(
        &self,
        a: u32,
        b: u32,
        want: bool,
        columns: &[ColumnView],
        selection: &mut Selection,
        below: &mut [Selection],
    ) {
        self.narrow(a, want, columns, selection, below);
        self.narrow(b, want, columns, selection, below);
    }
}

impl Leaf {
    /// Narrows `selection` to the rows this is `want` for, writing the rows it
    /// drops to `dropped`, if given.
    fn narrow(
        self,
        want: bool,
        columns: &[ColumnView],
        selection: &mut Selection,
        dropped: Option<&mut Selection>,
    ) {
        match self {
            Leaf::Compare { column, comparison, value } => {
                let comparison = if want { comparison } else { negated(comparison) };
                let column = at!(columns, column as usize);
                filter::compare(column, comparison, value, selection, dropped);
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

#[cfg(test)]
mod tests {
    use alloc::vec::Vec as StdVec;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::DataType;
    use crate::selection::Kept;

    /// `a` and `b`, with `None` for null.
    const A: [Option<i64>; 6] = [Some(1), None, Some(3), None, Some(5), Some(3)];
    const B: [Option<i64>; 6] = [None, Some(2), None, Some(4), Some(5), Some(1)];

    fn column(cells: [Option<i64>; 6]) -> ColumnView {
        let mut values = Buffer::allocate(Heap, 48).unwrap();
        let mut validity = Buffer::allocate(Heap, 1).unwrap();
        for (row, cell) in cells.iter().enumerate() {
            values.as_mut_slice::<i64>()[row] = cell.unwrap_or(0);
            validity.as_mut_slice::<u8>()[0] |= u8::from(cell.is_some()) << row;
        }
        ColumnView::new(DataType::Int64, values, Some(validity))
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
    fn build(expr: &Expr, nodes: &mut Vec<Node>) -> u32 {
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
        let mut nodes = Vec::new(Heap, PREDICATE_NODES_MAX).unwrap();
        build(expr, &mut nodes);
        let predicate = Predicate::new(nodes);
        let mut scratch = predicate.scratch(Heap).unwrap();
        let mut batch = RowBatch::new();
        batch.reset(6);
        assert!(batch.push_column(column(A)).is_ok() && batch.push_column(column(B)).is_ok());
        predicate.select(&mut batch, &mut scratch);
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
        for expr in exprs {
            let expected: StdVec<usize> =
                (0..6).filter(|&row| reference(expr, row) == Some(true)).collect();
            assert_eq!(kept(expr), expected);
        }
    }
}
