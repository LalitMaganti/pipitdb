//! `Condition`: what a plan's filters keep of one column, pushed down to
//! the scan that makes it, which may apply it as it reads.

use crate::allocator::{AllocError, Allocator};
use crate::column::{Bounds, ColumnView, Forms};
use crate::context::Context;
use crate::filter::{Entries, Range, Value};
use crate::predicate::{Leaf, Node, Predicate};
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::selection::{Kept, Selection};
use crate::slow_vec::SlowVec;

/// The most conditions a scan has.
pub const CONDITIONS_MAX: usize = 64;

/// What the rows of one of a scannable's columns must meet: the filters'
/// conjuncts that read only it. Scannables use it to skip what can't meet
/// it, and apply it where they say they do, in place of the filters.
pub struct Condition {
    /// Which of the scannable's columns.
    column: u32,
    /// The conjuncts, reading the column as column 0.
    predicate: Predicate,
    /// The values it keeps that aren't null.
    values: Values,
    /// Whether it keeps nulls, as `IS NULL` does.
    nulls: bool,
    /// Whether the scannable applies it, so no filter does.
    applied: bool,
}

impl Condition {
    /// The condition `predicate`, which reads only column 0, sets on
    /// `column`.
    pub fn new(column: u32, predicate: Predicate) -> Condition {
        #[expect(clippy::cast_possible_truncation, reason = "a predicate's nodes fit `u32`")]
        let root = predicate.nodes().len() as u32 - 1;
        let (values, nulls) = kept(predicate.nodes(), root);
        Condition { column, predicate, values, nulls: nulls == Some(true), applied: false }
    }

    /// The same condition, in memory from `allocator`.
    pub fn copy(&self, allocator: &dyn Allocator) -> Result<Condition, AllocError> {
        let predicate = self.predicate.renumbered(allocator, |column| column)?;
        Ok(Condition { predicate, ..*self })
    }

    /// Whether the scannable applies it, as it said it would.
    pub fn is_applied(&self) -> bool {
        self.applied
    }

    /// Marks it applied by its scannable, as the plan does when the
    /// scannable says it applies it.
    pub fn set_applied(&mut self) {
        self.applied = true;
    }

    /// The forms it can test its column in.
    pub fn accepts(&self) -> Forms {
        self.predicate.accepts(self.predicate.columns().next().unwrap_or(0))
    }

    /// Where testing it keeps the dictionary entries it finds, between
    /// batches, as a filter does. Reserves the selections testing it works
    /// in from `context`.
    pub fn new_entries(&self, context: &mut Context) -> Result<SlowVec<Entries>, AllocError> {
        context.reserve_selections(self.predicate.depth() as usize)?;
        self.predicate.new_entries(context.allocator())
    }

    /// Narrows `selection`, of `column`'s rows, to those meeting it. Its
    /// predicate must read column 0.
    pub fn select_column(
        &self,
        context: &mut Context,
        entries: &mut [Entries],
        column: &ColumnView,
        selection: &mut Selection,
    ) -> Result<(), AllocError> {
        let mut batch = RowBatch::new();
        batch.reset(column.row_count());
        let Ok(()) = batch.push_column(column.clone()) else {
            crate::check::check_failed(line!());
        };
        batch.selection_mut().clone_from(selection);
        let allocator = context.allocator();
        self.predicate.select(allocator, context.selections(), entries, &mut batch)?;
        selection.clone_from(batch.selection());
        Ok(())
    }

    /// Sets the bits, in `kept`, of the dictionary entries in `values` that
    /// meet it, one an entry from the lowest bit of the first word: each
    /// tested once. Its predicate must read column 0.
    pub fn kept_entries(
        &self,
        context: &mut Context,
        entries: &mut [Entries],
        values: &ColumnView,
        kept: &mut [u64],
    ) -> Result<(), AllocError> {
        let count = values.row_count();
        for start in (0..count).step_by(BATCH_ROWS_MAX as usize) {
            let rows = (count - start).min(BATCH_ROWS_MAX);
            let mut selection = Selection::all(rows);
            self.select_column(context, entries, &values.slice(start, rows), &mut selection)?;
            let mut set = |row: u32| {
                let at = (start + row) as usize;
                *at_mut!(kept, at / 64) |= 1 << (at % 64);
            };
            match selection.kept() {
                Kept::All => (0..rows).for_each(&mut set),
                Kept::None => {}
                Kept::Select(rows) => rows.iter().for_each(|&row| set(u32::from(row))),
            }
        }
        Ok(())
    }

    pub fn column(&self) -> u32 {
        self.column
    }

    /// Whether a row whose value is within `bounds`, or is null, could be
    /// kept: false only if no value within them is.
    pub fn may_keep(&self, bounds: Bounds) -> bool {
        let within = Values::Range(Range::between(bounds.min, bounds.max));
        self.nulls || !matches!(self.values.and(within), Values::None)
    }
}

/// Values: no integers, one wrapping range of them, as filters test, or
/// what one range doesn't say, such as two, or strings.
#[derive(Clone, Copy)]
enum Values {
    None,
    Range(Range),
    Ranges,
}

impl Values {
    /// Those not in this.
    fn not(self) -> Values {
        match self {
            Values::None => Values::Range(Range::between(i64::MIN, i64::MAX)),
            Values::Range(range) if range.span == u64::MAX => Values::None,
            // From past its end round to before its start.
            Values::Range(Range { lo, span }) => Values::Range(Range {
                lo: lo.wrapping_add_unsigned(span).wrapping_add(1),
                span: u64::MAX - span - 1,
            }),
            Values::Ranges => Values::Ranges,
        }
    }

    /// Those in both.
    fn and(self, other: Values) -> Values {
        let (a, b) = match (self, other) {
            (Values::None, _) | (_, Values::None) => return Values::None,
            (Values::Ranges, _) | (_, Values::Ranges) => return Values::Ranges,
            (Values::Range(a), Values::Range(b)) => (a, b),
        };
        if b.span == u64::MAX {
            return Values::Range(a);
        }
        // Counted from `a`'s start, `a` is `0..=a.span`, and `b` one or two
        // pieces that don't wrap.
        let start = b.lo.wrapping_sub(a.lo).cast_unsigned();
        let pieces = match start.checked_add(b.span) {
            Some(end) => [(start, end), (1, 0)],
            None => [(start, u64::MAX), (0, start.wrapping_add(b.span))],
        };
        let mut out = Values::None;
        for (lo, hi) in pieces {
            let hi = hi.min(a.span);
            if lo > hi {
                continue;
            }
            if !matches!(out, Values::None) {
                return Values::Ranges;
            }
            out = Values::Range(Range { lo: a.lo.wrapping_add_unsigned(lo), span: hi - lo });
        }
        out
    }
}

/// What node `node` of `nodes` keeps: the values that aren't null it's true
/// for, and what it is for a null, in SQL's three-valued logic, `None` for
/// null.
fn kept(nodes: &[Node], node: u32) -> (Values, Option<bool>) {
    match *at!(nodes, node as usize) {
        Node::Leaf(Leaf::Compare { comparison, value: Value::Int64(value), .. }) => {
            (Range::of(comparison, value).map_or(Values::None, Values::Range), None)
        }
        // No value that isn't null is null.
        Node::Leaf(Leaf::IsNull { .. }) => (Values::None, Some(true)),
        Node::Leaf(_) => (Values::Ranges, None),
        Node::And(a, b) => and(kept(nodes, a), kept(nodes, b)),
        // Not both false.
        Node::Or(a, b) => not(and(not(kept(nodes, a)), not(kept(nodes, b)))),
        Node::Not(a) => not(kept(nodes, a)),
    }
}

/// What `a AND b` keeps, given what each does.
#[inline(never)]
fn and(
    (a, a_null): (Values, Option<bool>),
    (b, b_null): (Values, Option<bool>),
) -> (Values, Option<bool>) {
    let null = match (a_null, b_null) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    };
    (a.and(b), null)
}

/// What `NOT a` keeps, given what `a` does.
#[inline(never)]
fn not((values, null): (Values, Option<bool>)) -> (Values, Option<bool>) {
    (values.not(), null.map(|null| !null))
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec as StdVec;

    use super::*;
    use crate::allocator::Heap;
    use crate::filter::Comparison;
    use crate::slow_vec::SlowVec;

    /// A predicate on one column, to build and to evaluate by hand.
    #[derive(Clone)]
    enum Expr {
        Compare(Comparison, i64),
        IsNull,
        And(alloc::boxed::Box<Expr>, alloc::boxed::Box<Expr>),
        Or(alloc::boxed::Box<Expr>, alloc::boxed::Box<Expr>),
        Not(alloc::boxed::Box<Expr>),
    }

    /// SQL's three-valued logic, for a row holding `value`: `None` is null.
    fn reference(expr: &Expr, value: Option<i64>) -> Option<bool> {
        match expr {
            Expr::Compare(comparison, v) => value.map(|value| match comparison {
                Comparison::Equal => value == *v,
                Comparison::NotEqual => value != *v,
                Comparison::Less => value < *v,
                Comparison::LessEqual => value <= *v,
                Comparison::Greater => value > *v,
                Comparison::GreaterEqual => value >= *v,
            }),
            Expr::IsNull => Some(value.is_none()),
            Expr::And(a, b) => match (reference(a, value), reference(b, value)) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            },
            Expr::Or(a, b) => match (reference(a, value), reference(b, value)) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            },
            Expr::Not(a) => reference(a, value).map(|value| !value),
        }
    }

    fn build(expr: &Expr, nodes: &mut SlowVec<Node>) -> u32 {
        let node = match expr {
            &Expr::Compare(comparison, value) => {
                Node::Leaf(Leaf::Compare { column: 0, comparison, value: Value::Int64(value) })
            }
            Expr::IsNull => Node::Leaf(Leaf::IsNull { column: 0 }),
            Expr::And(a, b) => Node::And(build(a, nodes), build(b, nodes)),
            Expr::Or(a, b) => Node::Or(build(a, nodes), build(b, nodes)),
            Expr::Not(a) => Node::Not(build(a, nodes)),
        };
        assert!(nodes.push(node).is_ok());
        u32::try_from(nodes.len() - 1).unwrap()
    }

    #[test]
    fn keeps_the_values_and_nulls_sql_does() {
        use alloc::boxed::Box;
        let comparisons = [
            Comparison::Equal,
            Comparison::NotEqual,
            Comparison::Less,
            Comparison::LessEqual,
            Comparison::Greater,
            Comparison::GreaterEqual,
        ];
        let mut leaves: StdVec<Expr> =
            comparisons.iter().map(|&comparison| Expr::Compare(comparison, 3)).collect();
        leaves.extend([
            Expr::IsNull,
            Expr::Compare(Comparison::Less, i64::MIN),
            Expr::Compare(Comparison::Greater, i64::MAX),
            Expr::Compare(Comparison::NotEqual, i64::MIN),
        ]);
        let combined = |exprs: &[Expr]| {
            let mut out: StdVec<Expr> =
                exprs.iter().map(|e| Expr::Not(Box::new(e.clone()))).collect();
            for a in exprs {
                for b in &leaves {
                    out.push(Expr::And(Box::new(a.clone()), Box::new(b.clone())));
                    out.push(Expr::Or(Box::new(a.clone()), Box::new(b.clone())));
                }
            }
            out
        };
        let first = combined(&leaves);
        let second = combined(&first);
        let values = (-2..=8).chain([i64::MIN, i64::MAX]);
        let mut said = 0;
        for expr in leaves.iter().chain(&first).chain(&second) {
            let mut nodes = SlowVec::new(&Heap, 64).unwrap();
            build(expr, &mut nodes);
            let condition = Condition::new(0, Predicate::new(nodes));
            assert_eq!(condition.nulls, reference(expr, None) == Some(true));
            let within = |v: i64| match condition.values {
                Values::None => Some(false),
                Values::Range(range) => Some(range.contains(v)),
                Values::Ranges => None,
            };
            for v in values.clone() {
                if let Some(within) = within(v) {
                    assert_eq!(within, reference(expr, Some(v)) == Some(true), "value {v}");
                }
            }
            said += usize::from(!matches!(condition.values, Values::Ranges));
        }
        // Every comparison, and most of what's made of them, is one range.
        assert!(leaves.iter().all(|leaf| {
            let mut nodes = SlowVec::new(&Heap, 64).unwrap();
            build(leaf, &mut nodes);
            !matches!(Condition::new(0, Predicate::new(nodes)).values, Values::Ranges)
        }));
        assert!(said * 2 > first.len() + second.len());
    }

    #[test]
    fn may_keep_rows_within_bounds_only_if_a_value_there_is_kept() {
        let condition = |expr: &Expr| {
            let mut nodes = SlowVec::new(&Heap, 8).unwrap();
            build(expr, &mut nodes);
            Condition::new(0, Predicate::new(nodes))
        };
        let between = |min, max| Bounds { min, max };
        let above_3 = condition(&Expr::Compare(Comparison::Greater, 3));
        assert!(!above_3.may_keep(between(0, 3)));
        assert!(above_3.may_keep(between(0, 4)));
        let not_3 = condition(&Expr::Compare(Comparison::NotEqual, 3));
        assert!(!not_3.may_keep(between(3, 3)));
        assert!(not_3.may_keep(between(3, 4)));
        // Nulls aren't within bounds, which say nothing of them.
        let null_or_above_3 = condition(&Expr::Or(
            alloc::boxed::Box::new(Expr::IsNull),
            alloc::boxed::Box::new(Expr::Compare(Comparison::Greater, 3)),
        ));
        assert!(null_or_above_3.may_keep(between(0, 3)));
    }
}
