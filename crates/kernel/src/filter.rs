//! Filters on one column, such as `x > 5` or `x IS NULL`.
//!
//! Each narrows a `Selection` to the rows its condition is true for. It
//! drops the rest: those it's false for, and for a comparison the null rows,
//! for which SQL's condition is neither true nor false. Given a second
//! selection, `dropped`, a filter also writes the rows it drops there, in the
//! same pass, so an `OR` can test its next condition on only those.

use crate::column::{ColumnView, DataType};
use crate::selection::Selection;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Comparison {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

/// A value to compare a column with, of the column's type.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Value {
    Int64(i64),
    Float64(f64),
}

/// Narrows `selection` to the rows where `column <comparison> value`, writing
/// the rows it drops to `dropped`, if given.
pub fn compare(
    column: &ColumnView,
    comparison: Comparison,
    value: Value,
    selection: &mut Selection,
    dropped: Option<&mut Selection>,
) {
    check_covers(column, selection);
    match value {
        Value::Int64(value) => {
            check!(column.data_type() == DataType::Int64);
            let cells = column.int64s();
            compare_cells(column, cells, |cell| cell, comparison, value, selection, dropped);
        }
        Value::Float64(value) => {
            check!(column.data_type() == DataType::Float64);
            let cells = column.float64s();
            let value = float_key(value);
            compare_cells(column, cells, float_key, comparison, value, selection, dropped);
        }
    }
}

/// Narrows `selection` to the null rows if `nulls`, or else to the others,
/// writing the rows it drops to `dropped`, if given.
pub fn is_null(
    column: &ColumnView,
    nulls: bool,
    selection: &mut Selection,
    dropped: Option<&mut Selection>,
) {
    check_covers(column, selection);
    match column.validity() {
        // No row is null.
        None => retain(selection, dropped, |_| !nulls),
        Some(validity) => retain(selection, dropped, move |row| {
            // SAFETY: `check_covers` checked every row is a row of `column`.
            let valid = unsafe { validity.is_valid_unchecked(u32::from(row)) };
            valid != nulls
        }),
    }
}

/// `compare` for `cells`, the values of `column`, compared by `key`. One loop
/// per comparison, so no loop branches on it.
fn compare_cells<T: Copy, K: Copy + PartialOrd>(
    column: &ColumnView,
    cells: &[T],
    key: impl Fn(T) -> K + Copy,
    comparison: Comparison,
    value: K,
    selection: &mut Selection,
    dropped: Option<&mut Selection>,
) {
    // SAFETY: `compare` checked every row is a row of `column`, whose values
    // `cells` are.
    let at = move |row: u16| key(unsafe { cell(cells, row) });
    let (s, d) = (selection, dropped);
    match comparison {
        Comparison::Equal => retain_valid(column, s, d, move |row| at(row) == value),
        Comparison::NotEqual => retain_valid(column, s, d, move |row| at(row) != value),
        Comparison::Less => retain_valid(column, s, d, move |row| at(row) < value),
        Comparison::LessEqual => retain_valid(column, s, d, move |row| at(row) <= value),
        Comparison::Greater => retain_valid(column, s, d, move |row| at(row) > value),
        Comparison::GreaterEqual => retain_valid(column, s, d, move |row| at(row) >= value),
    }
}

/// A float's place in the order SQL engines like DuckDB compare floats in:
/// NaN equals NaN and is above every other number, and -0 equals 0. The
/// bits of a negative float are flipped so integers order as floats do.
fn float_key(value: f64) -> i64 {
    if value.is_nan() {
        return i64::MAX;
    }
    // Adding 0 turns -0 into 0.
    let bits = (value + 0.0).to_bits().cast_signed();
    bits ^ ((bits >> 63).cast_unsigned() >> 1).cast_signed()
}

/// As `retain`, but also drops the null rows of `column`, with a loop that
/// skips checking for nulls when `column` has none. `test` still runs on
/// null rows, whose values are there to read, so the loop doesn't branch.
fn retain_valid(
    column: &ColumnView,
    selection: &mut Selection,
    dropped: Option<&mut Selection>,
    test: impl Fn(u16) -> bool + Copy,
) {
    match column.validity() {
        None => retain(selection, dropped, test),
        Some(validity) => retain(selection, dropped, move |row| {
            // SAFETY: callers check with `check_covers` first.
            let valid = unsafe { validity.is_valid_unchecked(u32::from(row)) };
            valid & test(row)
        }),
    }
}

/// Narrows `selection` to the rows `test` is true for, writing the rows it
/// drops to `dropped`, if given.
#[inline]
fn retain(
    selection: &mut Selection,
    dropped: Option<&mut Selection>,
    test: impl FnMut(u16) -> bool,
) {
    match dropped {
        None => selection.retain(test),
        Some(dropped) => selection.partition(dropped, test),
    }
}

/// Checks every row `selection` may keep is a row of `column`, so the rows
/// can then be read without checking each.
fn check_covers(column: &ColumnView, selection: &Selection) {
    check!(selection.rows() <= column.row_count());
}

/// `cells[row]`, without checking it's there.
///
/// # Safety
///
/// `row` must be below `cells.len()`.
#[inline]
unsafe fn cell<T: Copy>(cells: &[T], row: u16) -> T {
    // SAFETY: upheld by the caller.
    unsafe { *cells.get_unchecked(usize::from(row)) }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::selection::Kept;

    /// `0, 1, ..., 9`, with rows in `nulls` null.
    fn int64s(nulls: &[u16]) -> ColumnView {
        let mut values = Buffer::allocate(Heap, 80).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let validity = (!nulls.is_empty()).then(|| {
            let mut validity = Buffer::allocate(Heap, 2).unwrap();
            for row in 0..10_u16 {
                if !nulls.contains(&row) {
                    validity.as_mut_slice::<u8>()[row as usize / 8] |= 1 << (row % 8);
                }
            }
            validity
        });
        ColumnView::new(DataType::Int64, values, validity)
    }

    /// The rows `filter` keeps of all ten.
    fn kept(filter: impl FnOnce(&mut Selection)) -> Vec<u16> {
        let mut selection = Selection::all(10);
        filter(&mut selection);
        match selection.kept() {
            Kept::All => (0..10).collect(),
            Kept::None => Vec::new(),
            Kept::Select(rows) => rows.to_vec(),
        }
    }

    #[test]
    fn compares() {
        let column = int64s(&[]);
        let rows = |comparison| kept(|s| compare(&column, comparison, Value::Int64(6), s, None));
        assert_eq!(rows(Comparison::Equal), [6]);
        assert_eq!(rows(Comparison::NotEqual), [0, 1, 2, 3, 4, 5, 7, 8, 9]);
        assert_eq!(rows(Comparison::Less), [0, 1, 2, 3, 4, 5]);
        assert_eq!(rows(Comparison::LessEqual), [0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(rows(Comparison::Greater), [7, 8, 9]);
        assert_eq!(rows(Comparison::GreaterEqual), [6, 7, 8, 9]);
    }

    #[test]
    fn compares_floats_as_duckdb_does() {
        let mut values = Buffer::allocate(Heap, 48).unwrap();
        let cells = [-1.5, -0.0, 0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        values.as_mut_slice::<f64>().copy_from_slice(&cells);
        let column = ColumnView::new(DataType::Float64, values, None);
        let rows = |comparison, value| {
            let mut selection = Selection::all(6);
            compare(&column, comparison, Value::Float64(value), &mut selection, None);
            match selection.kept() {
                Kept::All => (0..6).collect(),
                Kept::None => Vec::new(),
                Kept::Select(rows) => rows.to_vec(),
            }
        };
        // NaN is above every number, infinity too, and equals NaN.
        assert_eq!(rows(Comparison::Greater, 0.0), [2, 3, 4]);
        assert_eq!(rows(Comparison::Less, f64::NAN), [0, 1, 2, 4, 5]);
        assert_eq!(rows(Comparison::Equal, f64::NAN), [3]);
        // -0 equals 0.
        assert_eq!(rows(Comparison::Equal, 0.0), [1]);
        assert_eq!(rows(Comparison::LessEqual, -1.5), [0, 5]);
    }

    #[test]
    fn never_keeps_nulls() {
        let column = int64s(&[2, 7]);
        assert_eq!(
            kept(|s| compare(&column, Comparison::NotEqual, Value::Int64(5), s, None)),
            [0, 1, 3, 4, 6, 8, 9]
        );
        assert_eq!(kept(|s| is_null(&column, true, s, None)), [2, 7]);
        assert_eq!(kept(|s| is_null(&column, false, s, None)).len(), 8);
        assert_eq!(kept(|s| is_null(&int64s(&[]), true, s, None)), []);
    }

    #[test]
    fn narrows_what_is_already_kept() {
        let column = int64s(&[]);
        let rows = kept(|s| {
            compare(&column, Comparison::Greater, Value::Int64(2), s, None);
            compare(&column, Comparison::NotEqual, Value::Int64(5), s, None);
        });
        assert_eq!(rows, [3, 4, 6, 7, 8, 9]);
    }
}
