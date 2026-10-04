//! Filters on one column: each narrows a `Selection` to the rows it keeps.
//! Shared by sources that filter as they read and by filters in a pipeline,
//! so both keep the same rows. A null row is never kept by a comparison, as
//! in SQL.

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

/// Keeps the rows where `column <comparison> value`.
pub fn compare(
    column: &ColumnView,
    comparison: Comparison,
    value: Value,
    selection: &mut Selection,
) {
    compare_into(column, comparison, value, selection, None);
}

/// As `compare`, and writes the rows not kept, null or not, to `rejected`, in
/// the same pass.
pub fn compare_split(
    column: &ColumnView,
    comparison: Comparison,
    value: Value,
    selection: &mut Selection,
    rejected: &mut Selection,
) {
    compare_into(column, comparison, value, selection, Some(rejected));
}

fn compare_into(
    column: &ColumnView,
    comparison: Comparison,
    value: Value,
    selection: &mut Selection,
    rejected: Option<&mut Selection>,
) {
    match value {
        Value::Int64(value) => {
            check!(column.data_type() == DataType::Int64);
            let cells = column.int64s();
            compare_values(column, cells, |cell| cell, comparison, value, selection, rejected);
        }
        Value::Float64(value) => {
            check!(column.data_type() == DataType::Float64);
            let cells = column.float64s();
            let value = float_key(value);
            compare_values(column, cells, float_key, comparison, value, selection, rejected);
        }
    }
}

/// Keeps the null rows if `nulls`, or the others.
pub fn is_null(column: &ColumnView, nulls: bool, selection: &mut Selection) {
    is_null_into(column, nulls, selection, None);
}

/// As `is_null`, and writes the rows not kept to `rejected`, in the same pass.
pub fn is_null_split(
    column: &ColumnView,
    nulls: bool,
    selection: &mut Selection,
    rejected: &mut Selection,
) {
    is_null_into(column, nulls, selection, Some(rejected));
}

fn is_null_into(
    column: &ColumnView,
    nulls: bool,
    selection: &mut Selection,
    rejected: Option<&mut Selection>,
) {
    let Some(validity) = covering(column, selection).validity() else {
        // No row is null.
        match rejected {
            None if nulls => selection.retain(|_| false),
            None => {}
            Some(rejected) => selection.partition(rejected, |_| !nulls),
        }
        return;
    };
    let test = move |row: u16| {
        // SAFETY: `covering` checked every kept row is a row of `column`.
        let valid = unsafe { validity.is_valid_unchecked(u32::from(row)) };
        valid != nulls
    };
    match rejected {
        None => selection.retain(test),
        Some(rejected) => selection.partition(rejected, test),
    }
}

/// One loop per comparison, so the loop has no branch on it.
fn compare_values<T: Copy, K: Copy + PartialOrd>(
    column: &ColumnView,
    cells: &[T],
    key: impl Fn(T) -> K + Copy,
    comparison: Comparison,
    value: K,
    selection: &mut Selection,
    rejected: Option<&mut Selection>,
) {
    covering(column, selection);
    // SAFETY: `covering` checked every kept row is a row of `column`, whose
    // values `cells` are.
    let at = move |row: u16| key(unsafe { cell(cells, row) });
    let (selection, rejected) = (selection, rejected);
    match comparison {
        Comparison::Equal => keep(column, selection, rejected, move |row| at(row) == value),
        Comparison::NotEqual => keep(column, selection, rejected, move |row| at(row) != value),
        Comparison::Less => keep(column, selection, rejected, move |row| at(row) < value),
        Comparison::LessEqual => keep(column, selection, rejected, move |row| at(row) <= value),
        Comparison::Greater => keep(column, selection, rejected, move |row| at(row) > value),
        Comparison::GreaterEqual => keep(column, selection, rejected, move |row| at(row) >= value),
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

/// Keeps the non-null rows `test` passes, writing the rest to `rejected` if
/// there is one, with a loop that skips the null check when `column` has no
/// nulls. `test` is still run for null rows, whose values are there to read,
/// so the loop has no branch on it.
fn keep(
    column: &ColumnView,
    selection: &mut Selection,
    rejected: Option<&mut Selection>,
    test: impl Fn(u16) -> bool + Copy,
) {
    match (column.validity(), rejected) {
        (None, None) => selection.retain(test),
        (None, Some(rejected)) => selection.partition(rejected, test),
        (Some(validity), rejected) => {
            let test = move |row: u16| {
                // SAFETY: callers check with `covering` first.
                let valid = unsafe { validity.is_valid_unchecked(u32::from(row)) };
                valid & test(row)
            };
            match rejected {
                None => selection.retain(test),
                Some(rejected) => selection.partition(rejected, test),
            }
        }
    }
}

/// Checks every row `selection` may keep is a row of `column`, so the rows
/// can be read without checking each, and returns `column`.
fn covering<'c>(column: &'c ColumnView, selection: &Selection) -> &'c ColumnView {
    check!(selection.rows() <= column.row_count());
    column
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
        let rows = |comparison| kept(|s| compare(&column, comparison, Value::Int64(6), s));
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
            compare(&column, comparison, Value::Float64(value), &mut selection);
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
            kept(|s| compare(&column, Comparison::NotEqual, Value::Int64(5), s)),
            [0, 1, 3, 4, 6, 8, 9]
        );
        assert_eq!(kept(|s| is_null(&column, true, s)), [2, 7]);
        assert_eq!(kept(|s| is_null(&column, false, s)).len(), 8);
        assert_eq!(kept(|s| is_null(&int64s(&[]), true, s)), []);
    }

    #[test]
    fn narrows_what_is_already_kept() {
        let column = int64s(&[]);
        let rows = kept(|s| {
            compare(&column, Comparison::Greater, Value::Int64(2), s);
            compare(&column, Comparison::NotEqual, Value::Int64(5), s);
        });
        assert_eq!(rows, [3, 4, 6, 7, 8, 9]);
    }
}
