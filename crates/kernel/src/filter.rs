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
    match value {
        Value::Int64(value) => {
            check!(column.data_type() == DataType::Int64);
            compare_values(column, column.int64s(), comparison, value, selection);
        }
        Value::Float64(value) => {
            check!(column.data_type() == DataType::Float64);
            compare_values(column, column.float64s(), comparison, value, selection);
        }
    }
}

/// Keeps the null rows if `nulls`, or the others.
pub fn is_null(column: &ColumnView, nulls: bool, selection: &mut Selection) {
    let Some(validity) = covering(column, selection).validity() else {
        if nulls {
            selection.retain(|_| false);
        }
        return;
    };
    // SAFETY: `covering` checked every kept row is a row of `column`.
    selection.retain(|row| unsafe { validity.is_valid_unchecked(u32::from(row)) } != nulls);
}

/// One loop per comparison, so the loop has no branch on it.
fn compare_values<T: Copy + PartialOrd>(
    column: &ColumnView,
    cells: &[T],
    comparison: Comparison,
    value: T,
    selection: &mut Selection,
) {
    covering(column, selection);
    // SAFETY: `covering` checked every kept row is a row of `column`, whose
    // values `cells` are.
    let at = move |row: u16| unsafe { cell(cells, row) };
    match comparison {
        Comparison::Equal => keep(column, selection, move |row| at(row) == value),
        Comparison::NotEqual => keep(column, selection, move |row| at(row) != value),
        Comparison::Less => keep(column, selection, move |row| at(row) < value),
        Comparison::LessEqual => keep(column, selection, move |row| at(row) <= value),
        Comparison::Greater => keep(column, selection, move |row| at(row) > value),
        Comparison::GreaterEqual => keep(column, selection, move |row| at(row) >= value),
    }
}

/// Keeps the non-null rows `test` passes, with a loop that skips the null
/// check when `column` has no nulls. `test` is still run for null rows, whose
/// values are there to read, so the loop has no branch on it.
fn keep(column: &ColumnView, selection: &mut Selection, test: impl Fn(u16) -> bool) {
    match column.validity() {
        None => selection.retain(test),
        Some(validity) => selection.retain(move |row| {
            // SAFETY: callers check with `covering` first.
            let valid = unsafe { validity.is_valid_unchecked(u32::from(row)) };
            valid & test(row)
        }),
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
    fn compares_floats() {
        let mut values = Buffer::allocate(Heap, 24).unwrap();
        values.as_mut_slice::<f64>().copy_from_slice(&[0.5, f64::NAN, 2.5]);
        let column = ColumnView::new(DataType::Float64, values, None);
        let mut selection = Selection::all(3);
        compare(&column, Comparison::Greater, Value::Float64(0.0), &mut selection);
        assert_eq!(selection.kept(), Kept::Select(&[0, 2]));
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
