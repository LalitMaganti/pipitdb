//! Filters on one column, such as `x > 5` or `x IS NULL`.
//!
//! Each narrows a `Selection` to the rows its condition is true for. It
//! drops the rest: those it's false for, and for a comparison the null rows,
//! for which SQL's condition is neither true nor false. Given a second
//! selection, `dropped`, a filter also writes the rows it drops there, in the
//! same pass, so an `OR` can test its next condition on only those.

use core::cmp::Ordering;

use crate::column::{ColumnView, DataType, Form};
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

impl Comparison {
    /// Whether it holds for a cell ordered `order` against the value.
    fn holds(self, order: Ordering) -> bool {
        match self {
            Comparison::Equal => order.is_eq(),
            Comparison::NotEqual => order.is_ne(),
            Comparison::Less => order.is_lt(),
            Comparison::LessEqual => order.is_le(),
            Comparison::Greater => order.is_gt(),
            Comparison::GreaterEqual => order.is_ge(),
        }
    }
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
    let key = match value {
        Value::Int64(value) => value,
        Value::Float64(value) => float_key(value),
    };
    let Some(range) = Range::of(comparison, key) else {
        return drop_all(selection, dropped);
    };
    match value {
        Value::Int64(_) => {
            check!(column.data_type() == DataType::Int64);
            compare_form(column, column.values().int64s(), |cell| cell, range, selection, dropped);
        }
        Value::Float64(_) => {
            check!(column.data_type() == DataType::Float64);
            compare_form(column, column.values().float64s(), float_key, range, selection, dropped);
        }
    }
}

/// Narrows `selection` to the rows where `column <comparison> string`, a
/// string column compared byte by byte, writing the rows it drops to
/// `dropped`, if given.
pub fn compare_string(
    column: &ColumnView,
    comparison: Comparison,
    string: &[u8],
    selection: &mut Selection,
    dropped: Option<&mut Selection>,
) {
    check_covers(column, selection);
    check!(column.data_type() == DataType::String);
    let strings = column.values().strings();
    let holds = |cell: &[u8]| comparison.holds(cell.cmp(string));
    match column.form() {
        Form::Flat => {}
        // One value for every row: kept or dropped together.
        Form::Constant if !column.is_null(0) && holds(strings.get(0)) => {
            return keep_all(selection, dropped);
        }
        Form::Constant => return drop_all(selection, dropped),
        // `Filter` refuses dictionary columns.
        Form::Dictionary(_) => crate::check::check_failed(line!()),
    }
    // A null row's view isn't read: it may not point at bytes.
    let validity = column.validity();
    let valid = move |row: u16| {
        // SAFETY: `check_covers` checked every row is a row of `column`.
        validity.is_none_or(|validity| unsafe { validity.is_valid_unchecked(u32::from(row)) })
    };
    let cell = move |row: u16| strings.get(usize::from(row));
    match comparison {
        // Equality needn't order: most cells differ in length, so aren't
        // read. One test for both `=` and `<>`, so the loop isn't copied.
        Comparison::Equal | Comparison::NotEqual => {
            let equal = comparison == Comparison::Equal;
            retain(selection, dropped, |row| valid(row) && (cell(row) == string) == equal);
        }
        _ => retain(selection, dropped, |row| valid(row) && holds(cell(row))),
    }
}

/// `compare` for any form of `column`, whose values `cells` reads.
fn compare_form<T: Copy>(
    column: &ColumnView,
    cells: &[T],
    key: impl Fn(T) -> i64 + Copy,
    range: Range,
    selection: &mut Selection,
    dropped: Option<&mut Selection>,
) {
    let values = column.values();
    match column.form() {
        Form::Flat => compare_cells(column, cells, key, range, selection, dropped),
        // One value for every row: kept or dropped together.
        Form::Constant => {
            if !values.is_null(0) && range.contains(key(*at!(cells, 0))) {
                keep_all(selection, dropped);
            } else {
                drop_all(selection, dropped);
            }
        }
        // `Filter` refuses dictionary columns.
        Form::Dictionary(_) => crate::check::check_failed(line!()),
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
    match column.form() {
        Form::Flat => {}
        // One value for every row: kept or dropped together.
        Form::Constant if column.is_null(0) == nulls => return keep_all(selection, dropped),
        Form::Constant => return drop_all(selection, dropped),
        // `Filter` refuses dictionary columns.
        Form::Dictionary(_) => crate::check::check_failed(line!()),
    }
    match column.validity() {
        // No row is null.
        None if nulls => drop_all(selection, dropped),
        None => keep_all(selection, dropped),
        Some(validity) => retain(selection, dropped, move |row| {
            // SAFETY: `check_covers` checked every row is a row of `column`.
            let valid = unsafe { validity.is_valid_unchecked(u32::from(row)) };
            valid != nulls
        }),
    }
}

/// The keys `lo` to `lo + span`, wrapping from `i64::MAX` to `i64::MIN`, so
/// one test covers every comparison: `!= v` is `v + 1` round to `v - 1`.
#[derive(Clone, Copy)]
struct Range {
    lo: i64,
    span: u64,
}

impl Range {
    /// The keys `key <comparison> value` is true for, or `None` if none.
    fn of(comparison: Comparison, value: i64) -> Option<Range> {
        let (lo, hi) = match comparison {
            Comparison::Equal => (value, value),
            Comparison::NotEqual => (value.wrapping_add(1), value.wrapping_sub(1)),
            Comparison::Less => (i64::MIN, value.checked_sub(1)?),
            Comparison::LessEqual => (i64::MIN, value),
            Comparison::Greater => (value.checked_add(1)?, i64::MAX),
            Comparison::GreaterEqual => (value, i64::MAX),
        };
        Some(Range { lo, span: hi.wrapping_sub(lo).cast_unsigned() })
    }

    /// One subtraction and one comparison, with no branch.
    #[inline]
    fn contains(self, key: i64) -> bool {
        key.wrapping_sub(self.lo).cast_unsigned() <= self.span
    }
}

/// `compare` for `cells`, the values of `column`, whose keys are `key`.
fn compare_cells<T: Copy>(
    column: &ColumnView,
    cells: &[T],
    key: impl Fn(T) -> i64 + Copy,
    range: Range,
    selection: &mut Selection,
    dropped: Option<&mut Selection>,
) {
    // SAFETY: `compare` checked every row is a row of `column`, whose values
    // `cells` are.
    let at = move |row: u16| key(unsafe { cell(cells, row) });
    retain_valid(column, selection, dropped, move |row| range.contains(at(row)));
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

/// Drops every row, writing them to `dropped`, if given.
fn drop_all(selection: &mut Selection, dropped: Option<&mut Selection>) {
    if let Some(dropped) = dropped {
        dropped.clone_from(selection);
    }
    selection.retain(never);
}

/// Keeps every row, so `dropped`, if given, is left with none.
fn keep_all(selection: &Selection, dropped: Option<&mut Selection>) {
    if let Some(dropped) = dropped {
        dropped.clone_from(selection);
        dropped.retain(never);
    }
}

/// One function, so `drop_all` and `keep_all` share a loop.
fn never(_: u16) -> bool {
    false
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
    use crate::context::Context;
    use crate::selection::Kept;

    /// `0, 1, ..., 9`, with rows in `nulls` null.
    fn int64s(nulls: &[u16]) -> ColumnView {
        let mut values = Buffer::allocate(&Heap, 80).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let validity = (!nulls.is_empty()).then(|| {
            let mut validity = Buffer::allocate(&Heap, 2).unwrap();
            for row in 0..10_u16 {
                if !nulls.contains(&row) {
                    validity.as_mut_slice::<u8>()[row as usize / 8] |= 1 << (row % 8);
                }
            }
            validity
        });
        ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, validity).unwrap()
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

    /// Whether a comparison holds, as Rust's operators say.
    type Holds = fn(&i64, &i64) -> bool;

    #[test]
    fn compares_at_the_ends_of_the_range() {
        let cells = [i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX - 1, i64::MAX];
        let mut values = Buffer::allocate(&Heap, 56).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(&cells);
        let column =
            ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, None).unwrap();
        let comparisons: [(Comparison, Holds); 6] = [
            (Comparison::Equal, i64::eq),
            (Comparison::NotEqual, i64::ne),
            (Comparison::Less, i64::lt),
            (Comparison::LessEqual, i64::le),
            (Comparison::Greater, i64::gt),
            (Comparison::GreaterEqual, i64::ge),
        ];
        for (comparison, holds) in comparisons {
            for value in cells {
                let mut selection = Selection::all(7);
                let mut dropped = Selection::all(0);
                compare(
                    &column,
                    comparison,
                    Value::Int64(value),
                    &mut selection,
                    Some(&mut dropped),
                );
                let rows = |selection: &Selection| match selection.kept() {
                    Kept::All => (0..7).collect(),
                    Kept::None => Vec::new(),
                    Kept::Select(rows) => rows.to_vec(),
                };
                let expected: Vec<u16> =
                    (0..7).filter(|&row| holds(&cells[usize::from(row)], &value)).collect();
                let others: Vec<u16> = (0..7).filter(|row| !expected.contains(row)).collect();
                assert_eq!(
                    (rows(&selection), rows(&dropped)),
                    (expected, others),
                    "{comparison:?} {value}"
                );
            }
        }
    }

    #[test]
    fn compares_floats_as_duckdb_does() {
        let mut values = Buffer::allocate(&Heap, 48).unwrap();
        let cells = [-1.5, -0.0, 0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        values.as_mut_slice::<f64>().copy_from_slice(&cells);
        let column =
            ColumnView::new(&mut Context::new(&Heap), DataType::Float64, values, None).unwrap();
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

    #[test]
    fn filters_constants_once() {
        let mut values = Buffer::allocate(&Heap, 2 * 8).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(&[20, 0]);
        let mut bits = Buffer::allocate(&Heap, 1).unwrap();
        bits.as_mut_slice::<u8>()[0] = 0b01;
        let values =
            ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, Some(bits)).unwrap();
        let compare_on = |column: &ColumnView, value| {
            let mut selection = Selection::all(column.row_count());
            compare(column, Comparison::GreaterEqual, Value::Int64(value), &mut selection, None);
            selection.len()
        };
        let twenties =
            ColumnView::constant(&mut Context::new(&Heap), &values.slice(0, 1), 4).unwrap();
        assert_eq!((compare_on(&twenties, 15), compare_on(&twenties, 25)), (4, 0));
        let nulls = ColumnView::constant(&mut Context::new(&Heap), &values.slice(1, 1), 4).unwrap();
        assert_eq!(compare_on(&nulls, -100), 0);
        let mut selection = Selection::all(4);
        is_null(&nulls, true, &mut selection, None);
        assert_eq!(selection.len(), 4);
    }

    /// Ten strings, ordered as bytes, not as text: `B` is before `a`, and
    /// `é`, two bytes from 0xc3, after `zz`. Row 3 is null, and its view
    /// points past the bytes, so reading it would fail.
    const STRINGS: [&[u8]; 10] =
        [b"", b"a", b"ab", b"", b"B", "é".as_bytes(), b"ab", b"abc", b"\0", b"zz"];

    fn strings() -> ColumnView {
        let bytes: Vec<u8> = STRINGS.concat();
        let mut views = Buffer::allocate(&Heap, 10 * 8).unwrap();
        let mut start = 0;
        for (row, string) in STRINGS.iter().enumerate() {
            let len = u32::try_from(string.len()).unwrap();
            let view = if row == 3 { [1000, 4] } else { [start, len] };
            views.as_mut_slice::<u32>()[2 * row..2 * row + 2].copy_from_slice(&view);
            start += len;
        }
        let mut stored = Buffer::allocate(&Heap, bytes.len()).unwrap();
        stored.as_mut_slice::<u8>().copy_from_slice(&bytes);
        let mut validity = Buffer::allocate(&Heap, 2).unwrap();
        validity.as_mut_slice::<u8>().copy_from_slice(&[!(1 << 3), 0b11]);
        ColumnView::strings(&mut Context::new(&Heap), views, stored, Some(validity)).unwrap()
    }

    #[test]
    fn compares_strings_byte_by_byte() {
        let column = strings();
        for comparison in [
            Comparison::Equal,
            Comparison::NotEqual,
            Comparison::Less,
            Comparison::LessEqual,
            Comparison::Greater,
            Comparison::GreaterEqual,
        ] {
            let rows = kept(|s| compare_string(&column, comparison, b"ab", s, None));
            let expected: Vec<u16> = (0..10)
                .filter(|&row| row != 3)
                .filter(|&row| comparison.holds(STRINGS[row as usize].cmp(b"ab".as_slice())))
                .collect();
            assert_eq!(rows, expected, "{comparison:?}");
        }
        // Bytes, not text: `B` and the empty string are below `a`, `é` above `zz`.
        assert_eq!(kept(|s| compare_string(&column, Comparison::Less, b"a", s, None)), [0, 4, 8]);
        assert_eq!(kept(|s| compare_string(&column, Comparison::Greater, b"zz", s, None)), [5]);
    }

    #[test]
    fn filters_constant_strings_once() {
        let mut views = Buffer::allocate(&Heap, 8).unwrap();
        views.as_mut_slice::<u32>().copy_from_slice(&[0, 2]);
        let mut bytes = Buffer::allocate(&Heap, 2).unwrap();
        bytes.as_mut_slice::<u8>().copy_from_slice(b"ab");
        let mut context = Context::new(&Heap);
        let value = ColumnView::strings(&mut context, views, bytes, None).unwrap();
        let column = ColumnView::constant(&mut context, &value, 10).unwrap();
        let equal = kept(|s| compare_string(&column, Comparison::Equal, b"ab", s, None));
        assert_eq!(equal, (0..10).collect::<Vec<u16>>());
        assert!(kept(|s| compare_string(&column, Comparison::Less, b"ab", s, None)).is_empty());
    }
}
