//! Bounds of a chunk's integers, from their type and the chunk's
//! statistics.

use pipit_kernel::column::{Bounds, ColumnView, DataType};

use crate::footer::{Chunk, Column, Logical, Physical};

/// `column`, with `bounds` if known.
pub(crate) fn bounded(column: ColumnView, bounds: Option<Bounds>) -> ColumnView {
    match bounds {
        Some(bounds) => column.with_bounds(bounds),
        None => column,
    }
}

/// Bounds of `column`'s values in `chunk`, if it's of integers: from their
/// type, for 32-bit ones, narrowed by the chunk's statistics.
pub(crate) fn of_chunk(column: Column, chunk: &Chunk) -> Option<Bounds> {
    if column.data_type() != Some(DataType::Int64) {
        return None;
    }
    let typed = match (column.physical, column.logical) {
        (Physical::Int32, Logical::Unsigned) => Some((0, i64::from(u32::MAX))),
        (Physical::Int32, _) => Some((i64::from(i32::MIN), i64::from(i32::MAX))),
        _ => None,
    };
    let (min, max) = match (typed, chunk.min.zip(chunk.max)) {
        (Some((lo, hi)), Some((min, max))) => (lo.max(min), hi.min(max)),
        (Some(bounds), None) | (None, Some(bounds)) => bounds,
        (None, None) => return None,
    };
    // Statistics at odds with the type are no use.
    (min <= max).then_some(Bounds { min, max })
}
