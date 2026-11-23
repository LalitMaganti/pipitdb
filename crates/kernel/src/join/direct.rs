//! Finding a join's build rows by their key's value, when it's one integer
//! whose values are close enough, as a perfect hash does.

use super::last_of;
use crate::allocator::AllocError;
use crate::column::{ColumnView, Form, Forms};
use crate::context::Context;
use crate::error::Error;
use crate::filter::{self, Range};
use crate::global::Global;
use crate::group_table::{GroupTable, NONE};
use crate::row_batch::RowBatch;
use crate::slow_vec::SlowVec;

/// The most a key can be above the smallest for `Direct` to find build
/// rows by their keys, as DuckDB's perfect hash join's: an array of 4 MB.
const DIRECT_SPAN_MAX: u64 = 1 << 20;

/// Finds the build row of a key by its value, as a perfect hash does: by
/// how far it is above the smallest key.
pub(super) struct Direct {
    /// The smallest key, and how far above it the largest is.
    pub(super) min: i64,
    pub(super) span: u64,
    /// The build row of each key, by how far it is above `min`, or `NONE`.
    /// Empty if each key's build row is that, as a table's ids' are.
    pub(super) rows: SlowVec<u32>,
}

impl Direct {
    /// For the build rows of `table`'s groups, of one integer key each,
    /// if their keys are close enough.
    pub(super) fn of(
        allocator: &Global,
        table: &GroupTable,
        count: u32,
    ) -> Result<Option<Direct>, AllocError> {
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let groups = table.len() as u32;
        let keys = (0..groups).map(|group| table.key(group, 0));
        let (Some(min), Some(max)) = (keys.clone().min(), keys.max()) else { return Ok(None) };
        let span = max.wrapping_sub(min).cast_unsigned();
        if max < min || span >= DIRECT_SPAN_MAX {
            return Ok(None);
        }
        let above = |group| table.key(group, 0).wrapping_sub(min).cast_unsigned();
        // Each build row its key's number above the first: none is a null
        // key's, which has no group.
        let each = (0..groups).all(|group| u64::from(last_of(table, group)) == above(group));
        if groups == count && each {
            return Ok(Some(Direct { min, span, rows: SlowVec::fixed(allocator, 0)? }));
        }
        #[expect(clippy::cast_possible_truncation, reason = "below `DIRECT_SPAN_MAX`")]
        let mut rows =
            SlowVec::fixed_from(allocator, core::iter::repeat_n(NONE, span as usize + 1))?;
        for group in 0..groups {
            #[expect(clippy::cast_possible_truncation, reason = "below `DIRECT_SPAN_MAX`")]
            let at = above(group) as usize;
            *at_mut!(rows, at) = last_of(table, group);
        }
        Ok(Some(Direct { min, span, rows }))
    }
}

/// Whether a join's build rows' keys have each been their number above the
/// first's.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Sequence {
    /// No build row yet.
    Empty,
    /// So far, from this first key.
    From(i64),
    /// Not, or not one integer key.
    Broken,
}

impl Direct {
    /// Narrows `pairs` to its rows whose key, at `key`, has a build row,
    /// writing each one's to `paired`, of the `count` build rows. Every row
    /// is gathered, so one without a build row takes the first's.
    pub(super) fn pair(
        &self,
        context: &mut Context,
        key: u32,
        count: u32,
        paired: &mut [u32],
        pairs: &mut RowBatch,
    ) -> Result<(), Error> {
        let mut key = pairs.column(key).clone();
        key.make_in(context, Forms::FLAT)?;
        let (min, span) = (self.min, self.span);
        filter::keep_within(&key, Range { lo: min, span }, pairs.selection_mut(), None);
        // A bound of 32 bits, so the loop works in 32 bits.
        #[expect(clippy::cast_possible_truncation, reason = "below `DIRECT_SPAN_MAX`")]
        let keys = span as u32 + 1;
        let row_of = |value: i64| {
            let above = value.wrapping_sub(min).cast_unsigned();
            #[expect(clippy::cast_possible_truncation, reason = "below `keys`")]
            let row = if above < u64::from(keys) { above as u32 } else { 0 };
            row
        };
        for (paired, &value) in paired.iter_mut().zip(key.int64s()) {
            *paired = row_of(value);
        }
        if self.rows.is_empty() {
            return Ok(());
        }
        let rows = &self.rows;
        for paired in paired.iter_mut() {
            *paired = *at!(rows, *paired as usize);
        }
        pairs.selection_mut().retain(|row| *at!(paired, usize::from(row)) != NONE);
        for paired in paired.iter_mut() {
            *paired = (*paired).min(count - 1);
        }
        Ok(())
    }
}

/// The first of `key`'s first `count` values if each is one above the one
/// before, as a table's ids are, flat or as a sequence, and `sequence` isn't
/// broken already.
pub(super) fn consecutive(key: &ColumnView, count: u32, sequence: Sequence) -> Option<i64> {
    match key.form() {
        _ if sequence == Sequence::Broken => None,
        Form::Sequence(start) => Some(start),
        Form::Flat if key.validity().is_none() => {
            let values = at!(key.int64s(), ..count as usize);
            let start = *values.first()?;
            let mut expected = start;
            let mut all = true;
            for &value in values {
                all &= value == expected;
                expected = expected.wrapping_add(1);
            }
            all.then_some(start)
        }
        _ => None,
    }
}
