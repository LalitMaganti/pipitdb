//! Finding a join's build rows by their key's value, when it's one integer
//! whose values are close enough, as a perfect hash does.

use super::last_of;
use crate::allocator::AllocError;
use crate::column::{ColumnView, Form, Forms};
use crate::containers::{Array, RankBitmap};
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;
use crate::filter::{self, Range};
use crate::group_table::GroupTable;
use crate::row_batch::RowBatch;

/// How far apart the keys can be for `Direct` to find build rows by them:
/// 256 keys a build row, so its bitmap takes at most 48 bytes a row.
const KEYS_PER_ROW: u64 = 256;

/// Finds the build row of a key by its value, as a perfect hash does: by
/// how far it is above the smallest key.
pub(super) struct Direct {
    /// The smallest key, and how far above it the largest is.
    min: i64,
    span: u64,
    /// Which keys have a build row, and which. None if each key's build row
    /// is its number above `min`, as a table's ids' are.
    keyed: Option<Keyed>,
}

/// Which keys have a build row, by their number above the smallest, and
/// their rows in order of key, so a key's row is at its rank, as in DuckDB's
/// perfect hash join.
struct Keyed {
    keys: RankBitmap,
    rows: Array<u32>,
}

impl Direct {
    /// For `count` build rows whose keys are each their number above the
    /// first, `min`, as a table's ids are.
    pub(super) fn sequence(min: i64, count: u32) -> Direct {
        Direct { min, span: u64::from(count - 1), keyed: None }
    }

    /// For the build rows of `table`'s groups, of one integer key each,
    /// if their keys are close enough.
    pub(super) fn of(
        env: &Env,
        table: &GroupTable,
        count: u32,
    ) -> Result<Option<Direct>, AllocError> {
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let groups = table.len() as u32;
        let keys = (0..groups).map(|group| table.key(group, 0));
        let (Some(min), Some(max)) = (keys.clone().min(), keys.max()) else { return Ok(None) };
        let span = max.wrapping_sub(min).cast_unsigned();
        if max < min || span > (u64::from(groups) * KEYS_PER_ROW).min(u32::MAX.into()) {
            return Ok(None);
        }
        #[expect(clippy::cast_possible_truncation, reason = "at most `span`, in 32 bits")]
        let above = |group| table.key(group, 0).wrapping_sub(min).cast_unsigned() as usize;
        // Each build row its key's number above the first: none is a null
        // key's, which has no group.
        let each = (0..groups).all(|group| last_of(table, group) as usize == above(group));
        if groups == count && each {
            return Ok(Some(Direct { min, span, keyed: None }));
        }
        #[expect(clippy::cast_possible_truncation, reason = "a span in 32 bits")]
        let keys = RankBitmap::new(env, span as usize + 1, (0..groups).map(above))?;
        let mut rows = Array::zeroed(env, groups as usize)?;
        for group in 0..groups {
            *at_mut!(rows, keys.rank(above(group)) as usize) = last_of(table, group);
        }
        Ok(Some(Direct { min, span, keyed: Some(Keyed { keys, rows }) }))
    }

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
        #[expect(clippy::cast_possible_truncation, reason = "a span in 32 bits")]
        let keys = span as u32 + 1;
        let above_of = |value: i64| {
            let above = value.wrapping_sub(min).cast_unsigned();
            #[expect(clippy::cast_possible_truncation, reason = "below `keys`")]
            let above = if above < u64::from(keys) { above as u32 } else { 0 };
            above
        };
        for (paired, &value) in paired.iter_mut().zip(key.int64s()) {
            *paired = above_of(value);
        }
        let Some(Keyed { keys, rows }) = &self.keyed else { return Ok(()) };
        // Most rows often have no build row, so those go before the rest's
        // rows are found.
        pairs.selection_mut().retain(|row| keys.get(*at!(paired, usize::from(row)) as usize));
        let selection = pairs.selection();
        for i in 0..selection.len() {
            let paired = at_mut!(paired, selection.kept().row(i) as usize);
            *paired = *at!(rows, keys.rank(*paired as usize) as usize);
        }
        for paired in paired.iter_mut() {
            *paired = (*paired).min(count - 1);
        }
        Ok(())
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
