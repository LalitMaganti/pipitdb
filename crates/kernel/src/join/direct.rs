//! Finding a join's build rows by their key's value, when it's one integer
//! whose values are close enough, as a perfect hash does.

use crate::allocator::AllocError;
use crate::column::{ColumnView, Form, Forms};
use crate::containers::Array;
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;
use crate::filter::{self, Range};
use crate::group_table::NONE;
use crate::key_set::KeySet;
use crate::row_batch::RowBatch;

/// How far apart the keys can be for `Direct` to find build rows by them:
/// 256 keys a build row, so its bitmap takes at most 48 bytes a row.
const KEYS_PER_ROW: u64 = 256;

/// Finds the build row of a key by its value, as a perfect hash does: by
/// how far it is above the smallest key.
pub(super) struct Direct {
    /// The keys that have a build row.
    keys: KeySet,
    /// The build rows in order of key, each at its key's place among them,
    /// as in DuckDB's perfect hash join. None if each key's build row is its
    /// number above the smallest, as a table's ids' are.
    rows: Option<Array<u32>>,
}

impl Direct {
    /// For `count` build rows whose keys are each their number above the
    /// first, `min`, as a table's ids are.
    pub(super) fn sequence(min: i64, count: u32) -> Direct {
        let keys = KeySet::all(Range { lo: min, span: u64::from(count - 1) });
        Direct { keys, rows: None }
    }

    /// For build rows of keys `keys`, one integer each, if they're close
    /// enough and none repeats.
    pub(super) fn of(env: &Env, keys: &[i64]) -> Result<Option<Direct>, AllocError> {
        let len = keys.len() as u64;
        // Keys that repeat are found below, so taken not to.
        let Some(set) = KeySet::of(env, keys, len * KEYS_PER_ROW, true)? else { return Ok(None) };
        // Each build row its key's number above the first.
        if keys.iter().enumerate().all(|(row, &key)| row == set.above(key) as usize) {
            return Ok(Some(Direct { keys: set, rows: None }));
        }
        let mut rows = Array::zeroed(env, keys.len())?;
        rows.fill(NONE);
        for (row, &key) in (0..).zip(keys) {
            let found = at_mut!(rows, set.rank(set.above(key)) as usize);
            // A key that repeats.
            if *found != NONE {
                return Ok(None);
            }
            *found = row;
        }
        Ok(Some(Direct { keys: set, rows: Some(rows) }))
    }

    /// Narrows `pairs` to its rows whose key, at `key`, has a build row,
    /// writing each one's to `paired`.
    pub(super) fn pair(
        &self,
        context: &mut Context,
        key: u32,
        paired: &mut [u32],
        pairs: &mut RowBatch,
    ) -> Result<(), Error> {
        let mut key = pairs.column(key).clone();
        key.make_in(context, Forms::FLAT)?;
        let Range { lo, span } = self.keys.range();
        filter::keep_within(&key, self.keys.range(), pairs.selection_mut(), None);
        // Each row's place above the smallest key, in one loop that
        // vectorizes: 0 for one outside the range, which isn't kept. A bound
        // of 32 bits, so the loop works in 32 bits.
        #[expect(clippy::cast_possible_truncation, reason = "a span in 32 bits")]
        let keys = span as u32 + 1;
        let above_of = |value: i64| {
            let above = value.wrapping_sub(lo).cast_unsigned();
            #[expect(clippy::cast_possible_truncation, reason = "below `keys`")]
            let above = if above < u64::from(keys) { above as u32 } else { 0 };
            above
        };
        for (paired, &value) in paired.iter_mut().zip(key.int64s()) {
            *paired = above_of(value);
        }
        let Some(rows) = self.rows.as_deref() else { return Ok(()) };
        let Some(present) = self.keys.present() else {
            // Every key within the range has a build row, at its place.
            for paired in paired.iter_mut() {
                *paired = *at!(rows, *paired as usize);
            }
            return Ok(());
        };
        // Most rows often have no build row, so those go before the rest's
        // rows are found.
        pairs.selection_mut().retain(|row| present.get(*at!(paired, usize::from(row)) as usize));
        let selection = pairs.selection();
        for i in 0..selection.len() {
            let paired = at_mut!(paired, selection.kept().row(i) as usize);
            *paired = *at!(rows, present.rank(*paired as usize) as usize);
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
    /// Not, but each build row's key is held, in `Built::keys`.
    Held,
    /// Not, or not one integer key: each build row is hashed.
    Broken,
}

/// The first of `key`'s first `count` values if each is one above the one
/// before, as a table's ids are, flat or as a sequence, and `sequence` isn't
/// broken already.
pub(super) fn consecutive(key: &ColumnView, count: u32, sequence: Sequence) -> Option<i64> {
    match key.form() {
        _ if matches!(sequence, Sequence::Held | Sequence::Broken) => None,
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
