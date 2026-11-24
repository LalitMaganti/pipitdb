//! `KeySet`: a set of integer keys, held as precisely as is cheap. It's
//! always the range they span. When that's narrow enough to hold a bit for
//! each key within it, it also says which of those keys are there. A
//! join's build side finds its rows by one, and its probe side's rows are
//! tested against one.

use crate::allocator::AllocError;
use crate::containers::RankBitmap;
use crate::env::Env;
use crate::filter::Range;

pub(crate) struct KeySet {
    /// The keys it may have.
    range: Range,
    /// Which keys within `range` it has, by their number above its start,
    /// unless it has all of them.
    present: Option<RankBitmap>,
}

impl KeySet {
    /// Every key in `range`.
    pub(crate) fn all(range: Range) -> KeySet {
        KeySet { range, present: None }
    }

    /// The set of `keys`, if their span is at most `span_max`, so a bit can
    /// be held for each key within it. If `distinct`, no key repeats, so
    /// when there are as many keys as the span holds, none is missing.
    pub(crate) fn of(
        env: &Env,
        keys: &[i64],
        span_max: u64,
        distinct: bool,
    ) -> Result<Option<KeySet>, AllocError> {
        let bounds = |(min, max): (i64, i64), &key: &i64| (key.min(min), key.max(max));
        let (min, max) = keys.iter().fold((i64::MAX, i64::MIN), bounds);
        let span = max.wrapping_sub(min).cast_unsigned();
        if keys.is_empty() || span > span_max.min(u32::MAX.into()) {
            return Ok(None);
        }
        let range = Range { lo: min, span };
        if distinct && span < keys.len() as u64 {
            return Ok(Some(KeySet::all(range)));
        }
        #[expect(clippy::cast_possible_truncation, reason = "a span in 32 bits")]
        let present = RankBitmap::new(
            env,
            span as usize + 1,
            keys.iter().map(|&key| key.wrapping_sub(min).cast_unsigned() as usize),
        )?;
        // No key within the range is missing: the range says it all.
        let all = u64::from(present.count()) == span + 1;
        Ok(Some(KeySet { range, present: (!all).then_some(present) }))
    }

    /// The range of keys it may have.
    pub(crate) fn range(&self) -> Range {
        self.range
    }

    /// Which keys within its range it has, by their number above its start,
    /// unless it has all of them.
    pub(crate) fn present(&self) -> Option<&RankBitmap> {
        self.present.as_ref()
    }

    /// How far above the range's start `key`, one of its keys, is.
    pub(crate) fn above(&self, key: i64) -> u32 {
        #[expect(clippy::cast_possible_truncation, reason = "within a span of 32 bits")]
        let above = key.wrapping_sub(self.range.lo).cast_unsigned() as u32;
        above
    }

    /// How many of its keys come before the one `above` the range's start,
    /// which it has: the key's place among them.
    pub(crate) fn rank(&self, above: u32) -> u32 {
        self.present.as_ref().map_or(above, |present| present.rank(above as usize))
    }
}
