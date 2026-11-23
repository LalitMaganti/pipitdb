//! `Estimate`: what planning knows about the rows a plan node will make,
//! before the query runs, so it can choose between plans. It follows the
//! rules Perfetto's dataframe planner uses to estimate the rows a query
//! leaves. Each op works out its estimate from its inputs' estimates.

use crate::plan::ColumnId;
use crate::scannable::Key;

/// The most rows there can be, and how many planning expects.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RowEstimate {
    pub max: u64,
    pub estimated: u64,
}

impl RowEstimate {
    /// Nothing known, so as many rows as there could possibly be.
    pub const UNKNOWN: RowEstimate = RowEstimate { max: u64::MAX, estimated: u64::MAX };

    /// Exactly `rows`.
    pub fn exactly(rows: u64) -> RowEstimate {
        RowEstimate { max: rows, estimated: rows }
    }
}

/// The rows a node will make, and what's known about their columns.
#[derive(Clone, Copy, Debug)]
pub struct Estimate {
    /// How many rows it will make.
    pub rows: RowEstimate,
    /// When several selective filters run one after another, the rows
    /// expected before the first of them. Each estimates from this count
    /// rather than from the previous filter's result: filters on one table
    /// tend to be correlated, so multiplying their selectivities would
    /// estimate far too few rows.
    selective_base: Option<u64>,
    /// The columns known to be keys, and what kind of key each is: the
    /// first `key_count` entries. Any more are dropped, which only means
    /// less is known.
    keys: [(ColumnId, Key); KEYS_MAX],
    /// How many of `keys` are known.
    key_count: u8,
}

/// The most keys an `Estimate` knows of.
const KEYS_MAX: usize = 4;

impl Estimate {
    /// Nothing known.
    pub const UNKNOWN: Estimate = Estimate {
        rows: RowEstimate::UNKNOWN,
        selective_base: None,
        keys: [(0, Key::None); KEYS_MAX],
        key_count: 0,
    };

    /// `rows` rows, with nothing known about their columns.
    pub fn of(rows: RowEstimate) -> Estimate {
        Estimate { rows, ..Estimate::UNKNOWN }
    }

    /// What kind of key `column` is known to be.
    pub fn key(&self, column: ColumnId) -> Key {
        let found = self.known_keys().iter().find(|&&(c, _)| c == column);
        found.map_or(Key::None, |&(_, key)| key)
    }

    /// Records that `column` is a key of kind `key`, if there's room.
    pub fn add_key(&mut self, column: ColumnId, key: Key) {
        let Some(slot) = self.keys.get_mut(self.key_count as usize) else { return };
        if key != Key::None {
            *slot = (column, key);
            self.key_count += 1;
        }
    }

    pub(crate) fn known_keys(&self) -> &[(ColumnId, Key)] {
        at!(self.keys, ..self.key_count as usize)
    }

    /// For an op that may drop rows: its keys stay unique, but a sequence
    /// may now have gaps, so none is known to be a sequence any more.
    pub fn demote_ids(&mut self) {
        for (_, key) in at_mut!(self.keys, ..self.key_count as usize) {
            *key = (*key).min(Key::Unique);
        }
    }

    /// For an op other than a filter: filters after it don't share a
    /// selective base with filters before it.
    pub fn after_filters(mut self) -> Estimate {
        self.selective_base = None;
        self
    }

    /// Applies a filter whose selectivity can't be estimated: it is assumed
    /// to keep half the rows.
    pub fn keep_some(&mut self) {
        if self.rows.estimated > 1 {
            self.rows.estimated /= 2;
        }
    }

    /// Applies a filter keeping the rows where `column` has one value. On a
    /// unique column that leaves at most one row. Otherwise it leaves the
    /// rows one value would have, assuming twice the log of the row count
    /// distinct values.
    pub fn keep_equal(&mut self, column: ColumnId) {
        self.keep_values(column, 1);
    }

    /// Applies a filter keeping the rows where `column` has one of `count`
    /// values: `keep_equal`'s estimate for each of them.
    pub fn keep_in(&mut self, column: ColumnId, count: u64) {
        self.keep_values(column, count);
    }

    fn keep_values(&mut self, column: ColumnId, count: u64) {
        if self.key(column) >= Key::Unique {
            self.rows.max = self.rows.max.min(count);
            self.rows.estimated = self.rows.estimated.min(count);
            return;
        }
        if self.rows.estimated <= 1 {
            return;
        }
        let base = *self.selective_base.get_or_insert(self.rows.estimated);
        let per_value = base / (2 * u64::from(base.ilog2())).max(1);
        let kept = per_value.saturating_mul(count).clamp(1, base);
        self.rows.estimated = self.rows.estimated.min(kept);
    }

    /// Applies a limit: skipping `skip` rows, then keeping at most `count`.
    pub fn keep_at_most(&mut self, skip: u64, count: u64) {
        self.rows.max = self.rows.max.saturating_sub(skip).min(count);
        self.rows.estimated = self.rows.estimated.saturating_sub(skip).min(self.rows.max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_keep_fewer_rows_the_more_selective_they_are() {
        let mut ids = Estimate::of(RowEstimate::exactly(1 << 20));
        ids.add_key(0, Key::Id);
        // A unique column has one row of each value.
        let mut one = ids;
        one.keep_equal(0);
        assert_eq!(one.rows, RowEstimate::exactly(1));
        one.keep_in(0, 3);
        assert_eq!(one.rows, RowEstimate::exactly(1));
        let mut three = ids;
        three.keep_in(0, 3);
        assert_eq!(three.rows, RowEstimate::exactly(3));

        // On another column, 1M rows / (2 * 20) per value, and a second
        // filter doesn't multiply the first's selectivity.
        let mut equal = ids;
        equal.keep_equal(1);
        assert_eq!(equal.rows.estimated, 26_214);
        equal.keep_equal(2);
        assert_eq!(equal.rows.estimated, 26_214);
        assert_eq!(equal.rows.max, 1 << 20);
        // A range keeps half.
        equal.keep_some();
        assert_eq!(equal.rows.estimated, 13_107);

        let mut limited = ids;
        limited.keep_at_most(10, 100);
        assert_eq!(limited.rows, RowEstimate::exactly(100));
    }
}
