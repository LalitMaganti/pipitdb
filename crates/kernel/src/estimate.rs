//! `Estimate`: what is known of a plan node's rows before any run, for
//! planning to choose by, as Perfetto's dataframe planner estimates the rows
//! a query leaves. Each op makes its own from its inputs'.

use crate::plan::ColumnId;
use crate::scannable::Key;

/// How many rows there can be, and how many planning expects.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RowEstimate {
    pub max: u64,
    pub estimated: u64,
}

impl RowEstimate {
    /// Nothing known: as many as can be.
    pub const UNKNOWN: RowEstimate = RowEstimate { max: u64::MAX, estimated: u64::MAX };

    /// Exactly `rows`.
    pub fn exactly(rows: u64) -> RowEstimate {
        RowEstimate { max: rows, estimated: rows }
    }
}

/// The rows a node makes, and what's known of their columns.
#[derive(Clone, Copy, Debug)]
pub struct Estimate {
    pub rows: RowEstimate,
    /// The rows expected before the first of the selective filters applied
    /// one after another, as each estimates from it: filters on one table
    /// tend to go together, so their selectivities don't multiply.
    selective_base: Option<u64>,
    /// Columns known to be keys, and what each is known to be: the first
    /// `key_count`. More are forgotten, which only means less is known.
    keys: [(ColumnId, Key); KEYS_MAX],
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

    /// `rows` rows, of whose columns nothing is known.
    pub fn of(rows: RowEstimate) -> Estimate {
        Estimate { rows, ..Estimate::UNKNOWN }
    }

    /// What `column` is known to be as a key.
    pub fn key(&self, column: ColumnId) -> Key {
        let found = self.known_keys().iter().find(|&&(c, _)| c == column);
        found.map_or(Key::None, |&(_, key)| key)
    }

    /// Knows `column` to be `key`, if there's room.
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

    /// Forgets which keys are sequences, for an op that may drop rows: they
    /// stay unique.
    pub fn demote_ids(&mut self) {
        for (_, key) in at_mut!(self.keys, ..self.key_count as usize) {
            *key = (*key).min(Key::Unique);
        }
    }

    /// As made by an op other than a filter, which a filter after it
    /// doesn't estimate together with any before it.
    pub fn after_filters(mut self) -> Estimate {
        self.selective_base = None;
        self
    }

    /// After a filter whose selectivity can't be told, which keeps half.
    pub fn keep_some(&mut self) {
        if self.rows.estimated > 1 {
            self.rows.estimated /= 2;
        }
    }

    /// After a filter keeping the rows with one value of `column`: at most
    /// one if it's unique, or else a share of them, as many as each of its
    /// values would have, the number of them taken as twice the rows' log.
    pub fn keep_equal(&mut self, column: ColumnId) {
        self.keep_values(column, 1);
    }

    /// After a filter keeping the rows with one of `count` values of
    /// `column`: as `keep_equal` for each.
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

    /// After keeping at most `count` rows from the `skip`th.
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

        // Of another, 1M rows / (2 * 20) each, and filters don't multiply.
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
