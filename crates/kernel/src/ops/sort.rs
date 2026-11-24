//! Sorting rows.

use crate::containers::SmallVec;
use crate::error::Error;
use crate::order_by::OrderByOp;
use crate::plan::{DynOp, SortKey};
use crate::query::{Builder, Column, Relation};

/// Rows being sorted, by keys added one at a time.
pub struct Sorting {
    /// The rows sorted.
    relation: Relation,
    /// The keys, in turn.
    keys: SmallVec<SortKey>,
}

impl Sorting {
    /// Sorts by `column`, in `order`, after the keys before.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if there are more keys than the sorting was
    /// made for.
    pub fn by(&mut self, column: Column, order: Order) -> Result<(), Error> {
        let descending = order == Order::Descending;
        Ok(self.keys.push(SortKey { column: column.0, descending })?)
    }
}

/// Which way a sort key orders rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Order {
    Ascending,
    Descending,
}

/// Sorts rows.
pub trait Sort {
    /// The rows of `relation`, to be sorted by up to `len` keys, which
    /// [`Sorting::by`] adds.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    fn sort(&mut self, relation: Relation, len: usize) -> Result<Sorting, Error>;

    /// The rows of `sorting`, sorted by its keys in turn, stably.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    fn sorted(&mut self, sorting: Sorting) -> Result<Relation, Error>;
}

impl Sort for Builder<'_> {
    fn sort(&mut self, relation: Relation, len: usize) -> Result<Sorting, Error> {
        Ok(Sorting { relation, keys: SmallVec::fixed(self.env, len)? })
    }

    fn sorted(&mut self, sorting: Sorting) -> Result<Relation, Error> {
        let Sorting { relation, keys } = sorting;
        self.add(DynOp::new(self.env, OrderByOp { keys })?, [relation])
    }
}
