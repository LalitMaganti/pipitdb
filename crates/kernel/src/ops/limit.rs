//! Keeping some of the rows.

use crate::error::Error;
use crate::limit::LimitOp;
use crate::plan::DynOp;
use crate::query::{Builder, Relation};

/// Limits rows.
pub trait Limit {
    /// The first `count` rows of `relation`, after skipping `offset`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    fn limit(&mut self, relation: Relation, count: u64, offset: u64) -> Result<Relation, Error>;
}

impl Limit for Builder<'_> {
    fn limit(&mut self, relation: Relation, count: u64, offset: u64) -> Result<Relation, Error> {
        self.add(DynOp::new(self.env, LimitOp { count, offset })?, [relation])
    }
}
