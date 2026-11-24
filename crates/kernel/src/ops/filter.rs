//! Keeping the rows a condition is true for.

use crate::error::Error;
use crate::ops::check_condition;
use crate::plan::{DynOp, FilterOp};
use crate::query::{Builder, Expr, Relation};

/// Filters rows.
pub trait Filter {
    /// The rows of `relation` that `condition`, an integer, is true for: not
    /// null or 0.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if `condition` isn't an integer, and
    /// [`Error::OutOfMemory`] if `env` has too little left.
    fn filter(&mut self, relation: Relation, condition: Expr) -> Result<Relation, Error>;
}

impl Filter for Builder<'_> {
    fn filter(&mut self, relation: Relation, condition: Expr) -> Result<Relation, Error> {
        check_condition(condition.data_type)?;
        let filter = FilterOp::new(self.env, &self.plan.exprs, condition.plan_expr())?;
        self.add(DynOp::new(self.env, filter)?, [relation])
    }
}
