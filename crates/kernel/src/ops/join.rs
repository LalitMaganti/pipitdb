//! Joining two relations' rows.

use crate::containers::{Bitmap, SmallVec};
use crate::error::Error;
pub use crate::join::JoinKind;
use crate::join::JoinOp;
use crate::ops::check_condition;
use crate::plan::{ColumnId, DynOp, conjuncts_of};
use crate::query::{Builder, Column, Expr, Relation};

/// Two relations being joined, until their condition is built.
pub struct Joining {
    /// How rows are paired.
    kind: JoinKind,
    /// The probe side, then the build side.
    inputs: [Relation; 2],
    /// Each side's columns.
    columns: [SmallVec<ColumnId>; 2],
}

/// Joins relations.
pub trait Join {
    /// A join of `probe` and `build`, as `kind` says, whose columns are
    /// `probe_columns` of `probe`, then `build_columns` of `build`, which a
    /// left join makes nullable. [`Join::joined`] gives it its condition.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    fn join(
        &mut self,
        kind: JoinKind,
        probe: (Relation, impl ExactSizeIterator<Item = Column>),
        build: (Relation, impl ExactSizeIterator<Item = Column>),
    ) -> Result<Joining, Error>;

    /// Each pair of `joining`'s rows that `condition` is true for, or every
    /// pair without one.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if `condition` isn't an integer, and
    /// [`Error::OutOfMemory`] if `env` has too little left.
    fn joined(&mut self, joining: Joining, condition: Option<Expr>) -> Result<Relation, Error>;
}

impl Join for Builder<'_> {
    fn join(
        &mut self,
        kind: JoinKind,
        (probe, probe_columns): (Relation, impl ExactSizeIterator<Item = Column>),
        (build, build_columns): (Relation, impl ExactSizeIterator<Item = Column>),
    ) -> Result<Joining, Error> {
        let env = self.env;
        let probe_columns = SmallVec::fixed_from(env, probe_columns.map(|column| column.0))?;
        let build_columns = SmallVec::fixed_from(env, build_columns.map(|column| column.0))?;
        if kind == JoinKind::Left {
            for &column in build_columns.iter() {
                at_mut!(self.plan.columns, column as usize).nullable = true;
            }
        }
        let mut columns = [probe_columns, build_columns];
        // Ids are places in the plan's columns, and no column is both sides'.
        let mut seen = Bitmap::new(env, self.plan.columns.len())?;
        for columns in &mut columns {
            distinct(&mut seen, columns);
        }
        Ok(Joining { kind, inputs: [probe, build], columns })
    }

    fn joined(&mut self, joining: Joining, condition: Option<Expr>) -> Result<Relation, Error> {
        let env = self.env;
        let Joining { kind, inputs, columns } = joining;
        let conjuncts = match condition {
            Some(condition) => {
                check_condition(condition.data_type)?;
                conjuncts_of(env, &self.plan.exprs, condition.plan_expr())?
            }
            None => SmallVec::fixed(env, 0)?,
        };
        let op = JoinOp::new(env, kind, conjuncts, columns)?;
        self.add(DynOp::new(env, op)?, inputs)
    }
}

/// Keeps the first of each column in `columns` not in `seen`, adding them:
/// a column of two names, as an alias makes, is joined once.
fn distinct(seen: &mut Bitmap, columns: &mut SmallVec<ColumnId>) {
    let mut kept = 0;
    for i in 0..columns.len() {
        let column = *at!(columns, i);
        if !seen.get(column as usize) {
            seen.set(column as usize);
            *at_mut!(columns, kept) = column;
            kept += 1;
        }
    }
    columns.truncate(kept);
}
