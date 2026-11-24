//! Computing columns from each row.

use crate::containers::SmallVec;
use crate::error::Error;
use crate::expression::{ComputeItem, ComputeOp, expr_can_be_null};
use crate::plan::DynOp;
use crate::query::{Builder, Column, Expr, Relation};

/// Rows being extended, by computed columns added one at a time.
pub struct Extension {
    /// The rows extended.
    relation: Relation,
    /// The columns, each with what computes it.
    items: SmallVec<ComputeItem>,
}

/// Computes columns.
pub trait Compute {
    /// The rows of `relation`, to be extended by up to `len` columns, which
    /// [`Compute::compute`] adds.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    fn extend(&mut self, relation: Relation, len: usize) -> Result<Extension, Error>;

    /// A column of `extension`, of `expr`, computed for each row.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    fn compute(&mut self, extension: &mut Extension, expr: Expr) -> Result<Column, Error>;

    /// The rows of `extension`, with its columns.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    fn extended(&mut self, extension: Extension) -> Result<Relation, Error>;
}

impl Compute for Builder<'_> {
    fn extend(&mut self, relation: Relation, len: usize) -> Result<Extension, Error> {
        Ok(Extension { relation, items: SmallVec::fixed(self.env, len)? })
    }

    fn compute(&mut self, extension: &mut Extension, expr: Expr) -> Result<Column, Error> {
        let (data_type, expr) = (expr.data_type, expr.plan_expr());
        let (columns, nodes) = (&self.plan.columns, self.plan.exprs.nodes(expr));
        let nullable = expr_can_be_null(nodes, &|column| at!(columns, column as usize).nullable);
        let column = self.add_column(data_type, nullable)?;
        extension.items.push(ComputeItem { output: column.0, expr })?;
        Ok(column)
    }

    fn extended(&mut self, extension: Extension) -> Result<Relation, Error> {
        let Extension { relation, items } = extension;
        if items.is_empty() {
            return Ok(relation);
        }
        self.add(DynOp::new(self.env, ComputeOp { items })?, [relation])
    }
}
