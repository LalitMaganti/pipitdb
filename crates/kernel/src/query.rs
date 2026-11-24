//! Building queries.
//!
//! A [`Builder`] builds a [`Query`] from tables, a relation at a time: each
//! operation takes the relation it reads and returns a new one, whose
//! columns the next operations read.

use crate::column::Forms;
use crate::containers::SmallVec;
use crate::env::Env;
use crate::error::Error;
use crate::plan::{ColumnId, DynOp, LogicalPlan, PlanNodeId, ScanColumn, ScanOp};
use crate::scannable::DynScannable;

/// What a front end asks to compute: a query, built but not yet prepared.
pub struct Query<'e> {
    pub(crate) plan: LogicalPlan<'e>,
}

impl<'e> Query<'e> {
    /// The plan the front end built.
    pub fn plan(&self) -> &LogicalPlan<'e> {
        &self.plan
    }
}

/// Builds a query, a relation at a time.
pub struct Builder<'e> {
    /// Where the query's memory comes from.
    env: &'e Env,
    /// The plan built so far.
    #[doc(hidden)]
    pub plan: LogicalPlan<'e>,
}

/// Rows an operation makes, which one other operation reads.
#[derive(Debug)]
pub struct Relation(PlanNodeId);

/// A column of a relation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Column(ColumnId);

/// Columns made together, as a scan's are.
#[derive(Clone, Copy, Debug)]
pub struct Columns {
    first: ColumnId,
    count: u32,
}

impl Columns {
    /// Each of them, in order.
    pub fn iter(self) -> impl Iterator<Item = Column> {
        (self.first..self.first + self.count).map(Column)
    }
}

impl<'e> Builder<'e> {
    /// A builder of a query whose memory comes from `env`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn new(env: &'e Env) -> Result<Builder<'e>, Error> {
        Ok(Builder { env, plan: LogicalPlan::new(env)? })
    }

    /// The query whose rows are those of `relation`.
    #[expect(clippy::needless_pass_by_value, reason = "a relation is read once")]
    pub fn finish(mut self, relation: Relation) -> Query<'e> {
        self.plan.root = relation.0;
        Query { plan: self.plan }
    }

    /// The rows of `table`, of each of its columns, in order.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if the query would have more columns than a
    /// plan can.
    pub fn scan(&mut self, table: &'e DynScannable<'e>) -> Result<(Relation, Columns), Error> {
        let count = table.column_count();
        let mut reads = SmallVec::fixed(self.env, count as usize)?;
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_COLUMNS_MAX`")]
        let first = self.plan.columns.len() as ColumnId;
        for column in 0..count {
            let (name, data_type) = (table.column_name(column), table.column_type(column));
            let binding = self.plan.add_column(name, data_type, table.column_nullable(column))?;
            reads.push(ScanColumn { column, binding, forms: Forms::FLAT })?;
        }
        let node = self.plan.add_node(DynOp::new(self.env, ScanOp::new(table, reads))?, &[])?;
        Ok((Relation(node), Columns { first, count }))
    }

    // What front ends not yet building through it reach the plan by.

    #[doc(hidden)]
    pub fn relation(node: PlanNodeId) -> Relation {
        Relation(node)
    }

    #[doc(hidden)]
    pub fn node(relation: &Relation) -> PlanNodeId {
        relation.0
    }

    #[doc(hidden)]
    pub fn column_id(column: Column) -> ColumnId {
        column.0
    }
}
