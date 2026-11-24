//! Building queries.
//!
//! A [`Builder`] builds a [`Query`] from tables, a relation at a time: each
//! operation takes the relation it reads and returns a new one, whose
//! columns the next operations read.

use crate::column::Forms;
use crate::containers::SmallVec;
use crate::env::Env;
use crate::error::Error;
use crate::expression::Expr;
use crate::limit::LimitOp;
use crate::order_by::OrderByOp;
use crate::plan::{
    ColumnId, DynOp, FilterOp, LogicalPlan, PlanNodeId, ScanColumn, ScanOp, SortKey,
};
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

    /// The rows of `relation` that `condition`, an integer, is true for: not
    /// null or 0.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn filter(&mut self, relation: Relation, condition: Expr) -> Result<Relation, Error> {
        let filter = FilterOp::new(self.env, &self.plan.exprs, condition)?;
        self.add(DynOp::new(self.env, filter)?, relation)
    }

    /// The rows of `relation`, to be sorted by up to `len` keys, which
    /// [`Sorting::by`] adds.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn sort(&mut self, relation: Relation, len: usize) -> Result<Sorting, Error> {
        Ok(Sorting { relation, keys: SmallVec::fixed(self.env, len)? })
    }

    /// The rows of `sorting`, sorted by its keys in turn, stably.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn sorted(&mut self, sorting: Sorting) -> Result<Relation, Error> {
        let Sorting { relation, keys } = sorting;
        self.add(DynOp::new(self.env, OrderByOp { keys })?, relation)
    }

    /// The first `count` rows of `relation`, after skipping `offset`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn limit(
        &mut self,
        relation: Relation,
        count: u64,
        offset: u64,
    ) -> Result<Relation, Error> {
        self.add(DynOp::new(self.env, LimitOp { count, offset })?, relation)
    }

    /// Adds `op`, reading `relation`.
    #[expect(clippy::needless_pass_by_value, reason = "a relation is read once")]
    fn add(&mut self, op: DynOp<'e>, relation: Relation) -> Result<Relation, Error> {
        Ok(Relation(self.plan.add_node(op, &[relation.0])?))
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

    #[doc(hidden)]
    pub fn column(id: ColumnId) -> Column {
        Column(id)
    }
}
