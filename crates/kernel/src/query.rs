//! Building queries.
//!
//! A [`Builder`] builds a [`Query`] from tables, a relation at a time: each
//! operation takes the relations it reads and returns a new one, whose
//! columns the next operations read. Expressions are built the same way,
//! from columns and constants, and checked when an operation takes them. Every
//! call takes `&self`, so calls nest, as the parts of an expression do.
//!
//! ```
//! use pipit_kernel::env::Env;
//! use pipit_kernel::error::Error;
//! use pipit_kernel::query::{Binary, Builder, Order, Query};
//! use pipit_kernel::scannable::DynScannable;
//!
//! /// `slices`' ten longest durations over 1000, of its columns `name` and
//! /// `dur`.
//! fn longest<'e>(env: &'e Env, slices: &'e DynScannable<'e>) -> Result<Query<'e>, Error> {
//!     let q = Builder::new(env)?;
//!     let slice = q.scan(slices, &[0, 1])?;
//!     let (name, dur) = (q.column(&slice, 0)?, q.column(&slice, 1)?);
//!     let long = q.binary(Binary::Greater, q.column_value(dur)?, q.int(1000)?)?;
//!     let slice = q.filter(slice, long)?;
//!     let slice = q.order_by(slice, &[(dur, Order::Descending)])?;
//!     let slice = q.limit(slice, 10, 0)?;
//!     q.finish(slice, &[(name, "name"), (dur, "dur")])
//! }
//! ```
//!
//! # Relations
//!
//! A relation is the input of one operation, as a query is a tree: a
//! [`Relation`] is moved into the operation that reads it. Its columns are
//! read through [`Column`]s, which stay valid for as long as the builder.
//!
//! # Builders
//!
//! Relations, columns and expressions belong to the builder that made them.
//! Using one with another fails with [`Error::WrongBuilder`].

use core::cell::{RefCell, RefMut};
use core::sync::atomic::{AtomicU32, Ordering};

pub use crate::aggregate::Function;
use crate::aggregate::{Aggregate, AggregateOp};
use crate::allocator::AllocError;
use crate::column::{DataType, Forms};
use crate::containers::SmallVec;
use crate::env::Env;
use crate::error::Error;
pub use crate::error::TypeError;
use crate::expression::{
    self, ComputeItem, ComputeOp, EXPR_NODES_MAX, ExprOp, ExprStep, PLAN_EXPR_NODES_MAX,
    PLAN_EXPR_VALUES_MAX, PRESENT, check_expr_type, columns_of, expr_can_be_null,
};
use crate::function::DynFunction;
pub use crate::join::JoinKind;
use crate::join::JoinOp;
use crate::limit::LimitOp;
use crate::order_by::OrderByOp;
use crate::plan::{
    ColumnId, DynOp, FilterOp, LogicalPlan, NamedColumn, PLAN_COLUMNS_MAX, PLAN_NODES_MAX,
    PlanNodeId, ScanColumn, ScanOp, SortKey, conjuncts_of,
};
use crate::scannable::DynScannable;
use crate::settings::PLAN_DEPTH_MAX;

/// The next builder's id.
static BUILDERS: AtomicU32 = AtomicU32::new(0);

/// The most columns of all a builder's relations, counted once for each.
const RELATION_COLUMNS_MAX: usize = 1 << 20;

/// What a front end asks to compute: a query, built but not yet prepared.
pub struct Query<'e> {
    pub(crate) plan: LogicalPlan<'e>,
}

impl<'e> Query<'e> {
    /// The query `build` makes, with memory from `env`.
    pub fn new<E: From<AllocError>>(
        env: &'e Env,
        build: impl FnOnce(&'e Env) -> Result<LogicalPlan<'e>, E>,
    ) -> Result<Query<'e>, E> {
        Ok(Query { plan: build(env)? })
    }

    /// The plan the front end built.
    pub fn plan(&self) -> &LogicalPlan<'e> {
        &self.plan
    }
}

/// Builds a query, a relation at a time.
pub struct Builder<'e> {
    /// What it has built, borrowed by each call.
    state: RefCell<State<'e>>,
}

/// What a builder has built.
struct State<'e> {
    /// Where the query's memory comes from.
    env: &'e Env,
    /// Which builder it is, so what another made isn't used with it.
    id: u32,
    /// The plan built so far.
    plan: LogicalPlan<'e>,
    /// Each relation's node, and where its columns are in `columns`.
    relations: SmallVec<(PlanNodeId, u32, u32)>,
    /// The columns of every relation, a run for each.
    columns: SmallVec<ColumnId>,
    /// Every expression's nodes, each reading only nodes before it.
    nodes: SmallVec<ExprStep>,
    /// For each of `nodes`, the copy that last copied it, and where to.
    copies: SmallVec<(u32, u32)>,
    /// The last copy's number.
    stamp: u32,
}

/// Rows an operation makes, which one other operation reads.
#[derive(Debug)]
pub struct Relation {
    /// The builder that made it.
    builder: u32,
    /// Which of its relations it is.
    index: u32,
}

/// A column of a relation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Column {
    /// The builder that made it.
    builder: u32,
    /// Which of the plan's columns it is.
    id: ColumnId,
}

/// A value computed for each row, from columns and constants.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Expr {
    /// The builder that made it.
    builder: u32,
    /// Its last node in the builder's.
    node: u32,
}

/// An operation of two values.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Binary {
    Add,
    Subtract,
    Multiply,
    Divide,
    /// The remainder of integers.
    Modulo,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
}

/// The step computing each `Binary`, by its order.
static BINARY_OPS: [ExprOp; 13] = [
    ExprOp::Add,
    ExprOp::Sub,
    ExprOp::Mul,
    ExprOp::Div,
    ExprOp::Mod,
    ExprOp::Eq,
    ExprOp::Ne,
    ExprOp::Lt,
    ExprOp::Le,
    ExprOp::Gt,
    ExprOp::Ge,
    ExprOp::And,
    ExprOp::Or,
];

/// An operation of one value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unary {
    Not,
    /// Whether it's null: never null itself.
    IsNull,
    /// Whether it isn't null: never null itself.
    IsNotNull,
}

/// The step computing each `Unary`, by its order.
static UNARY_OPS: [ExprOp; 3] = [ExprOp::Not, ExprOp::IsNull, ExprOp::IsNotNull];

/// Which way a sort key orders rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Order {
    Ascending,
    Descending,
}

/// An aggregate of a group's rows.
#[derive(Clone, Copy, Debug)]
pub struct Aggregation<'n> {
    /// What it computes.
    pub function: Function,
    /// The column it reads, or `None` for `Function::CountStar`.
    pub input: Option<Column>,
    /// Whether it reads each distinct value once.
    pub distinct: bool,
    /// The name of the column it makes.
    pub name: &'n str,
}

impl<'e> Builder<'e> {
    /// A builder of a query whose memory comes from `env`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn new(env: &'e Env) -> Result<Builder<'e>, Error> {
        Ok(Builder { state: RefCell::new(State::new(env)?) })
    }

    /// The query whose rows are those of `relation`, and whose columns are
    /// `output`, each named.
    ///
    /// # Errors
    ///
    /// [`Error::NotInScope`] if a column isn't `relation`'s.
    pub fn finish(self, relation: Relation, output: &[(Column, &str)]) -> Result<Query<'e>, Error> {
        self.state.into_inner().finish(relation, output)
    }

    /// The rows of `table`, of its columns `columns`, in that order.
    ///
    /// # Errors
    ///
    /// [`Error::NotInScope`] if `table` has no such column, and
    /// [`Error::TooManyColumns`] if the query would have more than a plan
    /// can.
    pub fn scan(&self, table: &'e DynScannable<'e>, columns: &[u32]) -> Result<Relation, Error> {
        self.state().scan(table, columns)
    }

    /// The rows of `relation` that `condition`, an integer, is true for: not
    /// null or 0.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if `condition` isn't an integer, and
    /// [`Error::NotInScope`] if it reads a column that isn't `relation`'s.
    pub fn filter(&self, relation: Relation, condition: Expr) -> Result<Relation, Error> {
        self.state().filter(relation, condition)
    }

    /// `relation`'s columns, then a column for each of `items`, computing its
    /// expression, under its name.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if an expression's types don't fit, and
    /// [`Error::NotInScope`] if it reads a column that isn't `relation`'s.
    pub fn extend(&self, relation: Relation, items: &[(Expr, &str)]) -> Result<Relation, Error> {
        self.state().extend(relation, items)
    }

    /// The rows of `relation`, of only `columns`, in that order.
    ///
    /// # Errors
    ///
    /// [`Error::NotInScope`] if a column isn't `relation`'s.
    pub fn select(&self, relation: Relation, columns: &[Column]) -> Result<Relation, Error> {
        self.state().select(relation, columns)
    }

    /// Each pair of a row of `left` and a row of `right` that `on`, an
    /// integer, is true for, or every pair with no condition, with `left`'s
    /// columns then `right`'s. A `JoinKind::Left` join also has each row of
    /// `left` with no such row of `right`, with nulls for `right`'s columns.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if `on` isn't an integer, and
    /// [`Error::NotInScope`] if it reads a column of neither.
    pub fn join(
        &self,
        kind: JoinKind,
        left: Relation,
        right: Relation,
        on: Option<Expr>,
    ) -> Result<Relation, Error> {
        self.state().join(kind, left, right, on)
    }

    /// A row for each group of `relation`'s rows with the same `keys`, or one
    /// for all its rows with none: the keys, then each of `aggregations`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if a key isn't an integer or a string, or an
    /// aggregation's function doesn't read its column's type or has no
    /// column when it needs one; and [`Error::NotInScope`] if a column isn't
    /// `relation`'s.
    pub fn aggregate(
        &self,
        relation: Relation,
        keys: &[Column],
        aggregations: &[Aggregation<'_>],
    ) -> Result<Relation, Error> {
        self.state().aggregate(relation, keys, aggregations)
    }

    /// `relation`'s rows sorted by `keys`, the first first, stably.
    ///
    /// # Errors
    ///
    /// [`Error::NotInScope`] if a key isn't `relation`'s.
    pub fn order_by(
        &self,
        relation: Relation,
        keys: &[(Column, Order)],
    ) -> Result<Relation, Error> {
        self.state().order_by(relation, keys)
    }

    /// `count` of `relation`'s rows, after skipping `offset`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `relation` is another builder's.
    pub fn limit(&self, relation: Relation, count: u64, offset: u64) -> Result<Relation, Error> {
        self.state().limit(relation, count, offset)
    }

    /// Column `index` of `relation`.
    ///
    /// # Errors
    ///
    /// [`Error::NotInScope`] if it has no such column.
    pub fn column(&self, relation: &Relation, index: usize) -> Result<Column, Error> {
        self.state().column(relation, index)
    }

    /// How many columns `relation` has.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `relation` is another builder's.
    pub fn column_count(&self, relation: &Relation) -> Result<usize, Error> {
        self.state().column_count(relation)
    }

    /// The type of `expr`'s values.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if its parts' types don't fit together.
    pub fn type_of(&self, expr: Expr) -> Result<DataType, Error> {
        self.state().type_of_expr(expr)
    }

    /// The type of `column`'s values.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `column` is another builder's.
    pub fn column_type(&self, column: Column) -> Result<DataType, Error> {
        self.state().column_type(column)
    }

    /// The value of `column`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `column` is another builder's.
    pub fn column_value(&self, column: Column) -> Result<Expr, Error> {
        self.state().column_value(column)
    }

    /// The integer `value`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the builder holds as many nodes as it can.
    pub fn int(&self, value: i64) -> Result<Expr, Error> {
        self.state().int(value)
    }

    /// The double `value`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the builder holds as many nodes as it can.
    pub fn double(&self, value: f64) -> Result<Expr, Error> {
        self.state().double(value)
    }

    /// The string `value`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left for it.
    pub fn string(&self, value: &[u8]) -> Result<Expr, Error> {
        self.state().string(value)
    }

    /// A null, of whatever type the expression around it needs.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the builder holds as many nodes as it can.
    pub fn null(&self) -> Result<Expr, Error> {
        self.state().null()
    }

    /// `a op b`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `a` or `b` is another builder's.
    pub fn binary(&self, op: Binary, a: Expr, b: Expr) -> Result<Expr, Error> {
        self.state().binary(op, a, b)
    }

    /// `op a`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `a` is another builder's.
    pub fn unary(&self, op: Unary, a: Expr) -> Result<Expr, Error> {
        self.state().unary(op, a)
    }

    /// Whether the string `a` matches the `LIKE` pattern `pattern`: `%` for
    /// any run of bytes, and `_` for any one.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `a` is another builder's.
    pub fn like(&self, a: Expr, pattern: &[u8]) -> Result<Expr, Error> {
        self.state().like(a, pattern)
    }

    /// Whether the integer `a` is one of `values`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the query would hold more values than a plan
    /// can.
    pub fn is_in(&self, a: Expr, values: &[i64]) -> Result<Expr, Error> {
        self.state().is_in(a, values)
    }

    /// Whether the string `a` is one of `values`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the query would hold more values than a plan
    /// can.
    pub fn is_in_strings(&self, a: Expr, values: &[&[u8]]) -> Result<Expr, Error> {
        self.state().is_in_strings(a, values)
    }

    /// For each row, the value of the first of `branches` whose condition
    /// isn't null or 0, else `otherwise`. A branch without a condition is
    /// taken where its value isn't null.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if an expression is another builder's.
    pub fn case(&self, branches: &[(Option<Expr>, Expr)], otherwise: Expr) -> Result<Expr, Error> {
        self.state().case(branches, otherwise)
    }

    /// `function` called with `arguments`, each converted to its type, and
    /// giving a value of type `result`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the query has as many functions as a plan can.
    pub fn call(
        &self,
        function: DynFunction,
        arguments: &[(Expr, DataType)],
        result: DataType,
    ) -> Result<Expr, Error> {
        self.state().call(function, arguments, result)
    }

    /// Its state, which no call into it holds while another runs.
    fn state(&self) -> RefMut<'_, State<'e>> {
        let Ok(state) = self.state.try_borrow_mut() else { failed!() };
        state
    }
}

#[expect(clippy::needless_pass_by_value, reason = "a relation is moved, as one operation reads it")]
impl<'e> State<'e> {
    fn new(env: &'e Env) -> Result<State<'e>, Error> {
        Ok(State {
            env,
            id: BUILDERS.fetch_add(1, Ordering::Relaxed),
            plan: LogicalPlan::new(env)?,
            relations: SmallVec::new(env, PLAN_NODES_MAX)?,
            columns: SmallVec::new(env, RELATION_COLUMNS_MAX)?,
            nodes: SmallVec::new(env, PLAN_EXPR_NODES_MAX)?,
            copies: SmallVec::new(env, PLAN_EXPR_NODES_MAX)?,
            stamp: 0,
        })
    }

    fn finish(mut self, relation: Relation, output: &[(Column, &str)]) -> Result<Query<'e>, Error> {
        let (node, ..) = self.relation(&relation)?;
        for &(column, name) in output {
            self.in_scope(&relation, column)?;
            let name = self.plan.names.add(name)?;
            self.plan.output.push(NamedColumn { name, id: column.id })?;
        }
        self.plan.root = node;
        Ok(Query { plan: self.plan })
    }

    // Relations.

    fn scan(&mut self, table: &'e DynScannable<'e>, columns: &[u32]) -> Result<Relation, Error> {
        if columns.iter().any(|&column| column >= table.column_count()) {
            return Err(Error::NotInScope);
        }
        self.room_for(columns.len())?;
        let mut reads = SmallVec::fixed(self.env, columns.len())?;
        let start = self.columns.len();
        for &column in columns {
            let (name, data_type) = (table.column_name(column), table.column_type(column));
            let binding = self.plan.add_column(name, data_type, table.column_nullable(column))?;
            reads.push(ScanColumn { column, binding, forms: Forms::FLAT })?;
            self.columns.push(binding.id)?;
        }
        let node = self.plan.add_node(DynOp::new(self.env, ScanOp::new(table, reads))?, &[])?;
        self.add_relation(node, start)
    }

    fn filter(&mut self, relation: Relation, condition: Expr) -> Result<Relation, Error> {
        let (child, start, end) = self.relation(&relation)?;
        let condition = self.condition(condition, &[(start, end)])?;
        let filter = FilterOp::new(self.env, &self.plan.exprs, condition)?;
        let node = self.plan.add_node(DynOp::new(self.env, filter)?, &[child])?;
        self.add_relation_of(node, &[(start, end)])
    }

    fn extend(&mut self, relation: Relation, items: &[(Expr, &str)]) -> Result<Relation, Error> {
        let (child, start, end) = self.relation(&relation)?;
        self.room_for(items.len())?;
        let mut computed = SmallVec::fixed(self.env, items.len())?;
        let mut outputs = SmallVec::fixed(self.env, items.len())?;
        for &(expr, name) in items {
            let (expr, data_type) = self.compile(expr, &[(start, end)])?;
            let columns = &self.plan.columns;
            let nullable = expr_can_be_null(self.plan.exprs.nodes(expr), &|column| {
                at!(columns, column as usize).nullable
            });
            let output = self.plan.add_column(name, data_type, nullable)?.id;
            computed.push(ComputeItem { output, expr })?;
            outputs.push(output)?;
        }
        let node = if computed.is_empty() {
            child
        } else {
            self.plan.add_node(DynOp::new(self.env, ComputeOp { items: computed })?, &[child])?
        };
        let first = self.columns.len();
        for i in start..end {
            let column = *at!(self.columns, i as usize);
            self.columns.push(column)?;
        }
        for &output in outputs.iter() {
            self.columns.push(output)?;
        }
        self.add_relation(node, first)
    }

    fn select(&mut self, relation: Relation, columns: &[Column]) -> Result<Relation, Error> {
        let (node, ..) = self.relation(&relation)?;
        for &column in columns {
            self.in_scope(&relation, column)?;
        }
        let start = self.columns.len();
        for column in columns {
            self.columns.push(column.id)?;
        }
        self.add_relation(node, start)
    }

    fn join(
        &mut self,
        kind: JoinKind,
        left: Relation,
        right: Relation,
        on: Option<Expr>,
    ) -> Result<Relation, Error> {
        let (left_node, left_start, left_end) = self.relation(&left)?;
        let (right_node, right_start, right_end) = self.relation(&right)?;
        let sides = [(left_start, left_end), (right_start, right_end)];
        if kind == JoinKind::Left {
            for &column in at!(self.columns, right_start as usize..right_end as usize) {
                at_mut!(self.plan.columns, column as usize).nullable = true;
            }
        }
        let conjuncts = match on {
            Some(on) => {
                let on = self.condition(on, &sides)?;
                conjuncts_of(self.env, &self.plan.exprs, on)?
            }
            None => SmallVec::fixed(self.env, 0)?,
        };
        let columns = sides.map(|(start, end)| {
            SmallVec::fixed_from(
                self.env,
                at!(self.columns, start as usize..end as usize).iter().copied(),
            )
        });
        let [left_columns, right_columns] = columns;
        let join = JoinOp::new(self.env, kind, conjuncts, [left_columns?, right_columns?])?;
        let node = self.plan.add_node(DynOp::new(self.env, join)?, &[left_node, right_node])?;
        self.add_relation_of(node, &sides)
    }

    fn aggregate(
        &mut self,
        relation: Relation,
        keys: &[Column],
        aggregations: &[Aggregation<'_>],
    ) -> Result<Relation, Error> {
        let (child, ..) = self.relation(&relation)?;
        self.room_for(aggregations.len())?;
        let mut key_ids = SmallVec::fixed(self.env, keys.len())?;
        for &key in keys {
            self.in_scope(&relation, key)?;
            if !matches!(self.type_of(key.id), DataType::Int64 | DataType::String) {
                return Err(Error::WrongType(TypeError::GroupKey));
            }
            key_ids.push(key.id)?;
        }
        let mut aggregates = SmallVec::fixed(self.env, aggregations.len())?;
        for aggregation in aggregations {
            let Aggregation { function, input, distinct, name } = *aggregation;
            if let Some(input) = input {
                self.in_scope(&relation, input)?;
            }
            let input_type = input.map_or(DataType::Int64, |input| self.type_of(input.id));
            let counts_rows = function == Function::CountStar;
            if counts_rows != input.is_none() || input.is_some() && !function.reads(input_type) {
                return Err(Error::WrongType(TypeError::Aggregated));
            }
            let nullable = !matches!(function, Function::CountStar | Function::Count);
            let binding = self.plan.add_column(name, function.output_type(input_type), nullable)?;
            // The least or greatest of the distinct values is that of them all.
            let distinct = distinct && !matches!(function, Function::Min | Function::Max);
            let input = input.map(|input| input.id);
            aggregates.push(Aggregate { function, input, distinct, binding })?;
        }
        let start = self.columns.len();
        for &key in key_ids.iter() {
            self.columns.push(key)?;
        }
        for aggregate in aggregates.iter() {
            self.columns.push(aggregate.binding.id)?;
        }
        let aggregate = AggregateOp { keys: key_ids, aggregates, answer: None };
        let node = self.plan.add_node(DynOp::new(self.env, aggregate)?, &[child])?;
        self.add_relation(node, start)
    }

    fn order_by(
        &mut self,
        relation: Relation,
        keys: &[(Column, Order)],
    ) -> Result<Relation, Error> {
        let (child, start, end) = self.relation(&relation)?;
        let mut sort_keys = SmallVec::fixed(self.env, keys.len())?;
        for &(column, order) in keys {
            self.in_scope(&relation, column)?;
            sort_keys
                .push(SortKey { column: column.id, descending: order == Order::Descending })?;
        }
        let order_by = DynOp::new(self.env, OrderByOp { keys: sort_keys })?;
        let node = self.plan.add_node(order_by, &[child])?;
        self.add_relation_of(node, &[(start, end)])
    }

    fn limit(&mut self, relation: Relation, count: u64, offset: u64) -> Result<Relation, Error> {
        let (child, start, end) = self.relation(&relation)?;
        let limit = DynOp::new(self.env, LimitOp { count, offset })?;
        let node = self.plan.add_node(limit, &[child])?;
        self.add_relation_of(node, &[(start, end)])
    }

    fn column(&self, relation: &Relation, index: usize) -> Result<Column, Error> {
        let (_, start, end) = self.relation(relation)?;
        let columns = at!(self.columns, start as usize..end as usize);
        let id = *columns.get(index).ok_or(Error::NotInScope)?;
        Ok(Column { builder: self.id, id })
    }

    fn column_count(&self, relation: &Relation) -> Result<usize, Error> {
        let (_, start, end) = self.relation(relation)?;
        Ok((end - start) as usize)
    }

    fn column_type(&self, column: Column) -> Result<DataType, Error> {
        self.mine(column.builder)?;
        Ok(self.type_of(column.id))
    }

    // Expressions.

    fn column_value(&mut self, column: Column) -> Result<Expr, Error> {
        self.mine(column.builder)?;
        self.push(ExprStep::new(ExprOp::Column, column.id, 0, 0))
    }

    fn int(&mut self, value: i64) -> Result<Expr, Error> {
        self.push(ExprStep::new(ExprOp::Constant, 0, 0, value))
    }

    fn double(&mut self, value: f64) -> Result<Expr, Error> {
        let mut step = ExprStep::new(ExprOp::Constant, 0, 0, 0);
        (step.data_type, step.double_value) = (DataType::Float64, value);
        self.push(step)
    }

    fn string(&mut self, value: &[u8]) -> Result<Expr, Error> {
        let id = self.plan.add_literal(self.env, value)?;
        let mut step = ExprStep::new(ExprOp::Constant, id.raw_id(), 0, 0);
        step.data_type = DataType::String;
        self.push(step)
    }

    fn null(&mut self) -> Result<Expr, Error> {
        self.push(ExprStep::new(ExprOp::Null, 0, 0, 0))
    }

    fn binary(&mut self, op: Binary, a: Expr, b: Expr) -> Result<Expr, Error> {
        let (a, b) = (self.node(a)?, self.node(b)?);
        self.push(ExprStep::new(*at!(BINARY_OPS, op as usize), a, b, 0))
    }

    fn unary(&mut self, op: Unary, a: Expr) -> Result<Expr, Error> {
        let a = self.node(a)?;
        self.push(ExprStep::new(*at!(UNARY_OPS, op as usize), a, 0, 0))
    }

    fn like(&mut self, a: Expr, pattern: &[u8]) -> Result<Expr, Error> {
        let a = self.node(a)?;
        let pattern = self.plan.add_literal(self.env, pattern)?;
        self.push(ExprStep::new(ExprOp::Like, a, pattern.raw_id(), 0))
    }

    fn is_in(&mut self, a: Expr, values: &[i64]) -> Result<Expr, Error> {
        self.is_in_values(a, values, DataType::Int64)
    }

    fn is_in_strings(&mut self, a: Expr, values: &[&[u8]]) -> Result<Expr, Error> {
        let mut ids = SmallVec::fixed(self.env, values.len())?;
        for value in values {
            ids.push(i64::from(self.plan.add_literal(self.env, value)?.raw_id()))?;
        }
        self.is_in_values(a, &ids, DataType::String)
    }

    fn case(&mut self, branches: &[(Option<Expr>, Expr)], otherwise: Expr) -> Result<Expr, Error> {
        let otherwise = self.node(otherwise)?;
        let Ok(count) = u32::try_from(branches.len()) else { return Err(Error::TooLarge) };
        let first = self.next_node()?;
        for &(condition, value) in branches {
            let condition = match condition {
                Some(condition) => self.node(condition)?,
                None => PRESENT,
            };
            let value = self.node(value)?;
            self.push(ExprStep::new(ExprOp::When, condition, value, 0))?;
        }
        let mut case = ExprStep::new(ExprOp::Case, first, count, 0);
        case.c = otherwise;
        self.push(case)
    }

    fn call(
        &mut self,
        function: DynFunction,
        arguments: &[(Expr, DataType)],
        result: DataType,
    ) -> Result<Expr, Error> {
        let Ok(count) = u32::try_from(arguments.len()) else { return Err(Error::TooLarge) };
        let mut values = SmallVec::fixed(self.env, arguments.len())?;
        for &(value, _) in arguments {
            values.push(self.node(value)?)?;
        }
        let function = self.plan.add_function(self.env, function)?.ok_or(Error::TooLarge)?;
        let first = self.next_node()?;
        for (&value, &(_, data_type)) in values.iter().zip(arguments) {
            let mut argument = ExprStep::new(ExprOp::Argument, value, 0, 0);
            argument.data_type = data_type;
            self.push(argument)?;
        }
        let mut call = ExprStep::new(ExprOp::Call, first, count, 0);
        (call.data_type, call.c) = (result, function);
        self.push(call)
    }

    // Checks.

    /// Fails unless `builder` is this builder.
    fn mine(&self, builder: u32) -> Result<(), Error> {
        if builder == self.id { Ok(()) } else { Err(Error::WrongBuilder) }
    }

    /// `relation`'s node, and where its columns are in `columns`.
    fn relation(&self, relation: &Relation) -> Result<(PlanNodeId, u32, u32), Error> {
        self.mine(relation.builder)?;
        Ok(*at!(self.relations, relation.index as usize))
    }

    /// Fails unless `column` is one of `relation`'s.
    fn in_scope(&self, relation: &Relation, column: Column) -> Result<(), Error> {
        let (_, start, end) = self.relation(relation)?;
        self.mine(column.builder)?;
        let columns = at!(self.columns, start as usize..end as usize);
        if columns.contains(&column.id) { Ok(()) } else { Err(Error::NotInScope) }
    }

    /// Fails unless a plan has room for `count` more columns.
    fn room_for(&self, count: usize) -> Result<(), Error> {
        if self.plan.columns.len() + count > PLAN_COLUMNS_MAX {
            return Err(Error::TooManyColumns);
        }
        Ok(())
    }

    fn type_of(&self, column: ColumnId) -> DataType {
        at!(self.plan.columns, column as usize).data_type
    }

    // Relations' columns.

    /// A relation of `node`, whose columns are those from `start` of
    /// `columns`.
    fn add_relation(&mut self, node: PlanNodeId, start: usize) -> Result<Relation, Error> {
        #[expect(clippy::cast_possible_truncation, reason = "at most `RELATION_COLUMNS_MAX`")]
        let (start, end) = (start as u32, self.columns.len() as u32);
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
        let index = self.relations.len() as u32;
        self.relations.push((node, start, end))?;
        Ok(Relation { builder: self.id, index })
    }

    /// A relation of `node`, whose columns are copies of the runs `sides` of
    /// `columns`.
    fn add_relation_of(
        &mut self,
        node: PlanNodeId,
        sides: &[(u32, u32)],
    ) -> Result<Relation, Error> {
        let start = self.columns.len();
        for &(from, to) in sides {
            for i in from..to {
                let column = *at!(self.columns, i as usize);
                self.columns.push(column)?;
            }
        }
        self.add_relation(node, start)
    }

    // Expressions' nodes.

    /// Where `expr` is in `nodes`.
    fn node(&self, expr: Expr) -> Result<u32, Error> {
        self.mine(expr.builder)?;
        Ok(expr.node)
    }

    /// Where the next node goes.
    fn next_node(&self) -> Result<u32, Error> {
        u32::try_from(self.nodes.len()).map_err(|_| Error::TooLarge)
    }

    /// Appends `step`, and returns its expression.
    fn push(&mut self, step: ExprStep) -> Result<Expr, Error> {
        if self.nodes.len() == PLAN_EXPR_NODES_MAX {
            return Err(Error::TooLarge);
        }
        let node = self.next_node()?;
        self.nodes.push(step)?;
        Ok(Expr { builder: self.id, node })
    }

    /// `a IN` the `values` of `kind`, which go in the plan's expressions'
    /// values.
    fn is_in_values(&mut self, a: Expr, values: &[i64], kind: DataType) -> Result<Expr, Error> {
        let a = self.node(a)?;
        let len = values.len();
        if self.plan.exprs.values().len() + len > PLAN_EXPR_VALUES_MAX {
            return Err(Error::TooLarge);
        }
        let Ok(start) = u32::try_from(self.plan.exprs.values().len()) else { failed!() };
        for &value in values {
            self.plan.exprs.add_value(value)?;
        }
        let mut test = ExprStep::new(ExprOp::In, a, start, 0);
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_EXPR_VALUES_MAX`")]
        let len = len as u32;
        (test.c, test.data_type) = (len, kind);
        self.push(test)
    }

    /// `expr`, which must be an integer, added to the plan's expressions,
    /// reading only the columns of the runs `scope` of `columns`.
    fn condition(&mut self, expr: Expr, scope: &[(u32, u32)]) -> Result<expression::Expr, Error> {
        let (expr, data_type) = self.compile(expr, scope)?;
        if data_type != DataType::Int64 {
            return Err(Error::WrongType(TypeError::Condition));
        }
        Ok(expr)
    }

    /// `expr` added to the plan's expressions, and its type, reading only
    /// the columns of the runs `scope` of `columns`.
    fn compile(
        &mut self,
        expr: Expr,
        scope: &[(u32, u32)],
    ) -> Result<(expression::Expr, DataType), Error> {
        let (nodes, data_type) = self.flatten(expr)?;
        let mut outside = false;
        let columns = &self.columns;
        let inside = |column| {
            let run = |&(start, end): &(u32, u32)| at!(columns, start as usize..end as usize);
            scope.iter().any(|side| run(side).contains(&column))
        };
        let Some(last) = nodes.len().checked_sub(1) else { failed!() };
        #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
        columns_of(&nodes, last as u32, &mut |column| outside |= !inside(column));
        if outside {
            return Err(Error::NotInScope);
        }
        Ok((self.plan.exprs.add(&nodes)?, data_type))
    }

    fn type_of_expr(&mut self, expr: Expr) -> Result<DataType, Error> {
        Ok(self.flatten(expr)?.1)
    }

    /// The nodes `expr` reads, then it, each once, and its type.
    fn flatten(&mut self, expr: Expr) -> Result<(SmallVec<ExprStep>, DataType), Error> {
        let root = self.node(expr)?;
        while self.copies.len() < self.nodes.len() {
            self.copies.push((0, 0))?;
        }
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.copies.iter_mut().for_each(|copy| *copy = (0, 0));
            self.stamp = 1;
        }
        let mut flat = Flattening {
            env: self.env,
            nodes: &self.nodes,
            copies: &mut self.copies,
            stamp: self.stamp,
            out: SmallVec::new(self.env, EXPR_NODES_MAX)?,
        };
        flat.copy(root, 0)?;
        let columns = &self.plan.columns;
        let type_of = |column: ColumnId| at!(columns, column as usize).data_type;
        let data_type = check_expr_type(self.env, &flat.out, &type_of)?;
        Ok((flat.out, data_type))
    }
}

/// An expression's nodes being copied out of a builder's, each once.
struct Flattening<'a> {
    env: &'a Env,
    /// The builder's nodes.
    nodes: &'a [ExprStep],
    /// For each of `nodes`, the copy that last copied it, and where to.
    copies: &'a mut [(u32, u32)],
    /// Which copy this is.
    stamp: u32,
    /// The nodes copied, each reading only those before it.
    out: SmallVec<ExprStep>,
}

impl Flattening<'_> {
    /// Copies the nodes `node` reads, then it, `depth` deep, and returns
    /// where it is in `out`. A `Case`'s `When`s and a `Call`'s `Argument`s
    /// stay together, as they're read as a run.
    fn copy(&mut self, node: u32, depth: usize) -> Result<u32, Error> {
        let (stamp, at) = *at!(self.copies, node as usize);
        if stamp == self.stamp {
            return Ok(at);
        }
        let mut step = *at!(self.nodes, node as usize);
        let at = if matches!(step.op, ExprOp::Case | ExprOp::Call) {
            let mut run = SmallVec::fixed(self.env, step.b as usize)?;
            for member in step.a..step.a + step.b {
                run.push(self.copy_operands(*at!(self.nodes, member as usize), depth)?)?;
            }
            if step.op == ExprOp::Case {
                step.c = self.copy(step.c, depth + 1)?;
            }
            step.a = u32::try_from(self.out.len()).map_err(|_| Error::TooLarge)?;
            for &member in run.iter() {
                self.push(member)?;
            }
            self.push(step)?
        } else {
            let step = self.copy_operands(step, depth)?;
            self.push(step)?
        };
        *at_mut!(self.copies, node as usize) = (self.stamp, at);
        Ok(at)
    }

    /// `step`, reading what it reads once that's copied.
    fn copy_operands(&mut self, mut step: ExprStep, depth: usize) -> Result<ExprStep, Error> {
        if depth == PLAN_DEPTH_MAX {
            return Err(Error::TooDeep);
        }
        let depth = depth + 1;
        match step.op {
            ExprOp::Column | ExprOp::Constant | ExprOp::Null => {}
            ExprOp::When if step.a == PRESENT => step.b = self.copy(step.b, depth)?,
            ExprOp::ToDouble
            | ExprOp::Like
            | ExprOp::IsNull
            | ExprOp::IsNotNull
            | ExprOp::Not
            | ExprOp::In
            | ExprOp::Argument => step.a = self.copy(step.a, depth)?,
            ExprOp::Case | ExprOp::Call => failed!(),
            _ => {
                step.a = self.copy(step.a, depth)?;
                step.b = self.copy(step.b, depth)?;
            }
        }
        Ok(step)
    }

    /// Appends `step`, and returns where it is.
    fn push(&mut self, step: ExprStep) -> Result<u32, Error> {
        if self.out.len() == EXPR_NODES_MAX {
            return Err(Error::TooLarge);
        }
        let at = u32::try_from(self.out.len()).map_err(|_| Error::TooLarge)?;
        self.out.push(step)?;
        Ok(at)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec::Vec;

    use super::*;
    use crate::buffer::Buffer;
    use crate::column::ColumnView;
    use crate::context::Context;
    use crate::session::Session;
    use crate::table::Table;

    fn int64s(values: &[i64]) -> ColumnView {
        let env = Env::current();
        let mut buffer = Buffer::allocate(env, values.len() * 8).unwrap();
        buffer.as_mut_slice::<i64>().copy_from_slice(values);
        ColumnView::new(&mut Context::new(env, None), DataType::Int64, buffer, None).unwrap()
    }

    /// A table of `a`, from 0 to 9, and `b`, `a` modulo 3.
    fn numbers() -> DynScannable<'static> {
        let a: Vec<i64> = (0..10).collect();
        let b: Vec<i64> = a.iter().map(|a| a % 3).collect();
        let group = [int64s(&a), int64s(&b)];
        let schema = [("a", DataType::Int64), ("b", DataType::Int64)];
        let table = Table::new(Env::current(), &schema, &[&group]).unwrap();
        DynScannable::new(Env::current(), table).unwrap()
    }

    /// Each row of `query`, its columns' values in order.
    fn rows(query: Query<'_>) -> Vec<Vec<i64>> {
        let mut session = Session::new(Env::current());
        let prepared = session.prepare(query).unwrap();
        let positions: Vec<u32> = prepared.columns().iter().map(|c| c.position).collect();
        let mut rows = Vec::new();
        let mut out = prepared.run(&mut session).unwrap();
        while let Some(batch) = out.next().unwrap() {
            let selection = batch.selection();
            for row in (0..selection.len()).map(|i| selection.kept().row(i) as usize) {
                rows.push(positions.iter().map(|&p| batch.column(p).int64s()[row]).collect());
            }
        }
        rows
    }

    #[test]
    fn builds_queries_that_run() {
        let table = numbers();
        let q = Builder::new(Env::current()).unwrap();
        let scan = q.scan(&table, &[0, 1]).unwrap();
        let (a, b) = (q.column(&scan, 0).unwrap(), q.column(&scan, 1).unwrap());
        let (a_value, two) = (q.column_value(a).unwrap(), q.int(2).unwrap());
        let more = q.binary(Binary::Greater, a_value, two).unwrap();
        let kept = q.filter(scan, more).unwrap();
        let b_value = q.column_value(b).unwrap();
        let sum = q.binary(Binary::Add, a_value, b_value).unwrap();
        let extended = q.extend(kept, &[(sum, "c")]).unwrap();
        let c = q.column(&extended, 2).unwrap();
        let sorted = q.order_by(extended, &[(c, Order::Descending)]).unwrap();
        let first = q.limit(sorted, 3, 0).unwrap();
        let query = q.finish(first, &[(a, "a"), (c, "c")]).unwrap();
        assert_eq!(rows(query), [[8, 10], [9, 9], [7, 8]]);
    }

    #[test]
    fn joins_and_aggregates() {
        let table = numbers();
        let q = Builder::new(Env::current()).unwrap();
        let (left, right) = (q.scan(&table, &[0, 1]).unwrap(), q.scan(&table, &[0]).unwrap());
        let (b, a) = (q.column(&left, 1).unwrap(), q.column(&right, 0).unwrap());
        let on = q.binary(Binary::Equal, q.column_value(b).unwrap(), q.column_value(a).unwrap());
        let on = on.unwrap();
        let joined = q.join(JoinKind::Inner, left, right, Some(on)).unwrap();
        let count =
            Aggregation { function: Function::CountStar, input: None, distinct: false, name: "n" };
        let total =
            Aggregation { function: Function::Sum, input: Some(a), distinct: false, name: "s" };
        let grouped = q.aggregate(joined, &[b], &[count, total]).unwrap();
        let (counted, summed) = (q.column(&grouped, 1).unwrap(), q.column(&grouped, 2).unwrap());
        let sorted = q.order_by(grouped, &[(b, Order::Ascending)]).unwrap();
        let query = q.finish(sorted, &[(b, "b"), (counted, "n"), (summed, "s")]).unwrap();
        // Of `a`s 0 to 9, four have `b` 0, and three each 1 and 2.
        assert_eq!(rows(query), [[0, 4, 0], [1, 3, 3], [2, 3, 6]]);
    }

    #[test]
    fn checks_what_it_is_given() {
        let table = numbers();
        let (q, other) =
            (Builder::new(Env::current()).unwrap(), Builder::new(Env::current()).unwrap());
        let (left, right) = (q.scan(&table, &[0]).unwrap(), q.scan(&table, &[1]).unwrap());
        let b = q.column(&right, 0).unwrap();
        assert_eq!(other.column_value(b).err(), Some(Error::WrongBuilder));
        let b_value = q.column_value(b).unwrap();
        assert_eq!(q.filter(left, b_value).err(), Some(Error::NotInScope));
        let text = q.string(b"x").unwrap();
        assert_eq!(q.filter(right, text).err(), Some(Error::WrongType(TypeError::Condition)));
        assert_eq!(q.scan(&table, &[2]).err(), Some(Error::NotInScope));
    }
}
