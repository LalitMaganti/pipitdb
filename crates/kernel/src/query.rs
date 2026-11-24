//! Building queries.
//!
//! A [`Builder`] builds a [`Query`] from tables, a relation at a time: each
//! operation takes the relations it reads and returns a new one, whose
//! columns the next operations read. Every call takes `&self`, so calls
//! nest, as the parts of an expression do.
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
//!     let (slice, columns) = q.scan(slices, &[0, 1])?;
//!     let (name, dur) = (columns.get(0)?, columns.get(1)?);
//!     let long = q.binary(Binary::Greater, q.column_value(dur)?, q.int(1000)?)?;
//!     let slice = q.filter(slice, long)?;
//!     let slice = q.order_by(slice, &[(dur, Order::Descending)])?;
//!     let slice = q.limit(slice, 10, 0)?;
//!     q.finish(slice, &[(name, "name"), (dur, "dur")])
//! }
//! ```
//!
//! # Expressions
//!
//! An expression is built straight into the query, a node at a time, and
//! typed as each node is: a call that makes one of the wrong types fails
//! itself. An expression is every node built since an operation last took
//! one, so each is finished, and given to an operation, before the next is
//! begun.
//!
//! # Relations
//!
//! A relation is the input of one operation, as a query is a tree: a
//! [`Relation`] is moved into the operation that reads it.
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
    self, ComputeItem, ComputeOp, EXPR_NODES_MAX, ExprOp, ExprStep, PLAN_EXPR_VALUES_MAX, PRESENT,
    Typed, type_last,
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

/// The next builder's id.
static BUILDERS: AtomicU32 = AtomicU32::new(0);

/// No node: the parent of one no operation has read yet, or the maker of a
/// column whose operation isn't added yet.
const NONE: PlanNodeId = PlanNodeId::MAX;

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
    /// For each of the plan's nodes, the one that reads it, once one does.
    parents: SmallVec<PlanNodeId>,
    /// For each of the plan's columns, the node that makes it.
    made_by: SmallVec<PlanNodeId>,
    /// The aggregates' nodes, above which only their keys and aggregates
    /// are in scope.
    aggregates: SmallVec<PlanNodeId>,
    /// Each aggregate's node and a key of it.
    keys: SmallVec<(PlanNodeId, ColumnId)>,
    /// Where the expression being built starts in the plan's nodes.
    start: u32,
    /// For each node of the expression being built, its type and whether
    /// computing it can fail.
    typed: SmallVec<Typed>,
}

/// Rows an operation makes, which one other operation reads.
#[derive(Debug)]
pub struct Relation {
    /// The builder that made it.
    builder: u32,
    /// Its node in the plan.
    node: PlanNodeId,
}

/// A column of a relation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Column {
    /// The builder that made it.
    builder: u32,
    /// Which of the plan's columns it is.
    id: ColumnId,
}

/// Columns made together, as a scan's are.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Columns {
    /// The builder that made them.
    builder: u32,
    /// The first's id: the rest follow it.
    first: ColumnId,
    /// How many there are.
    count: u32,
}

impl Columns {
    /// Column `index`.
    ///
    /// # Errors
    ///
    /// [`Error::NotInScope`] if there's no such column.
    pub fn get(self, index: u32) -> Result<Column, Error> {
        if index >= self.count {
            return Err(Error::NotInScope);
        }
        Ok(Column { builder: self.builder, id: self.first + index })
    }

    /// How many there are.
    pub fn len(self) -> u32 {
        self.count
    }

    /// Whether there are none.
    pub fn is_empty(self) -> bool {
        self.count == 0
    }
}

/// A value computed for each row, from columns and constants.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Expr {
    /// The builder that made it.
    builder: u32,
    /// Its root among the plan's expression nodes.
    node: u32,
}

/// Columns being computed over a relation's rows, one at a time.
pub struct Extension {
    /// The rows they're computed over.
    input: Relation,
    /// The columns, and what computes each.
    items: SmallVec<ComputeItem>,
}

/// Rows being grouped, by keys and into aggregates, one at a time.
pub struct Grouping {
    /// The rows grouped.
    input: Relation,
    /// The columns rows are grouped by.
    keys: SmallVec<ColumnId>,
    /// What each group's rows make.
    aggregates: SmallVec<Aggregate>,
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

#[expect(clippy::needless_pass_by_value, reason = "a relation is moved, as one operation reads it")]
impl<'e> Builder<'e> {
    /// A builder of a query whose memory comes from `env`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn new(env: &'e Env) -> Result<Builder<'e>, Error> {
        let state = State {
            env,
            id: BUILDERS.fetch_add(1, Ordering::Relaxed),
            plan: LogicalPlan::new(env)?,
            parents: SmallVec::new(env, PLAN_NODES_MAX)?,
            made_by: SmallVec::new(env, PLAN_COLUMNS_MAX)?,
            aggregates: SmallVec::new(env, PLAN_NODES_MAX)?,
            keys: SmallVec::new(env, PLAN_COLUMNS_MAX)?,
            start: 0,
            typed: SmallVec::new(env, EXPR_NODES_MAX)?,
        };
        Ok(Builder { state: RefCell::new(state) })
    }

    /// The query whose rows are those of `relation`, and whose columns are
    /// `output`, each named.
    ///
    /// # Errors
    ///
    /// [`Error::NotInScope`] if a column isn't `relation`'s.
    pub fn finish(self, relation: Relation, output: &[(Column, &str)]) -> Result<Query<'e>, Error> {
        let mut state = self.state.into_inner();
        let node = state.relation(&relation)?;
        for &(column, name) in output {
            state.in_scope(node, column)?;
            let name = state.plan.names.add(name)?;
            state.plan.output.push(NamedColumn { name, id: column.id })?;
        }
        state.plan.root = node;
        Ok(Query { plan: state.plan })
    }

    // Relations.

    /// The rows of `table`, of its columns `columns`, in that order.
    ///
    /// # Errors
    ///
    /// [`Error::NotInScope`] if `table` has no such column, and
    /// [`Error::TooManyColumns`] if the query would have more than a plan
    /// can.
    pub fn scan(
        &self,
        table: &'e DynScannable<'e>,
        columns: &[u32],
    ) -> Result<(Relation, Columns), Error> {
        let count = table.column_count();
        if columns.iter().any(|&column| column >= count) {
            return Err(Error::NotInScope);
        }
        let mut guard = self.state();
        let state = &mut *guard;
        state.room_for(columns.len())?;
        let mut reads = SmallVec::fixed(state.env, columns.len())?;
        let (first, node) = (state.next_column(), state.next_node_id());
        for &column in columns {
            let (name, data_type) = (table.column_name(column), table.column_type(column));
            let binding = state.add_column(name, data_type, table.column_nullable(column), node)?;
            reads.push(ScanColumn { column, binding, forms: Forms::FLAT })?;
        }
        state.add_node(DynOp::new(state.env, ScanOp::new(table, reads))?, &[])?;
        let Ok(count) = u32::try_from(columns.len()) else { failed!() };
        Ok((state.relation_of(node), Columns { builder: state.id, first, count }))
    }

    /// The rows of `relation` that `condition`, an integer, is true for: not
    /// null or 0.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if `condition` isn't an integer,
    /// [`Error::NotInScope`] if it reads a column that isn't `relation`'s,
    /// and [`Error::OutOfOrder`] if it isn't the expression being built.
    pub fn filter(&self, relation: Relation, condition: Expr) -> Result<Relation, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let child = state.relation(&relation)?;
        let condition = state.condition(condition, &[child])?;
        let filter = FilterOp::new(state.env, &state.plan.exprs, condition)?;
        let node = state.add_node(DynOp::new(state.env, filter)?, &[child])?;
        Ok(state.relation_of(node))
    }

    /// Begins computing columns over `relation`'s rows, which
    /// [`Builder::compute`] adds and [`Builder::extended`] ends.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `relation` is another builder's.
    pub fn extend(&self, relation: Relation) -> Result<Extension, Error> {
        let state = self.state();
        state.relation(&relation)?;
        Ok(Extension { input: relation, items: SmallVec::new(state.env, PLAN_COLUMNS_MAX)? })
    }

    /// A column of `extension`, named `name`, which `expr` computes from its
    /// input's columns.
    ///
    /// # Errors
    ///
    /// [`Error::NotInScope`] if `expr` reads a column that isn't the input's,
    /// and [`Error::OutOfOrder`] if it isn't the expression being built.
    pub fn compute(
        &self,
        extension: &mut Extension,
        expr: Expr,
        name: &str,
    ) -> Result<Column, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let input = state.relation(&extension.input)?;
        state.room_for(1)?;
        let (expr, data_type) = state.take(expr, &[input])?;
        let columns = &state.plan.columns;
        let nullable = expression::expr_can_be_null(state.plan.exprs.nodes(expr), &|column| {
            at!(columns, column as usize).nullable
        });
        let output = state.add_column(name, data_type, nullable, NONE)?.id;
        extension.items.push(ComputeItem { output, expr })?;
        Ok(Column { builder: state.id, id: output })
    }

    /// The rows of `extension`'s input, with its columns: its input's, then
    /// those computed.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `extension` is another builder's.
    pub fn extended(&self, extension: Extension) -> Result<Relation, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let Extension { input, items } = extension;
        let child = state.relation(&input)?;
        if items.is_empty() {
            return Ok(input);
        }
        let node = state.next_node_id();
        for item in items.iter() {
            *at_mut!(state.made_by, item.output as usize) = node;
        }
        state.add_node(DynOp::new(state.env, ComputeOp { items })?, &[child])?;
        Ok(state.relation_of(node))
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
        let mut guard = self.state();
        let state = &mut *guard;
        let sides = [state.relation(&left)?, state.relation(&right)?];
        let conjuncts = match on {
            Some(on) => {
                let on = state.condition(on, &sides)?;
                conjuncts_of(state.env, &state.plan.exprs, on)?
            }
            None => SmallVec::fixed(state.env, 0)?,
        };
        let [left_columns, right_columns] = [state.scope(sides[0])?, state.scope(sides[1])?];
        if kind == JoinKind::Left {
            for &column in right_columns.iter() {
                at_mut!(state.plan.columns, column as usize).nullable = true;
            }
        }
        let join = JoinOp::new(state.env, kind, conjuncts, [left_columns, right_columns])?;
        let node = state.add_node(DynOp::new(state.env, join)?, &sides)?;
        Ok(state.relation_of(node))
    }

    /// Begins grouping `relation`'s rows, by the keys [`Builder::group_by`]
    /// adds, into the aggregates [`Builder::aggregate`] adds, which
    /// [`Builder::grouped`] ends. With no keys, all its rows are one group.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `relation` is another builder's.
    pub fn group(&self, relation: Relation) -> Result<Grouping, Error> {
        let state = self.state();
        state.relation(&relation)?;
        Ok(Grouping {
            input: relation,
            keys: SmallVec::new(state.env, PLAN_COLUMNS_MAX)?,
            aggregates: SmallVec::new(state.env, PLAN_COLUMNS_MAX)?,
        })
    }

    /// Groups `grouping`'s rows by `key` too, which is then a column of the
    /// rows it makes.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if `key` isn't an integer or a string, and
    /// [`Error::NotInScope`] if it isn't a column of the rows grouped.
    pub fn group_by(&self, grouping: &mut Grouping, key: Column) -> Result<(), Error> {
        let state = self.state();
        state.in_scope(state.relation(&grouping.input)?, key)?;
        if !matches!(state.type_of(key.id), DataType::Int64 | DataType::String) {
            return Err(Error::WrongType(TypeError::GroupKey));
        }
        Ok(grouping.keys.push(key.id)?)
    }

    /// An aggregate of each of `grouping`'s groups: `function` of `input`, or
    /// of no column for `Function::CountStar`, of each distinct value once if
    /// `distinct`, named `name`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if `function` doesn't read `input`'s type, or
    /// needs a column and has none, and [`Error::NotInScope`] if `input`
    /// isn't a column of the rows grouped.
    pub fn aggregate(
        &self,
        grouping: &mut Grouping,
        function: Function,
        input: Option<Column>,
        distinct: bool,
        name: &str,
    ) -> Result<Column, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let rows = state.relation(&grouping.input)?;
        if let Some(input) = input {
            state.in_scope(rows, input)?;
        }
        let input_type = input.map_or(DataType::Int64, |input| state.type_of(input.id));
        let counts_rows = function == Function::CountStar;
        if counts_rows != input.is_none() || input.is_some() && !function.reads(input_type) {
            return Err(Error::WrongType(TypeError::Aggregated));
        }
        state.room_for(1)?;
        let nullable = !matches!(function, Function::CountStar | Function::Count);
        let output_type = function.output_type(input_type);
        let binding = state.add_column(name, output_type, nullable, NONE)?;
        // The least or greatest of the distinct values is that of them all.
        let distinct = distinct && !matches!(function, Function::Min | Function::Max);
        let input = input.map(|input| input.id);
        grouping.aggregates.push(Aggregate { function, input, distinct, binding })?;
        Ok(Column { builder: state.id, id: binding.id })
    }

    /// A row for each of `grouping`'s groups: its keys, then its aggregates.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `grouping` is another builder's.
    pub fn grouped(&self, grouping: Grouping) -> Result<Relation, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let Grouping { input, keys, aggregates } = grouping;
        let child = state.relation(&input)?;
        let node = state.next_node_id();
        for aggregate in aggregates.iter() {
            *at_mut!(state.made_by, aggregate.binding.id as usize) = node;
        }
        for &key in keys.iter() {
            state.keys.push((node, key))?;
        }
        state.aggregates.push(node)?;
        let aggregate = AggregateOp { keys, aggregates, answer: None };
        state.add_node(DynOp::new(state.env, aggregate)?, &[child])?;
        Ok(state.relation_of(node))
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
        let mut guard = self.state();
        let state = &mut *guard;
        let child = state.relation(&relation)?;
        let mut sort_keys = SmallVec::fixed(state.env, keys.len())?;
        for &(column, order) in keys {
            state.in_scope(child, column)?;
            sort_keys
                .push(SortKey { column: column.id, descending: order == Order::Descending })?;
        }
        let node =
            state.add_node(DynOp::new(state.env, OrderByOp { keys: sort_keys })?, &[child])?;
        Ok(state.relation_of(node))
    }

    /// `count` of `relation`'s rows, after skipping `offset`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `relation` is another builder's.
    pub fn limit(&self, relation: Relation, count: u64, offset: u64) -> Result<Relation, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let child = state.relation(&relation)?;
        let node = state.add_node(DynOp::new(state.env, LimitOp { count, offset })?, &[child])?;
        Ok(state.relation_of(node))
    }

    /// The type of `column`'s values.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `column` is another builder's.
    pub fn column_type(&self, column: Column) -> Result<DataType, Error> {
        let state = self.state();
        state.mine(column.builder)?;
        Ok(state.type_of(column.id))
    }

    // Expressions.

    /// The type of `expr`'s values: `None` for a bare `NULL`, which takes the
    /// type of what it meets.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfOrder`] if it isn't part of the expression being built.
    pub fn expr_type(&self, expr: Expr) -> Result<Option<DataType>, Error> {
        let state = self.state();
        let node = state.operand(expr)?;
        Ok(at!(state.typed, node as usize).0)
    }

    /// The value of `column`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongBuilder`] if `column` is another builder's.
    pub fn column_value(&self, column: Column) -> Result<Expr, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        state.mine(column.builder)?;
        state.push(ExprStep::new(ExprOp::Column, column.id, 0, 0))
    }

    /// The integer `value`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the expression has as many nodes as it can.
    pub fn int(&self, value: i64) -> Result<Expr, Error> {
        self.state().push(ExprStep::new(ExprOp::Constant, 0, 0, value))
    }

    /// The double `value`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the expression has as many nodes as it can.
    pub fn double(&self, value: f64) -> Result<Expr, Error> {
        let mut step = ExprStep::new(ExprOp::Constant, 0, 0, 0);
        (step.data_type, step.double_value) = (DataType::Float64, value);
        self.state().push(step)
    }

    /// The string `value`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left for it.
    pub fn string(&self, value: &[u8]) -> Result<Expr, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let id = state.plan.add_literal(state.env, value)?;
        let mut step = ExprStep::new(ExprOp::Constant, id.raw_id(), 0, 0);
        step.data_type = DataType::String;
        state.push(step)
    }

    /// A null, of whatever type the expression around it needs.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the expression has as many nodes as it can.
    pub fn null(&self) -> Result<Expr, Error> {
        self.state().push(ExprStep::new(ExprOp::Null, 0, 0, 0))
    }

    /// `a op b`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if `a` and `b` aren't of types `op` takes.
    pub fn binary(&self, op: Binary, a: Expr, b: Expr) -> Result<Expr, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let (a, b) = (state.operand(a)?, state.operand(b)?);
        state.push(ExprStep::new(*at!(BINARY_OPS, op as usize), a, b, 0))
    }

    /// `op a`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if `a` isn't of a type `op` takes.
    pub fn unary(&self, op: Unary, a: Expr) -> Result<Expr, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let a = state.operand(a)?;
        state.push(ExprStep::new(*at!(UNARY_OPS, op as usize), a, 0, 0))
    }

    /// Whether the string `a` matches the `LIKE` pattern `pattern`: `%` for
    /// any run of bytes, and `_` for any one.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if `a` isn't a string.
    pub fn like(&self, a: Expr, pattern: &[u8]) -> Result<Expr, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let a = state.operand(a)?;
        let pattern = state.plan.add_literal(state.env, pattern)?;
        state.push(ExprStep::new(ExprOp::Like, a, pattern.raw_id(), 0))
    }

    /// Whether the integer `a` is one of `values`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the query would hold more values than a plan
    /// can.
    pub fn is_in(&self, a: Expr, values: &[i64]) -> Result<Expr, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let a = state.operand(a)?;
        let start = state.room_for_values(values.len())?;
        for &value in values {
            state.plan.exprs.add_value(value)?;
        }
        state.push_in(a, start, values.len(), DataType::Int64)
    }

    /// Whether the string `a` is one of `values`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the query would hold more values than a plan
    /// can.
    pub fn is_in_strings(&self, a: Expr, values: &[&[u8]]) -> Result<Expr, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let a = state.operand(a)?;
        let start = state.room_for_values(values.len())?;
        for value in values {
            let id = state.plan.add_literal(state.env, value)?;
            state.plan.exprs.add_value(i64::from(id.raw_id()))?;
        }
        state.push_in(a, start, values.len(), DataType::String)
    }

    /// For each row, the value of the first of `branches` whose condition
    /// isn't null or 0, else `otherwise`. A branch without a condition is
    /// taken where its value isn't null.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if a condition isn't an integer, or the values
    /// aren't of one type.
    pub fn case(&self, branches: &[(Option<Expr>, Expr)], otherwise: Expr) -> Result<Expr, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let otherwise = state.operand(otherwise)?;
        let Ok(count) = u32::try_from(branches.len()) else { return Err(Error::TooLarge) };
        let first = state.next_node();
        for &(condition, value) in branches {
            let condition = match condition {
                Some(condition) => state.operand(condition)?,
                None => PRESENT,
            };
            let value = state.operand(value)?;
            state.push(ExprStep::new(ExprOp::When, condition, value, 0))?;
        }
        let mut case = ExprStep::new(ExprOp::Case, first, count, 0);
        case.c = otherwise;
        state.push(case)
    }

    /// `function` called with `arguments`, each converted to its type, and
    /// giving a value of type `result`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if an argument isn't of a type its function
    /// takes, and [`Error::TooLarge`] if the query has as many functions as a
    /// plan can.
    pub fn call(
        &self,
        function: DynFunction,
        arguments: &[(Expr, DataType)],
        result: DataType,
    ) -> Result<Expr, Error> {
        let mut guard = self.state();
        let state = &mut *guard;
        let Ok(count) = u32::try_from(arguments.len()) else { return Err(Error::TooLarge) };
        for &(value, _) in arguments {
            state.operand(value)?;
        }
        let function = state.plan.add_function(state.env, function)?.ok_or(Error::TooLarge)?;
        let first = state.next_node();
        for &(value, data_type) in arguments {
            let mut argument = ExprStep::new(ExprOp::Argument, state.operand(value)?, 0, 0);
            argument.data_type = data_type;
            state.push(argument)?;
        }
        let mut call = ExprStep::new(ExprOp::Call, first, count, 0);
        (call.data_type, call.c) = (result, function);
        state.push(call)
    }

    /// Its state, which no call into it holds while another runs.
    fn state(&self) -> RefMut<'_, State<'e>> {
        let Ok(state) = self.state.try_borrow_mut() else { failed!() };
        state
    }
}

impl<'e> State<'e> {
    // Checks.

    /// Fails unless `builder` is this builder.
    fn mine(&self, builder: u32) -> Result<(), Error> {
        if builder == self.id { Ok(()) } else { Err(Error::WrongBuilder) }
    }

    /// `relation`'s node.
    fn relation(&self, relation: &Relation) -> Result<PlanNodeId, Error> {
        self.mine(relation.builder)?;
        Ok(relation.node)
    }

    /// Fails unless `column` is one of the columns of `node`'s rows.
    fn in_scope(&self, node: PlanNodeId, column: Column) -> Result<(), Error> {
        self.mine(column.builder)?;
        if self.sees(node, column.id) { Ok(()) } else { Err(Error::NotInScope) }
    }

    /// Whether `column` is one of the columns of `node`'s rows: made by it or
    /// below it, and not grouped away by an aggregate between.
    fn sees(&self, node: PlanNodeId, column: ColumnId) -> bool {
        let mut at = *at!(self.made_by, column as usize);
        while at != node {
            if at == NONE {
                return false;
            }
            at = *at!(self.parents, at as usize);
            // Above an aggregate are only its keys and aggregates, which it
            // makes, so a column from below must be a key.
            if self.aggregates.contains(&at) && !self.keys.contains(&(at, column)) {
                return false;
            }
        }
        true
    }

    /// The columns of `node`'s rows, in order.
    fn scope(&self, node: PlanNodeId) -> Result<SmallVec<ColumnId>, Error> {
        let mut columns = SmallVec::new(self.env, PLAN_COLUMNS_MAX)?;
        for column in 0..self.next_column() {
            if self.sees(node, column) {
                columns.push(column)?;
            }
        }
        Ok(columns)
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

    // Relations.

    /// The id the next column gets.
    fn next_column(&self) -> ColumnId {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_COLUMNS_MAX`")]
        let next = self.plan.columns.len() as ColumnId;
        next
    }

    /// The id the next node gets.
    fn next_node_id(&self) -> PlanNodeId {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
        let next = self.plan.nodes.len() as PlanNodeId;
        next
    }

    /// Adds a column made by `node`, which may be `NONE` until it's added.
    fn add_column(
        &mut self,
        name: &str,
        data_type: DataType,
        nullable: bool,
        node: PlanNodeId,
    ) -> Result<NamedColumn, Error> {
        self.made_by.push(node)?;
        Ok(self.plan.add_column(name, data_type, nullable)?)
    }

    /// Adds `op` as a node reading `children`, each then read by it.
    fn add_node(&mut self, op: DynOp<'e>, children: &[PlanNodeId]) -> Result<PlanNodeId, Error> {
        let node = self.plan.add_node(op, children)?;
        self.parents.push(NONE)?;
        for &child in children {
            *at_mut!(self.parents, child as usize) = node;
        }
        Ok(node)
    }

    fn relation_of(&self, node: PlanNodeId) -> Relation {
        Relation { builder: self.id, node }
    }

    // Expressions.

    /// Where the next expression node goes, from the expression's start.
    fn next_node(&self) -> u32 {
        self.plan.exprs.len() - self.start
    }

    /// Where `expr` is in the expression being built.
    fn operand(&self, expr: Expr) -> Result<u32, Error> {
        self.mine(expr.builder)?;
        if expr.node < self.start || expr.node >= self.plan.exprs.len() {
            return Err(Error::OutOfOrder);
        }
        Ok(expr.node - self.start)
    }

    /// Appends `step`, reading nodes counted from the expression's start,
    /// and types it.
    fn push(&mut self, step: ExprStep) -> Result<Expr, Error> {
        if self.typed.len() == EXPR_NODES_MAX {
            return Err(Error::TooLarge);
        }
        let node = self.plan.exprs.len();
        self.plan.exprs.push(step)?;
        self.typed.push((None, 0, false))?;
        let columns = &self.plan.columns;
        let type_of = |column: ColumnId| at!(columns, column as usize).data_type;
        let nodes = self.plan.exprs.nodes_from(self.start);
        if let Err(error) = type_last(self.env, &type_of, nodes, &mut self.typed) {
            self.plan.exprs.pop();
            self.typed.pop();
            return Err(error.into());
        }
        Ok(Expr { builder: self.id, node })
    }

    /// Where `count` more `IN` values go, if a plan has room for them.
    fn room_for_values(&self, count: usize) -> Result<u32, Error> {
        let len = self.plan.exprs.values().len();
        if len + count > PLAN_EXPR_VALUES_MAX {
            return Err(Error::TooLarge);
        }
        u32::try_from(len).map_err(|_| Error::TooLarge)
    }

    /// `a IN` the `count` values from `start`, of `kind`.
    fn push_in(&mut self, a: u32, start: u32, count: usize, kind: DataType) -> Result<Expr, Error> {
        let mut test = ExprStep::new(ExprOp::In, a, start, 0);
        let Ok(count) = u32::try_from(count) else { failed!() };
        (test.c, test.data_type) = (count, kind);
        self.push(test)
    }

    /// Takes `expr`, the expression being built, which must be an integer
    /// and read only the columns of `scope`'s nodes.
    fn condition(&mut self, expr: Expr, scope: &[PlanNodeId]) -> Result<expression::Expr, Error> {
        let (expr, data_type) = self.take(expr, scope)?;
        if data_type != DataType::Int64 {
            return Err(Error::WrongType(TypeError::Condition));
        }
        Ok(expr)
    }

    /// Takes `expr`, the expression being built, whose root it must be, and
    /// which must read only the columns of `scope`'s nodes. Returns it and
    /// its type, and begins the next.
    fn take(
        &mut self,
        expr: Expr,
        scope: &[PlanNodeId],
    ) -> Result<(expression::Expr, DataType), Error> {
        let root = self.operand(expr)?;
        if root + 1 != self.next_node() {
            return Err(Error::OutOfOrder);
        }
        let nodes = self.plan.exprs.nodes_from(self.start);
        for node in nodes {
            if node.op == ExprOp::Column && !scope.iter().any(|&at| self.sees(at, node.a)) {
                return Err(Error::NotInScope);
            }
        }
        // A result which is only ever NULL is a column of Int64 nulls.
        let data_type = at!(self.typed, root as usize).0.unwrap_or(DataType::Int64);
        let taken = expression::Expr { start: self.start, root };
        self.start = self.plan.exprs.len();
        self.typed.truncate(0);
        Ok((taken, data_type))
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
        let (scan, columns) = q.scan(&table, &[0, 1]).unwrap();
        let (a, b) = (columns.get(0).unwrap(), columns.get(1).unwrap());
        let more = q.binary(Binary::Greater, q.column_value(a).unwrap(), q.int(2).unwrap());
        let kept = q.filter(scan, more.unwrap()).unwrap();
        let mut extension = q.extend(kept).unwrap();
        let sum = q.binary(Binary::Add, q.column_value(a).unwrap(), q.column_value(b).unwrap());
        let c = q.compute(&mut extension, sum.unwrap(), "c").unwrap();
        let extended = q.extended(extension).unwrap();
        let sorted = q.order_by(extended, &[(c, Order::Descending)]).unwrap();
        let first = q.limit(sorted, 3, 0).unwrap();
        let query = q.finish(first, &[(a, "a"), (c, "c")]).unwrap();
        assert_eq!(rows(query), [[8, 10], [9, 9], [7, 8]]);
    }

    #[test]
    fn joins_and_aggregates() {
        let table = numbers();
        let q = Builder::new(Env::current()).unwrap();
        let ((left, lefts), (right, rights)) =
            (q.scan(&table, &[0, 1]).unwrap(), q.scan(&table, &[0]).unwrap());
        let (b, a) = (lefts.get(1).unwrap(), rights.get(0).unwrap());
        let on = q.binary(Binary::Equal, q.column_value(b).unwrap(), q.column_value(a).unwrap());
        let joined = q.join(JoinKind::Inner, left, right, Some(on.unwrap())).unwrap();
        let mut grouping = q.group(joined).unwrap();
        q.group_by(&mut grouping, b).unwrap();
        let count = q.aggregate(&mut grouping, Function::CountStar, None, false, "n").unwrap();
        let sum = q.aggregate(&mut grouping, Function::Sum, Some(a), false, "s").unwrap();
        let grouped = q.grouped(grouping).unwrap();
        let sorted = q.order_by(grouped, &[(b, Order::Ascending)]).unwrap();
        let query = q.finish(sorted, &[(b, "b"), (count, "n"), (sum, "s")]).unwrap();
        // Of `a`s 0 to 9, four have `b` 0, and three each 1 and 2.
        assert_eq!(rows(query), [[0, 4, 0], [1, 3, 3], [2, 3, 6]]);
    }

    #[test]
    fn checks_what_it_is_given() {
        let table = numbers();
        let q = Builder::new(Env::current()).unwrap();
        let other = Builder::new(Env::current()).unwrap();
        let ((left, _), (_, rights)) =
            (q.scan(&table, &[0]).unwrap(), q.scan(&table, &[1]).unwrap());
        let b = rights.get(0).unwrap();
        assert_eq!(other.column_value(b).err(), Some(Error::WrongBuilder));
        // Typed as it's built: a string can't be added to.
        let (text, one) = (q.string(b"x").unwrap(), q.int(1).unwrap());
        let added = q.binary(Binary::Add, text, one);
        assert_eq!(added.err(), Some(Error::WrongType(TypeError::Numbers)));
        assert_eq!(q.filter(left, text).err(), Some(Error::OutOfOrder));
        let b_value = q.column_value(b).unwrap();
        let (left, _) = q.scan(&table, &[0]).unwrap();
        assert_eq!(q.filter(left, b_value).err(), Some(Error::NotInScope));
        assert_eq!(q.scan(&table, &[2]).err(), Some(Error::NotInScope));
        // A column grouped away is out of scope above the aggregate.
        let (scan, columns) = q.scan(&table, &[0, 1]).unwrap();
        let mut grouping = q.group(scan).unwrap();
        q.group_by(&mut grouping, columns.get(1).unwrap()).unwrap();
        let grouped = q.grouped(grouping).unwrap();
        let sorted = q.order_by(grouped, &[(columns.get(0).unwrap(), Order::Ascending)]);
        assert_eq!(sorted.err(), Some(Error::NotInScope));
    }
}
