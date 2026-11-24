//! Building queries.
//!
//! A [`Builder`] builds a [`Query`] from tables, a relation at a time: each
//! operation takes the relation it reads and returns a new one, whose
//! columns the next operations read.
//!
//! An expression is built a node at a time, each a [`Value`] the next ones
//! can read, and is every node built since an operation last took one:
//! [`Builder::expr`] takes it, typed.

pub use crate::aggregate::Function;
use crate::aggregate::{Aggregate, AggregateOp};
use crate::column::{DataType, Forms};
use crate::containers::SmallVec;
use crate::env::Env;
use crate::error::{Error, TypeError};
use crate::expression::{
    ComputeItem, ComputeOp, Expr, ExprBuilder, ExprOp, ExprStep, PLAN_EXPR_VALUES_MAX, PRESENT,
    check_expr_type, expr_can_be_null,
};
use crate::function::DynFunction;
pub use crate::join::JoinKind;
use crate::join::JoinOp;
use crate::limit::LimitOp;
use crate::names::Name;
use crate::order_by::OrderByOp;
use crate::plan::{
    ColumnId, DynOp, FilterOp, LogicalPlan, PlanNodeId, ScanColumn, ScanOp, SortKey, conjuncts_of,
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
    /// The nodes of the expression being built.
    nodes: ExprBuilder,
}

/// A value of the expression being built: one of its nodes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Value(u32);

/// An operation of two values.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Binary {
    Add,
    Subtract,
    Multiply,
    Divide,
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
static BINARY: [ExprOp; 13] = [
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
    IsNull,
    IsNotNull,
}

/// The step computing each `Unary`, by its order.
static UNARY: [ExprOp; 3] = [ExprOp::Not, ExprOp::IsNull, ExprOp::IsNotNull];

/// A list of integers or of strings being built, which a value is tested
/// for being in.
pub struct InList {
    /// The value tested.
    value: Value,
    /// Where its values start among the plan's.
    start: u32,
    /// The type of its values, once it has one.
    data_type: Option<DataType>,
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

/// Two relations being joined, until their condition is built.
pub struct Joining {
    /// How rows are paired.
    kind: JoinKind,
    /// The probe side's node, then the build side's.
    children: [PlanNodeId; 2],
    /// Each side's columns.
    columns: [SmallVec<ColumnId>; 2],
}

/// Rows being extended, by computed columns added one at a time.
pub struct Extension {
    /// The rows extended.
    relation: Relation,
    /// The columns, each with what computes it.
    items: SmallVec<ComputeItem>,
}

/// Rows being grouped, by keys and into aggregates added one at a time.
pub struct Grouping {
    /// The keys, in turn.
    keys: SmallVec<ColumnId>,
    /// The aggregates, in turn.
    aggregates: SmallVec<Aggregate>,
}

impl Grouping {
    /// Groups by `column` too.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if there are more keys than the grouping was
    /// made for.
    pub fn by(&mut self, column: Column) -> Result<(), Error> {
        Ok(self.keys.push(column.0)?)
    }
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
        Ok(Builder { env, plan: LogicalPlan::new(env)?, nodes: ExprBuilder::new(env)? })
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

    /// A join of `probe` and `build`, as `kind` says, whose columns are
    /// `probe_columns` of `probe`, then `build_columns` of `build`, which a
    /// left join makes nullable. [`Builder::joined`] gives it its condition.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn join(
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
        let columns = [probe_columns, build_columns];
        Ok(Joining { kind, children: [probe.0, build.0], columns })
    }

    /// Each pair of `joining`'s rows that `condition` is true for, or every
    /// pair without one.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn joined(&mut self, joining: Joining, condition: Option<Expr>) -> Result<Relation, Error> {
        let env = self.env;
        let Joining { kind, children, columns } = joining;
        let conjuncts = match condition {
            Some(condition) => conjuncts_of(env, &self.plan.exprs, condition)?,
            None => SmallVec::fixed(env, 0)?,
        };
        let op = JoinOp::new(env, kind, conjuncts, columns)?;
        Ok(Relation(self.plan.add_node(DynOp::new(env, op)?, &children)?))
    }

    /// The rows of `relation`, to be extended by up to `len` columns, which
    /// [`Builder::compute`] adds.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn extend(&mut self, relation: Relation, len: usize) -> Result<Extension, Error> {
        Ok(Extension { relation, items: SmallVec::fixed(self.env, len)? })
    }

    /// A column of `extension`, named `name`, of `expr`, of type
    /// `data_type`, computed for each row.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn compute(
        &mut self,
        extension: &mut Extension,
        name: Name,
        expr: Expr,
        data_type: DataType,
    ) -> Result<Column, Error> {
        let (columns, nodes) = (&self.plan.columns, self.plan.exprs.nodes(expr));
        let nullable = expr_can_be_null(nodes, &|column| at!(columns, column as usize).nullable);
        let column = self.plan.add_named_column(name, data_type, nullable)?;
        extension.items.push(ComputeItem { output: column.id, expr })?;
        Ok(Column(column.id))
    }

    /// The rows of `extension`, with its columns.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn extended(&mut self, extension: Extension) -> Result<Relation, Error> {
        let Extension { relation, items } = extension;
        if items.is_empty() {
            return Ok(relation);
        }
        self.add(DynOp::new(self.env, ComputeOp { items })?, relation)
    }

    /// A grouping by up to `keys` keys, which [`Grouping::by`] adds, into up
    /// to `aggregates` aggregates, which [`Builder::aggregate`] adds.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn group(&mut self, keys: usize, aggregates: usize) -> Result<Grouping, Error> {
        let keys = SmallVec::fixed(self.env, keys)?;
        Ok(Grouping { keys, aggregates: SmallVec::fixed(self.env, aggregates)? })
    }

    /// A column of `grouping`, named `name`: `function` of each group's
    /// values of `input`, of each distinct one if `distinct`, or of its rows
    /// if there's no input.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn aggregate(
        &mut self,
        grouping: &mut Grouping,
        name: Name,
        function: Function,
        input: Option<Column>,
        distinct: bool,
    ) -> Result<Column, Error> {
        let input = input.map(|column| column.0);
        let input_type =
            input.map_or(DataType::Int64, |id| at!(self.plan.columns, id as usize).data_type);
        // A count is 0 for a group of nulls, where the others are null.
        let nullable = !matches!(function, Function::CountStar | Function::Count);
        let output_type = function.output_type(input_type);
        let binding = self.plan.add_named_column(name, output_type, nullable)?;
        // The least or greatest of the distinct values is that of them all.
        let distinct = distinct && !matches!(function, Function::Min | Function::Max);
        grouping.aggregates.push(Aggregate { function, input, distinct, binding })?;
        Ok(Column(binding.id))
    }

    /// The rows of `relation`, grouped by `grouping`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn grouped(&mut self, relation: Relation, grouping: Grouping) -> Result<Relation, Error> {
        let Grouping { keys, aggregates } = grouping;
        let aggregate = AggregateOp { keys, aggregates, answer: None };
        self.add(DynOp::new(self.env, aggregate)?, relation)
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

    // Expressions.

    /// Takes the expression being built, whose root is the last node built,
    /// and the type it computes.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if its values aren't of the types they're used
    /// as.
    pub fn expr(&mut self) -> Result<(Expr, DataType), Error> {
        let columns = &self.plan.columns;
        let type_of = |column: ColumnId| at!(columns, column as usize).data_type;
        let typed = check_expr_type(self.env, self.nodes.nodes(), &type_of);
        let expr = self.plan.exprs.add(self.nodes.nodes());
        self.nodes.clear();
        let data_type = typed?;
        Ok((expr?, data_type))
    }

    /// The value of `column` in each row.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the expression has the most nodes it can.
    pub fn column_value(&mut self, column: Column) -> Result<Value, Error> {
        self.push(ExprStep::new(ExprOp::Column, column.0, 0, 0))
    }

    /// The integer `value`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the expression has the most nodes it can.
    pub fn int(&mut self, value: i64) -> Result<Value, Error> {
        self.push(ExprStep::new(ExprOp::Constant, 0, 0, value))
    }

    /// The string `value`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the expression has the most nodes it can.
    pub fn string(&mut self, value: &[u8]) -> Result<Value, Error> {
        let id = self.plan.add_literal(self.env, value)?;
        let mut constant = ExprStep::new(ExprOp::Constant, id.raw_id(), 0, 0);
        constant.data_type = DataType::String;
        self.push(constant)
    }

    /// A null, of whatever type it's used as.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the expression has the most nodes it can.
    pub fn null(&mut self) -> Result<Value, Error> {
        self.push(ExprStep::new(ExprOp::Null, 0, 0, 0))
    }

    /// `a op b`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the expression has the most nodes it can.
    pub fn binary(&mut self, op: Binary, a: Value, b: Value) -> Result<Value, Error> {
        self.push(ExprStep::new(*at!(BINARY, op as usize), a.0, b.0, 0))
    }

    /// `op a`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the expression has the most nodes it can.
    pub fn unary(&mut self, op: Unary, a: Value) -> Result<Value, Error> {
        self.push(ExprStep::new(*at!(UNARY, op as usize), a.0, 0, 0))
    }

    /// Whether `a` matches `pattern`, in which `%` is any bytes and `_` any
    /// one.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the expression has the most nodes it can.
    pub fn like(&mut self, a: Value, pattern: &[u8]) -> Result<Value, Error> {
        let pattern = self.plan.add_literal(self.env, pattern)?;
        self.push(ExprStep::new(ExprOp::Like, a.0, pattern.raw_id(), 0))
    }

    /// A test of whether `value` is in a list, which [`Builder::list_int`]
    /// and [`Builder::list_string`] add to, and [`Builder::listed`] ends.
    #[must_use]
    pub fn in_list(&self, value: Value) -> InList {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_EXPR_VALUES_MAX`")]
        let start = self.plan.exprs.values().len() as u32;
        InList { value, start, data_type: None }
    }

    /// Adds `value` to `list`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if the list has strings, and [`Error::TooLarge`]
    /// if the plan has the most values it can.
    pub fn list_int(&mut self, list: &mut InList, value: i64) -> Result<(), Error> {
        self.list(list, value, DataType::Int64)
    }

    /// Adds `value` to `list`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if the list has integers, and
    /// [`Error::TooLarge`] if the plan has the most values it can.
    pub fn list_string(&mut self, list: &mut InList, value: &[u8]) -> Result<(), Error> {
        let id = self.plan.add_literal(self.env, value)?;
        self.list(list, i64::from(id.raw_id()), DataType::String)
    }

    /// Adds `value`, of type `data_type`, to `list`.
    fn list(&mut self, list: &mut InList, value: i64, data_type: DataType) -> Result<(), Error> {
        if list.data_type.is_some_and(|listed| listed != data_type) {
            return Err(Error::WrongType(TypeError::Kinds));
        }
        if self.plan.exprs.values().len() == PLAN_EXPR_VALUES_MAX {
            return Err(Error::TooLarge);
        }
        list.data_type = Some(data_type);
        self.plan.exprs.add_value(value)?;
        Ok(())
    }

    /// Whether `list`'s value is in it.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the expression has the most nodes it can.
    #[expect(clippy::needless_pass_by_value, reason = "a list is ended once")]
    pub fn listed(&mut self, list: InList) -> Result<Value, Error> {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_EXPR_VALUES_MAX`")]
        let len = self.plan.exprs.values().len() as u32 - list.start;
        let mut test = ExprStep::new(ExprOp::In, list.value.0, list.start, 0);
        (test.c, test.data_type) = (len, list.data_type.unwrap_or(DataType::Int64));
        self.push(test)
    }

    /// The value of the first of `branches` whose condition is true, or
    /// without one, whose value isn't null; or else `otherwise`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the expression has the most nodes it can.
    pub fn case(
        &mut self,
        branches: &[(Option<Value>, Value)],
        otherwise: Value,
    ) -> Result<Value, Error> {
        let mut first = None;
        for &(condition, value) in branches {
            let condition = condition.map_or(PRESENT, |condition| condition.0);
            let when = self.push(ExprStep::new(ExprOp::When, condition, value.0, 0))?;
            first.get_or_insert(when.0);
        }
        #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
        let mut case = ExprStep::new(ExprOp::Case, first.unwrap_or(0), branches.len() as u32, 0);
        case.c = otherwise.0;
        self.push(case)
    }

    /// A call of `function` with `arguments`, each converted to its type,
    /// giving a value of type `result`.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the expression has the most nodes it can, or
    /// the plan the most functions.
    pub fn call(
        &mut self,
        function: DynFunction,
        arguments: &[(Value, DataType)],
        result: DataType,
    ) -> Result<Value, Error> {
        let Some(id) = self.plan.add_function(self.env, function)? else {
            return Err(Error::TooLarge);
        };
        let mut first = None;
        for &(value, data_type) in arguments {
            let mut argument = ExprStep::new(ExprOp::Argument, value.0, 0, 0);
            argument.data_type = data_type;
            first.get_or_insert(self.push(argument)?.0);
        }
        #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
        let mut call = ExprStep::new(ExprOp::Call, first.unwrap_or(0), arguments.len() as u32, 0);
        call.data_type = result;
        call.c = id;
        self.push(call)
    }

    /// Adds `step` to the expression being built.
    #[inline(never)]
    fn push(&mut self, step: ExprStep) -> Result<Value, Error> {
        if self.nodes.is_full() {
            return Err(Error::TooLarge);
        }
        Ok(Value(self.nodes.push(step)?))
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
