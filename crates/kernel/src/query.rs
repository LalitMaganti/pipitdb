//! Building queries.
//!
//! A [`Builder`] builds a [`Query`] from tables, a relation at a time: each
//! operation takes the relation it reads and returns a new one, whose
//! columns the next operations read.
//!
//! An expression is built a node at a time, each a [`Value`] the next ones
//! can read, and is every node built since an operation last took one:
//! [`Builder::expr`] takes it, typed.

use crate::column::DataType;
use crate::containers::{BitSlice, Bitmap, SmallVec};
use crate::env::Env;
use crate::error::{Error, TypeError};
pub use crate::expression::EXPR_NODES_MAX;
use crate::expression::{
    Expr as PlanExpr, ExprBuilder, ExprOp, ExprStep, PLAN_EXPR_VALUES_MAX, PRESENT, check_expr_type,
};
use crate::function::DynFunction;
use crate::names::{Name, Names};
use crate::plan::{CHILDREN_MAX, ColumnId, DynOp, LogicalPlan, PlanNodeId, Scope, has};
pub use crate::plan::{PLAN_COLUMNS_MAX, PLAN_NODES_MAX};

/// What a front end asks to compute: a query, built but not yet prepared.
pub struct Query<'e> {
    pub(crate) plan: LogicalPlan<'e>,
}

/// Builds a query, a relation at a time.
pub struct Builder<'e> {
    /// Where the query's memory comes from.
    pub(crate) env: &'e Env,
    /// The plan built so far.
    pub(crate) plan: LogicalPlan<'e>,
    /// The nodes of the expression being built.
    pub(crate) nodes: ExprBuilder,
    /// What checks each op as it's added, if it does: only a builder made
    /// to check refers to it, so one that isn't doesn't pay for it.
    check: Option<Check<'e>>,
    /// If it checks ops, each node's columns, a bit for each of the plan's
    /// columns when it was added.
    scopes: SmallVec<Bitmap>,
}

/// Checks `op`, to be added reading the nodes passed, reads only columns
/// they have; or with no op, that the query's output is the node's.
type Check<'e> = fn(&mut Builder<'e>, Option<&DynOp<'e>>, &[PlanNodeId]) -> Result<(), Error>;

/// The check of a builder that checks ops as they're added.
fn check_scope<'e>(
    builder: &mut Builder<'e>,
    op: Option<&DynOp<'e>>,
    children: &[PlanNodeId],
) -> Result<(), Error> {
    let mut inputs: [&BitSlice; CHILDREN_MAX] = [BitSlice::of(&[]); CHILDREN_MAX];
    for (input, &child) in inputs.iter_mut().zip(children) {
        *input = at!(builder.scopes, child as usize);
    }
    let inputs = at!(inputs, ..children.len());
    let Some(op) = op else {
        let root = at!(inputs, 0);
        let all = builder.plan.output.iter().all(|named| has(root, named.column.0));
        return if all { Ok(()) } else { Err(Error::NotInScope) };
    };
    let mut out = Bitmap::new(builder.env, builder.plan.columns.len())?;
    op.scope(&mut Scope::new(&builder.plan.exprs, inputs, &mut out))?;
    Ok(builder.scopes.push(out)?)
}

/// An expression, built, and the type of value it computes, which one
/// operation takes.
pub struct Expr {
    /// Where its nodes start among the plan's.
    pub(crate) start: u32,
    /// Which of its nodes is its root: at most `EXPR_NODES_MAX`.
    pub(crate) root: u16,
    /// The type of value it computes.
    pub(crate) data_type: DataType,
}

impl Expr {
    /// It, as the plan holds it.
    pub(crate) fn plan_expr(&self) -> PlanExpr {
        PlanExpr { start: self.start, root: u32::from(self.root) }
    }
}

impl Expr {
    /// The type of value it computes.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
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
pub struct Relation(pub(crate) PlanNodeId);

/// A column of a relation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Column(pub(crate) ColumnId);

/// A column, and what it's called: in a front end's scope, or in a query's
/// output.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Named {
    pub name: Name,
    pub column: Column,
}

/// Columns made together, as a scan's are.
#[derive(Clone, Copy, Debug)]
pub struct Columns {
    pub(crate) first: ColumnId,
    pub(crate) count: u32,
}

impl Columns {
    /// Each of them, in order.
    pub fn iter(self) -> impl Iterator<Item = Column> {
        (self.first..self.first + self.count).map(Column)
    }
}

impl<'e> Builder<'e> {
    /// A builder of a query whose memory comes from `env`, which checks each
    /// operation reads only columns in scope as it's added.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn new(env: &'e Env) -> Result<Builder<'e>, Error> {
        let mut builder = Builder::unchecked(env)?;
        builder.check = Some(check_scope);
        Ok(builder)
    }

    /// A builder of a query whose memory comes from `env`, which leaves
    /// checking operations read only columns in scope to when the query is
    /// prepared: for a front end that keeps its scope itself.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    pub fn unchecked(env: &'e Env) -> Result<Builder<'e>, Error> {
        let (plan, nodes) = (LogicalPlan::new(env)?, ExprBuilder::new(env)?);
        let scopes = SmallVec::new(env, PLAN_NODES_MAX)?;
        Ok(Builder { env, plan, nodes, check: None, scopes })
    }

    /// The query whose rows are those of `relation`, and whose columns are
    /// `output`, each named.
    ///
    /// # Errors
    ///
    /// [`Error::NotInScope`] if the builder checks scopes and an output
    /// column isn't `relation`'s.
    #[expect(clippy::needless_pass_by_value, reason = "a relation is read once")]
    pub fn finish(
        mut self,
        relation: Relation,
        output: SmallVec<Named>,
    ) -> Result<Query<'e>, Error> {
        self.plan.output = output;
        if let Some(check) = self.check {
            check(&mut self, None, &[relation.0])?;
        }
        self.plan.root = relation.0;
        Ok(Query { plan: self.plan })
    }

    /// The query's names, which its output's are among, and a front end's,
    /// such as of the columns in its scope.
    #[must_use]
    pub fn names(&self) -> &Names {
        &self.plan.names
    }

    /// As [`Builder::names`], to add names to.
    pub fn names_mut(&mut self) -> &mut Names {
        &mut self.plan.names
    }

    /// The type of `column`'s values.
    #[must_use]
    pub fn column_type(&self, column: Column) -> DataType {
        at!(self.plan.columns, column.0 as usize).data_type
    }

    // Expressions.

    /// Takes the expression being built, whose root is the last node built,
    /// typed.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if its values aren't of the types they're used
    /// as.
    pub fn expr(&mut self) -> Result<Expr, Error> {
        let columns = &self.plan.columns;
        let type_of = |column: ColumnId| at!(columns, column as usize).data_type;
        let typed = check_expr_type(self.env, self.nodes.nodes(), &type_of);
        let expr = self.plan.exprs.add(self.nodes.nodes());
        self.nodes.clear();
        let (data_type, expr) = (typed?, expr?);
        #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
        let root = expr.root as u16;
        Ok(Expr { start: expr.start, root, data_type })
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

    /// Adds `op`, reading `inputs`, and returns the rows it makes.
    pub(crate) fn add<const N: usize>(
        &mut self,
        op: DynOp<'e>,
        inputs: [Relation; N],
    ) -> Result<Relation, Error> {
        self.add_node(op, &inputs.map(|input| input.0))
    }

    /// As `add`, of the nodes `children`.
    fn add_node(&mut self, op: DynOp<'e>, children: &[PlanNodeId]) -> Result<Relation, Error> {
        if let Some(check) = self.check {
            check(self, Some(&op), children)?;
        }
        Ok(Relation(self.plan.add_node(op, children)?))
    }

    /// A column an op makes, of `data_type`, nullable if `nullable`.
    pub(crate) fn add_column(
        &mut self,
        data_type: DataType,
        nullable: bool,
    ) -> Result<Column, Error> {
        Ok(Column(self.plan.add_unnamed_column(data_type, nullable)?))
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
    use crate::ops::{Aggregate, Compute, Function, Order, Scan, Sort};
    use crate::scannable::DynScannable;
    use crate::table::Table;

    fn int64s(values: &[i64]) -> ColumnView {
        let mut buffer = Buffer::allocate(Env::current(), values.len() * 8).unwrap();
        buffer.as_mut_slice::<i64>().copy_from_slice(values);
        ColumnView::new(&mut Context::new(Env::current(), None), DataType::Int64, buffer, None)
            .unwrap()
    }

    #[test]
    fn works_out_which_columns_can_be_null() {
        let env = Env::current();
        let columns = [("a", DataType::Int64), ("b", DataType::Int64)];
        let table = Table::new(env, &columns, &[&[int64s(&[1, 2]), int64s(&[10, 20])]]).unwrap();
        let table = DynScannable::new(env, table).unwrap();
        let mut builder = Builder::new(env).unwrap();
        let (rows, columns) = builder.scan(&table).unwrap();
        let [a, b] = columns.iter().collect::<Vec<_>>()[..] else { panic!() };
        let mut extension = builder.extend(rows, 2).unwrap();
        // Dividing by a constant that isn't zero can't make a null, but
        // dividing by a column may.
        let (value, sixty) = (builder.column_value(a).unwrap(), builder.int(60).unwrap());
        builder.binary(Binary::Divide, value, sixty).unwrap();
        let expr = builder.expr().unwrap();
        let x = builder.compute(&mut extension, expr).unwrap();
        let (value, by) = (builder.column_value(a).unwrap(), builder.column_value(b).unwrap());
        builder.binary(Binary::Divide, value, by).unwrap();
        let expr = builder.expr().unwrap();
        let y = builder.compute(&mut extension, expr).unwrap();
        let rows = builder.extended(extension).unwrap();
        // A count is never null, but a sum of no values is.
        let mut grouping = builder.group(1, 2).unwrap();
        builder.group_by(&mut grouping, x).unwrap();
        let count = builder.aggregate(&mut grouping, Function::CountStar, None, false).unwrap();
        let total = builder.aggregate(&mut grouping, Function::Sum, Some(b), false).unwrap();
        builder.grouped(rows, grouping).unwrap();
        let nullable = |column: Column| builder.plan.columns[column.0 as usize].nullable;
        assert!(!nullable(a) && !nullable(x) && !nullable(count));
        assert!(nullable(y) && nullable(total));
    }

    /// Sorts the rows of table `a, b` grouped by `a` by `b`, which grouping
    /// leaves out of scope, then outputs `a`, with `builder`.
    fn sorted_out_of_scope<'e>(
        builder: &mut Builder<'e>,
        table: &'e DynScannable<'e>,
    ) -> Result<(Relation, SmallVec<Named>), Error> {
        let (rows, columns) = builder.scan(table)?;
        let [a, b] = columns.iter().collect::<Vec<_>>()[..] else { panic!() };
        let mut grouping = builder.group(1, 0)?;
        builder.group_by(&mut grouping, a)?;
        let rows = builder.grouped(rows, grouping)?;
        let mut sorting = builder.sort(rows, 1)?;
        sorting.by(b, Order::Ascending)?;
        let rows = builder.sorted(sorting)?;
        let name = builder.names_mut().add("a")?;
        Ok((rows, SmallVec::fixed_from(Env::current(), [Named { name, column: a }].into_iter())?))
    }

    #[test]
    fn reads_only_columns_in_scope() {
        let env = Env::current();
        let columns = [("a", DataType::Int64), ("b", DataType::Int64)];
        let table = Table::new(env, &columns, &[&[int64s(&[1, 2]), int64s(&[10, 20])]]).unwrap();
        let table = DynScannable::new(env, table).unwrap();
        // Checked as it's built, the sort fails.
        let mut checked = Builder::new(env).unwrap();
        assert_eq!(sorted_out_of_scope(&mut checked, &table).err(), Some(Error::NotInScope));
        // Unchecked, preparing it fails.
        let mut unchecked = Builder::unchecked(env).unwrap();
        let (rows, output) = sorted_out_of_scope(&mut unchecked, &table).unwrap();
        let query = unchecked.finish(rows, output).unwrap();
        let session = crate::session::Session::new(env);
        assert_eq!(session.prepare(query).err(), Some(Error::NotInScope));
    }
}
