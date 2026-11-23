//! Expressions over the columns in scope, compiled to the kernel's `Expr`,
//! as Perfetto's `CompileExpr`.

use pipit_kernel::column::DataType;
use pipit_kernel::containers::SmallVec;
use pipit_kernel::expression::{
    Expr, ExprBuilder, ExprError, ExprOp, ExprStep, PLAN_EXPR_VALUES_MAX, check_expr_type,
};
use pipit_kernel::function::DynFunction;
use pipit_kernel::string_pool::Id;

use crate::ast::{Node, Operator, Tag};
use crate::compile::Compiler;
use crate::error::{Error, ErrorCode, Unsupported};
use crate::keywords::same_word;
use crate::parser::NESTING_MAX;
use crate::stages::aggregate;

/// Compiles the expression `node` against the columns in scope, and the type
/// it computes.
pub fn compile_expr(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
) -> Result<(Expr, DataType), Error> {
    let mut nodes = ExprBuilder::new(compiler.env())?;
    compile(compiler, node, 0, &mut nodes)?;
    add_expr(compiler, &nodes, node)
}

/// Adds the expression of `nodes`, compiled from `node`, to the plan's, and
/// the type it computes.
pub fn add_expr(
    compiler: &mut Compiler<'_, '_>,
    nodes: &ExprBuilder,
    node: Node,
) -> Result<(Expr, DataType), Error> {
    let type_of = |column| at!(compiler.plan.columns, column as usize).data_type;
    let data_type = check_expr_type(compiler.env(), nodes.nodes(), &type_of).map_err(|error| {
        let span = compiler.span(node);
        let code = match error {
            ExprError::Numbers => ErrorCode::ExpectedNumber,
            ExprError::Integers => ErrorCode::ExpectedInteger,
            ExprError::Kinds => ErrorCode::ExpectedOneKind,
            ExprError::Types => ErrorCode::ExpectedOneType,
            ExprError::Inexact => return Error::unsupported(Unsupported::NumberType, span),
            ExprError::DoubleIn => return Error::unsupported(Unsupported::SelectExpression, span),
            ExprError::Strings => ErrorCode::ExpectedString,
            ExprError::SomeRows => ErrorCode::ArithmeticOnSomeRows,
            ExprError::Condition => ErrorCode::NumberCondition,
            ExprError::Choices => ErrorCode::ChoicesOfOneType,
            ExprError::Arguments => ErrorCode::ArgumentType,
            ExprError::OutOfMemory => ErrorCode::OutOfMemory,
        };
        Error::new(code, span)
    })?;
    let expr = compiler.plan.exprs.add(nodes.nodes())?;
    Ok((expr, data_type))
}

/// Whether `node` is the literal `NULL`.
pub fn is_null(compiler: &Compiler<'_, '_>, node: Node) -> bool {
    node.tag() == Tag::Name && same_word(compiler.text(node.span()).as_bytes(), b"null")
}

/// Appends `node`, and returns its index.
fn push_node(
    nodes: &mut ExprBuilder,
    node: ExprStep,
    span: Node,
    compiler: &Compiler<'_, '_>,
) -> Result<u32, Error> {
    if nodes.is_full() {
        return Err(Error::new(ErrorCode::ExpressionTooLarge, compiler.span(span)));
    }
    Ok(nodes.push(node)?)
}

/// `a IN (1, 2)` or `a IN ('x', 'y')`: a list of integers, or of strings,
/// whose values, or the strings' ids, go in the plan's expressions' values.
/// The node's type says which.
fn in_list(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
    nodes: &mut ExprBuilder,
) -> Result<(), Error> {
    compile(compiler, compiler.node(node.first_child()), depth + 1, nodes)?;
    let operand = nodes.last();
    let list = compiler.node(node.first_child() + 1);
    let len = list.child_count();
    if compiler.plan.exprs.values().len() + len as usize > PLAN_EXPR_VALUES_MAX {
        return Err(Error::new(ErrorCode::ConditionTooLarge, compiler.span(list)));
    }
    #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_EXPR_VALUES_MAX`")]
    let start = compiler.plan.exprs.values().len() as u32;
    let mut kind = None;
    for i in 0..len {
        // A literal, compiled as any is, then taken back off as a value.
        let item = compiler.node(list.first_child() + i);
        compile(compiler, item, depth + 1, nodes)?;
        let Some(constant) = nodes.pop() else { pipit_kernel::failed!() };
        let (listed, item_kind) = match (constant.op, constant.data_type) {
            (ExprOp::Constant, DataType::Int64) => (constant.int_value, DataType::Int64),
            (ExprOp::Constant, DataType::String) => (i64::from(constant.a), DataType::String),
            _ => return Err(Error::unsupported(Unsupported::Where, compiler.span(item))),
        };
        if kind.is_some_and(|kind| kind != item_kind) {
            return Err(Error::new(ErrorCode::ExpectedOneKind, compiler.span(item)));
        }
        kind = Some(item_kind);
        compiler.plan.exprs.add_value(listed)?;
    }
    let mut test = ExprStep::new(ExprOp::In, operand, start, 0);
    (test.c, test.data_type) = (len, kind.unwrap_or(DataType::Int64));
    push_node(nodes, test, node, compiler)?;
    Ok(())
}

/// Appends the nodes of the expression `node`, `depth` deep, its result
/// last.
fn compile(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
    nodes: &mut ExprBuilder,
) -> Result<(), Error> {
    // Each level recurses, so as deep as the parser lets a query nest: a
    // chain of operators is as deep as it is long, but nests no deeper.
    if depth == NESTING_MAX {
        return Err(Error::new(ErrorCode::NestingTooDeep, compiler.span(node)));
    }
    let child = |compiler: &Compiler<'_, '_>, i: u32| compiler.node(node.first_child() + i);
    let step = match node.tag() {
        Tag::Name if is_null(compiler, node) => ExprStep::new(ExprOp::Null, 0, 0, 0),
        Tag::Name => ExprStep::new(ExprOp::Column, compiler.find_column(node)?.id, 0, 0),
        Tag::Integer => ExprStep::new(ExprOp::Constant, 0, 0, integer(compiler, node, false)?),
        Tag::Float => {
            return Err(Error::unsupported(Unsupported::Decimal, node.span()));
        }
        Tag::String => {
            let id = string(compiler, node)?;
            let mut constant = ExprStep::new(ExprOp::Constant, id.raw_id(), 0, 0);
            constant.data_type = DataType::String;
            constant
        }
        Tag::Unary if node.operator() == Operator::Negate => {
            return negate(compiler, node, depth, nodes);
        }
        Tag::Unary => unary(compiler, node, depth, nodes)?,
        Tag::Binary if node.operator() == Operator::Like => like(compiler, node, depth, nodes)?,
        Tag::Binary if node.operator() == Operator::In => {
            return in_list(compiler, node, depth, nodes);
        }
        Tag::Call => return (compiler.registry().call)(compiler, node, depth, nodes),
        Tag::Case => return case(compiler, node, depth, nodes),
        Tag::Binary => {
            let op = match node.operator() {
                Operator::Add => ExprOp::Add,
                Operator::Subtract => ExprOp::Sub,
                Operator::Multiply => ExprOp::Mul,
                Operator::Divide => ExprOp::Div,
                Operator::Modulo => ExprOp::Mod,
                Operator::Equal => ExprOp::Eq,
                Operator::NotEqual => ExprOp::Ne,
                Operator::Less => ExprOp::Lt,
                Operator::LessEqual => ExprOp::Le,
                Operator::Greater => ExprOp::Gt,
                Operator::GreaterEqual => ExprOp::Ge,
                Operator::And => ExprOp::And,
                Operator::Or => ExprOp::Or,
                _ => {
                    return Err(Error::unsupported(
                        Unsupported::SelectExpression,
                        compiler.span(node),
                    ));
                }
            };
            compile(compiler, child(compiler, 0), depth + 1, nodes)?;
            let a = nodes.last();
            compile(compiler, child(compiler, 1), depth + 1, nodes)?;
            ExprStep::new(op, a, nodes.last(), 0)
        }
        _ => return Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(node))),
    };
    push_node(nodes, step, node, compiler)?;
    Ok(())
}

/// How a registry compiles a call of a function it may have: as
/// `call_function` does, or as `no_function` does when it has none.
pub(crate) type CallFunction =
    fn(&mut Compiler<'_, '_>, Node, usize, &mut ExprBuilder) -> Result<(), Error>;

/// Fails the call `node`, as a registry without functions can't compile it.
pub(crate) fn no_function(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    _: usize,
    _: &mut ExprBuilder,
) -> Result<(), Error> {
    Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(node)))
}

/// Appends the nodes of the call `node`, `depth` deep, as the registry's
/// rule for its function compiles them.
pub(crate) fn call_function(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
    nodes: &mut ExprBuilder,
) -> Result<(), Error> {
    let name = compiler.node(node.first_child()).span();
    let text = compiler.text(name).as_bytes();
    let Some(rule) = compiler.registry().function(text) else {
        // An aggregate can't be called here.
        if aggregate(text).is_some() {
            return Err(Error::unsupported(Unsupported::SelectExpression, name));
        }
        return Err(Error::new(ErrorCode::UnknownFunction, name));
    };
    (rule.compile)(compiler, node, depth, nodes)
}

/// Appends the nodes of each argument of the call `node`, `depth` deep, if
/// it has from `fewest` to `most`, and returns each one's result: the last
/// of the nodes computing it.
pub fn compile_arguments(
    compiler: &mut Compiler<'_, '_>,
    (node, depth): (Node, usize),
    nodes: &mut ExprBuilder,
    fewest: u32,
    most: u32,
) -> Result<SmallVec<u32>, Error> {
    let count = node.child_count() - 1;
    if count < fewest || count > most {
        return Err(Error::new(ErrorCode::ArgumentCount, compiler.span(node)));
    }
    let mut values = SmallVec::fixed(compiler.env(), count as usize)?;
    for i in 1..=count {
        compile(compiler, compiler.node(node.first_child() + i), depth + 1, nodes)?;
        values.push(nodes.last())?;
    }
    Ok(values)
}

/// Binds a function to the statement being compiled for the call node
/// passed, whose children after the name are its arguments: what needs only
/// doing once, such as compiling a pattern, is done here.
pub type Bind = fn(&mut Compiler<'_, '_>, Node) -> Result<DynFunction, Error>;

/// Appends the nodes calling the function `bind` binds for the call `node`,
/// `depth` deep, which takes `arguments`, of those types, and gives a
/// `result`: its arguments', an `Argument` for each, then the `Call`. A
/// `NULL` is an argument of any type, and an integer widens to a double.
pub fn call_bound(
    compiler: &mut Compiler<'_, '_>,
    (node, depth): (Node, usize),
    nodes: &mut ExprBuilder,
    (arguments, result): (&[DataType], DataType),
    bind: Bind,
) -> Result<(), Error> {
    check!(!arguments.is_empty());
    let Ok(count) = u32::try_from(arguments.len()) else { failed!() };
    let values = compile_arguments(compiler, (node, depth), nodes, count, count)?;
    let function = bind(compiler, node)?;
    let env = compiler.env();
    let Some(id) = compiler.plan.add_function(env, function)? else {
        return Err(Error::new(ErrorCode::ExpressionTooLarge, compiler.span(node)));
    };
    let mut first = None;
    for (&value, &data_type) in values.iter().zip(arguments) {
        let mut argument = ExprStep::new(ExprOp::Argument, value, 0, 0);
        argument.data_type = data_type;
        first.get_or_insert(push_node(nodes, argument, node, compiler)?);
    }
    let mut call = ExprStep::new(ExprOp::Call, first.unwrap_or(0), count, 0);
    call.data_type = result;
    call.c = id;
    push_node(nodes, call, node, compiler)?;
    Ok(())
}

/// Appends a `When` for each of `branches`, a condition and a value, then a
/// `Case` of them, with `otherwise` its `ELSE`: SQL's `CASE`, and the
/// functions that choose between their arguments.
pub(crate) fn push_case(
    compiler: &Compiler<'_, '_>,
    node: Node,
    nodes: &mut ExprBuilder,
    branches: &[(u32, u32)],
    otherwise: u32,
) -> Result<(), Error> {
    let mut first = None;
    for &(condition, value) in branches {
        let when =
            push_node(nodes, ExprStep::new(ExprOp::When, condition, value, 0), node, compiler)?;
        first.get_or_insert(when);
    }
    #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
    let mut case = ExprStep::new(ExprOp::Case, first.unwrap_or(0), branches.len() as u32, 0);
    case.c = otherwise;
    push_node(nodes, case, node, compiler)?;
    Ok(())
}

/// Appends the nodes of the `CASE` `node`, `depth` deep: each branch's
/// condition and value, then a `When` for each, then the `Case`. No `ELSE`
/// is `NULL`, and `CASE s WHEN v` tests `s = v`, with `s` computed once.
fn case(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
    nodes: &mut ExprBuilder,
) -> Result<(), Error> {
    let count = node.child_count();
    let child = |compiler: &Compiler<'_, '_>, i: u32| compiler.node(node.first_child() + i);
    let subject = child(compiler, 0);
    let subject = if subject.is_empty() {
        None
    } else {
        compile(compiler, subject, depth + 1, nodes)?;
        Some(nodes.last())
    };
    // Each `WHEN`'s condition and value.
    let mut branches = SmallVec::fixed(compiler.env(), (count as usize - 2) / 2)?;
    for i in (1..count - 1).step_by(2) {
        compile(compiler, child(compiler, i), depth + 1, nodes)?;
        let mut condition = nodes.last();
        if let Some(subject) = subject {
            let equal = ExprStep::new(ExprOp::Eq, subject, condition, 0);
            condition = push_node(nodes, equal, node, compiler)?;
        }
        compile(compiler, child(compiler, i + 1), depth + 1, nodes)?;
        branches.push((condition, nodes.last()))?;
    }
    let otherwise = child(compiler, count - 1);
    let otherwise = if otherwise.is_empty() {
        push_node(nodes, ExprStep::new(ExprOp::Null, 0, 0, 0), node, compiler)?
    } else {
        compile(compiler, otherwise, depth + 1, nodes)?;
        nodes.last()
    };
    push_case(compiler, node, nodes, &branches, otherwise)
}

/// Appends the nodes of `x` in `x IS [NOT] NULL` or `NOT x`, the node
/// `node`, `depth` deep, and returns the step computing it.
fn unary(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
    nodes: &mut ExprBuilder,
) -> Result<ExprStep, Error> {
    let op = match node.operator() {
        Operator::IsNull => ExprOp::IsNull,
        Operator::IsNotNull => ExprOp::IsNotNull,
        Operator::Not => ExprOp::Not,
        _ => return Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(node))),
    };
    compile(compiler, compiler.node(node.first_child()), depth + 1, nodes)?;
    Ok(ExprStep::new(op, nodes.last(), 0, 0))
}

/// Appends the nodes of `x` in `x LIKE 'pattern'`, the node `node`, `depth`
/// deep, and returns the step computing it. The pattern must be a string
/// literal, as Perfetto's `GLOB`'s must.
fn like(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
    nodes: &mut ExprBuilder,
) -> Result<ExprStep, Error> {
    let pattern = compiler.node(node.first_child() + 1);
    if pattern.tag() != Tag::String {
        return Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(pattern)));
    }
    compile(compiler, compiler.node(node.first_child()), depth + 1, nodes)?;
    let a = nodes.last();
    let pattern = string(compiler, pattern)?;
    Ok(ExprStep::new(ExprOp::Like, a, pattern.raw_id(), 0))
}

/// Appends the nodes of `-x`, the `Negate` node `node`, `depth` deep.
fn negate(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
    nodes: &mut ExprBuilder,
) -> Result<(), Error> {
    let operand = compiler.node(node.first_child());
    if operand.tag() == Tag::Integer {
        // Read as negative, so the smallest integer can be written.
        let value = integer(compiler, operand, true)?;
        push_node(nodes, ExprStep::new(ExprOp::Constant, 0, 0, value), node, compiler)?;
        return Ok(());
    }
    compile(compiler, operand, depth + 1, nodes)?;
    let x = nodes.last();
    let last = at_mut!(nodes.nodes_mut(), x as usize);
    if last.op == ExprOp::Constant
        && last.data_type == DataType::Int64
        && let Some(negated) = last.int_value.checked_neg()
    {
        last.int_value = negated;
        return Ok(());
    }
    // As in SQLite, -x is x * -1: an integer overflows only at its smallest.
    let minus_one = push_node(nodes, ExprStep::new(ExprOp::Constant, 0, 0, -1), node, compiler)?;
    push_node(nodes, ExprStep::new(ExprOp::Mul, x, minus_one, 0), node, compiler)?;
    Ok(())
}

/// The value of `node`, an `Integer`, negated if `negative`.
fn integer(compiler: &Compiler<'_, '_>, node: Node, negative: bool) -> Result<i64, Error> {
    let magnitude = i128::from(compiler.integer(node)?);
    let value = if negative { -magnitude } else { magnitude };
    i64::try_from(value).map_err(|_| Error::new(ErrorCode::NumberTooLarge, node.span()))
}

/// `node`, a string literal, added to the plan's literals, a quote for each
/// `''`, and its id there.
pub fn string(compiler: &mut Compiler<'_, '_>, node: Node) -> Result<Id, Error> {
    let quoted = compiler.bytes(node.span());
    // Between its quotes, which the lexer checked are there.
    let inside = at!(quoted, 1..quoted.len() - 1);
    let env = compiler.env();
    if !inside.windows(2).any(|pair| pair == b"''") {
        return Ok(compiler.plan.add_literal(env, inside)?);
    }
    let mut unquoted = SmallVec::fixed(env, inside.len())?;
    let mut quote = false;
    for &byte in inside {
        // The second quote of each `''` is the one kept.
        quote = byte == b'\'' && !quote;
        if !quote {
            unquoted.push(byte)?;
        }
    }
    Ok(compiler.plan.add_literal(env, &unquoted)?)
}
