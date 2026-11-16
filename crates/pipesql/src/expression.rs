//! Expressions over the columns in scope, compiled to the kernel's `Expr`,
//! as Perfetto's `CompileExpr`.

use pipit_kernel::column::DataType;
use pipit_kernel::expression::{
    EXPR_NODES_MAX, Expr, ExprError, ExprOp, ExprStep, PRESENT, check_expr_type,
};
use pipit_kernel::slow_vec::SlowVec;

use crate::ast::{Node, Operator, Tag};
use crate::compile::Compiler;
use crate::condition::string;
use crate::error::{Error, ErrorCode, Unsupported};
use crate::keywords::same_word;

/// Compiles the expression `node` against the columns in scope, and the type
/// it computes.
pub fn compile_expr(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
) -> Result<(Expr, DataType), Error> {
    let mut nodes = SlowVec::new(compiler.allocator(), EXPR_NODES_MAX)?;
    compile(compiler, node, 0, &mut nodes)?;
    let expr = Expr { nodes };
    let type_of = |column| at!(compiler.plan.columns, column as usize).data_type;
    let data_type = check_expr_type(&expr, &type_of).map_err(|error| {
        let span = compiler.span(node);
        let code = match error {
            ExprError::Numbers => ErrorCode::ExpectedNumber,
            ExprError::Integers => ErrorCode::ExpectedInteger,
            ExprError::Kinds => ErrorCode::ExpectedOneKind,
            ExprError::Types => ErrorCode::ExpectedOneType,
            ExprError::Inexact => return Error::unsupported(Unsupported::NumberType, span),
            ExprError::Strings => ErrorCode::ExpectedString,
            ExprError::SomeRows => ErrorCode::ArithmeticOnSomeRows,
            ExprError::Condition => ErrorCode::NumberCondition,
            ExprError::Choices => ErrorCode::ChoicesOfOneType,
            ExprError::OutOfMemory => ErrorCode::OutOfMemory,
        };
        Error::new(code, span)
    })?;
    Ok((expr, data_type))
}

/// Whether `node` is the literal `NULL`.
pub fn is_null(compiler: &Compiler<'_, '_>, node: Node) -> bool {
    node.tag() == Tag::Name && same_word(compiler.text(node.span()).as_bytes(), b"null")
}

/// A node of `op` reading nodes `a` and `b`, or of value `int_value`.
fn expr_node(op: ExprOp, a: u32, b: u32, int_value: i64) -> ExprStep {
    ExprStep { op, data_type: DataType::Int64, a, b, c: 0, int_value, double_value: 0.0 }
}

/// Appends `node`, and returns its index.
fn push_node(
    nodes: &mut SlowVec<ExprStep>,
    node: ExprStep,
    span: Node,
    compiler: &Compiler<'_, '_>,
) -> Result<u32, Error> {
    if nodes.len() == EXPR_NODES_MAX {
        return Err(Error::new(ErrorCode::ExpressionTooLarge, compiler.span(span)));
    }
    nodes.push(node)?;
    #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
    Ok(nodes.len() as u32 - 1)
}

/// Appends the nodes of the expression `node`, `depth` deep, its result
/// last.
fn compile(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
    nodes: &mut SlowVec<ExprStep>,
) -> Result<(), Error> {
    // An expression is at least as large as it is deep.
    if depth == EXPR_NODES_MAX {
        return Err(Error::new(ErrorCode::ExpressionTooLarge, compiler.span(node)));
    }
    let child = |compiler: &Compiler<'_, '_>, i: u32| compiler.node(node.first_child() + i);
    let push = |compiler: &Compiler<'_, '_>, nodes: &mut SlowVec<ExprStep>, added| {
        push_node(nodes, added, node, compiler)
    };
    match node.tag() {
        Tag::Name if is_null(compiler, node) => {
            push(compiler, nodes, expr_node(ExprOp::Null, 0, 0, 0))?;
        }
        Tag::Name => {
            let column = compiler.find_column(node)?.id;
            push(compiler, nodes, expr_node(ExprOp::Column, column, 0, 0))?;
        }
        Tag::Integer => {
            push(
                compiler,
                nodes,
                expr_node(ExprOp::Constant, 0, 0, integer(compiler, node, false)?),
            )?;
        }
        Tag::Float => {
            return Err(Error::unsupported(Unsupported::Decimal, node.span()));
        }
        Tag::String => {
            let id = string(compiler, node)?;
            let mut constant = expr_node(ExprOp::Constant, id.raw_id(), 0, 0);
            constant.data_type = DataType::String;
            push(compiler, nodes, constant)?;
        }
        Tag::Unary if node.operator() == Operator::Negate => negate(compiler, node, depth, nodes)?,
        Tag::Unary if matches!(node.operator(), Operator::IsNull | Operator::IsNotNull) => {
            null_test(compiler, node, depth, nodes)?;
        }
        Tag::Binary if node.operator() == Operator::Like => like(compiler, node, depth, nodes)?,
        Tag::Call => call(compiler, node, depth, nodes)?,
        Tag::Case => case(compiler, node, depth, nodes)?,
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
                _ => {
                    return Err(Error::unsupported(
                        Unsupported::SelectExpression,
                        compiler.span(node),
                    ));
                }
            };
            compile(compiler, child(compiler, 0), depth + 1, nodes)?;
            #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
            let a = nodes.len() as u32 - 1;
            compile(compiler, child(compiler, 1), depth + 1, nodes)?;
            #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
            let b = nodes.len() as u32 - 1;
            push(compiler, nodes, expr_node(op, a, b, 0))?;
        }
        _ => return Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(node))),
    }
    Ok(())
}

/// Appends the nodes of the call `node`, `depth` deep, as a `CASE`:
/// `iif(c, x, y)` is `CASE WHEN c THEN x ELSE y END`, and `coalesce(a, b,
/// c)` is `CASE WHEN a IS NOT NULL THEN a WHEN b IS NOT NULL THEN b ELSE c
/// END`, which `PRESENT` branches test. `ifnull` is `coalesce` of two.
fn call(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
    nodes: &mut SlowVec<ExprStep>,
) -> Result<(), Error> {
    let name = compiler.node(node.first_child());
    let text = compiler.text(name.span()).as_bytes();
    let (iif, ifnull) = (same_word(text, b"iif"), same_word(text, b"ifnull"));
    if !iif && !ifnull && !same_word(text, b"coalesce") {
        return Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(node)));
    }
    let count = node.child_count() - 1;
    let expected = if iif {
        count == 3
    } else if ifnull {
        count == 2
    } else {
        count >= 2
    };
    if !expected {
        return Err(Error::new(ErrorCode::ArgumentCount, compiler.span(node)));
    }
    // Each argument's result: the last of the nodes computing it.
    let mut values = SlowVec::fixed(compiler.allocator(), count as usize)?;
    for i in 0..count {
        compile(compiler, compiler.node(node.first_child() + 1 + i), depth + 1, nodes)?;
        #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
        values.push(nodes.len() as u32 - 1)?;
    }
    let branches = if iif { 1 } else { count - 1 };
    let mut first = None;
    for k in 0..branches as usize {
        let (condition, value) =
            if iif { (*at!(values, 0), *at!(values, 1)) } else { (PRESENT, *at!(values, k)) };
        let when = push_node(nodes, expr_node(ExprOp::When, condition, value, 0), node, compiler)?;
        first.get_or_insert(when);
    }
    let mut case = expr_node(ExprOp::Case, first.unwrap_or(0), branches, 0);
    case.c = *at!(values, count as usize - 1);
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
    nodes: &mut SlowVec<ExprStep>,
) -> Result<(), Error> {
    let count = node.child_count();
    let child = |compiler: &Compiler<'_, '_>, i: u32| compiler.node(node.first_child() + i);
    let last = |nodes: &SlowVec<ExprStep>| {
        #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
        let last = nodes.len() as u32 - 1;
        last
    };
    let subject = child(compiler, 0);
    let subject = if subject.is_empty() {
        None
    } else {
        compile(compiler, subject, depth + 1, nodes)?;
        Some(last(nodes))
    };
    // Each `WHEN`'s condition and value.
    let mut branches = SlowVec::fixed(compiler.allocator(), (count as usize - 2) / 2)?;
    for i in (1..count - 1).step_by(2) {
        compile(compiler, child(compiler, i), depth + 1, nodes)?;
        let mut condition = last(nodes);
        if let Some(subject) = subject {
            let equal = expr_node(ExprOp::Eq, subject, condition, 0);
            condition = push_node(nodes, equal, node, compiler)?;
        }
        compile(compiler, child(compiler, i + 1), depth + 1, nodes)?;
        branches.push((condition, last(nodes)))?;
    }
    let otherwise = child(compiler, count - 1);
    let otherwise = if otherwise.is_empty() {
        push_node(nodes, expr_node(ExprOp::Null, 0, 0, 0), node, compiler)?
    } else {
        compile(compiler, otherwise, depth + 1, nodes)?;
        last(nodes)
    };
    let mut first = None;
    for &(condition, value) in branches.iter() {
        let when = push_node(nodes, expr_node(ExprOp::When, condition, value, 0), node, compiler)?;
        first.get_or_insert(when);
    }
    #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
    let mut case = expr_node(ExprOp::Case, first.unwrap_or(0), branches.len() as u32, 0);
    case.c = otherwise;
    push_node(nodes, case, node, compiler)?;
    Ok(())
}

/// Appends the nodes of `x IS NULL` or `x IS NOT NULL`, the node `node`,
/// `depth` deep.
fn null_test(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
    nodes: &mut SlowVec<ExprStep>,
) -> Result<(), Error> {
    let op = if node.operator() == Operator::IsNull { ExprOp::IsNull } else { ExprOp::IsNotNull };
    compile(compiler, compiler.node(node.first_child()), depth + 1, nodes)?;
    #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
    let a = nodes.len() as u32 - 1;
    push_node(nodes, expr_node(op, a, 0, 0), node, compiler)?;
    Ok(())
}

/// Appends the nodes of `x LIKE 'pattern'`, the node `node`, `depth` deep.
/// The pattern must be a string literal, as Perfetto's `GLOB`'s must.
fn like(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
    nodes: &mut SlowVec<ExprStep>,
) -> Result<(), Error> {
    let pattern = compiler.node(node.first_child() + 1);
    if pattern.tag() != Tag::String {
        return Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(pattern)));
    }
    compile(compiler, compiler.node(node.first_child()), depth + 1, nodes)?;
    #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
    let a = nodes.len() as u32 - 1;
    let pattern = string(compiler, pattern)?;
    push_node(nodes, expr_node(ExprOp::Like, a, pattern.raw_id(), 0), node, compiler)?;
    Ok(())
}

/// Appends the nodes of `-x`, the `Negate` node `node`, `depth` deep.
fn negate(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
    nodes: &mut SlowVec<ExprStep>,
) -> Result<(), Error> {
    let operand = compiler.node(node.first_child());
    if operand.tag() == Tag::Integer {
        // Read as negative, so the smallest integer can be written.
        let value = integer(compiler, operand, true)?;
        push_node(nodes, expr_node(ExprOp::Constant, 0, 0, value), node, compiler)?;
        return Ok(());
    }
    compile(compiler, operand, depth + 1, nodes)?;
    #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
    let x = nodes.len() as u32 - 1;
    let last = at_mut!(nodes, x as usize);
    if last.op == ExprOp::Constant
        && last.data_type == DataType::Int64
        && let Some(negated) = last.int_value.checked_neg()
    {
        last.int_value = negated;
        return Ok(());
    }
    // As in SQLite, -x is x * -1: an integer overflows only at its smallest.
    let minus_one = push_node(nodes, expr_node(ExprOp::Constant, 0, 0, -1), node, compiler)?;
    push_node(nodes, expr_node(ExprOp::Mul, x, minus_one, 0), node, compiler)?;
    Ok(())
}

/// The value of `node`, an `Integer`, negated if `negative`.
fn integer(compiler: &Compiler<'_, '_>, node: Node, negative: bool) -> Result<i64, Error> {
    let magnitude = i128::from(compiler.integer(node)?);
    let value = if negative { -magnitude } else { magnitude };
    i64::try_from(value).map_err(|_| Error::new(ErrorCode::NumberTooLarge, node.span()))
}
