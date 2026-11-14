//! Expressions over the columns in scope, compiled to the kernel's `Expr`,
//! as Perfetto's `CompileExpr`.

use pipit_kernel::column::DataType;
use pipit_kernel::expression::{
    EXPR_NODES_MAX, Expr, ExprError, ExprOp, ExprStep, check_expr_type,
};
use pipit_kernel::slow_vec::SlowVec;

use crate::ast::{Node, Operator, Tag};
use crate::compile::Compiler;
use crate::error::{Error, ErrorCode, Unsupported};
use crate::keywords::same_word;

/// Compiles the expression `node` against the columns in scope, and the type
/// it computes.
pub fn compile_expr(compiler: &Compiler<'_, '_>, node: Node) -> Result<(Expr, DataType), Error> {
    let mut nodes = SlowVec::new(compiler.allocator(), EXPR_NODES_MAX)?;
    compile(compiler, node, 0, &mut nodes)?;
    let expr = Expr { nodes };
    let type_of = |column| at!(compiler.plan.columns, column as usize).data_type;
    let data_type = check_expr_type(&expr, &type_of).map_err(|error| {
        let code = match error {
            ExprError::Numbers => ErrorCode::ExpectedNumber,
            ExprError::Integers => ErrorCode::ExpectedInteger,
            ExprError::OutOfMemory => ErrorCode::OutOfMemory,
        };
        Error::new(code, compiler.span(node))
    })?;
    Ok((expr, data_type))
}

/// Whether `node` is the literal `NULL`.
pub fn is_null(compiler: &Compiler<'_, '_>, node: Node) -> bool {
    node.tag() == Tag::Name && same_word(compiler.text(node.span()).as_bytes(), b"null")
}

/// A node of `op` reading nodes `a` and `b`, or of value `int_value`.
fn expr_node(op: ExprOp, a: u32, b: u32, int_value: i64) -> ExprStep {
    ExprStep { op, data_type: DataType::Int64, a, b, int_value, double_value: 0.0 }
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
    compiler: &Compiler<'_, '_>,
    node: Node,
    depth: usize,
    nodes: &mut SlowVec<ExprStep>,
) -> Result<(), Error> {
    // An expression is at least as large as it is deep.
    if depth == EXPR_NODES_MAX {
        return Err(Error::new(ErrorCode::ExpressionTooLarge, compiler.span(node)));
    }
    let child = |i: u32| compiler.node(node.first_child() + i);
    let push = |nodes: &mut SlowVec<ExprStep>, added| push_node(nodes, added, node, compiler);
    match node.tag() {
        Tag::Name if is_null(compiler, node) => {
            push(nodes, expr_node(ExprOp::Null, 0, 0, 0))?;
        }
        Tag::Name => {
            let column = compiler.find_column(node)?.id;
            push(nodes, expr_node(ExprOp::Column, column, 0, 0))?;
        }
        Tag::Integer => {
            push(nodes, expr_node(ExprOp::Constant, 0, 0, integer(compiler, node, false)?))?;
        }
        Tag::Float => {
            return Err(Error::unsupported(Unsupported::Decimal, node.span()));
        }
        Tag::Unary if node.operator() == Operator::Negate => {
            let operand = child(0);
            if operand.tag() == Tag::Integer {
                // Read as negative, so the smallest integer can be written.
                let value = integer(compiler, operand, true)?;
                push(nodes, expr_node(ExprOp::Constant, 0, 0, value))?;
                return Ok(());
            }
            compile(compiler, operand, depth + 1, nodes)?;
            #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
            let x = nodes.len() as u32 - 1;
            let last = at_mut!(nodes, x as usize);
            if last.op == ExprOp::Constant
                && let Some(negated) = last.int_value.checked_neg()
            {
                last.int_value = negated;
                return Ok(());
            }
            // As in SQLite, -x is x * -1: an integer overflows only at its
            // smallest.
            let minus_one = push(nodes, expr_node(ExprOp::Constant, 0, 0, -1))?;
            push(nodes, expr_node(ExprOp::Mul, x, minus_one, 0))?;
        }
        Tag::Binary => {
            let op = match node.operator() {
                Operator::Add => ExprOp::Add,
                Operator::Subtract => ExprOp::Sub,
                Operator::Multiply => ExprOp::Mul,
                Operator::Divide => ExprOp::Div,
                Operator::Modulo => ExprOp::Mod,
                _ => {
                    return Err(Error::unsupported(
                        Unsupported::SelectExpression,
                        compiler.span(node),
                    ));
                }
            };
            compile(compiler, child(0), depth + 1, nodes)?;
            #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
            let a = nodes.len() as u32 - 1;
            compile(compiler, child(1), depth + 1, nodes)?;
            #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
            let b = nodes.len() as u32 - 1;
            push(nodes, expr_node(op, a, b, 0))?;
        }
        _ => return Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(node))),
    }
    Ok(())
}

/// The value of `node`, an `Integer`, negated if `negative`.
fn integer(compiler: &Compiler<'_, '_>, node: Node, negative: bool) -> Result<i64, Error> {
    let magnitude = i128::from(compiler.integer(node)?);
    let value = if negative { -magnitude } else { magnitude };
    i64::try_from(value).map_err(|_| Error::new(ErrorCode::NumberTooLarge, node.span()))
}
