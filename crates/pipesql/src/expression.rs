//! Expressions over the columns in scope, built with the kernel's query
//! builder, as Perfetto's `CompileExpr` compiles them.

use pipit_kernel::column::DataType;
use pipit_kernel::containers::SmallVec;
use pipit_kernel::function::DynFunction;
use pipit_kernel::query::{Binary, Expr, Unary};

use crate::ast::{Node, Operator, Tag};
use crate::compile::{Compiler, kernel_error};
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
    let expr = compile(compiler, node, 0)?;
    let data_type = typed(compiler, expr, node)?;
    Ok((expr, data_type))
}

/// The type `expr`, compiled from `node`, computes.
pub fn typed(compiler: &Compiler<'_, '_>, expr: Expr, node: Node) -> Result<DataType, Error> {
    compiler.builder.type_of(expr).map_err(|error| kernel_error(error, compiler.span(node)))
}

/// Whether `node` is the literal `NULL`.
pub fn is_null(compiler: &Compiler<'_, '_>, node: Node) -> bool {
    node.tag() == Tag::Name && same_word(compiler.text(node.span()).as_bytes(), b"null")
}

/// What the builder makes of `built`, for `node`.
fn built(
    compiler: &Compiler<'_, '_>,
    node: Node,
    built: Result<Expr, pipit_kernel::error::Error>,
) -> Result<Expr, Error> {
    built.map_err(|error| kernel_error(error, compiler.span(node)))
}

/// `a IN (1, 2)` or `a IN ('x', 'y')`: a list of integers, or of strings,
/// all literals.
fn in_list(compiler: &mut Compiler<'_, '_>, node: Node, depth: usize) -> Result<Expr, Error> {
    let operand = compile(compiler, compiler.node(node.first_child()), depth + 1)?;
    let list = compiler.node(node.first_child() + 1);
    let len = list.child_count() as usize;
    let mut integers = SmallVec::fixed(compiler.env(), len)?;
    let mut strings: SmallVec<SmallVec<u8>> = SmallVec::fixed(compiler.env(), len)?;
    for i in 0..list.child_count() {
        let item = compiler.node(list.first_child() + i);
        let negated = item.tag() == Tag::Unary && item.operator() == Operator::Negate;
        let literal = if negated { compiler.node(item.first_child()) } else { item };
        match literal.tag() {
            Tag::Integer => integers.push(integer(compiler, literal, negated)?)?,
            Tag::String if !negated => strings.push(unquoted(compiler, literal)?)?,
            _ => return Err(Error::unsupported(Unsupported::Where, compiler.span(item))),
        }
        if !integers.is_empty() && !strings.is_empty() {
            return Err(Error::new(ErrorCode::ExpectedOneKind, compiler.span(item)));
        }
    }
    let tested = if strings.is_empty() {
        compiler.builder.is_in(operand, &integers)
    } else {
        let mut listed = SmallVec::fixed(compiler.env(), strings.len())?;
        for string in strings.iter() {
            listed.push(&string[..])?;
        }
        compiler.builder.is_in_strings(operand, &listed)
    };
    let span = compiler.span(list);
    tested.map_err(|error| match error {
        pipit_kernel::error::Error::TooLarge => Error::new(ErrorCode::ConditionTooLarge, span),
        error => kernel_error(error, span),
    })
}

/// Compiles the expression `node`, `depth` deep.
fn compile(compiler: &mut Compiler<'_, '_>, node: Node, depth: usize) -> Result<Expr, Error> {
    // Each level recurses, so as deep as the parser lets a query nest: a
    // chain of operators is as deep as it is long, but nests no deeper.
    if depth == NESTING_MAX {
        return Err(Error::new(ErrorCode::NestingTooDeep, compiler.span(node)));
    }
    let child = |compiler: &Compiler<'_, '_>, i: u32| compiler.node(node.first_child() + i);
    let expr = match node.tag() {
        Tag::Name if is_null(compiler, node) => compiler.builder.null(),
        Tag::Name => {
            let column = compiler.find_column(node)?.column;
            compiler.builder.column_value(column)
        }
        Tag::Integer => {
            let value = integer(compiler, node, false)?;
            compiler.builder.int(value)
        }
        Tag::Float => return Err(Error::unsupported(Unsupported::Decimal, node.span())),
        Tag::String => {
            let value = unquoted(compiler, node)?;
            compiler.builder.string(&value)
        }
        Tag::Unary if node.operator() == Operator::Negate => return negate(compiler, node, depth),
        Tag::Unary => {
            let op = match node.operator() {
                Operator::IsNull => Unary::IsNull,
                Operator::IsNotNull => Unary::IsNotNull,
                Operator::Not => Unary::Not,
                _ => {
                    let span = compiler.span(node);
                    return Err(Error::unsupported(Unsupported::SelectExpression, span));
                }
            };
            let operand = compile(compiler, child(compiler, 0), depth + 1)?;
            compiler.builder.unary(op, operand)
        }
        Tag::Binary if node.operator() == Operator::Like => return like(compiler, node, depth),
        Tag::Binary if node.operator() == Operator::In => return in_list(compiler, node, depth),
        Tag::Call => return (compiler.registry().call)(compiler, node, depth),
        Tag::Case => return case(compiler, node, depth),
        Tag::Binary => {
            let op = match node.operator() {
                Operator::Add => Binary::Add,
                Operator::Subtract => Binary::Subtract,
                Operator::Multiply => Binary::Multiply,
                Operator::Divide => Binary::Divide,
                Operator::Modulo => Binary::Modulo,
                Operator::Equal => Binary::Equal,
                Operator::NotEqual => Binary::NotEqual,
                Operator::Less => Binary::Less,
                Operator::LessEqual => Binary::LessEqual,
                Operator::Greater => Binary::Greater,
                Operator::GreaterEqual => Binary::GreaterEqual,
                Operator::And => Binary::And,
                Operator::Or => Binary::Or,
                _ => {
                    let span = compiler.span(node);
                    return Err(Error::unsupported(Unsupported::SelectExpression, span));
                }
            };
            let a = compile(compiler, child(compiler, 0), depth + 1)?;
            let b = compile(compiler, child(compiler, 1), depth + 1)?;
            compiler.builder.binary(op, a, b)
        }
        _ => return Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(node))),
    };
    built(compiler, node, expr)
}

/// How a registry compiles a call of a function it may have: as
/// `call_function` does, or as `no_function` does when it has none.
pub(crate) type CallFunction = fn(&mut Compiler<'_, '_>, Node, usize) -> Result<Expr, Error>;

/// Fails the call `node`, as a registry without functions can't compile it.
pub(crate) fn no_function(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    _: usize,
) -> Result<Expr, Error> {
    Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(node)))
}

/// Compiles the call `node`, `depth` deep, as the registry's rule for its
/// function does.
pub(crate) fn call_function(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
) -> Result<Expr, Error> {
    let name = compiler.node(node.first_child()).span();
    let text = compiler.text(name).as_bytes();
    let Some(rule) = compiler.registry().function(text) else {
        // An aggregate can't be called here.
        if aggregate(text).is_some() {
            return Err(Error::unsupported(Unsupported::SelectExpression, name));
        }
        return Err(Error::new(ErrorCode::UnknownFunction, name));
    };
    (rule.compile)(compiler, node, depth)
}

/// Compiles each argument of the call `node`, `depth` deep, if it has from
/// `fewest` to `most`.
pub fn compile_arguments(
    compiler: &mut Compiler<'_, '_>,
    (node, depth): (Node, usize),
    fewest: u32,
    most: u32,
) -> Result<SmallVec<Expr>, Error> {
    let count = node.child_count() - 1;
    if count < fewest || count > most {
        return Err(Error::new(ErrorCode::ArgumentCount, compiler.span(node)));
    }
    let mut values = SmallVec::fixed(compiler.env(), count as usize)?;
    for i in 1..=count {
        values.push(compile(compiler, compiler.node(node.first_child() + i), depth + 1)?)?;
    }
    Ok(values)
}

/// Binds a function to the statement being compiled for the call node
/// passed, whose children after the name are its arguments: what needs only
/// doing once, such as compiling a pattern, is done here.
pub type Bind = fn(&mut Compiler<'_, '_>, Node) -> Result<DynFunction, Error>;

/// Calls the function `bind` binds for the call `node`, `depth` deep, which
/// takes `arguments`, of those types, and gives a `result`. A `NULL` is an
/// argument of any type, and an integer widens to a double.
pub fn call_bound(
    compiler: &mut Compiler<'_, '_>,
    (node, depth): (Node, usize),
    (arguments, result): (&[DataType], DataType),
    bind: Bind,
) -> Result<Expr, Error> {
    check!(!arguments.is_empty());
    let Ok(count) = u32::try_from(arguments.len()) else { failed!() };
    let values = compile_arguments(compiler, (node, depth), count, count)?;
    let function = bind(compiler, node)?;
    let mut typed = SmallVec::fixed(compiler.env(), arguments.len())?;
    for (&value, &data_type) in values.iter().zip(arguments) {
        typed.push((value, data_type))?;
    }
    let called = compiler.builder.call(function, &typed, result);
    let span = compiler.span(node);
    called.map_err(|error| match error {
        pipit_kernel::error::Error::TooLarge => Error::new(ErrorCode::ExpressionTooLarge, span),
        error => kernel_error(error, span),
    })
}

/// A `CASE` of `branches`, each a condition, or `None` to take its value
/// where it isn't null, and a value, with `otherwise` its `ELSE`: SQL's
/// `CASE`, and the functions that choose between their arguments.
pub(crate) fn case_of(
    compiler: &Compiler<'_, '_>,
    node: Node,
    branches: &[(Option<Expr>, Expr)],
    otherwise: Expr,
) -> Result<Expr, Error> {
    built(compiler, node, compiler.builder.case(branches, otherwise))
}

/// The `CASE` `node`, `depth` deep. No `ELSE` is `NULL`, and `CASE s WHEN
/// v` tests `s = v`, with `s` computed once.
fn case(compiler: &mut Compiler<'_, '_>, node: Node, depth: usize) -> Result<Expr, Error> {
    let count = node.child_count();
    let child = |compiler: &Compiler<'_, '_>, i: u32| compiler.node(node.first_child() + i);
    let subject = child(compiler, 0);
    let subject =
        if subject.is_empty() { None } else { Some(compile(compiler, subject, depth + 1)?) };
    // Each `WHEN`'s condition and value.
    let mut branches = SmallVec::fixed(compiler.env(), (count as usize - 2) / 2)?;
    for i in (1..count - 1).step_by(2) {
        let mut condition = compile(compiler, child(compiler, i), depth + 1)?;
        if let Some(subject) = subject {
            condition =
                built(compiler, node, compiler.builder.binary(Binary::Equal, subject, condition))?;
        }
        let value = compile(compiler, child(compiler, i + 1), depth + 1)?;
        branches.push((Some(condition), value))?;
    }
    let otherwise = child(compiler, count - 1);
    let otherwise = if otherwise.is_empty() {
        built(compiler, node, compiler.builder.null())?
    } else {
        compile(compiler, otherwise, depth + 1)?
    };
    case_of(compiler, node, &branches, otherwise)
}

/// `x LIKE 'pattern'`, the node `node`, `depth` deep. The pattern must be a
/// string literal, as Perfetto's `GLOB`'s must.
fn like(compiler: &mut Compiler<'_, '_>, node: Node, depth: usize) -> Result<Expr, Error> {
    let pattern = compiler.node(node.first_child() + 1);
    if pattern.tag() != Tag::String {
        return Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(pattern)));
    }
    let a = compile(compiler, compiler.node(node.first_child()), depth + 1)?;
    let pattern = unquoted(compiler, pattern)?;
    built(compiler, node, compiler.builder.like(a, &pattern))
}

/// `-x`, the `Negate` node `node`, `depth` deep.
fn negate(compiler: &mut Compiler<'_, '_>, node: Node, depth: usize) -> Result<Expr, Error> {
    let operand = compiler.node(node.first_child());
    if operand.tag() == Tag::Integer {
        // Read as negative, so the smallest integer can be written.
        let value = integer(compiler, operand, true)?;
        return built(compiler, node, compiler.builder.int(value));
    }
    let x = compile(compiler, operand, depth + 1)?;
    // As in SQLite, -x is x * -1: an integer overflows only at its smallest.
    let minus_one = built(compiler, node, compiler.builder.int(-1))?;
    built(compiler, node, compiler.builder.binary(Binary::Multiply, x, minus_one))
}

/// The value of `node`, an `Integer`, negated if `negative`.
fn integer(compiler: &Compiler<'_, '_>, node: Node, negative: bool) -> Result<i64, Error> {
    let magnitude = i128::from(compiler.integer(node)?);
    let value = if negative { -magnitude } else { magnitude };
    i64::try_from(value).map_err(|_| Error::new(ErrorCode::NumberTooLarge, node.span()))
}

/// The bytes of `node`, a string literal, a quote for each `''`.
pub fn unquoted(compiler: &Compiler<'_, '_>, node: Node) -> Result<SmallVec<u8>, Error> {
    let quoted = compiler.bytes(node.span());
    // Between its quotes, which the lexer checked are there.
    let inside = at!(quoted, 1..quoted.len() - 1);
    let mut unquoted = SmallVec::fixed(compiler.env(), inside.len())?;
    let mut quote = false;
    for &byte in inside {
        // The second quote of each `''` is the one kept.
        quote = byte == b'\'' && !quote;
        if !quote {
            unquoted.push(byte)?;
        }
    }
    Ok(unquoted)
}
