//! Expressions over the columns in scope, built into the query, as
//! Perfetto's `CompileExpr`.

use pipit_kernel::column::DataType;
use pipit_kernel::containers::SmallVec;
use pipit_kernel::expression::Expr;
use pipit_kernel::function::DynFunction;
use pipit_kernel::query::{Binary, Unary, Value};

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
    compile(compiler, node, 0)?;
    finish_expr(compiler, node)
}

/// Takes the expression built, compiled from `node`, and the type it
/// computes.
pub fn finish_expr(compiler: &mut Compiler<'_, '_>, node: Node) -> Result<(Expr, DataType), Error> {
    compiler.builder.expr().map_err(|error| Error::from(error).at(compiler.span(node)))
}

/// Whether `node` is the literal `NULL`.
pub fn is_null(compiler: &Compiler<'_, '_>, node: Node) -> bool {
    node.tag() == Tag::Name && same_word(compiler.text(node.span()).as_bytes(), b"null")
}

/// `a IN (1, 2)` or `a IN ('x', 'y')`: a list of integers, or of strings,
/// all literals.
fn in_list(compiler: &mut Compiler<'_, '_>, node: Node, depth: usize) -> Result<Value, Error> {
    let operand = compile(compiler, compiler.node(node.first_child()), depth + 1)?;
    let list = compiler.node(node.first_child() + 1);
    let mut listed = compiler.builder.in_list(operand);
    for i in 0..list.child_count() {
        let mut item = compiler.node(list.first_child() + i);
        let negative = item.tag() == Tag::Unary && item.operator() == Operator::Negate;
        if negative {
            item = compiler.node(item.first_child());
        }
        let added = match item.tag() {
            Tag::Integer => {
                let value = integer(compiler, item, negative)?;
                compiler.builder.list_int(&mut listed, value)
            }
            Tag::String if !negative => {
                let value = unquoted(compiler, item)?;
                compiler.builder.list_string(&mut listed, &value)
            }
            Tag::Float => return Err(Error::unsupported(Unsupported::Decimal, item.span())),
            _ => return Err(Error::unsupported(Unsupported::Where, compiler.span(item))),
        };
        added.map_err(|error| Error::from(error).at(compiler.span(item)))?;
    }
    Ok(compiler.builder.listed(listed)?)
}

/// Builds the expression `node`, `depth` deep, and returns its value. A
/// kernel error is about it, unless about a part of it.
fn compile(compiler: &mut Compiler<'_, '_>, node: Node, depth: usize) -> Result<Value, Error> {
    compile_node(compiler, node, depth).map_err(|error| error.at(compiler.span(node)))
}

fn compile_node(compiler: &mut Compiler<'_, '_>, node: Node, depth: usize) -> Result<Value, Error> {
    // Each level recurses, so as deep as the parser lets a query nest: a
    // chain of operators is as deep as it is long, but nests no deeper.
    if depth == NESTING_MAX {
        return Err(Error::new(ErrorCode::NestingTooDeep, compiler.span(node)));
    }
    let child = |compiler: &Compiler<'_, '_>, i: u32| compiler.node(node.first_child() + i);
    let value = match node.tag() {
        Tag::Name if is_null(compiler, node) => compiler.builder.null()?,
        Tag::Name => {
            let column = compiler.find_column(node)?.column;
            compiler.builder.column_value(column)?
        }
        Tag::Integer => {
            let value = integer(compiler, node, false)?;
            compiler.builder.int(value)?
        }
        Tag::Float => {
            return Err(Error::unsupported(Unsupported::Decimal, node.span()));
        }
        Tag::String => {
            let value = unquoted(compiler, node)?;
            compiler.builder.string(&value)?
        }
        Tag::Unary if node.operator() == Operator::Negate => {
            return negate(compiler, child(compiler, 0), depth);
        }
        Tag::Unary if let Some(op) = unary(node.operator()) => {
            let x = compile(compiler, child(compiler, 0), depth + 1)?;
            compiler.builder.unary(op, x)?
        }
        // The pattern must be a string literal, as Perfetto's `GLOB`'s must.
        Tag::Binary if node.operator() == Operator::Like => {
            let pattern = child(compiler, 1);
            if pattern.tag() != Tag::String {
                let span = compiler.span(pattern);
                return Err(Error::unsupported(Unsupported::SelectExpression, span));
            }
            let a = compile(compiler, child(compiler, 0), depth + 1)?;
            let pattern = unquoted(compiler, pattern)?;
            compiler.builder.like(a, &pattern)?
        }
        Tag::Binary if node.operator() == Operator::In => return in_list(compiler, node, depth),
        Tag::Call => return (compiler.registry().call)(compiler, node, depth),
        Tag::Case => return case(compiler, node, depth),
        Tag::Binary if let Some(op) = binary(node.operator()) => {
            let a = compile(compiler, child(compiler, 0), depth + 1)?;
            let b = compile(compiler, child(compiler, 1), depth + 1)?;
            compiler.builder.binary(op, a, b)?
        }
        _ => return Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(node))),
    };
    Ok(value)
}

/// `-x`, of the operand `x`, `depth` deep.
fn negate(compiler: &mut Compiler<'_, '_>, x: Node, depth: usize) -> Result<Value, Error> {
    if x.tag() == Tag::Integer {
        // Read as negative, so the smallest integer can be written.
        let value = integer(compiler, x, true)?;
        return Ok(compiler.builder.int(value)?);
    }
    // As in SQLite, -x is x * -1: an integer overflows only at its smallest.
    let x = compile(compiler, x, depth + 1)?;
    let minus_one = compiler.builder.int(-1)?;
    Ok(compiler.builder.binary(Binary::Multiply, x, minus_one)?)
}

/// The kernel's operation of the unary `operator`, if it has one.
fn unary(operator: Operator) -> Option<Unary> {
    match operator {
        Operator::IsNull => Some(Unary::IsNull),
        Operator::IsNotNull => Some(Unary::IsNotNull),
        Operator::Not => Some(Unary::Not),
        _ => None,
    }
}

/// The kernel's operation of the binary `operator`, if it has one.
fn binary(operator: Operator) -> Option<Binary> {
    Some(match operator {
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
        _ => return None,
    })
}

/// How a registry compiles a call of a function it may have: as
/// `call_function` does, or as `no_function` does when it has none.
pub(crate) type CallFunction = fn(&mut Compiler<'_, '_>, Node, usize) -> Result<Value, Error>;

/// Fails the call `node`, as a registry without functions can't compile it.
pub(crate) fn no_function(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    _: usize,
) -> Result<Value, Error> {
    Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(node)))
}

/// Builds the call `node`, `depth` deep, as the registry's rule for its
/// function does.
pub(crate) fn call_function(
    compiler: &mut Compiler<'_, '_>,
    node: Node,
    depth: usize,
) -> Result<Value, Error> {
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

/// Builds each argument of the call `node`, `depth` deep, if it has from
/// `fewest` to `most`, and returns their values.
pub fn compile_arguments(
    compiler: &mut Compiler<'_, '_>,
    (node, depth): (Node, usize),
    fewest: u32,
    most: u32,
) -> Result<SmallVec<Value>, Error> {
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

/// Builds a call of the function `bind` binds for the call `node`, `depth`
/// deep, which takes `arguments`, of those types, and gives a `result`. A
/// `NULL` is an argument of any type, and an integer widens to a double.
pub fn call_bound(
    compiler: &mut Compiler<'_, '_>,
    (node, depth): (Node, usize),
    (arguments, result): (&[DataType], DataType),
    bind: Bind,
) -> Result<Value, Error> {
    check!(!arguments.is_empty());
    let Ok(count) = u32::try_from(arguments.len()) else { failed!() };
    let values = compile_arguments(compiler, (node, depth), count, count)?;
    let function = bind(compiler, node)?;
    let mut typed = SmallVec::fixed(compiler.env(), arguments.len())?;
    for (&value, &data_type) in values.iter().zip(arguments) {
        typed.push((value, data_type))?;
    }
    Ok(compiler.builder.call(function, &typed, result)?)
}

/// Builds the `CASE` `node`, `depth` deep. No `ELSE` is `NULL`, and
/// `CASE s WHEN v` tests `s = v`, with `s` computed once.
fn case(compiler: &mut Compiler<'_, '_>, node: Node, depth: usize) -> Result<Value, Error> {
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
            condition = compiler.builder.binary(Binary::Equal, subject, condition)?;
        }
        let value = compile(compiler, child(compiler, i + 1), depth + 1)?;
        branches.push((Some(condition), value))?;
    }
    let otherwise = child(compiler, count - 1);
    let otherwise = if otherwise.is_empty() {
        compiler.builder.null()?
    } else {
        compile(compiler, otherwise, depth + 1)?
    };
    Ok(compiler.builder.case(&branches, otherwise)?)
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
