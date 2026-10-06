//! Conditions, such as `WHERE`'s, compiled to a `Predicate`.
//!
//! A condition is comparisons of a column with an integer, or of a string
//! column with a string, combined with `AND`, `OR`, `NOT` and parentheses.
//! Anything else is `Unsupported`.

use pipit_kernel::column::DataType;
use pipit_kernel::filter::{Comparison, Value};
use pipit_kernel::predicate::{Leaf, Node as PredicateNode, PREDICATE_NODES_MAX, Predicate};
use pipit_kernel::slow_vec::SlowVec;

use crate::ast::{Node, Operator, Tag};
use crate::compile::Compiler;
use crate::error::{Error, ErrorCode, Unsupported};

/// The most bytes a condition's strings can have, together.
const STRINGS_MAX: usize = 1 << 16;

/// `node`, a condition, as a predicate over the plan's columns.
pub fn compile_condition(compiler: &Compiler<'_, '_>, node: Node) -> Result<Predicate, Error> {
    let mut nodes = SlowVec::new(compiler.allocator(), PREDICATE_NODES_MAX)?;
    let mut strings = SlowVec::new(compiler.allocator(), STRINGS_MAX)?;
    add(compiler, &mut nodes, &mut strings, node)?;
    let strings = (!strings.is_empty()).then_some(strings);
    Ok(Predicate::with_strings(nodes, strings))
}

/// Adds `node`'s predicate nodes, children first, and returns the index of
/// its own. The parser limits how deep this recurses.
fn add(
    compiler: &Compiler<'_, '_>,
    nodes: &mut SlowVec<PredicateNode>,
    strings: &mut SlowVec<u8>,
    node: Node,
) -> Result<u32, Error> {
    let child = |i: u32| compiler.node(node.first_child() + i);
    let mut add = |node: Node| add(compiler, nodes, strings, node);
    let added = match node.tag() {
        Tag::Unary if node.operator() == Operator::Not => PredicateNode::Not(add(child(0))?),
        Tag::Binary if node.operator() == Operator::And => {
            PredicateNode::And(add(child(0))?, add(child(1))?)
        }
        Tag::Binary if node.operator() == Operator::Or => {
            PredicateNode::Or(add(child(0))?, add(child(1))?)
        }
        Tag::Binary => PredicateNode::Leaf(compare(compiler, strings, node)?),
        _ => return Err(Error::unsupported(Unsupported::Where, compiler.span(node))),
    };
    if nodes.len() == PREDICATE_NODES_MAX {
        return Err(Error::new(ErrorCode::ConditionTooLarge, compiler.span(node)));
    }
    nodes.push(added)?;
    #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX`")]
    Ok(nodes.len() as u32 - 1)
}

/// A comparison of a column with a number, or of a string column with a
/// string, whose bytes go in `strings`, either way round.
fn compare(
    compiler: &Compiler<'_, '_>,
    strings: &mut SlowVec<u8>,
    node: Node,
) -> Result<Leaf, Error> {
    let comparison = match node.operator() {
        Operator::Equal => Comparison::Equal,
        Operator::NotEqual => Comparison::NotEqual,
        Operator::Less => Comparison::Less,
        Operator::LessEqual => Comparison::LessEqual,
        Operator::Greater => Comparison::Greater,
        Operator::GreaterEqual => Comparison::GreaterEqual,
        _ => return Err(Error::unsupported(Unsupported::Where, compiler.span(node))),
    };
    let (left, right) = (compiler.node(node.first_child()), compiler.node(node.first_child() + 1));
    let (column, literal, comparison) = match (left.tag(), right.tag()) {
        (Tag::Name, _) => (left, right, comparison),
        // `1 < x` is `x > 1`.
        (_, Tag::Name) => (right, left, flipped(comparison)),
        _ => return Err(Error::unsupported(Unsupported::Where, compiler.span(node))),
    };
    let column = compiler.find_column(column)?;
    let data_type = at!(compiler.plan.columns, column.id as usize).data_type;
    if data_type == DataType::String && literal.tag() == Tag::String {
        let (start, len) = string(compiler, strings, literal)?;
        return Ok(Leaf::CompareString { column: column.id, comparison, start, len });
    }
    let value = value(compiler, literal, false, data_type)?;
    Ok(Leaf::Compare { column: column.id, comparison, value })
}

/// Adds the bytes of `node`, a string literal, to `strings`, a quote for
/// each `''`, and returns where they start and how many there are.
fn string(
    compiler: &Compiler<'_, '_>,
    strings: &mut SlowVec<u8>,
    node: Node,
) -> Result<(u32, u32), Error> {
    let quoted = compiler.bytes(node.span());
    // Between its quotes, which the lexer checked are there.
    let inside = at!(quoted, 1..quoted.len() - 1);
    if strings.len() + inside.len() > STRINGS_MAX {
        return Err(Error::new(ErrorCode::ConditionTooLarge, node.span()));
    }
    let start = strings.len();
    let mut quote = false;
    for &byte in inside {
        // The second quote of each `''` is the one kept.
        quote = byte == b'\'' && !quote;
        if !quote {
            strings.push(byte)?;
        }
    }
    #[expect(clippy::cast_possible_truncation, reason = "at most `STRINGS_MAX`")]
    Ok((start as u32, (strings.len() - start) as u32))
}

/// The comparison with its sides swapped.
fn flipped(comparison: Comparison) -> Comparison {
    match comparison {
        Comparison::Less => Comparison::Greater,
        Comparison::LessEqual => Comparison::GreaterEqual,
        Comparison::Greater => Comparison::Less,
        Comparison::GreaterEqual => Comparison::LessEqual,
        Comparison::Equal | Comparison::NotEqual => comparison,
    }
}

/// `node`, an integer, negated if `negative`, as a value of `data_type`,
/// which must hold it exactly.
fn value(
    compiler: &Compiler<'_, '_>,
    node: Node,
    negative: bool,
    data_type: DataType,
) -> Result<Value, Error> {
    let text = |node: Node| compiler.text(node.span());
    let inexact = Error::unsupported(Unsupported::NumberType, compiler.span(node));
    match node.tag() {
        Tag::Unary if node.operator() == Operator::Negate => {
            value(compiler, compiler.node(node.first_child()), !negative, data_type)
        }
        Tag::Integer => {
            // The lexer's integers are digits, so the only error is size.
            let too_large = Error::new(ErrorCode::NumberTooLarge, node.span());
            let magnitude = i128::from(text(node).parse::<u64>().map_err(|_| too_large)?);
            let value = if negative { -magnitude } else { magnitude };
            let value = i64::try_from(value).map_err(|_| too_large)?;
            match data_type {
                DataType::Int64 => Ok(Value::Int64(value)),
                #[expect(clippy::cast_precision_loss, reason = "checked to be exact")]
                DataType::Float64 => {
                    let float = value as f64;
                    #[expect(clippy::cast_possible_truncation, reason = "a check, not a use")]
                    let exact = float as i128 == i128::from(value);
                    if exact { Ok(Value::Float64(float)) } else { Err(inexact) }
                }
                DataType::String => {
                    Err(Error::unsupported(Unsupported::Where, compiler.span(node)))
                }
            }
        }
        Tag::Float => Err(Error::unsupported(Unsupported::Decimal, compiler.span(node))),
        _ => Err(Error::unsupported(Unsupported::Where, compiler.span(node))),
    }
}
