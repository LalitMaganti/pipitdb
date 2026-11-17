//! Conditions, such as `WHERE`'s, compiled to a `Predicate`.
//!
//! A condition is comparisons of a column with an integer, or of a string
//! column with a string or a `LIKE` pattern, `IN` lists of these, and
//! `IS [NOT] NULL`s, combined with `AND`, `OR`, `NOT` and parentheses.
//! Anything else is `Unsupported`.

use pipit_kernel::column::DataType;
use pipit_kernel::filter::{Comparison, Value};
use pipit_kernel::predicate::{Leaf, Node as PredicateNode, PREDICATE_NODES_MAX, Predicate};
use pipit_kernel::slow_vec::SlowVec;
use pipit_kernel::string_pool::Id;

use crate::ast::{Node, Operator, Tag};
use crate::compile::Compiler;
use crate::error::{Error, ErrorCode, Unsupported};
use crate::parser::IN_LIST_MAX;

/// The most values a condition's `IN` lists can have, together.
const VALUES_MAX: usize = IN_LIST_MAX;

/// `node`, a condition, as a predicate over the plan's columns. Its strings
/// are added to the plan's literals.
pub fn compile_condition(compiler: &mut Compiler<'_, '_>, node: Node) -> Result<Predicate, Error> {
    let mut nodes = SlowVec::new(compiler.allocator(), PREDICATE_NODES_MAX)?;
    // The values of its `IN` lists.
    let mut values = SlowVec::new(compiler.allocator(), VALUES_MAX)?;
    add(compiler, &mut nodes, &mut values, node)?;
    Ok(Predicate::with_values(compiler.allocator(), nodes, &values)?)
}

/// Adds `node`'s predicate nodes, children first, and returns the index of
/// its own, adding its `IN` lists' values to `values`. The parser limits how
/// deep this recurses.
fn add(
    compiler: &mut Compiler<'_, '_>,
    nodes: &mut SlowVec<PredicateNode>,
    values: &mut SlowVec<i64>,
    node: Node,
) -> Result<u32, Error> {
    let child = |compiler: &Compiler<'_, '_>, i: u32| compiler.node(node.first_child() + i);
    let added = match node.tag() {
        Tag::Unary if node.operator() == Operator::Not => {
            PredicateNode::Not(add(compiler, nodes, values, child(compiler, 0))?)
        }
        Tag::Unary if matches!(node.operator(), Operator::IsNull | Operator::IsNotNull) => {
            let name = child(compiler, 0);
            if name.tag() != Tag::Name {
                return Err(Error::unsupported(Unsupported::Where, compiler.span(name)));
            }
            let leaf = PredicateNode::Leaf(Leaf::IsNull { column: compiler.find_column(name)?.id });
            if node.operator() == Operator::IsNull {
                leaf
            } else {
                PredicateNode::Not(push(compiler, nodes, leaf, name)?)
            }
        }
        Tag::Binary if matches!(node.operator(), Operator::And | Operator::Or) => {
            let a = add(compiler, nodes, values, child(compiler, 0))?;
            let b = add(compiler, nodes, values, child(compiler, 1))?;
            if node.operator() == Operator::And {
                PredicateNode::And(a, b)
            } else {
                PredicateNode::Or(a, b)
            }
        }
        Tag::Binary if node.operator() == Operator::In => {
            PredicateNode::Leaf(in_list(compiler, values, node)?)
        }
        Tag::Binary => PredicateNode::Leaf(compare(compiler, node)?),
        _ => return Err(Error::unsupported(Unsupported::Where, compiler.span(node))),
    };
    push(compiler, nodes, added, node)
}

/// Pushes `added`, for `node`, to `nodes`, and returns its index.
fn push(
    compiler: &Compiler<'_, '_>,
    nodes: &mut SlowVec<PredicateNode>,
    added: PredicateNode,
    node: Node,
) -> Result<u32, Error> {
    if nodes.len() == PREDICATE_NODES_MAX {
        return Err(Error::new(ErrorCode::ConditionTooLarge, compiler.span(node)));
    }
    nodes.push(added)?;
    #[expect(clippy::cast_possible_truncation, reason = "at most `PREDICATE_NODES_MAX`")]
    Ok(nodes.len() as u32 - 1)
}

/// A comparison of a column with a number, or of a string column with a
/// string, either way round.
fn compare(compiler: &mut Compiler<'_, '_>, node: Node) -> Result<Leaf, Error> {
    let comparison = match node.operator() {
        Operator::Equal => Comparison::Equal,
        Operator::NotEqual => Comparison::NotEqual,
        Operator::Less => Comparison::Less,
        Operator::LessEqual => Comparison::LessEqual,
        Operator::Greater => Comparison::Greater,
        Operator::GreaterEqual => Comparison::GreaterEqual,
        Operator::Like => return like(compiler, node),
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
        let string = string(compiler, literal)?;
        return Ok(Leaf::CompareString { column: column.id, comparison, string });
    }
    let value = value(compiler, literal, false, data_type)?;
    Ok(Leaf::Compare { column: column.id, comparison, value })
}

/// `column LIKE 'pattern'`, a string column matched with a string.
fn like(compiler: &mut Compiler<'_, '_>, node: Node) -> Result<Leaf, Error> {
    let (left, right) = (compiler.node(node.first_child()), compiler.node(node.first_child() + 1));
    if left.tag() != Tag::Name || right.tag() != Tag::String {
        return Err(Error::unsupported(Unsupported::Where, compiler.span(node)));
    }
    let column = compiler.find_column(left)?;
    if at!(compiler.plan.columns, column.id as usize).data_type != DataType::String {
        return Err(Error::unsupported(Unsupported::Where, compiler.span(node)));
    }
    let pattern = string(compiler, right)?;
    Ok(Leaf::Like { column: column.id, pattern })
}

/// `column IN (1, 2)`, an integer column and a list of integers, or a string
/// column and a list of strings, whose values, or ids, go in `values`.
fn in_list(
    compiler: &mut Compiler<'_, '_>,
    values: &mut SlowVec<i64>,
    node: Node,
) -> Result<Leaf, Error> {
    let (column, list) = (compiler.node(node.first_child()), compiler.node(node.first_child() + 1));
    let unsupported = |compiler: &Compiler<'_, '_>, node| {
        Error::unsupported(Unsupported::Where, compiler.span(node))
    };
    if column.tag() != Tag::Name {
        return Err(unsupported(compiler, node));
    }
    let column = compiler.find_column(column)?;
    let data_type = at!(compiler.plan.columns, column.id as usize).data_type;
    let len = list.child_count();
    if values.len() + len as usize > VALUES_MAX {
        return Err(Error::new(ErrorCode::ConditionTooLarge, compiler.span(list)));
    }
    #[expect(clippy::cast_possible_truncation, reason = "at most `VALUES_MAX`")]
    let start = values.len() as u32;
    for i in 0..len {
        let item = compiler.node(list.first_child() + i);
        let value = match data_type {
            DataType::Int64 => {
                let Value::Int64(value) = value(compiler, item, false, data_type)? else {
                    pipit_kernel::failed!()
                };
                value
            }
            DataType::String if item.tag() == Tag::String => {
                i64::from(string(compiler, item)?.raw_id())
            }
            DataType::String | DataType::Float64 => return Err(unsupported(compiler, item)),
        };
        values.push(value)?;
    }
    Ok(match data_type {
        DataType::String => Leaf::InStrings { column: column.id, start, len },
        _ => Leaf::In { column: column.id, start, len },
    })
}

/// `node`, a string literal, added to the plan's literals, a quote for each
/// `''`, and its id there.
fn string(compiler: &mut Compiler<'_, '_>, node: Node) -> Result<Id, Error> {
    let quoted = compiler.bytes(node.span());
    // Between its quotes, which the lexer checked are there.
    let inside = at!(quoted, 1..quoted.len() - 1);
    let allocator = compiler.allocator();
    if !inside.windows(2).any(|pair| pair == b"''") {
        return Ok(compiler.plan.add_literal(allocator, inside)?);
    }
    let mut unquoted = SlowVec::fixed(allocator, inside.len())?;
    let mut quote = false;
    for &byte in inside {
        // The second quote of each `''` is the one kept.
        quote = byte == b'\'' && !quote;
        if !quote {
            unquoted.push(byte)?;
        }
    }
    Ok(compiler.plan.add_literal(allocator, &unquoted)?)
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
    let inexact = Error::unsupported(Unsupported::NumberType, compiler.span(node));
    match node.tag() {
        Tag::Unary if node.operator() == Operator::Negate => {
            value(compiler, compiler.node(node.first_child()), !negative, data_type)
        }
        Tag::Integer => {
            let too_large = Error::new(ErrorCode::NumberTooLarge, node.span());
            let magnitude = i128::from(compiler.integer(node)?);
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
