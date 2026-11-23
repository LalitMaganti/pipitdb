//! The stages PipeSQL comes with, as rules for a registry. They use the same
//! API as any extension.

use pipit_kernel::aggregate::{Aggregate, AggregateOp, Function};
use pipit_kernel::column::DataType;
use pipit_kernel::expression::{
    ComputeItem, ComputeOp, Expr, ExprBuilder, ExprOp, expr_can_be_null,
};
use pipit_kernel::join::{JoinKind, JoinOp};
use pipit_kernel::limit::LimitOp;
use pipit_kernel::names::Name;
use pipit_kernel::order_by::OrderByOp;
use pipit_kernel::plan::SortKey;
use pipit_kernel::plan::{DynOp, FilterOp, NamedColumn, PLAN_COLUMNS_MAX};
use pipit_kernel::slow_vec::SlowVec;

use crate::ast::{Node, Operator, Tag};
use crate::compile::{Compiler, Relation};
use crate::error::{Error, ErrorCode, Unsupported};
use crate::expression::{add_expr, compile_expr, is_null};
use crate::keywords::same_word;
use crate::parser::LIST_MAX;
use crate::registry::{Item, Point, Rule, Shared};

pub const FROM: Rule = Rule {
    keyword: "from",
    point: Point::Source,
    items: &[Item::One(Shared::Name), Item::Clause(&["as"], Shared::Name)],
    compile: compile_from,
};

pub const WHERE: Rule = Rule {
    keyword: "where",
    point: Point::Stage,
    items: &[Item::One(Shared::Expr)],
    compile: compile_where,
};

pub const SELECT: Rule = Rule {
    keyword: "select",
    point: Point::Stage,
    items: &[Item::List(Shared::Aliased)],
    compile: compile_select,
};

pub const EXTEND: Rule = Rule {
    keyword: "extend",
    point: Point::Stage,
    items: &[Item::List(Shared::Aliased)],
    compile: compile_extend,
};

pub const DROP: Rule = Rule {
    keyword: "drop",
    point: Point::Stage,
    items: &[Item::List(Shared::Name)],
    compile: compile_drop,
};

pub const RENAME: Rule = Rule {
    keyword: "rename",
    point: Point::Stage,
    items: &[Item::List(Shared::Renamed)],
    compile: compile_rename,
};

pub const SET: Rule = Rule {
    keyword: "set",
    point: Point::Stage,
    items: &[Item::List(Shared::Assigned)],
    compile: compile_set,
};

pub const LIMIT: Rule = Rule {
    keyword: "limit",
    point: Point::Stage,
    items: &[Item::One(Shared::Expr), Item::Clause(&["offset"], Shared::Expr)],
    compile: compile_limit,
};

pub const AGGREGATE: Rule = Rule {
    keyword: "aggregate",
    point: Point::Stage,
    items: &[Item::List(Shared::Aliased), Item::Clause(&["group", "by"], Shared::Aliased)],
    compile: compile_aggregate,
};

pub const ORDER_BY: Rule = Rule {
    keyword: "order",
    point: Point::Stage,
    items: &[Item::After(&["by"], Shared::Ordered)],
    compile: compile_order_by,
};

pub const AS: Rule = Rule {
    keyword: "as",
    point: Point::Stage,
    items: &[Item::One(Shared::Name)],
    compile: compile_as,
};

pub const JOIN: Rule = Rule {
    keyword: "join",
    point: Point::Stage,
    items: &[
        Item::One(Shared::Name),
        Item::Clause(&["as"], Shared::Name),
        Item::Clause(&["on"], Shared::Expr),
        Item::Parenthesized(&["using"], Shared::Name),
    ],
    compile: compile_inner_join,
};

pub const INNER_JOIN: Rule = Rule {
    keyword: "inner",
    point: Point::Stage,
    items: &[
        Item::After(&["join"], Shared::Name),
        Item::Clause(&["as"], Shared::Name),
        Item::Clause(&["on"], Shared::Expr),
        Item::Parenthesized(&["using"], Shared::Name),
    ],
    compile: compile_inner_join,
};

pub const LEFT_JOIN: Rule = Rule {
    keyword: "left",
    point: Point::Stage,
    items: &[
        Item::Optional("outer"),
        Item::After(&["join"], Shared::Name),
        Item::Clause(&["as"], Shared::Name),
        Item::Clause(&["on"], Shared::Expr),
        Item::Parenthesized(&["using"], Shared::Name),
    ],
    compile: compile_left_join,
};

pub const CROSS_JOIN: Rule = Rule {
    keyword: "cross",
    point: Point::Stage,
    items: &[Item::After(&["join"], Shared::Name), Item::Clause(&["as"], Shared::Name)],
    compile: compile_cross_join,
};

/// Selecting and filtering rows.
pub const RELATIONAL: &[Rule] =
    &[FROM, WHERE, SELECT, EXTEND, DROP, RENAME, SET, ORDER_BY, LIMIT, AS];

/// Aggregating rows, apart, so a registry without it doesn't pay for it.
pub const AGGREGATION: &[Rule] = &[AGGREGATE];

/// Joining tables, apart, so a registry without it doesn't pay for it.
pub const JOINS: &[Rule] = &[JOIN, INNER_JOIN, LEFT_JOIN, CROSS_JOIN];

/// The aggregate functions, by name.
const FUNCTIONS: [(&str, Function); 5] = [
    ("count", Function::Count),
    ("sum", Function::Sum),
    ("min", Function::Min),
    ("max", Function::Max),
    ("avg", Function::Avg),
];

/// The aggregate function called `name`, ignoring case, and its name.
pub(crate) fn aggregate(name: &[u8]) -> Option<(&'static str, Function)> {
    FUNCTIONS.iter().find(|(f, _)| same_word(f.as_bytes(), name)).copied()
}

/// `FROM t [AS a]`: scans the table `t`, whose columns are then in scope,
/// and named by alias `a`, or else `t`.
fn compile_from(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let table = compiler.node(stage.first_child());
    let relation = compiler.compile_relation(table)?;
    let alias = alias_or(compiler, compiler.node(stage.first_child() + 1), table)?;
    compiler.add_alias(alias, &relation.columns)?;
    compiler.scope = relation.columns;
    Ok(())
}

/// `AS a`: the columns in scope, named by alias `a`, the only one.
fn compile_as(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    compiler.clear_aliases();
    let mut scope = SlowVec::new(compiler.allocator(), PLAN_COLUMNS_MAX)?;
    scope.extend_from_slice(&compiler.scope)?;
    compiler.add_alias(compiler.node(stage.first_child()), &scope)
}

/// The alias in the list `alias`, `AS`'s, or `table` if it's empty.
fn alias_or(compiler: &Compiler<'_, '_>, alias: Node, table: Node) -> Result<Node, Error> {
    if alias.child_count() == 0 { Ok(table) } else { only(compiler, alias) }
}

/// The condition `node`, `ON`'s or `WHERE`'s: an integer, true where it
/// isn't 0, as a comparison is.
fn condition_expr(compiler: &mut Compiler<'_, '_>, node: Node) -> Result<Expr, Error> {
    let (condition, data_type) = compile_expr(compiler, node)?;
    match data_type {
        DataType::Int64 => Ok(condition),
        DataType::String => Err(Error::new(ErrorCode::NumberCondition, compiler.span(node))),
        DataType::Float64 => {
            Err(Error::unsupported(Unsupported::DoubleCondition, compiler.span(node)))
        }
    }
}

/// `WHERE c`: keeps the rows condition `c` is true for.
fn compile_where(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let condition = condition_expr(compiler, compiler.node(stage.first_child()))?;
    let allocator = compiler.allocator();
    let filter = DynOp::new(allocator, FilterOp::new(allocator, &compiler.plan.exprs, condition)?)?;
    let children = [compiler.plan.root];
    compiler.plan.add_node(filter, &children)?;
    Ok(())
}

/// `[INNER] JOIN t ON c`: each pair of a row so far and a row of table `t`
/// that condition `c` is true for, with the columns of both. Or `USING (k)`,
/// as `ON k = t.k`, with `k` once.
fn compile_inner_join(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let first = stage.first_child();
    let condition = (compiler.node(first + 2), compiler.node(first + 3));
    let (table, alias) = (compiler.node(first), compiler.node(first + 1));
    join(compiler, table, alias, JoinKind::Inner, Some(condition))
}

/// `LEFT [OUTER] JOIN t ON c`: as `JOIN`, and each row so far that `c` is
/// true for with no row of `t`, with nulls for `t`'s columns.
fn compile_left_join(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let first = stage.first_child();
    let condition = (compiler.node(first + 3), compiler.node(first + 4));
    let (table, alias) = (compiler.node(first + 1), compiler.node(first + 2));
    join(compiler, table, alias, JoinKind::Left, Some(condition))
}

/// `CROSS JOIN t`: each pair of a row so far and a row of table `t`.
fn compile_cross_join(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let first = stage.first_child();
    join(compiler, compiler.node(first), compiler.node(first + 1), JoinKind::Inner, None)
}

/// Joins the rows so far to table `table`, a name or a list of one, named by
/// the alias in the list `alias` or else its own name, of `kind`, on a
/// condition given as the lists of an `ON` and a `USING`, one of them
/// empty, or on none. The scope is `USING`'s columns, then the rest of both
/// sides', so another name in both is ambiguous.
fn join(
    compiler: &mut Compiler<'_, '_>,
    mut table: Node,
    alias: Node,
    kind: JoinKind,
    condition: Option<(Node, Node)>,
) -> Result<(), Error> {
    let probe = compiler.plan.root;
    let allocator = compiler.allocator();
    let probe_columns = SlowVec::fixed_from(allocator, compiler.scope.iter().map(|c| c.id))?;
    if table.tag() == Tag::List {
        table = only(compiler, table)?;
    }
    let build = compiler.compile_relation(table)?;
    compiler.add_alias(alias_or(compiler, alias, table)?, &build.columns)?;
    if kind == JoinKind::Left {
        for column in build.columns.iter() {
            at_mut!(compiler.plan.columns, column.id as usize).nullable = true;
        }
    }
    let condition = match condition {
        Some((on, using)) if on.child_count() == 0 && using.child_count() == 0 => {
            return Err(Error::new(ErrorCode::ExpectedWord, compiler.span(table)));
        }
        Some((on, using)) if on.child_count() != 0 && using.child_count() != 0 => {
            return Err(Error::new(ErrorCode::UnexpectedToken, compiler.span(using)));
        }
        Some((_, using)) if using.child_count() != 0 => {
            Some(using_columns(compiler, &build, using)?)
        }
        Some((on, _)) => {
            compiler.scope.extend_from_slice(&build.columns)?;
            Some(condition_expr(compiler, only(compiler, on)?)?)
        }
        None => {
            compiler.scope.extend_from_slice(&build.columns)?;
            None
        }
    };
    let allocator = compiler.allocator();
    let build_columns = SlowVec::fixed_from(allocator, build.columns.iter().map(|c| c.id))?;
    let columns = [probe_columns, build_columns];
    let op = DynOp::new(allocator, JoinOp::new(kind, condition, columns))?;
    let children = [probe, build.node];
    compiler.plan.add_node(op, &children)?;
    Ok(())
}

/// `USING`'s condition, that each column of the list `using` is equal on
/// both sides, and its scope: those columns, the rows so far's, then the
/// rest of `build`'s.
fn using_columns(
    compiler: &mut Compiler<'_, '_>,
    build: &Relation,
    using: Node,
) -> Result<Expr, Error> {
    let allocator = compiler.allocator();
    let mut nodes = ExprBuilder::new(allocator)?;
    let mut scope = SlowVec::new(allocator, PLAN_COLUMNS_MAX)?;
    // Each build column joined on, one for each name in the list.
    let mut joined = [0; LIST_MAX];
    for i in 0..using.child_count() {
        let name = listed_once(compiler, using, i, 0)?;
        let (left, right) = (compiler.find_column(name)?, compiler.find_in(&build.columns, name)?);
        let before = (i > 0).then(|| nodes.last());
        let (a, b) = (nodes.column(left.id)?, nodes.column(right.id)?);
        let equal = nodes.node(ExprOp::Eq, a, b)?;
        if let Some(before) = before {
            nodes.node(ExprOp::And, before, equal)?;
        }
        scope.push(left)?;
        *at_mut!(joined, i as usize) = right.id;
    }
    // The rest of the rows so far's, even another name for a column joined
    // on, or a second copy of one.
    let joined_on = scope.len();
    for column in compiler.scope.iter() {
        let named = |joined: &NamedColumn| (joined.name, joined.id) == (column.name, column.id);
        if !at!(scope, ..joined_on).iter().any(named) {
            scope.push(*column)?;
        }
    }
    for column in build.columns.iter() {
        if !at!(joined, ..using.child_count() as usize).contains(&column.id) {
            scope.push(*column)?;
        }
    }
    let (expr, _) = add_expr(compiler, &nodes, using)?;
    compiler.scope = scope;
    Ok(expr)
}

/// The one item of `list`.
fn only(compiler: &Compiler<'_, '_>, list: Node) -> Result<Node, Error> {
    if list.child_count() != 1 {
        let second = compiler.node(list.first_child() + 1);
        return Err(Error::new(ErrorCode::UnexpectedToken, compiler.span(second)));
    }
    Ok(compiler.node(list.first_child()))
}

// These stages change which columns are in scope and what they're called,
// as Perfetto's. An item which is just a column runs nothing; any other
// expression is added to a `ComputeOp`.

/// `SELECT a, b + 1 AS c`: the items, in that order, are the new scope.
fn compile_select(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    project(compiler, stage, false)
}

/// `EXTEND a * 2 AS c`: adds the items to the scope. Each sees only the
/// scope before the stage.
fn compile_extend(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    project(compiler, stage, true)
}

/// The column of each item of `stage`'s list, under its alias if it has
/// one, after the scope's columns if `extend`, are the new scope.
fn project(compiler: &mut Compiler<'_, '_>, stage: Node, extend: bool) -> Result<(), Error> {
    let mut scope = SlowVec::new(compiler.allocator(), PLAN_COLUMNS_MAX)?;
    if extend {
        scope.extend_from_slice(&compiler.scope)?;
    }
    let list = compiler.node(stage.first_child());
    let mut computed = SlowVec::fixed(compiler.allocator(), list.child_count() as usize)?;
    for i in 0..list.child_count() {
        let (value, alias) = unalias(compiler, compiler.node(list.first_child() + i));
        let name = match alias {
            Some(alias) => Some(compiler.plan.names.add(compiler.text(alias.span()))?),
            None => None,
        };
        scope.push(value_of(compiler, value, name, &mut computed)?)?;
    }
    add_compute(compiler, computed)?;
    compiler.scope = scope;
    // A new table, as GoogleSQL's `SELECT` makes, which no alias names.
    if !extend {
        compiler.clear_aliases();
    }
    Ok(())
}

/// `item`, and its name if it's a `Named`.
fn unalias(compiler: &Compiler<'_, '_>, item: Node) -> (Node, Option<Node>) {
    if item.tag() != Tag::Named {
        return (item, None);
    }
    (compiler.node(item.first_child()), Some(compiler.node(item.first_child() + 1)))
}

/// `column`, under the name `alias` if there is one.
fn named(
    compiler: &mut Compiler<'_, '_>,
    column: NamedColumn,
    alias: Option<Node>,
) -> Result<NamedColumn, Error> {
    let Some(alias) = alias else { return Ok(column) };
    let name = compiler.plan.names.add(compiler.text(alias.span()))?;
    Ok(NamedColumn { name, id: column.id })
}

/// `DROP a, b`: removes every column named `a` or `b`. One must be left.
fn compile_drop(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let list = compiler.node(stage.first_child());
    let mut scope = SlowVec::new(compiler.allocator(), PLAN_COLUMNS_MAX)?;
    scope.extend_from_slice(&compiler.scope)?;
    for i in 0..list.child_count() {
        let name = listed_once(compiler, list, i, 0)?;
        let text = compiler.text(name.span()).as_bytes();
        let kept = |column: &NamedColumn| {
            !same_word(compiler.plan.names.get(column.name).as_bytes(), text)
        };
        if scope.iter().all(kept) {
            return Err(Error::new(ErrorCode::UnknownColumn, name.span()));
        }
        scope.retain(kept);
    }
    if scope.is_empty() {
        let first = compiler.node(list.first_child());
        return Err(Error::new(ErrorCode::NoColumnsLeft, first.span()));
    }
    compiler.scope = scope;
    Ok(())
}

/// `RENAME a AS b, b AS a`: renames columns in place, all at once.
fn compile_rename(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    change_in_place(compiler, stage, true)
}

/// `SET a = b + 1`: replaces the values of columns in place, all at once,
/// computed from the scope before the stage.
fn compile_set(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    change_in_place(compiler, stage, false)
}

/// Renames, or sets the value of, the column each item of `stage`'s list
/// names, in a copy of the scope that replaces it once every item is done,
/// so each item sees the scope before the stage. `RENAME a AS b` names the
/// column first, and `SET a = b` second.
fn change_in_place(
    compiler: &mut Compiler<'_, '_>,
    stage: Node,
    renames: bool,
) -> Result<(), Error> {
    let list = compiler.node(stage.first_child());
    let mut scope = SlowVec::new(compiler.allocator(), PLAN_COLUMNS_MAX)?;
    scope.extend_from_slice(&compiler.scope)?;
    let mut computed = SlowVec::fixed(compiler.allocator(), list.child_count() as usize)?;
    let target = u32::from(!renames);
    for i in 0..list.child_count() {
        let found = compiler.find_column(listed_once(compiler, list, i, target)?)?;
        let Some(at) = compiler.scope.iter().position(|column| column.name == found.name) else {
            pipit_kernel::failed!()
        };
        let item = compiler.node(list.first_child() + i);
        let column = at_mut!(scope, at);
        if renames {
            let alias = compiler.node(item.first_child() + 1);
            column.name = compiler.plan.names.add(compiler.text(alias.span()))?;
        } else {
            let value = compiler.node(item.first_child());
            column.id = value_of(compiler, value, Some(found.name), &mut computed)?.id;
        }
    }
    add_compute(compiler, computed)?;
    compiler.scope = scope;
    Ok(())
}

/// The name item `i` of `list` lists: the item itself, as in `DROP`, or its
/// child `target`, as in `RENAME` and `SET`. Fails if an item before it lists
/// it too, ignoring case.
#[inline(never)]
fn listed_once(
    compiler: &Compiler<'_, '_>,
    list: Node,
    i: u32,
    target: u32,
) -> Result<Node, Error> {
    let name = |i: u32| {
        let item = compiler.node(list.first_child() + i);
        if item.tag() == Tag::Named { compiler.node(item.first_child() + target) } else { item }
    };
    let text = compiler.text(name(i).span()).as_bytes();
    if (0..i).any(|j| same_word(compiler.text(name(j).span()).as_bytes(), text)) {
        return Err(Error::new(ErrorCode::ListedTwice, name(i).span()));
    }
    Ok(name(i))
}

/// The column `value` names, under `name` if given; or else a new column
/// named `name`, which `value` computes, added to `computed`.
fn value_of(
    compiler: &mut Compiler<'_, '_>,
    value: Node,
    name: Option<Name>,
    computed: &mut SlowVec<ComputeItem>,
) -> Result<NamedColumn, Error> {
    if value.tag() == Tag::Name && !is_null(compiler, value) {
        let column = compiler.find_column(value)?;
        return Ok(NamedColumn { name: name.unwrap_or(column.name), id: column.id });
    }
    let Some(name) = name else {
        return Err(Error::new(ErrorCode::ExpectedName, compiler.span(value)));
    };
    let (expr, data_type) = compile_expr(compiler, value)?;
    let (columns, nodes) = (&compiler.plan.columns, compiler.plan.exprs.nodes(expr));
    let nullable = expr_can_be_null(nodes, &|column| at!(columns, column as usize).nullable);
    let column = compiler.plan.add_named_column(name, data_type, nullable)?;
    computed.push(ComputeItem { output: column.id, expr })?;
    Ok(column)
}

/// Adds the `ComputeOp` for a stage's `computed` columns, if it has any.
fn add_compute(
    compiler: &mut Compiler<'_, '_>,
    computed: SlowVec<ComputeItem>,
) -> Result<(), Error> {
    if computed.is_empty() {
        return Ok(());
    }
    let allocator = compiler.allocator();
    let compute = DynOp::new(allocator, ComputeOp { items: computed })?;
    let children = [compiler.plan.root];
    compiler.plan.add_node(compute, &children)?;
    Ok(())
}

/// `ORDER BY a, b DESC`: the rows sorted by `a`, then `b` descending,
/// stably. Keys are columns.
fn compile_order_by(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let list = compiler.node(stage.first_child());
    let allocator = compiler.allocator();
    let mut keys = SlowVec::fixed(allocator, list.child_count() as usize)?;
    for i in 0..list.child_count() {
        let key = compiler.node(list.first_child() + i);
        let column = compiler.node(key.first_child());
        let unsupported = Error::unsupported(Unsupported::OrderBy, compiler.span(column));
        if column.tag() != Tag::Name {
            return Err(unsupported);
        }
        let column = compiler.find_column(column)?.id;
        keys.push(SortKey { column, descending: key.descending() })?;
    }
    let order_by = DynOp::new(allocator, OrderByOp { keys })?;
    let children = [compiler.plan.root];
    compiler.plan.add_node(order_by, &children)?;
    Ok(())
}

/// `LIMIT 10 OFFSET 20`: the first 10 rows after skipping 20. Both are
/// integers.
fn compile_limit(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let count = count(compiler, compiler.node(stage.first_child()))?;
    let offsets = compiler.node(stage.first_child() + 1);
    let offset = match offsets.child_count() {
        0 => 0,
        1 => self::count(compiler, compiler.node(offsets.first_child()))?,
        _ => {
            let second = compiler.node(offsets.first_child() + 1);
            return Err(Error::unsupported(Unsupported::Limit, compiler.span(second)));
        }
    };
    let allocator = compiler.allocator();
    let limit = DynOp::new(allocator, LimitOp { count, offset })?;
    let children = [compiler.plan.root];
    compiler.plan.add_node(limit, &children)?;
    Ok(())
}

/// The number of rows `node`, an integer, says.
fn count(compiler: &Compiler<'_, '_>, node: Node) -> Result<u64, Error> {
    if node.tag() != Tag::Integer {
        return Err(Error::unsupported(Unsupported::Limit, compiler.span(node)));
    }
    compiler.integer(node)
}

/// `AGGREGATE COUNT(*), SUM(a) GROUP BY k`: a row for each group of rows
/// with the same keys, or one for all rows with none, is the new scope: the
/// keys, then the aggregates, named as `count(*)` and `sum(a)`.
fn compile_aggregate(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let list = compiler.node(stage.first_child());
    let by = compiler.node(stage.first_child() + 1);
    let allocator = compiler.allocator();
    let mut keys = SlowVec::fixed(allocator, by.child_count() as usize)?;
    let mut scope = SlowVec::new(allocator, PLAN_COLUMNS_MAX)?;
    for i in 0..by.child_count() {
        let (key, alias) = unalias(compiler, compiler.node(by.first_child() + i));
        if key.tag() != Tag::Name {
            return Err(Error::unsupported(Unsupported::GroupKey, compiler.span(key)));
        }
        let column = compiler.find_column(key)?;
        let data_type = at!(compiler.plan.columns, column.id as usize).data_type;
        if !matches!(data_type, DataType::Int64 | DataType::String) {
            return Err(Error::unsupported(Unsupported::GroupKey, compiler.span(key)));
        }
        keys.push(column.id)?;
        scope.push(named(compiler, column, alias)?)?;
    }
    let mut aggregates = SlowVec::fixed(allocator, list.child_count() as usize)?;
    for i in 0..list.child_count() {
        let (item, alias) = unalias(compiler, compiler.node(list.first_child() + i));
        let span = compiler.span(item);
        let unsupported = Error::unsupported(Unsupported::Aggregate, span);
        if item.tag() != Tag::Call || item.child_count() != 2 {
            return Err(unsupported);
        }
        let name = compiler.node(item.first_child()).span();
        let text = compiler.text(name);
        let Some((function_name, function)) = aggregate(text.as_bytes()) else {
            return Err(Error::new(ErrorCode::UnknownFunction, name));
        };
        let mut argument = compiler.node(item.first_child() + 1);
        let distinct = argument.tag() == Tag::Unary && argument.operator() == Operator::Distinct;
        if distinct {
            argument = compiler.node(argument.first_child());
        }
        let (function, input) = match argument.tag() {
            Tag::Star if function == Function::Count && !distinct => (Function::CountStar, None),
            Tag::Name => (function, Some(compiler.find_column(argument)?.id)),
            _ => return Err(unsupported),
        };
        let input_type =
            input.map_or(DataType::Int64, |id| at!(compiler.plan.columns, id as usize).data_type);
        if input.is_some() && !function.reads(input_type) {
            return Err(Error::unsupported(Unsupported::AggregateType, span));
        }
        let argument = if input.is_some() { compiler.text(argument.span()) } else { "*" };
        let name = match alias {
            Some(alias) => compiler.plan.names.add(compiler.text(alias.span()))?,
            None => aggregate_name(compiler, function_name, distinct, argument)?,
        };
        // A count is 0 for a group of nulls, where the others are null.
        let nullable = !matches!(function, Function::CountStar | Function::Count);
        let output_type = function.output_type(input_type);
        let binding = compiler.plan.add_named_column(name, output_type, nullable)?;
        // The least or greatest of the distinct values is that of them all.
        let distinct = distinct && !matches!(function, Function::Min | Function::Max);
        aggregates.push(Aggregate { function, input, distinct, binding })?;
        scope.push(binding)?;
    }
    let aggregate = DynOp::new(allocator, AggregateOp { keys, aggregates, answer: None })?;
    let children = [compiler.plan.root];
    compiler.plan.add_node(aggregate, &children)?;
    compiler.scope = scope;
    compiler.clear_aliases();
    Ok(())
}

/// The name of an aggregate without an alias, such as `count(DISTINCT a)`.
fn aggregate_name(
    compiler: &mut Compiler<'_, '_>,
    function: &str,
    distinct: bool,
    argument: &str,
) -> Result<Name, Error> {
    let distinct = if distinct { "DISTINCT " } else { "" };
    Ok(compiler.plan.names.add_parts(&[function, "(", distinct, argument, ")"])?)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::format;
    use std::string::{String, ToString};
    use std::vec::Vec;

    use crate::ast::{Ast, Tag};
    use crate::error::ErrorCode;
    use crate::parser::parse_query;
    use crate::registry::Registry;

    use super::*;
    use pipit_kernel::global::Global;

    static REGISTRY: Registry = Registry::new(&[RELATIONAL, AGGREGATION]);

    /// Renders the tree as an s-expression.
    fn render(source: &str, ast: &Ast, index: u32) -> String {
        let node = ast.node(index);
        let (head, count) = match node.tag() {
            Tag::Query => ("Query".to_string(), node.child_count()),
            Tag::List => ("List".to_string(), node.child_count()),
            Tag::Call => ("Call".to_string(), node.child_count()),
            Tag::Case => ("Case".to_string(), node.child_count()),
            Tag::Stage => {
                let rule = REGISTRY.rule(node.rule());
                (rule.keyword.to_uppercase(), u32::try_from(rule.items.len()).unwrap())
            }
            Tag::Named => ("Named".to_string(), 2),
            Tag::SortKey => ((if node.descending() { "DESC" } else { "ASC" }).to_string(), 1),
            Tag::Unary => (format!("{:?}", node.operator()), 1),
            Tag::Binary => (format!("{:?}", node.operator()), 2),
            _ => {
                let span = node.span();
                return source[span.start as usize..(span.start + span.len) as usize].to_string();
            }
        };
        let children: Vec<String> =
            (0..count).map(|i| render(source, ast, node.first_child() + i)).collect();
        format!("({head} {})", children.join(" "))
    }

    fn parse(source: &str) -> String {
        let global = Global::current();
        let ast = parse_query(global, &REGISTRY, source.as_bytes()).unwrap();
        render(source, &ast, ast.root())
    }

    fn error(source: &str) -> (ErrorCode, u32) {
        let global = Global::current();
        let error = parse_query(global, &REGISTRY, source.as_bytes()).unwrap_err();
        (error.code, error.span.start)
    }

    #[test]
    fn parses_queries() {
        assert_eq!(parse("from t"), "(Query (FROM t (List )))");
        assert_eq!(
            parse("FROM slice |> WHERE dur > 1000 |> SELECT name, dur / 1000, count(*)"),
            "(Query (FROM slice (List )) (WHERE (Greater dur 1000)) \
             (SELECT (List name (Divide dur 1000) (Call count *))))"
        );
        assert_eq!(
            parse("FROM t |> WHERE url LIKE '%x%' AND a = 1"),
            "(Query (FROM t (List )) (WHERE (And (Like url '%x%') (Equal a 1))))"
        );
        assert_eq!(
            parse("FROM t |> WHERE a IS NULL OR NOT b is not null"),
            "(Query (FROM t (List )) (WHERE (Or (IsNull a) (Not (IsNotNull b)))))"
        );
        assert_eq!(
            parse("FROM t |> LIMIT 10 offset 5"),
            "(Query (FROM t (List )) (LIMIT 10 (List 5)))"
        );
        // A qualified name is one name, spaces and all.
        assert_eq!(
            parse("FROM t |> WHERE t.a > u . b"),
            "(Query (FROM t (List )) (WHERE (Greater t.a u . b)))"
        );
        assert_eq!(parse("FROM t |> LIMIT 1"), "(Query (FROM t (List )) (LIMIT 1 (List )))");
        assert_eq!(
            parse("FROM t |> ORDER BY a, b + 1 DESC |> LIMIT 10"),
            "(Query (FROM t (List )) (ORDER (List (ASC a) (DESC (Add b 1)))) (LIMIT 10 (List )))"
        );
        assert_eq!(
            parse("FROM t |> SELECT a AS b, c |> RENAME b as d |> SET a = b"),
            "(Query (FROM t (List )) (SELECT (List (Named a b) c)) (RENAME (List (Named b d))) \
             (SET (List (Named b a))))"
        );
        assert_eq!(
            parse("FROM t |> AGGREGATE count(*), sum(x) group by a, b"),
            "(Query (FROM t (List )) (AGGREGATE (List (Call count *) (Call sum x)) (List a b)))"
        );
        assert_eq!(
            parse("FROM t |> AGGREGATE count(*)"),
            "(Query (FROM t (List )) (AGGREGATE (List (Call count *)) (List )))"
        );
        assert_eq!(
            parse("FROM t |> AGGREGATE count(*) AS n GROUP BY a AS k"),
            "(Query (FROM t (List )) (AGGREGATE (List (Named (Call count *) n)) (List (Named a k))))"
        );
    }

    #[test]
    fn reports_errors() {
        assert_eq!(error("WHERE x"), (ErrorCode::ExpectedSource, 0));
        assert_eq!(error("FROM t |> FROM u"), (ErrorCode::UnexpectedSource, 10));
        assert_eq!(error("FROM t |> SORT x"), (ErrorCode::UnknownStage, 10));
        assert_eq!(error("FROM t WHERE x"), (ErrorCode::UnexpectedToken, 7));
        assert_eq!(error("FROM t |> SELECT"), (ErrorCode::ExpectedExpression, 16));
        assert_eq!(error("FROM t |> WHERE a IS 1"), (ErrorCode::ExpectedWord, 21));
        assert_eq!(error("FROM t |> RENAME a b"), (ErrorCode::ExpectedAs, 19));
        assert_eq!(error("FROM t |> SET a b"), (ErrorCode::ExpectedToken, 16));
        assert_eq!(error("FROM t |> AGGREGATE count(*) GROUP a"), (ErrorCode::ExpectedWord, 35));
        assert_eq!(
            error("FROM t |> AGGREGATE count(*) GROUP BY"),
            (ErrorCode::ExpectedExpression, 37)
        );
        assert_eq!(error("FROM t |> ORDER a"), (ErrorCode::ExpectedWord, 16));
    }
}
