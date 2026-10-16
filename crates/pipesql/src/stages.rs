//! The stages PipeSQL comes with, as rules for a registry. They use the same
//! API as any extension.

use pipit_kernel::limit::LimitOp;
use pipit_kernel::plan::{DynOp, FilterOp, NamedColumn, PLAN_COLUMNS_MAX};
use pipit_kernel::slow_vec::SlowVec;

use crate::ast::{Node, Tag};
use crate::compile::Compiler;
use crate::condition::compile_condition;
use crate::error::{Error, ErrorCode, Unsupported};
use crate::keywords::same_word;
use crate::registry::{Item, Point, Rule, Shared};

pub const FROM: Rule = Rule {
    keyword: "from",
    point: Point::Source,
    items: &[Item::One(Shared::Name)],
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
    items: &[Item::One(Shared::Expr), Item::Clause("offset", Shared::Expr)],
    compile: compile_limit,
};

/// Selecting and filtering rows.
pub const RELATIONAL: &[Rule] = &[FROM, WHERE, SELECT, EXTEND, DROP, RENAME, SET, LIMIT];

/// `FROM t`: scans the table `t`, whose columns are then in scope.
fn compile_from(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let relation = compiler.compile_relation(compiler.node(stage.first_child()))?;
    compiler.scope = relation.columns;
    Ok(())
}

/// `WHERE c`: keeps the rows condition `c` is true for.
fn compile_where(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let predicate = compile_condition(compiler, compiler.node(stage.first_child()))?;
    let allocator = compiler.allocator();
    let filter = DynOp::new(allocator, FilterOp { predicate })?;
    let children = SlowVec::fixed_from(allocator, [compiler.plan.root].into_iter())?;
    compiler.plan.add_node(filter, children)?;
    Ok(())
}

// These stages only change which columns are in scope and what they're
// called, as Perfetto's: nothing runs. Where one could take an expression,
// it takes a column's name for now.

/// `SELECT a, b AS c`: the named columns, in that order, are the new scope.
fn compile_select(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    project(compiler, stage, false)
}

/// `EXTEND a AS c`: adds the named columns to the scope. Each is found in
/// the scope before the stage.
fn compile_extend(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    project(compiler, stage, true)
}

/// The column each item of `stage`'s list names, under its alias if it has
/// one, after the scope's columns if `extend`, are the new scope.
fn project(compiler: &mut Compiler<'_, '_>, stage: Node, extend: bool) -> Result<(), Error> {
    let mut scope = SlowVec::new(compiler.allocator(), PLAN_COLUMNS_MAX)?;
    if extend {
        scope.extend_from_slice(&compiler.scope)?;
    }
    let list = compiler.node(stage.first_child());
    for i in 0..list.child_count() {
        let item = compiler.node(list.first_child() + i);
        let (value, alias) = match item.tag() {
            Tag::Named => {
                (compiler.node(item.first_child()), Some(compiler.node(item.first_child() + 1)))
            }
            _ => (item, None),
        };
        let mut column = value_of(compiler, value)?;
        if let Some(alias) = alias {
            column.name = compiler.plan.names.add(compiler.text(alias.span()))?;
        }
        scope.push(column)?;
    }
    compiler.scope = scope;
    Ok(())
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

/// `SET a = b`: replaces the values of columns in place, all at once, as
/// they were before the stage.
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
            column.id = value_of(compiler, compiler.node(item.first_child()))?.id;
        }
    }
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

/// The column `value` names: only names, for now.
fn value_of(compiler: &Compiler<'_, '_>, value: Node) -> Result<NamedColumn, Error> {
    if value.tag() != Tag::Name {
        return Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(value)));
    }
    compiler.find_column(value)
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
    let children = SlowVec::fixed_from(allocator, [compiler.plan.root].into_iter())?;
    compiler.plan.add_node(limit, children)?;
    Ok(())
}

/// The number of rows `node`, an integer, says.
fn count(compiler: &Compiler<'_, '_>, node: Node) -> Result<u64, Error> {
    if node.tag() != Tag::Integer {
        return Err(Error::unsupported(Unsupported::Limit, compiler.span(node)));
    }
    compiler.integer(node)
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
    use pipit_kernel::allocator::Heap;

    use super::*;

    static REGISTRY: Registry = Registry::new(&[RELATIONAL]);

    /// Renders the tree as an s-expression.
    fn render(source: &str, ast: &Ast, index: u32) -> String {
        let node = ast.node(index);
        let (head, count) = match node.tag() {
            Tag::Query => ("Query".to_string(), node.child_count()),
            Tag::List => ("List".to_string(), node.child_count()),
            Tag::Call => ("Call".to_string(), node.child_count()),
            Tag::Stage => {
                let rule = REGISTRY.rule(node.rule());
                (rule.keyword.to_uppercase(), u32::try_from(rule.items.len()).unwrap())
            }
            Tag::Named => ("Named".to_string(), 2),
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
        let ast = parse_query(&Heap, &REGISTRY, source.as_bytes()).unwrap();
        render(source, &ast, ast.root())
    }

    fn error(source: &str) -> (ErrorCode, u32) {
        let error = parse_query(&Heap, &REGISTRY, source.as_bytes()).unwrap_err();
        (error.code, error.span.start)
    }

    #[test]
    fn parses_queries() {
        assert_eq!(parse("from t"), "(Query (FROM t))");
        assert_eq!(
            parse("FROM slice |> WHERE dur > 1000 |> SELECT name, dur / 1000, count(*)"),
            "(Query (FROM slice) (WHERE (Greater dur 1000)) \
             (SELECT (List name (Divide dur 1000) (Call count *))))"
        );
        assert_eq!(parse("FROM t |> LIMIT 10 offset 5"), "(Query (FROM t) (LIMIT 10 (List 5)))");
        assert_eq!(parse("FROM t |> LIMIT 1"), "(Query (FROM t) (LIMIT 1 (List )))");
        assert_eq!(
            parse("FROM t |> SELECT a AS b, c |> RENAME b as d |> SET a = b"),
            "(Query (FROM t) (SELECT (List (Named a b) c)) (RENAME (List (Named b d))) \
             (SET (List (Named b a))))"
        );
    }

    #[test]
    fn reports_errors() {
        assert_eq!(error("WHERE x"), (ErrorCode::ExpectedSource, 0));
        assert_eq!(error("FROM t |> FROM u"), (ErrorCode::UnexpectedSource, 10));
        assert_eq!(error("FROM t |> ORDER x"), (ErrorCode::UnknownStage, 10));
        assert_eq!(error("FROM t WHERE x"), (ErrorCode::UnexpectedToken, 7));
        assert_eq!(error("FROM t |> SELECT"), (ErrorCode::ExpectedExpression, 16));
        assert_eq!(error("FROM t |> RENAME a b"), (ErrorCode::ExpectedAs, 19));
        assert_eq!(error("FROM t |> SET a b"), (ErrorCode::ExpectedToken, 16));
    }
}
