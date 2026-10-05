//! The stages PipeSQL comes with, as rules for a registry. They use the same
//! API as any extension.

use pipit_kernel::aggregate::{Aggregate, AggregateOp, Function};
use pipit_kernel::plan::{DynOp, FilterOp, PLAN_COLUMNS_MAX, ScanColumn, ScanOp};
use pipit_kernel::slow_vec::SlowVec;

use crate::ast::{Node, Tag};
use crate::compile::Compiler;
use crate::condition::compile_condition;
use crate::error::{Error, ErrorCode, Unsupported};
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
    items: &[Item::List(Shared::Expr)],
    compile: compile_select,
};

pub const AGGREGATE: Rule = Rule {
    keyword: "aggregate",
    point: Point::Stage,
    items: &[Item::List(Shared::Expr)],
    compile: compile_aggregate,
};

/// Selecting, filtering and aggregating rows.
pub const RELATIONAL: &[Rule] = &[FROM, WHERE, SELECT, AGGREGATE];

/// The aggregate functions, by name.
const FUNCTIONS: [(&str, Function); 5] = [
    ("count", Function::Count),
    ("sum", Function::Sum),
    ("min", Function::Min),
    ("max", Function::Max),
    ("avg", Function::Avg),
];

/// `FROM t`: scans the table `t`, whose columns are then in scope.
fn compile_from(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let name = compiler.node(stage.first_child()).span();
    let Some(table) = compiler.catalog().find(compiler.text(name)) else {
        return Err(Error::new(ErrorCode::UnknownTable, name));
    };
    let allocator = compiler.allocator();
    let count = table.column_count();
    let mut columns = SlowVec::fixed(allocator, count as usize)?;
    let mut scope = SlowVec::new(allocator, PLAN_COLUMNS_MAX)?;
    for i in 0..count {
        let binding = compiler.plan.add_column(table.column_name(i), table.column_type(i))?;
        columns.push(ScanColumn { column: i, binding })?;
        scope.push(binding)?;
    }
    let scan = DynOp::new(allocator, ScanOp { scannable: table, columns })?;
    let children = SlowVec::fixed(allocator, 0)?;
    compiler.plan.add_node(scan, children)?;
    compiler.scope = scope;
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

/// `SELECT a, b`: the named columns, in that order, are the new scope. Only
/// names, for now.
fn compile_select(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let list = compiler.node(stage.first_child());
    let mut scope = SlowVec::new(compiler.allocator(), PLAN_COLUMNS_MAX)?;
    for i in 0..list.child_count() {
        let item = compiler.node(list.first_child() + i);
        if item.tag() != Tag::Name {
            return Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(item)));
        }
        scope.push(compiler.find_column(item)?)?;
    }
    compiler.scope = scope;
    Ok(())
}

/// `AGGREGATE COUNT(*), SUM(a)`: one row of the aggregates of every row is
/// the new scope, named as `count(*)` and `sum(a)`.
fn compile_aggregate(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let list = compiler.node(stage.first_child());
    let allocator = compiler.allocator();
    let mut aggregates = SlowVec::fixed(allocator, list.child_count() as usize)?;
    let mut scope = SlowVec::new(allocator, PLAN_COLUMNS_MAX)?;
    for i in 0..list.child_count() {
        let item = compiler.node(list.first_child() + i);
        let span = compiler.span(item);
        let unsupported = Error::unsupported(Unsupported::Aggregate, span);
        if item.tag() != Tag::Call || item.child_count() != 2 {
            return Err(unsupported);
        }
        let name = compiler.node(item.first_child()).span();
        let text = compiler.text(name);
        let Some(&(function_name, function)) =
            FUNCTIONS.iter().find(|(f, _)| f.eq_ignore_ascii_case(text))
        else {
            return Err(Error::new(ErrorCode::UnknownFunction, name));
        };
        let argument = compiler.node(item.first_child() + 1);
        let input = match argument.tag() {
            Tag::Star if function == Function::Count => None,
            Tag::Name => Some(compiler.find_column(argument)?.id),
            _ => return Err(unsupported),
        };
        let input_type = input.map(|id| at!(compiler.plan.columns, id as usize).data_type);
        let Some(output_type) = function.output_type(input_type) else {
            return Err(Error::unsupported(Unsupported::AggregateType, span));
        };
        let argument = if input.is_some() { compiler.text(argument.span()) } else { "*" };
        let name = compiler.plan.names.add_parts(&[function_name, "(", argument, ")"])?;
        let binding = compiler.plan.add_named_column(name, output_type)?;
        aggregates.push(Aggregate { function, input, binding })?;
        scope.push(binding)?;
    }
    let aggregate = DynOp::new(allocator, AggregateOp { aggregates })?;
    let children = SlowVec::fixed_from(allocator, [compiler.plan.root].into_iter())?;
    compiler.plan.add_node(aggregate, children)?;
    compiler.scope = scope;
    Ok(())
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
    }

    #[test]
    fn reports_errors() {
        assert_eq!(error("WHERE x"), (ErrorCode::ExpectedSource, 0));
        assert_eq!(error("FROM t |> FROM u"), (ErrorCode::UnexpectedSource, 10));
        assert_eq!(error("FROM t |> ORDER x"), (ErrorCode::UnknownStage, 10));
        assert_eq!(error("FROM t WHERE x"), (ErrorCode::UnexpectedToken, 7));
        assert_eq!(error("FROM t |> SELECT"), (ErrorCode::ExpectedExpression, 16));
    }
}
