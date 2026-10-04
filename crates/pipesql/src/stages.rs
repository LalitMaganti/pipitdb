//! The stages PipeSQL comes with, as rules for a registry. They use the same
//! API as any extension.

use pipit_kernel::plan::{DynOp, PLAN_COLUMNS_MAX, ScanColumn, ScanOp};
use pipit_kernel::vec::Vec;

use crate::ast::{Node, Tag};
use crate::compile::Compiler;
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

/// Selecting and filtering rows.
pub const RELATIONAL: &[Rule] = &[FROM, WHERE, SELECT];

/// `FROM t`: scans the table `t`, whose columns are then in scope.
fn compile_from(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let name = compiler.node(stage.first_child()).span();
    let Some(table) = compiler.catalog().find(compiler.text(name)) else {
        return Err(Error::new(ErrorCode::UnknownTable, name));
    };
    let allocator = compiler.allocator();
    let count = table.column_count();
    let mut columns = Vec::fixed(allocator.clone(), count as usize)?;
    let mut scope = Vec::new(allocator.clone(), PLAN_COLUMNS_MAX)?;
    for i in 0..count {
        let binding = compiler.plan.add_column(table.column_name(i), table.column_type(i))?;
        columns.push(ScanColumn { column: i, binding })?;
        scope.push(binding)?;
    }
    let scan = DynOp::new(allocator.clone(), ScanOp { scannable: table, columns })?;
    let children = Vec::fixed(allocator, 0)?;
    compiler.plan.add_node(scan, children)?;
    compiler.scope = scope;
    Ok(())
}

/// `WHERE`: needs expressions, which can't be compiled yet.
fn compile_where(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let span = compiler.span(compiler.node(stage.first_child()));
    Err(Error::unsupported(Unsupported::Where, span))
}

/// `SELECT a, b`: the named columns, in that order, are the new scope. Only
/// names, for now.
fn compile_select(compiler: &mut Compiler<'_, '_>, stage: Node) -> Result<(), Error> {
    let list = compiler.node(stage.first_child());
    let mut scope = Vec::new(compiler.allocator(), PLAN_COLUMNS_MAX)?;
    for i in 0..list.child_count() {
        let item = compiler.node(list.first_child() + i);
        if item.tag() != Tag::Name {
            return Err(Error::unsupported(Unsupported::SelectExpression, compiler.span(item)));
        }
        let name = compiler.text(item.span());
        let plan = &compiler.plan;
        let found = compiler.scope.iter().find(|column| plan.names.get(column.name) == name);
        let Some(&column) = found else {
            return Err(Error::new(ErrorCode::UnknownColumn, item.span()));
        };
        scope.push(column)?;
    }
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
        let ast = parse_query(Heap, &REGISTRY, source.as_bytes()).unwrap();
        render(source, &ast, ast.root())
    }

    fn error(source: &str) -> (ErrorCode, u32) {
        let error = parse_query(Heap, &REGISTRY, source.as_bytes()).unwrap_err();
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
