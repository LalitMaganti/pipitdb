//! pipitdb's standard stages, as rules for the kernel's registry. They use the
//! same API as any extension.

#![no_std]

use pipit_kernel::registry::{Item, Point, Rule, Shared};

pub const FROM: Rule =
    Rule { keyword: "from", point: Point::Source, items: &[Item::One(Shared::Name)] };

pub const WHERE: Rule =
    Rule { keyword: "where", point: Point::Stage, items: &[Item::One(Shared::Expr)] };

pub const SELECT: Rule =
    Rule { keyword: "select", point: Point::Stage, items: &[Item::List(Shared::Expr)] };

/// Selecting and filtering rows.
pub const RELATIONAL: &[Rule] = &[FROM, WHERE, SELECT];

#[cfg(test)]
mod tests {
    extern crate std;

    use std::format;
    use std::string::{String, ToString};
    use std::vec::Vec;

    use pipit_kernel::allocator::Heap;
    use pipit_kernel::ast::{Ast, Tag};
    use pipit_kernel::error::ErrorCode;
    use pipit_kernel::parser::parse_query;
    use pipit_kernel::registry::Registry;

    use super::*;

    static REGISTRY: Registry = Registry::new(&[RELATIONAL]);

    /// Renders the tree as an s-expression.
    fn render(ast: &Ast, source: &str, index: u32) -> String {
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
            (0..count).map(|i| render(ast, source, node.first_child() + i)).collect();
        format!("({head} {})", children.join(" "))
    }

    fn parse(source: &str) -> String {
        let ast = parse_query(Heap, &REGISTRY, source.as_bytes()).unwrap();
        render(&ast, source, ast.root())
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
