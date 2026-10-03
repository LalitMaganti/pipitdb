//! Parses query text into an `Ast`.

mod expression;

use crate::allocator::Allocator;
use crate::ast::Ast;
use crate::error::Error;

/// How deeply operators and parentheses can nest.
pub const NESTING_MAX: usize = 64;

/// Parses `source` as a single expression.
pub fn parse_expression<A: Allocator + 'static>(allocator: A, source: &[u8]) -> Result<Ast, Error> {
    expression::parse(allocator, source)
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::String;

    use super::*;
    use crate::allocator::Heap;
    use crate::ast::Tag;
    use crate::error::ErrorCode;

    /// Renders the tree as an s-expression.
    fn render(ast: &Ast, source: &str, index: u32) -> String {
        let node = ast.node(index);
        match node.tag() {
            Tag::Unary => {
                format!("({:?} {})", node.operator(), render(ast, source, node.first_child()))
            }
            Tag::Binary => format!(
                "({:?} {} {})",
                node.operator(),
                render(ast, source, node.first_child()),
                render(ast, source, node.first_child() + 1)
            ),
            _ => {
                let span = node.span();
                String::from(&source[span.start as usize..(span.start + span.len) as usize])
            }
        }
    }

    fn parse(source: &str) -> String {
        let ast = parse_expression(Heap, source.as_bytes()).unwrap();
        render(&ast, source, ast.root())
    }

    fn error(source: &str) -> (ErrorCode, u32) {
        let error = parse_expression(Heap, source.as_bytes()).err().unwrap();
        (error.code, error.span.start)
    }

    #[test]
    fn parses_with_precedence() {
        assert_eq!(
            parse("a + b * -c > 1 AND NOT d = 'x' or e"),
            "(Or (And (Greater (Add a (Multiply b (Negate c))) 1) (Not (Equal d 'x'))) e)"
        );
        assert_eq!(
            parse("(a - b) - c * (d / 2.5)"),
            "(Subtract (Subtract a b) (Multiply c (Divide d 2.5)))"
        );
    }

    #[test]
    fn reports_errors() {
        assert_eq!(error("a +"), (ErrorCode::ExpectedExpression, 3));
        assert_eq!(error("(a"), (ErrorCode::ExpectedToken, 2));
        assert_eq!(error("a b"), (ErrorCode::UnexpectedToken, 2));
        assert_eq!(error("a AND"), (ErrorCode::ExpectedExpression, 5));
        assert_eq!(error(&"(".repeat(NESTING_MAX + 1)), (ErrorCode::NestingTooDeep, 64));
    }
}
