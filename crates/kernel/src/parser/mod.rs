//! Parses query text into an `Ast`.
//!
//! `Parser` is the cursor over tokens that every part of the grammar uses, and
//! writes the tree. Expressions are parsed in `expression.rs`.

mod expression;

use crate::allocator::Allocator;
use crate::ast::{Ast, Node, Operator, Tag};
use crate::buffer::Buffer;
use crate::error::{Error, ErrorCode, Span};
use crate::lexer::{Lexer, Token, TokenKind};

/// How deeply operators and parentheses can nest.
pub const NESTING_MAX: usize = 64;

/// Parses `source` as a single expression.
pub fn parse_expression<A: Allocator + 'static>(allocator: A, source: &[u8]) -> Result<Ast, Error> {
    let mut parser = Parser::new(allocator, source)?;
    let root = parser.expression()?;
    let current = parser.current();
    if current.kind != TokenKind::End {
        return Err(Error::new(ErrorCode::UnexpectedToken, current.span));
    }
    Ok(parser.finish(root))
}

pub(crate) struct Parser<'a> {
    source: &'a [u8],
    lexer: Lexer<'a>,
    current: Token,
    nodes: Buffer,
    node_count: u32,
}

impl<'a> Parser<'a> {
    pub(crate) fn new<A: Allocator + 'static>(
        allocator: A,
        source: &'a [u8],
    ) -> Result<Parser<'a>, Error> {
        let mut lexer = Lexer::new(source)?;
        let current = lexer.next_token()?;
        // Each node takes a token of at least a byte, so there are at most
        // as many nodes as bytes.
        let first_byte = Span { start: 0, len: 1 };
        let Some(size_bytes) = source.len().checked_mul(size_of::<Node>()) else {
            return Err(Error::new(ErrorCode::QueryTooLarge, first_byte));
        };
        let nodes = Buffer::allocate(allocator, size_bytes)
            .map_err(|_| Error::new(ErrorCode::OutOfMemory, first_byte))?;
        Ok(Parser { source, lexer, current, nodes, node_count: 0 })
    }

    /// The token the parser is on.
    pub(crate) fn current(&self) -> Token {
        self.current
    }

    /// Moves to the next token and returns the one it was on.
    pub(crate) fn advance(&mut self) -> Result<Token, Error> {
        let token = self.current;
        self.current = self.lexer.next_token()?;
        Ok(token)
    }

    /// Moves past the current token if it is `kind`, and fails otherwise.
    pub(crate) fn expect(&mut self, kind: TokenKind) -> Result<Token, Error> {
        if self.current.kind != kind {
            let mut error = Error::new(ErrorCode::ExpectedToken, self.current.span);
            error.detail = u16::from(kind as u8);
            return Err(error);
        }
        self.advance()
    }

    pub(crate) fn text(&self, token: Token) -> &'a [u8] {
        let start = token.span.start as usize;
        at!(self.source, start..start + token.span.len as usize)
    }

    /// Writes `children` to the tree, next to each other, and returns their
    /// parent, which is not in the tree yet.
    pub(crate) fn operation(&mut self, tag: Tag, operator: Operator, children: &[Node]) -> Node {
        let first_child = self.node_count;
        for &child in children {
            self.push(child);
        }
        Node::operation(tag, operator, first_child)
    }

    /// Writes `root` to the tree, last, and returns the tree.
    pub(crate) fn finish(mut self, root: Node) -> Ast {
        self.push(root);
        Ast::new(self.nodes, self.node_count)
    }

    fn push(&mut self, node: Node) {
        let nodes = self.nodes.as_mut_slice::<Node>();
        check!((self.node_count as usize) < nodes.len());
        *at_mut!(nodes, self.node_count as usize) = node;
        self.node_count += 1;
    }
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
