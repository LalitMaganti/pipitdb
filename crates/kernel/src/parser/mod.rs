//! Parses query text into an `Ast`.
//!
//! `Parser` is the cursor over tokens that every part of the grammar uses, and
//! writes the tree. Expressions are parsed in `expression.rs`.

mod expression;

use crate::allocator::Allocator;
use crate::ast::{Ast, BLOCK_BYTES, BLOCK_NODES, Node, Operator, Tag};
use crate::buffer::Buffer;
use crate::error::{Error, ErrorCode, Span};
use crate::lexer::{Lexer, Token, TokenKind};

/// How deeply operators, parentheses and calls can nest. Each waiting
/// argument counts as a level.
pub const NESTING_MAX: usize = 64;

/// Parses `source` as a single expression.
pub fn parse_expression<A: Allocator + Clone + 'static>(
    allocator: A,
    source: &[u8],
) -> Result<Ast, Error> {
    let mut parser = Parser::new(allocator, source)?;
    let root = parser.expression()?;
    let current = parser.current();
    if current.kind != TokenKind::End {
        return Err(Error::new(ErrorCode::UnexpectedToken, current.span));
    }
    parser.finish(root)
}

pub(crate) struct Parser<'a> {
    source: &'a [u8],
    lexer: Lexer<'a>,
    current: Token,
    nodes: Buffer,
    node_count: u32,
    full: bool,
}

impl<'a> Parser<'a> {
    pub(crate) fn new<A: Allocator + Clone + 'static>(
        allocator: A,
        source: &'a [u8],
    ) -> Result<Parser<'a>, Error> {
        let mut lexer = Lexer::new(source)?;
        let current = lexer.next_token()?;
        // SAFETY: nodes are only read up to `node_count`, after they are
        // written.
        let nodes = unsafe { Buffer::allocate_uninit(allocator, BLOCK_BYTES) }
            .map_err(|_| Error::new(ErrorCode::OutOfMemory, Span { start: 0, len: 1 }))?;
        Ok(Parser { source, lexer, current, nodes, node_count: 0, full: false })
    }

    /// The token the parser is on.
    pub(crate) fn current(&self) -> Token {
        self.current
    }

    /// Moves to the next token and returns the one it was on.
    pub(crate) fn advance(&mut self) -> Result<Token, Error> {
        if self.full {
            return Err(Error::new(ErrorCode::QueryTooLarge, self.current.span));
        }
        let token = self.current;
        self.current = self.lexer.next_token()?;
        Ok(token)
    }

    /// Moves past the current token if it is `kind`, and fails otherwise.
    pub(crate) fn expect(&mut self, kind: TokenKind) -> Result<Token, Error> {
        if self.current.kind != kind {
            return Err(Error::expected(kind, self.current.span));
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
        Node::operation(tag, operator, self.write(children))
    }

    /// Writes `nodes` to the tree, next to each other, and returns the index
    /// of the first.
    pub(crate) fn write(&mut self, nodes: &[Node]) -> u32 {
        let first = self.node_count;
        for &node in nodes {
            self.push(node);
        }
        first
    }

    pub(crate) fn node_count(&self) -> u32 {
        self.node_count
    }

    /// Writes `root` to the tree, last, and returns the tree.
    pub(crate) fn finish(mut self, root: Node) -> Result<Ast, Error> {
        self.push(root);
        if self.full {
            return Err(Error::new(ErrorCode::QueryTooLarge, self.current.span));
        }
        Ok(Ast::new(self.nodes, self.node_count))
    }

    /// When the block is full, sets `full` instead of failing, so writing a
    /// node stays cheap. `advance` reports it at the next token.
    fn push(&mut self, node: Node) {
        if self.node_count as usize == BLOCK_NODES {
            self.full = true;
            return;
        }
        // SAFETY: `node_count` is below `BLOCK_NODES`, so this is in the block.
        unsafe { self.nodes.as_mut_ptr::<Node>().add(self.node_count as usize).write(node) };
        self.node_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;

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
            Tag::Call => {
                let children = (0..node.child_count())
                    .map(|i| render(ast, source, node.first_child() + i))
                    .collect::<alloc::vec::Vec<_>>();
                format!("(Call {})", children.join(" "))
            }
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
    fn parses_calls() {
        assert_eq!(parse("count(*)"), "(Call count *)");
        assert_eq!(parse("now()"), "(Call now)");
        assert_eq!(
            parse("max(a + 1, f(g(x), y)) * 2"),
            "(Multiply (Call max (Add a 1) (Call f (Call g x) y)) 2)"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under Miri")]
    fn reports_a_full_block() {
        let long = vec!["a"; crate::ast::BLOCK_NODES].join("+");
        assert_eq!(error(&long).0, ErrorCode::QueryTooLarge);
    }

    #[test]
    fn reports_errors() {
        assert_eq!(error("a +"), (ErrorCode::ExpectedExpression, 3));
        assert_eq!(error("(a"), (ErrorCode::ExpectedToken, 2));
        assert_eq!(error("a b"), (ErrorCode::UnexpectedToken, 2));
        assert_eq!(error("a AND"), (ErrorCode::ExpectedExpression, 5));
        assert_eq!(error(&"(".repeat(NESTING_MAX + 1)), (ErrorCode::NestingTooDeep, 64));
        assert_eq!(error("f(a b"), (ErrorCode::ExpectedToken, 4));
        assert_eq!(error("f(a,"), (ErrorCode::ExpectedExpression, 4));
        let many = format!("f({})", vec!["a"; NESTING_MAX + 1].join(","));
        assert_eq!(error(&many).0, ErrorCode::NestingTooDeep);
    }
}
