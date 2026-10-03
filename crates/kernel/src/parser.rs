//! Parses expressions by Pratt parsing, with an explicit stack instead of
//! recursion.
//!
//! Finished operands wait on the stack as nodes not yet in the tree. An
//! operator's operands are written to the tree together, so they end up next
//! to each other, as the tree needs.

use crate::allocator::Allocator;
use crate::ast::{Ast, LEAF_LEN_MAX, Node, Operator, Tag};
use crate::buffer::Buffer;
use crate::error::{Error, ErrorCode, Span};
use crate::lexer::{Lexer, Token, TokenKind};

/// How deeply operators and parentheses can nest.
pub const NESTING_MAX: usize = 64;

// Binding powers: higher binds tighter. A binary operator with power `p`
// parses its right operand at `p + 1`, so equal operators group to the left.
const OR: u8 = 1;
const AND: u8 = 3;
const NOT: u8 = 5;
const COMPARISON: u8 = 7;
const SUM: u8 = 9;
const PRODUCT: u8 = 11;
const NEGATE: u8 = 13;

const TOKEN_KIND_COUNT: usize = TokenKind::End as usize + 1;

/// The binary operator each token kind stands for, and its binding power.
/// `AND` and `OR` are identifiers, so they are handled separately.
static INFIX: [Option<(Operator, u8)>; TOKEN_KIND_COUNT] = infix();

const fn infix() -> [Option<(Operator, u8)>; TOKEN_KIND_COUNT] {
    let mut infix = [None; TOKEN_KIND_COUNT];
    infix[TokenKind::Equal as usize] = Some((Operator::Equal, COMPARISON));
    infix[TokenKind::NotEqual as usize] = Some((Operator::NotEqual, COMPARISON));
    infix[TokenKind::Less as usize] = Some((Operator::Less, COMPARISON));
    infix[TokenKind::LessEqual as usize] = Some((Operator::LessEqual, COMPARISON));
    infix[TokenKind::Greater as usize] = Some((Operator::Greater, COMPARISON));
    infix[TokenKind::GreaterEqual as usize] = Some((Operator::GreaterEqual, COMPARISON));
    infix[TokenKind::Plus as usize] = Some((Operator::Add, SUM));
    infix[TokenKind::Minus as usize] = Some((Operator::Subtract, SUM));
    infix[TokenKind::Star as usize] = Some((Operator::Multiply, PRODUCT));
    infix[TokenKind::Slash as usize] = Some((Operator::Divide, PRODUCT));
    infix
}

/// Parses `source` as a single expression.
pub fn parse_expression<A: Allocator + 'static>(allocator: A, source: &[u8]) -> Result<Ast, Error> {
    let mut parser = Parser::new(allocator, source)?;
    let root = parser.expression()?;
    if parser.current.kind != TokenKind::End {
        return Err(Error::new(ErrorCode::UnexpectedToken, parser.current.span));
    }
    parser.push(root);
    Ok(Ast::new(parser.nodes, parser.node_count))
}

#[derive(Clone, Copy)]
enum Keyword {
    None,
    And,
    Or,
    Not,
}

#[derive(Clone, Copy)]
enum Pending {
    Unary(Operator),
    Binary(Operator, Node),
    Parenthesis,
}

/// An operator or `(` waiting for its operand, and the binding power to go
/// back to once it has it.
#[derive(Clone, Copy)]
struct Frame {
    pending: Pending,
    min_power: u8,
}

struct Parser<'a> {
    source: &'a [u8],
    lexer: Lexer<'a>,
    current: Token,
    nodes: Buffer,
    node_count: u32,
}

impl<'a> Parser<'a> {
    fn new<A: Allocator + 'static>(allocator: A, source: &'a [u8]) -> Result<Parser<'a>, Error> {
        let mut lexer = Lexer::new(source)?;
        let current = lexer.next_token()?;
        // Each node takes a token of at least a byte, so there are at most
        // as many nodes as bytes.
        let nowhere = Span { start: 0, len: 0 };
        let Some(size_bytes) = source.len().checked_mul(size_of::<Node>()) else {
            return Err(Error::new(ErrorCode::QueryTooLarge, nowhere));
        };
        let nodes = Buffer::allocate(allocator, size_bytes)
            .map_err(|_| Error::new(ErrorCode::OutOfMemory, nowhere))?;
        Ok(Parser { source, lexer, current, nodes, node_count: 0 })
    }

    /// Parses an expression, stopping at the first token that can't continue
    /// it. Returns its root, which is not in the tree yet.
    fn expression(&mut self) -> Result<Node, Error> {
        let mut stack = [Frame { pending: Pending::Parenthesis, min_power: 0 }; NESTING_MAX];
        let mut depth = 0;
        let mut min_power = 0;
        'operand: loop {
            let token = self.advance()?;
            let pending = match token.kind {
                TokenKind::Integer | TokenKind::Float | TokenKind::String => None,
                TokenKind::Minus => Some((Pending::Unary(Operator::Negate), NEGATE)),
                TokenKind::LeftParen => Some((Pending::Parenthesis, 0)),
                TokenKind::Identifier => match self.keyword(token) {
                    Keyword::None => None,
                    Keyword::Not => Some((Pending::Unary(Operator::Not), NOT)),
                    Keyword::And | Keyword::Or => {
                        return Err(Error::new(ErrorCode::ExpectedExpression, token.span));
                    }
                },
                _ => return Err(Error::new(ErrorCode::ExpectedExpression, token.span)),
            };
            if let Some((pending, power)) = pending {
                if depth == NESTING_MAX {
                    return Err(Error::new(ErrorCode::NestingTooDeep, token.span));
                }
                stack[depth] = Frame { pending, min_power };
                depth += 1;
                min_power = power;
                continue;
            }
            let mut operand = leaf(token)?;
            loop {
                if let Some((operator, power)) = self.infix(self.current)
                    && power > min_power
                {
                    let token = self.advance()?;
                    if depth == NESTING_MAX {
                        return Err(Error::new(ErrorCode::NestingTooDeep, token.span));
                    }
                    stack[depth] = Frame { pending: Pending::Binary(operator, operand), min_power };
                    depth += 1;
                    min_power = power + 1;
                    continue 'operand;
                }
                if depth == 0 {
                    return Ok(operand);
                }
                depth -= 1;
                let frame = stack[depth];
                min_power = frame.min_power;
                operand = match frame.pending {
                    Pending::Unary(operator) => self.operation(Tag::Unary, operator, &[operand]),
                    Pending::Binary(operator, left) => {
                        self.operation(Tag::Binary, operator, &[left, operand])
                    }
                    Pending::Parenthesis => {
                        self.expect(TokenKind::RightParen)?;
                        operand
                    }
                };
            }
        }
    }

    fn infix(&self, token: Token) -> Option<(Operator, u8)> {
        if token.kind != TokenKind::Identifier {
            return INFIX[token.kind as usize];
        }
        match self.keyword(token) {
            Keyword::And => Some((Operator::And, AND)),
            Keyword::Or => Some((Operator::Or, OR)),
            Keyword::Not | Keyword::None => None,
        }
    }

    fn keyword(&self, token: Token) -> Keyword {
        check!(token.kind == TokenKind::Identifier);
        let text = self.text(token);
        let keyword = match text.len() {
            2 => (Keyword::Or, b"or".as_slice()),
            3 if text[0].eq_ignore_ascii_case(&b'a') => (Keyword::And, b"and".as_slice()),
            3 => (Keyword::Not, b"not".as_slice()),
            _ => return Keyword::None,
        };
        if text.eq_ignore_ascii_case(keyword.1) { keyword.0 } else { Keyword::None }
    }

    fn text(&self, token: Token) -> &'a [u8] {
        let start = token.span.start as usize;
        &self.source[start..start + token.span.len as usize]
    }

    /// Writes `children` to the tree, next to each other, and returns their
    /// parent.
    fn operation(&mut self, tag: Tag, operator: Operator, children: &[Node]) -> Node {
        let first_child = self.node_count;
        for &child in children {
            self.push(child);
        }
        Node::operation(tag, operator, first_child)
    }

    fn push(&mut self, node: Node) {
        let nodes = self.nodes.as_mut_slice::<Node>();
        check!((self.node_count as usize) < nodes.len());
        nodes[self.node_count as usize] = node;
        self.node_count += 1;
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token, Error> {
        if self.current.kind != kind {
            let mut error = Error::new(ErrorCode::ExpectedToken, self.current.span);
            error.detail = u16::from(kind as u8);
            return Err(error);
        }
        self.advance()
    }

    /// Moves to the next token and returns the one it was on.
    fn advance(&mut self) -> Result<Token, Error> {
        let token = self.current;
        self.current = self.lexer.next_token()?;
        Ok(token)
    }
}

fn leaf(token: Token) -> Result<Node, Error> {
    if token.span.len > LEAF_LEN_MAX {
        return Err(Error::new(ErrorCode::TokenTooLong, token.span));
    }
    let tag = match token.kind {
        TokenKind::Integer => Tag::Integer,
        TokenKind::Float => Tag::Float,
        TokenKind::String => Tag::String,
        _ => Tag::Column,
    };
    Ok(Node::leaf(tag, token.span))
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::String;

    use super::*;
    use crate::allocator::Heap;

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
