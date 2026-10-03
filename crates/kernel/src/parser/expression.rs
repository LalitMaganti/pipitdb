//! Pratt parsing of expressions, with an explicit stack instead of recursion.
//!
//! Finished operands wait on the stack as nodes not yet in the tree. An
//! operator's operands are written to the tree together, so they end up next
//! to each other, as the tree needs.

use crate::ast::{DATA_MAX, Node, Operator, Tag};
use crate::error::{Error, ErrorCode};
use crate::lexer::{Token, TokenKind};

use super::{NESTING_MAX, Parser};

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
    /// A call's name. Its arguments wait in `Argument` frames above it.
    Call(Node),
    Argument(Node),
}

/// What a token in operand position does: gives an operand, or starts
/// something that waits on the stack for one.
enum Step {
    Operand(Node),
    Push(Pending, u8),
}

/// An operator, `(` or call waiting for its operand, and the binding power to
/// go back to once it has it.
#[derive(Clone, Copy)]
struct Frame {
    pending: Pending,
    min_power: u8,
}

/// The frames waiting for operands. Bounded, as the parser doesn't recurse.
struct Stack {
    frames: [Frame; NESTING_MAX],
    depth: usize,
}

impl Stack {
    fn push(&mut self, pending: Pending, min_power: u8, token: Token) -> Result<(), Error> {
        if self.depth == NESTING_MAX {
            return Err(Error::new(ErrorCode::NestingTooDeep, token.span));
        }
        *at_mut!(self.frames, self.depth) = Frame { pending, min_power };
        self.depth += 1;
        Ok(())
    }
}

impl Parser<'_> {
    /// Parses an expression, stopping at the first token that can't continue
    /// it. Returns its root, which is not in the tree yet.
    pub(crate) fn expression(&mut self) -> Result<Node, Error> {
        let empty = Frame { pending: Pending::Parenthesis, min_power: 0 };
        let mut stack = Stack { frames: [empty; NESTING_MAX], depth: 0 };
        let mut min_power = 0;
        'operand: loop {
            let token = self.advance()?;
            let mut operand = match self.prefix(token)? {
                Step::Operand(operand) => operand,
                Step::Push(pending, power) => {
                    stack.push(pending, min_power, token)?;
                    min_power = power;
                    continue;
                }
            };
            loop {
                if let Some((operator, power)) = self.infix(self.current())
                    && power > min_power
                {
                    let token = self.advance()?;
                    stack.push(Pending::Binary(operator, operand), min_power, token)?;
                    min_power = power + 1;
                    continue 'operand;
                }
                if stack.depth == 0 {
                    return Ok(operand);
                }
                let frame = *at!(stack.frames, stack.depth - 1);
                if let Pending::Call(_) | Pending::Argument(_) = frame.pending {
                    let token = self.advance()?;
                    if token.kind == TokenKind::Comma {
                        stack.push(Pending::Argument(operand), 0, token)?;
                        min_power = 0;
                        continue 'operand;
                    }
                    if token.kind != TokenKind::RightParen {
                        return Err(Error::expected(TokenKind::RightParen, token.span));
                    }
                    (operand, min_power) = self.call(&mut stack, operand);
                    continue;
                }
                stack.depth -= 1;
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
                    Pending::Call(_) | Pending::Argument(_) => crate::check::check_failed(line!()),
                };
            }
        }
    }

    /// Pops a call and its arguments off `stack`, `last` being its last
    /// argument, and writes them to the tree. Returns the call and the binding
    /// power to go back to.
    fn call(&mut self, stack: &mut Stack, last: Node) -> (Node, u8) {
        let mut call = stack.depth - 1;
        while let Pending::Argument(_) = at!(stack.frames, call).pending {
            call -= 1;
        }
        let frames = at!(stack.frames, call..stack.depth);
        let first_child = self.node_count();
        for frame in frames {
            if let Pending::Call(node) | Pending::Argument(node) = frame.pending {
                self.write(&[node]);
            }
        }
        self.write(&[last]);
        stack.depth = call;
        let node = Node::list(Tag::Call, self.node_count() - first_child, first_child);
        (node, at!(stack.frames, call).min_power)
    }

    /// What `token`, in operand position, does.
    fn prefix(&mut self, token: Token) -> Result<Step, Error> {
        Ok(match token.kind {
            TokenKind::Integer | TokenKind::Float | TokenKind::String | TokenKind::Star => {
                Step::Operand(leaf(token)?)
            }
            TokenKind::Minus => Step::Push(Pending::Unary(Operator::Negate), NEGATE),
            TokenKind::LeftParen => Step::Push(Pending::Parenthesis, 0),
            TokenKind::Identifier => match self.keyword(token) {
                Keyword::None if self.current().kind == TokenKind::LeftParen => {
                    self.advance()?;
                    let name = leaf(token)?;
                    if self.current().kind == TokenKind::RightParen {
                        self.advance()?;
                        Step::Operand(Node::list(Tag::Call, 1, self.write(&[name])))
                    } else {
                        Step::Push(Pending::Call(name), 0)
                    }
                }
                Keyword::None => Step::Operand(leaf(token)?),
                Keyword::Not => Step::Push(Pending::Unary(Operator::Not), NOT),
                Keyword::And | Keyword::Or => {
                    return Err(Error::new(ErrorCode::ExpectedExpression, token.span));
                }
            },
            _ => return Err(Error::new(ErrorCode::ExpectedExpression, token.span)),
        })
    }

    fn infix(&self, token: Token) -> Option<(Operator, u8)> {
        if token.kind != TokenKind::Identifier {
            return *at!(INFIX, token.kind as usize);
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
}

fn leaf(token: Token) -> Result<Node, Error> {
    if token.span.len > DATA_MAX {
        return Err(Error::new(ErrorCode::TokenTooLong, token.span));
    }
    let tag = match token.kind {
        TokenKind::Integer => Tag::Integer,
        TokenKind::Float => Tag::Float,
        TokenKind::String => Tag::String,
        TokenKind::Star => Tag::Star,
        _ => Tag::Name,
    };
    Ok(Node::leaf(tag, token.span))
}
