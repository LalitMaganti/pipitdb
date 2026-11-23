//! Parses query text into an `Ast`.
//!
//! `Parser` is the cursor over tokens that every part of the grammar uses, and
//! writes the tree.

use core::mem::MaybeUninit;

use crate::ast::{Ast, DATA_MAX, Node, Nodes, Operator, Tag};
use crate::error::{Error, ErrorCode, Span};
use crate::keywords::same_word;
use crate::lexer::{Lexer, Token, TokenKind};
use crate::registry::{ITEMS_MAX, Item, Point, Registry, Shared};
use pipit_kernel::env::Env;
use pipit_kernel::small_vec::SmallVec;

/// How deeply operators, parentheses and calls can nest. Each waiting
/// argument counts as a level.
pub const NESTING_MAX: usize = 64;

/// How many stages a query, or items a list, can have.
pub const LIST_MAX: usize = 64;

/// How many values an `IN` list can have.
pub const IN_LIST_MAX: usize = 1 << 16;

/// Parses `source` as a query, with the stages in `registry`.
pub fn parse_query(env: &Env, registry: &Registry, source: &[u8]) -> Result<Ast, Error> {
    let mut parser = Parser::new(env, source)?;
    let root = parser.query(registry).and_then(|root| parser.end(root));
    parser.finish(root)
}

/// Parses `source` as a single expression.
pub fn parse_expression(env: &Env, source: &[u8]) -> Result<Ast, Error> {
    let mut parser = Parser::new(env, source)?;
    let root = parser.expression().and_then(|root| parser.end(root));
    parser.finish(root)
}

pub(crate) struct Parser<'a> {
    /// What lists too long for the stack are gathered with.
    env: &'a Env,
    source: &'a [u8],
    lexer: Lexer<'a>,
    current: Token,
    nodes: Nodes,
    failed: Option<Error>,
}

impl<'a> Parser<'a> {
    pub(crate) fn new(env: &'a Env, source: &'a [u8]) -> Result<Parser<'a>, Error> {
        let mut lexer = Lexer::new(source)?;
        let current = lexer.next_token()?;
        let nodes = Nodes::new(env)
            .map_err(|_| Error::new(ErrorCode::OutOfMemory, Span { start: 0, len: 1 }))?;
        Ok(Parser { env, source, lexer, current, nodes, failed: None })
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
        let first = self.nodes.len();
        for &node in nodes {
            self.push(node);
        }
        first
    }

    pub(crate) fn node_count(&self) -> u32 {
        self.nodes.len()
    }

    /// Returns `root` if the parser is at the end of the source.
    fn end(&self, root: Node) -> Result<Node, Error> {
        if self.current.kind != TokenKind::End {
            return Err(Error::new(ErrorCode::UnexpectedToken, self.current.span));
        }
        Ok(root)
    }

    /// Writes `root` to the tree, last, and returns the tree. Running out of
    /// memory comes first, as the parse went on past it.
    pub(crate) fn finish(mut self, root: Result<Node, Error>) -> Result<Ast, Error> {
        if let Ok(root) = root {
            self.push(root);
        }
        if let Some(error) = self.failed {
            return Err(error);
        }
        root?;
        Ok(Ast::new(self.nodes))
    }

    /// If the next block can't be had, later nodes overwrite the last block,
    /// which is safe as `finish` then fails, so the tree is never read.
    fn push(&mut self, node: Node) {
        self.nodes.push(node);
        if self.nodes.needs_block()
            && let Err(code) = self.nodes.grow()
        {
            self.failed.get_or_insert(Error::new(code, self.current.span));
        }
    }
}

// Expressions: Pratt parsing, with an explicit stack instead of recursion.
// Finished operands wait on the stack as nodes not yet in the tree. An
// operator's operands are written to the tree together, so they end up next
// to each other, as the tree needs.

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
/// Keywords, such as `AND`, are identifiers, so they have their own table.
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
    infix[TokenKind::Percent as usize] = Some((Operator::Modulo, PRODUCT));
    infix
}

const KEYWORD_COUNT: usize = Keyword::End as usize + 1;

/// The operator each keyword after an operand stands for, and its binding
/// power. `IS` stands for `IsNull`, as it starts `IS [NOT] NULL`.
static KEYWORD_INFIX: [Option<(Operator, u8)>; KEYWORD_COUNT] = keyword_infix();

const fn keyword_infix() -> [Option<(Operator, u8)>; KEYWORD_COUNT] {
    let mut infix = [None; KEYWORD_COUNT];
    infix[Keyword::And as usize] = Some((Operator::And, AND));
    infix[Keyword::Or as usize] = Some((Operator::Or, OR));
    infix[Keyword::Like as usize] = Some((Operator::Like, COMPARISON));
    infix[Keyword::Is as usize] = Some((Operator::IsNull, COMPARISON));
    infix[Keyword::In as usize] = Some((Operator::In, COMPARISON));
    infix
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Keyword {
    None,
    And,
    Or,
    Not,
    Like,
    Is,
    In,
    Distinct,
    Case,
    When,
    Then,
    Else,
    End,
}

#[derive(Clone, Copy)]
enum Pending {
    Unary(Operator),
    Binary(Operator, Node),
    Parenthesis,
    Call(Node),
    Argument(Node),
    /// A `CASE`, its parts on the frames above it.
    Case,
    /// A part of a `CASE`, and the keyword that ended it.
    CasePart(Node, Keyword),
}

/// What a part of a call or `CASE` ending did.
enum Ended {
    /// Another part follows.
    Next,
    /// The call or `CASE` closed, and the binding power to go back to.
    Closed(Node, u8),
    /// It wasn't part of one.
    No,
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
/// Frames from `depth` up are unwritten, so making one costs nothing.
struct Stack {
    frames: [MaybeUninit<Frame>; NESTING_MAX],
    depth: usize,
}

impl Stack {
    fn new() -> Stack {
        Stack { frames: [const { MaybeUninit::uninit() }; NESTING_MAX], depth: 0 }
    }

    fn push(&mut self, pending: Pending, min_power: u8, token: Token) -> Result<(), Error> {
        if self.depth == NESTING_MAX {
            return Err(Error::new(ErrorCode::NestingTooDeep, token.span));
        }
        at_mut!(self.frames, self.depth).write(Frame { pending, min_power });
        self.depth += 1;
        Ok(())
    }

    /// The frames, bottom first.
    fn frames(&self) -> &[Frame] {
        // SAFETY: `push` keeps `depth` within `frames`, the first `depth` were
        // written, and `MaybeUninit<Frame>` is laid out as `Frame`.
        unsafe { core::slice::from_raw_parts(self.frames.as_ptr().cast(), self.depth) }
    }

    /// Keeps the bottom `depth` frames.
    fn truncate(&mut self, depth: usize) {
        check!(depth <= self.depth);
        self.depth = depth;
    }
}

impl Parser<'_> {
    /// Parses an expression, stopping at the first token that can't continue
    /// it. Returns its root, which is not in the tree yet.
    pub(crate) fn expression(&mut self) -> Result<Node, Error> {
        let mut stack = Stack::new();
        let mut min_power = 0;
        'operand: loop {
            let token = self.advance()?;
            let mut operand = match self.prefix(token)? {
                Step::Operand(operand) => operand,
                Step::Push(pending, power) => {
                    self.push_pending(&mut stack, pending, min_power, token)?;
                    min_power = power;
                    continue;
                }
            };
            loop {
                if let Some((operator, power)) = self.infix(self.current())
                    && power > min_power
                {
                    operand = match operator {
                        Operator::IsNull => self.is_null(operand)?,
                        Operator::In => self.in_list(operand)?,
                        _ => {
                            let token = self.advance()?;
                            stack.push(Pending::Binary(operator, operand), min_power, token)?;
                            min_power = power + 1;
                            continue 'operand;
                        }
                    };
                    continue;
                }
                let Some(&frame) = stack.frames().last() else { return Ok(operand) };
                match self.end_part(&mut stack, frame, operand)? {
                    Ended::Next => {
                        min_power = 0;
                        continue 'operand;
                    }
                    Ended::Closed(closed, power) => {
                        (operand, min_power) = (closed, power);
                        continue;
                    }
                    Ended::No => {}
                }
                stack.truncate(stack.depth - 1);
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
                    Pending::Call(_)
                    | Pending::Argument(_)
                    | Pending::Case
                    | Pending::CasePart(..) => {
                        pipit_kernel::failed!()
                    }
                };
            }
        }
    }

    /// `operand IN (1, -2, 'a')`, from the `IN`. The list holds literals
    /// only, so it can be long without nesting: they're gathered as they're
    /// read, and written next to each other once it ends.
    fn in_list(&mut self, operand: Node) -> Result<Node, Error> {
        self.advance()?;
        self.expect(TokenKind::LeftParen)?;
        let mut items = SmallVec::new(self.env, IN_LIST_MAX)?;
        loop {
            let negative = self.current().kind == TokenKind::Minus;
            if negative {
                self.advance()?;
            }
            let token = self.advance()?;
            let number = matches!(token.kind, TokenKind::Integer | TokenKind::Float);
            if !number && (negative || token.kind != TokenKind::String) {
                return Err(Error::new(ErrorCode::ExpectedExpression, token.span));
            }
            let mut item = leaf(token)?;
            if negative {
                item = self.operation(Tag::Unary, Operator::Negate, &[item]);
            }
            if items.len() == IN_LIST_MAX {
                return Err(Error::new(ErrorCode::ListTooLong, token.span));
            }
            items.push(item)?;
            let token = self.advance()?;
            match token.kind {
                TokenKind::RightParen => break,
                TokenKind::Comma => {}
                _ => return Err(Error::expected(TokenKind::RightParen, token.span)),
            }
        }
        let list = self.list(Tag::List, &items);
        Ok(self.operation(Tag::Binary, Operator::In, &[operand, list]))
    }

    /// `operand IS NULL` or `operand IS NOT NULL`, from the `IS`.
    fn is_null(&mut self, operand: Node) -> Result<Node, Error> {
        self.advance()?;
        let not = self.at_word("not");
        if not {
            self.advance()?;
        }
        if !self.at_word("null") {
            return Err(Error::new(ErrorCode::ExpectedWord, self.current().span));
        }
        self.advance()?;
        let operator = if not { Operator::IsNotNull } else { Operator::IsNull };
        Ok(self.operation(Tag::Unary, operator, &[operand]))
    }

    /// Pops a call and its arguments off `stack`, `last` being its last
    /// argument, and writes them to the tree. Returns the call and the binding
    /// power to go back to.
    fn call(&mut self, stack: &mut Stack, last: Node) -> (Node, u8) {
        let frames = stack.frames();
        let mut call = frames.len() - 1;
        while let Pending::Argument(_) = at!(frames, call).pending {
            call -= 1;
        }
        let power = at!(frames, call).min_power;
        let frames = at!(frames, call..);
        let first_child = self.node_count();
        for frame in frames {
            if let Pending::Call(node) | Pending::Argument(node) = frame.pending {
                self.write(&[node]);
            }
        }
        self.write(&[last]);
        stack.truncate(call);
        (Node::list(Tag::Call, self.node_count() - first_child, first_child), power)
    }

    /// Pushes `pending`, from `token`, going back to `min_power` once it has
    /// its operand.
    fn push_pending(
        &mut self,
        stack: &mut Stack,
        pending: Pending,
        min_power: u8,
        token: Token,
    ) -> Result<(), Error> {
        // `DISTINCT` only starts a call's argument.
        let top = stack.frames().last().map(|top| top.pending);
        if let Pending::Unary(Operator::Distinct) = pending
            && !matches!(top, Some(Pending::Call(_) | Pending::Argument(_)))
        {
            return Err(Error::new(ErrorCode::ExpectedExpression, token.span));
        }
        stack.push(pending, min_power, token)?;
        // A `CASE` without a subject starts with `WHEN`.
        if let Pending::Case = pending
            && self.at_keyword(Keyword::When)
        {
            let token = self.advance()?;
            stack.push(Pending::CasePart(Node::empty(), Keyword::When), 0, token)?;
        }
        Ok(())
    }

    /// Ends `operand` as an argument of the call, or a part of the `CASE`,
    /// that `frame`, the top of `stack`, is part of, at the token after it:
    /// another follows, or the call or `CASE` closes.
    fn end_part(&mut self, stack: &mut Stack, frame: Frame, operand: Node) -> Result<Ended, Error> {
        match frame.pending {
            Pending::Case | Pending::CasePart(..) => {
                let token = self.advance()?;
                let keyword = self.keyword(token);
                // What may end this part, by what ended the one before: the
                // subject, a condition, a value, or what `ELSE` gives.
                let before = match frame.pending {
                    Pending::CasePart(_, keyword) => keyword,
                    _ => Keyword::Case,
                };
                let allowed = match before {
                    Keyword::Case => keyword == Keyword::When,
                    Keyword::When => keyword == Keyword::Then,
                    Keyword::Then => {
                        matches!(keyword, Keyword::When | Keyword::Else | Keyword::End)
                    }
                    _ => keyword == Keyword::End,
                };
                if !allowed {
                    return Err(Error::new(ErrorCode::ExpectedWord, token.span));
                }
                if keyword != Keyword::End {
                    stack.push(Pending::CasePart(operand, keyword), 0, token)?;
                    return Ok(Ended::Next);
                }
                let (case, power) = self.case(stack, operand, before);
                Ok(Ended::Closed(case, power))
            }
            Pending::Call(_) | Pending::Argument(_) => {
                let token = self.advance()?;
                if token.kind == TokenKind::Comma {
                    stack.push(Pending::Argument(operand), 0, token)?;
                    return Ok(Ended::Next);
                }
                if token.kind != TokenKind::RightParen {
                    return Err(Error::expected(TokenKind::RightParen, token.span));
                }
                let (call, power) = self.call(stack, operand);
                Ok(Ended::Closed(call, power))
            }
            Pending::Unary(_) | Pending::Binary(..) | Pending::Parenthesis => Ok(Ended::No),
        }
    }

    /// Pops a `CASE` and its parts off `stack`, checked to be in order as
    /// they ended, `last` its last, after a part ended by `before`, and
    /// writes them to the tree: its subject, each condition and value, and
    /// its `ELSE`'s value, an empty node for one it hasn't. Returns the `CASE`
    /// and the binding power to go back to.
    fn case(&mut self, stack: &mut Stack, last: Node, before: Keyword) -> (Node, u8) {
        let frames = stack.frames();
        let mut case = frames.len() - 1;
        while let Pending::CasePart(..) = at!(frames, case).pending {
            case -= 1;
        }
        let first_child = self.node_count();
        let power = at!(frames, case).min_power;
        for frame in at!(frames, case + 1..) {
            if let Pending::CasePart(node, _) = frame.pending {
                self.write(&[node]);
            }
        }
        self.write(&[last]);
        if before != Keyword::Else {
            self.write(&[Node::empty()]);
        }
        stack.truncate(case);
        (Node::list(Tag::Case, self.node_count() - first_child, first_child), power)
    }

    /// Whether the current token is the keyword `keyword`.
    fn at_keyword(&self, keyword: Keyword) -> bool {
        self.keyword(self.current()) == keyword
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
                // `IS` and `IN` are only keywords after an operand.
                Keyword::None | Keyword::Is | Keyword::In
                    if self.current().kind == TokenKind::LeftParen =>
                {
                    self.advance()?;
                    let name = leaf(token)?;
                    if self.current().kind == TokenKind::RightParen {
                        self.advance()?;
                        Step::Operand(Node::list(Tag::Call, 1, self.write(&[name])))
                    } else {
                        Step::Push(Pending::Call(name), 0)
                    }
                }
                // `t.x`, a column of alias `t`, as one name.
                Keyword::None if self.current().kind == TokenKind::Dot => {
                    self.advance()?;
                    let column = self.expect(TokenKind::Identifier)?;
                    let len = column.span.start + column.span.len - token.span.start;
                    let span = Span { start: token.span.start, len };
                    Step::Operand(leaf(Token { kind: token.kind, span })?)
                }
                Keyword::None | Keyword::Is | Keyword::In => Step::Operand(leaf(token)?),
                Keyword::Not => Step::Push(Pending::Unary(Operator::Not), NOT),
                // Over all that follows, as in `count(DISTINCT a)`.
                Keyword::Distinct => Step::Push(Pending::Unary(Operator::Distinct), 0),
                Keyword::Case => Step::Push(Pending::Case, 0),
                Keyword::And
                | Keyword::Or
                | Keyword::Like
                | Keyword::When
                | Keyword::Then
                | Keyword::Else
                | Keyword::End => {
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
        *at!(KEYWORD_INFIX, self.keyword(token) as usize)
    }

    /// The keyword `token` is, if it's one.
    fn keyword(&self, token: Token) -> Keyword {
        if token.kind != TokenKind::Identifier {
            return Keyword::None;
        }
        let text = self.text(token);
        if text.len() > 8 {
            return Keyword::None;
        }
        // Setting the bit only case changes lowers a letter, and keywords are
        // only letters.
        let word = text.iter().rev().fold(0, |word, &byte| word << 8 | u64::from(byte | 0x20));
        #[expect(clippy::cast_possible_truncation, reason = "the first letter")]
        let first = word as u8;
        let keyword = match (text.len(), first) {
            (2, b'o') => (Keyword::Or, const { packed(b"or") }),
            (2, _) if word >> 8 == u64::from(b's') => (Keyword::Is, const { packed(b"is") }),
            (2, _) => (Keyword::In, const { packed(b"in") }),
            (3, b'a') => (Keyword::And, const { packed(b"and") }),
            (3, b'e') => (Keyword::End, const { packed(b"end") }),
            (3, _) => (Keyword::Not, const { packed(b"not") }),
            (4, b'c') => (Keyword::Case, const { packed(b"case") }),
            (4, b'w') => (Keyword::When, const { packed(b"when") }),
            (4, b't') => (Keyword::Then, const { packed(b"then") }),
            (4, b'e') => (Keyword::Else, const { packed(b"else") }),
            (4, _) => (Keyword::Like, const { packed(b"like") }),
            (8, _) => (Keyword::Distinct, const { packed(b"distinct") }),
            _ => return Keyword::None,
        };
        if word == keyword.1 { keyword.0 } else { Keyword::None }
    }
}

/// A word of at most 8 bytes, as a number, its first byte lowest.
const fn packed(text: &[u8]) -> u64 {
    let mut word = 0;
    let mut i = text.len();
    while i > 0 {
        i -= 1;
        word = word << 8 | text[i] as u64;
    }
    word
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

// Queries: a source, then stages after `|>`. Each stage is found by its
// keyword in a `Registry`, and its rule's items are parsed in order.

impl Parser<'_> {
    /// Parses a query, returning its root, which is not in the tree yet.
    pub(crate) fn query(&mut self, registry: &Registry) -> Result<Node, Error> {
        let mut stages = [Node::empty(); LIST_MAX];
        let mut count = 0;
        loop {
            if count == LIST_MAX {
                return Err(Error::new(ErrorCode::ListTooLong, self.current().span));
            }
            let point = if count == 0 { Point::Source } else { Point::Stage };
            *at_mut!(stages, count) = self.stage(registry, point)?;
            count += 1;
            if self.current().kind != TokenKind::Pipe {
                return Ok(self.list(Tag::Query, at!(stages, ..count)));
            }
            self.advance()?;
        }
    }

    fn stage(&mut self, registry: &Registry, point: Point) -> Result<Node, Error> {
        let keyword = self.advance()?;
        let id = match keyword.kind {
            TokenKind::Identifier => registry.find(self.text(keyword)),
            _ => None,
        };
        let Some(id) = id else {
            let code = match point {
                Point::Source => ErrorCode::ExpectedSource,
                Point::Stage => ErrorCode::UnknownStage,
            };
            return Err(Error::new(code, keyword.span));
        };
        let rule = registry.rule(id);
        if rule.point != point {
            let code = match point {
                Point::Source => ErrorCode::ExpectedSource,
                Point::Stage => ErrorCode::UnexpectedSource,
            };
            return Err(Error::new(code, keyword.span));
        }
        let mut children = [Node::empty(); ITEMS_MAX];
        for (i, item) in rule.items.iter().enumerate() {
            *at_mut!(children, i) = match *item {
                Item::One(shared) => self.shared(shared)?,
                Item::List(shared) => self.list_of(shared)?,
                Item::Clause(words, shared) => self.clause(words, shared, false)?,
                Item::After(words, shared) => self.clause(words, shared, true)?,
                Item::Parenthesized(words, shared) => self.parenthesized(words, shared)?,
                Item::Optional(word) => {
                    if self.at_word(word) {
                        self.advance()?;
                    }
                    self.list(Tag::List, &[])
                }
            };
        }
        let first_child = self.write(at!(children, ..rule.items.len()));
        Ok(Node::stage(id, first_child))
    }

    fn shared(&mut self, shared: Shared) -> Result<Node, Error> {
        match shared {
            Shared::Name => {
                let token = self.expect(TokenKind::Identifier)?;
                Ok(Node::leaf(Tag::Name, token.span))
            }
            Shared::Expr => self.expression(),
            Shared::Aliased | Shared::Renamed | Shared::Assigned => self.named(shared),
            Shared::Ordered => {
                let expression = self.expression()?;
                let descending = self.at_word("desc");
                if descending || self.at_word("asc") {
                    self.advance()?;
                }
                Ok(Node::sort_key(descending, self.write(&[expression])))
            }
        }
    }

    /// An expression and its name, as `shared` has them, as a `Named`; or
    /// an expression alone, if `Aliased` has no `AS`.
    fn named(&mut self, shared: Shared) -> Result<Node, Error> {
        let (expression, name) = if shared == Shared::Assigned {
            let name = self.shared(Shared::Name)?;
            self.expect(TokenKind::Equal)?;
            (self.expression()?, name)
        } else {
            let first = if shared == Shared::Aliased { Shared::Expr } else { Shared::Name };
            let expression = self.shared(first)?;
            if !self.at_word("as") {
                if shared == Shared::Aliased {
                    return Ok(expression);
                }
                return Err(Error::new(ErrorCode::ExpectedAs, self.current().span));
            }
            self.advance()?;
            (expression, self.shared(Shared::Name)?)
        };
        Ok(Node::list(Tag::Named, 2, self.write(&[expression, name])))
    }

    /// One or more `shared`, separated by commas.
    fn list_of(&mut self, shared: Shared) -> Result<Node, Error> {
        let mut items = [Node::empty(); LIST_MAX];
        let mut count = 0;
        loop {
            if count == LIST_MAX {
                return Err(Error::new(ErrorCode::ListTooLong, self.current().span));
            }
            *at_mut!(items, count) = self.shared(shared)?;
            count += 1;
            if self.current().kind != TokenKind::Comma {
                return Ok(self.list(Tag::List, at!(items, ..count)));
            }
            self.advance()?;
        }
    }

    /// `words`, then one or more `shared`. If the first word isn't next, an
    /// empty list, or an error if `required`.
    fn clause(&mut self, words: &[&str], shared: Shared, required: bool) -> Result<Node, Error> {
        let Some((first, rest)) = words.split_first() else { pipit_kernel::failed!() };
        if !self.at_word(first) {
            if required {
                return Err(Error::new(ErrorCode::ExpectedWord, self.current().span));
            }
            return Ok(self.list(Tag::List, &[]));
        }
        self.advance()?;
        for word in rest {
            if !self.at_word(word) {
                return Err(Error::new(ErrorCode::ExpectedWord, self.current().span));
            }
            self.advance()?;
        }
        self.list_of(shared)
    }

    /// As `clause`, not required, with the list in parentheses.
    fn parenthesized(&mut self, words: &[&str], shared: Shared) -> Result<Node, Error> {
        let Some(first) = words.first() else { pipit_kernel::failed!() };
        if !self.at_word(first) {
            return Ok(self.list(Tag::List, &[]));
        }
        for word in words {
            if !self.at_word(word) {
                return Err(Error::new(ErrorCode::ExpectedWord, self.current().span));
            }
            self.advance()?;
        }
        self.expect(TokenKind::LeftParen)?;
        let list = self.list_of(shared)?;
        self.expect(TokenKind::RightParen)?;
        Ok(list)
    }

    /// Whether the current token is `word`, ignoring case.
    fn at_word(&self, word: &str) -> bool {
        let current = self.current();
        current.kind == TokenKind::Identifier && same_word(self.text(current), word.as_bytes())
    }

    /// Writes `children` to the tree, and returns a `tag` node holding them.
    fn list(&mut self, tag: Tag, children: &[Node]) -> Node {
        let first_child = self.write(children);
        Node::list(tag, self.node_count() - first_child, first_child)
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::BLOCK_BYTES;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;

    use super::*;
    use crate::ast::Tag;
    use crate::error::ErrorCode;
    use pipit_kernel::allocator::Heap;

    /// Renders the tree as an s-expression.
    fn render(source: &str, ast: &Ast, index: u32) -> String {
        let node = ast.node(index);
        match node.tag() {
            Tag::Unary => {
                format!("({:?} {})", node.operator(), render(source, ast, node.first_child()))
            }
            Tag::Binary => format!(
                "({:?} {} {})",
                node.operator(),
                render(source, ast, node.first_child()),
                render(source, ast, node.first_child() + 1)
            ),
            Tag::Call | Tag::Case | Tag::List => {
                let children = (0..node.child_count())
                    .map(|i| render(source, ast, node.first_child() + i))
                    .collect::<alloc::vec::Vec<_>>();
                format!("({:?} {})", node.tag(), children.join(" "))
            }
            _ => {
                let span = node.span();
                String::from(&source[span.start as usize..(span.start + span.len) as usize])
            }
        }
    }

    fn parse(source: &str) -> String {
        let env = Env::current();
        let ast = parse_expression(env, source.as_bytes()).unwrap();
        render(source, &ast, ast.root())
    }

    fn error(source: &str) -> (ErrorCode, u32) {
        let env = Env::current();
        let error = parse_expression(env, source.as_bytes()).err().unwrap();
        (error.code, error.span.start)
    }

    #[test]
    fn running_out_of_memory_comes_before_later_errors() {
        // The first block and the slot table, but not a second block.
        // SAFETY: declared before what's allocated from it, so dropped after.
        let env = unsafe { Env::new(Heap, 2 * BLOCK_BYTES) }.unwrap();
        let source = format!("{}+)", "1+".repeat(BLOCK_BYTES / 8));
        let error = parse_expression(&env, source.as_bytes()).err().unwrap();
        assert_eq!(error.code, ErrorCode::OutOfMemory);
    }

    #[test]
    fn parses_with_precedence() {
        assert_eq!(
            parse("a + b * -c > 1 AND NOT d = 'x' or e"),
            "(Or (And (Greater (Add a (Multiply b (Negate c))) 1) (Not (Equal d 'x'))) e)"
        );
        assert_eq!(
            parse("(a - b) - c * (d / 2.5) % e"),
            "(Subtract (Subtract a b) (Modulo (Multiply c (Divide d 2.5)) e))"
        );
    }

    #[test]
    fn parses_in_lists() {
        assert_eq!(
            parse("NOT a IN (-1, 6) AND b in ('x')"),
            "(And (Not (In a (List (Negate 1) 6))) (In b (List 'x')))"
        );
        // Lists hold only literals, and can be longer than calls can.
        let long = (0..1000).map(|i| format!("{i}")).collect::<alloc::vec::Vec<_>>().join(", ");
        assert!(parse(&format!("a IN ({long})")).starts_with("(In a (List 0 1 2 "));
        assert_eq!(error("a IN ()"), (ErrorCode::ExpectedExpression, 6));
        assert_eq!(error("a IN (b)"), (ErrorCode::ExpectedExpression, 6));
        assert_eq!(error("a IN (-'x')"), (ErrorCode::ExpectedExpression, 7));
        assert_eq!(error("a IN (1 2)"), (ErrorCode::ExpectedToken, 8));
        assert_eq!(error("a IN 1"), (ErrorCode::ExpectedToken, 5));
        // `IN` and `IS` are only keywords after an operand.
        assert_eq!(parse("in IN (1) OR is IS NULL"), "(Or (In in (List 1)) (IsNull is))");
    }

    #[test]
    fn parses_distinct_arguments() {
        assert_eq!(parse("count(DISTINCT a)"), "(Call count (Distinct a))");
        assert_eq!(parse("sum(distinct a + 1)"), "(Call sum (Distinct (Add a 1)))");
        // Only as the start of a call's argument, and errors point at it.
        assert_eq!(error("count((DISTINCT a))"), (ErrorCode::ExpectedExpression, 7));
        assert_eq!(error("count(1 + DISTINCT a)"), (ErrorCode::ExpectedExpression, 10));
        assert_eq!(error("DISTINCT a"), (ErrorCode::ExpectedExpression, 0));
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
    fn parses_past_the_first_block() {
        let env = Env::current();
        let long = vec!["a"; crate::ast::BLOCK_NODES].join("+");
        let ast = parse_expression(env, long.as_bytes()).unwrap();
        assert_eq!(ast.node_count() as usize, 2 * crate::ast::BLOCK_NODES - 1);
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under Miri")]
    fn reports_running_out_of_blocks() {
        let nodes_max = crate::ast::BLOCK_NODES * crate::ast::BLOCK_SLOTS;
        if nodes_max > 1 << 16 {
            // Too slow at the default settings. CI runs it with small blocks.
            return;
        }
        let long = vec!["a"; nodes_max / 2 + 1].join("+");
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
