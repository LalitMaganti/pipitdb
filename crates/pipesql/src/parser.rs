//! Parses query text into an `Ast`.
//!
//! `Parser` is the cursor over tokens that every part of the grammar uses, and
//! writes the tree.

use crate::ast::{Ast, BLOCK_BYTES, DATA_MAX, Node, Nodes, Operator, Tag};
use crate::error::{Error, ErrorCode, Span};
use crate::lexer::{Lexer, Token, TokenKind};
use crate::registry::{ITEMS_MAX, Item, Point, Registry, Shared};
use pipit_kernel::allocator::Allocator;
use pipit_kernel::buffer::Buffer;

/// How deeply operators, parentheses and calls can nest. Each waiting
/// argument counts as a level.
pub const NESTING_MAX: usize = 64;

/// How many stages a query, or items a list, can have.
pub const LIST_MAX: usize = 64;

/// Parses `source` as a query, with the stages in `registry`.
pub fn parse_query<A: Allocator + Clone + 'static>(
    allocator: A,
    registry: &Registry,
    source: &[u8],
) -> Result<Ast, Error> {
    let mut parser = Parser::new(allocator, source)?;
    let root = parser.query(registry)?;
    let current = parser.current();
    if current.kind != TokenKind::End {
        return Err(Error::new(ErrorCode::UnexpectedToken, current.span));
    }
    parser.finish(root)
}

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
    nodes: Nodes,
    failed: Option<ErrorCode>,
}

impl<'a> Parser<'a> {
    pub(crate) fn new<A: Allocator + Clone + 'static>(
        allocator: A,
        source: &'a [u8],
    ) -> Result<Parser<'a>, Error> {
        let mut lexer = Lexer::new(source)?;
        let current = lexer.next_token()?;
        // SAFETY: nodes are read only once written.
        let first = unsafe { Buffer::allocate_uninit(allocator, BLOCK_BYTES) }
            .map_err(|_| Error::new(ErrorCode::OutOfMemory, Span { start: 0, len: 1 }))?;
        Ok(Parser { source, lexer, current, nodes: Nodes::new(first), failed: None })
    }

    /// The token the parser is on.
    pub(crate) fn current(&self) -> Token {
        self.current
    }

    /// Moves to the next token and returns the one it was on.
    pub(crate) fn advance(&mut self) -> Result<Token, Error> {
        if let Some(code) = self.failed {
            return Err(Error::new(code, self.current.span));
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
        let first = self.nodes.len();
        for &node in nodes {
            self.push(node);
        }
        first
    }

    pub(crate) fn node_count(&self) -> u32 {
        self.nodes.len()
    }

    /// Writes `root` to the tree, last, and returns the tree.
    pub(crate) fn finish(mut self, root: Node) -> Result<Ast, Error> {
        self.push(root);
        if let Some(code) = self.failed {
            return Err(Error::new(code, self.current.span));
        }
        Ok(Ast::new(self.nodes))
    }

    /// If the next block can't be had, later nodes overwrite the last block,
    /// which is safe as `failed` stops the parse before the tree is read.
    fn push(&mut self, node: Node) {
        self.nodes.push(node);
        if self.nodes.needs_block()
            && let Err(code) = self.nodes.grow()
        {
            self.failed = Some(code);
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
                    Pending::Call(_) | Pending::Argument(_) => {
                        pipit_kernel::check::check_failed(line!())
                    }
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
        }
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

    /// Writes `children` to the tree, and returns a `tag` node holding them.
    fn list(&mut self, tag: Tag, children: &[Node]) -> Node {
        let first_child = self.write(children);
        Node::list(tag, self.node_count() - first_child, first_child)
    }
}

#[cfg(test)]
mod tests {
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
            Tag::Call => {
                let children = (0..node.child_count())
                    .map(|i| render(source, ast, node.first_child() + i))
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
        render(source, &ast, ast.root())
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
    fn parses_past_the_first_block() {
        let long = vec!["a"; crate::ast::BLOCK_NODES].join("+");
        let ast = parse_expression(Heap, long.as_bytes()).unwrap();
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
