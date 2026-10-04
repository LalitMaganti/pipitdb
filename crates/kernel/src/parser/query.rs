//! Queries: a source, then stages after `|>`. Each stage is found by its
//! keyword in a `Registry`, and its rule's items are parsed in order.

use crate::ast::{Node, Tag};
use crate::error::{Error, ErrorCode};
use crate::lexer::TokenKind;
use crate::registry::{ITEMS_MAX, Item, Point, Registry, Shared};

use super::{LIST_MAX, Parser};

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
