//! Splits query text into tokens.
//!
//! Keywords come out as identifiers: the parser decides what they mean, so
//! modules can add their own.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TokenKind {
    Identifier,
    Integer,
    Float,
    String,
    Pipe,
    LeftParen,
    RightParen,
    Comma,
    Dot,
    Semicolon,
    Star,
    Plus,
    Minus,
    Slash,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    End,
}

/// A token is the bytes `start..end` of the source.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LexError {
    UnexpectedCharacter { position: u32 },
    UnterminatedString { position: u32 },
}

pub struct Lexer<'a> {
    source: &'a [u8],
    position: u32,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a [u8]) -> Lexer<'a> {
        check!(u32::try_from(source.len()).is_ok());
        Lexer { source, position: 0 }
    }

    /// Returns `TokenKind::End` once the source is used up.
    pub fn next_token(&mut self) -> Result<Token, LexError> {
        self.skip_whitespace_and_comments();
        let start = self.position;
        let Some(byte) = self.peek(0) else {
            return Ok(Token { kind: TokenKind::End, start, end: start });
        };
        self.position += 1;
        let kind = match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                self.skip_while(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
                TokenKind::Identifier
            }
            b'0'..=b'9' => self.number(),
            b'\'' => self.string(start)?,
            b'|' if self.eat(b'>') => TokenKind::Pipe,
            b'(' => TokenKind::LeftParen,
            b')' => TokenKind::RightParen,
            b',' => TokenKind::Comma,
            b'.' => TokenKind::Dot,
            b';' => TokenKind::Semicolon,
            b'*' => TokenKind::Star,
            b'+' => TokenKind::Plus,
            b'-' => TokenKind::Minus,
            b'/' => TokenKind::Slash,
            b'=' => TokenKind::Equal,
            b'!' if self.eat(b'=') => TokenKind::NotEqual,
            b'<' if self.eat(b'=') => TokenKind::LessEqual,
            b'<' if self.eat(b'>') => TokenKind::NotEqual,
            b'<' => TokenKind::Less,
            b'>' if self.eat(b'=') => TokenKind::GreaterEqual,
            b'>' => TokenKind::Greater,
            _ => return Err(LexError::UnexpectedCharacter { position: start }),
        };
        Ok(Token { kind, start, end: self.position })
    }

    fn number(&mut self) -> TokenKind {
        self.skip_while(|byte| byte.is_ascii_digit());
        let has_fraction =
            self.peek(0) == Some(b'.') && self.peek(1).is_some_and(|byte| byte.is_ascii_digit());
        if !has_fraction {
            return TokenKind::Integer;
        }
        self.position += 1;
        self.skip_while(|byte| byte.is_ascii_digit());
        TokenKind::Float
    }

    /// Strings are in single quotes; `''` inside one is a quote.
    fn string(&mut self, start: u32) -> Result<TokenKind, LexError> {
        loop {
            match self.peek(0) {
                None => return Err(LexError::UnterminatedString { position: start }),
                Some(b'\'') if self.peek(1) == Some(b'\'') => self.position += 2,
                Some(b'\'') => {
                    self.position += 1;
                    return Ok(TokenKind::String);
                }
                Some(_) => self.position += 1,
            }
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            self.skip_while(|byte| byte.is_ascii_whitespace());
            if self.peek(0) != Some(b'-') || self.peek(1) != Some(b'-') {
                return;
            }
            self.skip_while(|byte| byte != b'\n');
        }
    }

    fn skip_while(&mut self, matches: fn(u8) -> bool) {
        while self.peek(0).is_some_and(matches) {
            self.position += 1;
        }
    }

    fn eat(&mut self, byte: u8) -> bool {
        if self.peek(0) != Some(byte) {
            return false;
        }
        self.position += 1;
        true
    }

    fn peek(&self, offset: u32) -> Option<u8> {
        self.source.get((self.position + offset) as usize).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds_and_text(source: &str) -> alloc::vec::Vec<(TokenKind, &str)> {
        let mut lexer = Lexer::new(source.as_bytes());
        let mut tokens = alloc::vec::Vec::new();
        loop {
            let token = lexer.next_token().unwrap();
            if token.kind == TokenKind::End {
                return tokens;
            }
            tokens.push((token.kind, &source[token.start as usize..token.end as usize]));
        }
    }

    #[test]
    fn lexes_a_query() {
        use TokenKind::*;
        let query = "FROM t -- the table\n|> WHERE x >= 1.5 AND name <> 'it''s' |> SELECT count(*)";
        assert_eq!(
            kinds_and_text(query),
            [
                (Identifier, "FROM"),
                (Identifier, "t"),
                (Pipe, "|>"),
                (Identifier, "WHERE"),
                (Identifier, "x"),
                (GreaterEqual, ">="),
                (Float, "1.5"),
                (Identifier, "AND"),
                (Identifier, "name"),
                (NotEqual, "<>"),
                (String, "'it''s'"),
                (Pipe, "|>"),
                (Identifier, "SELECT"),
                (Identifier, "count"),
                (LeftParen, "("),
                (Star, "*"),
                (RightParen, ")"),
            ]
        );
    }

    #[test]
    fn reports_errors() {
        let mut lexer = Lexer::new(b"x = 'abc");
        lexer.next_token().unwrap();
        lexer.next_token().unwrap();
        assert_eq!(lexer.next_token(), Err(LexError::UnterminatedString { position: 4 }));

        let mut lexer = Lexer::new(b"x | y");
        lexer.next_token().unwrap();
        assert_eq!(lexer.next_token(), Err(LexError::UnexpectedCharacter { position: 2 }));
    }
}
