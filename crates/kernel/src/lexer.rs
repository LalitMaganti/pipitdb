//! Splits query text into tokens.
//!
//! Keywords come out as identifiers: the parser decides what they mean, so
//! modules can add their own.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
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

// Classes of bytes that aren't a token on their own.
const SPACE: u8 = 32;
// Identifiers continue with `DIGIT..=IDENTIFIER`.
const DIGIT: u8 = 33;
const IDENTIFIER: u8 = 34;
const QUOTE: u8 = 35;
const PIPE: u8 = 36;
const BANG: u8 = 37;
const LESS: u8 = 38;
const GREATER: u8 = 39;
const INVALID: u8 = 40;

const _: () = assert!((TokenKind::End as u8) < SPACE);

/// The class of each byte, as in SQLite's `aiClass`. A byte that is a token
/// on its own maps to its `TokenKind`.
static CLASSES: [u8; 256] = classes();

const fn classes() -> [u8; 256] {
    let mut classes = [INVALID; 256];
    let mut byte: u8 = 0;
    loop {
        classes[byte as usize] = match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => IDENTIFIER,
            b'0'..=b'9' => DIGIT,
            _ if byte.is_ascii_whitespace() => SPACE,
            b'\'' => QUOTE,
            b'|' => PIPE,
            b'!' => BANG,
            b'<' => LESS,
            b'>' => GREATER,
            b'(' => TokenKind::LeftParen as u8,
            b')' => TokenKind::RightParen as u8,
            b',' => TokenKind::Comma as u8,
            b'.' => TokenKind::Dot as u8,
            b';' => TokenKind::Semicolon as u8,
            b'*' => TokenKind::Star as u8,
            b'+' => TokenKind::Plus as u8,
            b'-' => TokenKind::Minus as u8,
            b'/' => TokenKind::Slash as u8,
            b'=' => TokenKind::Equal as u8,
            _ => INVALID,
        };
        if byte == u8::MAX {
            return classes;
        }
        byte += 1;
    }
}

fn class(byte: u8) -> u8 {
    CLASSES[byte as usize]
}

fn is_digit(byte: u8) -> bool {
    class(byte) == DIGIT
}

/// The index of the first `needle` in `haystack`. Checks eight bytes at a
/// time: a byte of `word ^ pattern` is zero where it matches, and
/// `(x - 0x01..) & !x & 0x80..` sets the top bit of the first zero byte.
fn find_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGHS: u64 = 0x8080_8080_8080_8080;
    let pattern = ONES * u64::from(needle);
    let (words, tail) = haystack.as_chunks::<8>();
    for (i, word) in words.iter().enumerate() {
        let x = u64::from_le_bytes(*word) ^ pattern;
        let found = x.wrapping_sub(ONES) & !x & HIGHS;
        if found != 0 {
            return Some(i * 8 + found.trailing_zeros() as usize / 8);
        }
    }
    tail.iter().position(|&byte| byte == needle).map(|i| words.len() * 8 + i)
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
        let class = class(byte);
        if class < SPACE {
            // SAFETY: classes below `SPACE` are `TokenKind`s.
            let kind = unsafe { core::mem::transmute::<u8, TokenKind>(class) };
            return Ok(Token { kind, start, end: self.position });
        }
        let kind = match class {
            IDENTIFIER => {
                self.skip_classes(DIGIT, IDENTIFIER);
                TokenKind::Identifier
            }
            DIGIT => self.number(),
            QUOTE => self.string(start)?,
            PIPE if self.eat(b'>') => TokenKind::Pipe,
            BANG if self.eat(b'=') => TokenKind::NotEqual,
            LESS if self.eat(b'=') => TokenKind::LessEqual,
            LESS if self.eat(b'>') => TokenKind::NotEqual,
            LESS => TokenKind::Less,
            GREATER if self.eat(b'=') => TokenKind::GreaterEqual,
            GREATER => TokenKind::Greater,
            _ => return Err(LexError::UnexpectedCharacter { position: start }),
        };
        Ok(Token { kind, start, end: self.position })
    }

    fn number(&mut self) -> TokenKind {
        self.skip_classes(DIGIT, DIGIT);
        let has_fraction = self.peek(0) == Some(b'.') && self.peek(1).is_some_and(is_digit);
        if !has_fraction {
            return TokenKind::Integer;
        }
        self.position += 1;
        self.skip_classes(DIGIT, DIGIT);
        TokenKind::Float
    }

    /// Strings are in single quotes; `''` inside one is a quote.
    fn string(&mut self, start: u32) -> Result<TokenKind, LexError> {
        loop {
            let Some(quote) = find_byte(self.rest(), b'\'') else {
                return Err(LexError::UnterminatedString { position: start });
            };
            self.advance(quote + 1);
            if !self.eat(b'\'') {
                return Ok(TokenKind::String);
            }
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            self.skip_classes(SPACE, SPACE);
            if self.peek(0) != Some(b'-') || self.peek(1) != Some(b'-') {
                return;
            }
            let rest = self.rest();
            self.advance(find_byte(rest, b'\n').unwrap_or(rest.len()));
        }
    }

    /// Skips bytes whose class is in `first..=last`.
    fn skip_classes(&mut self, first: u8, last: u8) {
        while self.peek(0).is_some_and(|byte| (first..=last).contains(&class(byte))) {
            self.position += 1;
        }
    }

    fn rest(&self) -> &'a [u8] {
        &self.source[self.position as usize..]
    }

    fn advance(&mut self, count: usize) {
        check!(count <= self.rest().len());
        let Ok(count) = u32::try_from(count) else { crate::check::check_failed(line!()) };
        self.position += count;
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
    fn lexes_long_strings_and_comments() {
        use TokenKind::*;
        let query = "-- a comment longer than eight bytes\nx = 'a string longer than eight, with a '' in it'";
        assert_eq!(
            kinds_and_text(query),
            [
                (Identifier, "x"),
                (Equal, "="),
                (String, "'a string longer than eight, with a '' in it'")
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
