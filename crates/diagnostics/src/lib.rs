//! Turns pipitdb errors into messages, in the style of rustc.
//!
//! The kernel's errors carry no text so that small builds stay small. They can
//! be printed in a compact form, `pipit:E0007:2+0:5` (code, span start and
//! length, detail), and explained later with `pipit-explain`.

use std::fmt::Write;

use pipit_kernel::error::{Error, ErrorCode, Span};
use pipit_kernel::lexer::TokenKind;

/// `error` in its compact form, `pipit:E0007:2+0:5`.
pub fn compact(error: &Error) -> String {
    let Span { start, len } = error.span;
    format!("pipit:E{:04}:{start}+{len}:{}", error.code as u16, error.detail)
}

/// The error a compact form describes, or `None` if it isn't one.
pub fn parse_compact(text: &str) -> Option<Error> {
    let rest = text.strip_prefix("pipit:E")?;
    let mut parts = rest.split(':');
    let code = parts.next()?.parse::<u16>().ok()?;
    let (start, len) = parts.next()?.split_once('+')?;
    let detail = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    let code = ErrorCode::from_u16(code)?;
    let span = Span { start: start.parse().ok()?, len: len.parse().ok()? };
    Some(Error { code, detail, span })
}

/// `error` as a message about `source`, which is called `name`:
///
/// ```text
/// error[E0007]: expected `)`, found the end of the query
///  --> query:1:3
///   |
/// 1 | (a
///   |   ^ expected `)`
/// ```
pub fn render(error: &Error, source: &str, name: &str) -> String {
    let found = found(error.span, source);
    let (title, label) = describe(error, &found);
    let (line_number, line, column) = locate(source, error.span.start as usize);
    let width = underline_width(line, column, error.span.len as usize);
    let gutter = " ".repeat(line_number.to_string().len());

    let mut out = format!("error[E{:04}]: {title}\n", error.code as u16);
    let _ = writeln!(out, "{gutter}--> {name}:{line_number}:{}", column + 1);
    let _ = writeln!(out, "{gutter} |");
    let _ = writeln!(out, "{line_number} | {line}");
    let carets = "^".repeat(width);
    let _ = writeln!(out, "{gutter} | {}{carets} {label}", " ".repeat(column));
    out
}

/// The title and the label under the source for `error`.
fn describe(error: &Error, found: &str) -> (String, String) {
    let kind = u8::try_from(error.detail).ok().and_then(TokenKind::from_u8);
    let expected = kind.map_or("?", token);
    match error.code {
        ErrorCode::QueryTooLarge => {
            ("the query is too large".into(), "queries are limited to 4 GiB".into())
        }
        ErrorCode::UnexpectedCharacter => {
            (format!("unexpected character {found}"), "not part of any token".into())
        }
        ErrorCode::UnterminatedString => {
            ("unterminated string".into(), "this string has no closing `'`".into())
        }
        ErrorCode::TokenTooLong => ("token too long".into(), "tokens are limited to 16 MiB".into()),
        ErrorCode::ExpectedExpression => {
            (format!("expected an expression, found {found}"), "expected an expression".into())
        }
        ErrorCode::UnexpectedToken => (format!("unexpected {found}"), "unexpected".into()),
        ErrorCode::ExpectedToken => {
            (format!("expected {expected}, found {found}"), format!("expected {expected}"))
        }
        ErrorCode::NestingTooDeep => (
            "expression nests too deeply, or has too many arguments".into(),
            format!("nesting is limited to {} levels", pipit_kernel::parser::NESTING_MAX),
        ),
        ErrorCode::OutOfMemory => ("out of memory".into(), "while parsing this query".into()),
    }
}

/// What the error's span covers, for "found ...".
fn found(span: Span, source: &str) -> String {
    let start = span.start as usize;
    match source.get(start..start + span.len as usize) {
        Some("") | None => "the end of the query".into(),
        Some(text) => format!("`{text}`"),
    }
}

fn token(kind: TokenKind) -> &'static str {
    match kind {
        TokenKind::Identifier => "a name",
        TokenKind::Integer => "an integer",
        TokenKind::Float => "a number",
        TokenKind::String => "a string",
        TokenKind::Pipe => "`|>`",
        TokenKind::LeftParen => "`(`",
        TokenKind::RightParen => "`)`",
        TokenKind::Comma => "`,`",
        TokenKind::Dot => "`.`",
        TokenKind::Semicolon => "`;`",
        TokenKind::Star => "`*`",
        TokenKind::Plus => "`+`",
        TokenKind::Minus => "`-`",
        TokenKind::Slash => "`/`",
        TokenKind::Equal => "`=`",
        TokenKind::NotEqual => "`!=`",
        TokenKind::Less => "`<`",
        TokenKind::LessEqual => "`<=`",
        TokenKind::Greater => "`>`",
        TokenKind::GreaterEqual => "`>=`",
        TokenKind::End => "the end of the query",
    }
}

/// The 1-based line number of byte `offset`, that line's text, and the
/// 0-based column of `offset` in characters.
fn locate(source: &str, offset: usize) -> (usize, &str, usize) {
    let offset = offset.min(source.len());
    let line_start = source[..offset].rfind('\n').map_or(0, |i| i + 1);
    let line_end = source[offset..].find('\n').map_or(source.len(), |i| offset + i);
    let line_number = source[..line_start].matches('\n').count() + 1;
    let column = source[line_start..offset].chars().count();
    (line_number, &source[line_start..line_end], column)
}

/// How many carets to draw: the span's characters on its first line, and at
/// least one.
fn underline_width(line: &str, column: usize, len: usize) -> usize {
    let rest: String = line.chars().skip(column).collect();
    let covered = rest.char_indices().take_while(|&(i, _)| i < len).count();
    covered.max(1)
}

#[cfg(test)]
mod tests {
    use pipit_kernel::allocator::Heap;
    use pipit_kernel::parser::parse_expression;

    use super::*;

    #[test]
    fn renders_a_parse_error() {
        let source = "a +\n  (b * c";
        let error = parse_expression(Heap, source.as_bytes()).unwrap_err();
        assert_eq!(
            render(&error, source, "query"),
            "error[E0007]: expected `)`, found the end of the query\n \
             --> query:2:9\n  |\n2 |   (b * c\n  |         ^ expected `)`\n"
        );
    }

    #[test]
    fn compact_round_trips() {
        let error = parse_expression(Heap, b"a b").unwrap_err();
        let text = compact(&error);
        assert_eq!(text, "pipit:E0006:2+1:0");
        assert_eq!(parse_compact(&text), Some(error));
        assert_eq!(parse_compact("pipit:E0099:0+0:0"), None);
    }
}
