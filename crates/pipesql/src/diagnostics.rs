//! PipeSQL's error messages, laid out by `pipitdb-diagnostics`.

use alloc::format;
use alloc::string::String;

use pipitdb_diagnostics::{Compact, Message, found};

use crate::error::{Error, ErrorCode, Span, Unsupported};
use crate::lexer::TokenKind;

/// `error` in its compact form, `pipit:E0007:2+0:5`.
pub fn compact(error: &Error) -> String {
    let Span { start, len } = error.span;
    Compact { code: error.code as u16, start, len, detail: error.detail }.format()
}

/// The error a compact form describes, or `None` if it isn't one.
pub fn parse_compact(text: &str) -> Option<Error> {
    let Compact { code, start, len, detail } = Compact::parse(text)?;
    Some(Error { code: ErrorCode::from_u16(code)?, detail, span: Span { start, len } })
}

/// `error` as a message about `source`, which is called `name`.
pub fn render(source: &str, name: &str, error: &Error) -> String {
    let Span { start, len } = error.span;
    let message = describe(&found(source, start, len), error);
    pipitdb_diagnostics::render(source, name, error.code as u16, start, len, &message)
}

/// The title and the label under the source for `error`.
fn describe(found: &str, error: &Error) -> Message {
    let kind = u8::try_from(error.detail).ok().and_then(TokenKind::from_u8);
    let expected = kind.map_or("?", token);
    let (title, label): (String, String) = match error.code {
        ErrorCode::QueryTooLarge => {
            ("the query is too large to parse".into(), "parsing stopped here".into())
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
            format!("nesting is limited to {} levels", crate::parser::NESTING_MAX),
        ),
        ErrorCode::OutOfMemory => ("out of memory".into(), "while parsing this query".into()),
        ErrorCode::UnknownStage => (format!("unknown stage {found}"), "not a stage".into()),
        ErrorCode::ExpectedSource => (
            format!("a query must start with a source, found {found}"),
            "expected a source, such as `FROM`".into(),
        ),
        ErrorCode::UnexpectedSource => {
            (format!("{found} can only start a query"), "a source can't follow `|>`".into())
        }
        ErrorCode::ListTooLong => (
            "the list is too long".into(),
            format!("lists are limited to {} items", crate::parser::LIST_MAX),
        ),
        ErrorCode::UnknownTable => (format!("no table {found}"), "not in the catalog".into()),
        ErrorCode::UnknownColumn => (format!("no column {found}"), "not a column here".into()),
        ErrorCode::NumberTooLarge => {
            (format!("{found} is too large"), "doesn't fit in 64 bits".into())
        }
        ErrorCode::ConditionTooLarge => (
            "the condition is too large".into(),
            format!(
                "conditions are limited to {} parts",
                pipit_kernel::predicate::PREDICATE_NODES_MAX
            ),
        ),
        ErrorCode::Unsupported => match error.detail {
            what if what == Unsupported::SelectExpression as u16 => {
                ("only column names can be selected yet".into(), "not a column's name".into())
            }
            what if what == Unsupported::Where as u16 => (
                "only comparisons of a column with a number can filter yet".into(),
                "can't filter on this yet".into(),
            ),
            what if what == Unsupported::NumberType as u16 => (
                format!("{found} can't be compared with this column yet"),
                "a float can't hold this integer exactly".into(),
            ),
            what if what == Unsupported::Decimal as u16 => (
                "numbers with a decimal point can't be compared yet".into(),
                "not an integer".into(),
            ),
            _ => ("not supported yet".into(), "this can't be run yet".into()),
        },
    };
    Message { title, label }
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

#[cfg(test)]
mod tests {
    use pipit_kernel::allocator::Heap;

    use crate::parser::parse_expression;

    use super::*;

    #[test]
    fn renders_a_parse_error() {
        let source = "a +\n  (b * c";
        let error = parse_expression(Heap, source.as_bytes()).unwrap_err();
        assert_eq!(
            render(source, "query", &error),
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
