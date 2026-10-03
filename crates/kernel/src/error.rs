//! Errors a caller can cause, such as a bad query. They carry no text:
//! `pipitdb-diagnostics` turns them into messages.

/// The bytes `start..start + len` of the query.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Span {
    pub start: u32,
    pub len: u32,
}

/// Codes are only ever added, never renumbered, so an error printed by one
/// build can be explained by another.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u16)]
pub enum ErrorCode {
    QueryTooLarge = 1,
    UnexpectedCharacter = 2,
    UnterminatedString = 3,
    TokenTooLong = 4,
    ExpectedExpression = 5,
    UnexpectedToken = 6,
    /// `detail` is the `TokenKind` that was expected.
    ExpectedToken = 7,
    NestingTooDeep = 8,
    OutOfMemory = 9,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Error {
    pub code: ErrorCode,
    /// Depends on `code`.
    pub detail: u16,
    pub span: Span,
}

impl Error {
    pub fn new(code: ErrorCode, span: Span) -> Error {
        Error { code, detail: 0, span }
    }
}
