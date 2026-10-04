//! Errors a caller can cause, such as a bad query. They carry no text:
//! The `diagnostics` feature turns them into messages.

use pipit_kernel::allocator::AllocError;
use pipit_kernel::vec::Full;

/// The bytes `start..start + len` of the query.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Span {
    pub start: u32,
    pub len: u32,
}

/// Codes are only ever added, never renumbered, so an error printed by one
/// build can be explained by another. They start at 1 and have no gaps.
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
    UnknownStage = 10,
    /// A query must start with a source, like `FROM`.
    ExpectedSource = 11,
    /// A source, like `FROM`, can only start a query.
    UnexpectedSource = 12,
    ListTooLong = 13,
    /// `FROM` names a table the catalog doesn't have.
    UnknownTable = 14,
    /// A name isn't one of the columns so far.
    UnknownColumn = 15,
    /// Parsed, but not compiled yet. `detail` is the `Unsupported` it is.
    Unsupported = 16,
}

/// What an `Unsupported` error is about. Only ever added to, as codes are.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u16)]
pub enum Unsupported {
    /// Anything but a column's name in `SELECT`.
    SelectExpression = 1,
    Where = 2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Error {
    pub code: ErrorCode,
    /// Depends on `code`.
    pub detail: u16,
    pub span: Span,
}

/// Running out of memory, which isn't about any one part of the query, so
/// points at its start.
impl From<AllocError> for Error {
    fn from(_: AllocError) -> Error {
        Error::new(ErrorCode::OutOfMemory, Span { start: 0, len: 1 })
    }
}

impl<T> From<Full<T>> for Error {
    fn from(_: Full<T>) -> Error {
        Error::from(AllocError)
    }
}

// No code is 0, so `Option<ErrorCode>` uses 0 for `None` and needs no tag.
const _: () = assert!(size_of::<Option<ErrorCode>>() == size_of::<ErrorCode>());

impl ErrorCode {
    /// The highest code. Update it when adding one.
    pub const LAST: ErrorCode = ErrorCode::Unsupported;

    pub fn from_u16(value: u16) -> Option<ErrorCode> {
        if !(1..=ErrorCode::LAST as u16).contains(&value) {
            return None;
        }
        // SAFETY: `ErrorCode` is a `u16` with no gaps from 1 to `LAST`.
        Some(unsafe { core::mem::transmute::<u16, ErrorCode>(value) })
    }
}

impl Error {
    pub fn new(code: ErrorCode, span: Span) -> Error {
        Error { code, detail: 0, span }
    }

    /// `Unsupported`, for what isn't.
    pub fn unsupported(what: Unsupported, span: Span) -> Error {
        Error { code: ErrorCode::Unsupported, detail: what as u16, span }
    }

    /// `ExpectedToken`, where `kind` was expected.
    pub(crate) fn expected(kind: crate::lexer::TokenKind, span: Span) -> Error {
        Error { code: ErrorCode::ExpectedToken, detail: u16::from(kind as u8), span }
    }
}
