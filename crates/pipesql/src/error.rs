//! Errors a caller can cause, such as a bad query. They carry no text:
//! The `diagnostics` feature turns them into messages.

use pipit_kernel::allocator::AllocError;

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
    /// An integer doesn't fit in 64 bits.
    NumberTooLarge = 17,
    /// A condition has more than `PREDICATE_NODES_MAX` parts.
    ConditionTooLarge = 18,
    /// A table has two columns of one name, ignoring case.
    DuplicateColumn = 19,
    /// A name names more than one column, ignoring case.
    AmbiguousColumn = 20,
    /// `AS` isn't next where a stage needs it, as in `RENAME a AS b`.
    ExpectedAs = 21,
    /// A list names a column twice, ignoring case, where it can only once,
    /// as `DROP a, a`.
    ListedTwice = 22,
    /// `DROP` would leave no columns.
    NoColumnsLeft = 23,
}

/// What an `Unsupported` error is about. Only ever added to, as codes are.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u16)]
pub enum Unsupported {
    /// Anything but a column's name where a stage takes a column.
    SelectExpression = 1,
    /// A condition that isn't comparisons of a column with a number,
    /// combined with `AND`, `OR` and `NOT`.
    Where = 2,
    /// An integer compared with a float column that can't hold it exactly.
    NumberType = 3,
    /// A number with a decimal point, which conditions can't compare yet.
    Decimal = 4,
    /// A `LIMIT` or `OFFSET` that isn't one integer.
    Limit = 5,
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

// No code is 0, so `Option<ErrorCode>` uses 0 for `None` and needs no tag.
const _: () = assert!(size_of::<Option<ErrorCode>>() == size_of::<ErrorCode>());

impl ErrorCode {
    /// The highest code. Update it when adding one.
    pub const LAST: ErrorCode = ErrorCode::NoColumnsLeft;

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
