//! `Error`: why a query couldn't be prepared, or a run failed.

use crate::allocator::AllocError;
use crate::bytes::ReadError;
use crate::expression::ExprError;
use crate::lower::LowerError;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    /// The run's memory budget was used up.
    OutOfMemory,
    /// Data couldn't be read or written, such as from a disk or network
    /// failure.
    Io,
    /// Data is damaged, cut short, or not what it claims to be.
    Corrupt,
    /// Data uses something that isn't supported.
    Unsupported,
    /// An integer result didn't fit in its type.
    Overflow,
    /// A query was run in a session other than the one it was prepared in.
    WrongSession,
    /// A query needs more columns than a plan can have.
    TooManyColumns,
    /// A query's values aren't of the types they're used as.
    WrongType(TypeError),
    /// A query nests deeper than a plan can.
    TooDeep,
    /// A query, or one of its expressions, is bigger than a plan can hold.
    TooLarge,
    /// A relation, column or expression of one builder was given to
    /// another.
    WrongBuilder,
    /// A query reads a column where it isn't: not of the table, or not of
    /// the relation an operation reads.
    NotInScope,
}

/// How a query's values aren't of the types they're used as.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TypeError {
    /// Numbers, not strings, in arithmetic.
    Numbers,
    /// Integers on both sides of `%`.
    Integers,
    /// Values of one kind, numbers or strings, on both sides of a comparison.
    Kinds,
    /// Values of one type on both sides of a comparison, unless one is an
    /// integer literal compared with a double.
    Types,
    /// An integer literal compared with a double which can hold it exactly.
    Inexact,
    /// A double in a list of integers, as `IN`'s.
    DoubleIn,
    /// A string, not a number, on the left of `LIKE`.
    Strings,
    /// Integer arithmetic in a value only some rows use, such as a branch
    /// of `CASE`, which could overflow on the others.
    SomeRows,
    /// An integer as a condition: of a filter, a join, a branch of `CASE`,
    /// or an operand of `AND`, `OR` or `NOT`.
    Condition,
    /// Values of `CASE`, of one type.
    Choices,
    /// An argument of the type its function takes.
    Arguments,
    /// An integer or a string as a key rows are grouped by.
    GroupKey,
    /// A column of a type an aggregate reads, or none for a count of rows.
    Aggregated,
}

impl From<AllocError> for Error {
    fn from(_: AllocError) -> Error {
        Error::OutOfMemory
    }
}

impl From<LowerError> for Error {
    fn from(error: LowerError) -> Error {
        match error {
            LowerError::OutOfMemory => Error::OutOfMemory,
            LowerError::TooManyColumns => Error::TooManyColumns,
            LowerError::WrongType(_) => Error::WrongType(TypeError::Types),
            LowerError::TooDeep => Error::TooDeep,
        }
    }
}

impl From<ExprError> for Error {
    fn from(error: ExprError) -> Error {
        let wrong = match error {
            ExprError::OutOfMemory => return Error::OutOfMemory,
            ExprError::Numbers => TypeError::Numbers,
            ExprError::Integers => TypeError::Integers,
            ExprError::Kinds => TypeError::Kinds,
            ExprError::Types => TypeError::Types,
            ExprError::Inexact => TypeError::Inexact,
            ExprError::DoubleIn => TypeError::DoubleIn,
            ExprError::Strings => TypeError::Strings,
            ExprError::SomeRows => TypeError::SomeRows,
            ExprError::Condition => TypeError::Condition,
            ExprError::Choices => TypeError::Choices,
            ExprError::Arguments => TypeError::Arguments,
        };
        Error::WrongType(wrong)
    }
}

impl From<ReadError> for Error {
    fn from(_: ReadError) -> Error {
        Error::Io
    }
}
