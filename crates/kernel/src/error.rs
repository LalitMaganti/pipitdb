//! `Error`: why a query couldn't be prepared, or a run failed.

use crate::allocator::AllocError;
use crate::bytes::ReadError;
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
    /// Integer arithmetic in an argument of `coalesce`, or a branch of
    /// `CASE`, which only some rows use, which could overflow on the others.
    SomeRows,
    /// A number, not a string, as a condition of `CASE`, or an integer as
    /// an operand of `AND`, `OR` or `NOT`.
    Condition,
    /// Arguments of `coalesce`, or values of `CASE`, of one type.
    Choices,
    /// An argument of the type its function takes.
    Arguments,
    /// A column of a type the operation reading it doesn't take.
    Read,
    /// A double, not an integer, as the condition of a filter or a join.
    DoubleCondition,
    /// A key rows are grouped by that isn't an integer or a string.
    GroupKey,
    /// A column of a type an aggregate doesn't read.
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
            LowerError::WrongType(_) => Error::WrongType(TypeError::Read),
            LowerError::TooDeep => Error::TooDeep,
        }
    }
}

impl From<ReadError> for Error {
    fn from(_: ReadError) -> Error {
        Error::Io
    }
}
