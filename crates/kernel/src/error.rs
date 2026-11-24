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
    /// An operation of a query reads a column of a type it doesn't take.
    WrongType,
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
            LowerError::WrongType(_) => Error::WrongType,
            LowerError::TooDeep => Error::TooDeep,
        }
    }
}

impl From<ReadError> for Error {
    fn from(_: ReadError) -> Error {
        Error::Io
    }
}
