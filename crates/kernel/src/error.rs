//! `Error`: why a run failed.

use crate::allocator::AllocError;
use crate::bytes::ReadError;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    /// The run's memory budget was used up.
    OutOfMemory,
    /// Data couldn't be read, such as from a disk or network failure.
    Read,
    /// Data is damaged, cut short, or not what it claims to be.
    Corrupt,
    /// Data uses something that isn't supported.
    Unsupported,
}

impl From<AllocError> for Error {
    fn from(_: AllocError) -> Error {
        Error::OutOfMemory
    }
}

impl From<ReadError> for Error {
    fn from(_: ReadError) -> Error {
        Error::Read
    }
}
