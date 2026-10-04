//! A small Parquet reader, reading through a `ByteSource`, so files can be
//! in memory, on local disk or in object storage. Flat schemas only.

#![no_std]

#[macro_use]
extern crate pipit_kernel;

pub mod chunk;
pub mod footer;
mod hybrid;
mod thrift;

use pipit_kernel::allocator::AllocError;
use pipit_kernel::bytes::ReadError;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    /// The bytes couldn't be read.
    Read,
    OutOfMemory,
    /// They aren't Parquet, or are cut short or damaged.
    Corrupt,
    /// They use something this reader doesn't, such as nested columns.
    Unsupported,
}

impl From<ReadError> for Error {
    fn from(_: ReadError) -> Error {
        Error::Read
    }
}

impl From<AllocError> for Error {
    fn from(_: AllocError) -> Error {
        Error::OutOfMemory
    }
}
