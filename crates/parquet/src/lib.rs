//! A small Parquet reader, reading through a `ByteSource`, so files can be
//! in memory, on local disk or in object storage. Flat schemas only.

#![no_std]

#[macro_use]
extern crate pipit_kernel;

pub mod chunk;
pub mod footer;
mod hybrid;
mod thrift;

pub use pipit_kernel::error::Error;
