//! A small Parquet reader, reading through a `ByteSource`, so files can be
//! in memory, on local disk or in object storage. Flat schemas only.

#![no_std]

#[macro_use]
extern crate pipit_kernel;

pub mod chunk;
pub mod footer;
mod hybrid;
pub mod table;
mod thrift;

pub use pipit_kernel::error::Error;

/// Decompresses pages. The codecs aren't in this crate, which stays small:
/// `pipitdb-parquet-full` has them.
pub trait Codec {
    /// Decompresses `input`, compressed with Parquet's codec `codec`, into all
    /// of `output`.
    fn decompress(&self, codec: u8, input: &[u8], output: &mut [u8]) -> Result<(), Error>;
}

/// No codecs, so only uncompressed files can be read.
pub struct Uncompressed;

impl Codec for Uncompressed {
    fn decompress(&self, _: u8, _: &[u8], _: &mut [u8]) -> Result<(), Error> {
        Err(Error::Unsupported)
    }
}
