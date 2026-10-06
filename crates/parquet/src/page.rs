//! Page headers.

use crate::thrift::{Cursor, Value};

/// Parquet's number for the RLE and bit-packed hybrid encoding.
pub(crate) const RLE: i64 = 3;

pub(crate) struct Header {
    /// What the page is: 0 for data, 2 for a dictionary.
    pub(crate) kind: i64,
    /// How many bytes the page's body takes, as stored.
    pub(crate) len: u64,
    /// How many bytes the page's body takes once decompressed.
    pub(crate) size: u64,
    /// How many values the page has, nulls included.
    pub(crate) values: usize,
    /// How its values are encoded.
    pub(crate) encoding: i64,
    /// For a data page, how its definition levels are encoded.
    pub(crate) levels: i64,
}

/// A page's header: its type, sizes, and its data or dictionary page
/// header's value count and encoding, and a data page's levels' encoding.
pub(crate) fn header(c: &mut Cursor<'_>) -> Option<Header> {
    let mut header = [Value::Missing; 5];
    c.fields(&[1, 2, 3, 5, 7], &mut header)?;
    let mut data = [Value::Missing; 3];
    if let Some(at) = header[3].at().or(header[4].at()) {
        c.at(at).fields(&[1, 2, 3], &mut data)?;
    }
    Some(Header {
        kind: header[0].int()?,
        size: u64::try_from(header[1].int()?).ok()?,
        len: u64::try_from(header[2].int()?).ok()?,
        values: usize::try_from(data[0].int().unwrap_or(0)).ok()?,
        encoding: data[1].int().unwrap_or(0),
        levels: data[2].int().unwrap_or(RLE),
    })
}
