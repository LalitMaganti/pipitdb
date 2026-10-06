//! Parquet's plain encoding: values one after another, fixed-width ones as
//! they are, strings each after its length.

use pipit_kernel::buffer::Buffer;
use pipit_kernel::context::Context;

use crate::Error;
use crate::footer::{Column, Logical, Physical};

/// Decodes plain fixed-width values of `physical` from `bytes` at `at` into
/// `out`, as words, and returns where they end.
pub(crate) fn words(
    column: Column,
    bytes: &[u8],
    at: usize,
    out: &mut [i64],
) -> Result<usize, Error> {
    let end = at + out.len() * width(column.physical);
    let bytes = bytes.get(at..end).ok_or(Error::Corrupt)?;
    match column.physical {
        Physical::Int32 if column.logical == Logical::Unsigned => {
            for (word, value) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *word = i64::from(u32::from_le_bytes(*value));
            }
        }
        Physical::Int32 => {
            for (word, value) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *word = i64::from(i32::from_le_bytes(*value));
            }
        }
        Physical::Float => {
            for (word, value) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *word = f64::from(f32::from_le_bytes(*value)).to_bits().cast_signed();
            }
        }
        _ => {
            for (word, value) in out.iter_mut().zip(bytes.as_chunks::<8>().0) {
                *word = i64::from_le_bytes(*value);
            }
        }
    }
    Ok(end)
}

/// Finds the plain strings in `bytes` from `at`, filling `starts` and
/// `lens` with where each starts and how long it is, and returns where they
/// end.
pub(crate) fn strings(
    bytes: &[u8],
    mut at: usize,
    starts: &mut [u32],
    lens: &mut [u32],
) -> Result<usize, Error> {
    for (start, len_out) in starts.iter_mut().zip(lens) {
        let len = length(bytes, at)?;
        let first = at + 4;
        if first + len > bytes.len() {
            return Err(Error::Corrupt);
        }
        let span32 = (u32::try_from(first), u32::try_from(len));
        let (Ok(start32), Ok(len32)) = span32 else { return Err(Error::Unsupported) };
        (*start, *len_out) = (start32, len32);
        at = first + len;
    }
    Ok(at)
}

/// Copies the strings of `bytes` at `starts`, of `lens`, which are the values
/// of the rows `validity` says aren't null, into a column's offsets and bytes.
pub(crate) fn gather_strings(
    context: &mut Context,
    bytes: &[u8],
    starts: &[u32],
    lens: &[u32],
    rows: usize,
    validity: Option<&[u8]>,
) -> Result<(Buffer, Buffer), Error> {
    let total: usize = lens.iter().map(|&len| len as usize).sum();
    let mut offsets = context.indices_buffer((rows + 1) * 4)?;
    let mut copied = context.bytes_buffer(total)?;
    let (ends, out) = (offsets.as_mut_slice::<u32>(), copied.as_mut_slice::<u8>());
    let (mut to, mut next) = (0, starts.iter().zip(lens));
    *at_mut!(ends, 0) = 0;
    for row in 0..rows {
        if validity.is_none_or(|bits| *at!(bits, row / 8) >> (row % 8) & 1 != 0) {
            let Some((&start, &len)) = next.next() else { return Err(Error::Corrupt) };
            let (start, len) = (start as usize, len as usize);
            let value = bytes.get(start..start + len).ok_or(Error::Corrupt)?;
            at_mut!(out, to..to + len).copy_from_slice(value);
            to += len;
        }
        *at_mut!(ends, row + 1) = u32::try_from(to).map_err(|_| Error::Unsupported)?;
    }
    Ok((offsets, copied))
}

/// A dictionary page's `count` plain strings, and then an empty one, as
/// offsets and bytes.
pub(crate) fn dictionary_strings(
    context: &mut Context,
    page: &[u8],
    count: usize,
) -> Result<(Buffer, Buffer), Error> {
    // Walked once to size the bytes, then again to copy them.
    let (mut at, mut total) = (0, 0);
    for _ in 0..count {
        let len = length(page, at)?;
        (at, total) = (at + 4 + len, total + len);
    }
    let mut offsets = context.bytes_buffer((count + 2) * 4)?;
    let mut bytes = context.bytes_buffer(total)?;
    let (ends, out) = (offsets.as_mut_slice::<u32>(), bytes.as_mut_slice::<u8>());
    let (mut from, mut to) = (0, 0);
    *at_mut!(ends, 0) = 0;
    *at_mut!(ends, count + 1) = u32::try_from(total).map_err(|_| Error::Unsupported)?;
    for end in at_mut!(ends, 1..=count) {
        let len = length(page, from)?;
        let value = page.get(from + 4..from + 4 + len).ok_or(Error::Corrupt)?;
        at_mut!(out, to..to + len).copy_from_slice(value);
        (from, to) = (from + 4 + len, to + len);
        *end = u32::try_from(to).map_err(|_| Error::Unsupported)?;
    }
    Ok((offsets, bytes))
}

/// How many bytes a plain fixed-width value takes.
pub(crate) fn width(physical: Physical) -> usize {
    if matches!(physical, Physical::Int32 | Physical::Float) { 4 } else { 8 }
}

/// The little-endian `u32` length at `at`.
pub(crate) fn length(bytes: &[u8], at: usize) -> Result<usize, Error> {
    let len = bytes.get(at..at + 4).ok_or(Error::Corrupt)?;
    Ok(u32::from_le_bytes([len[0], len[1], len[2], len[3]]) as usize)
}
