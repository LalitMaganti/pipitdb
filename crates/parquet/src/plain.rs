//! Parquet's plain encoding: values one after another, fixed-width ones as
//! they are, strings each after its length.

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

/// Finds the first `valid` plain strings in `bytes` from `at`, a length and
/// then its bytes each, writing where each is to `words`, and returns where
/// they end.
pub(crate) fn string_words(
    bytes: &[u8],
    mut at: usize,
    words: &mut [u64],
    valid: usize,
) -> Result<usize, Error> {
    for word in at_mut!(words, ..valid) {
        let len = length(bytes, at)?;
        if at + 4 + len > bytes.len() {
            return Err(Error::Corrupt);
        }
        let Ok(start) = u32::try_from(at) else { return Err(Error::Unsupported) };
        *word = u64::from(start);
        at += 4 + len;
    }
    Ok(at)
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
