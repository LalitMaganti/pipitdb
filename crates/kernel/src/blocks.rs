//! `Blocks`: data that grows, as records of `width` values each, in blocks
//! that never move. Growing adds a block rather than copying everything
//! into one twice the size, so it never holds the old and the new at once.

use core::marker::PhantomData;

use crate::allocator::{AllocError, Allocator};
use crate::array::Array;
use crate::buffer::Primitive;
use crate::slow_vec::SlowVec;

/// How many records a block holds, as a power of two, so a record's block
/// is its number shifted. A block of 8-byte values is `width` times 2 MB, so
/// huge pages back it whole.
pub(crate) const BLOCK_SHIFT: u32 = if cfg!(test) { 2 } else { 18 };
const BLOCK_RECORDS: usize = 1 << BLOCK_SHIFT;

/// How many records the first block holds at first. It doubles from there
/// until it's a whole block, so small data stays small.
const FIRST_RECORDS: usize = if cfg!(test) { 1 } else { 1 << 10 };

/// The most blocks there can be.
const BLOCKS_MAX: usize = 1 << 16;

/// Where record `i` of `width` values starts, given where each block starts.
/// Callers only pass records whose block exists.
#[inline]
pub(crate) fn address<T>(starts: &[*mut T], width: usize, i: usize) -> *mut T {
    if cfg!(debug_assertions) {
        check!(i >> BLOCK_SHIFT < starts.len());
    }
    // SAFETY: as the caller promises, the record's block exists.
    unsafe { starts.get_unchecked(i >> BLOCK_SHIFT).add((i & (BLOCK_RECORDS - 1)) * width) }
}

pub(crate) struct Blocks<T: Primitive> {
    raw: RawBlocks,
    /// How many values a record is.
    width: usize,
    holds: PhantomData<T>,
}

/// What `Blocks` keeps that doesn't depend on its values' type, shared by
/// every `Blocks`, as `SlowVec` shares its code.
struct RawBlocks {
    /// The blocks, as bytes: their alignment suits any `Primitive`.
    arrays: SlowVec<Array<u8>>,
    /// Where each of `arrays` starts.
    starts: SlowVec<*mut u8>,
    /// How many bytes a record is.
    record_bytes: usize,
    /// How many records the first block has room for: `BLOCK_RECORDS` once
    /// there's a second.
    first: usize,
    /// How many records there are.
    len: usize,
}

impl RawBlocks {
    fn capacity(&self) -> usize {
        self.first + self.arrays.len().saturating_sub(1) * BLOCK_RECORDS
    }

    /// As `Blocks::reserve`.
    #[inline(never)]
    fn reserve(&mut self, allocator: &dyn Allocator, len: usize) -> Result<(), AllocError> {
        while self.capacity() < len {
            // The first block grows, in place of itself, until it's whole.
            let grow = self.first < BLOCK_RECORDS;
            let records = if grow {
                len.next_power_of_two().clamp(FIRST_RECORDS, BLOCK_RECORDS)
            } else {
                BLOCK_RECORDS
            };
            let mut block = Array::zeroed(allocator, records * self.record_bytes)?;
            if grow && let Some(old) = self.arrays.pop() {
                self.starts.pop();
                let used = self.len * self.record_bytes;
                at_mut!(block, ..used).copy_from_slice(at!(old, ..used));
            }
            if grow {
                self.first = records;
            }
            self.starts.push(block.as_mut_ptr())?;
            self.arrays.push(block)?;
        }
        Ok(())
    }
}

impl<T: Primitive> Blocks<T> {
    pub(crate) fn new(allocator: &dyn Allocator, width: usize) -> Result<Blocks<T>, AllocError> {
        check!(width > 0);
        let raw = RawBlocks {
            arrays: SlowVec::new(allocator, BLOCKS_MAX)?,
            starts: SlowVec::new(allocator, BLOCKS_MAX)?,
            record_bytes: width * size_of::<T>(),
            first: 0,
            len: 0,
        };
        Ok(Blocks { raw, width, holds: PhantomData })
    }

    pub(crate) fn len(&self) -> usize {
        self.raw.len
    }

    /// Where each block starts: record `i` is `width * (i % BLOCK_RECORDS)`
    /// values into block `i >> BLOCK_SHIFT`.
    pub(crate) fn starts(&self) -> &[*mut T] {
        let starts = &self.raw.starts;
        // SAFETY: pointers to bytes and to `T`s are alike, and every block
        // is aligned for any `Primitive`.
        unsafe { core::slice::from_raw_parts(starts.as_ptr().cast(), starts.len()) }
    }

    /// How many records there's room for.
    pub(crate) fn capacity(&self) -> usize {
        self.raw.capacity()
    }

    /// Makes room for `len` records, zeroed: the first block grows by
    /// doubling until it's whole, then whole blocks are added.
    pub(crate) fn reserve(
        &mut self,
        allocator: &dyn Allocator,
        len: usize,
    ) -> Result<(), AllocError> {
        self.raw.reserve(allocator, len)
    }

    /// Adds a record, zeroed, in the room `reserve` made, and returns it.
    pub(crate) fn push(&mut self) -> &mut [T] {
        check!(self.raw.len < self.capacity());
        self.raw.len += 1;
        // SAFETY: there's room for it, checked above, and it's this one's.
        unsafe { core::slice::from_raw_parts_mut(self.record_ptr(self.raw.len - 1), self.width) }
    }

    /// Record `i`'s first value, for loops that can't afford a check.
    /// Callers only pass `i` below the capacity.
    #[inline]
    pub(crate) fn record_ptr(&self, i: usize) -> *mut T {
        if cfg!(debug_assertions) {
            check!(i < self.capacity());
        }
        address(self.starts(), self.width, i)
    }

    /// Record `i`.
    pub(crate) fn record(&self, i: usize) -> &[T] {
        check!(i < self.raw.len);
        // SAFETY: the record exists, checked above.
        unsafe { core::slice::from_raw_parts(self.record_ptr(i), self.width) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;

    #[test]
    fn keeps_records_across_blocks() {
        let mut blocks = Blocks::<u64>::new(&Heap, 2).unwrap();
        for i in 0..11 {
            blocks.reserve(&Heap, i + 1).unwrap();
            blocks.push().copy_from_slice(&[i as u64, 0]);
        }
        assert_eq!(blocks.len(), 11);
        assert_eq!(
            (0..11).map(|i| blocks.record(i)[0]).collect::<alloc::vec::Vec<_>>(),
            (0..11).collect::<alloc::vec::Vec<_>>()
        );
        // The first block grew to a whole one; the rest are whole.
        assert_eq!(blocks.starts().len(), 11_usize.div_ceil(BLOCK_RECORDS));
    }
}
