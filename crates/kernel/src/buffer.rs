//! `Buffer`: the block of bytes every column is built from.
//!
//! A buffer is aligned, refcounted and immutable once shared. It points at
//! its owner, which knows how to free it: the allocator it came from today,
//! and a file mapping later. Releasing the last reference frees through the
//! owner, so buffers from different allocators mix freely.

use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::block::{BLOCK_ALIGNMENT_BYTES, Block, Owner};

/// Alignment of every buffer's bytes: a cache line, the widest SIMD register
/// (AVX-512), and what Arrow recommends.
pub const BUFFER_ALIGNMENT_BYTES: usize = BLOCK_ALIGNMENT_BYTES;

/// A refcounted, 64-byte aligned block of bytes.
///
/// Cloning retains the block; dropping releases it.
pub struct Buffer {
    data: NonNull<u8>,
    size_bytes: usize,
    owner: NonNull<Owner>,
}

impl Buffer {
    /// Allocates `size_bytes` bytes from `allocator`.
    ///
    /// The bytes are zeroed, so stale memory never leaks into a column.
    ///
    /// # Errors
    ///
    /// Returns `AllocError` if the allocator refuses or the size is too
    /// large to lay out.
    pub fn allocate<A: Allocator + 'static>(
        allocator: A,
        size_bytes: usize,
    ) -> Result<Buffer, AllocError> {
        let block = Block::allocate(allocator, (), size_bytes)?;
        // SAFETY: the block was just allocated.
        let data = unsafe { Block::trailing(block) };
        Ok(Buffer {
            data,
            size_bytes,
            owner: Block::owner(block),
        })
    }

    /// The number of bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// The bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `data` is valid for `size_bytes` reads while any reference
        // is alive, and nothing writes while it is shared (see
        // `as_bytes_mut`).
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.size_bytes) }
    }

    /// The bytes, for writing. Buffers are filled before they are shared.
    ///
    /// # Panics
    ///
    /// If the buffer is shared: writing would change other holders' bytes.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        assert!(self.is_unique());
        // SAFETY: as in `as_bytes`, and this is the only reference, so the
        // bytes are not aliased.
        unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr(), self.size_bytes) }
    }

    /// Whether this is the only reference to the bytes.
    #[must_use]
    pub fn is_unique(&self) -> bool {
        self.owner().is_unique()
    }

    fn owner(&self) -> &Owner {
        // SAFETY: the owner outlives every buffer that references it.
        unsafe { self.owner.as_ref() }
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Buffer {
        self.owner().retain();
        Buffer {
            data: self.data,
            size_bytes: self.size_bytes,
            owner: self.owner,
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: the owner is live and this buffer holds a reference.
        unsafe { Owner::release(self.owner) };
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;
    use core::cell::Cell;

    use super::*;
    use crate::allocator::Heap;
    use crate::allocator::testing::{Counting, Refusing};

    #[test]
    fn allocate_is_zeroed_and_aligned() {
        let buffer = Buffer::allocate(Heap, 100).unwrap();
        assert_eq!(buffer.size_bytes(), 100);
        assert!(buffer.as_bytes().iter().all(|&byte| byte == 0));
        assert!(
            buffer
                .as_bytes()
                .as_ptr()
                .addr()
                .is_multiple_of(BUFFER_ALIGNMENT_BYTES)
        );
    }

    #[test]
    fn allocate_empty() {
        let buffer = Buffer::allocate(Heap, 0).unwrap();
        assert!(buffer.as_bytes().is_empty());
    }

    #[test]
    fn clone_shares_bytes() {
        let mut buffer = Buffer::allocate(Heap, 8).unwrap();
        buffer.as_bytes_mut()[0] = 7;
        let clone = buffer.clone();
        assert_eq!(clone.as_bytes().as_ptr(), buffer.as_bytes().as_ptr());
        assert_eq!(clone.as_bytes()[0], 7);
    }

    #[test]
    fn last_drop_frees_through_allocator() {
        let live = Rc::new(Cell::new(0));
        let buffer = Buffer::allocate(Counting { live: live.clone() }, 8).unwrap();
        let clone = buffer.clone();
        assert_eq!(live.get(), 1);
        drop(buffer);
        assert_eq!(live.get(), 1);
        drop(clone);
        assert_eq!(live.get(), 0);
    }

    #[test]
    fn refused_allocation_is_an_error() {
        assert_eq!(Buffer::allocate(Refusing, 8).err(), Some(AllocError));
    }

    #[test]
    fn oversized_allocation_is_an_error() {
        assert_eq!(Buffer::allocate(Heap, usize::MAX).err(), Some(AllocError));
    }

    #[test]
    #[should_panic(expected = "is_unique")]
    fn writing_shared_bytes_panics() {
        let mut buffer = Buffer::allocate(Heap, 8).unwrap();
        let _clone = buffer.clone();
        buffer.as_bytes_mut();
    }
}
