//! `Buffer`: the block of bytes every column is built from.
//!
//! A buffer is aligned, refcounted and immutable once shared. It points at
//! its owner, which knows how to free it: the allocator it came from today,
//! and a file mapping later. Releasing the last reference frees through the
//! owner, so buffers from different allocators mix freely.

use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};

/// Alignment of every buffer's bytes: a cache line, the widest SIMD register
/// (AVX-512), and what Arrow recommends.
pub const BUFFER_ALIGNMENT_BYTES: usize = 64;

const _: () = assert!(BUFFER_ALIGNMENT_BYTES.is_power_of_two());

/// The part common to every kind of owner. It sits at the start of each
/// owner, so a buffer can release an owner without knowing its type.
///
/// The count is not atomic: execution is single-threaded for now.
#[repr(C)]
struct Owner {
    references: Cell<u32>,
    release: unsafe fn(NonNull<Owner>),
}

/// The owner of a buffer allocated from `A`. The bytes follow it in the same
/// block, so a buffer costs one allocation.
#[repr(C, align(64))]
struct AllocatedOwner<A: Allocator> {
    // Must be the first field: buffers cast between the two types.
    owner: Owner,
    allocator: A,
    // The whole block: this header followed by the bytes.
    layout: Layout,
}

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
        const { assert!(align_of::<AllocatedOwner<A>>() == BUFFER_ALIGNMENT_BYTES) };
        let header_bytes = size_of::<AllocatedOwner<A>>();
        let block_bytes = header_bytes.checked_add(size_bytes).ok_or(AllocError)?;
        let layout =
            Layout::from_size_align(block_bytes, BUFFER_ALIGNMENT_BYTES).map_err(|_| AllocError)?;

        let block = allocator.allocate(layout)?;
        assert!(block.as_ptr().addr().is_multiple_of(BUFFER_ALIGNMENT_BYTES));

        let header = block.cast::<AllocatedOwner<A>>();
        let owner = Owner {
            references: Cell::new(1),
            release: release_allocated::<A>,
        };
        // SAFETY: the block is valid for `layout`, which starts with room for
        // the header and is aligned for it.
        unsafe {
            header.write(AllocatedOwner {
                owner,
                allocator,
                layout,
            });
        }

        // SAFETY: `header_bytes <= block_bytes`, so this stays in the block.
        let data = unsafe { block.add(header_bytes) };
        assert!(data.as_ptr().addr().is_multiple_of(BUFFER_ALIGNMENT_BYTES));
        // SAFETY: `data` is valid for `size_bytes` writes, the rest of the
        // block.
        unsafe { data.write_bytes(0, size_bytes) };

        Ok(Buffer {
            data,
            size_bytes,
            owner: header.cast(),
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
        self.owner().references.get() == 1
    }

    fn owner(&self) -> &Owner {
        // SAFETY: the owner outlives every buffer that references it.
        unsafe { self.owner.as_ref() }
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Buffer {
        let references = &self.owner().references;
        let count = references.get();
        assert!(count > 0);
        assert!(count < u32::MAX);
        references.set(count + 1);
        Buffer {
            data: self.data,
            size_bytes: self.size_bytes,
            owner: self.owner,
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let owner = self.owner();
        let count = owner.references.get();
        assert!(count > 0);
        owner.references.set(count - 1);
        if count == 1 {
            let release = owner.release;
            // SAFETY: this was the last reference, and `release` matches the
            // owner's type: both were set together when it was created.
            unsafe { release(self.owner) };
        }
    }
}

/// Frees a block made by `Buffer::allocate::<A>`.
///
/// # Safety
///
/// `owner` must be the start of an `AllocatedOwner<A>` with no references
/// left.
unsafe fn release_allocated<A: Allocator>(owner: NonNull<Owner>) {
    let header = owner.cast::<AllocatedOwner<A>>();
    // SAFETY: per the contract, the header is valid. Reading moves the
    // allocator out, so it is dropped after the block is freed.
    let AllocatedOwner {
        owner,
        allocator,
        layout,
    } = unsafe { header.read() };
    assert!(owner.references.get() == 0);
    assert!(layout.size() >= size_of::<AllocatedOwner<A>>());
    // SAFETY: the block came from this allocator with this layout.
    unsafe { allocator.deallocate(header.cast(), layout) };
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use super::*;
    use crate::allocator::Heap;

    /// The heap, counting live blocks so tests can see when one is freed.
    #[derive(Clone)]
    struct Counting {
        live: Rc<Cell<u32>>,
    }

    // SAFETY: forwards to `Heap`.
    unsafe impl Allocator for Counting {
        fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
            self.live.set(self.live.get() + 1);
            Heap.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            self.live.set(self.live.get() - 1);
            // SAFETY: forwarded from our caller.
            unsafe { Heap.deallocate(ptr, layout) }
        }
    }

    /// Refuses every allocation.
    struct Refusing;

    // SAFETY: never hands out memory.
    unsafe impl Allocator for Refusing {
        fn allocate(&self, _: Layout) -> Result<NonNull<u8>, AllocError> {
            Err(AllocError)
        }

        unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {
            unreachable!()
        }
    }

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
