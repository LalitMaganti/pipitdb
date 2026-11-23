//! `Allocator`, what memory comes from, and `Heap`, the system's. A query's
//! `Global` owns the allocator it takes memory from, and what's allocated
//! keeps the `Global` alive, so can't outlive it.

use core::alloc::Layout;
use core::ptr::NonNull;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AllocError;

/// # Safety
///
/// `allocate` must return memory valid for `layout` until it is passed to
/// `deallocate`, and `allocate_zeroed` such memory with every byte zero.
pub unsafe trait Allocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError>;

    /// As `allocate`, with the bytes zeroed. An allocator whose fresh memory
    /// is zero already, such as pages from the system, can skip writing.
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        let ptr = self.allocate(layout)?;
        // SAFETY: the allocation holds `layout.size()` bytes.
        unsafe { ptr.write_bytes(0, layout.size()) };
        Ok(ptr)
    }

    /// # Safety
    ///
    /// `ptr` and `layout` must come from a matching call to `allocate` or
    /// `allocate_zeroed`.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// Says the `len` bytes at `ptr` won't be read before they're written
    /// again, so the whole pages among them can go back to the system.
    ///
    /// # Safety
    ///
    /// They must be within an allocation from this allocator, not yet
    /// deallocated.
    unsafe fn discard(&self, _ptr: NonNull<u8>, _len: usize) {}
}

#[derive(Clone, Copy)]
pub struct Heap;

// SAFETY: forwards to the global allocator.
unsafe impl Allocator for Heap {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        check!(layout.size() > 0);
        // SAFETY: the size is non-zero.
        NonNull::new(unsafe { alloc::alloc::alloc(layout) }).ok_or(AllocError)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        check!(layout.size() > 0);
        // SAFETY: the size is non-zero.
        NonNull::new(unsafe { alloc::alloc::alloc_zeroed(layout) }).ok_or(AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: guaranteed by the caller.
        unsafe { alloc::alloc::dealloc(ptr.as_ptr(), layout) }
    }
}
