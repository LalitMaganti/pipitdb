//! Every allocation goes through an `Allocator` the caller passes in.

use core::alloc::Layout;
use core::ptr::NonNull;

/// An allocation was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AllocError;

/// # Safety
///
/// `allocate` must return memory valid for `layout` until it is passed to
/// `deallocate`.
pub unsafe trait Allocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError>;

    /// # Safety
    ///
    /// `ptr` and `layout` must come from a matching call to `allocate`.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
}

/// The global heap.
#[derive(Clone, Copy)]
pub struct Heap;

// SAFETY: forwards to the global allocator.
unsafe impl Allocator for Heap {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        assert!(layout.size() > 0);
        // SAFETY: the size is non-zero.
        NonNull::new(unsafe { alloc::alloc::alloc(layout) }).ok_or(AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: guaranteed by the caller.
        unsafe { alloc::alloc::dealloc(ptr.as_ptr(), layout) }
    }
}
