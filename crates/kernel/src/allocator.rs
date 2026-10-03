//! Explicit allocators.
//!
//! Every allocation goes through an allocator handle the caller passes in;
//! nothing allocates behind the caller's back. This is what lets budgets be
//! enforced and lets an allocation be refused without aborting.

use core::alloc::Layout;
use core::ptr::NonNull;

/// An allocation was refused: out of memory, over budget, or too large.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AllocError;

/// A source of memory.
///
/// # Safety
///
/// `allocate` must return memory that is valid for reads and writes of
/// `layout` and aligned to `layout.align()`, until it is passed back to
/// `deallocate` on the same allocator.
pub unsafe trait Allocator {
    /// Allocates a block for `layout`, which must have a non-zero size.
    ///
    /// # Errors
    ///
    /// Returns `AllocError` when the allocation is refused.
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError>;

    /// Frees a block.
    ///
    /// # Safety
    ///
    /// `ptr` must have come from `allocate` on this allocator with the same
    /// `layout`, and must not have been freed already.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
}

/// The system heap, through Rust's global allocator.
#[derive(Clone, Copy, Default)]
pub struct Heap;

// SAFETY: the global allocator upholds the contract for non-zero sizes,
// which `allocate` asserts.
unsafe impl Allocator for Heap {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        // The global allocator's behavior is undefined for zero sizes.
        assert!(layout.size() > 0);
        // SAFETY: the size is non-zero, asserted above.
        let ptr = unsafe { alloc::alloc::alloc(layout) };
        NonNull::new(ptr).ok_or(AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        assert!(layout.size() > 0);
        // SAFETY: the caller guarantees `ptr` came from `allocate` with
        // `layout`.
        unsafe { alloc::alloc::dealloc(ptr.as_ptr(), layout) }
    }
}

/// Allocators for tests.
#[cfg(test)]
pub(crate) mod testing {
    use alloc::rc::Rc;
    use core::alloc::Layout;
    use core::cell::Cell;
    use core::ptr::NonNull;

    use super::{AllocError, Allocator, Heap};

    /// The heap, counting live blocks so tests can see when one is freed.
    #[derive(Clone)]
    pub(crate) struct Counting {
        pub(crate) live: Rc<Cell<u32>>,
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
    pub(crate) struct Refusing;

    // SAFETY: never hands out memory.
    unsafe impl Allocator for Refusing {
        fn allocate(&self, _: Layout) -> Result<NonNull<u8>, AllocError> {
            Err(AllocError)
        }

        unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {
            unreachable!()
        }
    }
}
