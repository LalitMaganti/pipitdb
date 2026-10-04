//! Every allocation goes through an `Allocator` the caller passes in.

use core::alloc::Layout;
use core::ptr::NonNull;

use crate::buffer::Buffer;

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

/// An allocator of any type, as one pointer. Made once, then cloned: every
/// clone, and everything allocated through one, refers to the same header,
/// which holds the allocator.
#[derive(Clone)]
pub struct DynAllocator {
    // No bytes, only a header holding the allocator.
    header: Buffer,
}

impl DynAllocator {
    pub fn new<A: Allocator + Clone + 'static>(allocator: A) -> Result<DynAllocator, AllocError> {
        Ok(DynAllocator { header: Buffer::allocate(allocator, 0)? })
    }
}

// SAFETY: forwards to the allocator in the header.
unsafe impl Allocator for DynAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.header.allocate_raw_like(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded from the caller.
        unsafe { self.header.deallocate_raw_like(ptr, layout) }
    }
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

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: guaranteed by the caller.
        unsafe { alloc::alloc::dealloc(ptr.as_ptr(), layout) }
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;
    use core::cell::Cell;

    use super::*;
    use crate::boxed::Box;
    use crate::vec::Vec;

    /// Counts what's live.
    #[derive(Clone)]
    struct Counting(Rc<Cell<u32>>);

    // SAFETY: forwards to `Heap`.
    unsafe impl Allocator for Counting {
        fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
            self.0.set(self.0.get() + 1);
            Heap.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            self.0.set(self.0.get() - 1);
            // SAFETY: forwarded from the caller.
            unsafe { Heap.deallocate(ptr, layout) }
        }
    }

    #[test]
    fn allocates_through_the_allocator_it_holds() {
        let live = Rc::new(Cell::new(0));
        let allocator = DynAllocator::new(Counting(live.clone())).unwrap();
        assert_eq!(live.get(), 1);

        let boxed = Box::new(allocator.clone(), 7_u64).unwrap();
        let mut values = Vec::new(allocator.clone(), 64).unwrap();
        for i in 0..10_u64 {
            assert!(values.push(i).is_ok());
        }
        assert!(live.get() > 2);

        // What it made keeps the allocator alive.
        drop(allocator);
        assert_eq!((*boxed, values.len()), (7, 10));
        drop((boxed, values));
        assert_eq!(live.get(), 0);
    }
}
