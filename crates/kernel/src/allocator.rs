//! Every allocation goes through an `Allocator` the caller passes in, by
//! reference. What's allocated refers back to it, to free itself, so an
//! allocator must outlive everything allocated from it.

use core::alloc::Layout;
use core::cell::Cell;
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

/// Allocates from another allocator while the bytes allocated stay within
/// `limit`, and fails past it, so a run can't use more memory than it's
/// given. `peak` is the most it allocated at once. Like any allocator, it
/// must outlive what's allocated from it: dropping it while anything is
/// still allocated fails a check.
pub struct Budget<'a> {
    allocator: &'a dyn Allocator,
    limit: usize,
    used: Cell<usize>,
    peak: Cell<usize>,
}

impl<'a> Budget<'a> {
    pub fn new(allocator: &'a dyn Allocator, limit: usize) -> Budget<'a> {
        Budget { allocator, limit, used: Cell::new(0), peak: Cell::new(0) }
    }

    /// The bytes allocated now.
    pub fn used(&self) -> usize {
        self.used.get()
    }

    /// The most bytes allocated at once.
    pub fn peak(&self) -> usize {
        self.peak.get()
    }
}

impl Budget<'_> {
    /// Allocates `layout`, zeroed if `zeroed`, if it fits within the limit,
    /// and counts it.
    fn charge(&self, layout: Layout, zeroed: bool) -> Result<NonNull<u8>, AllocError> {
        let used = self.used.get().checked_add(layout.size()).ok_or(AllocError)?;
        if used > self.limit {
            return Err(AllocError);
        }
        let ptr = if zeroed {
            self.allocator.allocate_zeroed(layout)
        } else {
            self.allocator.allocate(layout)
        }?;
        self.used.set(used);
        self.peak.set(self.peak.get().max(used));
        Ok(ptr)
    }
}

// SAFETY: forwards to the allocator it holds.
unsafe impl Allocator for Budget<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.charge(layout, false)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.charge(layout, true)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded from the caller.
        unsafe { self.allocator.deallocate(ptr, layout) };
        self.used.set(self.used.get() - layout.size());
    }
}

impl Drop for Budget<'_> {
    fn drop(&mut self) {
        check!(self.used.get() == 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boxed::Box;

    #[test]
    fn budgets_fail_past_their_limit_and_count_the_peak() {
        let budget = Budget::new(&Heap, 1000);
        let first = Box::new(&budget, [0_u8; 600]).unwrap();
        // A box's header counts too.
        let one = budget.used();
        assert!(one > 600);
        assert!(Box::new(&budget, [0_u8; 600]).is_err());
        assert_eq!(budget.used(), one);

        // What's freed can be allocated again.
        drop(first);
        assert_eq!(budget.used(), 0);
        let second = Box::new(&budget, [0_u8; 800]).unwrap();
        let two = budget.used();
        drop(second);
        assert_eq!((budget.used(), budget.peak()), (0, two));
    }

    #[test]
    fn zeroed_memory_is_zero_and_counted() {
        let budget = Budget::new(&Heap, 1000);
        let layout = Layout::from_size_align(256, 64).unwrap();
        let ptr = budget.allocate_zeroed(layout).unwrap();
        // SAFETY: the allocation holds 256 bytes.
        assert!(unsafe { core::slice::from_raw_parts(ptr.as_ptr(), 256) }.iter().all(|&b| b == 0));
        assert_eq!(budget.used(), 256);
        assert!(budget.allocate_zeroed(Layout::from_size_align(800, 64).unwrap()).is_err());
        // SAFETY: allocated just above with this layout.
        unsafe { budget.deallocate(ptr, layout) };
        assert_eq!(budget.used(), 0);
    }

    /// Hands out one static block, so what's allocated from it can be
    /// leaked without leaking heap memory.
    struct Static;

    #[repr(align(64))]
    struct Block {
        _bytes: [u8; 128],
    }

    static mut BLOCK: Block = Block { _bytes: [0; 128] };

    // SAFETY: the block is valid for the life of the program; each test
    // allocates from it once.
    unsafe impl Allocator for Static {
        fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
            check!(layout.size() <= 128 && layout.align() <= 64);
            NonNull::new((&raw mut BLOCK).cast::<u8>()).ok_or(AllocError)
        }

        unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {}
    }

    #[test]
    #[should_panic(expected = "used")]
    fn budgets_check_nothing_outlives_them() {
        let budget = Budget::new(&Static, 1000);
        // Leaked, so it's never freed through the dropped budget.
        core::mem::forget(Box::new(&budget, 7_u64).unwrap());
    }
}
