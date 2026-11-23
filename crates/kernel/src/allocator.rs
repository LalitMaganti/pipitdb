//! `Allocator`, what memory comes from, and `Heap`, the system's. A query's
//! `Env` owns the allocator it takes memory from, and what's allocated
//! keeps the `Env` alive, so can't outlive it.

use core::alloc::Layout;
use core::ptr::NonNull;

use crate::buffer::BUFFER_ALIGNMENT_BYTES;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AllocError;

/// 16 KB: the largest memory page among the systems we run on (Apple's
/// are 16 KB, most others' 4 KB), so memory aligned to it starts on a page
/// everywhere.
pub(crate) const PAGE_BYTES: usize = 1 << 14;

/// 256 KB: memory this big or bigger is aligned to a page.
const LARGE_BYTES: usize = 1 << 18;

/// 2 MB: Linux's huge page on x86 and on most ARM systems.
const HUGE_PAGE_BYTES: usize = 1 << 21;

/// The layout of `bytes` of memory that one owner holds, such as an
/// `Array`'s or a `SlowVec`'s.
///
/// From `LARGE_BYTES` up, it's aligned to a page, as a hint to the
/// allocator: a big block that one owner keeps for a long time, such as a
/// hash table, is best taken from the system as whole pages and given back
/// when freed, rather than kept in a heap. On Linux, from a huge page up,
/// it's aligned to one, so the system backs it with huge pages and reading
/// it at random doesn't also miss the TLB. Elsewhere that would only waste
/// memory, as WebAssembly's, which is all one heap.
pub(crate) fn layout(bytes: usize) -> Result<Layout, AllocError> {
    let huge = cfg!(any(target_os = "linux", target_os = "android")) && bytes >= HUGE_PAGE_BYTES;
    let align = match bytes {
        _ if huge => HUGE_PAGE_BYTES,
        LARGE_BYTES.. => PAGE_BYTES,
        _ => BUFFER_ALIGNMENT_BYTES,
    };
    Layout::from_size_align(bytes, align).map_err(|_| AllocError)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligns_large_memory_to_pages() {
        let align = |bytes| layout(bytes).unwrap().align();
        assert_eq!(align(LARGE_BYTES - 1), BUFFER_ALIGNMENT_BYTES);
        assert_eq!(align(LARGE_BYTES), PAGE_BYTES);
        let huge = if cfg!(target_os = "linux") { HUGE_PAGE_BYTES } else { PAGE_BYTES };
        assert_eq!(align(HUGE_PAGE_BYTES), huge);
    }
}
