//! `Buffer`: a refcounted, 64-byte aligned block of bytes.

use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};

pub const BUFFER_ALIGNMENT_BYTES: usize = 64;

/// A number type where any bit pattern is a valid value.
///
/// # Safety
///
/// Only implement this for such types.
pub unsafe trait Primitive: Copy {}

// SAFETY: any bit pattern is a valid value of each of these.
unsafe impl Primitive for u8 {}
// SAFETY: as above.
unsafe impl Primitive for u32 {}
// SAFETY: as above.
unsafe impl Primitive for i64 {}
// SAFETY: as above.
unsafe impl Primitive for f64 {}

/// Cloning shares the bytes; the last drop frees them, through the allocator
/// they came from, which must outlive them.
///
/// It's one pointer, to the bytes: what else it needs is in a header right
/// in front of them, so columns, which hold a few, stay small.
pub struct Buffer {
    data: NonNull<u8>,
}

/// Sits in front of the bytes, in the same allocation.
#[repr(C, align(64))]
struct Header {
    references: Cell<u32>,
    /// Whether the last drop keeps the bytes for a `ColumnPool`.
    pooled: Cell<bool>,
    size_bytes: usize,
    allocator: NonNull<dyn Allocator>,
}

const HEADER_BYTES: usize = size_of::<Header>();

impl Buffer {
    /// Allocates `size_bytes` zeroed bytes from `allocator`, which must
    /// outlive the buffer and its clones.
    pub fn allocate(allocator: &dyn Allocator, size_bytes: usize) -> Result<Buffer, AllocError> {
        // SAFETY: the bytes come zeroed.
        unsafe { Buffer::allocate_with(allocator, size_bytes, true) }
    }

    /// Allocates `size_bytes` bytes without zeroing them.
    ///
    /// # Safety
    ///
    /// No byte may be read, through `as_slice` or otherwise, before it is
    /// written.
    pub unsafe fn allocate_uninit(
        allocator: &dyn Allocator,
        size_bytes: usize,
    ) -> Result<Buffer, AllocError> {
        // SAFETY: upheld by the caller.
        unsafe { Buffer::allocate_with(allocator, size_bytes, false) }
    }

    /// Allocates `size_bytes` bytes, zeroed if `zeroed`.
    ///
    /// # Safety
    ///
    /// As for `allocate_uninit` unless `zeroed`.
    unsafe fn allocate_with(
        allocator: &dyn Allocator,
        size_bytes: usize,
        zeroed: bool,
    ) -> Result<Buffer, AllocError> {
        const { assert!(align_of::<Header>() == BUFFER_ALIGNMENT_BYTES) };
        let layout = layout(size_bytes)?;
        let header =
            if zeroed { allocator.allocate_zeroed(layout) } else { allocator.allocate(layout) }?
                .cast::<Header>();
        // The allocator outlives the buffer, as its callers promise, so its
        // lifetime can be forgotten.
        // SAFETY: only the lifetime changes.
        let allocator = unsafe {
            core::mem::transmute::<NonNull<dyn Allocator + '_>, NonNull<dyn Allocator>>(
                NonNull::from(allocator),
            )
        };
        // SAFETY: the allocation fits a header followed by `size_bytes` bytes.
        let data = unsafe {
            header.write(Header {
                references: Cell::new(1),
                pooled: Cell::new(false),
                size_bytes,
                allocator,
            });
            header.cast::<u8>().add(HEADER_BYTES)
        };
        Ok(Buffer { data })
    }

    /// Allocates `size_bytes` bytes, without zeroing them, from the allocator
    /// `self` came from.
    ///
    /// # Safety
    ///
    /// As for `allocate_uninit`.
    pub unsafe fn allocate_uninit_like(&self, size_bytes: usize) -> Result<Buffer, AllocError> {
        // SAFETY: the allocator outlives `self`; the caller upholds the rest.
        unsafe { Buffer::allocate_uninit(self.allocator(), size_bytes) }
    }

    /// The allocator the bytes came from.
    pub fn allocator(&self) -> &dyn Allocator {
        // SAFETY: the allocator outlives every buffer from it.
        unsafe { self.header().allocator.as_ref() }
    }

    pub fn size_bytes(&self) -> usize {
        self.header().size_bytes
    }

    pub fn as_slice<T: Primitive>(&self) -> &[T] {
        check!(self.size_bytes().is_multiple_of(size_of::<T>()));
        // SAFETY: the bytes are aligned for any `Primitive`, any bit pattern
        // is a valid `T`, and the bytes live as long as any reference to them.
        unsafe { core::slice::from_raw_parts(self.data.as_ptr().cast(), self.len::<T>()) }
    }

    /// Buffers are written before they are shared.
    pub fn as_mut_slice<T: Primitive>(&mut self) -> &mut [T] {
        check!(self.size_bytes().is_multiple_of(size_of::<T>()));
        check!(self.header().references.get() == 1);
        // SAFETY: as in `as_slice`, and this is the only reference.
        unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr().cast(), self.len::<T>()) }
    }

    /// The first byte, for buffers whose bytes are tracked as written by the
    /// caller, which `as_slice` can't read.
    pub fn as_ptr<T: Primitive>(&self) -> *const T {
        check!(self.size_bytes().is_multiple_of(size_of::<T>()));
        self.data.as_ptr().cast()
    }

    /// The first byte, for writing values that aren't `Primitive`, such as a
    /// `SlowVec`'s or a `Box`'s.
    pub fn as_mut_non_null(&mut self) -> NonNull<u8> {
        check!(self.header().references.get() == 1);
        self.data
    }

    /// As `as_ptr`, for writing.
    pub fn as_mut_ptr<T: Primitive>(&mut self) -> *mut T {
        check!(self.size_bytes().is_multiple_of(size_of::<T>()));
        check!(self.header().references.get() == 1);
        self.data.as_ptr().cast()
    }

    fn len<T: Primitive>(&self) -> usize {
        const { assert!(align_of::<T>() <= BUFFER_ALIGNMENT_BYTES) };
        self.size_bytes() / size_of::<T>()
    }

    fn header(&self) -> &Header {
        // SAFETY: the buffer references its bytes.
        unsafe { header(self.data) }
    }

    /// Allocates `size_bytes` zeroed bytes, as `allocate`, whose last drop
    /// keeps them for the `ColumnPool` that holds `as_non_null` of them, to
    /// hand out again, until it gives them up.
    pub(crate) fn allocate_pooled(
        allocator: &dyn Allocator,
        size_bytes: usize,
    ) -> Result<Buffer, AllocError> {
        let buffer = Buffer::allocate(allocator, size_bytes)?;
        buffer.header().pooled.set(true);
        Ok(buffer)
    }

    /// Whether this is the only reference to the bytes.
    pub(crate) fn is_unique(&self) -> bool {
        self.header().references.get() == 1
    }

    /// Where the bytes are, for a `ColumnPool` to find them again.
    pub(crate) fn as_non_null(&self) -> NonNull<u8> {
        self.data
    }

    /// The pooled buffer at `data`, if nothing references it and it has
    /// `size_bytes` bytes.
    ///
    /// # Safety
    ///
    /// `data` must be from a pooled buffer's `as_non_null`, not yet given up.
    pub(crate) unsafe fn reuse(data: NonNull<u8>, size_bytes: usize) -> Option<Buffer> {
        // SAFETY: pooled bytes live until given up.
        let header = unsafe { header(data) };
        (header.references.get() == 0 && header.size_bytes == size_bytes).then(|| {
            header.references.set(1);
            Buffer { data }
        })
    }

    /// Gives up the pooled buffer at `data`: it's freed now if nothing
    /// references it, and by its last drop if something does.
    ///
    /// # Safety
    ///
    /// As for `reuse`; `data` mustn't be used again.
    pub(crate) unsafe fn unpool(data: NonNull<u8>) {
        // SAFETY: pooled bytes live until given up.
        let header = unsafe { header(data) };
        header.pooled.set(false);
        if header.references.get() == 0 {
            // SAFETY: nothing references the bytes, and they're no longer
            // pooled.
            unsafe { deallocate(data) };
        }
    }
}

/// The header of the bytes at `data`.
///
/// # Safety
///
/// `data` must be a buffer's bytes, which haven't been freed.
unsafe fn header<'a>(data: NonNull<u8>) -> &'a Header {
    // SAFETY: the header is right in front of the bytes.
    unsafe { data.sub(HEADER_BYTES).cast::<Header>().as_ref() }
}

/// Frees the bytes at `data`, and their header.
///
/// # Safety
///
/// As for `header`, and nothing may use them again.
unsafe fn deallocate(data: NonNull<u8>) {
    // SAFETY: upheld by the caller.
    let header = unsafe { header(data) };
    let Ok(layout) = layout(header.size_bytes) else { crate::check::check_failed(line!()) };
    // SAFETY: the header and bytes were allocated together with this layout,
    // from this allocator, which outlives them.
    unsafe { header.allocator.as_ref().deallocate(data.sub(HEADER_BYTES), layout) };
}

/// A header and `size_bytes` bytes after it.
fn layout(size_bytes: usize) -> Result<Layout, AllocError> {
    let total_bytes = HEADER_BYTES.checked_add(size_bytes).ok_or(AllocError)?;
    Layout::from_size_align(total_bytes, BUFFER_ALIGNMENT_BYTES).map_err(|_| AllocError)
}

impl Clone for Buffer {
    fn clone(&self) -> Buffer {
        let references = &self.header().references;
        check!(references.get() < u32::MAX);
        references.set(references.get() + 1);
        Buffer { data: self.data }
    }
}

impl Drop for Buffer {
    // Out of line: buffers are dropped in many places, each of which would
    // otherwise carry a copy.
    #[inline(never)]
    fn drop(&mut self) {
        let header = self.header();
        let references = header.references.get() - 1;
        header.references.set(references);
        if references == 0 && !header.pooled.get() {
            // SAFETY: that was the last reference.
            unsafe { deallocate(self.data) };
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use super::*;
    use crate::allocator::Heap;

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

    #[derive(Clone)]
    struct Refusing;

    // SAFETY: never hands out memory.
    unsafe impl Allocator for Refusing {
        fn allocate(&self, _: Layout) -> Result<NonNull<u8>, AllocError> {
            Err(AllocError)
        }

        unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {}
    }

    #[test]
    fn is_one_pointer() {
        // Columns hold a few, and are moved for every batch.
        assert_eq!(size_of::<Buffer>(), size_of::<usize>());
        assert_eq!(size_of::<Option<Buffer>>(), size_of::<usize>());
    }

    #[test]
    fn allocate_is_zeroed_and_aligned() {
        let buffer = Buffer::allocate(&Heap, 100).unwrap();
        assert_eq!(buffer.as_slice::<u8>(), [0; 100]);
        assert!(buffer.as_slice::<u8>().as_ptr().addr().is_multiple_of(BUFFER_ALIGNMENT_BYTES));
    }

    #[test]
    fn clone_shares_bytes() {
        let mut buffer = Buffer::allocate(&Heap, 8).unwrap();
        buffer.as_mut_slice::<i64>()[0] = -7;
        let clone = buffer.clone();
        assert_eq!(clone.as_slice::<i64>(), [-7]);
        assert_eq!(clone.as_slice::<u8>().as_ptr(), buffer.as_slice::<u8>().as_ptr());
    }

    #[test]
    fn last_drop_frees() {
        let live = Rc::new(Cell::new(0));
        let counting = Counting(live.clone());
        let buffer = Buffer::allocate(&counting, 8).unwrap();
        let clone = buffer.clone();
        drop(buffer);
        assert_eq!(live.get(), 1);
        drop(clone);
        assert_eq!(live.get(), 0);
    }

    #[test]
    fn allocates_like_another_buffer() {
        let live = Rc::new(Cell::new(0));
        let counting = Counting(live.clone());
        let buffer = Buffer::allocate(&counting, 8).unwrap();
        // SAFETY: the bytes aren't read.
        let other = unsafe { buffer.allocate_uninit_like(16) }.unwrap();
        assert_eq!(other.size_bytes(), 16);
        assert_eq!(live.get(), 2);
        drop(buffer);
        drop(other);
        assert_eq!(live.get(), 0);
    }

    #[test]
    fn refused_allocation_fails() {
        assert_eq!(Buffer::allocate(&Refusing, 8).err(), Some(AllocError));
    }

    #[test]
    fn oversized_allocation_fails() {
        assert_eq!(Buffer::allocate(&Heap, usize::MAX).err(), Some(AllocError));
    }

    #[test]
    #[should_panic(expected = "references")]
    fn writing_shared_bytes_panics() {
        let mut buffer = Buffer::allocate(&Heap, 8).unwrap();
        let _clone = buffer.clone();
        buffer.as_mut_slice::<u8>();
    }
}
