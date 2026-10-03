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

/// Cloning shares the bytes; the last drop frees them.
pub struct Buffer {
    data: NonNull<u8>,
    size_bytes: usize,
    owner: NonNull<Owner>,
}

/// The start of every header, so a `Buffer` can free one without knowing
/// its allocator's type.
#[repr(C)]
struct Owner {
    references: Cell<u32>,
    free: unsafe fn(NonNull<Owner>),
}

/// Sits in front of the bytes, in the same allocation.
#[repr(C, align(64))]
struct Header<A> {
    owner: Owner,
    allocator: A,
    layout: Layout,
}

impl Buffer {
    /// Allocates `size_bytes` zeroed bytes.
    pub fn allocate<A: Allocator + 'static>(
        allocator: A,
        size_bytes: usize,
    ) -> Result<Buffer, AllocError> {
        // SAFETY: the bytes are zeroed before anything can read them.
        let buffer = unsafe { Buffer::allocate_uninit(allocator, size_bytes)? };
        // SAFETY: the buffer holds `size_bytes` bytes.
        unsafe { buffer.data.write_bytes(0, buffer.size_bytes) };
        Ok(buffer)
    }

    /// Allocates `size_bytes` bytes without zeroing them.
    ///
    /// # Safety
    ///
    /// No byte may be read, through `as_slice` or otherwise, before it is
    /// written.
    pub unsafe fn allocate_uninit<A: Allocator + 'static>(
        allocator: A,
        size_bytes: usize,
    ) -> Result<Buffer, AllocError> {
        const { assert!(align_of::<Header<A>>() == BUFFER_ALIGNMENT_BYTES) };
        let header_bytes = size_of::<Header<A>>();
        let total_bytes = header_bytes.checked_add(size_bytes).ok_or(AllocError)?;
        let layout =
            Layout::from_size_align(total_bytes, BUFFER_ALIGNMENT_BYTES).map_err(|_| AllocError)?;
        let header = allocator.allocate(layout)?.cast::<Header<A>>();
        let owner = Owner { references: Cell::new(1), free: free::<A> };

        // SAFETY: `layout` fits a header followed by `size_bytes` bytes.
        let data = unsafe {
            header.write(Header { owner, allocator, layout });
            header.cast::<u8>().add(header_bytes)
        };
        check!(data.addr().get().is_multiple_of(BUFFER_ALIGNMENT_BYTES));
        Ok(Buffer { data, size_bytes, owner: header.cast() })
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    pub fn as_slice<T: Primitive>(&self) -> &[T] {
        check!(self.size_bytes.is_multiple_of(size_of::<T>()));
        // SAFETY: the bytes are aligned for any `Primitive`, any bit pattern
        // is a valid `T`, and the bytes live as long as any reference to them.
        unsafe { core::slice::from_raw_parts(self.data.as_ptr().cast(), self.len::<T>()) }
    }

    /// Buffers are written before they are shared.
    pub fn as_mut_slice<T: Primitive>(&mut self) -> &mut [T] {
        check!(self.size_bytes.is_multiple_of(size_of::<T>()));
        check!(self.owner().references.get() == 1);
        // SAFETY: as in `as_slice`, and this is the only reference.
        unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr().cast(), self.len::<T>()) }
    }

    /// The first byte, for buffers whose bytes are tracked as written by the
    /// caller, which `as_slice` can't read.
    pub(crate) fn as_ptr<T: Primitive>(&self) -> *const T {
        check!(self.size_bytes.is_multiple_of(size_of::<T>()));
        self.data.as_ptr().cast()
    }

    /// As `as_ptr`, for writing.
    pub(crate) fn as_mut_ptr<T: Primitive>(&mut self) -> *mut T {
        check!(self.size_bytes.is_multiple_of(size_of::<T>()));
        check!(self.owner().references.get() == 1);
        self.data.as_ptr().cast()
    }

    fn len<T: Primitive>(&self) -> usize {
        const { assert!(align_of::<T>() <= BUFFER_ALIGNMENT_BYTES) };
        self.size_bytes / size_of::<T>()
    }

    fn owner(&self) -> &Owner {
        // SAFETY: the owner lives as long as any reference to it.
        unsafe { self.owner.as_ref() }
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Buffer {
        let references = &self.owner().references;
        check!(references.get() < u32::MAX);
        references.set(references.get() + 1);
        Buffer { data: self.data, size_bytes: self.size_bytes, owner: self.owner }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let owner = self.owner();
        let references = owner.references.get() - 1;
        owner.references.set(references);
        if references == 0 {
            let free = owner.free;
            // SAFETY: that was the last reference.
            unsafe { free(self.owner) };
        }
    }
}

unsafe fn free<A: Allocator>(owner: NonNull<Owner>) {
    let header = owner.cast::<Header<A>>();
    // SAFETY: `owner` starts a `Header<A>`. Reading it moves the allocator
    // out before its memory is freed.
    unsafe {
        let Header { allocator, layout, .. } = header.read();
        allocator.deallocate(header.cast(), layout);
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use super::*;
    use crate::allocator::Heap;

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

    struct Refusing;

    // SAFETY: never hands out memory.
    unsafe impl Allocator for Refusing {
        fn allocate(&self, _: Layout) -> Result<NonNull<u8>, AllocError> {
            Err(AllocError)
        }

        unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {}
    }

    #[test]
    fn allocate_is_zeroed_and_aligned() {
        let buffer = Buffer::allocate(Heap, 100).unwrap();
        assert_eq!(buffer.as_slice::<u8>(), [0; 100]);
        assert!(buffer.as_slice::<u8>().as_ptr().addr().is_multiple_of(BUFFER_ALIGNMENT_BYTES));
    }

    #[test]
    fn clone_shares_bytes() {
        let mut buffer = Buffer::allocate(Heap, 8).unwrap();
        buffer.as_mut_slice::<i64>()[0] = -7;
        let clone = buffer.clone();
        assert_eq!(clone.as_slice::<i64>(), [-7]);
        assert_eq!(clone.as_slice::<u8>().as_ptr(), buffer.as_slice::<u8>().as_ptr());
    }

    #[test]
    fn last_drop_frees() {
        let live = Rc::new(Cell::new(0));
        let buffer = Buffer::allocate(Counting(live.clone()), 8).unwrap();
        let clone = buffer.clone();
        drop(buffer);
        assert_eq!(live.get(), 1);
        drop(clone);
        assert_eq!(live.get(), 0);
    }

    #[test]
    fn refused_allocation_fails() {
        assert_eq!(Buffer::allocate(Refusing, 8).err(), Some(AllocError));
    }

    #[test]
    fn oversized_allocation_fails() {
        assert_eq!(Buffer::allocate(Heap, usize::MAX).err(), Some(AllocError));
    }

    #[test]
    #[should_panic(expected = "references")]
    fn writing_shared_bytes_panics() {
        let mut buffer = Buffer::allocate(Heap, 8).unwrap();
        let _clone = buffer.clone();
        buffer.as_mut_slice::<u8>();
    }
}
