//! `Buffer`: a refcounted block of bytes, aligned to 64 bytes or more.

use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::NonNull;

use crate::allocator::AllocError;
use crate::env::{Env, Kind};

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
unsafe impl Primitive for u64 {}
// SAFETY: as above.
unsafe impl Primitive for i64 {}
// SAFETY: as above.
unsafe impl Primitive for f64 {}

/// Cloning shares the bytes; the last drop frees them, through the `Env`
/// they came from, which outlives them.
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
    alignment: u32,
    size_bytes: usize,
    /// What the bytes came from.
    env: NonNull<Env>,
    /// Which of its allocators that is.
    kind: Kind,
}

pub(crate) const HEADER_BYTES: usize = size_of::<Header>();

impl Buffer {
    /// Allocates `size_bytes` zeroed bytes from `env`'s base allocator.
    #[inline(never)]
    pub fn allocate(env: &Env, size_bytes: usize) -> Result<Buffer, AllocError> {
        Buffer::allocate_in(env, Kind::Base, size_bytes)
    }

    /// As `allocate`, from `env`'s allocator of `kind`.
    pub fn allocate_in(env: &Env, kind: Kind, size_bytes: usize) -> Result<Buffer, AllocError> {
        // SAFETY: the bytes come zeroed.
        unsafe { Buffer::allocate_with((env, kind), size_bytes, true, BUFFER_ALIGNMENT_BYTES) }
    }

    /// Allocates `size_bytes` bytes without zeroing them.
    ///
    /// # Safety
    ///
    /// No byte may be read, through `as_slice` or otherwise, before it is
    /// written.
    #[inline(never)]
    pub unsafe fn allocate_uninit(env: &Env, size_bytes: usize) -> Result<Buffer, AllocError> {
        // SAFETY: upheld by the caller.
        unsafe { Buffer::allocate_uninit_in(env, Kind::Base, size_bytes) }
    }

    /// As `allocate_uninit`, from `env`'s allocator of `kind`.
    ///
    /// # Safety
    ///
    /// As for `allocate_uninit`.
    pub unsafe fn allocate_uninit_in(
        env: &Env,
        kind: Kind,
        size_bytes: usize,
    ) -> Result<Buffer, AllocError> {
        // SAFETY: upheld by the caller.
        unsafe { Buffer::allocate_with((env, kind), size_bytes, false, BUFFER_ALIGNMENT_BYTES) }
    }

    /// As `allocate_uninit`, with the header and bytes together starting at
    /// a multiple of `alignment`, a power of two of at least
    /// `BUFFER_ALIGNMENT_BYTES`: so `alignment - HEADER_BYTES` bytes fill an
    /// aligned block of `alignment`, such as a huge page.
    ///
    /// # Safety
    ///
    /// As for `allocate_uninit`.
    pub unsafe fn allocate_uninit_aligned(
        env: &Env,
        size_bytes: usize,
        alignment: usize,
    ) -> Result<Buffer, AllocError> {
        // SAFETY: upheld by the caller.
        unsafe { Buffer::allocate_with((env, Kind::Base), size_bytes, false, alignment) }
    }

    /// Allocates `size_bytes` bytes, zeroed if `zeroed`, aligned as
    /// `allocate_uninit_aligned`'s.
    ///
    /// # Safety
    ///
    /// As for `allocate_uninit` unless `zeroed`.
    #[inline(never)]
    unsafe fn allocate_with(
        (env, kind): (&Env, Kind),
        size_bytes: usize,
        zeroed: bool,
        alignment: usize,
    ) -> Result<Buffer, AllocError> {
        const { assert!(align_of::<Header>() == BUFFER_ALIGNMENT_BYTES) };
        check!(alignment >= BUFFER_ALIGNMENT_BYTES);
        let Ok(alignment) = u32::try_from(alignment) else { failed!() };
        let layout = layout(size_bytes, alignment)?;
        let allocator = env.allocator(kind);
        let header =
            if zeroed { allocator.allocate_zeroed(layout) } else { allocator.allocate(layout) }?
                .cast::<Header>();
        // SAFETY: the allocation fits a header followed by `size_bytes` bytes.
        let data = unsafe {
            let env = NonNull::from(env);
            header.write(Header { references: Cell::new(1), alignment, size_bytes, env, kind });
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
        // SAFETY: upheld by the caller.
        unsafe { Buffer::allocate_uninit_in(self.env(), self.header().kind, size_bytes) }
    }

    /// What the bytes came from.
    pub fn env(&self) -> &Env {
        // SAFETY: an `Env` outlives what's allocated from it.
        unsafe { self.header().env.as_ref() }
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
    /// `SmallVec`'s or a `Box`'s.
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

    /// Whether this is the only reference to the bytes.
    pub fn is_unique(&self) -> bool {
        self.header().references.get() == 1
    }

    /// Where the bytes are.
    pub(crate) fn as_non_null(&self) -> NonNull<u8> {
        self.data
    }

    /// Lets go of the buffer, and if it was the only reference, tells its
    /// allocator its bytes won't be read again, so their pages can go back
    /// to the system now rather than when they're next written.
    pub(crate) fn discard(self) {
        if self.is_unique() {
            let header = self.header();
            let allocator = self.env().allocator(header.kind);
            // SAFETY: the bytes are in the allocation, which isn't freed yet.
            unsafe { allocator.discard(self.data, header.size_bytes) };
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
    let Ok(layout) = layout(header.size_bytes, header.alignment) else { failed!() };
    // SAFETY: an `Env` outlives what's allocated from it.
    let allocator = unsafe { header.env.as_ref() }.allocator(header.kind);
    // SAFETY: the header and bytes were allocated together with this layout,
    // from this allocator.
    unsafe { allocator.deallocate(data.sub(HEADER_BYTES), layout) };
}

/// A header and `size_bytes` bytes after it, starting at a multiple of
/// `alignment`.
fn layout(size_bytes: usize, alignment: u32) -> Result<Layout, AllocError> {
    let total_bytes = HEADER_BYTES.checked_add(size_bytes).ok_or(AllocError)?;
    Layout::from_size_align(total_bytes, alignment as usize).map_err(|_| AllocError)
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
        if references == 0 {
            // SAFETY: that was the last reference.
            unsafe { deallocate(self.data) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;

    #[test]
    fn is_one_pointer() {
        // Columns hold a few, and are moved for every batch.
        assert_eq!(size_of::<Buffer>(), size_of::<usize>());
        assert_eq!(size_of::<Option<Buffer>>(), size_of::<usize>());
    }

    #[test]
    fn allocate_is_zeroed_and_aligned() {
        let env = Env::current();
        let buffer = Buffer::allocate(env, 100).unwrap();
        assert_eq!(buffer.as_slice::<u8>(), [0; 100]);
        assert!(buffer.as_slice::<u8>().as_ptr().addr().is_multiple_of(BUFFER_ALIGNMENT_BYTES));
    }

    /// The header and bytes start at the alignment, and are freed with it.
    #[test]
    fn allocates_aligned() {
        let env = Env::current();
        let empty = env.used();
        // SAFETY: the bytes aren't read.
        let buffer = unsafe { Buffer::allocate_uninit_aligned(env, 4096 - HEADER_BYTES, 4096) };
        let buffer = buffer.unwrap();
        let start = buffer.as_ptr::<u8>().addr() - HEADER_BYTES;
        assert!(start.is_multiple_of(4096));
        drop(buffer);
        assert_eq!(env.used(), empty);
    }

    #[test]
    fn clone_shares_bytes() {
        let env = Env::current();
        let mut buffer = Buffer::allocate(env, 8).unwrap();
        buffer.as_mut_slice::<i64>()[0] = -7;
        let clone = buffer.clone();
        assert_eq!(clone.as_slice::<i64>(), [-7]);
        assert_eq!(clone.as_slice::<u8>().as_ptr(), buffer.as_slice::<u8>().as_ptr());
    }

    #[test]
    fn last_drop_frees() {
        let env = Env::current();
        let empty = env.used();
        let buffer = Buffer::allocate(env, 8).unwrap();
        let clone = buffer.clone();
        drop(buffer);
        assert!(env.used() > empty);
        drop(clone);
        assert_eq!(env.used(), empty);
    }

    #[test]
    fn allocates_like_another_buffer() {
        let env = Env::current();
        let buffer = Buffer::allocate_in(env, Kind::Small, 8).unwrap();
        // SAFETY: the bytes aren't read.
        let other = unsafe { buffer.allocate_uninit_like(16) }.unwrap();
        assert_eq!(other.size_bytes(), 16);
        assert_eq!(other.header().kind, Kind::Small);
        drop((buffer, other));
    }

    #[test]
    fn allocation_past_the_budget_fails() {
        // SAFETY: declared before what's allocated from it, so dropped after.
        let env = unsafe { Env::new(Heap, 1000) }.unwrap();
        assert_eq!(Buffer::allocate(&env, 1000).err(), Some(AllocError));
    }

    #[test]
    fn oversized_allocation_fails() {
        let env = Env::current();
        assert_eq!(Buffer::allocate(env, usize::MAX).err(), Some(AllocError));
    }

    #[test]
    #[should_panic(expected = "references")]
    fn writing_shared_bytes_panics() {
        let env = Env::current();
        let mut buffer = Buffer::allocate(env, 8).unwrap();
        let _clone = buffer.clone();
        buffer.as_mut_slice::<u8>();
    }
}
