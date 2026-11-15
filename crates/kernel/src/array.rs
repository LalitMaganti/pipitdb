//! `Array`: a fixed number of values in one allocation of exactly their
//! size, for data that must be contiguous, such as what a sort splits into
//! slices. It never grows: data that grows goes in blocks that never move.

use core::alloc::Layout;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, Primitive};

/// 2 MB: Linux's huge page on x86 and on most ARM systems.
const HUGE_PAGE_BYTES: usize = 1 << 21;

/// `len` values, zeroed when made.
pub(crate) struct Array<T: Primitive> {
    values: NonNull<T>,
    len: usize,
    /// Where the values came from, which outlives them, as callers promise.
    allocator: NonNull<dyn Allocator>,
    owns: PhantomData<T>,
}

/// The layout of `bytes` of data that's read at random. On Linux, from a huge
/// page up, it's aligned to one, so the system backs it with huge pages and
/// reading it at random doesn't also miss the TLB. Elsewhere that alignment
/// would only waste memory, as WebAssembly's, which is all one heap.
pub(crate) fn layout(bytes: usize) -> Result<Layout, AllocError> {
    let huge = cfg!(any(target_os = "linux", target_os = "android")) && bytes >= HUGE_PAGE_BYTES;
    let align = if huge { HUGE_PAGE_BYTES } else { BUFFER_ALIGNMENT_BYTES };
    Layout::from_size_align(bytes, align).map_err(|_| AllocError)
}

impl<T: Primitive> Array<T> {
    /// `len` zeroes: fresh memory from the system is zero already, so a big
    /// one costs nothing to make.
    pub(crate) fn zeroed(allocator: &dyn Allocator, len: usize) -> Result<Array<T>, AllocError> {
        Array::allocate(allocator, len, true)
    }

    /// `len` values, value `i` being `value(i)`, each written once.
    pub(crate) fn from_fn(
        allocator: &dyn Allocator,
        len: usize,
        mut value: impl FnMut(usize) -> T,
    ) -> Result<Array<T>, AllocError> {
        let array = Array::<T>::allocate(allocator, len, false)?;
        for i in 0..len {
            // SAFETY: there's room for `len` values.
            unsafe { array.values.as_ptr().add(i).write(value(i)) };
        }
        Ok(array)
    }

    /// `len` values, written a chunk of up to `chunk` at a time by
    /// `write(start, out)`, which writes every one of `out`, values `start`
    /// on.
    pub(crate) fn from_chunks(
        allocator: &dyn Allocator,
        len: usize,
        chunk: usize,
        mut write: impl FnMut(usize, &mut [MaybeUninit<T>]),
    ) -> Result<Array<T>, AllocError> {
        check!(chunk > 0);
        let array = Array::<T>::allocate(allocator, len, false)?;
        for start in (0..len).step_by(chunk) {
            let count = chunk.min(len - start);
            // SAFETY: values `start..start + count` are within the array, and
            // only reached through this slice until written.
            let out = unsafe {
                core::slice::from_raw_parts_mut(array.values.as_ptr().add(start).cast(), count)
            };
            write(start, out);
        }
        Ok(array)
    }

    /// The values of `values`, in order.
    pub(crate) fn from_iter(
        allocator: &dyn Allocator,
        mut values: impl ExactSizeIterator<Item = T>,
    ) -> Result<Array<T>, AllocError> {
        let Some(first) = values.next() else { return Array::allocate(allocator, 0, false) };
        let len = values.len() + 1;
        Array::from_fn(
            allocator,
            len,
            |i| if i == 0 { first } else { values.next().unwrap_or(first) },
        )
    }

    /// Room for `len` values, zeroed if `zeroed`, else unwritten: callers
    /// that don't zero write each value before it's read.
    fn allocate(
        allocator: &dyn Allocator,
        len: usize,
        zeroed: bool,
    ) -> Result<Array<T>, AllocError> {
        let bytes = size_of::<T>().checked_mul(len).ok_or(AllocError)?;
        let values = match bytes {
            0 => NonNull::dangling(),
            _ if zeroed => allocator.allocate_zeroed(layout(bytes)?)?.cast(),
            _ => allocator.allocate(layout(bytes)?)?.cast(),
        };
        // SAFETY: only the lifetime changes; the allocator outlives the values.
        let allocator = unsafe {
            core::mem::transmute::<NonNull<dyn Allocator + '_>, NonNull<dyn Allocator>>(
                NonNull::from(allocator),
            )
        };
        Ok(Array { values, len, allocator, owns: PhantomData })
    }
}

impl<T: Primitive> Deref for Array<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: `len` values, all written: zeroed, or by `from_fn` or
        // `from_chunks`.
        unsafe { core::slice::from_raw_parts(self.values.as_ptr(), self.len) }
    }
}

impl<T: Primitive> DerefMut for Array<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`, and this is the only reference.
        unsafe { core::slice::from_raw_parts_mut(self.values.as_ptr(), self.len) }
    }
}

impl<T: Primitive> Drop for Array<T> {
    fn drop(&mut self) {
        let bytes = size_of::<T>() * self.len;
        if bytes == 0 {
            return;
        }
        let Ok(layout) = layout(bytes) else { failed!() };
        // SAFETY: allocated with this layout from this allocator, which
        // outlives it.
        unsafe { self.allocator.as_ref().deallocate(self.values.cast(), layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;

    #[test]
    fn holds_exactly_its_values() {
        let mut array = Array::<u32>::from_iter(&Heap, 0..5).unwrap();
        array[4] = 9;
        assert_eq!(&*array, &[0, 1, 2, 3, 9]);
        assert!(Array::<u64>::zeroed(&Heap, 0).unwrap().is_empty());
        let huge = layout(HUGE_PAGE_BYTES).unwrap().align();
        assert_eq!(huge, if cfg!(target_os = "linux") { HUGE_PAGE_BYTES } else { 64 });
    }
}
