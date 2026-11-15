//! `Array`: a fixed number of values in one allocation of exactly their
//! size, for data that must be contiguous, such as what a sort splits into
//! slices. It never grows: data that grows goes in blocks that never move.

use core::alloc::Layout;
use core::marker::PhantomData;
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
        let bytes = size_of::<T>().checked_mul(len).ok_or(AllocError)?;
        let values = if bytes == 0 {
            NonNull::dangling()
        } else {
            allocator.allocate_zeroed(layout(bytes)?)?.cast()
        };
        // SAFETY: only the lifetime changes; the allocator outlives the values.
        let allocator = unsafe {
            core::mem::transmute::<NonNull<dyn Allocator + '_>, NonNull<dyn Allocator>>(
                NonNull::from(allocator),
            )
        };
        Ok(Array { values, len, allocator, owns: PhantomData })
    }

    /// The values of `values`, in order.
    pub(crate) fn from_iter(
        allocator: &dyn Allocator,
        values: impl ExactSizeIterator<Item = T>,
    ) -> Result<Array<T>, AllocError> {
        let mut array = Array::zeroed(allocator, values.len())?;
        for (out, value) in array.iter_mut().zip(values) {
            *out = value;
        }
        Ok(array)
    }
}

impl<T: Primitive> Deref for Array<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: `len` values, all written, as zeroes at least.
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
