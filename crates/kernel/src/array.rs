//! `Array`: a fixed number of values in one allocation of exactly their
//! size, for data that must be contiguous, such as what a sort splits into
//! slices. It never grows: data that grows goes in blocks that never move.

use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator, layout};
use crate::buffer::Primitive;
use crate::global::Global;

/// `len` values, zeroed when made.
pub(crate) struct Array<T: Primitive> {
    values: NonNull<T>,
    len: usize,
    /// What the values came from.
    global: NonNull<Global>,
    owns: PhantomData<T>,
}

impl<T: Primitive> Array<T> {
    /// `len` zeroes: fresh memory from the system is zero already, so a big
    /// one costs nothing to make.
    pub(crate) fn zeroed(global: &Global, len: usize) -> Result<Array<T>, AllocError> {
        Array::allocate(global, len, true)
    }

    /// `len` values, value `i` being `value(i)`, each written once.
    pub(crate) fn from_fn(
        global: &Global,
        len: usize,
        mut value: impl FnMut(usize) -> T,
    ) -> Result<Array<T>, AllocError> {
        let array = Array::<T>::allocate(global, len, false)?;
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
        global: &Global,
        len: usize,
        chunk: usize,
        mut write: impl FnMut(usize, &mut [MaybeUninit<T>]),
    ) -> Result<Array<T>, AllocError> {
        check!(chunk > 0);
        let array = Array::<T>::allocate(global, len, false)?;
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

    /// Room for `len` values, zeroed if `zeroed`, else unwritten: callers
    /// that don't zero write each value before it's read.
    fn allocate(global: &Global, len: usize, zeroed: bool) -> Result<Array<T>, AllocError> {
        let bytes = size_of::<T>().checked_mul(len).ok_or(AllocError)?;
        let values = match bytes {
            0 => NonNull::dangling(),
            _ => allocate_bytes(global, bytes, zeroed)?.cast(),
        };
        Ok(Array { values, len, global: NonNull::from(global), owns: PhantomData })
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
        if bytes != 0 {
            // SAFETY: allocated by `allocate_bytes`, with these bytes, from
            // this `Global`, which outlives it.
            unsafe { free_bytes(self.global.as_ref(), self.values.cast(), bytes) };
        }
    }
}

/// `bytes` of memory, zeroed if `zeroed`. Out of line, and of no type, so
/// every `Array` shares it.
#[inline(never)]
fn allocate_bytes(global: &Global, bytes: usize, zeroed: bool) -> Result<NonNull<u8>, AllocError> {
    let layout = layout(bytes)?;
    if zeroed { global.allocate_zeroed(layout) } else { global.allocate(layout) }
}

/// Frees `bytes` at `values`.
///
/// # Safety
///
/// They must be from `allocate_bytes`, with `global`.
#[inline(never)]
unsafe fn free_bytes(global: &Global, values: NonNull<u8>, bytes: usize) {
    let Ok(layout) = layout(bytes) else { failed!() };
    // SAFETY: as the caller promises.
    unsafe { global.deallocate(values, layout) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_exactly_its_values() {
        let global = Global::current();
        let mut array = Array::<u32>::from_fn(global, 5, |i| u32::try_from(i).unwrap()).unwrap();
        array[4] = 9;
        assert_eq!(&*array, &[0, 1, 2, 3, 9]);
        assert!(Array::<u64>::zeroed(global, 0).unwrap().is_empty());
    }
}
