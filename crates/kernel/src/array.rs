//! `Array`: a fixed number of values in one allocation.

use core::marker::PhantomData;
use core::ops::Deref;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, Buffer};

/// Its values are made once, by `new`, and dropped with it.
pub struct Array<T> {
    buffer: Buffer,
    len: usize,
    values: PhantomData<T>,
}

impl<T> Array<T> {
    /// An array of `len` values, the `i`th being `value(i)`. If a value fails,
    /// so does the array, dropping the values made.
    pub fn new<A: Allocator + Clone + 'static>(
        allocator: A,
        len: usize,
        mut value: impl FnMut(usize) -> Result<T, AllocError>,
    ) -> Result<Array<T>, AllocError> {
        const { assert!(align_of::<T>() <= BUFFER_ALIGNMENT_BYTES) };
        let size_bytes = len.checked_mul(size_of::<T>()).ok_or(AllocError)?;
        // SAFETY: each value is written below before it can be read.
        let mut buffer = unsafe { Buffer::allocate_uninit(allocator, size_bytes)? };
        let values = buffer.as_mut_ptr::<u8>().cast::<T>();
        let mut array = Array { buffer, len: 0, values: PhantomData };
        for i in 0..len {
            // SAFETY: the buffer has room for `len` values, aligned for `T`.
            unsafe { values.add(i).write(value(i)?) };
            array.len = i + 1;
        }
        Ok(array)
    }
}

impl<T> Deref for Array<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        let values = self.buffer.as_ptr::<u8>().cast::<T>();
        // SAFETY: the first `len` values were written by `new`.
        unsafe { core::slice::from_raw_parts(values, self.len) }
    }
}

impl<T> Drop for Array<T> {
    fn drop(&mut self) {
        let values = self.buffer.as_mut_ptr::<u8>().cast::<T>();
        // SAFETY: the first `len` values were written by `new`, and are
        // dropped once.
        unsafe { core::ptr::slice_from_raw_parts_mut(values, self.len).drop_in_place() };
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use super::*;
    use crate::allocator::Heap;

    #[test]
    fn holds_and_drops_its_values() {
        let live = Rc::new(());
        let array = Array::new(Heap, 3, |i| Ok((i, live.clone()))).unwrap();
        assert_eq!(array.iter().map(|(i, _)| *i).sum::<usize>(), 3);
        assert_eq!(Rc::strong_count(&live), 4);
        drop(array);
        assert_eq!(Rc::strong_count(&live), 1);
    }

    #[test]
    fn can_be_empty() {
        let array = Array::<u64>::new(Heap, 0, |_| Ok(0)).unwrap();
        assert!(array.is_empty());
    }

    #[test]
    fn fails_with_a_value() {
        let live = Rc::new(());
        let array = Array::new(Heap, 3, |i| if i < 2 { Ok(live.clone()) } else { Err(AllocError) });
        assert!(array.is_err());
        assert_eq!(Rc::strong_count(&live), 1);
    }
}
