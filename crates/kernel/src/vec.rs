//! `Vec`: a growable list of values, in memory from the allocator it was made
//! with.

use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, Buffer};

/// Every `Vec` has a most values it can hold, `max`. It grows towards it as
/// needed, doubling, so one made with room for `max` never grows. `max` is a
/// power of two, and so is the room for values, unless it is 0.
///
/// Its values live in a `Buffer`, which also remembers the allocator, so a
/// `Vec` isn't generic over it.
pub struct Vec<T> {
    buffer: Buffer,
    values: NonNull<T>,
    len: usize,
    capacity: usize,
    max: usize,
}

impl<T> Vec<T> {
    /// An empty `Vec` that can grow to `max` values.
    pub fn new<A: Allocator + Clone + 'static>(
        allocator: A,
        max: usize,
    ) -> Result<Vec<T>, AllocError> {
        Vec::with_capacity(allocator, 0, max)
    }

    /// An empty `Vec` with room for `capacity` values, that can grow to `max`.
    pub fn with_capacity<A: Allocator + Clone + 'static>(
        allocator: A,
        capacity: usize,
        max: usize,
    ) -> Result<Vec<T>, AllocError> {
        const { assert!(size_of::<T>() > 0 && align_of::<T>() <= BUFFER_ALIGNMENT_BYTES) };
        check!(max.is_power_of_two());
        check!(capacity == 0 || capacity.is_power_of_two());
        check!(capacity <= max);
        max.checked_mul(size_of::<T>()).ok_or(AllocError)?;
        let size_bytes = capacity * size_of::<T>();
        // SAFETY: only the first `len` values are read, and each is written
        // first.
        let mut buffer = unsafe { Buffer::allocate_uninit(allocator, size_bytes)? };
        let values = data(&mut buffer).cast();
        Ok(Vec { buffer, values, len: 0, capacity, max })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Adds `value` at the end, growing if full. Fails, giving `value` back,
    /// if the `Vec` holds `max` values or can't grow.
    pub fn push(&mut self, value: T) -> Result<(), T> {
        if self.len == self.capacity {
            if self.len == self.max {
                return Err(value);
            }
            // Powers of two below `max` double to at most `max`.
            let capacity = if self.capacity == 0 { self.max.min(4) } else { self.capacity * 2 };
            let item = size_of::<T>();
            match grow(&mut self.buffer, self.len * item, capacity * item) {
                Ok(values) => self.values = values.cast(),
                Err(AllocError) => return Err(value),
            }
            self.capacity = capacity;
        }
        // SAFETY: there is room for a value at `len`.
        unsafe { self.values.add(self.len).write(value) };
        self.len += 1;
        Ok(())
    }
}

/// Moves the first `used_bytes` of `buffer` to one of `size_bytes`, from the
/// same allocator. Shared by every `Vec<T>`, so it isn't copied for each `T`.
#[cold]
#[inline(never)]
fn grow(
    buffer: &mut Buffer,
    used_bytes: usize,
    size_bytes: usize,
) -> Result<NonNull<u8>, AllocError> {
    // SAFETY: as in `Vec::with_capacity`.
    let mut grown = unsafe { buffer.allocate_uninit_like(size_bytes)? };
    let to = data(&mut grown);
    // SAFETY: both hold at least `used_bytes`. Values are moved, not dropped:
    // freeing a `Buffer` doesn't drop what's in it.
    unsafe { to.copy_from_nonoverlapping(data(buffer), used_bytes) };
    *buffer = grown;
    Ok(to)
}

fn data(buffer: &mut Buffer) -> NonNull<u8> {
    let Some(data) = NonNull::new(buffer.as_mut_ptr::<u8>()) else {
        crate::check::check_failed(line!());
    };
    data
}

impl<T> Deref for Vec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: the first `len` values were written by `push`.
        unsafe { core::slice::from_raw_parts(self.values.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for Vec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`, and this is the only reference.
        unsafe { core::slice::from_raw_parts_mut(self.values.as_ptr(), self.len) }
    }
}

impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        let values = core::ptr::slice_from_raw_parts_mut(self.values.as_ptr(), self.len);
        // SAFETY: the first `len` values were written by `push`, and are
        // dropped once: freeing the buffer doesn't drop them.
        unsafe { values.drop_in_place() };
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;
    use core::alloc::Layout;
    use core::cell::Cell;
    use core::ptr::NonNull;

    use super::*;
    use crate::allocator::Heap;

    /// Allows `left` allocations, counting those still live.
    #[derive(Clone)]
    struct Limited {
        left: Rc<Cell<u32>>,
        live: Rc<Cell<u32>>,
    }

    // SAFETY: forwards to `Heap`.
    unsafe impl Allocator for Limited {
        fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
            if self.left.get() == 0 {
                return Err(AllocError);
            }
            self.left.set(self.left.get() - 1);
            self.live.set(self.live.get() + 1);
            Heap.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            self.live.set(self.live.get() - 1);
            // SAFETY: forwarded from the caller.
            unsafe { Heap.deallocate(ptr, layout) }
        }
    }

    #[test]
    fn grows_from_its_allocator() {
        let live = Rc::new(Cell::new(0));
        let allocator = Limited { left: Rc::new(Cell::new(u32::MAX)), live: live.clone() };
        let mut values = Vec::new(allocator, 128).unwrap();
        for i in 0..100 {
            assert!(values.push(i).is_ok());
        }
        assert_eq!(values.iter().sum::<u64>(), 4950);
        assert_eq!(live.get(), 1);
        drop(values);
        assert_eq!(live.get(), 0);
    }

    #[test]
    fn drops_its_values() {
        let value = Rc::new(());
        let mut values = Vec::new(Heap, 16).unwrap();
        for _ in 0..10 {
            assert!(values.push(value.clone()).is_ok());
        }
        assert_eq!(Rc::strong_count(&value), 11);
        drop(values);
        assert_eq!(Rc::strong_count(&value), 1);
    }

    #[test]
    fn gives_the_value_back_if_it_cant_grow() {
        let left = Rc::new(Cell::new(2));
        let allocator = Limited { left, live: Rc::new(Cell::new(0)) };
        let mut values = Vec::new(allocator, 128).unwrap();
        for i in 0..4 {
            assert!(values.push(i).is_ok());
        }
        assert_eq!(values.push(4), Err(4));
        assert_eq!(*values, [0, 1, 2, 3]);
    }

    #[test]
    fn stops_at_its_max() {
        let left = Rc::new(Cell::new(1));
        let allocator = Limited { left: left.clone(), live: Rc::new(Cell::new(0)) };
        let mut values = Vec::with_capacity(allocator, 4, 4).unwrap();
        for i in 0..4 {
            assert!(values.push(i).is_ok());
        }
        assert_eq!(values.push(4), Err(4));
        // Made with room for its max, it never grew.
        assert_eq!(left.get(), 0);
        assert_eq!(values.capacity(), 4);
    }

    #[test]
    #[should_panic(expected = "power_of_two")]
    fn max_is_a_power_of_two() {
        let _ = Vec::<u64>::new(Heap, 100);
    }
}
