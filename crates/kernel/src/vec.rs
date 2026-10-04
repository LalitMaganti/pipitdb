//! `Vec`: a growable list of values, in memory from the allocator it was made
//! with.

use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, Buffer};

/// A value that didn't fit in a `Vec`. With `?`, it becomes an `AllocError`.
#[derive(PartialEq, Eq, Debug)]
pub struct Full<T>(pub T);

impl<T> From<Full<T>> for AllocError {
    fn from(_: Full<T>) -> AllocError {
        AllocError
    }
}

/// Every `Vec` has a most values it can hold, `max`, a power of two. One made
/// by `new` grows towards it as needed, doubling; one made by `fixed` has room
/// for `max` from the start, and never grows.
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
        check!(max.is_power_of_two());
        Vec::empty(allocator, 0, max)
    }

    /// An empty `Vec` with room for `len` values, rounded up, that never
    /// grows: for lists whose length is known.
    pub fn fixed<A: Allocator + Clone + 'static>(
        allocator: A,
        len: usize,
    ) -> Result<Vec<T>, AllocError> {
        let max = len.next_power_of_two();
        Vec::empty(allocator, max, max)
    }

    /// `values`, in a `Vec` that never grows.
    pub fn fixed_from<A: Allocator + Clone + 'static>(
        allocator: A,
        values: impl ExactSizeIterator<Item = T>,
    ) -> Result<Vec<T>, AllocError> {
        let mut vec = Vec::fixed(allocator, values.len())?;
        // An iterator can claim a wrong length, so it can't overrun.
        for value in values.take(vec.capacity) {
            // SAFETY: `len` is below the capacity.
            unsafe { vec.values.add(vec.len).write(value) };
            vec.len += 1;
        }
        Ok(vec)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// An empty `Vec` with room for `capacity` values, 0 or `max`.
    fn empty<A: Allocator + Clone + 'static>(
        allocator: A,
        capacity: usize,
        max: usize,
    ) -> Result<Vec<T>, AllocError> {
        const { assert!(size_of::<T>() > 0 && align_of::<T>() <= BUFFER_ALIGNMENT_BYTES) };
        max.checked_mul(size_of::<T>()).ok_or(AllocError)?;
        let mut buffer = allocate(allocator, capacity * size_of::<T>())?;
        let values = buffer.as_mut_non_null().cast();
        Ok(Vec { buffer, values, len: 0, capacity, max })
    }

    /// Adds `value` at the end, growing if full. Fails, giving `value` back,
    /// if the `Vec` holds `max` values or can't grow.
    pub fn push(&mut self, value: T) -> Result<(), Full<T>> {
        if self.len == self.capacity && self.make_room(1).is_err() {
            return Err(Full(value));
        }
        // SAFETY: there is room for a value at `len`.
        unsafe { self.values.add(self.len).write(value) };
        self.len += 1;
        Ok(())
    }

    /// Adds copies of `values` at the end, in one go, growing if there isn't
    /// room. Fails, adding none, if that would take it past `max` or it
    /// can't grow.
    pub fn extend_from_slice(&mut self, values: &[T]) -> Result<(), AllocError>
    where
        T: Copy,
    {
        if values.len() > self.capacity - self.len {
            self.make_room(values.len())?;
        }
        // SAFETY: there is room for `values` from `len`, and they can't
        // overlap the `Vec`, which `values` doesn't borrow.
        unsafe {
            self.values
                .add(self.len)
                .copy_from_nonoverlapping(NonNull::from(values).cast(), values.len());
        };
        self.len += values.len();
        Ok(())
    }

    /// Grows so `additional` more values fit, unless that's past `max`.
    #[cold]
    #[inline(never)]
    fn make_room(&mut self, additional: usize) -> Result<(), AllocError> {
        let needed = self.len.checked_add(additional).ok_or(AllocError)?;
        if needed > self.max {
            return Err(AllocError);
        }
        // Powers of two up to `max` stay powers of two up to `max`.
        let doubled = if self.capacity == 0 { 4 } else { self.capacity * 2 };
        let capacity = doubled.max(needed.next_power_of_two()).min(self.max);
        let item = size_of::<T>();
        self.values = grow(&mut self.buffer, self.len * item, capacity * item)?.cast();
        self.capacity = capacity;
        Ok(())
    }
}

/// Memory for a `Vec`'s values. Shared by every `Vec<T>` with an `A`, so it
/// isn't copied for each `T`.
#[inline(never)]
fn allocate<A: Allocator + Clone + 'static>(
    allocator: A,
    size_bytes: usize,
) -> Result<Buffer, AllocError> {
    // SAFETY: only the first `len` values are read, and each is written first.
    unsafe { Buffer::allocate_uninit(allocator, size_bytes) }
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
    // SAFETY: as in `Vec::empty`.
    let mut grown = unsafe { buffer.allocate_uninit_like(size_bytes)? };
    let to = grown.as_mut_non_null();
    // SAFETY: both hold at least `used_bytes`. Values are moved, not dropped:
    // freeing a `Buffer` doesn't drop what's in it.
    unsafe { to.copy_from_nonoverlapping(buffer.as_mut_non_null(), used_bytes) };
    *buffer = grown;
    Ok(to)
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
        assert_eq!(values.push(4), Err(Full(4)));
        assert_eq!(*values, [0, 1, 2, 3]);
    }

    #[test]
    fn stops_at_its_max() {
        let left = Rc::new(Cell::new(1));
        let allocator = Limited { left: left.clone(), live: Rc::new(Cell::new(0)) };
        let mut values = Vec::fixed(allocator, 3).unwrap();
        for i in 0..4 {
            assert!(values.push(i).is_ok());
        }
        assert_eq!(values.push(4), Err(Full(4)));
        // Made with room for its max, it never grew.
        assert_eq!(left.get(), 0);
        assert_eq!(values.capacity(), 4);
    }

    #[test]
    #[should_panic(expected = "power_of_two")]
    fn max_is_a_power_of_two() {
        let _ = Vec::<u64>::new(Heap, 100);
    }

    #[test]
    fn extends_in_one_go() {
        let left = Rc::new(Cell::new(u32::MAX));
        let live = Rc::new(Cell::new(0));
        let mut values = Vec::new(Limited { left: left.clone(), live }, 16).unwrap();
        assert!(values.extend_from_slice(&[1, 2, 3, 4, 5]).is_ok());
        assert!(values.extend_from_slice(&[6]).is_ok());
        assert_eq!(*values, [1, 2, 3, 4, 5, 6]);
        // One buffer to start, and one grown to fit five.
        assert_eq!(u32::MAX - left.get(), 2);
        assert_eq!(values.extend_from_slice(&[0; 11]), Err(AllocError));
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn rounds_its_room_up() {
        let values = Vec::<u64>::fixed(Heap, 5).unwrap();
        assert_eq!(values.capacity(), 8);
    }
}
