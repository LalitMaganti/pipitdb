//! `SlowVec`: a growable list of values, in memory from the allocator it was made
//! with.

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, Buffer};

/// A value that didn't fit in a `SlowVec`. With `?`, it becomes an `AllocError`.
#[derive(PartialEq, Eq, Debug)]
pub struct Full<T>(pub T);

impl<T> From<Full<T>> for AllocError {
    fn from(_: Full<T>) -> AllocError {
        AllocError
    }
}

/// A list for building things now and then, such as plans and footers. It
/// stays small rather than fast: what doesn't depend on its values' type is
/// shared by every `SlowVec`, not copied for each type. Hot loops work on
/// buffers directly.
///
/// Every `SlowVec` has a most values it can hold, `max`, a power of two. One made
/// by `new` grows towards it as needed, doubling; one made by `fixed` has room
/// for `max` from the start, and never grows.
pub struct SlowVec<T> {
    raw: RawVec,
    values: PhantomData<T>,
}

/// A `SlowVec`'s memory and counts, which don't depend on its values' type.
struct RawVec {
    buffer: Buffer,
    values: NonNull<u8>,
    len: usize,
    capacity: usize,
    max: usize,
}

impl<T> SlowVec<T> {
    /// An empty `SlowVec` that can grow to `max` values.
    pub fn new(allocator: &dyn Allocator, max: usize) -> Result<SlowVec<T>, AllocError> {
        check!(max.is_power_of_two());
        SlowVec::empty(allocator, 0, max)
    }

    /// An empty `SlowVec` with room for `len` values, rounded up, that never
    /// grows: for lists whose length is known.
    pub fn fixed(allocator: &dyn Allocator, len: usize) -> Result<SlowVec<T>, AllocError> {
        let max = len.next_power_of_two();
        SlowVec::empty(allocator, max, max)
    }

    /// `values`, in a `SlowVec` that never grows.
    pub fn fixed_from(
        allocator: &dyn Allocator,
        values: impl ExactSizeIterator<Item = T>,
    ) -> Result<SlowVec<T>, AllocError> {
        let mut vec = SlowVec::fixed(allocator, values.len())?;
        // An iterator can claim a wrong length, so it can't overrun.
        for value in values.take(vec.raw.capacity) {
            // SAFETY: `len` is below the capacity.
            unsafe { vec.slot(vec.raw.len).write(value) };
            vec.raw.len += 1;
        }
        Ok(vec)
    }

    pub fn capacity(&self) -> usize {
        self.raw.capacity
    }

    /// An empty `SlowVec` with room for `capacity` values, 0 or `max`.
    fn empty(
        allocator: &dyn Allocator,
        capacity: usize,
        max: usize,
    ) -> Result<SlowVec<T>, AllocError> {
        const { assert!(size_of::<T>() > 0 && align_of::<T>() <= BUFFER_ALIGNMENT_BYTES) };
        Ok(SlowVec {
            raw: RawVec::new(allocator, size_of::<T>(), capacity, max)?,
            values: PhantomData,
        })
    }

    /// Where value `i` goes.
    fn slot(&self, i: usize) -> NonNull<T> {
        // SAFETY: callers only ask for places within the capacity.
        unsafe { self.raw.values.cast::<T>().add(i) }
    }

    /// Adds `value` at the end, growing if full. Fails, giving `value` back,
    /// if the `SlowVec` holds `max` values or can't grow.
    pub fn push(&mut self, value: T) -> Result<(), Full<T>> {
        if self.raw.len == self.raw.capacity && self.raw.make_room(size_of::<T>(), 1).is_err() {
            return Err(Full(value));
        }
        // SAFETY: there is room for a value at `len`.
        unsafe { self.slot(self.raw.len).write(value) };
        self.raw.len += 1;
        Ok(())
    }

    /// Adds copies of `values` at the end, in one go, growing if there isn't
    /// room. Fails, adding none, if that would take it past `max` or it
    /// can't grow.
    pub fn extend_from_slice(&mut self, values: &[T]) -> Result<(), AllocError>
    where
        T: Copy,
    {
        if values.len() > self.raw.capacity - self.raw.len {
            self.raw.make_room(size_of::<T>(), values.len())?;
        }
        // SAFETY: there is room for `values` from `len`, and they can't
        // overlap the `SlowVec`, which `values` doesn't borrow.
        unsafe {
            self.slot(self.raw.len)
                .copy_from_nonoverlapping(NonNull::from(values).cast(), values.len());
        };
        self.raw.len += values.len();
        Ok(())
    }

    /// Removes the last value, if any.
    pub fn pop(&mut self) -> Option<T> {
        self.raw.len = self.raw.len.checked_sub(1)?;
        // SAFETY: the value at the old last place was written, and is no
        // longer counted, so it's moved out once.
        Some(unsafe { self.slot(self.raw.len).read() })
    }

    /// Keeps the values `keep` says to, in order, dropping the rest.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let len = self.raw.len;
        // Counted as empty while values move, so a panic in `keep` leaks
        // rather than drops twice.
        self.raw.len = 0;
        let mut kept = 0;
        for i in 0..len {
            // SAFETY: values below `len` were written, and each is read or
            // dropped once here; `kept` never passes `i`.
            unsafe {
                let value = self.slot(i);
                if keep(value.as_ref()) {
                    if kept != i {
                        self.slot(kept).write(value.read());
                    }
                    kept += 1;
                } else {
                    value.drop_in_place();
                }
            }
        }
        self.raw.len = kept;
    }
}

impl RawVec {
    #[inline(never)]
    fn new(
        allocator: &dyn Allocator,
        item: usize,
        capacity: usize,
        max: usize,
    ) -> Result<RawVec, AllocError> {
        max.checked_mul(item).ok_or(AllocError)?;
        // SAFETY: only the first `len` values are read, and each is written
        // first.
        let mut buffer = unsafe { Buffer::allocate_uninit(allocator, capacity * item)? };
        let values = buffer.as_mut_non_null();
        Ok(RawVec { buffer, values, len: 0, capacity, max })
    }

    /// Grows so `additional` more values of `item` bytes fit, unless that's
    /// past `max`.
    #[cold]
    #[inline(never)]
    fn make_room(&mut self, item: usize, additional: usize) -> Result<(), AllocError> {
        let needed = self.len.checked_add(additional).ok_or(AllocError)?;
        if needed > self.max {
            return Err(AllocError);
        }
        // Powers of two up to `max` stay powers of two up to `max`.
        let doubled = if self.capacity == 0 { 4 } else { self.capacity * 2 };
        let capacity = doubled.max(needed.next_power_of_two()).min(self.max);
        // SAFETY: as in `new`.
        let mut grown = unsafe { self.buffer.allocate_uninit_like(capacity * item)? };
        let to = grown.as_mut_non_null();
        // SAFETY: both hold at least `len` values. They're moved, not
        // dropped: freeing a `Buffer` doesn't drop what's in it.
        unsafe { to.copy_from_nonoverlapping(self.values, self.len * item) };
        self.buffer = grown;
        self.values = to;
        self.capacity = capacity;
        Ok(())
    }
}

impl<T> Deref for SlowVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: the first `len` values were written by `push`.
        unsafe { core::slice::from_raw_parts(self.slot(0).as_ptr(), self.raw.len) }
    }
}

impl<T> DerefMut for SlowVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`, and this is the only reference.
        unsafe { core::slice::from_raw_parts_mut(self.slot(0).as_ptr(), self.raw.len) }
    }
}

impl<T> Drop for SlowVec<T> {
    fn drop(&mut self) {
        let values = core::ptr::slice_from_raw_parts_mut(self.slot(0).as_ptr(), self.raw.len);
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
        let mut values = SlowVec::new(&allocator, 128).unwrap();
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
        let mut values = SlowVec::new(&Heap, 16).unwrap();
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
        let mut values = SlowVec::new(&allocator, 128).unwrap();
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
        let mut values = SlowVec::fixed(&allocator, 3).unwrap();
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
        let _ = SlowVec::<u64>::new(&Heap, 100);
    }

    #[test]
    fn extends_in_one_go() {
        let left = Rc::new(Cell::new(u32::MAX));
        let live = Rc::new(Cell::new(0));
        let allocator = Limited { left: left.clone(), live };
        let mut values = SlowVec::new(&allocator, 16).unwrap();
        assert!(values.extend_from_slice(&[1, 2, 3, 4, 5]).is_ok());
        assert!(values.extend_from_slice(&[6]).is_ok());
        assert_eq!(*values, [1, 2, 3, 4, 5, 6]);
        // One buffer to start, and one grown to fit five.
        assert_eq!(u32::MAX - left.get(), 2);
        assert_eq!(values.extend_from_slice(&[0; 11]), Err(AllocError));
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn retains_and_pops() {
        let value = Rc::new(());
        let mut values = SlowVec::new(&Heap, 8).unwrap();
        for i in 0..6 {
            assert!(values.push((i, value.clone())).is_ok());
        }
        values.retain(|(i, _)| i % 2 == 1);
        assert_eq!(values.iter().map(|(i, _)| *i).collect::<alloc::vec::Vec<_>>(), [1, 3, 5]);
        assert_eq!(Rc::strong_count(&value), 4);
        assert_eq!(values.pop().map(|(i, _)| i), Some(5));
        assert_eq!(Rc::strong_count(&value), 3);
    }

    #[test]
    fn rounds_its_room_up() {
        let values = SlowVec::<u64>::fixed(&Heap, 5).unwrap();
        assert_eq!(values.capacity(), 8);
    }
}
