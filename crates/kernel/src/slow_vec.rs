//! `SlowVec`: a growable list of values, in memory from the allocator it was made
//! with.

use core::alloc::Layout;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, Primitive};

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
/// The counts of values that fit are `u32`s, so a `SlowVec` is five words.
struct RawVec {
    /// Room for exactly `capacity` values, the first `len` of them written.
    values: NonNull<u8>,
    /// Where `values` came from, which outlives them, as callers promise.
    allocator: NonNull<dyn Allocator>,
    /// How many values there are.
    len: usize,
    /// How many values fit: a power of two, or 0 before any are added.
    capacity: u32,
    /// The most values that can ever fit, a power of two.
    max: u32,
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
        for value in values.take(vec.raw.capacity as usize) {
            // SAFETY: `len` is below the capacity.
            unsafe { vec.slot(vec.raw.len).write(value) };
            vec.raw.len += 1;
        }
        Ok(vec)
    }

    /// `len` zeros, in a `SlowVec` that never grows. Zeroed by the
    /// allocator, which on fresh pages needn't write them.
    #[inline]
    pub fn zeroed(allocator: &dyn Allocator, len: usize) -> Result<SlowVec<T>, AllocError>
    where
        T: Primitive,
    {
        const { assert!(size_of::<T>() > 0 && align_of::<T>() <= BUFFER_ALIGNMENT_BYTES) };
        let max = len.next_power_of_two();
        let mut raw = RawVec::new(allocator, size_of::<T>(), max, max, true)?;
        raw.len = len;
        Ok(SlowVec { raw, values: PhantomData })
    }

    pub fn capacity(&self) -> usize {
        self.raw.capacity as usize
    }

    /// An empty `SlowVec` with room for `capacity` values, 0 or `max`.
    fn empty(
        allocator: &dyn Allocator,
        capacity: usize,
        max: usize,
    ) -> Result<SlowVec<T>, AllocError> {
        const { assert!(size_of::<T>() > 0 && align_of::<T>() <= BUFFER_ALIGNMENT_BYTES) };
        Ok(SlowVec {
            raw: RawVec::new(allocator, size_of::<T>(), capacity, max, false)?,
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
        if self.raw.len == self.raw.capacity as usize
            && self.raw.make_room(size_of::<T>(), 1).is_err()
        {
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
        if values.len() > self.raw.capacity as usize - self.raw.len {
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

/// A type with the strictest alignment a `SlowVec`'s values can have. An
/// empty `SlowVec` allocates nothing, but its values must still start at an
/// aligned address, as slices of them need: a dangling pointer to this one
/// is that address.
#[repr(align(64))]
struct Aligned;

// `repr(align)` takes only a literal, so this checks it's the buffers'
// alignment.
const _: () = assert!(align_of::<Aligned>() == BUFFER_ALIGNMENT_BYTES);

/// 16 KB: the largest memory page among the systems we run on (Apple's
/// are 16 KB, most others' 4 KB), so memory aligned to it starts on a page
/// everywhere.
const PAGE_BYTES: usize = 1 << 14;

/// 256 KB: a `SlowVec` this big or bigger asks for page-aligned memory.
const LARGE_BYTES: usize = 1 << 18;

/// The layout of the memory for `capacity` values of `item` bytes each.
///
/// From `LARGE_BYTES` up, it asks for page alignment, as a hint to the
/// allocator: this is a big block that one owner keeps for a long time,
/// such as a hash table, so it's best taken from the system as whole pages
/// and given back to it when freed, rather than kept in a heap.
#[inline(never)]
fn layout(item: usize, capacity: usize) -> Result<Layout, AllocError> {
    let bytes = item.checked_mul(capacity).ok_or(AllocError)?;
    let align = if bytes >= LARGE_BYTES { PAGE_BYTES } else { BUFFER_ALIGNMENT_BYTES };
    Layout::from_size_align(bytes, align).map_err(|_| AllocError)
}

impl RawVec {
    /// Room for `capacity` values of `item` bytes, zeroed if `zeroed`, and
    /// at most `max`, which is at least `capacity`.
    #[inline(never)]
    fn new(
        allocator: &dyn Allocator,
        item: usize,
        capacity: usize,
        max: usize,
        zeroed: bool,
    ) -> Result<RawVec, AllocError> {
        let Ok(max) = u32::try_from(max) else { return Err(AllocError) };
        let values = match capacity {
            // Nothing to allocate; aligned for any value, as slices need.
            0 => NonNull::<Aligned>::dangling().cast(),
            _ if zeroed => allocator.allocate_zeroed(layout(item, capacity)?)?,
            _ => allocator.allocate(layout(item, capacity)?)?,
        };
        // The allocator outlives the values, as callers promise, so its
        // lifetime can be forgotten.
        // SAFETY: only the lifetime changes.
        let allocator = unsafe {
            core::mem::transmute::<NonNull<dyn Allocator + '_>, NonNull<dyn Allocator>>(
                NonNull::from(allocator),
            )
        };
        #[expect(clippy::cast_possible_truncation, reason = "at most `max`")]
        let capacity = capacity as u32;
        Ok(RawVec { values, allocator, len: 0, capacity, max })
    }

    /// Grows so `additional` more values of `item` bytes fit, unless that's
    /// past `max`.
    #[cold]
    #[inline(never)]
    fn make_room(&mut self, item: usize, additional: usize) -> Result<(), AllocError> {
        let needed = self.len.checked_add(additional).ok_or(AllocError)?;
        if needed > self.max as usize {
            return Err(AllocError);
        }
        // Powers of two up to `max` stay powers of two up to `max`.
        let doubled = if self.capacity == 0 { 4 } else { self.capacity as usize * 2 };
        let capacity = doubled.max(needed.next_power_of_two()).min(self.max as usize);
        // SAFETY: the allocator outlives the values.
        let allocator = unsafe { self.allocator.as_ref() };
        let to = allocator.allocate(layout(item, capacity)?)?;
        // SAFETY: both hold at least `len` values. They're moved, not
        // dropped.
        unsafe { to.copy_from_nonoverlapping(self.values, self.len * item) };
        // SAFETY: the old values are moved out, and nothing uses them again.
        unsafe { self.free(item) };
        self.values = to;
        #[expect(clippy::cast_possible_truncation, reason = "at most `max`")]
        let capacity = capacity as u32;
        self.capacity = capacity;
        Ok(())
    }

    /// Frees the memory, without dropping what's in it. Out of line, as
    /// every type's `drop` calls it.
    ///
    /// # Safety
    ///
    /// Nothing may use the values again; `item` is their size.
    #[inline(never)]
    unsafe fn free(&mut self, item: usize) {
        if self.capacity == 0 {
            return;
        }
        let Ok(layout) = layout(item, self.capacity as usize) else {
            crate::check::check_failed(line!())
        };
        // SAFETY: allocated with this layout from this allocator, which
        // outlives it.
        unsafe { self.allocator.as_ref().deallocate(self.values, layout) };
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
        // dropped once, then their memory is freed.
        unsafe {
            values.drop_in_place();
            self.raw.free(size_of::<T>());
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;
    use alloc::vec::Vec as StdVec;
    use core::cell::{Cell, RefCell};
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
        // An empty `SlowVec` allocates nothing; this is the room for four.
        let left = Rc::new(Cell::new(1));
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
    fn zeroed_holds_zeros() {
        let values = SlowVec::<i64>::zeroed(&Heap, 5).unwrap();
        assert_eq!(*values, [0; 5]);
        assert_eq!(values.capacity(), 8);
    }

    /// Records the layouts asked for.
    struct Layouts(RefCell<StdVec<Layout>>);

    // SAFETY: forwards to `Heap`.
    unsafe impl Allocator for Layouts {
        fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
            self.0.borrow_mut().push(layout);
            Heap.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            // SAFETY: forwarded from the caller.
            unsafe { Heap.deallocate(ptr, layout) }
        }
    }

    #[test]
    fn asks_for_exact_sizes_and_pages_when_large() {
        let layouts = Layouts(RefCell::new(StdVec::new()));
        drop(SlowVec::<u64>::fixed(&layouts, 100).unwrap());
        drop(SlowVec::<u64>::fixed(&layouts, 1 << 15).unwrap());
        let asked: StdVec<_> = layouts.0.borrow().iter().map(|l| (l.size(), l.align())).collect();
        assert_eq!(asked, [(128 * 8, 64), (1 << 18, PAGE_BYTES)]);
    }

    #[test]
    fn empty_allocates_nothing() {
        let left = Rc::new(Cell::new(0));
        let allocator = Limited { left, live: Rc::new(Cell::new(0)) };
        let values = SlowVec::<u64>::new(&allocator, 16).unwrap();
        assert!(values.is_empty());
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
        // Nothing to start, then room grown to fit five.
        assert_eq!(u32::MAX - left.get(), 1);
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
