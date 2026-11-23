//! Containers: what holds values in memory from an `Env`, other than a
//! column's `Buffer`s, which batches share.
//!
//! - `Array`: a fixed number of values, allocated exactly.
//! - `Box` and `ErasedBox`: one value.
//! - `SmallVec`: a growable list, as long as the query, not its data, says.
//! - `InlineVec`: a short list held in place.
//! - `Blocks`: data that grows with the rows, in blocks that never move.
//! - `BitSlice`, `Bitmap` and `RankBitmap`: bits packed in words.

#![allow(unsafe_code)]

use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator, layout};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, Primitive};
use crate::env::Env;
// `Array`: a fixed number of values in one allocation of exactly their
// size, for data that must be contiguous, such as what a sort splits into
// slices. It never grows: data that grows goes in blocks that never move.

/// `len` values, zeroed when made.
pub struct Array<T: Primitive> {
    /// Where the values start.
    values: NonNull<T>,
    /// How many values there are.
    len: usize,
    /// What the values came from.
    env: NonNull<Env>,
    /// Says it owns `T`s, for the drop checker.
    owns: PhantomData<T>,
}

impl<T: Primitive> Array<T> {
    /// `len` zeroes: fresh memory from the system is zero already, so a big
    /// one costs nothing to make.
    pub fn zeroed(env: &Env, len: usize) -> Result<Array<T>, AllocError> {
        Array::allocate(env, len, true)
    }

    /// `len` values, value `i` being `value(i)`, each written once.
    pub fn from_fn(
        env: &Env,
        len: usize,
        mut value: impl FnMut(usize) -> T,
    ) -> Result<Array<T>, AllocError> {
        let array = Array::<T>::allocate(env, len, false)?;
        for i in 0..len {
            // SAFETY: there's room for `len` values.
            unsafe { array.values.as_ptr().add(i).write(value(i)) };
        }
        Ok(array)
    }

    /// `len` values, written a chunk of up to `chunk` at a time by
    /// `write(start, out)`, values `start` on.
    ///
    /// # Safety
    ///
    /// `write` must write every one of `out`.
    pub unsafe fn from_chunks(
        env: &Env,
        len: usize,
        chunk: usize,
        mut write: impl FnMut(usize, &mut [MaybeUninit<T>]),
    ) -> Result<Array<T>, AllocError> {
        check!(chunk > 0);
        let array = Array::<T>::allocate(env, len, false)?;
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

    /// No values, which allocates nothing.
    pub fn empty(env: &Env) -> Array<T> {
        let values = NonNull::<Aligned>::dangling().cast();
        Array { values, len: 0, env: NonNull::from(env), owns: PhantomData }
    }

    /// Room for `len` values, not written.
    ///
    /// # Safety
    ///
    /// No value may be read before it's written.
    pub unsafe fn uninit(env: &Env, len: usize) -> Result<Array<T>, AllocError> {
        Array::allocate(env, len, false)
    }

    /// The `Env` the values came from.
    pub fn env(&self) -> &Env {
        // SAFETY: an `Env` outlives what's allocated from it.
        unsafe { self.env.as_ref() }
    }

    /// Where the values start, for writing values that aren't `T`, such as
    /// a `SmallVec`'s.
    pub fn as_non_null(&self) -> NonNull<T> {
        self.values
    }

    /// Room for `len` values, zeroed if `zeroed`, else unwritten: callers
    /// that don't zero write each value before it's read.
    fn allocate(env: &Env, len: usize, zeroed: bool) -> Result<Array<T>, AllocError> {
        let bytes = size_of::<T>().checked_mul(len).ok_or(AllocError)?;
        let values = match bytes {
            0 => return Ok(Array::empty(env)),
            _ => allocate_bytes(env, bytes, zeroed)?.cast(),
        };
        Ok(Array { values, len, env: NonNull::from(env), owns: PhantomData })
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
    /// Out of line, so dropping one, or a `SmallVec`, is a call.
    #[inline(never)]
    fn drop(&mut self) {
        let bytes = size_of::<T>() * self.len;
        if bytes != 0 {
            // SAFETY: allocated by `allocate_bytes`, with these bytes, from
            // this `Env`, which outlives it.
            unsafe { free_bytes(self.env.as_ref(), self.values.cast(), bytes) };
        }
    }
}

/// A type with the strictest alignment an `Array`'s values can have, even
/// when it holds another type's, as a `SmallVec`'s does. An empty one
/// allocates nothing, but its values must still start at an aligned
/// address, as slices of them need: a dangling pointer to this is that
/// address.
#[repr(align(64))]
struct Aligned;

// `repr(align)` takes only a literal, so this checks it's the buffers'
// alignment.
const _: () = assert!(align_of::<Aligned>() == BUFFER_ALIGNMENT_BYTES);

/// `bytes` of memory, zeroed if `zeroed`. Out of line, and of no type, so
/// every `Array` and `Box` shares it.
#[inline(never)]
pub(crate) fn allocate_bytes(
    env: &Env,
    bytes: usize,
    zeroed: bool,
) -> Result<NonNull<u8>, AllocError> {
    let layout = layout(bytes)?;
    if zeroed { env.allocate_zeroed(layout) } else { env.allocate(layout) }
}

/// Frees `bytes` at `values`.
///
/// # Safety
///
/// They must be from `allocate_bytes`, with `env`.
#[inline(never)]
pub(crate) unsafe fn free_bytes(env: &Env, values: NonNull<u8>, bytes: usize) {
    let Ok(layout) = layout(bytes) else { failed!() };
    // SAFETY: as the caller promises.
    unsafe { env.deallocate(values, layout) };
}

// `Box`: one value, in memory from the `Env` it was made with.
// `ErasedBox`: a `Box` whose type has been forgotten.

/// The value lives in memory taken as an `Array`'s is, and remembers its
/// `Env`, so a `Box` isn't generic over it. It holds no length, as an
/// `Array` would: a `Box<T>`'s is a `T`'s.
pub struct Box<T> {
    value: NonNull<T>,
    env: NonNull<Env>,
}

impl<T> Box<T> {
    pub fn new(env: &Env, value: T) -> Result<Box<T>, AllocError> {
        const { assert!(align_of::<T>() <= BUFFER_ALIGNMENT_BYTES) };
        let value_ptr = match size_of::<T>() {
            0 => NonNull::dangling(),
            bytes => allocate_bytes(env, bytes, false)?.cast::<T>(),
        };
        // SAFETY: there's room for a `T`, aligned for it.
        unsafe { value_ptr.write(value) };
        Ok(Box { value: value_ptr, env: NonNull::from(env) })
    }

    /// Forgets the value's type, keeping what's needed to drop it.
    pub fn erase(self) -> ErasedBox {
        let this = ManuallyDrop::new(self);
        ErasedBox { value: this.value.cast(), env: this.env, drop: drop_and_free::<T> }
    }
}

/// Drops the `T` at `value` and frees its memory.
///
/// # Safety
///
/// `value` must be a `Box<T>`'s, from `env`, and not used again.
unsafe fn drop_and_free<T>(value: NonNull<()>, env: &Env) {
    let value = value.cast::<T>();
    // SAFETY: as the caller promises.
    unsafe {
        value.drop_in_place();
        if size_of::<T>() != 0 {
            free_bytes(env, value.cast(), size_of::<T>());
        }
    }
}

impl<T> Deref for Box<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: the value was written by `new`, and lives as long as `self`.
        unsafe { self.value.as_ref() }
    }
}

impl<T> DerefMut for Box<T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as in `deref`, and this is the only reference.
        unsafe { self.value.as_mut() }
    }
}

impl<T> Drop for Box<T> {
    fn drop(&mut self) {
        // SAFETY: the value is this `Box`'s, from its `Env`, which outlives
        // it, and isn't used again.
        unsafe { drop_and_free::<T>(self.value.cast(), self.env.as_ref()) };
    }
}

/// A value of a type only its maker knows, which it drops.
pub struct ErasedBox {
    /// The value, of the type `drop` was made for.
    value: NonNull<()>,
    /// What the value's memory came from.
    env: NonNull<Env>,
    /// Drops the value and frees its memory.
    drop: unsafe fn(NonNull<()>, &Env),
}

impl ErasedBox {
    /// The value, for code that knows its type to cast back.
    pub fn as_ptr(&self) -> NonNull<()> {
        self.value
    }
}

impl Drop for ErasedBox {
    fn drop(&mut self) {
        // SAFETY: `drop` was made for the value's type, runs once, and the
        // `Env` outlives what's allocated from it.
        unsafe { (self.drop)(self.value, self.env.as_ref()) }
    }
}

// `SmallVec`: a growable list of values whose count is bounded by the
// query, not by its data, in memory from the `Env` it was made with.

/// A list of things a query has as many of as its text or its constants
/// say, such as plan nodes, a footer's columns or a pipeline's steps: never
/// as many as its rows. Data that grows with the rows goes in an `Array`,
/// if its size is known, or in `Blocks`. It stays small rather than fast:
/// what doesn't depend on its values' type is shared by every `SmallVec`,
/// not copied for each type.
///
/// Every `SmallVec` has a most values it can hold, `max`, a power of two. One made
/// by `new` grows towards it as needed, doubling; one made by `fixed` has room
/// for `max` from the start, and never grows.
pub struct SmallVec<T> {
    raw: RawVec,
    values: PhantomData<T>,
}

/// A `SmallVec`'s memory and counts, which don't depend on its values' type.
/// Its counts are `u32`s, so a `SmallVec` is four words.
struct RawVec {
    /// Room for a power of two values, or none before any are added, the
    /// first `len` of them written. Dropped by `SmallVec`'s drop, so a
    /// `SmallVec` is dropped in one call.
    memory: ManuallyDrop<Array<u8>>,
    /// How many values there are.
    len: u32,
    /// The most values that can ever fit, a power of two.
    max: u32,
}

impl<T> SmallVec<T> {
    /// An empty `SmallVec` that can grow to `max` values. It allocates
    /// nothing until a value is added, so never fails: a `Result` only so
    /// callers can treat every constructor alike.
    #[inline]
    pub fn new(env: &Env, max: usize) -> Result<SmallVec<T>, AllocError> {
        const { assert!(size_of::<T>() > 0 && align_of::<T>() <= BUFFER_ALIGNMENT_BYTES) };
        check!(max.is_power_of_two());
        let Ok(max) = u32::try_from(max) else { failed!() };
        let raw = RawVec { memory: ManuallyDrop::new(Array::empty(env)), len: 0, max };
        Ok(SmallVec { raw, values: PhantomData })
    }

    /// An empty `SmallVec` with room for `len` values, rounded up, that never
    /// grows: for lists whose length is known.
    #[inline]
    pub fn fixed(env: &Env, len: usize) -> Result<SmallVec<T>, AllocError> {
        SmallVec::empty(env, len, len)
    }

    /// An empty `SmallVec` that allocates nothing and never grows.
    pub fn none(env: &Env) -> SmallVec<T> {
        let raw = RawVec { memory: ManuallyDrop::new(Array::empty(env)), len: 0, max: 0 };
        SmallVec { raw, values: PhantomData }
    }

    /// `values`, in a `SmallVec` that never grows.
    #[inline]
    pub fn fixed_from(
        env: &Env,
        values: impl ExactSizeIterator<Item = T>,
    ) -> Result<SmallVec<T>, AllocError> {
        let mut vec = SmallVec::fixed(env, values.len())?;
        vec.extend_within(values);
        Ok(vec)
    }

    /// Adds `values` at the end, as many as there's room for without
    /// growing: an iterator can claim a wrong length, so it can't overrun.
    #[inline]
    pub fn extend_within(&mut self, values: impl Iterator<Item = T>) {
        for value in values.take(self.capacity() - self.raw.len()) {
            // SAFETY: `len` is below the capacity.
            unsafe { self.slot(self.raw.len()).write(value) };
            self.raw.len += 1;
        }
    }

    pub fn capacity(&self) -> usize {
        self.raw.memory.len() / size_of::<T>()
    }

    /// An empty `SmallVec` with room for `capacity` values, 0 or `max`,
    /// rounded up.
    #[inline]
    fn empty(env: &Env, capacity: usize, max: usize) -> Result<SmallVec<T>, AllocError> {
        const { assert!(size_of::<T>() > 0 && align_of::<T>() <= BUFFER_ALIGNMENT_BYTES) };
        Ok(SmallVec { raw: RawVec::new(env, size_of::<T>(), capacity, max)?, values: PhantomData })
    }

    /// Where value `i` goes.
    fn slot(&self, i: usize) -> NonNull<T> {
        // SAFETY: callers only ask for places within the capacity.
        unsafe { self.raw.memory.as_non_null().cast::<T>().add(i) }
    }

    /// Adds `value` at the end, growing if full. Fails, dropping `value`, if
    /// the `SmallVec` holds `max` values or can't grow.
    pub fn push(&mut self, value: T) -> Result<(), AllocError> {
        let slot = self.raw.next_slot(size_of::<T>())?;
        // SAFETY: `next_slot` made room for a value there, and counted it.
        unsafe { slot.cast::<T>().write(value) };
        Ok(())
    }

    /// Adds copies of `values` at the end, in one go, growing if there isn't
    /// room. Fails, adding none, if that would take it past `max` or it
    /// can't grow.
    pub fn extend_from_slice(&mut self, values: &[T]) -> Result<(), AllocError>
    where
        T: Copy,
    {
        if values.len() > self.capacity() - self.raw.len() {
            self.raw.make_room(size_of::<T>(), values.len())?;
        }
        // SAFETY: there is room for `values` from `len`, and they can't
        // overlap the `SmallVec`, which `values` doesn't borrow.
        unsafe {
            self.slot(self.raw.len())
                .copy_from_nonoverlapping(NonNull::from(values).cast(), values.len());
        };
        self.raw.set_len(self.raw.len() + values.len());
        Ok(())
    }

    /// Adds `count` zeros at the end, growing if there isn't room. Fails,
    /// adding none, if that would take it past `max` or it can't grow.
    pub fn extend_zeroed(&mut self, count: usize) -> Result<(), AllocError>
    where
        T: Primitive,
    {
        if count > self.capacity() - self.raw.len() {
            self.raw.make_room(size_of::<T>(), count)?;
        }
        // SAFETY: there is room for `count` values from `len`, and zero bits
        // are a valid `Primitive`.
        unsafe { self.slot(self.raw.len()).write_bytes(0, count) };
        self.raw.set_len(self.raw.len() + count);
        Ok(())
    }

    /// Empties it and frees its memory, so it holds no more than one made by
    /// `new`.
    pub fn free(&mut self)
    where
        T: Primitive,
    {
        // The values need no dropping.
        self.raw.replace(Array::empty(self.raw.memory.env()));
        self.raw.len = 0;
    }

    /// Empties it, keeping its memory.
    pub fn clear(&mut self)
    where
        T: Copy,
    {
        self.raw.len = 0;
    }

    /// Removes the last value, if any.
    pub fn pop(&mut self) -> Option<T> {
        self.raw.len = self.raw.len.checked_sub(1)?;
        // SAFETY: the value at the old last place was written, and is no
        // longer counted, so it's moved out once.
        Some(unsafe { self.slot(self.raw.len()).read() })
    }

    /// Keeps the values `keep` says to, in order, dropping the rest.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let len = self.raw.len();
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
        self.raw.set_len(kept);
    }
}

impl RawVec {
    /// Room for `capacity` values of `item` bytes, and at most `max`, which
    /// is at least `capacity`: both rounded up to a power of two, but for a
    /// `capacity` of 0.
    #[inline(never)]
    fn new(env: &Env, item: usize, capacity: usize, max: usize) -> Result<RawVec, AllocError> {
        let capacity = if capacity == 0 { 0 } else { capacity.next_power_of_two() };
        let Ok(max) = u32::try_from(max.next_power_of_two()) else { return Err(AllocError) };
        // SAFETY: a `SmallVec` reads only the values below `len`, which it
        // has written.
        let memory = unsafe { RawVec::room(env, item, capacity)? };
        Ok(RawVec { memory: ManuallyDrop::new(memory), len: 0, max })
    }

    fn len(&self) -> usize {
        self.len as usize
    }

    /// Frees the memory, without dropping the values in it, for `memory`.
    fn replace(&mut self, memory: Array<u8>) {
        // SAFETY: the old memory isn't used again.
        unsafe { ManuallyDrop::drop(&mut self.memory) };
        self.memory = ManuallyDrop::new(memory);
    }

    /// Counts `len` values, at most `max`.
    fn set_len(&mut self, len: usize) {
        check!(len <= self.max as usize);
        #[expect(clippy::cast_possible_truncation, reason = "at most `max`")]
        let len = len as u32;
        self.len = len;
    }

    /// Unwritten memory for `capacity` values of `item` bytes.
    ///
    /// # Safety
    ///
    /// As for `Array::uninit`: no value may be read before it's written.
    unsafe fn room(env: &Env, item: usize, capacity: usize) -> Result<Array<u8>, AllocError> {
        let bytes = item.checked_mul(capacity).ok_or(AllocError)?;
        // SAFETY: upheld by the caller.
        unsafe { Array::uninit(env, bytes) }
    }

    /// Where the next value of `item` bytes goes, growing if full, and
    /// counted: the caller writes it. Out of line, so each type's `push` is
    /// only that write.
    #[inline(never)]
    fn next_slot(&mut self, item: usize) -> Result<NonNull<u8>, AllocError> {
        if self.len() * item == self.memory.len() {
            self.make_room(item, 1)?;
        }
        // SAFETY: there's room for a value at `len`.
        let slot = unsafe { self.memory.as_non_null().add(self.len() * item) };
        self.len += 1;
        Ok(slot)
    }

    /// Grows so `additional` more values of `item` bytes fit, unless that's
    /// past `max`.
    #[cold]
    #[inline(never)]
    fn make_room(&mut self, item: usize, additional: usize) -> Result<(), AllocError> {
        let needed = self.len().checked_add(additional).ok_or(AllocError)?;
        if needed > self.max as usize {
            return Err(AllocError);
        }
        // Powers of two up to `max` stay powers of two up to `max`.
        let capacity = self.memory.len() / item;
        let doubled = if capacity == 0 { 4 } else { capacity * 2 };
        let capacity = doubled.max(needed.next_power_of_two()).min(self.max as usize);
        // SAFETY: the first `len` values are copied in below, and the rest
        // are read only once written.
        let to = unsafe { RawVec::room(self.memory.env(), item, capacity)? };
        // SAFETY: both hold at least `len` values. They're moved, not
        // dropped, and the old memory is freed when replaced.
        unsafe {
            to.as_non_null().copy_from_nonoverlapping(self.memory.as_non_null(), self.len() * item);
        }
        self.replace(to);
        Ok(())
    }
}

impl<T> Deref for SmallVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: the first `len` values were written by `push`.
        unsafe { core::slice::from_raw_parts(self.slot(0).as_ptr(), self.raw.len()) }
    }
}

impl<T> DerefMut for SmallVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`, and this is the only reference.
        unsafe { core::slice::from_raw_parts_mut(self.slot(0).as_ptr(), self.raw.len()) }
    }
}

impl<T> Drop for SmallVec<T> {
    #[inline(never)]
    fn drop(&mut self) {
        let values = core::ptr::slice_from_raw_parts_mut(self.slot(0).as_ptr(), self.raw.len());
        // SAFETY: the first `len` values were written by `push`, and are
        // dropped once, then their memory is freed.
        unsafe {
            values.drop_in_place();
            ManuallyDrop::drop(&mut self.raw.memory);
        }
    }
}

// `InlineVec`: up to `N` values, held in place rather than allocated.

/// A short list with a known most, such as a node's children or a sort's
/// keys: held in place, so there's no memory to allocate or free.
#[derive(Clone, Copy)]
pub struct InlineVec<T: Copy + Default, const N: usize> {
    values: [T; N],
    len: u32,
}

impl<T: Copy + Default, const N: usize> InlineVec<T, N> {
    pub fn new() -> InlineVec<T, N> {
        InlineVec { values: [T::default(); N], len: 0 }
    }

    /// `values`, failing as running out of memory if there are more than
    /// `N`, as a full `SmallVec` does.
    pub fn from_slice(values: &[T]) -> Result<InlineVec<T, N>, AllocError> {
        let mut list = InlineVec::new();
        let Some(room) = list.values.get_mut(..values.len()) else { return Err(AllocError) };
        room.copy_from_slice(values);
        #[expect(clippy::cast_possible_truncation, reason = "at most `N`")]
        let len = values.len() as u32;
        list.len = len;
        Ok(list)
    }

    /// Adds `value` at the end, failing if it holds `N`.
    pub fn push(&mut self, value: T) -> Result<(), AllocError> {
        let Some(slot) = self.values.get_mut(self.len as usize) else { return Err(AllocError) };
        *slot = value;
        self.len += 1;
        Ok(())
    }

    /// Keeps the values `keep` says to, in order.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let mut kept = 0;
        for i in 0..self.len as usize {
            let value = *at!(self.values, i);
            if keep(&value) {
                *at_mut!(self.values, kept) = value;
                kept += 1;
            }
        }
        #[expect(clippy::cast_possible_truncation, reason = "at most `N`")]
        let kept = kept as u32;
        self.len = kept;
    }
}

impl<T: Copy + Default, const N: usize> Default for InlineVec<T, N> {
    fn default() -> InlineVec<T, N> {
        InlineVec::new()
    }
}

impl<T: Copy + Default, const N: usize> Deref for InlineVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        at!(self.values, ..self.len as usize)
    }
}

impl<T: Copy + Default, const N: usize> DerefMut for InlineVec<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        at_mut!(self.values, ..self.len as usize)
    }
}

// `Blocks`: data that grows, as records of `width` values each, in blocks
// that never move. Growing adds a block rather than copying everything
// into one twice the size, so it never holds the old and the new at once.

/// How many records a block holds, as a power of two, so a record's block
/// is its number shifted. A block of 8-byte values is `width` times 2 MB, so
/// huge pages back it whole.
pub(crate) const BLOCK_SHIFT: u32 = if cfg!(test) { 2 } else { 18 };
const BLOCK_RECORDS: usize = 1 << BLOCK_SHIFT;

/// How many records the first block holds at first. It doubles from there
/// until it's a whole block, so small data stays small.
const FIRST_RECORDS: usize = if cfg!(test) { 1 } else { 1 << 10 };

/// The most blocks there can be.
const BLOCKS_MAX: usize = 1 << 16;

/// Where record `i` of `width` values starts, given where each block starts.
///
/// # Safety
///
/// Record `i`'s block must exist.
#[inline]
pub(crate) unsafe fn address<T>(starts: &[*mut T], width: usize, i: usize) -> *mut T {
    if cfg!(debug_assertions) {
        check!(i >> BLOCK_SHIFT < starts.len());
    }
    // SAFETY: as the caller promises, the record's block exists.
    unsafe { starts.get_unchecked(i >> BLOCK_SHIFT).add((i & (BLOCK_RECORDS - 1)) * width) }
}

pub(crate) struct Blocks<T: Primitive> {
    /// The blocks, as bytes.
    raw: RawBlocks,
    /// How many values a record is.
    width: usize,
    /// Says it holds `T`s.
    holds: PhantomData<T>,
}

/// What `Blocks` keeps that doesn't depend on its values' type, shared by
/// every `Blocks`, as `SmallVec` shares its code.
struct RawBlocks {
    /// The blocks, as bytes: their alignment suits any `Primitive`.
    arrays: SmallVec<Array<u8>>,
    /// Where each of `arrays` starts.
    starts: SmallVec<*mut u8>,
    /// How many bytes a record is.
    record_bytes: usize,
    /// How many records the first block has room for: `BLOCK_RECORDS` once
    /// there's a second.
    first: usize,
    /// How many records there are.
    len: usize,
}

impl RawBlocks {
    fn capacity(&self) -> usize {
        self.first + self.arrays.len().saturating_sub(1) * BLOCK_RECORDS
    }

    /// As `Blocks::reserve`.
    #[inline(never)]
    fn reserve(&mut self, env: &Env, len: usize) -> Result<(), AllocError> {
        while self.capacity() < len {
            // The first block grows, in place of itself, until it's whole.
            let grow = self.first < BLOCK_RECORDS;
            let records = if grow {
                len.next_power_of_two().clamp(FIRST_RECORDS, BLOCK_RECORDS)
            } else {
                BLOCK_RECORDS
            };
            let mut block = Array::zeroed(env, records * self.record_bytes)?;
            if grow && let Some(old) = self.arrays.pop() {
                self.starts.pop();
                let used = self.len * self.record_bytes;
                at_mut!(block, ..used).copy_from_slice(at!(old, ..used));
            }
            if grow {
                self.first = records;
            }
            self.starts.push(block.as_mut_ptr())?;
            self.arrays.push(block)?;
        }
        Ok(())
    }
}

impl<T: Primitive> Blocks<T> {
    pub(crate) fn new(env: &Env, width: usize) -> Result<Blocks<T>, AllocError> {
        check!(width > 0);
        let raw = RawBlocks {
            arrays: SmallVec::new(env, BLOCKS_MAX)?,
            starts: SmallVec::new(env, BLOCKS_MAX)?,
            record_bytes: width * size_of::<T>(),
            first: 0,
            len: 0,
        };
        Ok(Blocks { raw, width, holds: PhantomData })
    }

    pub(crate) fn len(&self) -> usize {
        self.raw.len
    }

    /// Where each block starts: record `i` is `width * (i % BLOCK_RECORDS)`
    /// values into block `i >> BLOCK_SHIFT`.
    pub(crate) fn starts(&self) -> &[*mut T] {
        let starts = &self.raw.starts;
        // SAFETY: pointers to bytes and to `T`s are alike, and every block
        // is aligned for any `Primitive`.
        unsafe { core::slice::from_raw_parts(starts.as_ptr().cast(), starts.len()) }
    }

    /// How many records there's room for.
    pub(crate) fn capacity(&self) -> usize {
        self.raw.capacity()
    }

    /// Makes room for `len` records, zeroed: the first block grows by
    /// doubling until it's whole, then whole blocks are added.
    pub(crate) fn reserve(&mut self, env: &Env, len: usize) -> Result<(), AllocError> {
        self.raw.reserve(env, len)
    }

    /// Adds a record, zeroed, in the room `reserve` made, and returns it.
    pub(crate) fn push(&mut self) -> &mut [T] {
        check!(self.raw.len < self.capacity());
        self.raw.len += 1;
        // SAFETY: there's room for it, checked above, and it's this one's.
        unsafe { core::slice::from_raw_parts_mut(self.record_ptr(self.raw.len - 1), self.width) }
    }

    /// Record `i`'s first value, for loops that can't afford a check.
    ///
    /// # Safety
    ///
    /// `i` must be below the capacity.
    #[inline]
    pub(crate) unsafe fn record_ptr(&self, i: usize) -> *mut T {
        if cfg!(debug_assertions) {
            check!(i < self.capacity());
        }
        // SAFETY: below the capacity, so its block exists.
        unsafe { address(self.starts(), self.width, i) }
    }

    /// Record `i`.
    pub(crate) fn record(&self, i: usize) -> &[T] {
        check!(i < self.raw.len);
        // SAFETY: the record exists, checked above.
        unsafe { core::slice::from_raw_parts(self.record_ptr(i), self.width) }
    }
}

// Bitmaps: bits packed in words, the lowest first, as a column's validity
// is, a byte at a time, and a join's matches are, a `u64` at a time.

/// A word bits are packed in.
pub trait Word: Copy {
    /// How many bits it holds.
    const BITS: usize;

    /// Whether bit `bit` is set.
    fn bit(self, bit: usize) -> bool;

    /// With bit `bit` set too if `value`.
    fn or(self, bit: usize, value: bool) -> Self;

    /// With bit `bit` cleared.
    fn without(self, bit: usize) -> Self;
}

macro_rules! word {
    ($($t:ty),*) => {$(
        impl Word for $t {
            const BITS: usize = <$t>::BITS as usize;

            #[inline]
            fn bit(self, bit: usize) -> bool {
                self >> bit & 1 != 0
            }

            #[inline]
            fn or(self, bit: usize, value: bool) -> Self {
                self | <$t>::from(value) << bit
            }

            #[inline]
            fn without(self, bit: usize) -> Self {
                self & !(1 << bit)
            }
        }
    )*};
}

word!(u8, u64);

/// Bits in words of `W`, borrowed, as a slice is a `Vec`'s.
#[repr(transparent)]
pub struct BitSlice<W: Word = u64>([W]);

impl<W: Word> BitSlice<W> {
    /// `words`, as bits.
    #[inline]
    pub fn of(words: &[W]) -> &BitSlice<W> {
        // SAFETY: `BitSlice` is `[W]`, by `repr(transparent)`.
        unsafe { &*(core::ptr::from_ref(words) as *const BitSlice<W>) }
    }

    /// As `of`, to change them.
    #[inline]
    pub fn of_mut(words: &mut [W]) -> &mut BitSlice<W> {
        // SAFETY: as in `of`, and the borrow is unique.
        unsafe { &mut *(core::ptr::from_mut(words) as *mut BitSlice<W>) }
    }

    /// How many words hold `bits` bits.
    #[inline]
    pub fn words_for(bits: usize) -> usize {
        bits.div_ceil(W::BITS)
    }

    /// Whether bit `at` is set.
    #[inline]
    pub fn get(&self, at: usize) -> bool {
        at!(self.0, at / W::BITS).bit(at % W::BITS)
    }

    /// As `get`, without checking `at` is within the bits.
    ///
    /// # Safety
    ///
    /// `at` must be within the bits.
    #[inline]
    pub unsafe fn get_unchecked(&self, at: usize) -> bool {
        // SAFETY: guaranteed by the caller.
        unsafe { self.0.get_unchecked(at / W::BITS) }.bit(at % W::BITS)
    }

    /// Sets bit `at`.
    #[inline]
    pub fn set(&mut self, at: usize) {
        self.set_if(at, true);
    }

    /// Sets bit `at` if `value`, so a clear bit comes to hold it.
    #[inline]
    pub fn set_if(&mut self, at: usize, value: bool) {
        let word = at_mut!(self.0, at / W::BITS);
        *word = word.or(at % W::BITS, value);
    }

    /// Clears bit `at`.
    #[inline]
    pub fn clear(&mut self, at: usize) {
        let word = at_mut!(self.0, at / W::BITS);
        *word = word.without(at % W::BITS);
    }

    /// The words the bits are in.
    #[inline]
    pub fn words(&self) -> &[W] {
        &self.0
    }
}

impl BitSlice<u8> {
    /// Sets bits `start..start + count`.
    pub fn set_run(&mut self, start: usize, count: usize) {
        let end = start + count;
        let mut bit = start;
        // Up to a whole byte, then whole bytes, then the rest.
        while bit < end && !bit.is_multiple_of(8) {
            self.set(bit);
            bit += 1;
        }
        let whole = (end - bit) / 8;
        at_mut!(self.0, bit / 8..bit / 8 + whole).fill(0xff);
        bit += whole * 8;
        while bit < end {
            self.set(bit);
            bit += 1;
        }
    }
}

/// Bits of its own, a `u64` at a time.
pub struct Bitmap {
    words: Array<u64>,
}

impl Bitmap {
    /// `len` bits, all clear.
    pub fn new(env: &Env, len: usize) -> Result<Bitmap, AllocError> {
        Ok(Bitmap { words: Array::zeroed(env, BitSlice::<u64>::words_for(len))? })
    }

    /// Clears every bit.
    pub fn clear_all(&mut self) {
        self.words.fill(0);
    }
}

impl Deref for Bitmap {
    type Target = BitSlice;

    fn deref(&self) -> &BitSlice {
        BitSlice::of(&self.words)
    }
}

impl DerefMut for Bitmap {
    fn deref_mut(&mut self) -> &mut BitSlice {
        BitSlice::of_mut(&mut self.words)
    }
}

// A bitmap that also says how many bits are set before any bit: its rank,
// which numbers whatever the set bits stand for, in order, without gaps.

/// Bits, with how many are set before each word of them.
pub struct RankBitmap {
    bits: Bitmap,
    before: Array<u32>,
}

impl RankBitmap {
    /// `len` bits, of which those in `set` are set.
    pub fn new(
        env: &Env,
        len: usize,
        set: impl Iterator<Item = usize>,
    ) -> Result<RankBitmap, AllocError> {
        let mut bits = Bitmap::new(env, len)?;
        for at in set {
            bits.set(at);
        }
        let mut before = Array::zeroed(env, bits.words().len())?;
        let mut count = 0;
        for (before, word) in before.iter_mut().zip(bits.words()) {
            *before = count;
            count += word.count_ones();
        }
        Ok(RankBitmap { bits, before })
    }

    /// Whether bit `at` is set.
    #[inline]
    pub fn get(&self, at: usize) -> bool {
        self.bits.get(at)
    }

    /// How many bits are set before bit `at`.
    #[inline]
    pub fn rank(&self, at: usize) -> u32 {
        let word = *at!(self.bits.words(), at / 64);
        *at!(self.before, at / 64) + (word & ((1 << (at % 64)) - 1)).count_ones()
    }
}

#[cfg(test)]
mod tests {
    mod array {
        use crate::containers::*;

        #[test]
        fn holds_exactly_its_values() {
            let env = Env::current();
            let mut array = Array::<u32>::from_fn(env, 5, |i| u32::try_from(i).unwrap()).unwrap();
            array[4] = 9;
            assert_eq!(&*array, &[0, 1, 2, 3, 9]);
            assert!(Array::<u64>::zeroed(env, 0).unwrap().is_empty());
        }
    }

    mod boxed {
        use alloc::rc::Rc;

        use crate::containers::*;

        #[test]
        fn holds_and_drops_its_value() {
            let env = Env::current();
            let live = Rc::new(());
            let mut boxed = Box::new(env, (1, live.clone())).unwrap();
            boxed.0 += 1;
            assert_eq!(boxed.0, 2);
            assert_eq!(Rc::strong_count(&live), 2);
            drop(boxed);
            assert_eq!(Rc::strong_count(&live), 1);
        }

        #[test]
        fn drops_its_value_once_erased() {
            let env = Env::current();
            let live = Rc::new(());
            let erased = Box::new(env, live.clone()).unwrap().erase();
            // SAFETY: it holds an `Rc<()>`.
            assert_eq!(unsafe { erased.as_ptr().cast::<Rc<()>>().as_ref() }, &live);
            assert_eq!(Rc::strong_count(&live), 2);
            drop(erased);
            assert_eq!(Rc::strong_count(&live), 1);
        }
    }

    mod small_vec {
        use alloc::rc::Rc;
        use alloc::vec::Vec as StdVec;
        use core::cell::{Cell, RefCell};
        use core::ptr::NonNull;

        use crate::containers::*;
        use core::alloc::Layout;

        use crate::allocator::{Allocator, Heap, PAGE_BYTES};

        /// Allows `left` allocations, counting those still live. An `Env`
        /// over it takes one, for itself, and holds it while it lives.
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
            // SAFETY: declared before what's allocated from it, so dropped after.
            let env = unsafe { Env::new(allocator, usize::MAX) }.unwrap();
            let mut values = SmallVec::new(&env, 128).unwrap();
            for i in 0..100 {
                assert!(values.push(i).is_ok());
            }
            assert_eq!(values.iter().sum::<u64>(), 4950);
            assert_eq!(live.get(), 2);
            drop(values);
            assert_eq!(live.get(), 1);
        }

        #[test]
        fn drops_its_values() {
            let env = Env::current();
            let value = Rc::new(());
            let mut values = SmallVec::new(env, 16).unwrap();
            for _ in 0..10 {
                assert!(values.push(value.clone()).is_ok());
            }
            assert_eq!(Rc::strong_count(&value), 11);
            drop(values);
            assert_eq!(Rc::strong_count(&value), 1);
        }

        #[test]
        fn gives_the_value_back_if_it_cant_grow() {
            // An empty `SmallVec` allocates nothing; the second is the room for
            // four.
            let left = Rc::new(Cell::new(2));
            let allocator = Limited { left, live: Rc::new(Cell::new(0)) };
            // SAFETY: declared before what's allocated from it, so dropped after.
            let env = unsafe { Env::new(allocator, usize::MAX) }.unwrap();
            let mut values = SmallVec::new(&env, 128).unwrap();
            for i in 0..4 {
                assert!(values.push(i).is_ok());
            }
            assert_eq!(values.push(4), Err(AllocError));
            assert_eq!(*values, [0, 1, 2, 3]);
        }

        #[test]
        fn stops_at_its_max() {
            let left = Rc::new(Cell::new(2));
            let allocator = Limited { left: left.clone(), live: Rc::new(Cell::new(0)) };
            // SAFETY: declared before what's allocated from it, so dropped after.
            let env = unsafe { Env::new(allocator, usize::MAX) }.unwrap();
            let mut values = SmallVec::fixed(&env, 3).unwrap();
            for i in 0..4 {
                assert!(values.push(i).is_ok());
            }
            assert_eq!(values.push(4), Err(AllocError));
            // Made with room for its max, it never grew.
            assert_eq!(left.get(), 0);
            assert_eq!(values.capacity(), 4);
        }

        /// Records the layouts asked for.
        struct Layouts(Rc<RefCell<StdVec<Layout>>>);

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
            let layouts = Rc::new(RefCell::new(StdVec::new()));
            // SAFETY: declared before what's allocated from it, so dropped after.
            let env = unsafe { Env::new(Layouts(layouts.clone()), usize::MAX) }.unwrap();
            drop(SmallVec::<u64>::fixed(&env, 100).unwrap());
            drop(SmallVec::<u64>::fixed(&env, 1 << 15).unwrap());
            // After the `Env`'s own.
            let asked: StdVec<_> =
                layouts.borrow()[1..].iter().map(|l| (l.size(), l.align())).collect();
            assert_eq!(asked, [(128 * 8, 64), (1 << 18, PAGE_BYTES)]);
        }

        #[test]
        fn empty_allocates_nothing() {
            let left = Rc::new(Cell::new(1));
            let allocator = Limited { left, live: Rc::new(Cell::new(0)) };
            // SAFETY: declared before what's allocated from it, so dropped after.
            let env = unsafe { Env::new(allocator, usize::MAX) }.unwrap();
            let values = SmallVec::<u64>::new(&env, 16).unwrap();
            assert!(values.is_empty());
        }

        #[test]
        #[should_panic(expected = "power_of_two")]
        fn max_is_a_power_of_two() {
            let env = Env::current();
            let _ = SmallVec::<u64>::new(env, 100);
        }

        #[test]
        fn frees_and_grows_again() {
            let live = Rc::new(Cell::new(0));
            let allocator = Limited { left: Rc::new(Cell::new(u32::MAX)), live: live.clone() };
            // SAFETY: declared before what's allocated from it, so dropped after.
            let env = unsafe { Env::new(allocator, usize::MAX) }.unwrap();
            let mut values = SmallVec::new(&env, 16).unwrap();
            assert!(values.extend_from_slice(&[1_u64, 2, 3]).is_ok());
            values.free();
            assert!(values.is_empty() && live.get() == 1);
            assert!(values.push(4).is_ok());
            assert_eq!(*values, [4]);
        }

        #[test]
        fn extends_in_one_go() {
            let left = Rc::new(Cell::new(u32::MAX));
            let live = Rc::new(Cell::new(0));
            let allocator = Limited { left: left.clone(), live };
            // SAFETY: declared before what's allocated from it, so dropped after.
            let env = unsafe { Env::new(allocator, usize::MAX) }.unwrap();
            let mut values = SmallVec::new(&env, 16).unwrap();
            assert!(values.extend_from_slice(&[1, 2, 3, 4, 5]).is_ok());
            assert!(values.extend_from_slice(&[6]).is_ok());
            assert_eq!(*values, [1, 2, 3, 4, 5, 6]);
            // Nothing to start, then room grown to fit five.
            assert_eq!(u32::MAX - left.get(), 2);
            assert_eq!(values.extend_from_slice(&[0; 11]), Err(AllocError));
            assert_eq!(values.len(), 6);
        }

        #[test]
        fn retains_and_pops() {
            let env = Env::current();
            let value = Rc::new(());
            let mut values = SmallVec::new(env, 8).unwrap();
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
            let env = Env::current();
            let values = SmallVec::<u64>::fixed(env, 5).unwrap();
            assert_eq!(values.capacity(), 8);
        }
    }

    mod inline_vec {
        use crate::containers::*;

        #[test]
        fn holds_up_to_its_most() {
            let mut list = InlineVec::<u32, 3>::from_slice(&[1, 2]).unwrap();
            assert!(list.push(3).is_ok());
            assert!(list.push(4).is_err());
            list.retain(|&value| value != 2);
            assert_eq!(*list, [1, 3]);
            assert!(InlineVec::<u32, 1>::from_slice(&[1, 2]).is_err());
        }
    }

    mod blocks {
        use crate::containers::*;

        #[test]
        fn keeps_records_across_blocks() {
            let env = Env::current();
            let mut blocks = Blocks::<u64>::new(env, 2).unwrap();
            for i in 0..11 {
                blocks.reserve(env, i + 1).unwrap();
                blocks.push().copy_from_slice(&[i as u64, 0]);
            }
            assert_eq!(blocks.len(), 11);
            assert_eq!(
                (0..11).map(|i| blocks.record(i)[0]).collect::<alloc::vec::Vec<_>>(),
                (0..11).collect::<alloc::vec::Vec<_>>()
            );
            // The first block grew to a whole one; the rest are whole.
            assert_eq!(blocks.starts().len(), 11_usize.div_ceil(BLOCK_RECORDS));
        }
    }

    mod bitmap {
        use crate::containers::*;

        #[test]
        fn sets_and_clears_bits_of_either_width() {
            let mut bytes = [0_u8; 3];
            let bits = BitSlice::of_mut(&mut bytes);
            bits.set_run(3, 15);
            bits.clear(10);
            bits.set_if(20, false);
            bits.set_if(21, true);
            let set: std::vec::Vec<usize> = (0..24).filter(|&at| bits.get(at)).collect();
            let expected: std::vec::Vec<usize> =
                (3..18).filter(|&at| at != 10).chain([21]).collect();
            assert_eq!(set, expected);

            let mut bitmap = Bitmap::new(Env::current(), 130).unwrap();
            bitmap.set(129);
            assert!(bitmap.get(129) && !bitmap.get(128));
            assert_eq!(bitmap.words().len(), 3);
            bitmap.clear_all();
            assert!(!bitmap.get(129));
        }
    }

    mod rank_bitmap {
        use crate::containers::*;

        #[test]
        fn ranks_set_bits_in_order() {
            let set = [3, 64, 65, 200];
            let bitmap = RankBitmap::new(Env::current(), 201, set.into_iter()).unwrap();
            for (rank, &at) in set.iter().enumerate() {
                assert!(bitmap.get(at));
                assert_eq!(bitmap.rank(at) as usize, rank);
            }
            assert!(!bitmap.get(4));
            assert_eq!(bitmap.rank(100), 3);
        }
    }
}
