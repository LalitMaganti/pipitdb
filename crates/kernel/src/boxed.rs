//! `Box`: one value, in memory from the allocator it was made with.
//! `ErasedBox`: a `Box` whose type has been forgotten.

use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, Buffer};

/// The value lives in a `Buffer`, which also remembers the allocator, so a
/// `Box` isn't generic over it.
pub struct Box<T> {
    buffer: Buffer,
    value: NonNull<T>,
}

impl<T> Box<T> {
    pub fn new(allocator: &dyn Allocator, value: T) -> Result<Box<T>, AllocError> {
        const { assert!(align_of::<T>() <= BUFFER_ALIGNMENT_BYTES) };
        // SAFETY: the value is written before anything reads it.
        let mut buffer = unsafe { Buffer::allocate_uninit(allocator, size_of::<T>())? };
        let value_ptr = buffer.as_mut_non_null().cast::<T>();
        // SAFETY: the buffer has room for a `T`, aligned for it.
        unsafe { value_ptr.write(value) };
        Ok(Box { buffer, value: value_ptr })
    }

    /// Forgets the value's type, keeping what's needed to drop it.
    pub fn erase(self) -> ErasedBox {
        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is never used or dropped again, so the buffer moves
        // out once.
        let buffer = unsafe { core::ptr::read(&raw const this.buffer) };
        // SAFETY: only called with this value, once.
        let drop = |value: NonNull<()>| unsafe { value.cast::<T>().drop_in_place() };
        ErasedBox { _buffer: buffer, value: this.value.cast(), drop }
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
        // SAFETY: the value was written by `new`, and is dropped once: freeing
        // the buffer doesn't drop it.
        unsafe { self.value.drop_in_place() };
    }
}

/// A value of a type only its maker knows, which it drops.
pub struct ErasedBox {
    // Holds the value, and frees it after `drop` runs.
    _buffer: Buffer,
    value: NonNull<()>,
    drop: unsafe fn(NonNull<()>),
}

impl ErasedBox {
    /// The value, for code that knows its type to cast back.
    pub fn as_ptr(&self) -> NonNull<()> {
        self.value
    }
}

impl Drop for ErasedBox {
    fn drop(&mut self) {
        // SAFETY: `drop` was made for the value's type, and runs once.
        unsafe { (self.drop)(self.value) }
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use super::*;
    use crate::allocator::Heap;

    #[test]
    fn holds_and_drops_its_value() {
        let live = Rc::new(());
        let mut boxed = Box::new(&Heap, (1, live.clone())).unwrap();
        boxed.0 += 1;
        assert_eq!(boxed.0, 2);
        assert_eq!(Rc::strong_count(&live), 2);
        drop(boxed);
        assert_eq!(Rc::strong_count(&live), 1);
    }

    #[test]
    fn drops_its_value_once_erased() {
        let live = Rc::new(());
        let erased = Box::new(&Heap, live.clone()).unwrap().erase();
        // SAFETY: it holds an `Rc<()>`.
        assert_eq!(unsafe { erased.as_ptr().cast::<Rc<()>>().as_ref() }, &live);
        assert_eq!(Rc::strong_count(&live), 2);
        drop(erased);
        assert_eq!(Rc::strong_count(&live), 1);
    }
}
