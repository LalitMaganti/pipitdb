//! `Box`: one value, in memory from the `Env` it was made with.
//! `ErasedBox`: a `Box` whose type has been forgotten.

use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::allocator::AllocError;
use crate::array::{allocate_bytes, free_bytes};
use crate::buffer::BUFFER_ALIGNMENT_BYTES;
use crate::env::Env;

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

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use super::*;

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
