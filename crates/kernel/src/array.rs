//! `Array`: a fixed number of values in one allocation of exactly their
//! size, for data that must be contiguous, such as what a sort splits into
//! slices. It never grows: data that grows goes in blocks that never move.

use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator, layout};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, Primitive};
use crate::env::Env;

/// `len` values, zeroed when made.
pub(crate) struct Array<T: Primitive> {
    values: NonNull<T>,
    len: usize,
    /// What the values came from.
    env: NonNull<Env>,
    owns: PhantomData<T>,
}

impl<T: Primitive> Array<T> {
    /// `len` zeroes: fresh memory from the system is zero already, so a big
    /// one costs nothing to make.
    pub(crate) fn zeroed(env: &Env, len: usize) -> Result<Array<T>, AllocError> {
        Array::allocate(env, len, true)
    }

    /// `len` values, value `i` being `value(i)`, each written once.
    pub(crate) fn from_fn(
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
    /// `write(start, out)`, which writes every one of `out`, values `start`
    /// on.
    pub(crate) fn from_chunks(
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
    pub(crate) fn empty(env: &Env) -> Array<T> {
        let values = NonNull::<Aligned>::dangling().cast();
        Array { values, len: 0, env: NonNull::from(env), owns: PhantomData }
    }

    /// Room for `len` values, not written.
    ///
    /// # Safety
    ///
    /// No value may be read before it's written.
    pub(crate) unsafe fn uninit(env: &Env, len: usize) -> Result<Array<T>, AllocError> {
        Array::allocate(env, len, false)
    }

    /// The `Env` the values came from.
    pub(crate) fn env(&self) -> &Env {
        // SAFETY: an `Env` outlives what's allocated from it.
        unsafe { self.env.as_ref() }
    }

    /// Where the values start, for writing values that aren't `T`, such as
    /// a `SmallVec`'s.
    pub(crate) fn as_non_null(&self) -> NonNull<T> {
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
/// every `Array` shares it.
#[inline(never)]
fn allocate_bytes(env: &Env, bytes: usize, zeroed: bool) -> Result<NonNull<u8>, AllocError> {
    let layout = layout(bytes)?;
    if zeroed { env.allocate_zeroed(layout) } else { env.allocate(layout) }
}

/// Frees `bytes` at `values`.
///
/// # Safety
///
/// They must be from `allocate_bytes`, with `env`.
#[inline(never)]
unsafe fn free_bytes(env: &Env, values: NonNull<u8>, bytes: usize) {
    let Ok(layout) = layout(bytes) else { failed!() };
    // SAFETY: as the caller promises.
    unsafe { env.deallocate(values, layout) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_exactly_its_values() {
        let env = Env::current();
        let mut array = Array::<u32>::from_fn(env, 5, |i| u32::try_from(i).unwrap()).unwrap();
        array[4] = 9;
        assert_eq!(&*array, &[0, 1, 2, 3, 9]);
        assert!(Array::<u64>::zeroed(env, 0).unwrap().is_empty());
    }
}
