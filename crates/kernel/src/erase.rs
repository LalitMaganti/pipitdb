//! A `Dyn*` holds a value of any type behind an untyped pointer, and a table
//! of functions that know that type: one table for each type, whose
//! functions take the value, and its state if it has one, as untyped
//! pointers. These are helpers for those functions.
//!
//! TODO: make the tables `#[repr(C)]`, with `extern "C"` functions over C
//! types, so code in other languages can implement these interfaces.

#![allow(unsafe_code)]

use core::ptr::NonNull;

/// `value` as the `T` it is.
///
/// # Safety
///
/// `value` must point to a live `T`, for `'a`.
pub(crate) unsafe fn value_of<'a, T>(value: NonNull<()>) -> &'a T {
    // SAFETY: upheld by the caller.
    unsafe { value.cast().as_ref() }
}

/// `value` as the `T` it is, to change.
///
/// # Safety
///
/// `value` must point to a live `T`, not otherwise borrowed, for `'a`.
pub(crate) unsafe fn value_mut_of<'a, T>(value: NonNull<()>) -> &'a mut T {
    // SAFETY: upheld by the caller.
    unsafe { value.cast().as_mut() }
}

/// `state` as the `S` it is.
///
/// # Safety
///
/// `state` must point to a live `S`, not otherwise borrowed, for `'a`.
pub(crate) unsafe fn state_of<'a, S>(state: NonNull<u8>) -> &'a mut S {
    // SAFETY: upheld by the caller.
    unsafe { state.cast().as_mut() }
}

/// Writes `value` to `state`.
///
/// # Safety
///
/// `state` must be valid for writes of an `S`.
pub(crate) unsafe fn write_state<S>(state: NonNull<u8>, value: S) {
    // SAFETY: upheld by the caller.
    unsafe { state.cast().write(value) }
}

/// Drops the `S` at `state`.
///
/// # Safety
///
/// `state` must point to a live `S`, which isn't used again.
pub(crate) unsafe fn drop_state<S>(state: NonNull<u8>) {
    // SAFETY: upheld by the caller.
    unsafe { state.cast::<S>().drop_in_place() }
}
