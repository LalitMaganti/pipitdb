//! `Shared`: a refcounted value from an explicit allocator.
//!
//! Like `Rc`, but the memory comes from an allocator the caller passes in,
//! never the global heap behind its back.

use core::marker::PhantomData;
use core::ops::Deref;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::block::{Block, Owner};

/// A refcounted, immutable value.
///
/// Cloning retains the value; dropping releases it. The last release drops
/// the value and frees its memory through the allocator it came from.
pub struct Shared<T> {
    value: NonNull<T>,
    owner: NonNull<Owner>,
    // `Shared` owns a `T`, for drop checking and variance.
    _value: PhantomData<T>,
}

impl<T> Shared<T> {
    /// Moves `value` into memory from `allocator`.
    ///
    /// # Errors
    ///
    /// Returns `AllocError` if the allocator refuses; `value` is dropped.
    pub fn new<A: Allocator + 'static>(allocator: A, value: T) -> Result<Shared<T>, AllocError> {
        let block = Block::allocate(allocator, value, 0)?;
        // SAFETY: the block was just allocated.
        let value = unsafe { Block::value(block) };
        Ok(Shared {
            value,
            owner: Block::owner(block),
            _value: PhantomData,
        })
    }

    /// Whether this is the only reference to the value.
    #[must_use]
    pub fn is_unique(&self) -> bool {
        self.owner().is_unique()
    }

    fn owner(&self) -> &Owner {
        // SAFETY: the owner outlives every `Shared` that references it.
        unsafe { self.owner.as_ref() }
    }
}

impl<T> Deref for Shared<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: the value lives as long as any reference, and is never
        // mutated while shared.
        unsafe { self.value.as_ref() }
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Shared<T> {
        self.owner().retain();
        Shared {
            value: self.value,
            owner: self.owner,
            _value: PhantomData,
        }
    }
}

impl<T> Drop for Shared<T> {
    fn drop(&mut self) {
        // SAFETY: the owner is live and this `Shared` holds a reference.
        unsafe { Owner::release(self.owner) };
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;
    use core::cell::Cell;

    use super::*;
    use crate::allocator::Heap;
    use crate::allocator::testing::{Counting, Refusing};

    /// Counts how many times it is dropped.
    struct DropCounter {
        drops: Rc<Cell<u32>>,
    }

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    #[test]
    fn new_holds_the_value() {
        let shared = Shared::new(Heap, 42_u64).unwrap();
        assert_eq!(*shared, 42);
        assert!(shared.is_unique());
    }

    #[test]
    fn clone_shares_the_value() {
        let shared = Shared::new(Heap, 42_u64).unwrap();
        let clone = shared.clone();
        assert!(core::ptr::eq(&raw const *shared, &raw const *clone));
        assert!(!shared.is_unique());
    }

    #[test]
    fn last_drop_drops_the_value_once_and_frees() {
        let live = Rc::new(Cell::new(0));
        let drops = Rc::new(Cell::new(0));
        let value = DropCounter {
            drops: drops.clone(),
        };
        let shared = Shared::new(Counting { live: live.clone() }, value).unwrap();
        let clone = shared.clone();
        drop(shared);
        assert_eq!(drops.get(), 0);
        assert_eq!(live.get(), 1);
        drop(clone);
        assert_eq!(drops.get(), 1);
        assert_eq!(live.get(), 0);
    }

    #[test]
    fn refused_allocation_drops_the_value() {
        let drops = Rc::new(Cell::new(0));
        let value = DropCounter {
            drops: drops.clone(),
        };
        assert!(Shared::new(Refusing, value).is_err());
        assert_eq!(drops.get(), 1);
    }
}
