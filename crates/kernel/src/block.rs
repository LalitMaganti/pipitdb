//! Refcounted blocks from an explicit allocator: what `Buffer` and `Shared`
//! are made of.
//!
//! A block is a header holding a value, then optional trailing bytes, all in
//! one allocation. The header starts with an `Owner`, which holds the
//! refcount and how to free the block, so holders can release a block
//! without knowing its allocator's or value's type.

use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};

/// Alignment of every block, and so of the trailing bytes after its header.
pub(crate) const BLOCK_ALIGNMENT_BYTES: usize = 64;

const _: () = assert!(BLOCK_ALIGNMENT_BYTES.is_power_of_two());

/// The start of every block.
///
/// The count is not atomic: execution is single-threaded for now.
#[repr(C)]
pub(crate) struct Owner {
    references: Cell<u32>,
    release: unsafe fn(NonNull<Owner>),
}

impl Owner {
    /// Whether exactly one reference is held.
    pub(crate) fn is_unique(&self) -> bool {
        self.references.get() == 1
    }

    /// Adds a reference.
    pub(crate) fn retain(&self) {
        let count = self.references.get();
        assert!(count > 0);
        assert!(count < u32::MAX);
        self.references.set(count + 1);
    }

    /// Gives up a reference, freeing the block if it was the last.
    ///
    /// # Safety
    ///
    /// `owner` must start a live block, and the caller must hold a reference
    /// to it, which this consumes.
    pub(crate) unsafe fn release(owner: NonNull<Owner>) {
        // SAFETY: the block is live, per the contract.
        let owner_ref = unsafe { owner.as_ref() };
        let count = owner_ref.references.get();
        assert!(count > 0);
        owner_ref.references.set(count - 1);
        if count == 1 {
            let release = owner_ref.release;
            // SAFETY: this was the last reference, and `release` matches the
            // block's type: both were set together in `Block::allocate`.
            unsafe { release(owner) };
        }
    }
}

/// A block's header: the owner, what is needed to free the block, and the
/// value. Trailing bytes, if any, follow it.
#[repr(C, align(64))]
pub(crate) struct Block<A: Allocator, T> {
    // Must be the first field: holders cast between the two pointer types.
    owner: Owner,
    allocator: A,
    // The whole allocation: this header followed by the trailing bytes.
    layout: Layout,
    value: T,
}

impl<A: Allocator + 'static, T> Block<A, T> {
    /// Allocates a block holding `value`, followed by `trailing_bytes` zeroed
    /// bytes, with one reference.
    ///
    /// # Errors
    ///
    /// Returns `AllocError` if the allocator refuses or the size is too
    /// large to lay out.
    pub(crate) fn allocate(
        allocator: A,
        value: T,
        trailing_bytes: usize,
    ) -> Result<NonNull<Block<A, T>>, AllocError> {
        const { assert!(align_of::<Block<A, T>>() == BLOCK_ALIGNMENT_BYTES) };
        let header_bytes = size_of::<Block<A, T>>();
        let block_bytes = header_bytes.checked_add(trailing_bytes).ok_or(AllocError)?;
        let layout =
            Layout::from_size_align(block_bytes, BLOCK_ALIGNMENT_BYTES).map_err(|_| AllocError)?;

        let pointer = allocator.allocate(layout)?;
        assert!(
            pointer
                .as_ptr()
                .addr()
                .is_multiple_of(BLOCK_ALIGNMENT_BYTES)
        );

        let block = pointer.cast::<Block<A, T>>();
        let owner = Owner {
            references: Cell::new(1),
            release: Self::free,
        };
        // SAFETY: the allocation is valid for `layout`, which starts with
        // room for the header and is aligned for it.
        unsafe {
            block.write(Block {
                owner,
                allocator,
                layout,
                value,
            });
        }
        // SAFETY: the block was just allocated above.
        let trailing = unsafe { Self::trailing(block) };
        // SAFETY: the trailing bytes are the rest of the allocation.
        unsafe { trailing.write_bytes(0, trailing_bytes) };
        Ok(block)
    }

    /// The block's owner.
    pub(crate) fn owner(block: NonNull<Block<A, T>>) -> NonNull<Owner> {
        block.cast()
    }

    /// The block's value.
    ///
    /// # Safety
    ///
    /// `block` must come from `allocate` and still be live.
    pub(crate) unsafe fn value(block: NonNull<Block<A, T>>) -> NonNull<T> {
        // SAFETY: the block is live, so its `value` field is in bounds.
        unsafe { NonNull::new_unchecked(&raw mut (*block.as_ptr()).value) }
    }

    /// The first trailing byte, aligned to `BLOCK_ALIGNMENT_BYTES`.
    ///
    /// # Safety
    ///
    /// `block` must come from `allocate` and still be live.
    pub(crate) unsafe fn trailing(block: NonNull<Block<A, T>>) -> NonNull<u8> {
        // SAFETY: the allocation is at least one header long, so the end of
        // the header is in bounds or one past the end.
        let trailing = unsafe { block.cast::<u8>().add(size_of::<Block<A, T>>()) };
        assert!(
            trailing
                .as_ptr()
                .addr()
                .is_multiple_of(BLOCK_ALIGNMENT_BYTES)
        );
        trailing
    }

    /// Drops the value and frees the block. `Owner::release` calls this
    /// through the fn pointer stored in the owner.
    ///
    /// # Safety
    ///
    /// `owner` must start a block from `allocate::<A, T>` with no references
    /// left.
    unsafe fn free(owner: NonNull<Owner>) {
        let block = owner.cast::<Block<A, T>>();
        // SAFETY: the header is valid, per the contract. Reading moves the
        // value and allocator out, so the memory can be freed.
        let Block {
            owner,
            allocator,
            layout,
            value,
        } = unsafe { block.read() };
        assert!(owner.references.get() == 0);
        assert!(layout.size() >= size_of::<Block<A, T>>());
        drop(value);
        // SAFETY: the block came from this allocator with this layout.
        unsafe { allocator.deallocate(block.cast(), layout) };
    }
}
