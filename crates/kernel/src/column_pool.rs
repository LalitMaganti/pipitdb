//! `ColumnPool`: column buffers kept to hand out again.

use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::slow_vec::SlowVec;

/// The most buffers a pool keeps track of.
const BUFFERS_MAX: usize = 1 << 6;

/// Hands out buffers for filling columns, and takes each back when the last
/// column holding it is dropped, to hand out again for the same size. Batch
/// after batch asks for the same sizes, so filling them neither allocates
/// nor zeroes: a buffer handed out again holds what it last held.
///
/// The buffers it keeps count against the allocator's budget. Dropping the
/// pool frees those nothing holds; the rest are freed by their last drop.
pub(crate) struct ColumnPool {
    /// Every buffer not given up, held or not. Made with the first buffer.
    buffers: Option<SlowVec<NonNull<u8>>>,
}

impl ColumnPool {
    pub(crate) fn new() -> ColumnPool {
        ColumnPool { buffers: None }
    }

    /// A buffer of `size_bytes`, one the pool has if one's free, or else a
    /// new one, zeroed, which it keeps track of.
    pub(crate) fn take(
        &mut self,
        allocator: &dyn Allocator,
        size_bytes: usize,
    ) -> Result<Buffer, AllocError> {
        let buffers = match &mut self.buffers {
            Some(buffers) => buffers,
            None => self.buffers.insert(SlowVec::new(allocator, BUFFERS_MAX)?),
        };
        for &data in buffers.iter() {
            // SAFETY: the pool hasn't given up its buffers.
            if let Some(buffer) = unsafe { Buffer::reuse(data, size_bytes) } {
                return Ok(buffer);
            }
        }
        let buffer = Buffer::allocate(allocator, size_bytes)?;
        if buffers.len() == BUFFERS_MAX {
            // Full: frees those nothing holds, such as of sizes no longer
            // asked for, to make room.
            // SAFETY: the pool hasn't given up its buffers, and forgets those
            // freed.
            buffers.retain(|&data| !unsafe { Buffer::free_if_unused(data) });
        }
        // Pooled only once kept track of, so it's freed if it can't be.
        if buffers.push(buffer.as_non_null()).is_ok() {
            buffer.set_pooled();
        }
        Ok(buffer)
    }
}

impl Drop for ColumnPool {
    fn drop(&mut self) {
        for &data in self.buffers.iter().flat_map(|buffers| buffers.iter()) {
            // SAFETY: the pool hasn't given up its buffers, and is going.
            unsafe { Buffer::unpool(data) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::{Budget, Heap};

    #[test]
    fn hands_out_buffers_again_once_nothing_holds_them() {
        let budget = Budget::new(&Heap, 1 << 20);
        let mut pool = ColumnPool::new();
        let first = pool.take(&budget, 1000).unwrap();
        let held = first.clone();
        let ptr = first.as_ptr::<u8>();
        drop(first);
        // Still held, so a new one is made; and another for another size.
        let second = pool.take(&budget, 1000).unwrap();
        assert_ne!(second.as_ptr::<u8>(), ptr);
        let other = pool.take(&budget, 8).unwrap();
        drop((held, second, other));
        let used = budget.used();
        let again = pool.take(&budget, 1000).unwrap();
        assert_eq!((again.as_ptr::<u8>(), again.size_bytes(), budget.used()), (ptr, 1000, used));
    }

    #[test]
    fn makes_room_by_freeing_buffers_nothing_holds() {
        let budget = Budget::new(&Heap, 1 << 20);
        let mut pool = ColumnPool::new();
        for size in 1..=BUFFERS_MAX {
            drop(pool.take(&budget, size).unwrap());
        }
        let used = budget.used();
        drop(pool.take(&budget, 4096).unwrap());
        assert!(budget.used() < used);
    }

    #[test]
    fn buffers_held_past_the_pool_are_freed_by_their_last_drop() {
        let budget = Budget::new(&Heap, 1 << 20);
        let mut pool = ColumnPool::new();
        let held = pool.take(&budget, 64).unwrap();
        drop(pool.take(&budget, 64).unwrap());
        drop(pool);
        assert!(budget.used() > 0);
        drop(held);
        assert_eq!(budget.used(), 0);
    }
}
