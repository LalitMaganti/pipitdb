//! `RowBatchPool`: memory for filling row batches' columns, reused batch
//! after batch.

use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::row_batch::BATCH_ROWS_MAX;
use crate::slow_vec::SlowVec;

/// How big each block is: enough for any part of a full batch's column, be
/// it values, offsets or validity.
pub const BLOCK_BYTES: usize = BATCH_ROWS_MAX as usize * 8;

/// The most blocks a pool keeps.
const BLOCKS_MAX: usize = 1 << 16;

/// Hands out blocks of `BLOCK_BYTES` for filling batches' columns. A block
/// comes back when the last column holding it is dropped, to be handed out
/// again, so a run allocates only as many blocks as it holds at once.
///
/// The blocks it keeps count against the allocator's budget until
/// `release`d. Dropping the pool frees those nothing holds; the rest are
/// freed by their last drop.
pub struct RowBatchPool<'a> {
    allocator: &'a dyn Allocator,
    /// Every block not given up, held or not. Made with the first block.
    blocks: Option<SlowVec<NonNull<u8>>>,
}

impl<'a> RowBatchPool<'a> {
    pub fn new(allocator: &'a dyn Allocator) -> RowBatchPool<'a> {
        RowBatchPool { allocator, blocks: None }
    }

    /// A block for one part of a batch's column. It holds whatever was last
    /// written to it, so whoever fills it writes every byte they read.
    pub fn take(&mut self) -> Result<Buffer, AllocError> {
        let blocks = match &mut self.blocks {
            Some(blocks) => blocks,
            None => self.blocks.insert(SlowVec::new(self.allocator, BLOCKS_MAX)?),
        };
        for &block in blocks.iter() {
            // SAFETY: the pool hasn't given up its blocks.
            if let Some(buffer) = unsafe { Buffer::reuse(block) } {
                return Ok(buffer);
            }
        }
        let buffer = Buffer::allocate(self.allocator, BLOCK_BYTES)?;
        // Pooled only once kept track of, so it's freed if it can't be.
        blocks.push(buffer.as_non_null())?;
        buffer.set_pooled();
        Ok(buffer)
    }

    /// Frees the blocks nothing holds.
    pub fn release(&mut self) {
        if let Some(blocks) = &mut self.blocks {
            // SAFETY: the pool hasn't given up its blocks, and forgets those
            // freed.
            blocks.retain(|&block| !unsafe { Buffer::free_if_unused(block) });
        }
    }
}

impl Drop for RowBatchPool<'_> {
    fn drop(&mut self) {
        for &block in self.blocks.iter().flat_map(|blocks| blocks.iter()) {
            // SAFETY: the pool hasn't given up its blocks, and is going.
            unsafe { Buffer::unpool(block) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::{Budget, Heap};

    #[test]
    fn hands_out_blocks_again_once_nothing_holds_them() {
        let budget = Budget::new(&Heap, 1 << 20);
        let mut pool = RowBatchPool::new(&budget);
        let first = pool.take().unwrap();
        let held = first.clone();
        let ptr = first.as_ptr::<u8>();
        drop(first);
        // Still held, so a new block is made.
        let second = pool.take().unwrap();
        assert_ne!(second.as_ptr::<u8>(), ptr);
        drop((held, second));
        let used = budget.used();
        let again = pool.take().unwrap();
        assert_eq!((again.as_ptr::<u8>(), budget.used()), (ptr, used));
    }

    #[test]
    fn releases_blocks_nothing_holds() {
        let budget = Budget::new(&Heap, 1 << 20);
        let mut pool = RowBatchPool::new(&budget);
        let (held, free) = (pool.take().unwrap(), pool.take().unwrap());
        drop(free);
        let used = budget.used();
        pool.release();
        assert!(budget.used() <= used - BLOCK_BYTES);
        // The held block is still the pool's, to hand out once dropped.
        let ptr = held.as_ptr::<u8>();
        drop(held);
        assert_eq!(pool.take().unwrap().as_ptr::<u8>(), ptr);
    }

    #[test]
    fn blocks_held_past_the_pool_are_freed_by_their_last_drop() {
        let budget = Budget::new(&Heap, 1 << 20);
        let mut pool = RowBatchPool::new(&budget);
        let held = pool.take().unwrap();
        drop(pool.take().unwrap());
        drop(pool);
        assert!(budget.used() > 0);
        drop(held);
        assert_eq!(budget.used(), 0);
    }
}
