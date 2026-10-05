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
/// It keeps track of at most `BUFFERS_MAX`, giving them up in turn to make
/// room for new sizes. Those it keeps count against the allocator's budget.
/// A buffer given up, by that or by dropping the pool, is freed now if
/// nothing holds it, and else by its last drop.
pub(crate) struct ColumnPool {
    /// The buffers kept track of, held or not. Made with the first buffer.
    buffers: Option<SlowVec<NonNull<u8>>>,
    /// Which buffer to give up next when there's no room.
    next: usize,
}

impl ColumnPool {
    pub(crate) fn new() -> ColumnPool {
        ColumnPool { buffers: None, next: 0 }
    }

    /// A buffer of `size_bytes`: one the pool has, if one's free, or else a
    /// new one, zeroed.
    pub(crate) fn take(
        &mut self,
        allocator: &dyn Allocator,
        size_bytes: usize,
    ) -> Result<Buffer, AllocError> {
        let buffers = match &mut self.buffers {
            Some(buffers) => buffers,
            None => self.buffers.insert(SlowVec::fixed(allocator, BUFFERS_MAX)?),
        };
        for &data in buffers.iter() {
            // SAFETY: the pool hasn't given up its buffers.
            if let Some(buffer) = unsafe { Buffer::reuse(data, size_bytes) } {
                return Ok(buffer);
            }
        }
        let buffer = Buffer::allocate_pooled(allocator, size_bytes)?;
        let data = buffer.as_non_null();
        if buffers.len() < BUFFERS_MAX {
            let Ok(()) = buffers.push(data) else { crate::check::check_failed(line!()) };
        } else {
            let slot = at_mut!(buffers, self.next);
            // SAFETY: the pool hasn't given up its buffers, and forgets this
            // one.
            unsafe { Buffer::unpool(*slot) };
            *slot = data;
            self.next = (self.next + 1) % BUFFERS_MAX;
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
    fn makes_room_by_giving_buffers_up_in_turn() {
        let budget = Budget::new(&Heap, 1 << 20);
        let mut pool = ColumnPool::new();
        // The first is held when given up, so it's freed by its last drop.
        let held = pool.take(&budget, 1).unwrap();
        for size in 2..=BUFFERS_MAX {
            drop(pool.take(&budget, size).unwrap());
        }
        let used = budget.used();
        drop(pool.take(&budget, 4096).unwrap());
        assert_eq!(budget.used(), used + 64 + 4096);
        drop(held);
        assert_eq!(budget.used(), used + 4096 - 1);
        // The second, free when given up, is freed then.
        drop(pool.take(&budget, 8192).unwrap());
        assert_eq!(budget.used(), used + 4096 - 1 + 8192 - 2);
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
