//! `Context`: what a run's steps share, passed to each as it runs, so what
//! they share can grow without changing every step's functions.

use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::query_allocators::QueryAllocators;
use crate::selection::Selection;
use crate::slow_vec::SlowVec;

/// The most scratch selections a run can have.
pub const SCRATCH_SELECTIONS_MAX: usize = 1 << 6;

pub struct Context<'a> {
    /// What states and other metadata allocate from.
    allocator: &'a dyn Allocator,
    /// What column buffers of each kind allocate from.
    values: &'a dyn Allocator,
    indices: &'a dyn Allocator,
    small: &'a dyn Allocator,
    /// Made when first reserved: most runs need none.
    selections: Option<SlowVec<Selection>>,
}

impl<'a> Context<'a> {
    /// A context whose states and column buffers all allocate from
    /// `allocator`, for steps run outside a pipeline, such as in tests.
    pub fn new(allocator: &'a dyn Allocator) -> Context<'a> {
        let (values, indices, small) = (allocator, allocator, allocator);
        Context { allocator, values, indices, small, selections: None }
    }

    /// A query's context, which allocates from `query`'s allocators.
    pub fn for_query(query: &'a QueryAllocators<'a>) -> Context<'a> {
        Context {
            allocator: query.metadata,
            values: &query.values,
            indices: &query.indices,
            small: &query.small,
            selections: None,
        }
    }

    /// What the run's memory comes from, for states to allocate with.
    pub fn allocator(&self) -> &'a dyn Allocator {
        self.allocator
    }

    /// Memory for a batch's column of 8-byte values, `size_bytes` of them,
    /// at most a batch's. Steps get all their columns' memory from these
    /// functions, by kind, so how it's found can change in one place. It
    /// isn't zeroed: whoever fills it writes every byte that's read.
    pub fn values_buffer(&mut self, size_bytes: usize) -> Result<Buffer, AllocError> {
        buffer(self.values, size_bytes)
    }

    /// As `values_buffer`, for 4-byte values, such as dictionary indices, or
    /// string offsets, of which there's one more than rows.
    pub fn indices_buffer(&mut self, size_bytes: usize) -> Result<Buffer, AllocError> {
        buffer(self.indices, size_bytes)
    }

    /// As `values_buffer`, for small things a column needs: its header, its
    /// validity bitmap, a lazy column's handle.
    pub fn small_buffer(&mut self, size_bytes: usize) -> Result<Buffer, AllocError> {
        buffer(self.small, size_bytes)
    }

    /// As `values_buffer`, for column data of no fixed size, such as
    /// strings' bytes.
    pub fn bytes_buffer(&mut self, size_bytes: usize) -> Result<Buffer, AllocError> {
        buffer(self.allocator, size_bytes)
    }

    /// Makes sure there are at least `count` scratch selections. Steps call
    /// this when making their states, so no batch allocates. Steps never
    /// run at once, so they share them: there are as many as the most any
    /// step reserved.
    pub fn reserve_selections(&mut self, count: usize) -> Result<(), AllocError> {
        if count > SCRATCH_SELECTIONS_MAX {
            return Err(AllocError);
        }
        let selections = match &mut self.selections {
            Some(selections) => selections,
            None => self.selections.insert(SlowVec::new(self.allocator, SCRATCH_SELECTIONS_MAX)?),
        };
        while selections.len() < count {
            selections.push(Selection::all(0))?;
        }
        Ok(())
    }

    /// The scratch selections, for a step to use while it runs. They hold
    /// whatever the last step left in them.
    pub fn selections(&mut self) -> &mut [Selection] {
        self.selections.as_deref_mut().unwrap_or_default()
    }
}

/// `size_bytes` bytes from `allocator`, not zeroed.
fn buffer(allocator: &dyn Allocator, size_bytes: usize) -> Result<Buffer, AllocError> {
    // SAFETY: whoever fills it writes every byte that's read.
    unsafe { Buffer::allocate_uninit(allocator, size_bytes) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;

    #[test]
    fn shares_scratch_between_steps() {
        let mut context = Context::new(&Heap);
        assert!(context.selections().is_empty());
        // As many as the most reserved, not the sum.
        assert!(context.reserve_selections(2).is_ok() && context.reserve_selections(1).is_ok());
        assert_eq!(context.selections().len(), 2);
        assert_eq!(context.reserve_selections(SCRATCH_SELECTIONS_MAX + 1), Err(AllocError));
    }
}
