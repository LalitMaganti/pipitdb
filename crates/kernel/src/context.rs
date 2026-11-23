//! `Context`: what a run's steps share, passed to each as it runs, so what
//! they share can grow without changing every step's functions.

#![allow(unsafe_code)]

use crate::allocator::AllocError;
use crate::buffer::Buffer;
use crate::containers::SmallVec;
use crate::env::{Env, MemoryKind};
use crate::function::DynFunction;
use crate::selection::Selection;
use crate::statement::Statement;
use crate::string_pool::StringPool;

/// The most scratch selections a run can have.
pub const SCRATCH_SELECTIONS_MAX: usize = 1 << 10;

/// The most cutoffs a run's `TopK`s can publish: more are not published.
pub const CUTOFFS_MAX: u32 = 1 << 6;

pub struct Context<'a> {
    /// What states, other metadata and column buffers allocate from.
    env: &'a Env,
    /// Made when first reserved: most runs need none.
    selections: Option<SmallVec<Selection>>,
    /// Made when first used: most runs intern no strings.
    string_pool: Option<StringPool>,
    /// The strings the statement's conditions compare with, if it has any.
    literals: Option<&'a StringPool>,
    /// The functions the statement's expressions call, by id.
    functions: &'a [DynFunction],
    /// By slot, the first key of the last row a `TopK` keeps, once it has
    /// published one, so its scan can skip rows that sort after it. Made
    /// when first reserved: most runs publish none.
    cutoffs: Option<SmallVec<Option<i64>>>,
}

impl<'a> Context<'a> {
    /// A context allocating from `env`, of `statement`, if any.
    pub fn new(env: &'a Env, statement: Option<&'a Statement>) -> Context<'a> {
        Context {
            env,
            selections: None,
            string_pool: None,
            literals: statement.and_then(Statement::literals),
            functions: statement.map_or(&[], Statement::functions),
            cutoffs: None,
        }
    }

    /// The strings the statement's conditions compare with, if it has any.
    pub fn literals(&self) -> Option<&'a StringPool> {
        self.literals
    }

    /// The function the statement's expressions call as `id`.
    pub fn function(&self, id: u32) -> &'a DynFunction {
        at!(self.functions, id as usize)
    }

    /// What the run's memory comes from, for states to allocate with.
    pub fn env(&self) -> &'a Env {
        self.env
    }

    /// Memory for a batch's column of 8-byte values, `size_bytes` of them,
    /// at most a batch's. Steps get all their columns' memory from these
    /// functions, by kind, so how it's found can change in one place. It
    /// isn't zeroed: whoever fills it writes every byte that's read.
    pub fn values_buffer(&mut self, size_bytes: usize) -> Result<Buffer, AllocError> {
        buffer(self.env, MemoryKind::Values, size_bytes)
    }

    /// As `values_buffer`, for 4-byte values, such as dictionary indices, or
    /// string offsets, of which there's one more than rows.
    pub fn indices_buffer(&mut self, size_bytes: usize) -> Result<Buffer, AllocError> {
        buffer(self.env, MemoryKind::Indices, size_bytes)
    }

    /// As `values_buffer`, for small things a column needs: its header, its
    /// validity bitmap, a lazy column's handle.
    pub fn small_buffer(&mut self, size_bytes: usize) -> Result<Buffer, AllocError> {
        buffer(self.env, MemoryKind::Small, size_bytes)
    }

    /// As `values_buffer`, for column data of no fixed size, such as
    /// strings' bytes.
    pub fn bytes_buffer(&mut self, size_bytes: usize) -> Result<Buffer, AllocError> {
        buffer(self.env, MemoryKind::Base, size_bytes)
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
            None => self.selections.insert(SmallVec::new(self.env, SCRATCH_SELECTIONS_MAX)?),
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

    /// Makes sure there is a cutoff in `slot`, below `CUTOFFS_MAX`, for a
    /// `TopK` to publish in. `TopK`s call this when making their states, so
    /// publishing doesn't allocate.
    pub fn reserve_cutoff(&mut self, slot: u32) -> Result<(), AllocError> {
        check!(slot < CUTOFFS_MAX);
        let cutoffs = match &mut self.cutoffs {
            Some(cutoffs) => cutoffs,
            None => self.cutoffs.insert(SmallVec::new(self.env, CUTOFFS_MAX as usize)?),
        };
        while cutoffs.len() <= slot as usize {
            cutoffs.push(None)?;
        }
        Ok(())
    }

    /// The cutoff a `TopK` published in `slot`, if any.
    pub fn cutoff(&self, slot: u32) -> Option<i64> {
        self.cutoffs.as_deref()?.get(slot as usize).copied().flatten()
    }

    /// Publishes `cutoff` in `slot`, which must be reserved.
    pub fn set_cutoff(&mut self, slot: u32, cutoff: Option<i64>) {
        let cutoffs = self.cutoffs.as_deref_mut().unwrap_or_default();
        *at_mut!(cutoffs, slot as usize) = cutoff;
    }

    /// The run's string pool, so the ids of strings interned anywhere in it
    /// can be compared.
    pub fn string_pool(&mut self) -> Result<&mut StringPool, AllocError> {
        if self.string_pool.is_none() {
            self.string_pool = Some(StringPool::new(self.env)?);
        }
        let Some(pool) = &mut self.string_pool else { failed!() };
        Ok(pool)
    }
}

/// `size_bytes` bytes from `env`, not zeroed.
fn buffer(env: &Env, kind: MemoryKind, size_bytes: usize) -> Result<Buffer, AllocError> {
    // SAFETY: whoever fills it writes every byte that's read.
    unsafe { Buffer::allocate_uninit(env, kind, size_bytes) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_scratch_between_steps() {
        let env = Env::current();
        let mut context = Context::new(env, None);
        assert!(context.selections().is_empty());
        // As many as the most reserved, not the sum.
        assert!(context.reserve_selections(2).is_ok() && context.reserve_selections(1).is_ok());
        assert_eq!(context.selections().len(), 2);
        assert_eq!(context.reserve_selections(SCRATCH_SELECTIONS_MAX + 1), Err(AllocError));
    }
}
