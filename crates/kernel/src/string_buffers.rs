//! `StringBuffers`: the buffers of strings an `Env`'s columns point into,
//! by id, so a string column's words can point into many of them. A word's
//! high half is its buffer's id, or 0 for its column's own bytes.

#![allow(unsafe_code)]

use core::cell::{Cell, UnsafeCell};
use core::ptr::NonNull;

use crate::allocator::AllocError;
use crate::buffer::Buffer;
use crate::containers::SmallVec;
use crate::env::Env;

/// The most buffers registered in an `Env`'s life: ids aren't reused.
const BUFFERS_MAX: usize = 1 << 31;

pub(crate) struct StringBuffers {
    /// Each buffer registered, at its id less one, made with the first.
    /// Only its own methods borrow it, and none while another runs.
    entries: UnsafeCell<Option<SmallVec<Entry>>>,
}

/// A buffer registered, its bytes, found in one step, and how many hold
/// words into it: freed when none does.
struct Entry {
    buffer: Option<Buffer>,
    bytes: NonNull<[u8]>,
    holders: Cell<u32>,
}

impl StringBuffers {
    pub(crate) const fn new() -> StringBuffers {
        StringBuffers { entries: UnsafeCell::new(None) }
    }

    /// Registers `buffer`, with one holder, returning its id.
    pub(crate) fn register(&self, env: &Env, buffer: Buffer) -> Result<u32, AllocError> {
        // SAFETY: no other borrow of the entries is live, as each method's
        // ends before it returns.
        let entries = unsafe { &mut *self.entries.get() };
        let entries = match entries {
            Some(entries) => entries,
            None => entries.insert(SmallVec::new(env, BUFFERS_MAX)?),
        };
        let bytes = NonNull::from(buffer.as_slice::<u8>());
        entries.push(Entry { buffer: Some(buffer), bytes, holders: Cell::new(1) })?;
        let Ok(id) = u32::try_from(entries.len()) else { failed!() };
        Ok(id)
    }

    /// Adds a holder of buffer `id`, which must have one.
    pub(crate) fn retain(&self, id: u32) {
        let entry = self.entry(id);
        check!(entry.holders.get() > 0);
        entry.holders.set(entry.holders.get() + 1);
    }

    /// Takes away a holder of buffer `id`, freeing it with the last.
    pub(crate) fn release(&self, id: u32) {
        let holders = self.entry(id).holders.get() - 1;
        self.entry(id).holders.set(holders);
        if holders == 0 {
            // SAFETY: as in `register`. No slice of the buffer is left, as
            // whoever holds one holds the buffer.
            let entries = unsafe { &mut *self.entries.get() };
            let Some(entries) = entries else { failed!() };
            at_mut!(entries, id as usize - 1).buffer = None;
        }
    }

    /// The bytes of buffer `id`, which must have a holder for as long as
    /// they're borrowed.
    #[inline]
    pub(crate) fn bytes(&self, id: u32) -> &[u8] {
        // SAFETY: the buffer is held, so it's where the entry says.
        unsafe { self.entry(id).bytes.as_ref() }
    }

    /// Frees what holds the entries if every buffer is freed, before its
    /// `Env` is, returning whether it did.
    pub(crate) fn free(&self) -> bool {
        // SAFETY: as in `register`.
        let entries = unsafe { &mut *self.entries.get() };
        let freed = entries.iter().flat_map(|entries| entries.iter()).all(|e| e.buffer.is_none());
        if freed {
            *entries = None;
        }
        freed
    }

    #[inline]
    fn entry(&self, id: u32) -> &Entry {
        // SAFETY: as in `register`; an entry is never removed while its
        // `Env` lives, so this one stays where it is but for `register`
        // growing the list, which no caller does while it holds this.
        let entries = unsafe { &*self.entries.get() };
        let entries = entries.as_deref().unwrap_or_default();
        // Id 0, which names none, wraps past every entry.
        at!(entries, (id as usize).wrapping_sub(1))
    }
}
