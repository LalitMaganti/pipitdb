//! `SlabAllocator`, which hands out slots of one size, and the sizes of a
//! query's slots for its batches.

use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, HEADER_BYTES};
use crate::row_batch::BATCH_ROWS_MAX;

/// Hands out slots of one size, carved from slabs taken from the
/// allocator under it, and takes each back to hand out again, the last
/// freed first, while it's likely still in cache. Slabs are given back
/// when it's dropped, which must be after every slot is freed.
///
/// Any layout up to its slot size gets a whole slot. It's one type for
/// every size, so its code isn't copied for each.
pub(crate) struct SlabAllocator<'a> {
    /// What slabs are taken from.
    allocator: &'a dyn Allocator,
    /// How many bytes each slot has.
    slot_bytes: usize,
    /// Slots freed, each holding the next.
    free: Cell<Option<NonNull<u8>>>,
    /// Where the next slot is carved from in the newest slab, and how
    /// many bytes are left after it.
    next: Cell<Option<(NonNull<u8>, usize)>>,
    /// The newest slab, which holds the one before it, and so on.
    slabs: Cell<Option<NonNull<u8>>>,
    /// How many slabs were taken: each is twice the one before, up to
    /// `SLAB_BYTES_MAX`, so a short run takes little and a long one few.
    slab_count: Cell<u8>,
    /// Slots handed out and not freed.
    live: Cell<usize>,
}

/// A first slab's size, at least, and the most a slab takes, at least a
/// slot's size more.
const SLAB_BYTES_MIN: usize = 4 << 10;
const SLAB_BYTES_MAX: usize = 1 << 20;

/// What a slab's first bytes hold: the slab taken before it, and its size.
type Link = (Option<NonNull<u8>>, usize);

impl<'a> SlabAllocator<'a> {
    pub(crate) fn new(allocator: &'a dyn Allocator, slot_bytes: usize) -> SlabAllocator<'a> {
        check!(slot_bytes.is_multiple_of(BUFFER_ALIGNMENT_BYTES));
        SlabAllocator {
            allocator,
            slot_bytes,
            free: Cell::new(None),
            next: Cell::new(None),
            slabs: Cell::new(None),
            slab_count: Cell::new(0),
            live: Cell::new(0),
        }
    }

    /// Whether no slot is handed out, and so the bytes it holds are all
    /// its slabs', `slab_bytes`.
    pub(crate) fn is_idle(&self) -> bool {
        self.live.get() == 0
    }

    /// The bytes of the slabs it took.
    pub(crate) fn slab_bytes(&self) -> usize {
        let (mut bytes, mut slab) = (0, self.slabs.get());
        while let Some(at) = slab {
            // SAFETY: a slab's first bytes hold its link.
            let (before, size) = unsafe { at.cast::<Link>().read() };
            (bytes, slab) = (bytes + size, before);
        }
        bytes
    }

    /// The layout of the `slab_count`th slab: four slots, and the link,
    /// at first, then twice the one before.
    fn slab_layout(&self) -> Layout {
        let first = (4 * self.slot_bytes + BUFFER_ALIGNMENT_BYTES).max(SLAB_BYTES_MIN);
        let max = SLAB_BYTES_MAX.max(first);
        let bytes = (first << self.slab_count.get().min(16)).min(max);
        let Ok(layout) = Layout::from_size_align(bytes, BUFFER_ALIGNMENT_BYTES) else { failed!() };
        layout
    }

    /// Takes a new slab, and returns where its first slot is and the
    /// bytes after it.
    #[cold]
    fn take_slab(&self) -> Result<(NonNull<u8>, usize), AllocError> {
        let layout = self.slab_layout();
        let slab = self.allocator.allocate(layout)?;
        self.slab_count.set(self.slab_count.get().saturating_add(1));
        // SAFETY: a slab's first bytes hold its link.
        unsafe { slab.cast::<Link>().write((self.slabs.get(), layout.size())) };
        self.slabs.set(Some(slab));
        // SAFETY: slots start after the link, within the slab.
        Ok((unsafe { slab.add(BUFFER_ALIGNMENT_BYTES) }, layout.size() - BUFFER_ALIGNMENT_BYTES))
    }
}

// SAFETY: slots are carved from slabs that live until it's dropped, which
// checks none is still handed out.
unsafe impl Allocator for SlabAllocator<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        // Asking for more than a slot is a caller's mistake, not running out
        // of memory, which `AllocError` would say, and spilling can't fix.
        check!(layout.size() <= self.slot_bytes && layout.align() <= BUFFER_ALIGNMENT_BYTES);
        let slot = if let Some(slot) = self.free.get() {
            // SAFETY: a freed slot holds the next one.
            self.free.set(unsafe { slot.cast::<Option<NonNull<u8>>>().read() });
            slot
        } else {
            let (next, left) = match self.next.get() {
                Some((next, left)) if left >= self.slot_bytes => (next, left),
                _ => self.take_slab()?,
            };
            // SAFETY: a slot's bytes are left after `next`, in its slab.
            let after = unsafe { next.add(self.slot_bytes) };
            self.next.set(Some((after, left - self.slot_bytes)));
            next
        };
        self.live.set(self.live.get() + 1);
        Ok(slot)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _: Layout) {
        // SAFETY: the slot is freed, so it can hold the next free one.
        unsafe { ptr.cast::<Option<NonNull<u8>>>().write(self.free.get()) };
        self.free.set(Some(ptr));
        self.live.set(self.live.get() - 1);
    }

    unsafe fn discard(&self, ptr: NonNull<u8>, len: usize) {
        // SAFETY: a slot is within a slab taken from the allocator under it.
        unsafe { self.allocator.discard(ptr, len) };
    }
}

impl Drop for SlabAllocator<'_> {
    fn drop(&mut self) {
        check!(self.live.get() == 0);
        let mut slab = self.slabs.take();
        while let Some(at) = slab {
            // SAFETY: a slab's first bytes hold its link.
            let (before, bytes) = unsafe { at.cast::<Link>().read() };
            slab = before;
            let Ok(layout) = Layout::from_size_align(bytes, BUFFER_ALIGNMENT_BYTES) else {
                failed!()
            };
            // SAFETY: taken with this layout, and no slot in it is used.
            unsafe { self.allocator.deallocate(at, layout) };
        }
    }
}

/// A slot for a batch's 8-byte values, with their buffer's header.
pub(crate) const VALUES_BYTES: usize = HEADER_BYTES + BATCH_ROWS_MAX as usize * 8;

/// A slot for a batch's 4-byte values, such as dictionary indices, or
/// string offsets, of which there's one more than rows.
pub(crate) const INDICES_BYTES: usize =
    (HEADER_BYTES + (BATCH_ROWS_MAX as usize + 1) * 4).next_multiple_of(BUFFER_ALIGNMENT_BYTES);

/// A slot for the small things a batch's columns need: their headers,
/// validity bitmaps, lazy columns' handles.
pub(crate) const SMALL_BYTES: usize = 512;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::env::{Env, MemoryKind};

    #[test]
    fn carves_blocks_exactly_and_hands_them_out_again() {
        let env = Env::current();
        let words = BATCH_ROWS_MAX as usize * 8;
        let before = env.used();
        let a = Buffer::allocate_in(env, MemoryKind::Values, words).unwrap();
        let b = Buffer::allocate_in(env, MemoryKind::Values, words).unwrap();
        // One slab counts, and the second slot is right after the first.
        let slab = env.used() - before;
        let (pa, pb) = (a.as_slice::<u8>().as_ptr(), b.as_slice::<u8>().as_ptr());
        assert_eq!(pb.addr() - pa.addr(), VALUES_BYTES);
        drop(b);
        let c = Buffer::allocate_in(env, MemoryKind::Values, words).unwrap();
        assert_eq!(c.as_slice::<u8>().as_ptr(), pb);
        assert_eq!(env.used() - before, slab);
        drop((a, c));
    }

    #[test]
    fn hands_out_a_whole_block_for_less() {
        let env = Env::current();
        let rows = BATCH_ROWS_MAX as usize;
        let offsets = Buffer::allocate_in(env, MemoryKind::Indices, (rows + 1) * 4).unwrap();
        let short = Buffer::allocate_in(env, MemoryKind::Indices, 10 * 4).unwrap();
        assert_eq!(short.size_bytes(), 40);
        drop((offsets, short));
    }

    #[test]
    #[should_panic(expected = "slot_bytes")]
    fn checks_nothing_asks_for_more_than_a_block() {
        let env = Env::current();
        // With its header, it's more than a small slot.
        let _ = Buffer::allocate_in(env, MemoryKind::Small, SMALL_BYTES);
    }

    #[test]
    fn zeroes_blocks_handed_out_again_when_asked() {
        let env = Env::current();
        let mut a = Buffer::allocate_in(env, MemoryKind::Small, 100).unwrap();
        a.as_mut_slice::<u8>().fill(7);
        drop(a);
        let b = Buffer::allocate_in(env, MemoryKind::Small, 100).unwrap();
        assert_eq!(b.as_slice::<u8>(), [0; 100]);
    }

    #[test]
    fn chunks_count_against_a_limit() {
        // SAFETY: declared before what's allocated from it, so dropped after.
        let env = unsafe { Env::new(Heap, 1000) }.unwrap();
        let before = env.used();
        assert!(Buffer::allocate_in(&env, MemoryKind::Small, 100).is_err());
        assert_eq!(env.used(), before);
    }
}
