//! `FixedAllocator`, which hands out blocks of one size, and the sizes of a
//! query's blocks for its batches.

use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, HEADER_BYTES};
use crate::row_batch::BATCH_ROWS_MAX;

/// Hands out blocks of one size, carved from chunks taken from the
/// allocator under it, and takes each back to hand out again, the last
/// freed first, while it's likely still in cache. Chunks are given back
/// when it's dropped, which must be after every block is freed.
///
/// Any layout up to its block size gets a whole block. It's one type for
/// every size, so its code isn't copied for each.
pub(crate) struct FixedAllocator<'a> {
    allocator: &'a dyn Allocator,
    /// How many bytes each block has.
    block_bytes: usize,
    /// Blocks freed, each holding the next.
    free: Cell<Option<NonNull<u8>>>,
    /// Where the next block is carved from in the newest chunk, and how
    /// many bytes are left after it.
    next: Cell<Option<(NonNull<u8>, usize)>>,
    /// The newest chunk, which holds the one before it, and so on.
    chunks: Cell<Option<NonNull<u8>>>,
    /// How many chunks were taken: each is twice the one before, up to
    /// `CHUNK_BYTES_MAX`, so a short run takes little and a long one few.
    chunk_count: Cell<u8>,
    /// Blocks handed out and not freed.
    live: Cell<usize>,
}

/// A first chunk's size, at least, and the most a chunk takes, at least a
/// block's size more.
const CHUNK_BYTES_MIN: usize = 4 << 10;
const CHUNK_BYTES_MAX: usize = 1 << 20;

/// What a chunk's first bytes hold: the chunk taken before it, and its size.
type Link = (Option<NonNull<u8>>, usize);

impl<'a> FixedAllocator<'a> {
    pub(crate) fn new(allocator: &'a dyn Allocator, block_bytes: usize) -> FixedAllocator<'a> {
        check!(block_bytes.is_multiple_of(BUFFER_ALIGNMENT_BYTES));
        FixedAllocator {
            allocator,
            block_bytes,
            free: Cell::new(None),
            next: Cell::new(None),
            chunks: Cell::new(None),
            chunk_count: Cell::new(0),
            live: Cell::new(0),
        }
    }

    /// Whether no block is handed out, and so the bytes it holds are all
    /// its chunks', `chunk_bytes`.
    pub(crate) fn is_idle(&self) -> bool {
        self.live.get() == 0
    }

    /// The bytes of the chunks it took.
    pub(crate) fn chunk_bytes(&self) -> usize {
        let (mut bytes, mut chunk) = (0, self.chunks.get());
        while let Some(at) = chunk {
            // SAFETY: a chunk's first bytes hold its link.
            let (before, size) = unsafe { at.cast::<Link>().read() };
            (bytes, chunk) = (bytes + size, before);
        }
        bytes
    }

    /// The layout of the `chunk_count`th chunk: four blocks, and the link,
    /// at first, then twice the one before.
    fn chunk_layout(&self) -> Layout {
        let first = (4 * self.block_bytes + BUFFER_ALIGNMENT_BYTES).max(CHUNK_BYTES_MIN);
        let max = CHUNK_BYTES_MAX.max(first);
        let bytes = (first << self.chunk_count.get().min(16)).min(max);
        let Ok(layout) = Layout::from_size_align(bytes, BUFFER_ALIGNMENT_BYTES) else { failed!() };
        layout
    }

    /// Takes a new chunk, and returns where its first block is and the
    /// bytes after it.
    #[cold]
    fn take_chunk(&self) -> Result<(NonNull<u8>, usize), AllocError> {
        let layout = self.chunk_layout();
        let chunk = self.allocator.allocate(layout)?;
        self.chunk_count.set(self.chunk_count.get().saturating_add(1));
        // SAFETY: a chunk's first bytes hold its link.
        unsafe { chunk.cast::<Link>().write((self.chunks.get(), layout.size())) };
        self.chunks.set(Some(chunk));
        // SAFETY: blocks start after the link, within the chunk.
        Ok((unsafe { chunk.add(BUFFER_ALIGNMENT_BYTES) }, layout.size() - BUFFER_ALIGNMENT_BYTES))
    }
}

// SAFETY: blocks are carved from chunks that live until it's dropped, which
// checks none is still handed out.
unsafe impl Allocator for FixedAllocator<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        // Asking for more than a block is a caller's mistake, not running out
        // of memory, which `AllocError` would say, and spilling can't fix.
        check!(layout.size() <= self.block_bytes && layout.align() <= BUFFER_ALIGNMENT_BYTES);
        let block = if let Some(block) = self.free.get() {
            // SAFETY: a freed block holds the next one.
            self.free.set(unsafe { block.cast::<Option<NonNull<u8>>>().read() });
            block
        } else {
            let (next, left) = match self.next.get() {
                Some((next, left)) if left >= self.block_bytes => (next, left),
                _ => self.take_chunk()?,
            };
            // SAFETY: a block's bytes are left after `next`, in its chunk.
            let after = unsafe { next.add(self.block_bytes) };
            self.next.set(Some((after, left - self.block_bytes)));
            next
        };
        self.live.set(self.live.get() + 1);
        Ok(block)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _: Layout) {
        // SAFETY: the block is freed, so it can hold the next free one.
        unsafe { ptr.cast::<Option<NonNull<u8>>>().write(self.free.get()) };
        self.free.set(Some(ptr));
        self.live.set(self.live.get() - 1);
    }

    unsafe fn discard(&self, ptr: NonNull<u8>, len: usize) {
        // SAFETY: a block is within a chunk taken from the allocator under it.
        unsafe { self.allocator.discard(ptr, len) };
    }
}

impl Drop for FixedAllocator<'_> {
    fn drop(&mut self) {
        check!(self.live.get() == 0);
        let mut chunk = self.chunks.take();
        while let Some(at) = chunk {
            // SAFETY: a chunk's first bytes hold its link.
            let (before, bytes) = unsafe { at.cast::<Link>().read() };
            chunk = before;
            let Ok(layout) = Layout::from_size_align(bytes, BUFFER_ALIGNMENT_BYTES) else {
                failed!()
            };
            // SAFETY: taken with this layout, and no block in it is used.
            unsafe { self.allocator.deallocate(at, layout) };
        }
    }
}

/// A block for a batch's 8-byte values, with their buffer's header.
pub(crate) const VALUES_BYTES: usize = HEADER_BYTES + BATCH_ROWS_MAX as usize * 8;

/// A block for a batch's 4-byte values, such as dictionary indices, or
/// string offsets, of which there's one more than rows.
pub(crate) const INDICES_BYTES: usize =
    (HEADER_BYTES + (BATCH_ROWS_MAX as usize + 1) * 4).next_multiple_of(BUFFER_ALIGNMENT_BYTES);

/// A block for the small things a batch's columns need: their headers,
/// validity bitmaps, lazy columns' handles.
pub(crate) const SMALL_BYTES: usize = 512;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::env::{Env, Kind};

    #[test]
    fn carves_blocks_exactly_and_hands_them_out_again() {
        let env = Env::current();
        let words = BATCH_ROWS_MAX as usize * 8;
        let before = env.used();
        let a = Buffer::allocate_in(env, Kind::Values, words).unwrap();
        let b = Buffer::allocate_in(env, Kind::Values, words).unwrap();
        // One chunk counts, and the second block is right after the first.
        let chunk = env.used() - before;
        let (pa, pb) = (a.as_slice::<u8>().as_ptr(), b.as_slice::<u8>().as_ptr());
        assert_eq!(pb.addr() - pa.addr(), VALUES_BYTES);
        drop(b);
        let c = Buffer::allocate_in(env, Kind::Values, words).unwrap();
        assert_eq!(c.as_slice::<u8>().as_ptr(), pb);
        assert_eq!(env.used() - before, chunk);
        drop((a, c));
    }

    #[test]
    fn hands_out_a_whole_block_for_less() {
        let env = Env::current();
        let rows = BATCH_ROWS_MAX as usize;
        let offsets = Buffer::allocate_in(env, Kind::Indices, (rows + 1) * 4).unwrap();
        let short = Buffer::allocate_in(env, Kind::Indices, 10 * 4).unwrap();
        assert_eq!(short.size_bytes(), 40);
        drop((offsets, short));
    }

    #[test]
    #[should_panic(expected = "block_bytes")]
    fn checks_nothing_asks_for_more_than_a_block() {
        let env = Env::current();
        // With its header, it's more than a small block.
        let _ = Buffer::allocate_in(env, Kind::Small, SMALL_BYTES);
    }

    #[test]
    fn zeroes_blocks_handed_out_again_when_asked() {
        let env = Env::current();
        let mut a = Buffer::allocate_in(env, Kind::Small, 100).unwrap();
        a.as_mut_slice::<u8>().fill(7);
        drop(a);
        let b = Buffer::allocate_in(env, Kind::Small, 100).unwrap();
        assert_eq!(b.as_slice::<u8>(), [0; 100]);
    }

    #[test]
    fn chunks_count_against_a_limit() {
        // SAFETY: declared before what's allocated from it, so dropped after.
        let env = unsafe { Env::new(Heap, 1000) }.unwrap();
        let before = env.used();
        assert!(Buffer::allocate_in(&env, Kind::Small, 100).is_err());
        assert_eq!(env.used(), before);
    }
}
