//! `Global`: what every part of a query shares, such as its memory and its
//! budget.
//!
//! What's allocated from a `Global` points back to it, without a lifetime,
//! so a `Global` must outlive everything allocated from it. One that's
//! `'static`, from `leak` or `current`, always does; one from `new` lives
//! as long as its `OwnedGlobal`, which whoever makes it promises is dropped
//! last.

use core::alloc::Layout;
use core::cell::Cell;
use core::ops::Deref;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::fixed_allocator::{FixedAllocator, INDICES_BYTES, SMALL_BYTES, VALUES_BYTES};

/// A query's memory: its budget, which plans, states, lists and column data
/// of no fixed size come from, and a `FixedAllocator` for each kind of
/// column buffer its batches are made of, taking their chunks from it, so
/// everything a query takes counts against its budget, this included.
///
/// It lives in memory from the allocator it was made with, so it never
/// moves.
pub struct Global {
    budget: Budget,
    values: FixedAllocator<'static>,
    indices: FixedAllocator<'static>,
    small: FixedAllocator<'static>,
    /// Each of the allocators above, by `Kind`, so finding one doesn't
    /// branch.
    allocators: [NonNull<dyn Allocator>; KIND_COUNT],
    /// Frees the block this is in, and the allocator in it.
    free: unsafe fn(NonNull<Global>),
    /// The bytes of the block this is in.
    block_bytes: usize,
}

/// Allocates from the allocator a `Global` was made with while the bytes
/// allocated stay within `limit`, and fails past it.
struct Budget {
    /// The allocator, in the same block as the `Global`.
    base: NonNull<dyn Allocator>,
    limit: usize,
    /// The bytes allocated now.
    used: Cell<usize>,
    /// The most bytes allocated at once.
    peak: Cell<usize>,
}

/// Which of a `Global`'s allocators memory comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Kind {
    /// Its budget.
    Base,
    /// Blocks for a batch's 8-byte values, with their buffer's header.
    Values,
    /// Blocks for a batch's 4-byte values, such as dictionary indices.
    Indices,
    /// Blocks for the small things a batch's columns need.
    Small,
}

const KIND_COUNT: usize = Kind::Small as usize + 1;

/// A `Global` and the allocator it was made with, in one allocation from it.
#[repr(C)]
struct Block<A> {
    global: Global,
    base: A,
}

impl Global {
    /// A query's memory, all of it from `base`, which it owns, at most
    /// `limit` bytes of it.
    ///
    /// # Safety
    ///
    /// Everything allocated from it must be dropped before the
    /// `OwnedGlobal`.
    #[expect(clippy::new_ret_no_self, reason = "it lives in memory it owns, held by the owner")]
    pub unsafe fn new<A: Allocator + 'static>(
        base: A,
        limit: usize,
    ) -> Result<OwnedGlobal, AllocError> {
        let layout = Layout::new::<Block<A>>();
        if layout.size() > limit {
            return Err(AllocError);
        }
        let block = base.allocate(layout)?.cast::<Block<A>>();
        // SAFETY: the block has room for a `Block<A>`, each of whose fields
        // is written before anything reads it. It's freed only by `free`,
        // with the fixed allocators borrowing its budget, so the budget can
        // be lent for `'static` here.
        unsafe {
            let at = block.as_ptr();
            (&raw mut (*at).base).write(base);
            let base: &dyn Allocator = &(*at).base;
            let budget = &raw mut (*at).global.budget;
            budget.write(Budget {
                base: NonNull::from(base),
                limit,
                used: Cell::new(layout.size()),
                peak: Cell::new(layout.size()),
            });
            let budget: &'static Budget = &*budget;
            let global = &raw mut (*at).global;
            (&raw mut (*global).values).write(FixedAllocator::new(budget, VALUES_BYTES));
            (&raw mut (*global).indices).write(FixedAllocator::new(budget, INDICES_BYTES));
            (&raw mut (*global).small).write(FixedAllocator::new(budget, SMALL_BYTES));
            let allocators: [NonNull<dyn Allocator>; KIND_COUNT] = [
                NonNull::from(budget),
                NonNull::from(&(*global).values),
                NonNull::from(&(*global).indices),
                NonNull::from(&(*global).small),
            ];
            (&raw mut (*global).allocators).write(allocators);
            (&raw mut (*global).free).write(free::<A>);
            (&raw mut (*global).block_bytes).write(layout.size());
        }
        Ok(OwnedGlobal(block.cast()))
    }

    /// As `new`, one that's never freed, so outlives everything: for one
    /// kept for a whole program or thread.
    pub fn leak<A: Allocator + 'static>(
        base: A,
        limit: usize,
    ) -> Result<&'static Global, AllocError> {
        // SAFETY: it's never dropped.
        let owned = unsafe { Global::new(base, limit) }?;
        let global = owned.0;
        core::mem::forget(owned);
        // SAFETY: it's never freed.
        Ok(unsafe { global.as_ref() })
    }

    /// This thread's `Global`, from the system's heap, without a limit, made
    /// the first time it's asked for, and freed when the thread exits unless
    /// something allocated from it is still held: for tools, and tests.
    ///
    /// Not quite `'static`: a thread-local's destructor that runs after its
    /// own mustn't use what was allocated from it, as an empty list, which
    /// holds no memory, would then grow from a freed `Global`.
    #[cfg(any(test, feature = "std"))]
    pub fn current() -> &'static Global {
        std::thread_local! {
            // SAFETY: kept in a thread-local.
            static CURRENT: ThreadGlobal = unsafe { ThreadGlobal::new() };
        }
        CURRENT.with(ThreadGlobal::get)
    }

    /// The allocator memory of `kind` comes from.
    pub fn allocator(&self, kind: Kind) -> &dyn Allocator {
        // SAFETY: each points into this `Global`, which outlives the borrow.
        unsafe { at!(self.allocators, kind as usize).as_ref() }
    }

    /// Whether nothing allocated from it is held: only its own block and
    /// its block allocators' chunks are.
    pub fn holds_nothing(&self) -> bool {
        let fixed = [&self.values, &self.indices, &self.small];
        let chunks: usize = fixed.iter().map(|fixed| fixed.chunk_bytes()).sum();
        fixed.iter().all(|fixed| fixed.is_idle()) && self.used() == self.block_bytes + chunks
    }

    /// The bytes allocated now, this included.
    pub fn used(&self) -> usize {
        self.budget.used.get()
    }

    /// The most bytes allocated at once.
    pub fn peak(&self) -> usize {
        self.budget.peak.get()
    }
}

/// Owns a `Global` from `Global::new`, and frees it when dropped, checking
/// everything allocated from it was freed first.
pub struct OwnedGlobal(NonNull<Global>);

impl Deref for OwnedGlobal {
    type Target = Global;

    fn deref(&self) -> &Global {
        // SAFETY: this owns the `Global`, which lives until it's dropped.
        unsafe { self.0.as_ref() }
    }
}

impl Drop for OwnedGlobal {
    fn drop(&mut self) {
        // SAFETY: `free` was made for this `Global`'s block, and whoever made
        // it promised nothing allocated from it is left.
        unsafe { (self.free)(self.0) }
    }
}

/// Holds a `Global` for a thread, and frees it as `OwnedGlobal::release`
/// does when the thread exits.
pub struct ThreadGlobal(core::mem::ManuallyDrop<OwnedGlobal>);

impl ThreadGlobal {
    /// A `Global` from the system's heap, without a limit.
    ///
    /// # Safety
    ///
    /// It must be kept in a thread-local, as `get` lends the `Global` for
    /// as long as the thread lives, under `Global::current`'s rule.
    pub unsafe fn new() -> ThreadGlobal {
        // SAFETY: it's released, not dropped, so it's freed only if nothing
        // allocated from it is held.
        let Ok(global) = (unsafe { Global::new(crate::allocator::Heap, usize::MAX) }) else {
            failed!()
        };
        ThreadGlobal(core::mem::ManuallyDrop::new(global))
    }

    /// The `Global`, for as long as the thread lives, as
    /// `Global::current` says.
    pub fn get(&self) -> &'static Global {
        // SAFETY: it's freed only when the thread exits.
        unsafe { self.0.0.as_ref() }
    }
}

impl Drop for ThreadGlobal {
    fn drop(&mut self) {
        // SAFETY: taken once, here.
        unsafe { core::mem::ManuallyDrop::take(&mut self.0) }.release();
    }
}

impl OwnedGlobal {
    /// Frees the `Global` if nothing allocated from it is held, and keeps it
    /// for good if something is, which can then still use it.
    pub fn release(self) {
        if !self.holds_nothing() {
            core::mem::forget(self);
        }
    }
}

/// Frees `global`, a `Block<A>`'s, and its allocator.
///
/// # Safety
///
/// `global` must be in a `Block<A>` from `Global::new`, with nothing
/// allocated from it left.
unsafe fn free<A: Allocator>(global: NonNull<Global>) {
    let block = global.cast::<Block<A>>();
    // SAFETY: the block holds both fields. Dropping the fixed allocators
    // first gives their chunks back, so all that's left is the block, which
    // `base` then frees from outside it. They're dropped one by one, not as
    // a `Global`, whose `&mut` would end the budget's borrow they hold; the
    // rest needs no drop.
    unsafe {
        let global = &raw mut (*block.as_ptr()).global;
        (&raw mut (*global).values).drop_in_place();
        (&raw mut (*global).indices).drop_in_place();
        (&raw mut (*global).small).drop_in_place();
        check!((*block.as_ptr()).global.used() == size_of::<Block<A>>());
        let base = (&raw const (*block.as_ptr()).base).read();
        base.deallocate(block.cast(), Layout::new::<Block<A>>());
    }
}

impl Budget {
    /// Allocates `layout`, zeroed if `zeroed`, if it fits within the limit,
    /// and counts it.
    fn charge(&self, layout: Layout, zeroed: bool) -> Result<NonNull<u8>, AllocError> {
        let used = self.used.get().checked_add(layout.size()).ok_or(AllocError)?;
        if used > self.limit {
            return Err(AllocError);
        }
        // SAFETY: the allocator lives in the `Global`'s block, as this does.
        let base = unsafe { self.base.as_ref() };
        let ptr = if zeroed { base.allocate_zeroed(layout) } else { base.allocate(layout) }?;
        self.used.set(used);
        self.peak.set(self.peak.get().max(used));
        Ok(ptr)
    }
}

// SAFETY: forwards to the allocator the `Global` was made with.
unsafe impl Allocator for Budget {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.charge(layout, false)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.charge(layout, true)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded from the caller; the allocator lives as this does.
        unsafe { self.base.as_ref().deallocate(ptr, layout) };
        self.used.set(self.used.get() - layout.size());
    }

    unsafe fn discard(&self, ptr: NonNull<u8>, len: usize) {
        // SAFETY: forwarded from the caller; the allocator lives as this does.
        unsafe { self.base.as_ref().discard(ptr, len) }
    }
}

// SAFETY: forwards to its budget.
unsafe impl Allocator for Global {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.budget.allocate(layout)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.budget.allocate_zeroed(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded from the caller.
        unsafe { self.budget.deallocate(ptr, layout) }
    }

    unsafe fn discard(&self, ptr: NonNull<u8>, len: usize) {
        // SAFETY: forwarded from the caller.
        unsafe { self.budget.discard(ptr, len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;

    #[test]
    fn fails_past_its_limit_and_counts_the_peak() {
        // SAFETY: declared before what's allocated from it, so dropped after.
        let global = unsafe { Global::new(Heap, 2000) }.unwrap();
        let empty = global.used();
        let first = Buffer::allocate(&global, 600).unwrap();
        // A buffer's header counts too.
        let one = global.used() - empty;
        assert!(one > 600);
        assert!(Buffer::allocate(&global, 1200).is_err());
        assert_eq!(global.used() - empty, one);

        // What's freed can be allocated again.
        drop(first);
        assert_eq!(global.used(), empty);
        let second = Buffer::allocate(&global, 1000).unwrap();
        let two = global.used();
        drop(second);
        assert_eq!((global.used(), global.peak()), (empty, two));
    }

    #[test]
    fn cannot_be_made_past_its_limit() {
        // SAFETY: nothing is allocated from it.
        assert!(unsafe { Global::new(Heap, 16) }.is_err());
    }

    /// A static block, which `Bump` hands out, so what's allocated from it
    /// can be kept for good without leaking heap memory.
    #[repr(align(64))]
    struct Arena {
        bytes: core::cell::UnsafeCell<[u8; 4096]>,
        next: core::sync::atomic::AtomicUsize,
    }

    // SAFETY: each allocation is a distinct part of the block.
    unsafe impl Sync for Arena {}

    impl Arena {
        const fn new() -> Arena {
            Arena {
                bytes: core::cell::UnsafeCell::new([0; 4096]),
                next: core::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    /// Hands out an `Arena`, bump by bump, and never takes any back.
    struct Bump(&'static Arena);

    // SAFETY: each allocation is a distinct part of the arena, which lives
    // for the program.
    unsafe impl Allocator for Bump {
        fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
            use core::sync::atomic::Ordering::Relaxed;
            let start = self.0.next.load(Relaxed).next_multiple_of(layout.align());
            if start + layout.size() > 4096 {
                return Err(AllocError);
            }
            self.0.next.store(start + layout.size(), Relaxed);
            // SAFETY: within the block.
            NonNull::new(unsafe { self.0.bytes.get().cast::<u8>().add(start) }).ok_or(AllocError)
        }

        unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {}
    }

    static OUTLIVED: Arena = Arena::new();
    static KEPT: Arena = Arena::new();

    #[test]
    #[should_panic(expected = "used")]
    fn checks_nothing_outlives_it() {
        // SAFETY: the buffer is forgotten, so never freed through it after
        // the drop, whose check is what's tested.
        let global = unsafe { Global::new(Bump(&OUTLIVED), usize::MAX) }.unwrap();
        core::mem::forget(Buffer::allocate(&global, 8).unwrap());
        drop(global);
    }

    #[test]
    fn released_once_nothing_is_held() {
        // SAFETY: what's allocated from it is dropped before it's released.
        let global = unsafe { Global::new(Heap, usize::MAX) }.unwrap();
        let empty = global.used();
        drop(Buffer::allocate_in(&global, Kind::Small, 8).unwrap());
        assert!(global.holds_nothing());
        assert!(global.used() > empty);
        global.release();
    }

    #[test]
    fn kept_while_something_is_held() {
        // SAFETY: released, not dropped, so the buffer it keeps is never
        // freed through a freed `Global`.
        let global = unsafe { Global::new(Bump(&KEPT), usize::MAX) }.unwrap();
        let buffer = Buffer::allocate(&global, 8).unwrap();
        assert!(!global.holds_nothing());
        global.release();
        assert_eq!(buffer.as_slice::<u64>(), [0]);
        core::mem::forget(buffer);
    }

    #[test]
    fn each_thread_has_its_own() {
        let here = core::ptr::from_ref(Global::current());
        assert!(core::ptr::eq(here, Global::current()));
        let there = std::thread::spawn(|| core::ptr::from_ref(Global::current()).addr());
        assert_ne!(here.addr(), there.join().unwrap());
    }
}
