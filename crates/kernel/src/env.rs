//! `Env`: the environment a query runs in, which every part of it is
//! passed: its memory and budget now, and what else they share later, such
//! as its I/O.
//!
//! What's allocated from an `Env` points back to it, without a lifetime,
//! so an `Env` must outlive everything allocated from it. One that's
//! `'static`, from `leak` or `current`, always does; one from `new` lives
//! as long as its `OwnedEnv`, which whoever makes it promises is dropped
//! last.

use core::alloc::Layout;
use core::cell::Cell;
use core::ops::Deref;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::slab_allocator::{INDICES_BYTES, SMALL_BYTES, SlabAllocator, VALUES_BYTES};

/// A query's environment. Its memory is its budget, which plans, states, lists and column data
/// of no fixed size come from, and a `SlabAllocator` for each kind of
/// column buffer its batches are made of, taking their slabs from it, so
/// everything a query takes counts against its budget, this included.
///
/// It lives in memory from the allocator it was made with, so it never
/// moves.
pub struct Env {
    budget: Budget,
    values: SlabAllocator<'static>,
    indices: SlabAllocator<'static>,
    small: SlabAllocator<'static>,
    /// Each of the allocators above, by `MemoryKind`, so finding one doesn't
    /// branch.
    allocators: [NonNull<dyn Allocator>; KIND_COUNT],
    /// Frees the allocation this is in, and the allocator in it.
    free: unsafe fn(NonNull<Env>),
    /// The bytes of the allocation this is in.
    allocation_bytes: usize,
}

/// Allocates from the allocator an `Env` was made with while the bytes
/// allocated stay within `limit`, and fails past it.
struct Budget {
    /// The allocator, in the same allocation as the `Env`.
    base: NonNull<dyn Allocator>,
    limit: usize,
    /// The bytes allocated now.
    used: Cell<usize>,
    /// The most bytes allocated at once.
    peak: Cell<usize>,
}

/// Which of an `Env`'s allocators memory comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum MemoryKind {
    /// Its budget.
    Base,
    /// Slots for a batch's 8-byte values, with their buffer's header.
    Values,
    /// Slots for a batch's 4-byte values, such as dictionary indices.
    Indices,
    /// Slots for the small things a batch's columns need.
    Small,
}

const KIND_COUNT: usize = MemoryKind::Small as usize + 1;

/// An `Env` and the allocator it was made with, in one allocation from it.
#[repr(C)]
struct Allocation<A> {
    env: Env,
    base: A,
}

impl Env {
    /// A query's memory, all of it from `base`, which it owns, at most
    /// `limit` bytes of it.
    ///
    /// # Safety
    ///
    /// Everything allocated from it must be dropped before the
    /// `OwnedEnv`.
    #[expect(clippy::new_ret_no_self, reason = "it lives in memory it owns, held by the owner")]
    pub unsafe fn new<A: Allocator + 'static>(
        base: A,
        limit: usize,
    ) -> Result<OwnedEnv, AllocError> {
        let layout = Layout::new::<Allocation<A>>();
        if layout.size() > limit {
            return Err(AllocError);
        }
        let allocation = base.allocate(layout)?.cast::<Allocation<A>>();
        // SAFETY: the allocation has room for a `Allocation<A>`, each of whose fields
        // is written before anything reads it. It's freed only by `free`,
        // with the slab allocators borrowing its budget, so the budget can
        // be lent for `'static` here.
        unsafe {
            let at = allocation.as_ptr();
            (&raw mut (*at).base).write(base);
            let base: &dyn Allocator = &(*at).base;
            let budget = &raw mut (*at).env.budget;
            budget.write(Budget {
                base: NonNull::from(base),
                limit,
                used: Cell::new(layout.size()),
                peak: Cell::new(layout.size()),
            });
            let budget: &'static Budget = &*budget;
            let env = &raw mut (*at).env;
            (&raw mut (*env).values).write(SlabAllocator::new(budget, VALUES_BYTES));
            (&raw mut (*env).indices).write(SlabAllocator::new(budget, INDICES_BYTES));
            (&raw mut (*env).small).write(SlabAllocator::new(budget, SMALL_BYTES));
            let allocators: [NonNull<dyn Allocator>; KIND_COUNT] = [
                NonNull::from(budget),
                NonNull::from(&(*env).values),
                NonNull::from(&(*env).indices),
                NonNull::from(&(*env).small),
            ];
            (&raw mut (*env).allocators).write(allocators);
            (&raw mut (*env).free).write(free::<A>);
            (&raw mut (*env).allocation_bytes).write(layout.size());
        }
        Ok(OwnedEnv(allocation.cast()))
    }

    /// As `new`, one that's never freed, so outlives everything: for one
    /// kept for a whole program or thread.
    pub fn leak<A: Allocator + 'static>(base: A, limit: usize) -> Result<&'static Env, AllocError> {
        // SAFETY: it's never dropped.
        let owned = unsafe { Env::new(base, limit) }?;
        let env = owned.0;
        core::mem::forget(owned);
        // SAFETY: it's never freed.
        Ok(unsafe { env.as_ref() })
    }

    /// This thread's `Env`, from the system's heap, without a limit, made
    /// the first time it's asked for, and freed when the thread exits unless
    /// something allocated from it is still held: for tools, and tests.
    ///
    /// Not quite `'static`: a thread-local's destructor that runs after its
    /// own mustn't use what was allocated from it, as an empty list, which
    /// holds no memory, would then grow from a freed `Env`.
    #[cfg(any(test, feature = "std"))]
    pub fn current() -> &'static Env {
        std::thread_local! {
            // SAFETY: kept in a thread-local.
            static CURRENT: ThreadEnv = unsafe { ThreadEnv::new() };
        }
        CURRENT.with(ThreadEnv::get)
    }

    /// The allocator memory of `kind` comes from.
    pub fn allocator(&self, kind: MemoryKind) -> &dyn Allocator {
        // SAFETY: each points into this `Env`, which outlives the borrow.
        unsafe { at!(self.allocators, kind as usize).as_ref() }
    }

    /// Whether nothing allocated from it is held: only its own allocation and
    /// its slab allocators' slabs are.
    pub fn holds_nothing(&self) -> bool {
        let fixed = [&self.values, &self.indices, &self.small];
        let slabs: usize = fixed.iter().map(|fixed| fixed.slab_bytes()).sum();
        fixed.iter().all(|fixed| fixed.is_idle()) && self.used() == self.allocation_bytes + slabs
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

/// Owns an `Env` from `Env::new`, and frees it when dropped, checking
/// everything allocated from it was freed first.
pub struct OwnedEnv(NonNull<Env>);

impl Deref for OwnedEnv {
    type Target = Env;

    fn deref(&self) -> &Env {
        // SAFETY: this owns the `Env`, which lives until it's dropped.
        unsafe { self.0.as_ref() }
    }
}

impl Drop for OwnedEnv {
    fn drop(&mut self) {
        // SAFETY: `free` was made for this `Env`'s allocation, and whoever made
        // it promised nothing allocated from it is left.
        unsafe { (self.free)(self.0) }
    }
}

/// Holds an `Env` for a thread, and frees it as `OwnedEnv::release`
/// does when the thread exits.
pub struct ThreadEnv(core::mem::ManuallyDrop<OwnedEnv>);

impl ThreadEnv {
    /// An `Env` from the system's heap, without a limit.
    ///
    /// # Safety
    ///
    /// It must be kept in a thread-local, as `get` lends the `Env` for
    /// as long as the thread lives, under `Env::current`'s rule.
    pub unsafe fn new() -> ThreadEnv {
        // SAFETY: it's released, not dropped, so it's freed only if nothing
        // allocated from it is held.
        let Ok(env) = (unsafe { Env::new(crate::allocator::Heap, usize::MAX) }) else { failed!() };
        ThreadEnv(core::mem::ManuallyDrop::new(env))
    }

    /// The `Env`, for as long as the thread lives, as
    /// `Env::current` says.
    pub fn get(&self) -> &'static Env {
        // SAFETY: it's freed only when the thread exits.
        unsafe { self.0.0.as_ref() }
    }
}

impl Drop for ThreadEnv {
    fn drop(&mut self) {
        // SAFETY: taken once, here.
        unsafe { core::mem::ManuallyDrop::take(&mut self.0) }.release();
    }
}

impl OwnedEnv {
    /// Frees the `Env` if nothing allocated from it is held, and keeps it
    /// for good if something is, which can then still use it.
    pub fn release(self) {
        if !self.holds_nothing() {
            core::mem::forget(self);
        }
    }
}

/// Frees `env`, a `Allocation<A>`'s, and its allocator.
///
/// # Safety
///
/// `env` must be in a `Allocation<A>` from `Env::new`, with nothing
/// allocated from it left.
unsafe fn free<A: Allocator>(env: NonNull<Env>) {
    let allocation = env.cast::<Allocation<A>>();
    // SAFETY: the allocation holds both fields. Dropping the slab allocators
    // first gives their slabs back, so all that's left is the allocation, which
    // `base` then frees from outside it. They're dropped one by one, not as
    // an `Env`, whose `&mut` would end the budget's borrow they hold; the
    // rest needs no drop.
    unsafe {
        let env = &raw mut (*allocation.as_ptr()).env;
        (&raw mut (*env).values).drop_in_place();
        (&raw mut (*env).indices).drop_in_place();
        (&raw mut (*env).small).drop_in_place();
        check!((*allocation.as_ptr()).env.used() == size_of::<Allocation<A>>());
        let base = (&raw const (*allocation.as_ptr()).base).read();
        base.deallocate(allocation.cast(), Layout::new::<Allocation<A>>());
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
        // SAFETY: the allocator lives in the `Env`'s allocation, as this does.
        let base = unsafe { self.base.as_ref() };
        let ptr = if zeroed { base.allocate_zeroed(layout) } else { base.allocate(layout) }?;
        self.used.set(used);
        self.peak.set(self.peak.get().max(used));
        Ok(ptr)
    }
}

// SAFETY: forwards to the allocator the `Env` was made with.
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
unsafe impl Allocator for Env {
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
        let env = unsafe { Env::new(Heap, 2000) }.unwrap();
        let empty = env.used();
        let first = Buffer::allocate(&env, 600).unwrap();
        // A buffer's header counts too.
        let one = env.used() - empty;
        assert!(one > 600);
        assert!(Buffer::allocate(&env, 1200).is_err());
        assert_eq!(env.used() - empty, one);

        // What's freed can be allocated again.
        drop(first);
        assert_eq!(env.used(), empty);
        let second = Buffer::allocate(&env, 1000).unwrap();
        let two = env.used();
        drop(second);
        assert_eq!((env.used(), env.peak()), (empty, two));
    }

    #[test]
    fn cannot_be_made_past_its_limit() {
        // SAFETY: nothing is allocated from it.
        assert!(unsafe { Env::new(Heap, 16) }.is_err());
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
        let env = unsafe { Env::new(Bump(&OUTLIVED), usize::MAX) }.unwrap();
        core::mem::forget(Buffer::allocate(&env, 8).unwrap());
        drop(env);
    }

    #[test]
    fn released_once_nothing_is_held() {
        // SAFETY: what's allocated from it is dropped before it's released.
        let env = unsafe { Env::new(Heap, usize::MAX) }.unwrap();
        let empty = env.used();
        drop(Buffer::allocate_in(&env, MemoryKind::Small, 8).unwrap());
        assert!(env.holds_nothing());
        assert!(env.used() > empty);
        env.release();
    }

    #[test]
    fn kept_while_something_is_held() {
        // SAFETY: released, not dropped, so the buffer it keeps is never
        // freed through a freed `Env`.
        let env = unsafe { Env::new(Bump(&KEPT), usize::MAX) }.unwrap();
        let buffer = Buffer::allocate(&env, 8).unwrap();
        assert!(!env.holds_nothing());
        env.release();
        assert_eq!(buffer.as_slice::<u64>(), [0]);
        core::mem::forget(buffer);
    }

    #[test]
    fn each_thread_has_its_own() {
        let here = core::ptr::from_ref(Env::current());
        assert!(core::ptr::eq(here, Env::current()));
        let there = std::thread::spawn(|| core::ptr::from_ref(Env::current()).addr());
        assert_ne!(here.addr(), there.join().unwrap());
    }
}
