//! `Env`: the environment queries run in, which every part of them is
//! passed: the resources they use, each a piece of its own, such as its
//! `Memory`.
//!
//! What's allocated from an `Env` points back to it, without a lifetime,
//! so it must be dropped before its `Env`. Dropping an `Env` while
//! something allocated from it is held is a bug, which fails a check before
//! anything is freed, so nothing is left pointing to freed memory.

#![allow(unsafe_code)]

use core::alloc::Layout;
use core::cell::Cell;
use core::ops::Deref;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
#[cfg(feature = "spill")]
use crate::manager::Manager;
use crate::slab_allocator::{INDICES_BYTES, SMALL_BYTES, SlabAllocator, VALUES_BYTES};
#[cfg(feature = "spill")]
use crate::spill::SpillStore;

/// A query's environment. Its memory is its budget, which plans, states, lists and column data
/// of no fixed size come from, and a `SlabAllocator` for each kind of
/// column buffer its batches are made of, taking their slabs from it, so
/// everything a query takes counts against its budget, this included.
///
/// It lives in memory from the allocator it was made with, so it never
/// moves.
pub struct Env {
    /// Where plans, states, lists and column data of no fixed size come
    /// from, and the slab allocators' slabs.
    budget: Budget,
    /// Slots for a batch's 8-byte values.
    values: SlabAllocator<'static>,
    /// Slots for a batch's 4-byte values.
    indices: SlabAllocator<'static>,
    /// Slots for the small things a batch's columns need.
    small: SlabAllocator<'static>,
    /// Each of the allocators above, by `MemoryKind`, so finding one doesn't
    /// branch.
    allocators: [NonNull<dyn Allocator>; KIND_COUNT],
    /// Frees the allocation this is in, and the allocator in it.
    free: unsafe fn(NonNull<Env>),
    /// The bytes of the allocation this is in.
    allocation_bytes: usize,
    /// Where what doesn't fit in its budget goes, if anywhere.
    #[cfg(feature = "spill")]
    spill: Cell<Option<&'static dyn SpillStore>>,
}

/// Allocates from the allocator an `Env` was made with while the bytes
/// allocated stay within `limit`, and fails past it.
struct Budget {
    /// The allocator, in the same allocation as the `Env`.
    base: NonNull<dyn Allocator>,
    /// The most bytes that can be allocated at once.
    limit: usize,
    /// The bytes allocated now.
    used: Cell<usize>,
    /// The most bytes allocated at once.
    peak: Cell<usize>,
    /// Makes room when an allocation doesn't fit.
    #[cfg(feature = "spill")]
    manager: Manager,
    /// The `Env` it's in, which its manager spills sinks with.
    #[cfg(feature = "spill")]
    env: NonNull<Env>,
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

/// An `Env`'s memory: the allocator it comes from, and the most bytes of
/// it the `Env` takes, itself included.
pub struct Memory<A> {
    /// Where the memory comes from.
    allocator: A,
    /// The most bytes taken at once.
    limit: usize,
}

impl<A: Allocator + 'static> Memory<A> {
    pub fn new(allocator: A, limit: usize) -> Memory<A> {
        Memory { allocator, limit }
    }
}

/// An `Env` and the allocator it was made with, in one allocation from it.
#[repr(C)]
struct Allocation<A> {
    env: Env,
    base: A,
}

impl Env {
    /// An environment with `memory`, which it owns.
    #[expect(clippy::new_ret_no_self, reason = "it lives in memory it owns, held by the owner")]
    pub fn new<A: Allocator + 'static>(memory: Memory<A>) -> Result<OwnedEnv, AllocError> {
        let Memory { allocator: base, limit } = memory;
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
                #[cfg(feature = "spill")]
                manager: Manager::new(),
                #[cfg(feature = "spill")]
                env: NonNull::new_unchecked(&raw mut (*at).env),
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
            #[cfg(feature = "spill")]
            (&raw mut (*env).spill).write(Cell::new(None));
        }
        Ok(OwnedEnv(allocation.cast()))
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
    pub(crate) fn allocator(&self, kind: MemoryKind) -> &dyn Allocator {
        // SAFETY: each points into this `Env`, which outlives the borrow.
        unsafe { at!(self.allocators, kind as usize).as_ref() }
    }

    /// Whether nothing allocated from it is held: only its own allocation and
    /// its slab allocators' slabs are.
    pub(crate) fn holds_nothing(&self) -> bool {
        let fixed = [&self.values, &self.indices, &self.small];
        let slabs: usize = fixed.iter().map(|fixed| fixed.slab_bytes()).sum();
        let used = self.budget.used.get();
        fixed.iter().all(|fixed| fixed.is_idle()) && used == self.allocation_bytes + slabs
    }

    /// Runs `f`, a sink applying a batch it prepared, which allocates
    /// nothing: with the `spill` feature, that's checked.
    #[inline]
    #[cfg_attr(not(feature = "spill"), expect(clippy::unused_self, reason = "checked to spill"))]
    pub(crate) fn applying<R>(&self, f: impl FnOnce() -> R) -> R {
        #[cfg(feature = "spill")]
        return self.budget.manager.still(f);
        #[cfg(not(feature = "spill"))]
        f()
    }

    /// Makes `store` where what doesn't fit in its budget goes.
    #[cfg(feature = "spill")]
    pub fn spill_to(&self, store: &'static dyn SpillStore) {
        self.spill.set(Some(store));
    }

    #[cfg(feature = "spill")]
    pub(crate) fn manager(&self) -> &Manager {
        &self.budget.manager
    }
}

/// Owns an `Env` from `Env::new`, and frees it when dropped, checking first
/// that nothing allocated from it is held.
pub struct OwnedEnv(NonNull<Env>);

impl Deref for OwnedEnv {
    type Target = Env;

    fn deref(&self) -> &Env {
        // SAFETY: this owns the `Env`, which lives until it's dropped.
        unsafe { self.0.as_ref() }
    }
}

impl Drop for OwnedEnv {
    #[inline(never)]
    fn drop(&mut self) {
        // Checked before anything is freed, so if it fails, what's still
        // allocated from it points to memory that's still there.
        check!(self.holds_nothing());
        // SAFETY: `free` was made for this `Env`'s allocation, and nothing
        // allocated from it is left.
        unsafe { (self.free)(self.0) }
    }
}

/// What it holds is measured only by its owner: what runs in it doesn't
/// decide by how much memory is left, which its manager alone does.
impl OwnedEnv {
    /// The bytes allocated now, this included.
    pub fn used(&self) -> usize {
        self.budget.used.get()
    }

    /// The most bytes allocated at once.
    pub fn peak(&self) -> usize {
        self.budget.peak.get()
    }

    /// Frees the `Env` if nothing allocated from it is held, and keeps it
    /// for good if something is, which can then still use it: for a
    /// thread's `Env`, which may hold what's kept for the whole program.
    fn release(self) {
        if !self.holds_nothing() {
            core::mem::forget(self);
        }
    }
}

/// Holds an `Env` for a thread, and frees it as `OwnedEnv::release` does
/// when the thread exits.
pub struct ThreadEnv(core::mem::ManuallyDrop<OwnedEnv>);

impl ThreadEnv {
    /// An `Env` from the system's heap, without a limit.
    ///
    /// # Safety
    ///
    /// It must be kept in a thread-local, as `get` lends the `Env` for
    /// as long as the thread lives, under `Env::current`'s rule.
    pub unsafe fn new() -> ThreadEnv {
        let memory = Memory::new(crate::allocator::Heap, usize::MAX);
        let Ok(env) = Env::new(memory) else { failed!() };
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
        let base = (&raw const (*allocation.as_ptr()).base).read();
        base.deallocate(allocation.cast(), Layout::new::<Allocation<A>>());
    }
}

impl Budget {
    /// Allocates `layout`, zeroed if `zeroed`, if it fits within the limit,
    /// and counts it.
    fn charge(&self, layout: Layout, zeroed: bool) -> Result<NonNull<u8>, AllocError> {
        let after =
            || self.used.get().checked_add(layout.size()).filter(|&used| used <= self.limit);
        #[cfg(feature = "spill")]
        self.manager.check_may_allocate();
        #[cfg(feature = "spill")]
        if after().is_none() {
            // SAFETY: the `Env` this is in outlives it.
            let env = unsafe { self.env.as_ref() };
            self.manager.make_room(env, || after().is_some());
        }
        let used = after().ok_or(AllocError)?;
        // SAFETY: the allocator lives in the `Env`'s allocation, as this does.
        let base = unsafe { self.base.as_ref() };
        let ptr = if zeroed { base.allocate_zeroed(layout) } else { base.allocate(layout) }?;
        self.used.set(used);
        self.peak.set(self.peak.get().max(used));
        #[cfg(feature = "spill")]
        self.manager.allocated(layout.size());
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
        #[cfg(feature = "spill")]
        self.manager.freed(layout.size());
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
        let env = Env::new(Memory::new(Heap, 2000)).unwrap();
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
        assert!(Env::new(Memory::new(Heap, 16)).is_err());
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

    static KEPT: Arena = Arena::new();

    static OUTLIVED: Arena = Arena::new();

    #[test]
    #[should_panic(expected = "holds_nothing")]
    fn checks_nothing_outlives_it() {
        let env = Env::new(Memory::new(Bump(&OUTLIVED), usize::MAX)).unwrap();
        // Forgotten, so never freed through the `Env` the check keeps.
        core::mem::forget(Buffer::allocate(&env, 8).unwrap());
        drop(env);
    }

    #[test]
    fn freed_once_nothing_is_held() {
        let env = Env::new(Memory::new(Heap, usize::MAX)).unwrap();
        let empty = env.used();
        drop(Buffer::allocate_in(&env, MemoryKind::Small, 8).unwrap());
        assert!(env.holds_nothing());
        assert!(env.used() > empty);
        drop(env);
    }

    #[test]
    fn kept_while_something_is_held() {
        let env = Env::new(Memory::new(Bump(&KEPT), usize::MAX)).unwrap();
        let buffer = Buffer::allocate(&env, 8).unwrap();
        assert!(!env.holds_nothing());
        // Kept for good, as the buffer still points to it.
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
