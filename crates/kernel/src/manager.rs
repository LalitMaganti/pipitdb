//! `Manager`: makes room when an allocation doesn't fit in an `Env`'s
//! budget, by spilling the sinks that aren't consuming, the largest first.
//!
//! It's the one part of a query that decides by how much memory is left.
//! A sink learns its budget is full only when one of its allocations fails
//! with nothing else left to spill, and then spills itself and consumes the
//! batch again.
//!
//! So that's always possible, a sink allocates only while it prepares a
//! batch, before `Env::applying`, and changes nothing then that applying
//! the batch wouldn't. Applying it and spilling allocate nothing, which the
//! budget checks.

#![allow(unsafe_code)]

use core::cell::Cell;
use core::ptr::NonNull;

use crate::env::Env;
use crate::error::Error;

/// Writes the local state of a sink to its `Env`'s spill store and frees
/// it, returning whether it held anything to write: `Sink::spill`, erased.
pub(crate) type SpillFn = unsafe fn(NonNull<()>, &Env, NonNull<u8>) -> Result<bool, Error>;

/// The sinks whose pipelines are running.
pub(crate) struct Manager {
    /// The newest registered.
    first: Cell<Option<NonNull<Entry>>>,
    /// The one consuming or spilling, which what's allocated and freed
    /// counts toward.
    current: Cell<Option<NonNull<Entry>>>,
    /// Whether nothing may allocate: a sink is applying a batch, or
    /// spilling.
    still: Cell<bool>,
}

/// A sink's local state, registered while its pipeline runs.
pub(crate) struct Entry {
    /// The sink.
    sink: NonNull<()>,
    /// Its local state.
    local: NonNull<u8>,
    /// Its `spill`.
    spill: SpillFn,
    /// The entry registered after it, newer.
    newer: Cell<Option<NonNull<Entry>>>,
    /// The entry registered before it, older.
    older: Cell<Option<NonNull<Entry>>>,
    /// Whether its sink is consuming or spilling, so it can't be spilled.
    busy: Cell<bool>,
    /// Whether the room being made has tried spilling it.
    tried: Cell<bool>,
    /// The bytes allocated less those freed while it was busy.
    bytes: Cell<usize>,
    /// What spilling it to make room failed with, which it returns next.
    failed: Cell<Option<Error>>,
}

impl Manager {
    pub(crate) const fn new() -> Manager {
        Manager { first: Cell::new(None), current: Cell::new(None), still: Cell::new(false) }
    }

    /// Registers `entry` until what's returned is dropped.
    pub(crate) fn register<'e>(&'e self, entry: &'e Entry) -> Registered<'e> {
        let at = NonNull::from(entry);
        entry.older.set(self.first.get());
        if let Some(older) = self.first.get() {
            // SAFETY: a registered entry lives until it's unregistered.
            unsafe { older.as_ref() }.newer.set(Some(at));
        }
        self.first.set(Some(at));
        Registered { manager: self, entry }
    }

    /// Fails a check if nothing may allocate now.
    pub(crate) fn check_may_allocate(&self) {
        check!(!self.still.get());
    }

    /// Runs `f`, which allocates nothing.
    pub(crate) fn still<R>(&self, f: impl FnOnce() -> R) -> R {
        let _still = Still { manager: self, before: self.still.replace(true) };
        f()
    }

    /// Counts `bytes` allocated toward the entry busy, if any.
    pub(crate) fn allocated(&self, bytes: usize) {
        if let Some(entry) = self.current.get() {
            // SAFETY: the current entry is registered.
            let entry = unsafe { entry.as_ref() };
            entry.bytes.set(entry.bytes.get() + bytes);
        }
    }

    /// Counts `bytes` freed toward the entry busy, if any.
    pub(crate) fn freed(&self, bytes: usize) {
        if let Some(entry) = self.current.get() {
            // SAFETY: the current entry is registered.
            let entry = unsafe { entry.as_ref() };
            entry.bytes.set(entry.bytes.get().saturating_sub(bytes));
        }
    }

    /// Spills the sinks that aren't busy, each at most once, the one holding
    /// the most first, until `fits`.
    pub(crate) fn make_room(&self, env: &Env, fits: impl Fn() -> bool) {
        while !fits() {
            let Some(entry) = self.largest_idle() else { break };
            // SAFETY: a registered entry lives until it's unregistered, which
            // spilling it doesn't do.
            let entry = unsafe { entry.as_ref() };
            entry.tried.set(true);
            if let Err(error) = self.while_busy(entry, || entry.spill_now(env)) {
                entry.failed.set(Some(error));
            }
        }
        let mut at = self.first.get();
        while let Some(entry) = at {
            // SAFETY: as above.
            let entry = unsafe { entry.as_ref() };
            entry.tried.set(false);
            at = entry.older.get();
        }
    }

    /// The entry that isn't busy, tried or failed holding the most bytes, if
    /// any holds some.
    fn largest_idle(&self) -> Option<NonNull<Entry>> {
        let (mut largest, mut most) = (None, 0);
        let mut at = self.first.get();
        while let Some(pointer) = at {
            // SAFETY: a registered entry lives until it's unregistered.
            let entry = unsafe { pointer.as_ref() };
            let idle = !entry.busy.get() && !entry.tried.get() && entry.failed.get().is_none();
            if idle && entry.bytes.get() > most {
                (largest, most) = (Some(pointer), entry.bytes.get());
            }
            at = entry.older.get();
        }
        largest
    }

    /// Runs `f` with `entry` busy, and current.
    fn while_busy<R>(&self, entry: &Entry, f: impl FnOnce() -> R) -> R {
        check!(!entry.busy.get());
        let current = self.current.replace(Some(NonNull::from(entry)));
        entry.busy.set(true);
        let _busy = Busy { manager: self, entry, current };
        f()
    }
}

/// Ends `Manager::still` when dropped, even by a panic.
struct Still<'m> {
    manager: &'m Manager,
    /// Whether nothing could allocate before.
    before: bool,
}

impl Drop for Still<'_> {
    fn drop(&mut self) {
        self.manager.still.set(self.before);
    }
}

/// Ends an entry's being busy when dropped, even by a panic.
struct Busy<'e> {
    manager: &'e Manager,
    entry: &'e Entry,
    /// The entry current before.
    current: Option<NonNull<Entry>>,
}

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.entry.busy.set(false);
        self.manager.current.set(self.current);
    }
}

impl Entry {
    /// An entry for `local`, a local state of `sink`, whose `spill` is
    /// `spill`.
    ///
    /// # Safety
    ///
    /// `spill` must take `sink` and `local`, and both must live as long as
    /// the entry is registered, and be used only through it meanwhile.
    pub(crate) unsafe fn new(sink: NonNull<()>, local: NonNull<u8>, spill: SpillFn) -> Entry {
        Entry {
            sink,
            local,
            spill,
            newer: Cell::new(None),
            older: Cell::new(None),
            busy: Cell::new(false),
            tried: Cell::new(false),
            bytes: Cell::new(0),
            failed: Cell::new(None),
        }
    }

    /// Spills it, which allocates nothing.
    fn spill_now(&self, env: &Env) -> Result<bool, Error> {
        // SAFETY: as `new` requires, and busy, so nothing else uses `local`.
        env.manager().still(|| unsafe { (self.spill)(self.sink, env, self.local) })
    }
}

/// An entry while it's registered.
pub(crate) struct Registered<'e> {
    manager: &'e Manager,
    entry: &'e Entry,
}

impl Registered<'_> {
    /// Runs `consume` with the entry busy. If it fails for want of memory,
    /// which it can only do while preparing, the sink spills and it runs
    /// again, until the sink holds nothing to spill.
    pub(crate) fn consume(
        &self,
        env: &Env,
        mut consume: impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        loop {
            if let Some(error) = self.entry.failed.get() {
                return Err(error);
            }
            match self.manager.while_busy(self.entry, &mut consume) {
                Err(Error::OutOfMemory) => {
                    if !self.manager.while_busy(self.entry, || self.entry.spill_now(env))? {
                        return Err(Error::OutOfMemory);
                    }
                }
                consumed => return consumed,
            }
        }
    }

    /// Runs `f` with the entry busy, so it isn't spilled meanwhile, failing
    /// if spilling it failed before.
    pub(crate) fn run<R>(&self, f: impl FnOnce() -> Result<R, Error>) -> Result<R, Error> {
        if let Some(error) = self.entry.failed.get() {
            return Err(error);
        }
        self.manager.while_busy(self.entry, f)
    }
}

impl Drop for Registered<'_> {
    fn drop(&mut self) {
        let (entry, manager) = (self.entry, self.manager);
        let (newer, older) = (entry.newer.get(), entry.older.get());
        match newer {
            // SAFETY: registered entries live until they're unregistered.
            Some(newer) => unsafe { newer.as_ref() }.older.set(older),
            None => manager.first.set(older),
        }
        if let Some(older) = older {
            // SAFETY: as above.
            unsafe { older.as_ref() }.newer.set(newer);
        }
        check!(manager.current.get() != Some(NonNull::from(entry)));
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::cell::Cell;

    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::{ColumnView, DataType};
    use crate::containers::SmallVec;
    use crate::context::Context;
    use crate::env::{Env, Memory, OwnedEnv};
    use crate::error::Error;
    use crate::pipeline::Pipeline;
    use crate::pipeline::tests::after_breaker;
    use crate::row_batch::RowBatch;
    use crate::step::{Breaker, DynSource, Source};

    /// Batches of a row each, numbered, every fourth read into `bytes` of
    /// memory held until the next.
    struct Reads {
        batches: i64,
        bytes: usize,
    }

    impl Source for Reads {
        const NAME: &'static str = "Reads";
        type State = (i64, Option<Buffer>);

        fn new_state(&self, _: &mut Context) -> Result<Self::State, Error> {
            Ok((0, None))
        }

        fn next(
            &self,
            context: &mut Context,
            (i, read): &mut Self::State,
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            *read = None;
            if *i == self.batches {
                return Ok(false);
            }
            let bytes = if *i % 4 == 3 { self.bytes } else { 0 };
            *read = Some(Buffer::allocate(context.env(), bytes)?);
            let mut values = context.values_buffer(8)?;
            values.as_mut_slice::<i64>()[0] = *i;
            batch.reset(1);
            let column = ColumnView::new(context, DataType::Int64, values, None)?;
            assert!(batch.push_column(column).is_ok());
            *i += 1;
            Ok(true)
        }
    }

    /// Holds `bytes` of memory for each row, until it spills, which frees
    /// them; then serves how many rows it had, how often it spilled, and how
    /// often to make room for something else.
    struct Hoard {
        bytes: usize,
        /// What spilling fails with, if anything.
        fails: Option<Error>,
        /// Whether it can spill.
        spills: bool,
        /// Whether it allocates applying a batch, which fails a check.
        late: bool,
    }

    #[derive(Default)]
    struct Hoarded {
        held: Vec<Buffer>,
        rows: i64,
        spills: i64,
        /// Spills not after its own allocation failed.
        made_room: i64,
        /// Whether its last allocation failed.
        failed: bool,
        /// Whether it was consuming, to check it's never spilled meanwhile.
        consuming: Cell<bool>,
    }

    impl Breaker for Hoard {
        const NAME: &'static str = "Hoard";
        type State = Hoarded;

        fn new_state(&self, _: &mut Context) -> Result<Hoarded, Error> {
            Ok(Hoarded::default())
        }

        fn consume(
            &self,
            context: &mut Context,
            hoarded: &mut Hoarded,
            input: &RowBatch,
        ) -> Result<(), Error> {
            hoarded.consuming.set(true);
            let held = Buffer::allocate(context.env(), self.bytes);
            hoarded.consuming.set(false);
            hoarded.failed = held.is_err();
            let held = held?;
            context.env().applying(|| {
                if self.late {
                    hoarded.held.push(Buffer::allocate(context.env(), 8)?);
                }
                hoarded.held.push(held);
                hoarded.rows += i64::from(input.row_count());
                Ok(())
            })
        }

        fn finalize(&self, _: &mut Context, _: &mut Hoarded) -> Result<(), Error> {
            Ok(())
        }

        fn serve(
            &self,
            context: &mut Context,
            hoarded: &mut Hoarded,
            output: &mut RowBatch,
        ) -> Result<bool, Error> {
            let mut values = context.values_buffer(24)?;
            let counts = [hoarded.rows, hoarded.spills, hoarded.made_room];
            values.as_mut_slice::<i64>()[..3].copy_from_slice(&counts);
            output.reset(3);
            let column = ColumnView::new(context, DataType::Int64, values, None)?;
            assert!(output.push_column(column).is_ok());
            Ok(false)
        }

        fn spill(&self, _: &Env, hoarded: &mut Hoarded) -> Result<bool, Error> {
            assert!(!hoarded.consuming.get());
            if let Some(error) = self.fails {
                return Err(error);
            }
            if !self.spills || hoarded.held.is_empty() {
                return Ok(false);
            }
            hoarded.held.clear();
            hoarded.spills += 1;
            hoarded.made_room += i64::from(!hoarded.failed);
            Ok(true)
        }
    }

    /// What `hoard` serves, reading `reads` within `limit` bytes.
    fn run(limit: usize, reads: Reads, hoard: Hoard) -> Result<[i64; 3], Error> {
        let setup = Env::current();
        let input =
            Pipeline::new(DynSource::new(setup, reads).unwrap(), SmallVec::new(setup, 1).unwrap());
        let (sinks, pipeline) = after_breaker(input, hoard, []);
        let env: OwnedEnv = Env::new(Memory::new(Heap, limit)).unwrap();
        let mut execution = pipeline.start_with_sinks(&env, None, &sinks, None)?;
        let mut batch = RowBatch::new();
        execution.next(&mut batch)?;
        let column = batch.column(0).int64s();
        Ok([column[0], column[1], column[2]])
    }

    const KB: usize = 1 << 10;

    #[test]
    fn spills_a_sink_not_consuming_to_make_room_for_a_read() {
        let reads = Reads { batches: 40, bytes: 64 * KB };
        let hoard = Hoard { bytes: 16 * KB, fails: None, spills: true, late: false };
        // The reads fill the budget, so the hoard is spilled while one is
        // read, not while it consumes.
        let [rows, _, made_room] = run(256 * KB, reads, hoard).unwrap();
        assert_eq!(rows, 40);
        assert!(made_room > 0);
    }

    #[test]
    fn spills_a_sink_whose_own_allocation_does_not_fit_and_consumes_again() {
        let reads = Reads { batches: 40, bytes: 0 };
        let hoard = Hoard { bytes: 32 * KB, fails: None, spills: true, late: false };
        let [rows, spills, made_room] = run(256 * KB, reads, hoard).unwrap();
        assert_eq!((rows, made_room), (40, 0));
        assert!(spills > 0);
    }

    #[test]
    fn runs_out_of_memory_with_nothing_to_spill() {
        let reads = Reads { batches: 40, bytes: 0 };
        let hoard = Hoard { bytes: 32 * KB, fails: None, spills: false, late: false };
        assert_eq!(run(256 * KB, reads, hoard), Err(Error::OutOfMemory));
    }

    #[test]
    fn fails_with_what_spilling_failed_with() {
        let reads = Reads { batches: 40, bytes: 64 * KB };
        let hoard = Hoard { bytes: 16 * KB, fails: Some(Error::Io), spills: true, late: false };
        assert_eq!(run(256 * KB, reads, hoard), Err(Error::Io));
    }

    #[test]
    #[should_panic(expected = "still")]
    fn checks_nothing_allocates_applying_a_batch() {
        let reads = Reads { batches: 1, bytes: 0 };
        let hoard = Hoard { bytes: 8, fails: None, spills: true, late: true };
        let _ = run(usize::MAX, reads, hoard);
    }

    #[test]
    fn spills_nothing_with_room() {
        let reads = Reads { batches: 40, bytes: 64 * KB };
        let hoard = Hoard { bytes: 16 * KB, fails: None, spills: true, late: false };
        assert_eq!(run(usize::MAX, reads, hoard), Ok([40, 0, 0]));
    }
}
