//! How an aggregation spills: it writes each group's keys and states to a
//! partition by its keys' hash and forgets them, and once every row is in,
//! aggregates a partition at a time by merging the states written.

#![allow(unsafe_code)]

use core::ptr::NonNull;

use crate::aggregate_function::{GroupStates, StringExtreme, make_aggregate_function};
use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType};
use crate::containers::{Blocks, SmallVec};
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;
use crate::group_table::{GroupTable, KEYS_MAX};
use crate::manager::Entry;
use crate::murmur_hash;
use crate::row_batch::BATCH_COLUMNS_MAX;
use crate::spill::{LogId, Rows, SpillStore, read_batch, write_batch};
use crate::string_pool::{Id, StringPool};

use super::{HashAggregate, Prepared, State, string_extreme};

/// How many partitions groups are written to, by four bits of their keys'
/// hash.
const PARTITIONS: usize = 16;

/// How many times a partition can be split, by the next four bits each.
const LEVELS: u32 = u64::BITS / 4;

/// How many groups of a partition are written as a batch, at most.
const CHUNK_GROUPS: usize = 256;

/// The partitions an aggregation's groups are written to.
pub(super) struct Spilled {
    /// Which four bits of a group's hash, from the top, pick its partition.
    level: u32,
    /// Each partition being written at `level`, once it is.
    writing: [Option<Partition>; PARTITIONS],
    /// The partitions written and sealed, to aggregate once every row is in.
    pending: SmallVec<Partition>,
    /// The store they're in, once any is.
    store: Option<&'static dyn SpillStore>,
    /// Each batch of states being merged, once any is.
    partials: Option<Partials>,
}

/// A partition's log, of batches of groups' keys and states.
#[derive(Clone, Copy)]
struct Partition {
    log: LogId,
    /// How many batches it holds.
    batches: u32,
    /// The level it was written at.
    level: u32,
}

/// A batch of groups' states read from a partition, laid out as a table's.
struct Partials {
    /// Each group's states, every aggregate's.
    states: Blocks<i64>,
    /// For each aggregate, which groups have seen a value.
    seen: SmallVec<Blocks<u8>>,
}

impl Spilled {
    pub(super) fn new(env: &Env) -> Result<Spilled, Error> {
        let pending = SmallVec::new(env, 1 << 24)?;
        Ok(Spilled { level: 0, writing: [None; PARTITIONS], pending, store: None, partials: None })
    }

    /// Whether any partition is left to aggregate.
    pub(super) fn is_pending(&self) -> bool {
        !self.pending.is_empty()
    }
}

impl Drop for Spilled {
    fn drop(&mut self) {
        let Some(store) = self.store else { return };
        for partition in self.writing.iter().flatten().chain(self.pending.iter()) {
            store.delete(partition.log);
        }
    }
}

/// What a column written for a group is.
#[derive(Clone, Copy)]
enum Written {
    /// Key `k`.
    Key(usize),
    /// Word `w` of its states, null if it's an aggregate's first and that
    /// aggregate, `seen`, tracks which have seen a value, and it hasn't.
    State { w: usize, seen: Option<usize> },
    /// Word `w` of its states, a string `MIN` or `MAX`'s.
    Best(usize),
}

/// The columns written for a group: its keys, then each aggregate's
/// states, as one column a word or a string column.
fn written(aggregate: &HashAggregate) -> ([Written; BATCH_COLUMNS_MAX as usize], usize) {
    let mut columns = [Written::Key(0); BATCH_COLUMNS_MAX as usize];
    let mut width = 0;
    let mut add = |column| {
        *at_mut!(columns, width) = column;
        width += 1;
    };
    for k in 0..aggregate.spec.key_columns.len() {
        add(Written::Key(k));
    }
    let mut offset = 0;
    for (a, call) in aggregate.spec.aggregates.iter().enumerate() {
        if string_extreme(*call).is_some() {
            add(Written::Best(offset));
            offset += StringExtreme::STATE_WORDS;
            continue;
        }
        let function = make_aggregate_function(call.function, call.input_type);
        for w in offset..offset + function.state_words() {
            let seen = (w == offset && function.tracks_seen()).then_some(a);
            add(Written::State { w, seen });
        }
        offset += function.state_words();
    }
    (columns, width)
}

/// Groups of a table, as `write_batch` writes them.
struct Groups<'s> {
    state: &'s State,
    columns: &'s [Written],
    numbers: &'s [u32],
}

impl Groups<'_> {
    fn group(&self, row: u32) -> u32 {
        *at!(self.numbers, row as usize)
    }

    /// Row `row`'s string of string column `column`, if it isn't null.
    fn some_string(&self, column: usize, row: u32) -> Option<&[u8]> {
        let (table, group) = (&self.state.groups, self.group(row));
        match *at!(self.columns, column) {
            Written::Key(k) => table.string(table.key(group, k)?),
            Written::Best(w) => {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "an id"
                )]
                let id = Id::raw(table.payload_word(group, w) as u32);
                self.state.extremes.as_ref()?.get(id)
            }
            Written::State { .. } => failed!(),
        }
    }
}

impl Rows for Groups<'_> {
    fn count(&self) -> u32 {
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
        let count = self.numbers.len() as u32;
        count
    }

    fn width(&self) -> usize {
        self.columns.len()
    }

    fn column(&self, column: usize) -> (DataType, bool) {
        let table = &self.state.groups;
        match *at!(self.columns, column) {
            Written::Key(k) if table.is_string(k) => (DataType::String, true),
            Written::Key(k) => (DataType::Int64, table.is_nullable(k)),
            Written::Best(_) => (DataType::String, true),
            Written::State { seen, .. } => (DataType::Int64, seen.is_some()),
        }
    }

    fn is_null(&self, column: usize, row: u32) -> bool {
        let (table, group) = (&self.state.groups, self.group(row));
        match *at!(self.columns, column) {
            Written::Key(k) if !table.is_string(k) => table.key(group, k).is_none(),
            Written::Key(_) | Written::Best(_) => self.some_string(column, row).is_none(),
            Written::State { seen, .. } => {
                seen.is_some_and(|a| !at!(self.state.seen, a).seen(group))
            }
        }
    }

    fn word(&self, column: usize, row: u32) -> u64 {
        let (table, group) = (&self.state.groups, self.group(row));
        match *at!(self.columns, column) {
            Written::Key(k) => table.key(group, k).unwrap_or(0).cast_unsigned(),
            Written::State { w, .. } => table.payload_word(group, w).cast_unsigned(),
            Written::Best(_) => failed!(),
        }
    }

    fn string(&self, column: usize, row: u32) -> &[u8] {
        self.some_string(column, row).unwrap_or_default()
    }
}

/// The partition group `group` of `table` is written to at `level`: four
/// bits of a hash of its keys' values, strings' bytes rather than their
/// words, which differ from pool to pool.
fn partition(table: &GroupTable, keys: usize, group: u32, level: u32) -> usize {
    let mut hash = murmur_hash::SEED;
    for k in 0..keys {
        let word = match table.key(group, k) {
            Some(word) if table.is_string(k) => {
                table.string(word).map_or(u64::MAX, murmur_hash::bytes)
            }
            Some(word) => word.cast_unsigned(),
            None => u64::MAX,
        };
        hash = murmur_hash::combine(hash, word);
    }
    (murmur_hash::mix(hash) >> (u64::BITS - 4 * (level + 1)) & 15) as usize
}

impl HashAggregate {
    /// Whether it can spill: it has keys, and no `DISTINCT` aggregate, as
    /// the values each group has had can't be merged.
    pub(super) fn spills(&self) -> bool {
        !self.spec.key_columns.is_empty() && self.spec.aggregates.iter().all(|call| !call.distinct)
    }

    /// Writes `state`'s groups a batch has been applied to to partitions,
    /// and forgets every group, allocating nothing: `Breaker::spill`.
    pub(super) fn spill_groups(&self, env: &Env, state: &mut State) -> Result<bool, Error> {
        let count = state.applied;
        let Some(store) = env.spill() else { return Ok(false) };
        if !self.spills() || count == 0 || state.spilled.level == LEVELS {
            return Ok(false);
        }
        state.spilled.store = Some(store);
        let (columns, width) = written(self);
        let columns = at!(columns, ..width);
        let keys = self.spec.key_columns.len();
        let level = state.spilled.level;
        let mut chunks = [[0_u32; CHUNK_GROUPS]; PARTITIONS];
        let mut filled = [0; PARTITIONS];
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        for group in 0..count as u32 {
            let p = partition(&state.groups, keys, group, level);
            let chunk = at_mut!(chunks, p);
            *at_mut!(chunk, *at!(filled, p)) = group;
            *at_mut!(filled, p) += 1;
            if *at!(filled, p) == CHUNK_GROUPS {
                write_groups(store, state, columns, p, at!(chunks, p))?;
                *at_mut!(filled, p) = 0;
            }
        }
        for (p, (chunk, &filled)) in chunks.iter().zip(&filled).enumerate() {
            if filled > 0 {
                write_groups(store, state, columns, p, at!(chunk, ..filled))?;
            }
        }
        forget(env, state);
        Ok(true)
    }

    /// Spills what `state` holds if it's spilled before, so every group is in
    /// a partition, and seals the partitions it's written.
    pub(super) fn seal(&self, env: &Env, state: &mut State) -> Result<(), Error> {
        if state.spilled.writing.iter().all(Option::is_none) {
            return Ok(());
        }
        self.spill_groups(env, state)?;
        let Some(store) = state.spilled.store else { failed!() };
        for writing in &mut state.spilled.writing {
            let Some(partition) = *writing else { continue };
            // Pushed before it's sealed, so it's deleted if either fails.
            state.spilled.pending.push(partition)?;
            *writing = None;
            store.seal(partition.log)?;
        }
        Ok(())
    }

    /// Forgets the groups `state` has output, and aggregates the next
    /// partition left into it, merging the states written. If it doesn't
    /// fit, it's split into partitions at the next level, and `state` holds
    /// no groups.
    pub(super) fn merge_next(&self, context: &mut Context, state: &mut State) -> Result<(), Error> {
        forget(context.env(), state);
        let (Some(partition), Some(store)) = (state.spilled.pending.pop(), state.spilled.store)
        else {
            failed!()
        };
        let merged = self.merge(context, state, store, partition);
        store.delete(partition.log);
        merged?;
        self.seal(context.env(), state)
    }

    fn merge(
        &self,
        context: &mut Context,
        state: &mut State,
        store: &'static dyn SpillStore,
        partition: Partition,
    ) -> Result<(), Error> {
        let env = context.env();
        state.spilled.level = partition.level + 1;
        if state.spilled.partials.is_none() {
            let mut states = Blocks::new(env, self.state_words.max(1))?;
            states.reserve(env, CHUNK_GROUPS)?;
            let mut seen = SmallVec::fixed(env, self.spec.aggregates.len())?;
            for _ in 0..self.spec.aggregates.len() {
                let mut bytes = Blocks::new(env, 1)?;
                bytes.reserve(env, CHUNK_GROUPS)?;
                seen.push(bytes)?;
            }
            state.spilled.partials = Some(Partials { states, seen });
        }
        // While it's merging, the state can be spilled to make room, as when
        // consuming, so it's used only through the entry meanwhile.
        let local = NonNull::from(&mut *state);
        // SAFETY: `spill_state` takes this and the state, which outlive the
        // entry, and the state is used only through it while it's registered.
        let entry = unsafe { Entry::new(NonNull::from(self).cast(), local.cast(), spill_state) };
        let registered = env.manager().register(&entry);
        let mut at = 0;
        for _ in 0..partition.batches {
            let mut columns = SmallVec::fixed(env, BATCH_COLUMNS_MAX as usize)?;
            let (rows, next) = read_batch(context, store, partition.log, at, |_, column| {
                Ok(columns.push(column)?)
            })?;
            at = next;
            registered.consume(env, || {
                // SAFETY: the entry is busy, so the manager doesn't use it.
                let state = unsafe { &mut *local.as_ptr() };
                self.merge_batch(context, state, &columns, rows as usize)
            })?;
        }
        Ok(())
    }

    /// Merges the `rows` groups of `columns`, as `written` lays them out,
    /// into `state`'s: prepared, then applied.
    fn merge_batch(
        &self,
        context: &mut Context,
        state: &mut State,
        columns: &[ColumnView],
        rows: usize,
    ) -> Result<(), Error> {
        let (numbers, from) = self.prepare_merge(context, state, columns, rows)?;
        let (numbers, from) =
            (at!(numbers.as_slice::<u32>(), ..rows), at!(from.as_slice(), ..rows));
        let merged = context.env().applying(|| self.apply_merge(state, numbers, from));
        state.applied = state.groups.len();
        for prepared in state.prepared.iter_mut() {
            *prepared = None;
        }
        merged
    }

    /// Finds or adds each group's, and lays out its states as a table's,
    /// allocating all merging them needs, changing no group's state.
    /// Returns each one's group, and its place in the batch.
    fn prepare_merge(
        &self,
        context: &mut Context,
        state: &mut State,
        columns: &[ColumnView],
        rows: usize,
    ) -> Result<(Buffer, Buffer), Error> {
        let keys_count = self.spec.key_columns.len();
        let mut keys = [at!(columns, 0); KEYS_MAX];
        for (key, column) in keys.iter_mut().zip(columns) {
            *key = column;
        }
        let mut numbers = context.indices_buffer(rows * 4)?;
        let groups = at_mut!(numbers.as_mut_slice::<u32>(), ..rows);
        state.groups.find_or_create(context, at!(keys, ..keys_count), &[], groups)?;
        let mut from = context.indices_buffer(rows * 4)?;
        #[expect(clippy::cast_possible_truncation, reason = "a batch's row")]
        for (i, place) in at_mut!(from.as_mut_slice::<u32>(), ..rows).iter_mut().enumerate() {
            *place = i as u32;
        }
        let group_count = state.groups.len();
        let mut c = keys_count;
        let State { groups: table, total, seen, extremes, prepared, spilled, .. } = state;
        let Some(partials) = &mut spilled.partials else { failed!() };
        let total = [total.as_mut_ptr()];
        let base = self.state_base(table, &total);
        let mut offset = 0;
        for (a, call) in self.spec.aggregates.iter().enumerate() {
            if let Some(extreme) = string_extreme(*call) {
                let states = Self::states(base, offset, None);
                offset += StringExtreme::STATE_WORDS;
                let pool = StringPool::made(extremes, context.env())?;
                let column = at!(columns, c);
                c += 1;
                let (found, count) =
                    extreme.prepare((context, pool), column, &[], rows, Some(groups), states)?;
                *at_mut!(prepared, a) = Some(Prepared::Strings(found, count));
                continue;
            }
            let function = make_aggregate_function(call.function, call.input_type);
            let first = at!(columns, c);
            for w in offset..offset + function.state_words() {
                let words = at!(columns, c).int64s();
                for (i, &word) in at!(words, ..rows).iter().enumerate() {
                    // SAFETY: room for a batch written was reserved.
                    unsafe { *partials.states.record_ptr(i).add(w) = word };
                }
                c += 1;
            }
            offset += function.state_words();
            if function.tracks_seen() {
                let bytes = at_mut!(partials.seen, a);
                let mut unseen = false;
                #[expect(clippy::cast_possible_truncation, reason = "a batch's row")]
                for i in 0..rows {
                    let null = first.is_null(i as u32);
                    unseen |= null;
                    // SAFETY: as above.
                    unsafe { *bytes.record_ptr(i) = u8::from(!null) };
                }
                at_mut!(seen, a).get(context.env(), group_count, unseen)?;
            }
        }
        Ok((numbers, from))
    }

    /// Merges the states `prepare_merge` laid out into those of their
    /// groups, `numbers`, allocating nothing.
    fn apply_merge(&self, state: &mut State, numbers: &[u32], from: &[u32]) -> Result<(), Error> {
        let State { groups: table, total, seen, prepared, spilled, .. } = state;
        let Some(partials) = &spilled.partials else { failed!() };
        let total = [total.as_mut_ptr()];
        let base = self.state_base(table, &total);
        let mut overflow = false;
        let mut offset = 0;
        for (a, call) in self.spec.aggregates.iter().enumerate() {
            if string_extreme(*call).is_some() {
                let states = Self::states(base, offset, None);
                offset += StringExtreme::STATE_WORDS;
                let Some(Prepared::Strings(found, count)) = at!(prepared, a) else { failed!() };
                StringExtreme::apply(found, *count, states);
                continue;
            }
            let function = make_aggregate_function(call.function, call.input_type);
            let tracks = function.tracks_seen();
            let states =
                Self::states(base, offset, if tracks { at!(seen, a).tracked() } else { None });
            let other = GroupStates {
                blocks: partials.states.starts(),
                stride: self.state_words.max(1),
                offset,
                seen: tracks.then(|| at!(partials.seen, a).starts()),
            };
            offset += function.state_words();
            function.merge(states, numbers, &other, from, &mut overflow);
        }
        if overflow { Err(Error::Overflow) } else { Ok(()) }
    }
}

/// Forgets `state`'s groups, freeing what they took, allocating nothing.
fn forget(env: &Env, state: &mut State) {
    state.groups.free(env);
    for seen in state.seen.iter_mut() {
        seen.free(env);
    }
    state.extremes = None;
    state.emitted = 0;
    state.applied = 0;
}

/// Writes `groups` of `state` to partition `p` at its level, as `columns`.
fn write_groups(
    store: &dyn SpillStore,
    state: &mut State,
    columns: &[Written],
    p: usize,
    groups: &[u32],
) -> Result<(), Error> {
    let level = state.spilled.level;
    let partition = if let Some(partition) = *at!(state.spilled.writing, p) {
        partition
    } else {
        let partition = Partition { log: store.create()?, batches: 0, level };
        *at_mut!(state.spilled.writing, p) = Some(partition);
        partition
    };
    write_batch(store, partition.log, &Groups { state, columns, numbers: groups })?;
    if let Some(partition) = at_mut!(state.spilled.writing, p) {
        partition.batches += 1;
    }
    Ok(())
}

/// `HashAggregate::spill_groups` of the aggregation at `sink` and its state
/// at `state`, as the manager calls it.
///
/// # Safety
///
/// As `Entry::new` requires.
unsafe fn spill_state(sink: NonNull<()>, env: &Env, state: NonNull<u8>) -> Result<bool, Error> {
    // SAFETY: upheld by the caller.
    let (sink, state) =
        unsafe { (sink.cast::<HashAggregate>().as_ref(), state.cast::<State>().as_mut()) };
    sink.spill_groups(env, state)
}
