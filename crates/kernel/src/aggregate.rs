//! `AggregateOp`: aggregates, such as `COUNT(*)` and `SUM(x)`, over each
//! group of its child's rows with the same keys, or over all of them if it
//! has none; and `hash_aggregate`, the breaker that runs it.

use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::column::{Bounds, ColumnView, DataType, Forms, Validity};
use crate::context::Context;
use crate::error::Error;
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::plan::{ColumnId, NamedColumn, Op, PlanNode};
use crate::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use crate::selection::Kept;
use crate::slow_vec::SlowVec;
use crate::step::{Breaker, DynOperator, Step};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Function {
    /// How many rows there are, or, of a column, how many aren't null: an
    /// `Int64`.
    Count,
    /// Of the values that aren't null, as are the rest: of the column's
    /// type, but `Float64` for `Avg`, and null if there are none.
    Sum,
    Min,
    Max,
    Avg,
}

impl Function {
    /// What it makes of a column of `data_type`, or `None` if it can't.
    pub fn output_type(self, data_type: Option<DataType>) -> Option<DataType> {
        match (self, data_type) {
            (Function::Count, _) => Some(DataType::Int64),
            (_, None | Some(DataType::String)) => None,
            (Function::Avg, Some(_)) => Some(DataType::Float64),
            (_, Some(data_type)) => Some(data_type),
        }
    }
}

/// An aggregate in a plan: its function, the column it reads, `None` only for
/// `COUNT(*)`, and the column it makes.
#[derive(Clone, Copy)]
pub struct Aggregate {
    pub function: Function,
    pub input: Option<ColumnId>,
    pub binding: NamedColumn,
}

/// Outputs a row for each group: its keys, then its aggregates. With no
/// keys, all rows are one group, and there's a row even if there are none.
pub struct AggregateOp {
    /// The `Int64` columns rows are grouped by.
    pub keys: SlowVec<ColumnId>,
    /// What each group's rows make.
    pub aggregates: SlowVec<Aggregate>,
}

impl<'c> Op<'c> for AggregateOp {
    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        lowering.lower(*at!(node.children, 0))?;
        let allocator = lowering.allocator();
        check!(self.keys.iter().all(|&key| lowering.query_type(key) == DataType::Int64));
        let keys = self.keys.iter().map(|&key| lowering.query_position(key));
        let keys = SlowVec::fixed_from(allocator, keys)?;
        let aggregates = self.aggregates.iter().map(|aggregate| {
            let input =
                aggregate.input.map(|id| (lowering.query_position(id), lowering.query_type(id)));
            Lowered { function: aggregate.function, input }
        });
        let aggregates = SlowVec::fixed_from(allocator, aggregates)?;
        lowering.declare_new_batches();
        for &key in self.keys.iter() {
            lowering.declare_column(key)?;
        }
        for aggregate in self.aggregates.iter() {
            lowering.declare_column(aggregate.binding.id)?;
        }
        let operator = hash_aggregate(allocator, Grouping { keys, aggregates })?;
        lowering.add_step(Step::Operator(operator))
    }

    /// Drops the aggregates nothing reads, keeping one if there are no keys:
    /// a batch with no columns has no rows.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        let any = self.aggregates.iter().any(|aggregate| needed.is_needed(aggregate.binding.id));
        let keep_first = !any && self.keys.is_empty();
        let mut first = true;
        self.aggregates.retain(|aggregate| {
            let keep = needed.is_needed(aggregate.binding.id) || (keep_first && first);
            first = false;
            keep
        });
        self.reads(needed);
        Pruned::Keep
    }

    /// A key's batch of one value is found once. With more keys, it's made
    /// flat unless they all are, so it gains only for one.
    fn accepts(&self, column: ColumnId) -> Forms {
        if *self.keys == [column] { Forms::FLAT | Forms::CONSTANT } else { Forms::FLAT }
    }

    fn reads(&self, reads: &mut Needed) {
        for &key in self.keys.iter() {
            reads.need(key);
        }
        for input in self.aggregates.iter().filter_map(|aggregate| aggregate.input) {
            reads.need(input);
        }
    }
}

/// What a breaker that aggregates is given, once lowered.
pub struct Grouping {
    /// Where the keys are in batches.
    pub keys: SlowVec<u32>,
    /// Each aggregate's function, and where its column is in batches, and of
    /// what type.
    pub aggregates: SlowVec<Lowered>,
}

#[derive(Clone, Copy)]
pub struct Lowered {
    pub function: Function,
    pub input: Option<(u32, DataType)>,
}

/// The breaker that aggregates by `grouping`.
pub fn hash_aggregate<'a>(
    allocator: &dyn Allocator,
    grouping: Grouping,
) -> Result<DynOperator<'a>, AllocError> {
    DynOperator::new(allocator, HashAggregate::new(allocator, grouping)?)
}

/// The most keys a grouping can have: one bit each says which are null.
const KEYS_MAX: usize = 64;

/// The most aggregates a grouping can have: one bit each says which have
/// seen a value.
const AGGREGATES_MAX: usize = 64;

/// The most groups an aggregation can hold.
const GROUPS_MAX: usize = 1 << 26;

/// The fewest slots a table has: a batch of new groups fits.
const SLOTS_MIN: usize = 2 * BATCH_ROWS_MAX as usize;

/// The most keys a table maps to their groups directly: 512 KB of them.
const DIRECT_MAX: usize = 1 << 17;

/// The bits of a slot that hold the top bits of its group's hash.
const SALT: u64 = 0xffff << 48;

/// What a group's keys are hashed from, and the hash of no keys.
const SEED: u64 = 0x243f_6a88_85a3_08d3;

/// Groups rows a batch at a time, as Perfetto's `HashAggregate` does: a
/// `Table` turns a batch's rows into group numbers, then each aggregate
/// updates the groups over the whole batch in a loop of its own. A batch
/// whose keys are each one value is one group, found once; and integer sums
/// are kept in one word while the bounds of what's summed say they fit.
///
/// A group's record is a word of which keys are null, the keys, a word of
/// which aggregates have seen a value, then each aggregate's state.
struct HashAggregate {
    /// What rows are grouped by, and what each group's make.
    grouping: Grouping,
    /// Where each aggregate's state starts in a record.
    offsets: SlowVec<u32>,
    /// A new group's record, but for its key: 0 for that, and for its seen
    /// word, then each aggregate's empty state.
    empty: SlowVec<i64>,
}

/// The groups so far.
struct Groups {
    /// The groups' keys and states.
    table: Table,
    /// The aggregates whose seen bits are kept, a bit each. One that skips
    /// nulls starts at its first null: until then, every group has seen a
    /// value, as a group is only made for a row.
    tracked: u64,
    /// Whether integer sums are kept in their first word alone: so long as
    /// `reach` says none can pass an `Int64`.
    narrow: bool,
    /// The most any group's integer sum could be in magnitude: the sum, over
    /// batches, of their rows times the largest bound of the columns summed.
    reach: u64,
    /// How many groups have been output.
    served: usize,
}

/// Perfetto's `GroupTable`: numbers each distinct key, in the order first
/// seen. As in DuckDB's, each slot keeps the top bits of its group's hash,
/// so keys are only compared when those match; and a group's record keeps
/// its keys together, so comparing them reads one cache line.
struct Table {
    /// A power of two many, at most two thirds full: each 0 if empty, or the
    /// top 16 bits of a group's hash and its number plus one.
    slots: SlowVec<u64>,
    /// Each group's hash, so growing never hashes again.
    hashes: SlowVec<u64>,
    /// Each group's record, `width` words, one after another.
    records: SlowVec<i64>,
    /// How many words a record takes.
    width: usize,
    /// For one key whose batches' bounds are close enough together: the
    /// group of each key from `direct_min` plus one, or 0 if not looked up
    /// yet, so most rows skip hashing.
    direct: SlowVec<u32>,
    /// The least key `direct` covers.
    direct_min: i64,
}

impl Table {
    fn new(allocator: &dyn Allocator, width: usize) -> Result<Table, AllocError> {
        Ok(Table {
            slots: SlowVec::new(allocator, 1)?,
            hashes: SlowVec::new(allocator, GROUPS_MAX)?,
            // A `SlowVec` holds at most 2^31 values.
            records: SlowVec::new(
                allocator,
                (GROUPS_MAX * width).next_power_of_two().min(1 << 31),
            )?,
            width,
            direct: SlowVec::new(allocator, 1)?,
            direct_min: 0,
        })
    }

    fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Makes room for `more` groups, keeping the slots at most two thirds
    /// full, and puts each group back by its hash.
    fn reserve(&mut self, allocator: &dyn Allocator, more: usize) -> Result<(), Error> {
        let needed = self.len() + more;
        let mut capacity = self.slots.len().max(SLOTS_MIN);
        while needed * 3 > capacity * 2 {
            capacity *= 2;
        }
        if capacity == self.slots.len() {
            return Ok(());
        }
        let mut slots = SlowVec::<u64>::zeroed(allocator, capacity)?;
        let mask = capacity - 1;
        for (group, &hash) in (0..).zip(self.hashes.iter()) {
            #[expect(clippy::cast_possible_truncation, reason = "masked")]
            let mut at = hash as usize & mask;
            while *at!(slots, at) != 0 {
                at = (at + 1) & mask;
            }
            *at_mut!(slots, at) = slot(hash, group);
        }
        self.slots = slots;
        Ok(())
    }

    /// The group whose key is `key`, its null word and keys, hashed as
    /// `hash`, added with the record `empty` but for the key if new.
    fn find_or_add(&mut self, hash: u64, key: &[i64], empty: &[i64]) -> Result<u32, Error> {
        let Table { slots, hashes, records, width, .. } = self;
        find_or_add(slots, hashes, records, *width, hash, key, empty)
    }

    /// Writes the group of each row to `numbers`, of the rows whose keys
    /// are `keys`, a word a row for each, 0 where null, and `nulls`, a word a
    /// row saying which are, unless none are. A new group's record is
    /// `empty` but for its key. `bounds` are the first key's, if known.
    fn find_or_add_rows(
        &mut self,
        context: &mut Context,
        keys: &[&[i64]],
        nulls: Option<&[u64]>,
        bounds: Option<Bounds>,
        empty: &[i64],
        numbers: &mut Numbers,
    ) -> Result<(), Error> {
        self.reserve(context.allocator(), numbers.groups.len())?;
        if let ([keys], None) = (keys, nulls) {
            if let Some(bounds) = bounds
                && self.cover(context.allocator(), bounds)?
            {
                return self.find_or_add_direct(keys, empty, numbers);
            }
            return self.find_or_add_values(keys, empty, numbers);
        }
        let mut hashes = context.values_buffer(numbers.groups.len() * 8)?;
        let hashes = hashes.as_mut_slice::<u64>();
        for (hash, &key) in hashes.iter_mut().zip(*at!(keys, 0)) {
            *hash = mix(SEED, key.cast_unsigned());
        }
        for keys in at!(keys, 1..) {
            for (hash, &key) in hashes.iter_mut().zip(*keys) {
                *hash = mix(*hash, key.cast_unsigned());
            }
        }
        if let Some(nulls) = nulls {
            for (hash, &null) in hashes.iter_mut().zip(nulls) {
                *hash = if null == 0 { *hash } else { mix(*hash, null) };
            }
        }
        let rows = numbers.groups.len();
        check!(keys.iter().all(|keys| keys.len() >= rows));
        check!(nulls.is_none_or(|nulls| nulls.len() >= rows));
        // SAFETY: each key has a word for each row, as do `nulls`, checked
        // above, and a record has the null word and keys.
        unsafe {
            // Two keys, the most common after one, compared without a loop.
            if let (None, &[first, second]) = (nulls, keys) {
                let equal = |record: *const i64, i: usize| {
                    (*record == 0)
                        & (*record.add(1) == *first.get_unchecked(i))
                        & (*record.add(2) == *second.get_unchecked(i))
                };
                return self.find_or_add_hashed(hashes, keys, |_| 0, equal, empty, numbers);
            }
            let null = |i: usize| nulls.map_or(0, |nulls| *nulls.get_unchecked(i)).cast_signed();
            let equal = |record: *const i64, i: usize| {
                (*record == null(i))
                    && keys
                        .iter()
                        .enumerate()
                        .all(|(k, keys)| *record.add(1 + k) == *keys.get_unchecked(i))
            };
            self.find_or_add_hashed(hashes, keys, null, equal, empty, numbers)
        }
    }

    /// As `find_or_add_rows`, given each row's hash, its null word by `null`,
    /// and whether a record's key is a row's by `equal`.
    #[inline(never)]
    fn find_or_add_hashed(
        &mut self,
        hashes: &[u64],
        keys: &[&[i64]],
        null: impl Fn(usize) -> i64,
        equal: impl Fn(*const i64, usize) -> bool,
        empty: &[i64],
        numbers: &mut Numbers,
    ) -> Result<(), Error> {
        let Table { slots, hashes: stored_hashes, records, width, .. } = self;
        let (slots, width) = (&mut slots[..], *width);
        check!(records.len() == stored_hashes.len() * width && width > keys.len());
        check!(slots.len().is_power_of_two());
        let mut stored: &[i64] = &records[..];
        let rows = numbers.groups.len();
        check!(hashes.len() >= rows);
        let mut i = 0;
        while i < rows {
            let hash = *at!(hashes, i);
            // SAFETY: there are a power of two many slots, checked above, and
            // a group in a slot has a record.
            let found =
                unsafe { probe(slots, hash, |group| equal(stored.as_ptr().add(group * width), i)) };
            let group = match found {
                Ok(group) => group,
                Err(at) => {
                    let mut key = [0_i64; 1 + KEYS_MAX];
                    *at_mut!(key, 0) = null(i);
                    for (word, keys) in at_mut!(key, 1..).iter_mut().zip(keys) {
                        *word = *at!(keys, i);
                    }
                    let key = at!(key, ..=keys.len());
                    let group = add(slots, stored_hashes, records, at, hash, key, empty)?;
                    stored = &records[..];
                    group
                }
            };
            // Rows of sorted keys come in runs: the rest of this one are
            // compared with the group's key, not hashed.
            // SAFETY: as above.
            let record = unsafe { stored.as_ptr().add(group as usize * width) };
            i = run(numbers, i, group, |i| equal(record, i));
        }
        Ok(())
    }

    /// Whether `direct` covers keys within `bounds`, made to if together
    /// with those it covers they're few enough.
    fn cover(&mut self, allocator: &dyn Allocator, bounds: Bounds) -> Result<bool, Error> {
        let len = i64::try_from(self.direct.len()).unwrap_or(i64::MAX);
        let end = self.direct_min.saturating_add(len - 1);
        if len > 0 && self.direct_min <= bounds.min && bounds.max <= end {
            return Ok(true);
        }
        let (min, max) = match len {
            0 => (bounds.min, bounds.max),
            _ => (self.direct_min.min(bounds.min), end.max(bounds.max)),
        };
        match max.checked_sub(min).and_then(|span| usize::try_from(span).ok()) {
            Some(span) if span < DIRECT_MAX => {
                self.direct = SlowVec::zeroed(allocator, span + 1)?;
                self.direct_min = min;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// As `find_or_add_rows`, for one key that's never null, within the
    /// keys `direct` covers.
    #[inline(never)]
    fn find_or_add_direct(
        &mut self,
        keys: &[i64],
        empty: &[i64],
        numbers: &mut Numbers,
    ) -> Result<(), Error> {
        let Table { slots, hashes, records, width, direct, direct_min } = self;
        let (direct, min, width) = (&mut direct[..], *direct_min, *width);
        let rows = numbers.groups.len();
        check!(keys.len() >= rows);
        let mut i = 0;
        while i < rows {
            let key = *at!(keys, i);
            #[expect(clippy::cast_possible_truncation, reason = "within the bounds")]
            let at = key.wrapping_sub(min).cast_unsigned() as usize;
            let entry = at_mut!(direct, at);
            let group = if let Some(group) = entry.checked_sub(1) {
                group
            } else {
                let hash = mix(SEED, key.cast_unsigned());
                let group = find_or_add(slots, hashes, records, width, hash, &[0, key], empty)?;
                *entry = group + 1;
                group
            };
            // SAFETY: checked above.
            i = run(numbers, i, group, |i| unsafe { *keys.get_unchecked(i) } == key);
        }
        Ok(())
    }

    /// As `find_or_add_rows`, for one key that's never null, compared as it
    /// is.
    #[inline(never)]
    fn find_or_add_values(
        &mut self,
        keys: &[i64],
        empty: &[i64],
        numbers: &mut Numbers,
    ) -> Result<(), Error> {
        let Table { slots, hashes, records, width, .. } = self;
        let (slots, width) = (&mut slots[..], *width);
        check!(records.len() == hashes.len() * width && width >= 2);
        check!(slots.len().is_power_of_two());
        let mut stored: &[i64] = &records[..];
        let rows = numbers.groups.len();
        check!(keys.len() >= rows);
        // SAFETY: checked just above.
        let key = |i: usize| unsafe { *keys.get_unchecked(i) };
        let mut i = 0;
        while i < rows {
            let key = key(i);
            let hash = mix(SEED, key.cast_unsigned());
            // SAFETY: there are a power of two many slots, checked above, and
            // a group in a slot has a record.
            let found = unsafe {
                probe(slots, hash, |group| {
                    stored.get_unchecked(group * width..group * width + 2) == [0, key]
                })
            };
            let group = match found {
                Ok(group) => group,
                Err(at) => {
                    let group = add(slots, hashes, records, at, hash, &[0, key], empty)?;
                    stored = &records[..];
                    group
                }
            };
            // SAFETY: as above.
            i = run(numbers, i, group, |i| unsafe { *keys.get_unchecked(i) } == key);
        }
        Ok(())
    }
}

/// A batch's group numbers: each row's, and where each run of rows of one
/// group starts, as rows of sorted keys come in runs.
struct Numbers<'a> {
    groups: &'a mut [u32],
    starts: &'a mut [u32],
    runs: usize,
}

/// Writes `group` to `numbers` for row `start` and those after it while
/// `same` says they have its key, and returns the row after them.
#[inline]
fn run(numbers: &mut Numbers, start: usize, group: u32, same: impl Fn(usize) -> bool) -> usize {
    #[expect(clippy::cast_possible_truncation, reason = "a batch's row")]
    let row = start as u32;
    *at_mut!(numbers.starts, numbers.runs) = row;
    numbers.runs += 1;
    let groups = &mut *numbers.groups;
    *at_mut!(groups, start) = group;
    let mut i = start + 1;
    while i < groups.len() && same(i) {
        // SAFETY: checked just above.
        unsafe { *groups.get_unchecked_mut(i) = group };
        i += 1;
    }
    i
}

/// The group whose key is `key`, hashed as `hash`, added with the record
/// `empty` but for the key if new. Out of line, as `add` is.
#[cold]
#[inline(never)]
fn find_or_add(
    slots: &mut SlowVec<u64>,
    hashes: &mut SlowVec<u64>,
    records: &mut SlowVec<i64>,
    width: usize,
    hash: u64,
    key: &[i64],
    empty: &[i64],
) -> Result<u32, Error> {
    check!(slots.len().is_power_of_two());
    // SAFETY: checked just above.
    let found = unsafe {
        probe(slots, hash, |group| at!(records, group * width..group * width + key.len()) == key)
    };
    match found {
        Ok(group) => Ok(group),
        Err(at) => add(slots, hashes, records, at, hash, key, empty),
    }
}

/// Adds a group for `key`, hashed as `hash`, in slot `at`, which is empty,
/// with the record `empty` but for the key, and returns its number. Out of
/// line, so loops that find groups keep what they read in registers.
#[cold]
#[inline(never)]
fn add(
    slots: &mut [u64],
    hashes: &mut SlowVec<u64>,
    records: &mut SlowVec<i64>,
    at: usize,
    hash: u64,
    key: &[i64],
    empty: &[i64],
) -> Result<u32, Error> {
    let group = hashes.len();
    if group == GROUPS_MAX {
        return Err(Error::OutOfMemory);
    }
    hashes.push(hash).map_err(|_| AllocError)?;
    records.extend_from_slice(empty)?;
    let start = records.len() - empty.len();
    at_mut!(records, start..start + key.len()).copy_from_slice(key);
    #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
    let group = group as u32;
    *at_mut!(slots, at) = slot(hash, group);
    Ok(group)
}

/// A slot for `group`, hashed as `hash`.
fn slot(hash: u64, group: u32) -> u64 {
    hash & SALT | (u64::from(group) + 1)
}

/// The group `equal` says is the one hashed as `hash`, or the empty slot to
/// put it in.
///
/// # Safety
///
/// There must be a power of two many `slots`.
#[inline]
unsafe fn probe(slots: &[u64], hash: u64, equal: impl Fn(usize) -> bool) -> Result<u32, usize> {
    let mask = slots.len().wrapping_sub(1);
    #[expect(clippy::cast_possible_truncation, reason = "masked")]
    let mut at = hash as usize & mask;
    loop {
        // SAFETY: masked below the slots' count, a power of two.
        let slot = unsafe { *slots.get_unchecked(at) };
        if slot == 0 {
            return Err(at);
        }
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let group = (slot & !SALT) as u32 - 1;
        if (slot ^ hash) & SALT == 0 && equal(group as usize) {
            return Ok(group);
        }
        at = (at + 1) & mask;
    }
}

/// `hash` with `word` mixed in: their xor multiplied out to 128 bits, and
/// the halves folded, so every bit of the word reaches both the low bits a
/// slot is picked by and the top ones it keeps.
#[inline]
fn mix(hash: u64, word: u64) -> u64 {
    let product = u128::from(hash ^ word) * u128::from(0x9e37_79b9_7f4a_7c15_u64);
    #[expect(clippy::cast_possible_truncation, reason = "folded")]
    let folded = product as u64 ^ (product >> 64) as u64;
    folded
}

/// The hash of a group's key, its null word and keys, as `Table` hashes a
/// batch's.
fn hash(key: &[i64]) -> u64 {
    let (nulls, keys) = (at!(key, 0).cast_unsigned(), at!(key, 1..));
    let hash = keys.iter().fold(SEED, |hash, &key| mix(hash, key.cast_unsigned()));
    if nulls == 0 { hash } else { mix(hash, nulls) }
}

impl HashAggregate {
    fn new(allocator: &dyn Allocator, grouping: Grouping) -> Result<HashAggregate, AllocError> {
        check!(grouping.keys.len() <= KEYS_MAX);
        check!(grouping.aggregates.len() <= AGGREGATES_MAX);
        let mut offsets = SlowVec::fixed(allocator, grouping.aggregates.len())?;
        let mut empty = SlowVec::new(allocator, 4 * (KEYS_MAX + AGGREGATES_MAX))?;
        for _ in 0..2 + grouping.keys.len() {
            empty.push(0).map_err(|_| AllocError)?;
        }
        for aggregate in grouping.aggregates.iter() {
            let offset = empty.len();
            #[expect(clippy::cast_possible_truncation, reason = "a few words each")]
            offsets.push(offset as u32).map_err(|_| AllocError)?;
            empty.extend_from_slice(aggregate.empty())?;
        }
        Ok(HashAggregate { grouping, offsets, empty })
    }

    /// Where a record's seen word is.
    fn seen_at(&self) -> usize {
        1 + self.grouping.keys.len()
    }

    /// Writes the group of each row `input` keeps to `numbers`, adding
    /// groups for keys not seen before.
    fn find_or_add(
        &self,
        context: &mut Context,
        table: &mut Table,
        input: &Flat,
        numbers: &mut Numbers,
    ) -> Result<(), Error> {
        let rows = numbers.groups.len();
        let select = input.select();
        // Each key as a word a row, 0 where null, copied unless every row is
        // kept and none is null.
        let mut copies = [const { None }; KEYS_MAX];
        let mut nulls: Option<Buffer> = None;
        let keys = self.grouping.keys.iter().zip(copies.iter_mut());
        for (k, (&position, copy)) in keys.enumerate() {
            let column = input.column(position);
            let values = column.int64s();
            let Some(validity) = column.validity() else {
                gathered(context, select, values, copy)?;
                continue;
            };
            if nulls.is_none() {
                let mut words = context.values_buffer(rows * 8)?;
                words.as_mut_slice::<u64>().fill(0);
                nulls = Some(words);
            }
            let Some(nulls) = &mut nulls else { crate::check::check_failed(line!()) };
            let out = copy.insert(context.values_buffer(rows * 8)?).as_mut_slice::<i64>();
            for (i, (out, null)) in out.iter_mut().zip(nulls.as_mut_slice::<u64>()).enumerate() {
                let row = if select.is_empty() { i } else { usize::from(*at!(select, i)) };
                let valid = is_valid(validity, row);
                *out = if valid { *at!(values, row) } else { 0 };
                *null |= u64::from(!valid) << k;
            }
        }
        let mut keys: [&[i64]; KEYS_MAX] = [&[]; KEYS_MAX];
        for ((keys, copy), &position) in keys.iter_mut().zip(&copies).zip(self.grouping.keys.iter())
        {
            *keys = match copy {
                Some(copy) => copy.as_slice(),
                None => at!(input.column(position).int64s(), ..rows),
            };
        }
        let keys = at!(keys, ..self.grouping.keys.len());
        let nulls = nulls.as_ref().map(Buffer::as_slice);
        let bounds = input.column(*at!(self.grouping.keys, 0)).bounds();
        table.find_or_add_rows(context, keys, nulls, bounds, &self.empty, numbers)
    }

    /// Updates aggregate `a`'s state in each group by the rows of `input`,
    /// each in the group `rows` says. Groups numbered below `before` were
    /// there before this batch.
    fn update(
        &self,
        context: &mut Context,
        groups: &mut Groups,
        input: &Flat,
        before: usize,
        a: usize,
        rows: Rows,
    ) -> Result<(), Error> {
        let aggregate = *at!(self.grouping.aggregates, a);
        let (width, seen_at) = (groups.table.width, self.seen_at());
        let offset = *at!(self.offsets, a) as usize;
        let narrow = groups.narrow;
        let records = &mut groups.table.records[..];
        check!(records.len() == groups.table.hashes.len() * width);
        let Some((position, _)) = aggregate.input else {
            count(records, Place { width, offset, seen: (seen_at, 0) }, rows);
            return Ok(());
        };
        let column = input.column(position);
        let select = input.select();
        let loaded = validity(context, column, select, rows.len())?;
        let bit = 1_u64 << a;
        if aggregate.function.skips_nulls() && loaded.is_some() && groups.tracked & bit == 0 {
            groups.tracked |= bit;
            for record in records.chunks_exact_mut(width).take(before) {
                *at_mut!(record, seen_at) |= bit.cast_signed();
            }
        }
        let tracked = groups.tracked & bit != 0;
        let valid = loaded.as_ref().map(Buffer::as_slice);
        let seen = (seen_at, if tracked { bit.cast_signed() } else { 0 });
        let place = Place { width, offset, seen };
        let values = Values { select, column, valid };
        accumulate(context, records, place, rows, aggregate.function, narrow, values)
    }

    /// Keeps integer sums in one word if `rows` more values within the
    /// bounds of their columns in `input` can't take any past an `Int64`, or
    /// else widens them to two.
    fn narrow(&self, groups: &mut Groups, input: &Flat, rows: usize) {
        if !groups.narrow {
            return;
        }
        let mut largest = Some(0_u64);
        for aggregate in self.grouping.aggregates.iter().filter(|aggregate| aggregate.is_int_sum())
        {
            let Some((position, _)) = aggregate.input else { continue };
            let bounds = input.column(position).bounds();
            largest = largest.zip(bounds).map(|(largest, bounds)| {
                largest.max(bounds.min.unsigned_abs()).max(bounds.max.unsigned_abs())
            });
        }
        let reach = largest
            .and_then(|largest| largest.checked_mul(rows as u64))
            .and_then(|reach| reach.checked_add(groups.reach))
            .filter(|&reach| reach <= i64::MAX.cast_unsigned());
        match reach {
            Some(reach) => groups.reach = reach,
            None => self.widen(groups),
        }
    }

    /// Widens integer sums from their first word to both.
    fn widen(&self, groups: &mut Groups) {
        groups.narrow = false;
        let sums = self.grouping.aggregates.iter().zip(self.offsets.iter());
        for (_, &offset) in sums.filter(|(aggregate, _)| aggregate.is_int_sum()) {
            let offset = offset as usize;
            for record in groups.table.records.chunks_exact_mut(groups.table.width) {
                *at_mut!(record, offset + 1) = *at!(record, offset) >> 63;
            }
        }
    }

    /// Folds the states of `theirs`, a group's record, into `ours`, another
    /// record of the same group.
    fn combine(&self, ours: &mut [i64], theirs: &[i64]) {
        let float = |word: i64| f64::from_bits(word.cast_unsigned());
        let word = |value: f64| value.to_bits().cast_signed();
        for (aggregate, &offset) in self.grouping.aggregates.iter().zip(self.offsets.iter()) {
            let offset = offset as usize;
            let words = aggregate.empty().len();
            let ours = at_mut!(ours, offset..offset + words);
            let theirs = at!(theirs, offset..offset + words);
            let (a, b) = (*at!(ours, 0), *at!(theirs, 0));
            let data_type = aggregate.input.map(|(_, data_type)| data_type);
            match (aggregate.function, data_type) {
                (Function::Count, _) => *at_mut!(ours, 0) = a + b,
                (Function::Sum | Function::Avg, Some(DataType::Int64)) => {
                    let sum = int128(ours) + int128(theirs);
                    #[expect(clippy::cast_possible_truncation, reason = "split in two")]
                    at_mut!(ours, ..2).copy_from_slice(&[sum as i64, (sum >> 64) as i64]);
                }
                (Function::Sum | Function::Avg, _) => *at_mut!(ours, 0) = word(float(a) + float(b)),
                (Function::Min, Some(DataType::Int64)) => *at_mut!(ours, 0) = a.min(b),
                (Function::Max, Some(DataType::Int64)) => *at_mut!(ours, 0) = a.max(b),
                (Function::Min, _) => *at_mut!(ours, 0) = word(float(a).min(float(b))),
                (Function::Max, _) => *at_mut!(ours, 0) = word(float(a).max(float(b))),
            }
            if aggregate.function == Function::Avg {
                *at_mut!(ours, words - 1) += *at!(theirs, words - 1);
            }
        }
    }

    /// Output column `c`, a key or then an aggregate, of the groups in
    /// `range`.
    fn column(
        &self,
        context: &mut Context,
        groups: &Groups,
        c: usize,
        range: core::ops::Range<usize>,
    ) -> Result<ColumnView, Error> {
        let width = groups.table.width;
        let Some(a) = c.checked_sub(self.grouping.keys.len()) else {
            let place = Place { width, offset: 1 + c, seen: (0, 0) };
            let records = &groups.table.records;
            return output_column(context, DataType::Int64, records, place, range, Output::Key(c));
        };
        let aggregate = *at!(self.grouping.aggregates, a);
        let data_type = aggregate.input.map(|(_, data_type)| data_type);
        let Some(output_type) = aggregate.function.output_type(data_type) else {
            crate::check::check_failed(line!());
        };
        let output = match (aggregate.function, data_type) {
            (Function::Count, _) => Output::Count,
            (Function::Avg, Some(DataType::Int64)) => Output::IntAvg,
            (Function::Avg, _) => Output::FloatAvg,
            (Function::Sum, Some(DataType::Int64)) => Output::Sum,
            _ => Output::Word,
        };
        let tracked = groups.tracked & 1 << a;
        let offset = *at!(self.offsets, a) as usize;
        let place = Place { width, offset, seen: (self.seen_at(), tracked.cast_signed()) };
        output_column(context, output_type, &groups.table.records, place, range, output)
    }

    /// Where the columns it reads are in batches: the keys, then the
    /// aggregates'.
    fn read(&self) -> impl Iterator<Item = u32> + '_ {
        let inputs = self.grouping.aggregates.iter().filter_map(|aggregate| aggregate.input);
        self.grouping.keys.iter().copied().chain(inputs.map(|(position, _)| position))
    }

    /// Whether every key is one value in `batch`, as with no keys, writing
    /// the null word and keys to `key`.
    fn constant_key(&self, batch: &RowBatch, key: &mut [i64]) -> bool {
        let mut nulls = 0;
        let words = at_mut!(key, 1..).iter_mut();
        for (k, (&position, word)) in self.grouping.keys.iter().zip(words).enumerate() {
            let column = batch.column(position);
            if !column.is_in(Forms::CONSTANT) {
                return false;
            }
            let values = column.values();
            let null = values.is_null(0);
            *word = if null { 0 } else { *at!(values.int64s(), 0) };
            nulls |= u64::from(null) << k;
        }
        *at_mut!(key, 0) = nulls.cast_signed();
        true
    }
}

impl Lowered {
    /// Its state in a group with no values: a count, a sum of two words for
    /// integers, a least or greatest, or for `Avg` a sum and a count.
    fn empty(&self) -> &'static [i64] {
        const NAN: i64 = f64::NAN.to_bits().cast_signed();
        let data_type = self.input.map(|(_, data_type)| data_type);
        match (self.function, data_type) {
            (Function::Count, _) | (Function::Sum, Some(DataType::Float64)) => &[0],
            (Function::Sum, _) | (Function::Avg, Some(DataType::Float64)) => &[0, 0],
            (Function::Avg, _) => &[0, 0, 0],
            (Function::Min, Some(DataType::Int64)) => &[i64::MAX],
            (Function::Max, Some(DataType::Int64)) => &[i64::MIN],
            // `min` and `max` take the other of a NaN.
            (Function::Min | Function::Max, _) => &[NAN],
        }
    }
}

impl Lowered {
    /// Whether it's a sum of integers, as `Avg` of them keeps.
    fn is_int_sum(&self) -> bool {
        matches!(self.function, Function::Sum | Function::Avg)
            && matches!(self.input, Some((_, DataType::Int64)))
    }
}

impl Function {
    /// Whether its result is null for a group with only nulls, so groups
    /// need to say whether they've seen a value.
    fn skips_nulls(self) -> bool {
        matches!(self, Function::Sum | Function::Min | Function::Max)
    }
}

/// The two words of an integer sum, as one.
fn int128(state: &[i64]) -> i128 {
    i128::from(*at!(state, 1)) << 64 | i128::from(at!(state, 0).cast_unsigned())
}

/// The groups of a batch's rows.
#[derive(Clone, Copy)]
enum Rows<'a> {
    /// Each row's.
    Grouped(&'a [u32]),
    /// Each row's, and where each run of rows of one group starts, then the
    /// rows' end.
    Runs(&'a [u32], &'a [u32]),
    /// This many rows, all in this group.
    All(u32, usize),
}

impl Rows<'_> {
    fn len(self) -> usize {
        match self {
            Rows::Grouped(numbers) | Rows::Runs(numbers, _) => numbers.len(),
            Rows::All(_, rows) => rows,
        }
    }
    /// Row `i`'s group.
    fn group(self, i: usize) -> u32 {
        match self {
            Rows::Grouped(numbers) | Rows::Runs(numbers, _) => *at!(numbers, i),
            Rows::All(group, _) => group,
        }
    }
}

/// Where an aggregate's state is in each record, `width` words apart: at
/// `offset`, and the word and bit that say a group has seen a value, the bit
/// 0 if not kept.
#[derive(Clone, Copy)]
struct Place {
    width: usize,
    offset: usize,
    seen: (usize, i64),
}

/// Whether `row` isn't null by `validity`.
fn is_valid(validity: Validity, row: usize) -> bool {
    check!(row < validity.row_count() as usize);
    #[expect(clippy::cast_possible_truncation, reason = "a batch's row")]
    // SAFETY: checked just above.
    unsafe {
        validity.is_valid_unchecked(row as u32)
    }
}

/// Which of `column`'s rows `select` keeps, all if empty, aren't null: a bit
/// for each of `rows`, from the first. `None` if none are.
fn validity(
    context: &mut Context,
    column: &ColumnView,
    select: &[u16],
    rows: usize,
) -> Result<Option<Buffer>, Error> {
    let Some(validity) = column.validity() else { return Ok(None) };
    let mut words = context.small_buffer(rows.div_ceil(64) * 8)?;
    let out = words.as_mut_slice::<u64>();
    out.fill(0);
    for i in 0..rows {
        let row = if select.is_empty() { i } else { usize::from(*at!(select, i)) };
        *at_mut!(out, i / 64) |= u64::from(is_valid(validity, row)) << (i % 64);
    }
    Ok(Some(words))
}

/// The words of `column` at the rows `select` keeps: all of them if it's
/// empty, or else a copy, in `copy`.
fn gathered<'a>(
    context: &mut Context,
    select: &[u16],
    column: &'a [i64],
    copy: &'a mut Option<Buffer>,
) -> Result<&'a [i64], Error> {
    if select.is_empty() {
        return Ok(column);
    }
    let copy = copy.insert(context.values_buffer(select.len() * 8)?);
    for (out, &row) in copy.as_mut_slice::<i64>().iter_mut().zip(select) {
        *out = *at!(column, usize::from(row));
    }
    Ok(copy.as_slice())
}

/// A flat column's values as words: integers, or floats' bits.
fn words(column: &ColumnView) -> &[i64] {
    // SAFETY: any bits are an `i64`.
    let (before, words, after) = unsafe { column.value_bytes().align_to::<i64>() };
    check!(before.is_empty() && after.is_empty());
    words
}

/// The rows `select` keeps, all if empty, of an aggregate's `column`, and
/// which aren't null, unless none are.
#[derive(Clone, Copy)]
struct Values<'a> {
    select: &'a [u16],
    column: &'a ColumnView,
    valid: Option<&'a [u64]>,
}

/// Adds `values` to the states at `place` of their rows' groups by
/// `function`: an integer sum in one word if `narrow`.
fn accumulate(
    context: &mut Context,
    records: &mut [i64],
    place: Place,
    rows: Rows,
    function: Function,
    narrow: bool,
    values: Values,
) -> Result<(), Error> {
    let Values { select, column, valid } = values;
    // Loops of their own need no nulls, nor seen bits kept.
    let plain = valid.is_none() && place.seen.1 == 0;
    if function == Function::Count && plain {
        count(records, place, rows);
        return Ok(());
    }
    let mut copy = None;
    let words = gathered(context, select, words(column), &mut copy)?;
    let data_type = column.data_type();
    match (function, data_type) {
        (Function::Sum | Function::Avg, DataType::Int64) if narrow && plain => {
            sum(records, place, rows, words);
        }
        _ => update(records, place, rows, words, valid, Fold::of(function, data_type, narrow)),
    }
    // An average's count is after its sum.
    if function == Function::Avg {
        let offset = place.offset + if data_type == DataType::Int64 { 2 } else { 1 };
        let place = Place { offset, ..place };
        match valid {
            None => count(records, place, rows),
            Some(_) => update(records, place, rows, words, valid, Fold::Count),
        }
    }
    Ok(())
}

/// What an aggregate other than a count does with each value.
#[derive(Clone, Copy)]
enum Fold {
    /// Counts it, if it isn't null.
    Count,
    /// Adds it to an integer sum in one word.
    Sum,
    /// Adds it to an integer sum in two.
    WideSum,
    /// Adds it to a float sum.
    FloatSum,
    /// Keeps the least integer.
    Min,
    /// Keeps the greatest integer.
    Max,
    /// Keeps the least float.
    FloatMin,
    /// Keeps the greatest float.
    FloatMax,
}

impl Fold {
    fn of(function: Function, data_type: DataType, narrow: bool) -> Fold {
        match (function, data_type) {
            (Function::Sum | Function::Avg, DataType::Int64) if narrow => Fold::Sum,
            (Function::Sum | Function::Avg, DataType::Int64) => Fold::WideSum,
            (Function::Min, DataType::Int64) => Fold::Min,
            (Function::Max, DataType::Int64) => Fold::Max,
            (Function::Min, _) => Fold::FloatMin,
            (Function::Max, _) => Fold::FloatMax,
            (Function::Count, _) => Fold::Count,
            (Function::Sum | Function::Avg, _) => Fold::FloatSum,
        }
    }

    /// Adds `value`, an integer or a float's bits, to `state`, unless it's
    /// not `valid`. `min` and `max` take the other of a NaN, as of a null.
    fn add(self, state: &mut [i64], value: i64, valid: bool) {
        let float = |word: i64| f64::from_bits(word.cast_unsigned());
        let word = |value: f64| value.to_bits().cast_signed();
        let (old, added) = (*at!(state, 0), if valid { value } else { 0 });
        let ignored = if valid { float(value) } else { f64::NAN };
        *at_mut!(state, 0) = match self {
            Fold::Count => old + i64::from(valid),
            Fold::Sum => old.wrapping_add(added),
            Fold::WideSum => {
                let sum = int128(state) + i128::from(added);
                #[expect(clippy::cast_possible_truncation, reason = "split in two")]
                let (low, high) = (sum as i64, (sum >> 64) as i64);
                *at_mut!(state, 1) = high;
                low
            }
            Fold::FloatSum => word(float(old) + float(added)),
            Fold::Min => old.min(if valid { value } else { i64::MAX }),
            Fold::Max => old.max(if valid { value } else { i64::MIN }),
            Fold::FloatMin => word(float(old).min(ignored)),
            Fold::FloatMax => word(float(old).max(ignored)),
        };
    }
}

/// Adds each row's value in `values` to the state at `place` of its group
/// by `fold`, given whether it isn't null by `valid`, and if so marks the group
/// seen. A row at a time, choosing what to do for each: slower than a loop
/// for each, but small, for all but the commonest aggregates.
fn update(
    records: &mut [i64],
    place: Place,
    rows: Rows,
    values: &[i64],
    valid: Option<&[u64]>,
    fold: Fold,
) {
    let Place { width, offset, seen: (seen_at, bit) } = place;
    for (i, &value) in at!(values, ..rows.len()).iter().enumerate() {
        let group = rows.group(i) as usize;
        let valid = valid.is_none_or(|valid| *at!(valid, i / 64) >> (i % 64) & 1 != 0);
        let record = at_mut!(records, group * width..(group + 1) * width);
        fold.add(at_mut!(record, offset..), value, valid);
        *at_mut!(record, seen_at) |= bit & -i64::from(valid);
    }
}

/// Adds each row's value in `values`, none null, to the integer sum in one
/// word at `place` of its group: the commonest aggregate, so it has loops of
/// its own, out of line so they keep what they read in registers.
#[inline(never)]
fn sum(records: &mut [i64], place: Place, rows: Rows, values: &[i64]) {
    let Place { width, offset, .. } = place;
    check!(offset < width && values.len() >= rows.len());
    let numbers = match rows {
        Rows::Grouped(numbers) => numbers,
        Rows::Runs(numbers, starts) => {
            for run in starts.windows(2) {
                let (start, end) = (*at!(run, 0) as usize, *at!(run, 1) as usize);
                let values = at!(values, start..end).iter();
                let sum = values.fold(0_i64, |sum, &value| sum.wrapping_add(value));
                let state = at_mut!(records, *at!(numbers, start) as usize * width + offset);
                *state = state.wrapping_add(sum);
            }
            return;
        }
        Rows::All(group, rows) => {
            let values = at!(values, ..rows).iter();
            let sum = values.fold(0_i64, |sum, &value| sum.wrapping_add(value));
            let state = at_mut!(records, group as usize * width + offset);
            *state = state.wrapping_add(sum);
            return;
        }
    };
    if cfg!(debug_assertions) {
        check!(numbers.iter().all(|&group| (group as usize + 1) * width <= records.len()));
    }
    let records = records.as_mut_ptr();
    for (&group, &value) in numbers.iter().zip(values) {
        // SAFETY: each number is a group's, whose record is in `records`.
        unsafe {
            let state = records.add(group as usize * width + offset);
            *state = (*state).wrapping_add(value);
        }
    }
}

/// Adds 1 to the count at `place` of the group of each row: with `sum`, the
/// commonest aggregates, so with a loop of their own.
fn count(records: &mut [i64], place: Place, rows: Rows) {
    let Place { width, offset, .. } = place;
    check!(offset < width);
    let numbers = match rows {
        Rows::Grouped(numbers) => numbers,
        Rows::Runs(numbers, starts) => {
            for run in starts.windows(2) {
                let (start, end) = (*at!(run, 0), *at!(run, 1));
                let count =
                    at_mut!(records, *at!(numbers, start as usize) as usize * width + offset);
                *count += i64::from(end - start);
            }
            return;
        }
        Rows::All(group, rows) => {
            *at_mut!(records, group as usize * width + offset) += i64::try_from(rows).unwrap_or(0);
            return;
        }
    };
    if cfg!(debug_assertions) {
        check!(numbers.iter().all(|&group| (group as usize + 1) * width <= records.len()));
    }
    let records = records.as_mut_ptr();
    for &group in numbers {
        // SAFETY: each number is a group's, whose record is in `records`.
        unsafe { *records.add(group as usize * width + offset) += 1 };
    }
}

/// What a column of output holds, a word for each group: one of its keys,
/// or what an aggregate made of its rows.
#[derive(Clone, Copy)]
enum Output {
    /// The key, null if its bit in the null word says so.
    Key(usize),
    /// A count.
    Count,
    /// An integer sum, null if it saw no values.
    Sum,
    /// A float sum, or a least or greatest, null if it saw no values.
    Word,
    /// An integer sum, then a count.
    IntAvg,
    /// A float sum, then a count.
    FloatAvg,
}

/// The column of `output` of the groups in `range`, whose records are in
/// `records`, their states at `place`.
fn output_column(
    context: &mut Context,
    data_type: DataType,
    records: &[i64],
    place: Place,
    range: core::ops::Range<usize>,
    output: Output,
) -> Result<ColumnView, Error> {
    let Place { width, offset, seen: (seen_at, bit) } = place;
    let records = at!(records, range.start * width..range.end * width);
    let rows = range.len();
    let mut values = context.values_buffer(rows * 8)?;
    let mut validity = context.small_buffer(rows.div_ceil(8))?;
    let mut any_null = false;
    let float = |word: i64| f64::from_bits(word.cast_unsigned());
    #[expect(clippy::cast_precision_loss, reason = "averages are approximate")]
    let average = |sum: f64, count: i64| (sum / count as f64).to_bits().cast_signed();
    // Keys and counts, the commonest, are copied: a key is null if its bit
    // in the null word is set, a count never.
    if let Output::Key(_) | Output::Count = output {
        let null = if let Output::Key(k) = output { 1 << k } else { 0 };
        let out = values.as_mut_slice::<i64>().iter_mut();
        for (out, record) in out.zip(records.chunks_exact(width)) {
            *out = *at!(record, offset);
            any_null |= *at!(record, 0) & null != 0;
        }
        if !any_null {
            return Ok(ColumnView::new(context, data_type, values, None)?);
        }
    }
    // Eight rows at a time, a byte of their bits.
    let out = values.as_mut_slice::<i64>().chunks_mut(8);
    let records = records.chunks(8 * width);
    for (bits, (out, records)) in validity.as_mut_slice::<u8>().iter_mut().zip(out.zip(records)) {
        *bits = 0;
        for (i, (out, record)) in out.iter_mut().zip(records.chunks_exact(width)).enumerate() {
            let state = at!(record, offset..);
            let seen = || *at!(record, seen_at) & bit == bit;
            #[expect(clippy::cast_precision_loss, reason = "averages are approximate")]
            let word = match output {
                Output::Key(k) => (*at!(record, 0) >> k & 1 == 0).then(|| *at!(state, 0)),
                Output::Count => Some(*at!(state, 0)),
                // A sum that doesn't fit in an `Int64` would need a wider type.
                Output::Sum => {
                    let sum = i64::try_from(int128(state)).map_err(|_| Error::Unsupported)?;
                    seen().then_some(sum)
                }
                Output::Word => seen().then(|| *at!(state, 0)),
                Output::IntAvg => {
                    let count = *at!(state, 2);
                    (count != 0).then(|| average(int128(state) as f64, count))
                }
                Output::FloatAvg => {
                    let count = *at!(state, 1);
                    (count != 0).then(|| average(float(*at!(state, 0)), count))
                }
            };
            *out = word.unwrap_or(0);
            *bits |= u8::from(word.is_some()) << i;
            any_null |= word.is_none();
        }
    }
    Ok(ColumnView::new(context, data_type, values, any_null.then_some(validity))?)
}

/// A batch whose columns this reads are flat: they're in whatever form the
/// first op to read them takes, so those that aren't are made flat.
struct Flat<'b, 'r> {
    /// The batch, whose columns that are flat are read as they are.
    batch: &'b RowBatch<'r>,
    /// A flat copy of each column that isn't, by its position.
    copies: [Option<ColumnView>; BATCH_COLUMNS_MAX as usize],
}

impl<'b, 'r> Flat<'b, 'r> {
    fn new(
        context: &mut Context,
        batch: &'b RowBatch<'r>,
        read: impl Iterator<Item = u32>,
    ) -> Result<Flat<'b, 'r>, Error> {
        let mut copies = [const { None }; BATCH_COLUMNS_MAX as usize];
        for position in read {
            let column = batch.column(position);
            if !column.is_in(Forms::FLAT) && at!(copies, position as usize).is_none() {
                let mut copy = column.clone();
                copy.make_in(context, Forms::FLAT)?;
                *at_mut!(copies, position as usize) = Some(copy);
            }
        }
        Ok(Flat { batch, copies })
    }

    fn column(&self, position: u32) -> &ColumnView {
        at!(self.copies, position as usize).as_ref().unwrap_or_else(|| self.batch.column(position))
    }

    /// The rows kept, by index, or none if all are.
    fn select(&self) -> &[u16] {
        match self.batch.selection().kept() {
            Kept::Select(select) => select,
            Kept::All | Kept::None => &[],
        }
    }
}

impl Breaker for HashAggregate {
    type State = Groups;

    fn new_state(&self, context: &mut Context) -> Result<Groups, Error> {
        let width = self.empty.len();
        let table = Table::new(context.allocator(), width)?;
        Ok(Groups { table, tracked: 0, narrow: true, reach: 0, served: 0 })
    }

    fn consume(
        &self,
        context: &mut Context,
        groups: &mut Groups,
        input: &RowBatch,
    ) -> Result<(), Error> {
        let rows = input.selection().len() as usize;
        if rows == 0 {
            return Ok(());
        }
        let before = groups.table.len();
        let mut key = [0_i64; 1 + KEYS_MAX];
        let key = at_mut!(key, ..=self.grouping.keys.len());
        let mut numbers = None;
        let (input, rows) = if self.constant_key(input, key) {
            // All rows are one group: with no keys, the one there is.
            let input = Flat::new(context, input, self.read().skip(self.grouping.keys.len()))?;
            groups.table.reserve(context.allocator(), 1)?;
            let group = groups.table.find_or_add(hash(key), key, &self.empty)?;
            (input, Rows::All(group, rows))
        } else {
            let input = Flat::new(context, input, self.read())?;
            let (groups_of, starts) = numbers.insert((
                context.indices_buffer(rows * 4)?,
                context.indices_buffer((rows + 1) * 4)?,
            ));
            let mut numbers = Numbers {
                groups: groups_of.as_mut_slice(),
                starts: starts.as_mut_slice(),
                runs: 0,
            };
            self.find_or_add(context, &mut groups.table, &input, &mut numbers)?;
            let Numbers { groups: numbers, starts, runs } = numbers;
            // Folding a run at a time pays where runs are long enough.
            if runs * 2 <= rows {
                #[expect(clippy::cast_possible_truncation, reason = "a batch's rows")]
                let end = rows as u32;
                *at_mut!(starts, runs) = end;
                (input, Rows::Runs(numbers, at!(starts, ..=runs)))
            } else {
                (input, Rows::Grouped(numbers))
            }
        };
        self.narrow(groups, &input, rows.len());
        for a in 0..self.grouping.aggregates.len() {
            self.update(context, groups, &input, before, a, rows)?;
        }
        Ok(())
    }

    fn merge(
        &self,
        context: &mut Context,
        groups: &mut Groups,
        mut other: Groups,
    ) -> Result<(), Error> {
        let (width, seen_at) = (groups.table.width, self.seen_at());
        // Sums are merged wide: nothing is consumed after.
        for state in [&mut *groups, &mut other] {
            if state.narrow {
                self.widen(state);
            }
        }
        // Those `other` keeps that these don't: every group here has seen a
        // value.
        let starting = other.tracked & !groups.tracked;
        if starting != 0 {
            for record in groups.table.records.chunks_exact_mut(width) {
                *at_mut!(record, seen_at) |= starting.cast_signed();
            }
            groups.tracked |= starting;
        }
        groups.table.reserve(context.allocator(), other.table.len())?;
        let theirs = other.table.records.chunks_exact(width);
        for (theirs, &hash) in theirs.zip(other.table.hashes.iter()) {
            let key = at!(theirs, ..seen_at);
            let group = groups.table.find_or_add(hash, key, &self.empty)? as usize;
            let ours = at_mut!(groups.table.records, group * width..(group + 1) * width);
            self.combine(ours, theirs);
            *at_mut!(ours, seen_at) |= *at!(theirs, seen_at) | (!other.tracked).cast_signed();
        }
        Ok(())
    }

    fn finalize(&self, context: &mut Context, groups: &mut Groups) -> Result<(), Error> {
        // Served wide, as merged.
        if groups.narrow {
            self.widen(groups);
        }
        // With no keys, all rows are one group, which is there even if there
        // are none, having seen no values.
        if self.grouping.keys.is_empty() && groups.table.len() == 0 {
            groups.table.reserve(context.allocator(), 1)?;
            groups.table.find_or_add(hash(&[0]), &[0], &self.empty)?;
            groups.tracked = u64::MAX;
        }
        Ok(())
    }

    fn serve(
        &self,
        context: &mut Context,
        groups: &mut Groups,
        output: &mut RowBatch,
    ) -> Result<bool, Error> {
        let first = groups.served;
        let rows = (groups.table.len() - first).min(BATCH_ROWS_MAX as usize);
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
        output.reset(rows as u32);
        let keys = self.grouping.keys.len();
        for c in 0..keys + self.grouping.aggregates.len() {
            let column = self.column(context, groups, c, first..first + rows)?;
            let Ok(()) = output.push_column(column) else { crate::check::check_failed(line!()) };
        }
        groups.served += rows;
        Ok(groups.served < groups.table.len())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::pipeline::Pipeline;
    use crate::query_allocators::QueryAllocators;
    use crate::step::Source;

    /// A column of `values`, null where `None`.
    fn column(data_type: DataType, values: &[Option<i64>]) -> ColumnView {
        let mut words = Buffer::allocate(&Heap, values.len() * 8).unwrap();
        let mut bits = Buffer::allocate(&Heap, values.len().div_ceil(8)).unwrap();
        for (i, value) in values.iter().enumerate() {
            words.as_mut_slice::<i64>()[i] = value.unwrap_or(0);
            bits.as_mut_slice::<u8>()[i / 8] |= u8::from(value.is_some()) << (i % 8);
        }
        let validity = values.iter().any(Option::is_none).then_some(bits);
        ColumnView::new(&mut Context::new(&Heap), data_type, words, validity).unwrap()
    }

    /// Batches of a key `k` and a value `x`, each a list of rows, keeping
    /// all but those in `dropped`.
    struct Rows {
        batches: Vec<Vec<(Option<i64>, Option<i64>)>>,
        dropped: Vec<u16>,
    }

    impl Source for Rows {
        type State = usize;

        fn new_state(&self, _: &mut Context) -> Result<usize, Error> {
            Ok(0)
        }

        fn next(
            &self,
            _: &mut Context,
            i: &mut usize,
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            let Some(rows) = self.batches.get(*i) else { return Ok(false) };
            *i += 1;
            batch.reset(u32::try_from(rows.len()).unwrap());
            let keys: Vec<_> = rows.iter().map(|row| row.0).collect();
            let values: Vec<_> = rows.iter().map(|row| row.1).collect();
            assert!(batch.push_column(column(DataType::Int64, &keys)).is_ok());
            assert!(batch.push_column(column(DataType::Int64, &values)).is_ok());
            batch.selection_mut().retain(|row| !self.dropped.contains(&row));
            Ok(true)
        }
    }

    /// Batches of the columns given.
    struct Columns(Vec<Vec<ColumnView>>);

    impl Source for Columns {
        type State = usize;

        fn new_state(&self, _: &mut Context) -> Result<usize, Error> {
            Ok(0)
        }

        fn next(
            &self,
            _: &mut Context,
            i: &mut usize,
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            let Some(columns) = self.0.get(*i) else { return Ok(false) };
            *i += 1;
            batch.reset(columns[0].row_count());
            for column in columns {
                assert!(batch.push_column(column.clone()).is_ok());
            }
            Ok(true)
        }
    }

    /// Groups by `k` if `grouped`, with `COUNT(*)`, `COUNT(x)`, `SUM(x)`,
    /// `MIN(x)`, `MAX(x)` and `AVG(x)`.
    fn grouping(grouped: bool) -> Grouping {
        grouping_by(if grouped { &[0] } else { &[] }, DataType::Int64)
    }

    /// As `grouping`, by the columns at `keys`, with `x` at 1 of `data_type`.
    fn grouping_by(keys: &[u32], data_type: DataType) -> Grouping {
        let x = Some((1, data_type));
        let functions = [
            (Function::Count, None),
            (Function::Count, x),
            (Function::Sum, x),
            (Function::Min, x),
            (Function::Max, x),
            (Function::Avg, x),
        ];
        let aggregates = functions.into_iter().map(|(function, input)| Lowered { function, input });
        Grouping {
            keys: SlowVec::fixed_from(&Heap, keys.iter().copied()).unwrap(),
            aggregates: SlowVec::fixed_from(&Heap, aggregates).unwrap(),
        }
    }

    /// Each row out, its columns as integers, or floats' bits, or `None` if null,
    /// sorted.
    fn aggregated(rows: Rows, grouped: bool) -> Vec<Vec<Option<i64>>> {
        run(rows, grouping(grouped))
    }

    /// As `aggregated`, of `source`'s batches by `grouping`.
    fn run(source: impl Source, grouping: Grouping) -> Vec<Vec<Option<i64>>> {
        let mut steps = SlowVec::new(&Heap, 1).unwrap();
        assert!(steps.push(Step::Operator(hash_aggregate(&Heap, grouping).unwrap())).is_ok());
        let pipeline = Pipeline::new(crate::step::DynSource::new(&Heap, source).unwrap(), steps);
        let query = QueryAllocators::new(&Heap);
        let mut execution = pipeline.start(&query).unwrap();
        let mut batch = RowBatch::new();
        let mut out = Vec::new();
        while execution.next(&mut batch).unwrap() {
            for row in 0..batch.row_count() {
                let values = (0..batch.column_count()).map(|i| {
                    let column = batch.column(i);
                    (!column.is_null(row)).then(|| match column.data_type() {
                        DataType::Float64 => {
                            column.float64s()[row as usize].to_bits().cast_signed()
                        }
                        _ => column.int64s()[row as usize],
                    })
                });
                out.push(values.collect());
            }
        }
        out.sort();
        out
    }

    /// A float's bits, as `aggregated` gives them.
    fn bits(value: f64) -> i64 {
        value.to_bits().cast_signed()
    }

    #[test]
    fn aggregates_each_group_of_rows_kept() {
        // Row 1 of each batch is dropped.
        let rows = Rows {
            batches: alloc::vec![
                alloc::vec![
                    (Some(1), Some(10)),
                    (Some(1), Some(1000)),
                    (None, Some(5)),
                    (Some(2), None)
                ],
                alloc::vec![(Some(1), Some(30)), (None, None), (Some(2), None)],
            ],
            dropped: alloc::vec![1],
        };
        let expected = [
            // `k` null: one row, of 5.
            alloc::vec![None, Some(1), Some(1), Some(5), Some(5), Some(5), Some(bits(5.0))],
            // `k` = 1: 10 and 30.
            alloc::vec![Some(1), Some(2), Some(2), Some(40), Some(10), Some(30), Some(bits(20.0))],
            // `k` = 2: only nulls.
            alloc::vec![Some(2), Some(2), Some(0), None, None, None, None],
        ];
        assert_eq!(aggregated(rows, true), expected);
    }

    #[test]
    fn aggregates_no_rows_as_one_row_without_keys() {
        let rows = Rows { batches: Vec::new(), dropped: Vec::new() };
        assert_eq!(
            aggregated(rows, false),
            [alloc::vec![Some(0), Some(0), None, None, None, None]]
        );
        let rows = Rows { batches: Vec::new(), dropped: Vec::new() };
        assert!(aggregated(rows, true).is_empty());
    }

    #[test]
    fn merged_states_hold_every_group() {
        // Enough groups to grow the table several times, half of them in
        // each state, and some in both.
        let batch = |range: core::ops::Range<i64>| -> Vec<(Option<i64>, Option<i64>)> {
            range.map(|k| (Some(k), Some(1))).collect()
        };
        let breaker = HashAggregate::new(&Heap, grouping(true)).unwrap();
        let mut context = Context::new(&Heap);
        let consumed = |context: &mut Context, range| {
            let mut state = Breaker::new_state(&breaker, context).unwrap();
            let rows = Rows { batches: alloc::vec![batch(range)], dropped: Vec::new() };
            let mut source = Source::new_state(&rows, context).unwrap();
            let mut input = RowBatch::new();
            while rows.next(context, &mut source, &mut input).unwrap() {
                breaker.consume(context, &mut state, &input).unwrap();
            }
            state
        };
        let (mut first, second) =
            (consumed(&mut context, 0..1500), consumed(&mut context, 1000..2000));
        breaker.merge(&mut context, &mut first, second).unwrap();
        assert_eq!(first.table.len(), 2000);
        let mut output = RowBatch::new();
        let mut counts = Vec::new();
        loop {
            let more = breaker.serve(&mut context, &mut first, &mut output).unwrap();
            for row in 0..output.row_count() as usize {
                counts.push((output.column(0).int64s()[row], output.column(1).int64s()[row]));
            }
            if !more {
                break;
            }
        }
        counts.sort_unstable();
        let expected: Vec<_> =
            (0..2000).map(|k| (k, if (1000..1500).contains(&k) { 2 } else { 1 })).collect();
        assert_eq!(counts, expected);
    }

    /// An integer column of `values`, null where `None`, within `bounds` if
    /// given.
    fn bounded(values: &[Option<i64>], bounds: Option<(i64, i64)>) -> ColumnView {
        let column = column(DataType::Int64, values);
        match bounds {
            Some((min, max)) => column.with_bounds(Bounds { min, max }),
            None => column,
        }
    }

    #[test]
    fn finds_a_keys_group_however_its_batch_comes() {
        let one =
            |value| ColumnView::constant(&mut Context::new(&Heap), &bounded(&[value], None), 3);
        let batches = alloc::vec![
            // One key for the batch, then within bounds, then not, then nullable.
            alloc::vec![one(Some(7)).unwrap(), bounded(&[Some(1), Some(2), Some(3)], None)],
            alloc::vec![
                bounded(&[Some(7), Some(8)], Some((7, 8))),
                bounded(&[Some(10), Some(20)], None)
            ],
            alloc::vec![bounded(&[Some(7), Some(9)], None), bounded(&[Some(100), Some(200)], None)],
            alloc::vec![bounded(&[Some(7), None], None), bounded(&[Some(1000), Some(2000)], None)],
        ];
        let count_and_sum = |keys: &[u32]| {
            let mut grouping = grouping_by(keys, DataType::Int64);
            grouping.aggregates.retain(|aggregate| {
                aggregate.input.is_none() || aggregate.function == Function::Sum
            });
            run(Columns(batches.clone()), grouping)
        };
        let expected = [
            alloc::vec![None, Some(1), Some(2000)],
            alloc::vec![Some(7), Some(6), Some(1116)],
            alloc::vec![Some(8), Some(1), Some(20)],
            alloc::vec![Some(9), Some(1), Some(200)],
        ];
        assert_eq!(count_and_sum(&[0]), expected);
        // By both columns, every row is a group of its own.
        let pairs = count_and_sum(&[0, 1]);
        assert_eq!(pairs.len(), 9);
        assert!(pairs.iter().all(|row| row[2] == Some(1) && row[3] == row[1]));
    }

    #[test]
    fn sums_stay_exact_when_they_widen() {
        // One word holds the first batch's sum, not the second's, which
        // passes an `Int64` until the third.
        let half = i64::MAX / 2;
        let x = |values: &[i64], bounds| {
            let values: Vec<_> = values.iter().copied().map(Some).collect();
            alloc::vec![
                bounded(&alloc::vec![Some(1); values.len()], None),
                bounded(&values, bounds)
            ]
        };
        let batches = alloc::vec![
            x(&[half, half], Some((half, half))),
            x(&[half], Some((half, half))),
            x(&[-half, -half], None),
        ];
        #[expect(clippy::cast_precision_loss, reason = "as `AVG` does")]
        let average = bits(half as f64 / 5.0);
        let totals = [Some(5), Some(5), Some(half), Some(-half), Some(half), Some(average)];
        assert_eq!(run(Columns(batches.clone()), grouping(false)), [totals.to_vec()]);
        let grouped = [&[Some(1)][..], &totals].concat();
        assert_eq!(run(Columns(batches), grouping(true)), [grouped]);
    }

    #[test]
    fn aggregates_floats() {
        let x: Vec<_> = [Some(1.5), None, Some(-2.0), None].map(|x| x.map(bits)).into();
        let batches = alloc::vec![alloc::vec![
            bounded(&[Some(1), Some(1), Some(1), Some(2)], None),
            column(DataType::Float64, &x),
        ]];
        assert_eq!(
            run(Columns(batches), grouping_by(&[0], DataType::Float64)),
            [
                alloc::vec![
                    Some(1),
                    Some(3),
                    Some(2),
                    Some(bits(-0.5)),
                    Some(bits(-2.0)),
                    Some(bits(1.5)),
                    Some(bits(-0.25))
                ],
                alloc::vec![Some(2), Some(1), Some(0), None, None, None, None],
            ]
        );
    }

    #[test]
    fn merges_groups_whether_or_not_their_values_were_null() {
        // The first state's values are never null, the second's are.
        let breaker = HashAggregate::new(&Heap, grouping(true)).unwrap();
        let mut context = Context::new(&Heap);
        let consumed = |context: &mut Context, rows: Vec<(Option<i64>, Option<i64>)>| {
            let mut state = Breaker::new_state(&breaker, context).unwrap();
            let rows = Rows { batches: alloc::vec![rows], dropped: Vec::new() };
            let mut source = Source::new_state(&rows, context).unwrap();
            let mut input = RowBatch::new();
            while rows.next(context, &mut source, &mut input).unwrap() {
                breaker.consume(context, &mut state, &input).unwrap();
            }
            state
        };
        let mut first = consumed(&mut context, alloc::vec![(Some(1), Some(5)), (Some(2), Some(6))]);
        let second = consumed(
            &mut context,
            alloc::vec![(Some(1), None), (Some(3), Some(7)), (Some(4), None)],
        );
        breaker.merge(&mut context, &mut first, second).unwrap();
        Breaker::finalize(&breaker, &mut context, &mut first).unwrap();
        let mut output = RowBatch::new();
        assert!(!breaker.serve(&mut context, &mut first, &mut output).unwrap());
        let sums: Vec<_> = (0..output.row_count())
            .map(|row| (output.column(0).int64s()[row as usize], !output.column(3).is_null(row)))
            .collect();
        assert_eq!(sums, [(1, true), (2, true), (3, true), (4, false)]);
        let sum = output.column(3);
        assert_eq!((sum.int64s()[0], sum.int64s()[2]), (5, 7));
    }
}
