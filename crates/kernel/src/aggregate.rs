//! `AggregateOp`: aggregates, such as `COUNT(*)` and `SUM(x)`, over each
//! group of its child's rows with the same keys, or over all of them if it
//! has none; and `hash_aggregate`, the breaker that runs it.

use crate::allocator::{AllocError, Allocator};
use crate::buffer::Primitive;
use crate::column::{Bounds, ColumnView, DataType, Form, Forms, Validity};
use crate::context::Context;
use crate::error::Error;
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::plan::{ColumnId, NamedColumn, Op, PlanNode};
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::selection::{Kept, Selection};
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

    fn reads(&self, reads: &mut Needed) {
        for &key in self.keys.iter() {
            reads.need(key);
        }
        for input in self.aggregates.iter().filter_map(|aggregate| aggregate.input) {
            reads.need(input);
        }
    }

    /// Constants, whose one value it reads once: a batch whose keys are all
    /// constant is one group, found once.
    fn accepts(&self, _: ColumnId) -> Forms {
        Forms::FLAT | Forms::CONSTANT
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
    check!(grouping.keys.len() <= KEYS_MAX);
    DynOperator::new(allocator, HashAggregate { grouping })
}

/// The most keys a grouping can have: one bit each says which are null.
const KEYS_MAX: usize = 64;

/// The most groups an aggregation can hold.
const GROUPS_MAX: usize = 1 << 26;

/// Groups rows a batch at a time, in a Swiss table: it hashes every kept
/// row's keys, then finds each row's group, then folds each aggregate's
/// column into its groups' states.
struct HashAggregate {
    grouping: Grouping,
}

/// The groups so far. Each list of something for each group has room for
/// as many as the table can take before it grows, so adding a group writes
/// into each rather than growing it.
struct Groups {
    /// Each group's keys, and the table that finds a group by them.
    table: Table,
    /// How many rows each group has.
    rows: SlowVec<u64>,
    /// Each aggregate's states.
    states: SlowVec<States>,
    /// What a batch's rows need on the way to their groups.
    scratch: Scratch,
    /// How many groups have been output.
    served: usize,
}

/// An aggregate's state for each group.
struct States {
    /// How many values each group has that aren't null, or `None` while no
    /// value has been null: then it's each group's row count.
    counts: Option<SlowVec<u64>>,
    /// Each group's sum, least or greatest so far.
    totals: Totals,
}

/// Each group's total, by what the aggregate keeps.
enum Totals {
    /// None: `COUNT` keeps only counts.
    None,
    /// The least or greatest integer.
    Int(SlowVec<i64>),
    /// The sum of integers, in 64 bits, while no sum can overflow them:
    /// and how much more the magnitudes of the values summed can add up to
    /// before one could.
    Sum(SlowVec<i64>, u64),
    /// The sum of integers, in 128 bits, so none can overflow, for when 64
    /// bits might not do.
    Wide(SlowVec<i128>),
    /// The sum, least or greatest float. Least and greatest start as NaN,
    /// which `f64::min` and `f64::max` pass over.
    Float(SlowVec<f64>),
}

/// Room for a batch's worth of what each kept row needs on the way to its
/// group.
struct Scratch {
    /// Copies of the keys that can't be read in place, `BATCH_ROWS_MAX` a
    /// key.
    keys: SlowVec<i64>,
    /// Which of each row's keys are null, a bit each.
    nulls: SlowVec<u64>,
    /// Each row's hash.
    hashes: SlowVec<u64>,
    /// Each row's group.
    ids: SlowVec<u32>,
    /// Where each run of rows of the same group starts, where runs are
    /// found, then where the last ends.
    starts: SlowVec<u16>,
}

/// Groups by their keys: a Swiss table, as abseil's and hashbrown's. Each
/// slot has a control byte, holding 7 bits of its group's hash, and the
/// table is probed a word of `WIDTH` control bytes at a time, so most
/// slots that can't hold the group are skipped without reading its keys.
struct Table {
    /// A byte for each slot: 0 if empty, else 0x80 and the top 7 bits of
    /// its group's hash. Then a copy of the first `WIDTH`, so a word can be
    /// read from any slot.
    controls: SlowVec<u8>,
    /// The group in each slot that isn't empty.
    slots: SlowVec<u32>,
    /// Each group's keys, a list for each key.
    keys: SlowVec<SlowVec<i64>>,
    /// Which of each group's keys are null, a bit each.
    nulls: SlowVec<u64>,
    /// How many groups there are.
    len: usize,
    /// Whether any group has a key that's null.
    any_null: bool,
}

/// The groups of a batch's kept rows.
#[derive(Clone, Copy)]
enum Ids<'a> {
    /// Every row's is this one.
    One(u32),
    /// Each row's, in order.
    Each(&'a [u32]),
    /// Each run's, of runs of rows of the same group, in order, each
    /// starting at one of these and ending at the next, the first at 0 and
    /// the last at the number of rows.
    Runs(&'a [u32], &'a [u16]),
}

/// How a batch's kept rows were grouped, into `Scratch`.
#[derive(Clone, Copy)]
enum Grouped {
    /// Every row is in this group.
    One(u32),
    /// Each row's group is in `ids`.
    Each,
    /// This many runs of rows of the same group: each run's group is in
    /// `ids`, and it starts at one of `starts` and ends at the next.
    Runs(usize),
}

/// How many control bytes are probed at a time: a word's.
const WIDTH: usize = 8;

/// A byte of 1 in each byte of a word.
const LOW: u64 = 0x0101_0101_0101_0101;

/// The top bit of each byte of a word.
const HIGH: u64 = 0x8080_8080_8080_8080;

/// The slots a table starts with.
const SLOTS_MIN: usize = 16;

impl Table {
    fn new(allocator: &dyn Allocator, width: usize) -> Result<Table, AllocError> {
        let mut keys = SlowVec::fixed(allocator, width.max(1))?;
        for _ in 0..width {
            keys.push(SlowVec::new(allocator, GROUPS_MAX)?).map_err(|_| AllocError)?;
        }
        let mut table = Table {
            controls: SlowVec::new(allocator, 1)?,
            slots: SlowVec::new(allocator, 1)?,
            keys,
            nulls: SlowVec::new(allocator, GROUPS_MAX)?,
            len: 0,
            any_null: false,
        };
        table.grow(allocator, SLOTS_MIN)?;
        Ok(table)
    }

    /// How many groups fit before the table grows: 7/8 of its slots.
    fn room(&self) -> usize {
        self.slots.len() / 8 * 7
    }

    /// Finds the group of each of `ids`' rows, whose keys are `keys`, a
    /// list for each key, and `nulls`, if any are, adding groups for keys
    /// not seen before, with room for each row's hash in `hashes`. There must
    /// be room for the groups. Either it finds runs of rows of the same group
    /// and returns how many, with each run's group in `ids` and where it
    /// starts in `starts`, then where the last ends; or it returns `None`,
    /// with each row's group in `ids`.
    fn assign(
        &mut self,
        keys: &[&[i64]],
        nulls: Option<&[u64]>,
        hashes: &mut [u64],
        ids: &mut [u32],
        starts: &mut [u16],
    ) -> Option<usize> {
        let rows = ids.len();
        check!(self.len + rows <= self.room() && hashes.len() == rows && starts.len() > rows);
        check!(keys.iter().all(|keys| keys.len() >= rows) && nulls.is_none_or(|n| n.len() >= rows));
        match keys {
            [key] if nulls.is_none() && !self.any_null => {
                Some(self.assign_unnulled(&[key], ids, starts))
            }
            [a, b] if nulls.is_none() && !self.any_null => {
                Some(self.assign_unnulled(&[a, b], ids, starts))
            }
            _ => {
                self.assign_with(keys, nulls, hashes, ids, |table, group, row| {
                    let null = nulls.map_or(0, |nulls| *at!(nulls, row));
                    *at!(table.nulls, group) == null
                        && table
                            .keys
                            .iter()
                            .zip(keys)
                            .all(|(ours, keys)| at!(ours, group) == at!(keys, row))
                });
                None
            }
        }
    }

    /// As `assign`, for `N` keys, where no key is null in any row or group,
    /// finding runs: a row is hashed only if its keys aren't the last row's.
    fn assign_unnulled<const N: usize>(
        &mut self,
        keys: &[&[i64]; N],
        ids: &mut [u32],
        starts: &mut [u16],
    ) -> usize {
        let key = |k: usize, row: usize| {
            // SAFETY: there are `N` keys; `row` is one of `hashes`', and
            // `assign` checked there are as many keys.
            unsafe { *keys.get_unchecked(k).get_unchecked(row) }
        };
        let rows = ids.len();
        let (mut row, mut runs) = (0, 0);
        while row < rows {
            let hash = (0..N).fold(SEED, |hash, k| mix(hash, key(k, row)));
            let found = self.find(hash, |group| {
                (0..N).all(|k| {
                    // SAFETY: the group is in a slot, so below `len`, within
                    // the room each list of keys has.
                    let ours = unsafe { *self.keys.get_unchecked(k).get_unchecked(group as usize) };
                    ours == key(k, row)
                })
            });
            // `runs` is at most `row`, so this writes over no row's id.
            *at_mut!(ids, runs) = match found {
                Ok(group) => group,
                Err(slot) => self.insert_unnulled(slot, hash, keys, row),
            };
            #[expect(clippy::cast_possible_truncation, reason = "a row")]
            let start = row as u16;
            *at_mut!(starts, runs) = start;
            runs += 1;
            // Rows often come in runs of the same keys, as from sorted data.
            let first = row;
            row += 1;
            while row < rows && (0..N).all(|k| key(k, row) == key(k, first)) {
                row += 1;
            }
        }
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
        let end = rows as u16;
        *at_mut!(starts, runs) = end;
        runs
    }

    /// Adds a group with row `row`'s keys, none null, hashed as `hash`, in
    /// `slot`.
    #[cold]
    #[inline(never)]
    fn insert_unnulled<const N: usize>(
        &mut self,
        slot: usize,
        hash: u64,
        keys: &[&[i64]; N],
        row: usize,
    ) -> u32 {
        let group = self.len;
        for (ours, keys) in self.keys.iter_mut().zip(keys) {
            *at_mut!(ours, group) = *at!(keys, row);
        }
        // Each list of nulls holds 0 until a group's set.
        self.len += 1;
        self.set(slot, hash, group);
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let group = group as u32;
        group
    }

    /// As `assign`, with `same` saying whether a group has a row's keys.
    #[inline]
    fn assign_with(
        &mut self,
        keys: &[&[i64]],
        nulls: Option<&[u64]>,
        hashes: &mut [u64],
        ids: &mut [u32],
        same: impl Fn(&Table, usize, usize) -> bool,
    ) {
        hash_rows(keys, nulls, hashes);
        for (row, (&hash, id)) in hashes.iter().zip(ids.iter_mut()).enumerate() {
            *id = match self.find(hash, |group| same(self, group as usize, row)) {
                Ok(group) => group,
                Err(slot) => self.insert(slot, hash, keys, nulls, row),
            };
        }
    }

    /// Adds a group with row `row`'s keys, hashed as `hash`, in `slot`.
    fn insert(
        &mut self,
        slot: usize,
        hash: u64,
        keys: &[&[i64]],
        nulls: Option<&[u64]>,
        row: usize,
    ) -> u32 {
        let group = self.len;
        for (ours, keys) in self.keys.iter_mut().zip(keys) {
            *at_mut!(ours, group) = *at!(keys, row);
        }
        let null = nulls.map_or(0, |nulls| *at!(nulls, row));
        *at_mut!(self.nulls, group) = null;
        self.any_null |= null != 0;
        self.len += 1;
        self.set(slot, hash, group);
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let group = group as u32;
        group
    }

    /// The group hashed as `hash` that `same` says has the keys sought, or
    /// else the empty slot where it goes.
    #[inline]
    fn find(&self, hash: u64, same: impl Fn(u32) -> bool) -> Result<u32, usize> {
        let mask = self.slots.len() - 1;
        let control = LOW * (0x80 | hash >> 57);
        #[expect(clippy::cast_possible_truncation, reason = "masked")]
        let mut at = hash as usize & mask;
        let mut stride = 0;
        loop {
            let word = self.word(at);
            // A high bit for each byte that's `control`, and maybe for some
            // just above one that is, which `same` turns down.
            let x = word ^ control;
            let mut matches = x.wrapping_sub(LOW) & !x & HIGH;
            while matches != 0 {
                let slot = (at + matches.trailing_zeros() as usize / 8) & mask;
                // SAFETY: masked, so below the number of slots.
                let group = unsafe { *self.slots.get_unchecked(slot) };
                if same(group) {
                    return Ok(group);
                }
                matches &= matches - 1;
            }
            let empty = !word & HIGH;
            if empty != 0 {
                return Err((at + empty.trailing_zeros() as usize / 8) & mask);
            }
            // Triangular steps of whole words visit every word once.
            stride += WIDTH;
            at = (at + stride) & mask;
        }
    }

    /// The `WIDTH` control bytes from slot `at`, the first lowest.
    #[inline]
    fn word(&self, at: usize) -> u64 {
        // SAFETY: `at` is a slot, and `grow` made room for a word from any.
        let bytes = unsafe { self.controls.get_unchecked(at..at + WIDTH) };
        let Ok(bytes) = <[u8; WIDTH]>::try_from(bytes) else { crate::check::check_failed(line!()) };
        u64::from_le_bytes(bytes)
    }

    /// Puts `group`, hashed as `hash`, in `slot`, which is empty.
    #[inline]
    fn set(&mut self, slot: usize, hash: u64, group: usize) {
        let mask = self.slots.len() - 1;
        let slot = slot & mask;
        let control = control(hash);
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let group = group as u32;
        // SAFETY: masked, `slot` is a slot, and `grow` made a control byte
        // for each, then a copy of the first `WIDTH`.
        unsafe {
            *self.slots.get_unchecked_mut(slot) = group;
            *self.controls.get_unchecked_mut(slot) = control;
            // The copy, for the first `WIDTH` slots; for the rest, the slot
            // again.
            *self.controls.get_unchecked_mut((slot.wrapping_sub(WIDTH) & mask) + WIDTH) = control;
        }
    }

    /// Moves every group into `slots` new slots, and makes room in its lists
    /// for as many groups as they take.
    #[cold]
    #[inline(never)]
    fn grow(&mut self, allocator: &dyn Allocator, slots: usize) -> Result<(), AllocError> {
        self.controls = SlowVec::zeroed(allocator, slots + WIDTH)?;
        self.slots = SlowVec::zeroed(allocator, slots)?;
        check!(self.controls.len() == self.slots.len() + WIDTH && slots.is_power_of_two());
        let mut hashes = [0; 256];
        for first in (0..self.len).step_by(hashes.len()) {
            let groups = first..self.len.min(first + hashes.len());
            let mut keys: [&[i64]; KEYS_MAX] = [&[]; KEYS_MAX];
            for (keys, ours) in keys.iter_mut().zip(self.keys.iter()) {
                *keys = at!(ours, groups.clone());
            }
            let nulls = self.any_null.then(|| at!(self.nulls, groups.clone()));
            let hashes = at_mut!(hashes, ..groups.len());
            hash_rows(at!(keys, ..self.keys.len()), nulls, hashes);
            for (group, &hash) in groups.zip(hashes.iter()) {
                self.set(self.empty(hash), hash, group);
            }
        }
        let room = self.room();
        for keys in self.keys.iter_mut() {
            resize(keys, room, 0)?;
        }
        resize(&mut self.nulls, room, 0)
    }

    /// The first empty slot of those a group hashed as `hash` may be in.
    fn empty(&self, hash: u64) -> usize {
        let mask = self.slots.len() - 1;
        #[expect(clippy::cast_possible_truncation, reason = "masked")]
        let mut at = hash as usize & mask;
        let mut stride = 0;
        loop {
            let empty = !self.word(at) & HIGH;
            if empty != 0 {
                return (at + empty.trailing_zeros() as usize / 8) & mask;
            }
            stride += WIDTH;
            at = (at + stride) & mask;
        }
    }

    /// The column of key `k` for groups `first..first + rows`.
    fn key(
        &self,
        context: &mut Context,
        k: usize,
        first: usize,
        rows: usize,
    ) -> Result<ColumnView, Error> {
        let mut values = context.values_buffer(rows * 8)?;
        values.as_mut_slice::<i64>().copy_from_slice(at!(at!(self.keys, k), first..first + rows));
        let nulls = at!(self.nulls, first..first + rows);
        let mut validity = None;
        if nulls.iter().any(|&null| null >> k & 1 != 0) {
            let mut bits = context.small_buffer(rows.div_ceil(8))?;
            let bytes = bits.as_mut_slice::<u8>();
            bytes.fill(0);
            for (i, &null) in nulls.iter().enumerate() {
                *at_mut!(bytes, i / 8) |= u8::from(null >> k & 1 == 0) << (i % 8);
            }
            validity = Some(bits);
        }
        Ok(ColumnView::new(context, DataType::Int64, values, validity)?)
    }
}

/// The control byte of a group hashed as `hash`: never 0, which is empty.
fn control(hash: u64) -> u8 {
    let top = (hash >> 57) as u8;
    0x80 | top
}

/// What a hash starts as, with the bits of the keys that are null.
const SEED: u64 = 0x243f_6a88_85a3_08d3;

/// `hash` with `key` mixed in: multiplied, the high half folded into the
/// low, so each bit of the result depends on every bit of both.
fn mix(hash: u64, key: i64) -> u64 {
    let product = u128::from(hash ^ key.cast_unsigned()) * 0x9e37_79b9_7f4a_7c15;
    #[expect(clippy::cast_possible_truncation, reason = "the low half")]
    let low = product as u64;
    let high = (product >> 64) as u64;
    low ^ high
}

/// Hashes each row of `keys`, a list for each key, and `nulls`, if any are,
/// into `hashes`, a key at a time.
fn hash_rows(keys: &[&[i64]], nulls: Option<&[u64]>, hashes: &mut [u64]) {
    let Some((first, rest)) = keys.split_first() else { crate::check::check_failed(line!()) };
    match nulls {
        None => {
            for (hash, &key) in hashes.iter_mut().zip(*first) {
                *hash = mix(SEED, key);
            }
        }
        Some(nulls) => {
            for ((hash, &key), &null) in hashes.iter_mut().zip(*first).zip(nulls) {
                *hash = mix(SEED ^ null, key);
            }
        }
    }
    for keys in rest {
        for (hash, &key) in hashes.iter_mut().zip(*keys) {
            *hash = mix(*hash, key);
        }
    }
}

/// Turns `ids` from the group of each of `runs` runs, which start at
/// `starts` and end at the next, into the group of each row.
fn spread(ids: &mut [u32], starts: &[u16], runs: usize) {
    // From the last run, so none's group is written over before it's read:
    // a run starts at or after its place.
    for run in (0..runs).rev() {
        let (start, end) = (*at!(starts, run), *at!(starts, run + 1));
        let id = *at!(ids, run);
        at_mut!(ids, usize::from(start)..usize::from(end)).fill(id);
    }
}

/// Calls `f` with each row `selection` keeps, and its place among them.
#[inline]
fn for_kept(selection: &Selection, mut f: impl FnMut(usize, usize)) {
    match selection.kept() {
        Kept::All => (0..selection.len() as usize).for_each(|row| f(row, row)),
        Kept::Select(rows) => {
            rows.iter().enumerate().for_each(|(i, &row)| f(i, usize::from(row)));
        }
        Kept::None => {}
    }
}

/// Folds each row `selection` keeps, whose value `read` gives with whether
/// it isn't null, into the state of its group in `ids` with `op`. `states`
/// has room for every group.
#[inline]
fn fold<S: Copy, T>(
    states: &mut [S],
    ids: Ids,
    selection: &Selection,
    read: impl Fn(usize) -> (T, bool),
    op: impl Fn(S, T, bool) -> S,
) {
    let ids = match ids {
        // One run, rather than loops of its own.
        Ids::One(group) => {
            #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
            let rows = selection.len() as u16;
            return fold(states, Ids::Runs(&[group], &[0, rows]), selection, read, op);
        }
        Ids::Each(ids) => ids,
        Ids::Runs(ids, starts) => {
            check!(starts.last().is_some_and(|&end| u32::from(end) == selection.len()));
            for (&id, run) in ids.iter().zip(starts.windows(2)) {
                let (start, end) = (usize::from(*at!(run, 0)), usize::from(*at!(run, 1)));
                // SAFETY: each id is a group, which `states` has room for.
                let state = unsafe { states.get_unchecked_mut(id as usize) };
                let mut folded = *state;
                let mut fold_row = |row| {
                    let (value, valid) = read(row);
                    folded = op(folded, value, valid);
                };
                match selection.kept() {
                    Kept::All => (start..end).for_each(&mut fold_row),
                    Kept::Select(rows) => {
                        at!(rows, start..end).iter().for_each(|&row| fold_row(usize::from(row)));
                    }
                    Kept::None => {}
                }
                *state = folded;
            }
            return;
        }
    };
    check!(ids.len() == selection.len() as usize);
    let mut fold_row = |id: u32, row: usize| {
        let (value, valid) = read(row);
        // SAFETY: each id is a group, which `states` has room for.
        let state = unsafe { states.get_unchecked_mut(id as usize) };
        *state = op(*state, value, valid);
    };
    match selection.kept() {
        Kept::All => ids.iter().enumerate().for_each(|(row, &id)| fold_row(id, row)),
        Kept::Select(rows) => {
            rows.iter().zip(ids).for_each(|(&row, &id)| fold_row(id, usize::from(row)));
        }
        Kept::None => {}
    }
}

/// As `fold`, of `column`'s values, which are `T`s.
#[inline]
fn fold_column<S: Copy, T: Primitive>(
    states: &mut [S],
    ids: Ids,
    selection: &Selection,
    column: &ColumnView,
    op: impl Fn(S, T, bool) -> S,
) {
    match read(column, selection) {
        Read::Constant(value, valid) => fold(states, ids, selection, |_| (value, valid), op),
        Read::Flat(flat) if flat.validity.is_none() => {
            fold(states, ids, selection, |row| (flat.value(row), true), op);
        }
        Read::Flat(flat) => {
            fold(states, ids, selection, |row| (flat.value(row), flat.is_valid(row)), op);
        }
    }
}

/// `column`, an aggregate's input of `data_type`, with a constant made
/// flat, so rows of many groups fold it by the same loops as any, or
/// `None` if it's a constant null.
fn flat_input(
    context: &mut Context,
    column: &ColumnView,
    data_type: DataType,
    selection: &Selection,
) -> Result<Option<ColumnView>, AllocError> {
    let (value, valid) = match read::<i64>(column, selection) {
        Read::Flat(_) => return Ok(Some(column.clone())),
        Read::Constant(value, valid) => (value, valid),
    };
    if !valid {
        return Ok(None);
    }
    let mut values = context.values_buffer(selection.rows() as usize * 8)?;
    values.as_mut_slice::<i64>().fill(value);
    let flat = ColumnView::new(context, data_type, values, None)?;
    Ok(Some(match data_type {
        DataType::Int64 => flat.with_bounds(Bounds { min: value, max: value }),
        _ => flat,
    }))
}

/// The greatest magnitude of any value of `column`, an integer one, if
/// known.
fn magnitude(column: &ColumnView) -> Option<u64> {
    if column.is_in(Forms::CONSTANT) {
        return Some(at!(column.values().int64s(), 0).unsigned_abs());
    }
    let bounds = column.bounds()?;
    Some(bounds.min.unsigned_abs().max(bounds.max.unsigned_abs()))
}

/// `sums`, in 128 bits.
#[cold]
#[inline(never)]
fn widen(allocator: &dyn Allocator, sums: &[i64]) -> Result<SlowVec<i128>, AllocError> {
    let mut wide = SlowVec::new(allocator, GROUPS_MAX)?;
    for &sum in sums {
        wide.push(i128::from(sum)).map_err(|_| AllocError)?;
    }
    Ok(wide)
}

/// A column's values, for the rows a selection keeps.
enum Read<'c, T> {
    /// Each row's own.
    Flat(Flat<'c, T>),
    /// Every row's, and whether it isn't null.
    Constant(T, bool),
}

/// A flat column's values, read only for the rows a selection keeps.
#[derive(Clone, Copy)]
struct Flat<'c, T> {
    /// Each row's value.
    values: &'c [T],
    /// Which values aren't null, if any are.
    validity: Option<Validity<'c>>,
}

/// `column`'s values, which are `T`s, checked to cover every row
/// `selection` keeps.
fn read<'c, T: Primitive>(column: &'c ColumnView, selection: &Selection) -> Read<'c, T> {
    let values = column.values();
    let (words, validity) = (values.slice_of::<T>(), values.validity());
    match column.form() {
        Form::Flat => {
            // Every row kept is below `rows()`.
            check!(selection.rows() as usize <= words.len());
            Read::Flat(Flat { values: words, validity })
        }
        Form::Constant => Read::Constant(*at!(words, 0), !values.is_null(0)),
        Form::Dictionary(_) => crate::check::check_failed(line!()),
    }
}

impl<T: Primitive> Flat<'_, T> {
    /// Row `row`'s value, which is a row kept.
    #[inline]
    fn value(self, row: usize) -> T {
        // SAFETY: `read` checked each row kept is one of the values.
        unsafe { *self.values.get_unchecked(row) }
    }

    /// Whether row `row`, which is a row kept, isn't null.
    #[inline]
    fn is_valid(self, row: usize) -> bool {
        #[expect(clippy::cast_possible_truncation, reason = "a row")]
        let row = row as u32;
        // SAFETY: as in `value`; the validity has a bit for each value.
        self.validity.is_none_or(|validity| unsafe { validity.is_valid_unchecked(row) })
    }
}

/// Folds `theirs`, the states of groups `range` of another `Groups`, into
/// `ours`, those of groups `ids`, for `function`. `counted` is each side's
/// row count for each group.
fn merge_states(
    allocator: &dyn Allocator,
    function: Function,
    ours: &mut States,
    theirs: &States,
    counted: (&SlowVec<u64>, &SlowVec<u64>),
    ids: Ids,
    range: &core::ops::Range<usize>,
) -> Result<(), AllocError> {
    #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
    let all = &Selection::all(range.len() as u32);
    if ours.counts.is_some() || theirs.counts.is_some() {
        let counts = ours.counts(allocator, counted.0)?;
        let added = at!(theirs.counts.as_ref().unwrap_or(counted.1), range.clone());
        fold(counts, ids, all, |i| (*at!(added, i), true), |count, x, _| count + x);
    }
    if let Totals::Sum(sums, headroom) = &mut ours.totals {
        match theirs.totals {
            Totals::Sum(_, theirs) if i64::MAX.unsigned_abs() - theirs <= *headroom => {
                *headroom -= i64::MAX.unsigned_abs() - theirs;
            }
            _ => ours.totals = Totals::Wide(widen(allocator, sums)?),
        }
    }
    match (&mut ours.totals, &theirs.totals, function) {
        (Totals::Sum(ours, _), Totals::Sum(totals, _), _) => {
            let totals = at!(totals, range.clone());
            fold(ours, ids, all, |i| (*at!(totals, i), true), |sum, x, _| sum + x);
        }
        (Totals::Wide(ours), Totals::Sum(totals, _), _) => {
            let totals = at!(totals, range.clone());
            fold(ours, ids, all, |i| (*at!(totals, i), true), |sum, x, _| sum + i128::from(x));
        }
        (Totals::Wide(ours), Totals::Wide(totals), _) => {
            let totals = at!(totals, range.clone());
            fold(ours, ids, all, |i| (*at!(totals, i), true), |sum, x, _| sum + x);
        }
        (Totals::Int(ours), Totals::Int(totals), function) => {
            let read = |i| (*at!(totals, range.start + i), true);
            match function {
                Function::Min => fold(ours, ids, all, read, |a, b: i64, _| a.min(b)),
                _ => fold(ours, ids, all, read, |a, b: i64, _| a.max(b)),
            }
        }
        (Totals::Float(ours), Totals::Float(totals), function) => {
            let read = |i| (*at!(totals, range.start + i), true);
            match function {
                Function::Min => fold(ours, ids, all, read, |a, b: f64, _| a.min(b)),
                Function::Max => fold(ours, ids, all, read, |a, b: f64, _| a.max(b)),
                _ => fold(ours, ids, all, read, |a, b, _| a + b),
            }
        }
        _ => {}
    }
    Ok(())
}

/// Adds `value` to the end of `values` until it holds `len`.
fn resize<T: Copy>(values: &mut SlowVec<T>, len: usize, value: T) -> Result<(), AllocError> {
    let values_in_one_go = [value; 256];
    while values.len() < len {
        values.extend_from_slice(at!(values_in_one_go, ..(len - values.len()).min(256)))?;
    }
    Ok(())
}

impl States {
    /// How many values each group has that aren't null, kept from now on,
    /// given each group's row count, `counted`.
    fn counts(
        &mut self,
        allocator: &dyn Allocator,
        counted: &[u64],
    ) -> Result<&mut SlowVec<u64>, AllocError> {
        if self.counts.is_none() {
            let mut counts = SlowVec::new(allocator, GROUPS_MAX)?;
            counts.extend_from_slice(counted)?;
            self.counts = Some(counts);
        }
        let Some(counts) = &mut self.counts else { crate::check::check_failed(line!()) };
        Ok(counts)
    }
}

impl HashAggregate {
    /// Makes room for `more` groups in the table and every list of
    /// something for each group.
    fn reserve(
        &self,
        allocator: &dyn Allocator,
        groups: &mut Groups,
        more: usize,
    ) -> Result<(), Error> {
        let len = groups.table.len + more;
        if len <= groups.table.room() {
            return Ok(());
        }
        if len > GROUPS_MAX {
            return Err(Error::OutOfMemory);
        }
        groups.table.grow(allocator, (len * 8 / 7 + 1).next_power_of_two())?;
        Ok(self.resize_states(groups)?)
    }

    /// Makes room in every list of states for as many groups as the table
    /// has room for.
    fn resize_states(&self, groups: &mut Groups) -> Result<(), AllocError> {
        let room = groups.table.room();
        resize(&mut groups.rows, room, 0)?;
        for (aggregate, states) in self.grouping.aggregates.iter().zip(groups.states.iter_mut()) {
            if let Some(counts) = &mut states.counts {
                resize(counts, room, 0)?;
            }
            match &mut states.totals {
                Totals::None => {}
                Totals::Int(totals) => {
                    let min = aggregate.function == Function::Min;
                    resize(totals, room, if min { i64::MAX } else { i64::MIN })?;
                }
                Totals::Sum(totals, _) => resize(totals, room, 0)?,
                Totals::Wide(totals) => resize(totals, room, 0)?,
                Totals::Float(totals) => {
                    let sum = matches!(aggregate.function, Function::Sum | Function::Avg);
                    resize(totals, room, if sum { 0.0 } else { f64::NAN })?;
                }
            }
        }
        Ok(())
    }

    /// Finds the group of each row `input` keeps, adding groups for keys not
    /// seen before. There must be room for them.
    fn group(&self, groups: &mut Groups, input: &RowBatch) -> Grouped {
        let positions = self.grouping.keys.iter();
        let constant = positions.clone().all(|&k| input.column(k).is_in(Forms::CONSTANT));
        // Every row's keys are its first's: only one is read.
        let one = Selection::all(1);
        let selection = if constant { &one } else { input.selection() };
        let rows = selection.len() as usize;
        let Scratch { keys: copies, nulls, hashes, ids, starts } = &mut groups.scratch;
        let mut keys: [&[i64]; KEYS_MAX] = [&[]; KEYS_MAX];
        let mut any_null = false;
        let copies = copies.chunks_mut(BATCH_ROWS_MAX as usize);
        for ((k, &position), copy) in positions.enumerate().zip(copies) {
            let column = input.column(position);
            let copy = at_mut!(copy, ..rows);
            let flat = match read::<i64>(column, selection) {
                Read::Flat(flat) if flat.validity.is_none() && selection.kept() == Kept::All => {
                    *at_mut!(keys, k) = at!(flat.values, ..rows);
                    continue;
                }
                Read::Flat(flat) => flat,
                // Null keys are 0, so they compare and hash the same.
                Read::Constant(value, valid) => {
                    copy.fill(if valid { value } else { 0 });
                    if !valid {
                        if !any_null {
                            at_mut!(nulls, ..rows).fill(0);
                            any_null = true;
                        }
                        at_mut!(nulls, ..rows).iter_mut().for_each(|null| *null |= 1 << k);
                    }
                    *at_mut!(keys, k) = copy;
                    continue;
                }
            };
            if flat.validity.is_none() {
                for_kept(selection, |i, row| *at_mut!(copy, i) = flat.value(row));
            } else {
                if !any_null {
                    at_mut!(nulls, ..rows).fill(0);
                    any_null = true;
                }
                for_kept(selection, |i, row| {
                    let valid = flat.is_valid(row);
                    *at_mut!(copy, i) = if valid { flat.value(row) } else { 0 };
                    *at_mut!(nulls, i) |= u64::from(!valid) << k;
                });
            }
            *at_mut!(keys, k) = copy;
        }
        let keys = at!(keys, ..self.grouping.keys.len());
        let nulls = any_null.then_some(at!(nulls, ..rows));
        let (hashes, ids) = (at_mut!(hashes, ..rows), at_mut!(ids, ..rows));
        match groups.table.assign(keys, nulls, hashes, ids, starts) {
            _ if constant => Grouped::One(*at!(ids, 0)),
            // Folding a run at a time pays where runs are long enough.
            Some(runs) if runs * 2 <= rows => Grouped::Runs(runs),
            Some(runs) => {
                spread(ids, starts, runs);
                Grouped::Each
            }
            None => Grouped::Each,
        }
    }

    /// Folds `input`'s column for `aggregate` into `states`, given each
    /// group's row count before it, `counted`.
    fn update(
        context: &mut Context,
        aggregate: Lowered,
        states: &mut States,
        counted: &[u64],
        input: &RowBatch,
        ids: Ids,
    ) -> Result<(), AllocError> {
        let Some((position, data_type)) = aggregate.input else { return Ok(()) };
        let allocator = context.allocator();
        let selection = input.selection();
        let column = match ids {
            // One group folds a constant as fast as it counts rows.
            Ids::One(_) => Some(input.column(position).clone()),
            _ => flat_input(context, input.column(position), data_type, selection)?,
        };
        // A constant null adds no value, but counts are kept apart from rows.
        let Some(column) = &column else {
            states.counts(allocator, counted)?;
            return Ok(());
        };
        // Once a value's been null, counts are kept apart from rows.
        if column.has_nulls() || states.counts.is_some() {
            let counts = states.counts(allocator, counted)?;
            fold_column(counts, ids, selection, column, |count, _: u64, valid| {
                count + u64::from(valid)
            });
        }
        if let Totals::Sum(sums, headroom) = &mut states.totals {
            let added = magnitude(column)
                .and_then(|magnitude| magnitude.checked_mul(u64::from(selection.len())));
            match added {
                Some(added) if added <= *headroom => *headroom -= added,
                _ => states.totals = Totals::Wide(widen(allocator, sums)?),
            }
        }
        match (&mut states.totals, aggregate.function) {
            (Totals::None, _) => {}
            (Totals::Sum(sums, _), _) => {
                fold_column(sums, ids, selection, column, |sum, x: i64, valid| {
                    sum + if valid { x } else { 0 }
                });
            }
            (Totals::Wide(sums), _) => {
                fold_column(sums, ids, selection, column, |sum, x: i64, valid| {
                    sum + if valid { i128::from(x) } else { 0 }
                });
            }
            (Totals::Int(least), Function::Min) => {
                fold_column(least, ids, selection, column, |least: i64, x, valid| {
                    if valid { least.min(x) } else { least }
                });
            }
            (Totals::Int(greatest), _) => {
                fold_column(greatest, ids, selection, column, |greatest: i64, x, valid| {
                    if valid { greatest.max(x) } else { greatest }
                });
            }
            (Totals::Float(least), Function::Min) => {
                fold_column(least, ids, selection, column, |least: f64, x, valid| {
                    least.min(if valid { x } else { f64::NAN })
                });
            }
            (Totals::Float(greatest), Function::Max) => {
                fold_column(greatest, ids, selection, column, |greatest: f64, x, valid| {
                    greatest.max(if valid { x } else { f64::NAN })
                });
            }
            (Totals::Float(sums), _) => {
                fold_column(sums, ids, selection, column, |sum: f64, x, valid| {
                    sum + if valid { x } else { 0.0 }
                });
            }
        }
        Ok(())
    }

    /// The column of `aggregate`'s results for groups `first..first + rows`.
    fn result(
        &self,
        context: &mut Context,
        groups: &Groups,
        a: usize,
        first: usize,
        rows: usize,
    ) -> Result<ColumnView, Error> {
        let aggregate = *at!(self.grouping.aggregates, a);
        let data_type = aggregate.input.map(|(_, data_type)| data_type);
        let Some(output_type) = aggregate.function.output_type(data_type) else {
            crate::check::check_failed(line!());
        };
        let states = at!(groups.states, a);
        let counts = states.counts.as_ref().unwrap_or(&groups.rows);
        let mut values = context.values_buffer(rows * 8)?;
        let mut validity = context.small_buffer(rows.div_ceil(8))?;
        validity.as_mut_slice::<u8>().fill(0);
        let mut any_null = false;
        for (i, value) in values.as_mut_slice::<i64>().iter_mut().enumerate() {
            let group = first + i;
            let count = *at!(counts, group);
            #[expect(clippy::cast_precision_loss, reason = "averages are approximate")]
            let word = match (aggregate.function, &states.totals) {
                (Function::Count, _) => Some(i64::try_from(count).map_err(|_| Error::Unsupported)?),
                _ if count == 0 => None,
                (Function::Avg, Totals::Sum(sums, _)) => {
                    Some((*at!(sums, group) as f64 / count as f64).to_bits().cast_signed())
                }
                (Function::Avg, Totals::Wide(sums)) => {
                    Some((*at!(sums, group) as f64 / count as f64).to_bits().cast_signed())
                }
                (Function::Avg, Totals::Float(sums)) => {
                    Some((*at!(sums, group) / count as f64).to_bits().cast_signed())
                }
                // A sum that doesn't fit in an `Int64` would need a wider type.
                (_, Totals::Wide(sums)) => {
                    Some(i64::try_from(*at!(sums, group)).map_err(|_| Error::Unsupported)?)
                }
                (_, Totals::Int(totals) | Totals::Sum(totals, _)) => Some(*at!(totals, group)),
                (_, Totals::Float(totals)) => Some(at!(totals, group).to_bits().cast_signed()),
                (_, Totals::None) => crate::check::check_failed(line!()),
            };
            *value = word.unwrap_or(0);
            match word {
                Some(_) => *at_mut!(validity.as_mut_slice::<u8>(), i / 8) |= 1 << (i % 8),
                None => any_null = true,
            }
        }
        Ok(ColumnView::new(context, output_type, values, any_null.then_some(validity))?)
    }
}

impl Breaker for HashAggregate {
    type State = Groups;

    fn new_state(&self, context: &mut Context) -> Result<Groups, Error> {
        let allocator = context.allocator();
        let width = self.grouping.keys.len();
        let mut states = SlowVec::fixed(allocator, self.grouping.aggregates.len().max(1))?;
        for aggregate in self.grouping.aggregates.iter() {
            let integers =
                aggregate.input.is_some_and(|(_, data_type)| data_type == DataType::Int64);
            let totals = match aggregate.function {
                Function::Count => Totals::None,
                Function::Sum | Function::Avg if integers => {
                    Totals::Sum(SlowVec::new(allocator, GROUPS_MAX)?, i64::MAX.unsigned_abs())
                }
                Function::Min | Function::Max if integers => {
                    Totals::Int(SlowVec::new(allocator, GROUPS_MAX)?)
                }
                _ => Totals::Float(SlowVec::new(allocator, GROUPS_MAX)?),
            };
            states.push(States { counts: None, totals }).map_err(|_| AllocError)?;
        }
        let batch = BATCH_ROWS_MAX as usize;
        let mut groups = Groups {
            table: Table::new(allocator, width)?,
            rows: SlowVec::new(allocator, GROUPS_MAX)?,
            states,
            scratch: Scratch {
                keys: SlowVec::zeroed(allocator, width * batch)?,
                nulls: SlowVec::zeroed(allocator, batch)?,
                hashes: SlowVec::zeroed(allocator, batch)?,
                ids: SlowVec::zeroed(allocator, batch)?,
                starts: SlowVec::fixed_from(allocator, core::iter::repeat_n(0, batch + 1))?,
            },
            served: 0,
        };
        self.resize_states(&mut groups)?;
        // All rows are one group, which is there even if there are none.
        if width == 0 {
            groups.table.len = 1;
        }
        Ok(groups)
    }

    fn consume(
        &self,
        context: &mut Context,
        groups: &mut Groups,
        input: &RowBatch,
    ) -> Result<(), Error> {
        let allocator = context.allocator();
        let selection = input.selection();
        if selection.is_empty() {
            return Ok(());
        }
        let grouped = if self.grouping.keys.is_empty() {
            Grouped::One(0)
        } else {
            self.reserve(allocator, groups, selection.len() as usize)?;
            self.group(groups, input)
        };
        let ids = at!(groups.scratch.ids, ..selection.len() as usize);
        let ids = match grouped {
            Grouped::One(group) => Ids::One(group),
            Grouped::Each => Ids::Each(ids),
            Grouped::Runs(runs) => Ids::Runs(at!(ids, ..runs), at!(groups.scratch.starts, ..=runs)),
        };
        for (&aggregate, states) in self.grouping.aggregates.iter().zip(groups.states.iter_mut()) {
            HashAggregate::update(context, aggregate, states, &groups.rows, input, ids)?;
        }
        fold(&mut groups.rows, ids, selection, |_| ((), true), |count, (), _| count + 1);
        Ok(())
    }

    fn merge(
        &self,
        context: &mut Context,
        groups: &mut Groups,
        other: Groups,
    ) -> Result<(), Error> {
        let allocator = context.allocator();
        let width = self.grouping.keys.len();
        let batch = BATCH_ROWS_MAX as usize;
        for first in (0..other.table.len).step_by(batch) {
            let rows = (other.table.len - first).min(batch);
            let theirs = first..first + rows;
            let ids = if width == 0 {
                Ids::One(0)
            } else {
                self.reserve(allocator, groups, rows)?;
                let mut keys: [&[i64]; KEYS_MAX] = [&[]; KEYS_MAX];
                for (keys, theirs_by_key) in keys.iter_mut().zip(other.table.keys.iter()) {
                    *keys = at!(theirs_by_key, theirs.clone());
                }
                let keys = at!(keys, ..width);
                let nulls = Some(at!(other.table.nulls, theirs.clone()));
                let Scratch { hashes, ids, starts, .. } = &mut groups.scratch;
                let (hashes, ids) = (at_mut!(hashes, ..rows), at_mut!(ids, ..rows));
                if let Some(runs) = groups.table.assign(keys, nulls, hashes, ids, starts) {
                    spread(ids, starts, runs);
                }
                Ids::Each(ids)
            };
            let aggregates = self.grouping.aggregates.iter().map(|aggregate| aggregate.function);
            for ((function, ours), theirs_states) in
                aggregates.zip(groups.states.iter_mut()).zip(other.states.iter())
            {
                let counted = (&groups.rows, &other.rows);
                merge_states(allocator, function, ours, theirs_states, counted, ids, &theirs)?;
            }
            let counted = at!(other.rows, theirs);
            #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
            let all = &Selection::all(rows as u32);
            fold(&mut groups.rows, ids, all, |i| (*at!(counted, i), true), |a, b, _| a + b);
        }
        Ok(())
    }

    fn finalize(&self, _: &mut Context, _: &mut Groups) -> Result<(), Error> {
        Ok(())
    }

    fn serve(
        &self,
        context: &mut Context,
        groups: &mut Groups,
        output: &mut RowBatch,
    ) -> Result<bool, Error> {
        let first = groups.served;
        let rows = (groups.table.len - first).min(BATCH_ROWS_MAX as usize);
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
        output.reset(rows as u32);
        for k in 0..self.grouping.keys.len() {
            let column = groups.table.key(context, k, first, rows)?;
            let Ok(()) = output.push_column(column) else { crate::check::check_failed(line!()) };
        }
        for a in 0..self.grouping.aggregates.len() {
            let column = self.result(context, groups, a, first, rows)?;
            let Ok(()) = output.push_column(column) else { crate::check::check_failed(line!()) };
        }
        groups.served += rows;
        Ok(groups.served < groups.table.len)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::Bounds;
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

    /// Groups by `k` if `grouped`, with `COUNT(*)`, `COUNT(x)`, `SUM(x)`,
    /// `MIN(x)`, `MAX(x)` and `AVG(x)`.
    fn grouping(grouped: bool) -> Grouping {
        let keys: &[u32] = if grouped { &[0] } else { &[] };
        let x = Some((1, DataType::Int64));
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
        let mut steps = SlowVec::new(&Heap, 1).unwrap();
        assert!(
            steps.push(Step::Operator(hash_aggregate(&Heap, grouping(grouped)).unwrap())).is_ok()
        );
        let pipeline = Pipeline::new(crate::step::DynSource::new(&Heap, rows).unwrap(), steps);
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
        let breaker = HashAggregate { grouping: grouping(true) };
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
        assert_eq!(first.table.len, 2000);
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

    /// Batches of columns, each keeping all but the rows in `dropped`.
    struct Columns {
        batches: Vec<Vec<ColumnView>>,
        dropped: Vec<u16>,
    }

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
            let Some(columns) = self.batches.get(*i) else { return Ok(false) };
            *i += 1;
            batch.reset(columns[0].row_count());
            for column in columns {
                assert!(batch.push_column(column.clone()).is_ok());
            }
            batch.selection_mut().retain(|row| !self.dropped.contains(&row));
            Ok(true)
        }
    }

    /// Each row out of `grouping` over `columns`, as `aggregated` gives
    /// them, sorted. If `apart`, each batch is consumed into a state of its
    /// own, and the states merged; if not, all into one, merged into an
    /// empty one.
    fn merged(grouping: Grouping, columns: &Columns, apart: bool) -> Vec<Vec<Option<i64>>> {
        let breaker = HashAggregate { grouping };
        let mut context = Context::new(&Heap);
        let mut source = Source::new_state(columns, &mut context).unwrap();
        let mut input = RowBatch::new();
        let mut merged = Breaker::new_state(&breaker, &mut context).unwrap();
        let mut state = Breaker::new_state(&breaker, &mut context).unwrap();
        while columns.next(&mut context, &mut source, &mut input).unwrap() {
            breaker.consume(&mut context, &mut state, &input).unwrap();
            if apart {
                let consumed = Breaker::new_state(&breaker, &mut context).unwrap();
                let consumed = core::mem::replace(&mut state, consumed);
                breaker.merge(&mut context, &mut merged, consumed).unwrap();
            }
        }
        breaker.merge(&mut context, &mut merged, state).unwrap();
        let mut output = RowBatch::new();
        let mut out = Vec::new();
        loop {
            let more = breaker.serve(&mut context, &mut merged, &mut output).unwrap();
            for row in 0..output.row_count() {
                let values = (0..output.column_count()).map(|i| {
                    let column = output.column(i);
                    (!column.is_null(row)).then(|| column.int64s()[row as usize])
                });
                out.push(values.collect());
            }
            if !more {
                break;
            }
        }
        out.sort();
        out
    }

    /// A grouping by `keys` of `COUNT(*)`, then each of `functions` of `x`,
    /// an integer column.
    fn by(keys: &[u32], x: u32, functions: &[Function]) -> Grouping {
        let x = Some((x, DataType::Int64));
        let mut aggregates = alloc::vec![Lowered { function: Function::Count, input: None }];
        aggregates.extend(functions.iter().map(|&function| Lowered { function, input: x }));
        Grouping {
            keys: SlowVec::fixed_from(&Heap, keys.iter().copied()).unwrap(),
            aggregates: SlowVec::fixed_from(&Heap, aggregates.into_iter()).unwrap(),
        }
    }

    /// Each group of the rows of `k1`, `k2` and `x` in `batches`, but row
    /// `dropped` of each: its keys, then its rows, values, and the values'
    /// sum, least and greatest, sorted.
    fn expected(batches: &[&[[Option<i64>; 3]]], dropped: usize) -> Vec<Vec<Option<i64>>> {
        let mut expected: Vec<Vec<Option<i64>>> = Vec::new();
        for (i, &[k1, k2, x]) in batches.iter().flat_map(|rows| rows.iter().enumerate()) {
            if i == dropped {
                continue;
            }
            if !expected.iter().any(|group| group[..2] == [k1, k2]) {
                expected.push(alloc::vec![k1, k2, Some(0), Some(0), None, None, None]);
            }
            let group = expected.iter_mut().find(|group| group[..2] == [k1, k2]).unwrap();
            group[2] = group[2].map(|n| n + 1);
            if let Some(x) = x {
                group[3] = group[3].map(|n| n + 1);
                group[4] = Some(group[4].unwrap_or(0) + x);
                group[5] = Some(group[5].map_or(x, |least| least.min(x)));
                group[6] = Some(group[6].map_or(x, |greatest| greatest.max(x)));
            }
        }
        expected.sort();
        expected
    }

    #[test]
    fn groups_by_keys_with_nulls_constants_and_runs() {
        // Rows of `k1`, `k2` and `x`, a batch each; in the second, `k1` is
        // constant; in the third, both keys are, and no `x` is null; in the
        // last two, `x` is. Row 3 of each is dropped.
        let batches: [&[[Option<i64>; 3]]; 5] = [
            &[
                [Some(1), Some(10), Some(1)],
                [Some(1), Some(10), Some(2)],
                [Some(1), None, Some(3)],
                [Some(2), Some(10), Some(4)],
                [Some(2), Some(10), None],
                [None, Some(5), Some(6)],
                [None, Some(5), Some(7)],
                [Some(1), Some(10), Some(8)],
            ],
            &[
                [Some(2), Some(10), Some(1)],
                [Some(2), Some(11), Some(2)],
                [Some(2), Some(10), Some(3)],
                [Some(2), Some(11), Some(4)],
                [Some(2), None, Some(5)],
                [Some(2), Some(10), None],
                [Some(2), Some(10), Some(7)],
            ],
            &[
                [None, Some(5), Some(100)],
                [None, Some(5), Some(200)],
                [None, Some(5), Some(300)],
                [None, Some(5), Some(400)],
            ],
            &[[Some(1), Some(10), Some(9)]; 5],
            &[[Some(2), Some(11), None]; 4],
        ];
        let constant = [
            [false, false, false],
            [true, false, false],
            [true, true, false],
            [false, false, true],
            [false, false, true],
        ];
        let mut context = Context::new(&Heap);
        let mut columns = Columns { batches: Vec::new(), dropped: alloc::vec![3] };
        for (rows, constant) in batches.iter().zip(constant) {
            let count = u32::try_from(rows.len()).unwrap();
            let batch = (0..3).map(|c| {
                let values: Vec<_> = rows.iter().map(|row| row[c]).collect();
                if constant.get(c) == Some(&true) {
                    let one = column(DataType::Int64, &values[..1]);
                    ColumnView::constant(&mut context, &one, count).unwrap()
                } else {
                    column(DataType::Int64, &values)
                }
            });
            columns.batches.push(batch.collect());
        }
        let expected = expected(&batches, 3);
        let functions = [Function::Count, Function::Sum, Function::Min, Function::Max];
        for apart in [true, false] {
            assert_eq!(merged(by(&[0, 1], 2, &functions), &columns, apart), expected);
        }
    }

    #[test]
    fn sums_widen_before_they_could_overflow() {
        // Summed in 64 bits, the second batch's first two values would
        // overflow, though the whole sum fits.
        let big = 1 << 62;
        let bounded = |values: &[Option<i64>], min, max| {
            column(DataType::Int64, values).with_bounds(Bounds { min, max })
        };
        let columns = Columns {
            batches: alloc::vec![
                alloc::vec![
                    column(DataType::Int64, &[Some(7); 2]),
                    bounded(&[Some(1), Some(2)], 1, 2)
                ],
                alloc::vec![
                    column(DataType::Int64, &[Some(7); 3]),
                    bounded(&[Some(big), Some(big), Some(-big)], -big, big),
                ],
            ],
            dropped: Vec::new(),
        };
        let functions = [Function::Sum];
        let expected = [Some(5), Some(3 + big)];
        let grouped: Vec<_> = [Some(7)].into_iter().chain(expected).collect();
        // Apart, the second batch's state widens, then the first's as it's
        // merged with it; together, the one state widens as it goes.
        for apart in [true, false] {
            assert_eq!(merged(by(&[], 1, &functions), &columns, apart), [expected.to_vec()]);
            let got = merged(by(&[0], 1, &functions), &columns, apart);
            assert_eq!(got, core::slice::from_ref(&grouped));
        }
    }
}
