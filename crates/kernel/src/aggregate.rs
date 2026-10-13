//! `AggregateOp`: aggregates, such as `COUNT(*)` and `SUM(x)`, over each
//! group of its child's rows with the same keys, or over all of them if it
//! has none; and `hash_aggregate`, the breaker that runs it.

use crate::allocator::{AllocError, Allocator};
use crate::buffer::Primitive;
use crate::column::{ColumnView, DataType, Form, Forms};
use crate::context::Context;
use crate::error::Error;
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::plan::{ColumnId, NamedColumn, Op, PlanNode};
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
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

/// The groups so far, and room to find a batch's.
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

/// Each group's total, by what the aggregate keeps.
enum Totals {
    /// None: `COUNT` keeps only counts.
    None,
    /// The least or greatest integer.
    Int(SlowVec<i64>),
    /// The sum of integers, in 128 bits, so no sum of a batch's can
    /// overflow.
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
}

/// The groups of a batch's kept rows.
#[derive(Clone, Copy)]
enum Ids<'a> {
    /// Every row's is this one.
    One(u32),
    /// Each row's, in order.
    Each(&'a [u32]),
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
        let keys = (0..width).map(|_| SlowVec::new(allocator, GROUPS_MAX));
        let mut keys_by_key = SlowVec::fixed(allocator, width.max(1))?;
        for key in keys {
            keys_by_key.push(key?).map_err(|_| AllocError)?;
        }
        Ok(Table {
            controls: SlowVec::zeroed(allocator, SLOTS_MIN + WIDTH)?,
            slots: SlowVec::zeroed(allocator, SLOTS_MIN)?,
            keys: keys_by_key,
            nulls: SlowVec::new(allocator, GROUPS_MAX)?,
        })
    }

    fn len(&self) -> usize {
        self.nulls.len()
    }

    /// Finds the group of each of `hashes`' rows, whose keys are `keys`, a
    /// list for each key, and `nulls`, if any are, into `ids`, adding groups
    /// for keys not seen before.
    fn assign(
        &mut self,
        allocator: &dyn Allocator,
        keys: &[&[i64]],
        nulls: Option<&[u64]>,
        hashes: &[u64],
        ids: &mut [u32],
    ) -> Result<(), Error> {
        self.reserve(allocator, hashes.len())?;
        for (row, (&hash, id)) in hashes.iter().zip(ids.iter_mut()).enumerate() {
            let null = nulls.map_or(0, |nulls| *at!(nulls, row));
            let found = self.find(hash, |group| {
                let group = group as usize;
                *at!(self.nulls, group) == null
                    && self
                        .keys
                        .iter()
                        .zip(keys)
                        .all(|(ours, keys)| at!(ours, group) == at!(keys, row))
            });
            *id = match found {
                Ok(group) => group,
                Err(slot) => {
                    let group = self.len();
                    for (ours, keys) in self.keys.iter_mut().zip(keys) {
                        ours.push(*at!(keys, row)).map_err(|_| AllocError)?;
                    }
                    self.nulls.push(null).map_err(|_| AllocError)?;
                    self.set(slot, hash, group);
                    #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
                    let group = group as u32;
                    group
                }
            };
        }
        Ok(())
    }

    /// The group hashed as `hash` that `same` says has the keys sought, or
    /// else the empty slot where it goes.
    #[inline]
    fn find(&self, hash: u64, same: impl Fn(u32) -> bool) -> Result<u32, usize> {
        let mask = self.slots.len() - 1;
        let control = LOW * u64::from(control(hash));
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
                let group = *at!(self.slots, slot);
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
    fn word(&self, at: usize) -> u64 {
        let bytes = at!(self.controls, at..at + WIDTH);
        let Ok(bytes) = <[u8; WIDTH]>::try_from(bytes) else { crate::check::check_failed(line!()) };
        u64::from_le_bytes(bytes)
    }

    /// Puts `group`, hashed as `hash`, in `slot`, which is empty.
    fn set(&mut self, slot: usize, hash: u64, group: usize) {
        let mask = self.slots.len() - 1;
        let control = control(hash);
        *at_mut!(self.controls, slot) = control;
        // The copy, for the first `WIDTH` slots; for the rest, the slot again.
        *at_mut!(self.controls, (slot.wrapping_sub(WIDTH) & mask) + WIDTH) = control;
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let group = group as u32;
        *at_mut!(self.slots, slot) = group;
    }

    /// Makes room for `more` groups, keeping slots at most 7/8 full.
    fn reserve(&mut self, allocator: &dyn Allocator, more: usize) -> Result<(), Error> {
        let len = self.len() + more;
        if len > GROUPS_MAX {
            return Err(Error::OutOfMemory);
        }
        if len * 8 <= self.slots.len() * 7 {
            return Ok(());
        }
        self.grow(allocator, (len * 8 / 7 + 1).next_power_of_two())
    }

    /// Moves every group into `slots` new slots.
    #[cold]
    #[inline(never)]
    fn grow(&mut self, allocator: &dyn Allocator, slots: usize) -> Result<(), Error> {
        self.controls = SlowVec::zeroed(allocator, slots + WIDTH)?;
        self.slots = SlowVec::zeroed(allocator, slots)?;
        for group in 0..self.len() {
            let keys = self.keys.iter().map(|keys| *at!(keys, group));
            let hash = keys.fold(SEED ^ *at!(self.nulls, group), mix);
            let Err(slot) = self.find(hash, |_| false) else { crate::check::check_failed(line!()) };
            self.set(slot, hash, group);
        }
        Ok(())
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

/// Calls `f` with each of the first `rows` rows `kept` keeps, and its place
/// among them.
#[inline]
fn for_kept(kept: Kept, rows: usize, mut f: impl FnMut(usize, usize)) {
    match kept {
        Kept::All => (0..rows).for_each(|row| f(row, row)),
        Kept::Select(rows) => {
            rows.iter().enumerate().for_each(|(i, &row)| f(i, usize::from(row)));
        }
        Kept::None => {}
    }
}

/// Folds each kept row's value, which `read` gives with whether it isn't
/// null, into its group's state with `op`.
#[inline]
fn fold<S: Copy, T>(
    states: &mut [S],
    ids: Ids,
    kept: Kept,
    rows: usize,
    read: impl Fn(usize) -> (T, bool),
    op: impl Fn(S, T, bool) -> S,
) {
    match ids {
        Ids::One(group) => {
            let state = at_mut!(states, group as usize);
            let mut folded = *state;
            for_kept(kept, rows, |_, row| {
                let (value, valid) = read(row);
                folded = op(folded, value, valid);
            });
            *state = folded;
        }
        Ids::Each(ids) => for_kept(kept, rows, |i, row| {
            let (value, valid) = read(row);
            let state = at_mut!(states, *at!(ids, i) as usize);
            *state = op(*state, value, valid);
        }),
    }
}

/// As `fold`, of `column`'s values, which are `T`s.
#[inline]
fn fold_column<S: Copy, T: Primitive>(
    states: &mut [S],
    ids: Ids,
    kept: Kept,
    rows: usize,
    column: &ColumnView,
    op: impl Fn(S, T, bool) -> S,
) {
    let (values, mask) = values_of(column);
    let words = values.slice_of::<T>();
    match values.validity() {
        None => fold(states, ids, kept, rows, |row| (*at!(words, row & mask), true), op),
        Some(validity) => fold(
            states,
            ids,
            kept,
            rows,
            |row| {
                let i = row & mask;
                // SAFETY: `i` is a row of the column, so one of its values.
                #[expect(clippy::cast_possible_truncation, reason = "a row")]
                (*at!(words, i), unsafe { validity.is_valid_unchecked(i as u32) })
            },
            op,
        ),
    }
}

/// `column`'s values, and which bits of a row are its value's place: all,
/// or none if it's constant.
fn values_of(column: &ColumnView) -> (crate::column::Values<'_>, usize) {
    match column.form() {
        Form::Flat => (column.values(), usize::MAX),
        Form::Constant => (column.values(), 0),
        Form::Dictionary(_) => crate::check::check_failed(line!()),
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
    let (all, rows) = (Kept::All, range.len());
    if ours.counts.is_some() || theirs.counts.is_some() {
        let counts = ours.counts(allocator, counted.0)?;
        let added = at!(theirs.counts.as_ref().unwrap_or(counted.1), range.clone());
        fold(counts, ids, all, rows, |i| (*at!(added, i), true), |count, x, _| count + x);
    }
    match (&mut ours.totals, &theirs.totals, function) {
        (Totals::Wide(ours), Totals::Wide(totals), _) => {
            let totals = at!(totals, range.clone());
            fold(ours, ids, all, rows, |i| (*at!(totals, i), true), |sum, x, _| sum + x);
        }
        (Totals::Int(ours), Totals::Int(totals), function) => {
            let read = |i| (*at!(totals, range.start + i), true);
            match function {
                Function::Min => fold(ours, ids, all, rows, read, |a, b: i64, _| a.min(b)),
                _ => fold(ours, ids, all, rows, read, |a, b: i64, _| a.max(b)),
            }
        }
        (Totals::Float(ours), Totals::Float(totals), function) => {
            let read = |i| (*at!(totals, range.start + i), true);
            match function {
                Function::Min => fold(ours, ids, all, rows, read, |a, b: f64, _| a.min(b)),
                Function::Max => fold(ours, ids, all, rows, read, |a, b: f64, _| a.max(b)),
                _ => fold(ours, ids, all, rows, read, |a, b, _| a + b),
            }
        }
        _ => {}
    }
    Ok(())
}

/// Adds `value` to the end of `values` until it holds `len`.
fn extend<T: Copy>(values: &mut SlowVec<T>, len: usize, value: T) -> Result<(), AllocError> {
    while values.len() < len {
        values.push(value).map_err(|_| AllocError)?;
    }
    Ok(())
}

impl Table {
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

impl HashAggregate {
    /// Finds the group of each of `input`'s `rows` kept rows, into
    /// `groups.scratch.ids`, adding groups for keys not seen before; or
    /// returns the one group of every row, if each key is constant.
    fn group(
        &self,
        allocator: &dyn Allocator,
        groups: &mut Groups,
        input: &RowBatch,
        rows: usize,
    ) -> Result<Option<u32>, Error> {
        let positions = self.grouping.keys.iter();
        let constant = positions.clone().all(|&k| input.column(k).is_in(Forms::CONSTANT));
        // Every row's keys are its first's: only one is read.
        let (kept, rows) = if constant { (Kept::All, 1) } else { (input.selection().kept(), rows) };
        let Scratch { keys: copies, nulls, hashes, ids } = &mut groups.scratch;
        let mut keys: [&[i64]; KEYS_MAX] = [&[]; KEYS_MAX];
        let mut any_null = false;
        let copies = copies.chunks_mut(BATCH_ROWS_MAX as usize);
        for ((k, &position), copy) in positions.enumerate().zip(copies) {
            let column = input.column(position);
            let (values, mask) = values_of(column);
            let words = values.int64s();
            let validity = values.validity();
            if mask != 0 && validity.is_none() && kept == Kept::All {
                *at_mut!(keys, k) = at!(words, ..rows);
                continue;
            }
            let copy = at_mut!(copy, ..rows);
            let Some(validity) = validity else {
                for_kept(kept, rows, |i, row| *at_mut!(copy, i) = *at!(words, row & mask));
                *at_mut!(keys, k) = copy;
                continue;
            };
            if !any_null {
                at_mut!(nulls, ..rows).fill(0);
                any_null = true;
            }
            for_kept(kept, rows, |i, row| {
                let value = row & mask;
                // SAFETY: `value` is a row of the column, so one of its values.
                #[expect(clippy::cast_possible_truncation, reason = "a row")]
                let valid = unsafe { validity.is_valid_unchecked(value as u32) };
                // Null keys are 0, so they compare and hash the same.
                *at_mut!(copy, i) = if valid { *at!(words, value) } else { 0 };
                *at_mut!(nulls, i) |= u64::from(!valid) << k;
            });
            *at_mut!(keys, k) = copy;
        }
        let width = self.grouping.keys.len();
        let keys = at!(keys, ..width);
        let nulls = any_null.then_some(at!(nulls, ..rows));
        let (hashes, ids) = (at_mut!(hashes, ..rows), at_mut!(ids, ..rows));
        hash_rows(keys, nulls, hashes);
        groups.table.assign(allocator, keys, nulls, hashes, ids)?;
        Ok(constant.then(|| *at!(ids, 0)))
    }

    /// Adds states for groups added since they were last added.
    fn add_states(&self, groups: &mut Groups) -> Result<(), AllocError> {
        let len = groups.table.len();
        if groups.rows.len() == len {
            return Ok(());
        }
        extend(&mut groups.rows, len, 0)?;
        for (aggregate, states) in self.grouping.aggregates.iter().zip(groups.states.iter_mut()) {
            if let Some(counts) = &mut states.counts {
                extend(counts, len, 0)?;
            }
            match &mut states.totals {
                Totals::None => {}
                Totals::Int(totals) => {
                    let first =
                        if aggregate.function == Function::Min { i64::MAX } else { i64::MIN };
                    extend(totals, len, first)?;
                }
                Totals::Wide(totals) => extend(totals, len, 0)?,
                Totals::Float(totals) => {
                    let sum = matches!(aggregate.function, Function::Sum | Function::Avg);
                    extend(totals, len, if sum { 0.0 } else { f64::NAN })?;
                }
            }
        }
        Ok(())
    }

    /// Folds `input`'s column for `aggregate` into `states`.
    fn update(
        allocator: &dyn Allocator,
        aggregate: Lowered,
        states: &mut States,
        counted: &SlowVec<u64>,
        input: &RowBatch,
        ids: Ids,
        rows: usize,
    ) -> Result<(), AllocError> {
        let Some((position, _)) = aggregate.input else { return Ok(()) };
        let kept = input.selection().kept();
        let column = input.column(position);
        if column.has_nulls() {
            let counts = states.counts(allocator, counted)?;
            fold_column(counts, ids, kept, rows, column, |count, _: u64, valid| {
                count + u64::from(valid)
            });
        }
        match (&mut states.totals, aggregate.function) {
            (Totals::None, _) => {}
            (Totals::Wide(sums), _) => {
                fold_column(sums, ids, kept, rows, column, |sum, x: i64, valid| {
                    sum + if valid { i128::from(x) } else { 0 }
                });
            }
            (Totals::Int(least), Function::Min) => {
                fold_column(least, ids, kept, rows, column, |least: i64, x, valid| {
                    if valid { least.min(x) } else { least }
                });
            }
            (Totals::Int(greatest), _) => {
                fold_column(greatest, ids, kept, rows, column, |greatest: i64, x, valid| {
                    if valid { greatest.max(x) } else { greatest }
                });
            }
            (Totals::Float(least), Function::Min) => {
                fold_column(least, ids, kept, rows, column, |least: f64, x, valid| {
                    least.min(if valid { x } else { f64::NAN })
                });
            }
            (Totals::Float(greatest), Function::Max) => {
                fold_column(greatest, ids, kept, rows, column, |greatest: f64, x, valid| {
                    greatest.max(if valid { x } else { f64::NAN })
                });
            }
            (Totals::Float(sums), _) => {
                fold_column(sums, ids, kept, rows, column, |sum: f64, x, valid| {
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
                (_, Totals::Int(totals)) => Some(*at!(totals, group)),
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
                    Totals::Wide(SlowVec::new(allocator, GROUPS_MAX)?)
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
            },
            served: 0,
        };
        // All rows are one group, which is there even if there are none.
        if width == 0 {
            groups.table.nulls.push(0).map_err(|_| AllocError)?;
            self.add_states(&mut groups)?;
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
        let kept = input.selection().kept();
        let rows = match kept {
            Kept::All => input.row_count() as usize,
            Kept::Select(rows) => rows.len(),
            Kept::None => return Ok(()),
        };
        let one = if self.grouping.keys.is_empty() {
            Some(0)
        } else {
            self.group(allocator, groups, input, rows)?
        };
        self.add_states(groups)?;
        let ids = match one {
            Some(group) => Ids::One(group),
            None => Ids::Each(at!(groups.scratch.ids, ..rows)),
        };
        for (&aggregate, states) in self.grouping.aggregates.iter().zip(groups.states.iter_mut()) {
            HashAggregate::update(allocator, aggregate, states, &groups.rows, input, ids, rows)?;
        }
        fold(&mut groups.rows, ids, kept, rows, |_| ((), true), |count, (), _| count + 1);
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
        for first in (0..other.table.len()).step_by(batch) {
            let rows = (other.table.len() - first).min(batch);
            let theirs = first..first + rows;
            let ids = if width == 0 {
                Ids::One(0)
            } else {
                let mut keys: [&[i64]; KEYS_MAX] = [&[]; KEYS_MAX];
                for (keys, theirs_by_key) in keys.iter_mut().zip(other.table.keys.iter()) {
                    *keys = at!(theirs_by_key, theirs.clone());
                }
                let keys = at!(keys, ..width);
                let nulls = Some(at!(other.table.nulls, theirs.clone()));
                let Scratch { hashes, ids, .. } = &mut groups.scratch;
                let (hashes, ids) = (at_mut!(hashes, ..rows), at_mut!(ids, ..rows));
                hash_rows(keys, nulls, hashes);
                groups.table.assign(allocator, keys, nulls, hashes, ids)?;
                self.add_states(groups)?;
                Ids::Each(at!(groups.scratch.ids, ..rows))
            };
            let aggregates = self.grouping.aggregates.iter().map(|aggregate| aggregate.function);
            for ((function, ours), theirs_states) in
                aggregates.zip(groups.states.iter_mut()).zip(other.states.iter())
            {
                let counted = (&groups.rows, &other.rows);
                merge_states(allocator, function, ours, theirs_states, counted, ids, &theirs)?;
            }
            let counted = at!(other.rows, theirs);
            fold(
                &mut groups.rows,
                ids,
                Kept::All,
                rows,
                |i| (*at!(counted, i), true),
                |a, b, _| a + b,
            );
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
        let rows = (groups.table.len() - first).min(BATCH_ROWS_MAX as usize);
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
}
