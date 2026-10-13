//! `AggregateOp`: aggregates, such as `COUNT(*)` and `SUM(x)`, over each
//! group of its child's rows with the same keys, or over all of them if it
//! has none; and `hash_aggregate`, the breaker that runs it.

use crate::allocator::{AllocError, Allocator};
use crate::column::{ColumnView, DataType, Forms};
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
    check!(grouping.keys.len() <= KEYS_MAX);
    DynOperator::new(allocator, RadixAggregate { grouping })
}

/// The most keys a grouping can have: one bit each says which are null.
const KEYS_MAX: usize = 64;

/// The most groups an aggregation can hold.
const GROUPS_MAX: usize = 1 << 26;

/// How many bits of a hash pick a group's partition, once there are many.
const RADIX_BITS: u32 = 4;

/// How many partitions there are once there are many.
const PARTITIONS: usize = 1 << RADIX_BITS;

/// How many groups there are before they're split into partitions: as many
/// as one table holds before it outgrows cache. Splitting sooner cost more,
/// sorting each batch's runs, than it saved.
const SPLIT_GROUPS: usize = 1 << 17;

/// How many rows, or groups, are found at a time: a batch's.
const ROWS: usize = BATCH_ROWS_MAX as usize;

/// Groups rows a batch at a time, into partitions by the top bits of their
/// keys' hash, each with its own hash table of open addressing. There's one
/// partition until there are many groups, then many, each small enough to
/// stay in cache. Partitions are independent: states merge, and could
/// spill, one partition at a time.
///
/// Rows come in runs of the same keys, as data is often sorted, so a batch
/// is cut into runs first: each run's group is found once, the runs sorted
/// by partition to find them, and a run's values added up together.
struct RadixAggregate {
    grouping: Grouping,
}

/// The groups so far.
struct Groups {
    /// One, or `PARTITIONS` once split: a group is in the one the top bits
    /// of its hash pick.
    partitions: SlowVec<Partition>,
    /// The partition being output, and how many of its groups have been.
    served: (usize, usize),
    /// Room for a batch, `ROWS` words apart: each run's tag, saying which
    /// rows it has, and hash; the same, sorted by partition; where each run,
    /// so sorted, has its group's totals; and the rows kept.
    scratch: SlowVec<u64>,
    /// Each run's entry, as a group's; then the same, sorted.
    entries: SlowVec<i64>,
}

/// Some of the groups, in a hash table.
struct Partition {
    /// A power of two many, at most half full: each the index of a group
    /// plus one, or 0 if empty.
    slots: SlowVec<u32>,
    /// Each group's entry, one after another: its keys, as 0 if null, then
    /// a word whose bits say which are null.
    entries: SlowVec<i64>,
    /// Each group's totals, one for each aggregate.
    totals: SlowVec<Total>,
    /// How many groups it has.
    len: usize,
}

/// An aggregate's total for a group: how many values it's seen, and their
/// sum, least or greatest so far, as an integer or a float by the column's
/// type: an integer in 128 bits, so no sum of a batch's can overflow, as
/// its `low` and `high` words; a float as `low`'s bits.
#[derive(Clone, Copy, Default)]
struct Total {
    count: u64,
    low: u64,
    high: u64,
}

/// Totals of no values, to start groups' with.
const NONE: [Total; 4] = [Total { count: 0, low: 0, high: 0 }; 4];

impl Total {
    #[expect(clippy::cast_possible_truncation, reason = "split into words")]
    fn of_int(count: u64, int: i128) -> Total {
        let bits = int.cast_unsigned();
        Total { count, low: bits as u64, high: (bits >> 64) as u64 }
    }

    fn of_float(count: u64, float: f64) -> Total {
        Total { count, low: float.to_bits(), high: 0 }
    }

    fn int(self) -> i128 {
        i128::from(self.high.cast_signed()) << 64 | i128::from(self.low)
    }

    fn float(self) -> f64 {
        f64::from_bits(self.low)
    }
}

impl Partition {
    /// An empty partition, for entries `width` words wide and `count`
    /// totals.
    fn new(allocator: &dyn Allocator, width: usize, count: usize) -> Result<Partition, Error> {
        Ok(Partition {
            slots: SlowVec::zeroed(allocator, 16)?,
            entries: SlowVec::new(allocator, (GROUPS_MAX * width).next_power_of_two())?,
            totals: SlowVec::new(allocator, (GROUPS_MAX * count.max(1)).next_power_of_two())?,
            len: 0,
        })
    }

    /// Adds a group with `entry`, and `count` totals, at `slot`.
    #[inline(never)]
    fn add(
        &mut self,
        allocator: &dyn Allocator,
        count: usize,
        entry: &[i64],
        slot: usize,
    ) -> Result<(), Error> {
        self.entries.extend_from_slice(entry)?;
        for start in (0..count).step_by(NONE.len()) {
            self.totals.extend_from_slice(at!(NONE, ..(count - start).min(NONE.len())))?;
        }
        self.len += 1;
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let index = self.len as u32;
        *at_mut!(self.slots, slot) = index;
        if self.len * 2 <= self.slots.len() {
            return Ok(());
        }
        // Doubles the slots, putting each group back.
        let mut slots = SlowVec::zeroed(allocator, self.slots.len() * 2)?;
        let mask = slots.len() - 1;
        for (index, entry) in (1..).zip(self.entries.chunks_exact(entry.len())) {
            #[expect(clippy::cast_possible_truncation, reason = "masked")]
            let mut slot = hash(entry) as usize & mask;
            while *at!(slots, slot) != 0 {
                slot = (slot + 1) & mask;
            }
            *at_mut!(slots, slot) = index;
        }
        self.slots = slots;
        Ok(())
    }
}

/// Which of `partitions` partitions, one or `PARTITIONS`, a group hashed as
/// `hash` is in.
fn partition_of(hash: u64, partitions: usize) -> usize {
    let top = (hash >> (64 - RADIX_BITS)) as usize;
    top & (partitions - 1)
}

impl RadixAggregate {
    /// `count` empty partitions.
    fn partitions(
        &self,
        allocator: &dyn Allocator,
        count: usize,
    ) -> Result<SlowVec<Partition>, Error> {
        let (width, totals) = (self.grouping.keys.len() + 1, self.grouping.aggregates.len());
        let mut partitions = SlowVec::fixed(allocator, count)?;
        for _ in 0..count {
            let partition = Partition::new(allocator, width, totals)?;
            partitions.push(partition).map_err(|_| AllocError)?;
        }
        Ok(partitions)
    }

    /// Finds the group of each of `entries`, hashed as `hashes` say, in
    /// `partition`, adding those not there: where its totals start, into
    /// `starts`.
    fn find(
        &self,
        allocator: &dyn Allocator,
        partition: &mut Partition,
        (entries, hashes): (&[i64], &[u64]),
        starts: &mut [u64],
    ) -> Result<(), Error> {
        let width = self.grouping.keys.len() + 1;
        let count = self.grouping.aggregates.len();
        for ((entry, &hash), start) in entries.chunks_exact(width).zip(hashes).zip(starts) {
            let mask = partition.slots.len() - 1;
            #[expect(clippy::cast_possible_truncation, reason = "masked")]
            let mut slot = hash as usize & mask;
            let group = loop {
                let group = *at!(partition.slots, slot) as usize;
                if group == 0 {
                    partition.add(allocator, count, entry, slot)?;
                    break partition.len - 1;
                }
                let stored = at!(partition.entries, (group - 1) * width..);
                if stored.iter().zip(entry).all(|(stored, word)| stored == word) {
                    break group - 1;
                }
                slot = (slot + 1) & mask;
            };
            *start = (group * count) as u64;
        }
        Ok(())
    }

    /// Finds the groups of a batch's `len` runs: their tags and hashes are
    /// `scratch`'s first, and their entries `entries`'. If there are many
    /// partitions, sorts the runs by partition first, into the room after
    /// each. Returns where each partition's runs end, and whether they're
    /// sorted.
    fn find_all(
        &self,
        allocator: &dyn Allocator,
        groups: &mut Groups,
        len: usize,
    ) -> Result<([usize; PARTITIONS], bool), Error> {
        let width = self.grouping.keys.len() + 1;
        let (scratch, starts) = at_mut!(groups.scratch, ..).split_at_mut(4 * ROWS);
        let (entries, sorted_entries) = at_mut!(groups.entries, ..).split_at_mut(ROWS * width);
        let (tags, scratch) = scratch.split_at_mut(ROWS);
        let (hashes, sorted) = scratch.split_at_mut(ROWS);
        let partitions = &mut groups.partitions;
        let mut ends = [len; PARTITIONS];
        let many = partitions.len() > 1;
        if many {
            let mut next = [0_usize; PARTITIONS];
            for &hash in at!(hashes, ..len) {
                *at_mut!(next, partition_of(hash, PARTITIONS)) += 1;
            }
            let mut sum = 0;
            for (next, end) in next.iter_mut().zip(&mut ends) {
                (*next, sum) = (sum, sum + *next);
                *end = sum;
            }
            let entries = entries.chunks_exact(width).zip(at!(hashes, ..len)).zip(&*tags);
            for ((entry, &hash), &tag) in entries {
                let next = at_mut!(next, partition_of(hash, PARTITIONS));
                let j = *next;
                *next += 1;
                (*at_mut!(sorted, j), *at_mut!(sorted, ROWS + j)) = (tag, hash);
                let to = at_mut!(sorted_entries, j * width..(j + 1) * width);
                for (to, &word) in to.iter_mut().zip(entry) {
                    *to = word;
                }
            }
        }
        let (entries, hashes) =
            if many { (&*sorted_entries, at!(sorted, ROWS..)) } else { (&*entries, &*hashes) };
        let mut first = 0;
        for (partition, &end) in partitions.iter_mut().zip(&ends) {
            let entries = at!(entries, first * width..end * width);
            let starts = at_mut!(starts, first..end);
            self.find(allocator, partition, (entries, at!(hashes, first..end)), starts)?;
            first = end;
        }
        Ok((ends, many))
    }

    /// Adds each of `other`'s groups to `groups`, in the partition its
    /// hash picks, folding its totals into those of the same keys.
    fn absorb(
        &self,
        allocator: &dyn Allocator,
        groups: &mut Groups,
        other: &Partition,
    ) -> Result<(), Error> {
        let width = self.grouping.keys.len() + 1;
        let count = self.grouping.aggregates.len();
        for found in 0..other.len {
            let entry = at!(other.entries, found * width..(found + 1) * width);
            let theirs = at!(other.totals, found * count..(found + 1) * count);
            let hash = hash(entry);
            let partitions = groups.partitions.len();
            let partition = at_mut!(groups.partitions, partition_of(hash, partitions));
            let mut start = [0];
            self.find(allocator, partition, (entry, &[hash]), &mut start)?;
            let ours = at_mut!(partition.totals, index(start[0])..);
            for ((aggregate, ours), &theirs) in
                self.grouping.aggregates.iter().zip(ours).zip(theirs)
            {
                let float =
                    aggregate.input.is_some_and(|(_, data_type)| data_type == DataType::Float64);
                fold(aggregate.function, float, ours, theirs);
            }
        }
        Ok(())
    }

    /// Splits the one partition into many once it holds `SPLIT_GROUPS`, and
    /// fails if there are too many groups.
    fn split(&self, allocator: &dyn Allocator, groups: &mut Groups) -> Result<(), Error> {
        let len: usize = groups.partitions.iter().map(|partition| partition.len).sum();
        if len > GROUPS_MAX {
            return Err(Error::OutOfMemory);
        }
        if groups.partitions.len() == 1 && len > SPLIT_GROUPS {
            let many = self.partitions(allocator, PARTITIONS)?;
            let one = core::mem::replace(&mut groups.partitions, many);
            self.absorb(allocator, groups, at!(one, 0))?;
        }
        Ok(())
    }

    /// Finds the runs of rows with the same keys among `rows` of `input`:
    /// rows come in such runs, as data is often sorted, and a run's group
    /// is found once. Each run's first row, and in the high half how many
    /// it has, go in `tags`; `same` is room to mark rows with the last's
    /// keys. Returns how many runs there are.
    fn runs(&self, input: &Flat, rows: &[u64], same: &mut [u64], tags: &mut [u64]) -> usize {
        let same = at_mut!(same, ..rows.len());
        same.fill(1);
        *at_mut!(same, 0) = 0;
        for &position in self.grouping.keys.iter() {
            let column = input.column(position);
            let values = column.int64s();
            let rest = at!(rows, 1..).iter().zip(at_mut!(same, 1..));
            let Some(validity) = column.validity() else {
                let mut last = *at!(values, index(*at!(rows, 0)));
                for (&row, same) in rest {
                    // SAFETY: as `at_row` asks.
                    let value = unsafe { at_row(values, row) };
                    *same &= u64::from(value == last);
                    last = value;
                }
                continue;
            };
            // A null is 0, and not valid.
            #[expect(clippy::cast_possible_truncation, reason = "a batch's row")]
            let key = |row: u64| {
                // SAFETY: kept rows are below the batch's row count.
                let valid = unsafe { validity.is_valid_unchecked(row as u32) };
                (valid, if valid { *at!(values, row as usize) } else { 0 })
            };
            let mut last = key(*at!(rows, 0));
            for (&row, same) in rest {
                let key = key(row);
                *same &= u64::from(key == last);
                last = key;
            }
        }
        let mut runs = 0;
        for (i, &same) in same.iter().enumerate() {
            if same == 0 {
                *at_mut!(tags, runs) = i as u64;
                runs += 1;
            }
        }
        // Each run ends where the next starts.
        let mut end = rows.len() as u64;
        for tag in at_mut!(tags, ..runs).iter_mut().rev() {
            (*tag, end) = (*tag | (end - *tag) << 32, *tag);
        }
        runs
    }

    /// Writes the entries of the runs `tags` says of rows `rows` of `input`
    /// into `entries`, and their hashes into `hashes`.
    fn entries(
        &self,
        input: &Flat,
        rows: &[u64],
        tags: &[u64],
        entries: &mut [i64],
        hashes: &mut [u64],
    ) {
        let width = self.grouping.keys.len() + 1;
        let entries = at_mut!(entries, ..tags.len() * width);
        for (k, &position) in self.grouping.keys.iter().enumerate() {
            let column = input.column(position);
            let (values, validity) = (column.int64s(), column.validity());
            for (entry, &tag) in entries.chunks_exact_mut(width).zip(tags) {
                #[expect(clippy::cast_possible_truncation, reason = "a batch's row")]
                let row = *at!(rows, tag as u32 as usize) as u32;
                // SAFETY: kept rows are below the batch's row count.
                let valid =
                    validity.is_none_or(|validity| unsafe { validity.is_valid_unchecked(row) });
                *at_mut!(entry, k) = if valid { *at!(values, row as usize) } else { 0 };
                let nulls = at_mut!(entry, width - 1);
                *nulls = if k == 0 { 0 } else { *nulls } | i64::from(!valid) << k;
            }
        }
        for (hash, entry) in hashes.iter_mut().zip(entries.chunks_exact(width)) {
            *hash = self::hash(entry);
        }
    }

    /// Adds the runs `tags` says of rows `rows` of `input` to the totals of
    /// their groups, which start at `starts` in `totals`.
    #[inline(never)]
    fn add(
        &self,
        input: &Flat,
        totals: &mut [Total],
        rows: &[u64],
        (tags, starts): (&[u64], &[u64]),
    ) {
        for (a, aggregate) in self.grouping.aggregates.iter().enumerate() {
            let function = aggregate.function;
            let column = aggregate.input.map(|(position, _)| input.column(position));
            let data_type = aggregate.input.map(|(_, data_type)| data_type);
            let validity = column.and_then(ColumnView::validity);
            // A count needs no values.
            let values = column.filter(|_| function != Function::Count);
            let ints = match values {
                Some(column) if data_type == Some(DataType::Int64) => column.int64s(),
                _ => &[],
            };
            let floats = match values {
                Some(column) if data_type == Some(DataType::Float64) => column.float64s(),
                _ => &[],
            };
            let float = !floats.is_empty();
            // Counts, and sums of integers, that can't be null add up a run
            // at a time.
            let runs = validity.is_none()
                && !float
                && matches!(function, Function::Count | Function::Sum | Function::Avg);
            for (&tag, &start) in tags.iter().zip(starts) {
                #[expect(clippy::cast_possible_truncation, reason = "a batch's rows")]
                let (first, len) = (tag as u32 as usize, (tag >> 32) as usize);
                let total = at_mut!(totals, index(start) + a);
                let rows = at!(rows, first..first + len);
                if runs {
                    let mut sum = total.int();
                    if !ints.is_empty() {
                        for &row in rows {
                            // SAFETY: as `at_row` asks.
                            sum += i128::from(unsafe { at_row(ints, row) });
                        }
                    }
                    *total = Total::of_int(total.count + len as u64, sum);
                    continue;
                }
                for &row in rows {
                    #[expect(clippy::cast_possible_truncation, reason = "a batch's row")]
                    let row = row as u32;
                    // SAFETY: kept rows are below the batch's row count.
                    if validity.is_some_and(|validity| !unsafe { validity.is_valid_unchecked(row) })
                    {
                        continue;
                    }
                    let value = match floats.get(row as usize) {
                        Some(&float) => Total::of_float(1, float),
                        None => {
                            Total::of_int(1, ints.get(row as usize).copied().unwrap_or(0).into())
                        }
                    };
                    fold(function, float, total, value);
                }
            }
        }
    }

    /// Column `c` of the output, the keys then the aggregates, for groups
    /// `first..first + rows` of `partition`.
    fn output(
        &self,
        context: &mut Context,
        partition: &Partition,
        c: usize,
        first: usize,
        rows: usize,
    ) -> Result<ColumnView, Error> {
        let width = self.grouping.keys.len();
        let count = self.grouping.aggregates.len();
        let aggregate = self.grouping.aggregates.get(c.wrapping_sub(width)).copied();
        let data_type =
            aggregate.and_then(|aggregate| aggregate.input).map(|(_, data_type)| data_type);
        let output_type = match aggregate {
            None => Some(DataType::Int64),
            Some(aggregate) => aggregate.function.output_type(data_type),
        };
        let Some(output_type) = output_type else { crate::check::check_failed(line!()) };
        let mut values = context.values_buffer(rows * 8)?;
        let mut validity = context.small_buffer(rows.div_ceil(8))?;
        validity.as_mut_slice::<u8>().fill(0);
        let mut any_null = false;
        for (i, value) in values.as_mut_slice::<i64>().iter_mut().enumerate() {
            let group = first + i;
            let word = match aggregate {
                None => {
                    let entry = at!(partition.entries, group * (width + 1)..);
                    let null = *at!(entry, width) >> c & 1 != 0;
                    (!null).then_some(*at!(entry, c))
                }
                Some(aggregate) => {
                    let total = *at!(partition.totals, group * count + c - width);
                    result(aggregate.function, data_type, total)?
                }
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

/// The tags of the runs `find_all` found, `sorted` or not, and where their
/// groups' totals start, from `scratch`.
fn found(scratch: &[u64], sorted: bool) -> (&[u64], &[u64]) {
    let tags = if sorted { at!(scratch, 2 * ROWS..3 * ROWS) } else { at!(scratch, ..ROWS) };
    (tags, at!(scratch, 4 * ROWS..5 * ROWS))
}

/// Folds `theirs` into `ours`, both totals of `function`, of floats if
/// `float`.
fn fold(function: Function, float: bool, ours: &mut Total, theirs: Total) {
    let count = ours.count + theirs.count;
    *ours = match function {
        _ if theirs.count == 0 => *ours,
        _ if ours.count == 0 => theirs,
        Function::Min if float => Total::of_float(count, ours.float().min(theirs.float())),
        Function::Max if float => Total::of_float(count, ours.float().max(theirs.float())),
        _ if float => Total::of_float(count, ours.float() + theirs.float()),
        Function::Min => Total::of_int(count, ours.int().min(theirs.int())),
        Function::Max => Total::of_int(count, ours.int().max(theirs.int())),
        _ => Total::of_int(count, ours.int() + theirs.int()),
    };
}

/// What `function` makes of `total`, of a column of `data_type`, as a
/// word: an integer, or a float's bits.
fn result(
    function: Function,
    data_type: Option<DataType>,
    total: Total,
) -> Result<Option<i64>, Error> {
    Ok(match (function, data_type) {
        (Function::Count, _) => Some(i64::try_from(total.count).map_err(|_| Error::Unsupported)?),
        _ if total.count == 0 => None,
        #[expect(clippy::cast_precision_loss, reason = "averages are approximate")]
        (Function::Avg, Some(DataType::Int64)) => {
            Some((total.int() as f64 / total.count as f64).to_bits().cast_signed())
        }
        #[expect(clippy::cast_precision_loss, reason = "averages are approximate")]
        (Function::Avg, _) => Some((total.float() / total.count as f64).to_bits().cast_signed()),
        // A sum that doesn't fit in an `Int64` would need a wider type.
        (_, Some(DataType::Int64)) => {
            Some(i64::try_from(total.int()).map_err(|_| Error::Unsupported)?)
        }
        _ => Some(total.float().to_bits().cast_signed()),
    })
}

/// A batch whose columns this reads are flat: they're in whatever form the
/// first op to read them takes, so those that aren't are made flat.
struct Flat<'b, 'r> {
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
}

/// A word of scratch as an index: a row, or where a group's totals start,
/// which fit.
#[expect(clippy::cast_possible_truncation, reason = "they fit")]
fn index(word: u64) -> usize {
    word as usize
}

/// Row `row` of `values`, without checking it's there.
///
/// # Safety
///
/// `values` must be a flat column's of a batch, and `row` one it keeps:
/// kept rows are below the batch's row count, every column's length.
#[expect(clippy::cast_possible_truncation, reason = "a batch's row")]
unsafe fn at_row(values: &[i64], row: u64) -> i64 {
    // SAFETY: upheld by the caller.
    unsafe { *values.get_unchecked(row as usize) }
}

/// `word` mixed into `hash`.
fn mix(hash: u64, word: u64) -> u64 {
    (hash ^ word).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29)
}

/// A group's entry, hashed.
fn hash(entry: &[i64]) -> u64 {
    entry.iter().fold(0, |hash, &word| mix(hash, word.cast_unsigned()))
}

impl Breaker for RadixAggregate {
    type State = Groups;

    fn new_state(&self, context: &mut Context) -> Result<Groups, Error> {
        let allocator = context.allocator();
        let width = self.grouping.keys.len() + 1;
        let mut groups = Groups {
            partitions: self.partitions(allocator, 1)?,
            served: (0, 0),
            scratch: SlowVec::zeroed(allocator, 6 * ROWS)?,
            entries: SlowVec::zeroed(allocator, 2 * ROWS * width)?,
        };
        // All rows are one group, which is there even if there are none.
        if width == 1 {
            self.find(allocator, at_mut!(groups.partitions, 0), (&[0], &[0]), &mut [0])?;
        }
        Ok(groups)
    }

    fn consume(
        &self,
        context: &mut Context,
        groups: &mut Groups,
        batch: &RowBatch,
    ) -> Result<(), Error> {
        let allocator = context.allocator();
        let inputs = self.grouping.aggregates.iter().filter_map(|aggregate| aggregate.input);
        let read = self.grouping.keys.iter().copied().chain(inputs.map(|(position, _)| position));
        let input = Flat::new(context, batch, read)?;
        let len = match batch.selection().kept() {
            Kept::All => batch.row_count() as usize,
            Kept::Select(kept) => kept.len(),
            Kept::None => return Ok(()),
        };
        let (scratch, rows) = at_mut!(groups.scratch, ..).split_at_mut(5 * ROWS);
        let rows = at_mut!(rows, ..len);
        match batch.selection().kept() {
            Kept::Select(kept) => {
                for (row, &kept) in rows.iter_mut().zip(kept) {
                    *row = u64::from(kept);
                }
            }
            _ => {
                for (i, row) in rows.iter_mut().enumerate() {
                    *row = i as u64;
                }
            }
        }
        let (tags, scratch) = scratch.split_at_mut(ROWS);
        let (ends, sorted) = if self.grouping.keys.is_empty() {
            // All rows are one run, of the one group, whose totals start at
            // 0, as the starts were made: nothing else writes them.
            *at_mut!(tags, 0) = (len as u64) << 32;
            ([1; PARTITIONS], false)
        } else {
            // The hashes' room marks rows in runs first.
            let hashes = at_mut!(scratch, ..ROWS);
            let runs = self.runs(&input, rows, hashes, tags);
            let tags = at!(tags, ..runs);
            self.entries(&input, rows, tags, &mut groups.entries, hashes);
            self.find_all(allocator, groups, runs)?
        };
        let (tags, starts) = found(&groups.scratch, sorted);
        let rows = at!(groups.scratch, 5 * ROWS..);
        let mut first = 0;
        for (partition, &end) in groups.partitions.iter_mut().zip(&ends) {
            let runs = (at!(tags, first..end), at!(starts, first..end));
            self.add(&input, &mut partition.totals, rows, runs);
            first = end;
        }
        self.split(allocator, groups)
    }

    fn merge(
        &self,
        context: &mut Context,
        groups: &mut Groups,
        mut other: Groups,
    ) -> Result<(), Error> {
        // Into the one split, if either is, which keeps its partitions.
        if other.partitions.len() > groups.partitions.len() {
            core::mem::swap(groups, &mut other);
        }
        let allocator = context.allocator();
        for partition in other.partitions.iter() {
            self.absorb(allocator, groups, partition)?;
        }
        self.split(allocator, groups)
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
        let (mut p, mut first) = groups.served;
        while p + 1 < groups.partitions.len() && first == at!(groups.partitions, p).len {
            (p, first) = (p + 1, 0);
        }
        let partition = at!(groups.partitions, p);
        let rows = (partition.len - first).min(ROWS);
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
        output.reset(rows as u32);
        for c in 0..self.grouping.keys.len() + self.grouping.aggregates.len() {
            let column = self.output(context, partition, c, first, rows)?;
            let Ok(()) = output.push_column(column) else { crate::check::check_failed(line!()) };
        }
        groups.served = (p, first + rows);
        let rest = at!(groups.partitions, p + 1..);
        Ok(first + rows < partition.len || rest.iter().any(|partition| partition.len > 0))
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
    fn aggregates_all_rows_kept_without_keys() {
        // Row 1 of each batch is dropped.
        let rows = Rows {
            batches: alloc::vec![
                alloc::vec![(None, Some(-4)), (None, Some(1000)), (None, None)],
                alloc::vec![(None, Some(7)), (None, Some(1000)), (None, Some(3))],
            ],
            dropped: alloc::vec![1],
        };
        let expected = [Some(4), Some(3), Some(6), Some(-4), Some(7), Some(bits(2.0))];
        assert_eq!(aggregated(rows, false), [expected]);
    }

    /// Each key and its `COUNT(*)`, from merging a state that consumed
    /// keys `first` into one that consumed `second`, a batch at a time.
    fn merged(first: core::ops::Range<i64>, second: core::ops::Range<i64>) -> Vec<(i64, i64)> {
        let breaker = RadixAggregate { grouping: grouping(true) };
        let mut context = Context::new(&Heap);
        let consumed = |context: &mut Context, range: core::ops::Range<i64>| {
            let mut state = Breaker::new_state(&breaker, context).unwrap();
            let keys: Vec<_> = range.map(|k| (Some(k), Some(1))).collect();
            let batches = keys.chunks(BATCH_ROWS_MAX as usize).map(<[_]>::to_vec).collect();
            let rows = Rows { batches, dropped: Vec::new() };
            let mut source = Source::new_state(&rows, context).unwrap();
            let mut input = RowBatch::new();
            while rows.next(context, &mut source, &mut input).unwrap() {
                breaker.consume(context, &mut state, &input).unwrap();
            }
            state
        };
        let (mut state, other) = (consumed(&mut context, second), consumed(&mut context, first));
        breaker.merge(&mut context, &mut state, other).unwrap();
        let mut output = RowBatch::new();
        let mut counts = Vec::new();
        loop {
            let more = breaker.serve(&mut context, &mut state, &mut output).unwrap();
            for row in 0..output.row_count() as usize {
                counts.push((output.column(0).int64s()[row], output.column(1).int64s()[row]));
            }
            if !more {
                break;
            }
        }
        counts.sort_unstable();
        counts
    }

    #[test]
    fn merged_states_hold_every_group() {
        // Enough groups to grow the table several times, half of them in
        // each state, and some in both.
        let expected: Vec<_> =
            (0..2000).map(|k| (k, if (1000..1500).contains(&k) { 2 } else { 1 })).collect();
        assert_eq!(merged(0..1500, 1000..2000), expected);
    }

    #[test]
    fn splits_into_partitions_and_merges_them() {
        // Enough groups to split one state into partitions but not the
        // other, which it's merged into.
        let groups = i64::try_from(SPLIT_GROUPS * 2).unwrap();
        let expected: Vec<_> =
            (0..groups).map(|k| (k, if (100..200).contains(&k) { 2 } else { 1 })).collect();
        assert_eq!(merged(0..groups, 100..200), expected);
    }
}
