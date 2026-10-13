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
    DynOperator::new(allocator, HashAggregate { grouping })
}

/// The most keys a grouping can have: one bit each says which are null.
const KEYS_MAX: usize = 64;

/// The most groups an aggregation can hold.
const GROUPS_MAX: usize = 1 << 26;

/// Groups rows a row at a time, in a hash table of open addressing.
struct HashAggregate {
    grouping: Grouping,
}

/// The groups so far.
struct Groups {
    /// A power of two many, at most half full: each the index of a group
    /// plus one, or 0 if empty.
    slots: SlowVec<u32>,
    /// Each group's keys, one after another.
    keys: SlowVec<i64>,
    /// Which of each group's keys are null, a bit each.
    nulls: SlowVec<u64>,
    /// Each group's totals, one for each aggregate.
    totals: SlowVec<Total>,
    /// How many groups have been output.
    served: usize,
}

/// An aggregate's total for a group: how many values it's seen, and their
/// sum, least or greatest so far, as an integer or a float by the column's
/// type. Integers are summed in 128 bits, so no sum of a batch's can
/// overflow.
#[derive(Clone, Copy, Default)]
struct Total {
    count: u64,
    int: i128,
    float: f64,
}

impl HashAggregate {
    /// The group with `keys` and `nulls`, hashed as `hash`, added if new.
    fn group(
        &self,
        allocator: &dyn Allocator,
        groups: &mut Groups,
        hash: u64,
        keys: &[i64],
        nulls: u64,
    ) -> Result<usize, Error> {
        let mask = groups.slots.len() - 1;
        #[expect(clippy::cast_possible_truncation, reason = "masked")]
        let mut slot = hash as usize & mask;
        loop {
            let group = *at!(groups.slots, slot) as usize;
            if group == 0 {
                break;
            }
            let found = group - 1;
            let width = keys.len();
            if *at!(groups.nulls, found) == nulls
                && at!(groups.keys, found * width..(found + 1) * width) == keys
            {
                return Ok(found);
            }
            slot = (slot + 1) & mask;
        }
        let found = groups.nulls.len();
        if found == GROUPS_MAX {
            return Err(Error::OutOfMemory);
        }
        groups.keys.extend_from_slice(keys)?;
        groups.nulls.push(nulls).map_err(|_| AllocError)?;
        for _ in 0..self.grouping.aggregates.len() {
            groups.totals.push(Total::default()).map_err(|_| AllocError)?;
        }
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let index = found as u32 + 1;
        *at_mut!(groups.slots, slot) = index;
        if (found + 1) * 2 > groups.slots.len() {
            self.grow(allocator, groups)?;
        }
        Ok(found)
    }

    /// Doubles `groups`' slots, putting each group back.
    fn grow(&self, allocator: &dyn Allocator, groups: &mut Groups) -> Result<(), Error> {
        let mut slots = SlowVec::zeroed(allocator, groups.slots.len() * 2)?;
        let mask = slots.len() - 1;
        let width = self.grouping.keys.len();
        for found in 0..groups.nulls.len() {
            let keys = at!(groups.keys, found * width..(found + 1) * width);
            #[expect(clippy::cast_possible_truncation, reason = "masked")]
            let mut slot = hash(keys, *at!(groups.nulls, found)) as usize & mask;
            while *at!(slots, slot) != 0 {
                slot = (slot + 1) & mask;
            }
            #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
            let index = found as u32 + 1;
            *at_mut!(slots, slot) = index;
        }
        groups.slots = slots;
        Ok(())
    }

    /// Adds row `row` of `input` to `totals`, a group's.
    fn add(&self, input: &Flat, row: u32, totals: &mut [Total]) {
        for (aggregate, total) in self.grouping.aggregates.iter().zip(totals) {
            let Some((position, data_type)) = aggregate.input else {
                total.count += 1;
                continue;
            };
            let column = input.column(position);
            if column.is_null(row) {
                continue;
            }
            match data_type {
                DataType::Int64 => {
                    let value = i128::from(*at!(column.int64s(), row as usize));
                    total.int = match aggregate.function {
                        Function::Min if total.count > 0 => total.int.min(value),
                        Function::Max if total.count > 0 => total.int.max(value),
                        Function::Min | Function::Max => value,
                        _ => total.int + value,
                    };
                }
                DataType::Float64 => {
                    let value = *at!(column.float64s(), row as usize);
                    total.float = match aggregate.function {
                        Function::Min if total.count > 0 => total.float.min(value),
                        Function::Max if total.count > 0 => total.float.max(value),
                        Function::Min | Function::Max => value,
                        _ => total.float + value,
                    };
                }
                DataType::String => {}
            }
            total.count += 1;
        }
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
        let count = self.grouping.aggregates.len();
        let mut values = context.values_buffer(rows * 8)?;
        let mut validity = context.small_buffer(rows.div_ceil(8))?;
        validity.as_mut_slice::<u8>().fill(0);
        let mut any_null = false;
        for (i, value) in values.as_mut_slice::<i64>().iter_mut().enumerate() {
            let total = *at!(groups.totals, (first + i) * count + a);
            let word = match (aggregate.function, data_type) {
                (Function::Count, _) => {
                    Some(i64::try_from(total.count).map_err(|_| Error::Unsupported)?)
                }
                _ if total.count == 0 => None,
                #[expect(clippy::cast_precision_loss, reason = "averages are approximate")]
                (Function::Avg, Some(DataType::Int64)) => {
                    Some((total.int as f64 / total.count as f64).to_bits().cast_signed())
                }
                #[expect(clippy::cast_precision_loss, reason = "averages are approximate")]
                (Function::Avg, _) => {
                    Some((total.float / total.count as f64).to_bits().cast_signed())
                }
                // A sum that doesn't fit in an `Int64` would need a wider type.
                (_, Some(DataType::Int64)) => {
                    Some(i64::try_from(total.int).map_err(|_| Error::Unsupported)?)
                }
                _ => Some(total.float.to_bits().cast_signed()),
            };
            *value = word.unwrap_or(0);
            match word {
                Some(_) => *at_mut!(validity.as_mut_slice::<u8>(), i / 8) |= 1 << (i % 8),
                None => any_null = true,
            }
        }
        Ok(ColumnView::new(context, output_type, values, any_null.then_some(validity))?)
    }

    /// The column of key `k` for groups `first..first + rows`.
    fn key(
        &self,
        context: &mut Context,
        groups: &Groups,
        k: usize,
        first: usize,
        rows: usize,
    ) -> Result<ColumnView, Error> {
        let width = self.grouping.keys.len();
        let mut values = context.values_buffer(rows * 8)?;
        let mut validity = context.small_buffer(rows.div_ceil(8))?;
        validity.as_mut_slice::<u8>().fill(0);
        let mut any_null = false;
        for (i, value) in values.as_mut_slice::<i64>().iter_mut().enumerate() {
            let group = first + i;
            *value = *at!(groups.keys, group * width + k);
            if *at!(groups.nulls, group) >> k & 1 == 0 {
                *at_mut!(validity.as_mut_slice::<u8>(), i / 8) |= 1 << (i % 8);
            } else {
                any_null = true;
            }
        }
        Ok(ColumnView::new(context, DataType::Int64, values, any_null.then_some(validity))?)
    }
}

impl HashAggregate {
    /// Where the columns it reads are in batches.
    fn read(&self) -> impl Iterator<Item = u32> + '_ {
        let inputs = self.grouping.aggregates.iter().filter_map(|aggregate| aggregate.input);
        self.grouping.keys.iter().copied().chain(inputs.map(|(position, _)| position))
    }
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

/// `keys` and which are `nulls`, hashed.
fn hash(keys: &[i64], nulls: u64) -> u64 {
    let mix =
        |hash: u64, word: u64| (hash ^ word).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29);
    keys.iter().fold(mix(0, nulls), |hash, &key| mix(hash, key.cast_unsigned()))
}

impl Breaker for HashAggregate {
    type State = Groups;

    fn new_state(&self, context: &mut Context) -> Result<Groups, Error> {
        let allocator = context.allocator();
        let width = self.grouping.keys.len().max(1);
        let count = self.grouping.aggregates.len().max(1);
        let mut groups = Groups {
            slots: SlowVec::zeroed(allocator, 16)?,
            keys: SlowVec::new(allocator, (GROUPS_MAX * width).next_power_of_two())?,
            nulls: SlowVec::new(allocator, GROUPS_MAX)?,
            totals: SlowVec::new(allocator, (GROUPS_MAX * count).next_power_of_two())?,
            served: 0,
        };
        // All rows are one group, which is there even if there are none.
        if self.grouping.keys.is_empty() {
            self.group(allocator, &mut groups, hash(&[], 0), &[], 0)?;
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
        let input = Flat::new(context, input, self.read())?;
        let mut keys = [0_i64; KEYS_MAX];
        let keys = at_mut!(keys, ..self.grouping.keys.len());
        let count = self.grouping.aggregates.len();
        let mut add = |row: u32| -> Result<(), Error> {
            let mut nulls = 0;
            for (k, (key, &position)) in keys.iter_mut().zip(self.grouping.keys.iter()).enumerate()
            {
                let column = input.column(position);
                if column.is_null(row) {
                    nulls |= 1 << k;
                    *key = 0;
                } else {
                    *key = *at!(column.int64s(), row as usize);
                }
            }
            let group = self.group(allocator, groups, hash(keys, nulls), keys, nulls)?;
            self.add(&input, row, at_mut!(groups.totals, group * count..(group + 1) * count));
            Ok(())
        };
        match input.batch.selection().kept() {
            Kept::All => (0..input.batch.row_count()).try_for_each(&mut add),
            Kept::Select(rows) => rows.iter().try_for_each(|&row| add(u32::from(row))),
            Kept::None => Ok(()),
        }
    }

    fn merge(
        &self,
        context: &mut Context,
        groups: &mut Groups,
        other: Groups,
    ) -> Result<(), Error> {
        let allocator = context.allocator();
        let width = self.grouping.keys.len();
        let count = self.grouping.aggregates.len();
        for found in 0..other.nulls.len() {
            let keys = at!(other.keys, found * width..(found + 1) * width);
            let nulls = *at!(other.nulls, found);
            let group = self.group(allocator, groups, hash(keys, nulls), keys, nulls)?;
            let theirs = at!(other.totals, found * count..(found + 1) * count);
            let ours = at_mut!(groups.totals, group * count..(group + 1) * count);
            for ((aggregate, ours), theirs) in self.grouping.aggregates.iter().zip(ours).zip(theirs)
            {
                if theirs.count == 0 {
                    continue;
                }
                let first = ours.count == 0;
                (ours.int, ours.float) = match aggregate.function {
                    _ if first => (theirs.int, theirs.float),
                    Function::Min => (ours.int.min(theirs.int), ours.float.min(theirs.float)),
                    Function::Max => (ours.int.max(theirs.int), ours.float.max(theirs.float)),
                    _ => (ours.int + theirs.int, ours.float + theirs.float),
                };
                ours.count += theirs.count;
            }
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
        let rows = (groups.nulls.len() - first).min(BATCH_ROWS_MAX as usize);
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
        output.reset(rows as u32);
        for k in 0..self.grouping.keys.len() {
            let column = self.key(context, groups, k, first, rows)?;
            let Ok(()) = output.push_column(column) else { crate::check::check_failed(line!()) };
        }
        for a in 0..self.grouping.aggregates.len() {
            let column = self.result(context, groups, a, first, rows)?;
            let Ok(()) = output.push_column(column) else { crate::check::check_failed(line!()) };
        }
        groups.served += rows;
        Ok(groups.served < groups.nulls.len())
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
        assert_eq!(first.nulls.len(), 2000);
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
