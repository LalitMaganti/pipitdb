//! `hash_aggregate`: the breaker that aggregates rows by their keys, through
//! a `GroupTable` and the aggregate functions, as Perfetto's
//! `HashAggregate` does.

use crate::aggregate::{Function, HashAggregateSpec};
use crate::aggregate_function::{
    AggregateInput, AggregateInputLoader, GroupStates, SeenBytes, finalize_to_buffer,
    make_aggregate_function,
};
use crate::allocator::{AllocError, Allocator};
use crate::column::{ColumnView, Forms};
use crate::context::Context;
use crate::error::Error;
use crate::group_table::{GroupTable, KEYS_MAX, Numbers};
use crate::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use crate::selection::Kept;
use crate::slow_vec::SlowVec;
use crate::step::{Breaker, DynOperator};

/// The breaker that aggregates by `spec`.
pub fn hash_aggregate<'a>(
    allocator: &dyn Allocator,
    spec: HashAggregateSpec,
) -> Result<DynOperator<'a>, AllocError> {
    DynOperator::new(allocator, HashAggregate::new(spec))
}

/// Groups rows by their keys and outputs a row for each group: its keys,
/// then one column for each aggregate, in the order groups were first seen.
///
/// Input rows are not stored. Each row is added to its group's state for
/// each aggregate, which is stored next to the group's keys, so memory grows
/// with the groups, not the rows.
struct HashAggregate {
    spec: HashAggregateSpec,
    /// How many words of state a group keeps: each aggregate's, one after
    /// another.
    state_words: usize,
}

struct State {
    /// The groups, by their keys.
    groups: GroupTable,
    /// With no keys, the state of the single group.
    total: SlowVec<i64>,
    /// Whether any input row was read.
    consumed: bool,
    /// For each aggregate, which groups have seen a value.
    seen: SlowVec<SeenBytes>,
    /// How many groups have been output.
    emitted: usize,
}

impl HashAggregate {
    fn new(spec: HashAggregateSpec) -> HashAggregate {
        check!(spec.key_columns.len() <= KEYS_MAX);
        let calls = spec.aggregates.iter();
        let state_words = calls
            .map(|call| make_aggregate_function(call.function, call.data_type).state_words())
            .sum();
        HashAggregate { spec, state_words }
    }

    /// The states of every group, from the first, and how many words apart
    /// groups' states are.
    fn state_base<'s>(
        &self,
        groups: &'s mut GroupTable,
        total: &'s mut [i64],
    ) -> (&'s mut [i64], usize) {
        if self.spec.key_columns.is_empty() {
            return (total, self.state_words);
        }
        let stride = groups.row_stride();
        (groups.payloads(), stride)
    }

    /// The states of the aggregate whose words start at `offset` of each
    /// group's, with `seen`.
    fn states<'s>(
        base: &'s mut [i64],
        stride: usize,
        offset: usize,
        seen: Option<&'s mut [u8]>,
    ) -> GroupStates<'s> {
        // No groups, no words.
        let words = base.get_mut(offset..).unwrap_or_default();
        GroupStates { words, stride, seen }
    }

    fn group_count(&self, state: &State) -> usize {
        if self.spec.key_columns.is_empty() { 1 } else { state.groups.len() }
    }

    /// Where the columns it reads are in batches: the keys, then the
    /// aggregates'.
    fn read(&self) -> impl Iterator<Item = u32> + '_ {
        let inputs =
            self.spec.aggregates.iter().filter(|call| call.function != Function::CountStar);
        self.spec.key_columns.iter().copied().chain(inputs.map(|call| call.column))
    }
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
    type State = State;

    fn new_state(&self, context: &mut Context) -> Result<State, Error> {
        let allocator = context.allocator();
        let groups = GroupTable::new(allocator, self.spec.key_columns.len(), self.state_words)?;
        let mut total = SlowVec::new(allocator, self.state_words.next_power_of_two().max(1))?;
        total.extend_zeroed(self.state_words)?;
        let mut seen = SlowVec::fixed(allocator, self.spec.aggregates.len())?;
        for _ in 0..self.spec.aggregates.len() {
            seen.push(SeenBytes::new(allocator)?)?;
        }
        Ok(State { groups, total, consumed: false, seen, emitted: 0 })
    }

    fn consume(
        &self,
        context: &mut Context,
        state: &mut State,
        input: &RowBatch,
    ) -> Result<(), Error> {
        let rows = input.selection().len() as usize;
        if rows == 0 {
            return Ok(());
        }
        let input = Flat::new(context, input, self.read())?;
        // Find each row's group. Groups are numbered in the order first seen,
        // so a row starts a new group when its number equals the group count.
        // With no keys, every row is in the one group.
        state.consumed = true;
        let mut buffers = None;
        // Each row's group, where each run of rows of one group starts, and
        // whether runs are long enough to fold a run at a time.
        let mut found: Option<(&[u32], &[u32], bool)> = None;
        if !self.spec.key_columns.is_empty() {
            let mut keys = [input.column(0); KEYS_MAX];
            for (key, &column) in keys.iter_mut().zip(self.spec.key_columns.iter()) {
                *key = input.column(column);
            }
            let keys = at!(keys, ..self.spec.key_columns.len());
            let (groups, starts) = buffers.insert((
                context.indices_buffer(rows * 4)?,
                context.indices_buffer((rows + 1) * 4)?,
            ));
            let mut numbers =
                Numbers { groups: groups.as_mut_slice(), starts: starts.as_mut_slice(), runs: 0 };
            state.groups.find_or_create(context, keys, input.select(), &mut numbers)?;
            let Numbers { groups, starts, runs } = numbers;
            #[expect(clippy::cast_possible_truncation, reason = "a batch's rows")]
            let end = rows as u32;
            *at_mut!(starts, runs) = end;
            found = Some((groups, at!(starts, ..=runs), runs * 2 <= rows));
        }
        let group_count = self.group_count(state);
        let State { groups, total, seen, .. } = state;
        let (base, stride) = self.state_base(groups, total);
        let mut overflow = false;
        let mut offset = 0;
        for (a, call) in self.spec.aggregates.iter().enumerate() {
            let function = make_aggregate_function(call.function, call.data_type);
            let mut buffers = AggregateInputLoader::default();
            let mut loaded = AggregateInput { values: &[], valid: None, rows };
            if function.reads_input() {
                loaded = buffers.load(context, input.column(call.column), input.select(), rows)?;
            }
            let seen = if function.tracks_seen() {
                at_mut!(seen, a).get(group_count, loaded.valid.is_some())?
            } else {
                None
            };
            let mut seen = seen;
            match found {
                // A run's rows are one group's: folded as a batch of no keys is.
                Some((groups, starts, true)) if loaded.valid.is_none() => {
                    for run in starts.windows(2) {
                        let (start, end) = (*at!(run, 0) as usize, *at!(run, 1) as usize);
                        let values = loaded.values.get(start..end).unwrap_or_default();
                        let input = AggregateInput { values, valid: None, rows: end - start };
                        let states = Self::states(base, stride, offset, seen.as_deref_mut());
                        function.update_group(&input, *at!(groups, start), states, &mut overflow);
                    }
                }
                Some((groups, _, _)) => {
                    let states = Self::states(base, stride, offset, seen);
                    function.update(&loaded, groups, states, &mut overflow);
                }
                None => {
                    let states = Self::states(base, stride, offset, seen);
                    function.update_group(&loaded, 0, states, &mut overflow);
                }
            }
            offset += function.state_words();
        }
        if overflow { Err(Error::Overflow) } else { Ok(()) }
    }

    fn finalize(&self, _: &mut Context, _: &mut State) -> Result<(), Error> {
        Ok(())
    }

    fn serve(
        &self,
        context: &mut Context,
        state: &mut State,
        output: &mut RowBatch,
    ) -> Result<bool, Error> {
        let groups = self.group_count(state);
        // Output up to a full batch of groups.
        let first = state.emitted;
        let count = (groups - first).min(BATCH_ROWS_MAX as usize);
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
        output.reset(count as u32);
        for k in 0..self.spec.key_columns.len() {
            let column = state.groups.write_keys(context, k, first..first + count)?;
            let Ok(()) = output.push_column(column) else { failed!() };
        }
        let mut numbers = context.indices_buffer(count * 4)?;
        let numbers = numbers.as_mut_slice::<u32>();
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        for (number, group) in numbers.iter_mut().zip(first..) {
            *number = group as u32;
        }
        let State { groups: table, total, seen, consumed, .. } = state;
        let (base, stride) = self.state_base(table, total);
        let mut offset = 0;
        for (a, call) in self.spec.aggregates.iter().enumerate() {
            let function = make_aggregate_function(call.function, call.data_type);
            // Without keys there is always one group, even with no rows. In
            // that case it has no value.
            let seen = if function.tracks_seen() {
                at_mut!(seen, a).get(groups, !*consumed)?
            } else {
                None
            };
            let states = Self::states(base, stride, offset, seen);
            offset += function.state_words();
            let column =
                finalize_to_buffer(context, function, call.output_type(), states, numbers)?;
            let Ok(()) = output.push_column(column) else { failed!() };
        }
        state.emitted += count;
        Ok(state.emitted < groups)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::aggregate::AggregateCall;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::DataType;
    use crate::pipeline::Pipeline;
    use crate::query_allocators::QueryAllocators;
    use crate::step::{DynSource, Source, Step};

    /// An `Int64` column of `values`, null where `None`.
    fn column(values: &[Option<i64>]) -> ColumnView {
        typed(DataType::Int64, values)
    }

    /// A column of `data_type` of `values`, integers or floats' bits, null
    /// where `None`.
    fn typed(data_type: DataType, values: &[Option<i64>]) -> ColumnView {
        let mut words = Buffer::allocate(&Heap, values.len() * 8).unwrap();
        let mut bits = Buffer::allocate(&Heap, values.len().div_ceil(8)).unwrap();
        bits.as_mut_slice::<u8>().fill(0);
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
            assert!(batch.push_column(column(&keys)).is_ok());
            assert!(batch.push_column(column(&values)).is_ok());
            batch.selection_mut().retain(|row| !self.dropped.contains(&row));
            Ok(true)
        }
    }

    /// By the columns at `keys`, `COUNT(*)`, then `COUNT`, `SUM`, `MIN` and
    /// `MAX` of the column at 1, an `Int64`.
    fn spec(keys: &[u32]) -> HashAggregateSpec {
        spec_of(keys, DataType::Int64, false)
    }

    /// As `spec`, of a column of `data_type`, and then `AVG` if `avg`.
    fn spec_of(keys: &[u32], data_type: DataType, avg: bool) -> HashAggregateSpec {
        let functions = [
            Function::CountStar,
            Function::Count,
            Function::Sum,
            Function::Min,
            Function::Max,
            Function::Avg,
        ];
        let functions = &functions[..if avg { 6 } else { 5 }];
        let aggregates =
            functions.iter().map(|&function| AggregateCall { function, column: 1, data_type });
        HashAggregateSpec {
            key_columns: SlowVec::fixed_from(&Heap, keys.iter().copied()).unwrap(),
            aggregates: SlowVec::fixed_from(&Heap, aggregates).unwrap(),
        }
    }

    /// Each row out, its columns as integers, or `None` if null, sorted.
    fn run(rows: impl Source, spec: HashAggregateSpec) -> Result<Vec<Vec<Option<i64>>>, Error> {
        let mut steps = SlowVec::new(&Heap, 1).unwrap();
        assert!(steps.push(Step::Operator(hash_aggregate(&Heap, spec).unwrap())).is_ok());
        let pipeline = Pipeline::new(DynSource::new(&Heap, rows).unwrap(), steps);
        let query = QueryAllocators::new(&Heap);
        let mut execution = pipeline.start(&query)?;
        let mut batch = RowBatch::new();
        let mut out = Vec::new();
        while execution.next(&mut batch)? {
            for row in 0..batch.row_count() {
                let values = (0..batch.column_count()).map(|i| {
                    let column = batch.column(i);
                    (!column.is_null(row)).then(|| match column.data_type() {
                        DataType::Float64 => bits(column.float64s()[row as usize]),
                        _ => column.int64s()[row as usize],
                    })
                });
                out.push(values.collect());
            }
        }
        out.sort();
        Ok(out)
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
            alloc::vec![None, Some(1), Some(1), Some(5), Some(5), Some(5)],
            // `k` = 1: 10 and 30.
            alloc::vec![Some(1), Some(2), Some(2), Some(40), Some(10), Some(30)],
            // `k` = 2: only nulls, so `SUM`, `MIN` and `MAX` are null.
            alloc::vec![Some(2), Some(2), Some(0), None, None, None],
        ];
        assert_eq!(run(rows, spec(&[0])).unwrap(), expected);
    }

    #[test]
    fn groups_by_two_keys() {
        let rows = Rows {
            batches: alloc::vec![alloc::vec![
                (Some(1), Some(1)),
                (Some(1), Some(2)),
                (Some(1), Some(1)),
                (None, Some(1)),
                (None, None),
            ]],
            dropped: Vec::new(),
        };
        let mut by_both = spec(&[0, 1]);
        by_both.aggregates.retain(|call| call.function == Function::CountStar);
        let expected = [
            alloc::vec![None, None, Some(1)],
            alloc::vec![None, Some(1), Some(1)],
            alloc::vec![Some(1), Some(1), Some(2)],
            alloc::vec![Some(1), Some(2), Some(1)],
        ];
        assert_eq!(run(rows, by_both).unwrap(), expected);
    }

    #[test]
    fn aggregates_all_rows_without_keys() {
        let rows = Rows { batches: Vec::new(), dropped: Vec::new() };
        // With no rows, `COUNT` is 0 and the rest are null.
        assert_eq!(
            run(rows, spec(&[])).unwrap(),
            [alloc::vec![Some(0), Some(0), None, None, None]]
        );
        let rows = Rows { batches: Vec::new(), dropped: Vec::new() };
        assert!(run(rows, spec(&[0])).unwrap().is_empty());
        let half = i64::MAX / 2;
        let rows = Rows {
            batches: alloc::vec![
                alloc::vec![(None, Some(-7)), (None, None), (None, Some(half))],
                alloc::vec![(None, Some(-half)), (None, Some(3))],
            ],
            dropped: Vec::new(),
        };
        let expected = [alloc::vec![Some(5), Some(4), Some(-4), Some(-half), Some(half)]];
        assert_eq!(run(rows, spec(&[])).unwrap(), expected);
    }

    /// A float's bits, as `run` gives them.
    fn bits(value: f64) -> i64 {
        value.to_bits().cast_signed()
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

    #[test]
    fn aggregates_floats_and_means() {
        let x = [Some(1.5), None, Some(-2.0), None].map(|x| x.map(bits));
        let floats = typed(DataType::Float64, &x);
        let batches = Columns(alloc::vec![alloc::vec![
            column(&[Some(1), Some(1), Some(1), Some(2)]),
            floats
        ]]);
        let expected = [
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
        ];
        assert_eq!(run(batches, spec_of(&[0], DataType::Float64, true)).unwrap(), expected);
        // Integers' means are exact, though their sum passes an `Int64`.
        let half = i64::MAX / 2;
        let values = [half, half, half, -half, -half].map(Some);
        let batches = Columns(alloc::vec![alloc::vec![column(&[Some(1); 5]), column(&values)]]);
        let mut means = spec_of(&[], DataType::Int64, true);
        means.aggregates.retain(|call| call.function == Function::Avg);
        #[expect(clippy::cast_precision_loss, reason = "as `AVG` does")]
        let mean = bits(half as f64 / 5.0);
        assert_eq!(run(batches, means).unwrap(), [[Some(mean)]]);
    }

    #[test]
    fn folds_runs_of_one_group() {
        // Runs of keys 1, 2, then 1 again, by one key and by two.
        let keys = [1, 1, 1, 2, 2, 1, 1, 1].map(Some);
        let values = [1, 2, 3, 4, 5, 6, 7, 8].map(Some);
        let rows = || Rows {
            batches: alloc::vec![keys.into_iter().zip(values).collect()],
            dropped: Vec::new(),
        };
        let expected = [
            alloc::vec![Some(1), Some(6), Some(6), Some(27), Some(1), Some(8)],
            alloc::vec![Some(2), Some(2), Some(2), Some(9), Some(4), Some(5)],
        ];
        assert_eq!(run(rows(), spec(&[0])).unwrap(), expected);
        let mut by_both = spec(&[0, 0]);
        by_both.aggregates.retain(|call| call.function == Function::Sum);
        assert_eq!(
            run(rows(), by_both).unwrap(),
            [[Some(1), Some(1), Some(27)], [Some(2), Some(2), Some(9)]]
        );
    }

    #[test]
    fn keeps_zero_apart_from_an_earlier_batchs_nulls() {
        // The second batch has no nulls, so its keys are found by value alone,
        // past the group of the first's nulls, stored with a key of 0.
        let batches = [[None, Some(5)], [Some(0), Some(5)]].map(|keys| {
            let key = column(&keys);
            alloc::vec![key.clone(), key]
        });
        let mut counts = spec(&[0]);
        counts.aggregates.retain(|call| call.function == Function::CountStar);
        let expected = [[None, Some(1)], [Some(0), Some(1)], [Some(5), Some(2)]];
        assert_eq!(run(Columns(batches.into()), counts).unwrap(), expected);
    }

    #[test]
    fn sums_fail_past_int64() {
        for keys in [&[][..], &[0]] {
            // The least `Int64` fits; one less doesn't.
            let rows = Rows {
                batches: alloc::vec![
                    alloc::vec![(Some(1), Some(i64::MIN)), (Some(1), Some(0))],
                    alloc::vec![(Some(1), Some(-1))],
                ],
                dropped: Vec::new(),
            };
            assert_eq!(run(rows, spec(keys)), Err(Error::Overflow));
        }
    }
}
