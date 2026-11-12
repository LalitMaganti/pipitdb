//! `hash_aggregate`: the breaker that aggregates rows by their keys, through
//! a `GroupTable` and the aggregate functions, as Perfetto's
//! `HashAggregate` does.

use crate::aggregate::{Function, HashAggregateSpec};
use crate::aggregate_function::{
    AggregateInput, AggregateInputLoader, GroupStates, SeenBytes, finalize_to_buffer,
    make_aggregate_function,
};
use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType, Forms};
use crate::context::Context;
use crate::error::Error;
use crate::flat_hash_map::{FlatHashMap, Key};
use crate::group_table::{GroupTable, KEYS_MAX};
use crate::murmur_hash;
use crate::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use crate::selection::Kept;
use crate::slow_vec::SlowVec;
use crate::step::{Breaker, DynOperator};
use crate::string_pool::CACHED_SLOTS;

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
    /// For each `DISTINCT` aggregate, the values of each group it's read.
    distinct: SlowVec<Option<FlatHashMap<GroupValue, ()>>>,
    /// How many groups have been output.
    emitted: usize,
}

impl HashAggregate {
    fn new(spec: HashAggregateSpec) -> HashAggregate {
        check!(spec.key_columns.len() <= KEYS_MAX);
        let calls = spec.aggregates.iter();
        let state_words =
            calls.map(|call| make_aggregate_function(call.function).state_words()).sum();
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

/// A value of a group, as a `DISTINCT` aggregate keeps the ones it's read.
#[derive(Clone, Copy, PartialEq, Eq)]
struct GroupValue {
    group: u32,
    value: i64,
}

impl Key for GroupValue {
    fn hash(&self) -> u64 {
        murmur_hash::mix(self.value.cast_unsigned() ^ murmur_hash::mix(u64::from(self.group)))
    }
}

/// How many rows ahead `first_values` fetches a row's slot.
const SLOT_AHEAD: usize = 16;

/// The values of `input`'s rows that aren't null and that their group, in
/// `groups`, all 0 if `None`, hadn't had before, adding them to `set`; and
/// each one's group, and how many there are.
fn first_values(
    context: &mut Context,
    set: &mut FlatHashMap<GroupValue, ()>,
    input: &AggregateInput,
    groups: Option<&[u32]>,
) -> Result<(Buffer, Buffer, usize), Error> {
    // A set too big for the cache misses it a row, so each row's slot is
    // fetched ahead. `prefetch` does nothing on other targets, so they don't.
    let ahead = cfg!(target_arch = "x86_64") && set.capacity() > CACHED_SLOTS;
    let mut values = context.values_buffer(input.rows * 8)?;
    let mut firsts = context.indices_buffer(input.rows * 4)?;
    let (kept_values, kept_groups) = (values.as_mut_slice::<i64>(), firsts.as_mut_slice::<u32>());
    let all = at!(input.values, ..input.rows);
    let group_of = |row: usize| groups.map_or(0, |groups| *at!(groups, row));
    let mut kept = 0;
    for (row, &value) in all.iter().enumerate() {
        if ahead && let Some(&next) = all.get(row + SLOT_AHEAD) {
            set.prefetch(GroupValue { group: group_of(row + SLOT_AHEAD), value: next }.hash());
        }
        let group = group_of(row);
        if input.is_valid(row)
            && set.insert(context.allocator(), GroupValue { group, value }, ())?.1
        {
            *at_mut!(kept_values, kept) = value;
            *at_mut!(kept_groups, kept) = group;
            kept += 1;
        }
    }
    Ok((values, firsts, kept))
}

/// A batch whose columns this reads are flat, but for string keys'
/// dictionaries: they're in whatever form the first op to read them takes,
/// so those that aren't are made flat.
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
            let forms = if column.data_type() == DataType::String {
                Forms::FLAT | Forms::DICTIONARY
            } else {
                Forms::FLAT
            };
            if !column.is_in(forms) && at!(copies, position as usize).is_none() {
                let mut copy = column.clone();
                copy.make_in(context, forms)?;
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
            Kept::Prefix | Kept::None => &[],
        }
    }
}

impl Breaker for HashAggregate {
    type State = State;

    fn new_state(&self, context: &mut Context) -> Result<State, Error> {
        let allocator = context.allocator();
        let groups = GroupTable::new(
            allocator,
            self.spec.key_columns.len(),
            self.spec.string_keys,
            self.state_words,
        )?;
        let mut total = SlowVec::new(allocator, self.state_words.next_power_of_two().max(1))?;
        total.extend_zeroed(self.state_words)?;
        let mut seen = SlowVec::fixed(allocator, self.spec.aggregates.len())?;
        for _ in 0..self.spec.aggregates.len() {
            seen.push(SeenBytes::new(allocator)?)?;
        }
        let mut distinct = SlowVec::fixed(allocator, self.spec.aggregates.len())?;
        for call in self.spec.aggregates.iter() {
            check!(!call.distinct || call.function != Function::CountStar);
            let set = if call.distinct { Some(FlatHashMap::for_len(allocator, 0)?) } else { None };
            distinct.push(set)?;
        }
        Ok(State { groups, total, consumed: false, seen, distinct, emitted: 0 })
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
        let mut numbers = None;
        if !self.spec.key_columns.is_empty() {
            let mut keys = [input.column(0); KEYS_MAX];
            for (key, &column) in keys.iter_mut().zip(self.spec.key_columns.iter()) {
                *key = input.column(column);
            }
            let keys = at!(keys, ..self.spec.key_columns.len());
            let groups = numbers.insert(context.indices_buffer(rows * 4)?).as_mut_slice::<u32>();
            state.groups.find_or_create(context, keys, input.select(), groups)?;
        }
        let group_count = self.group_count(state);
        let State { groups, total, seen, distinct, .. } = state;
        let (base, stride) = self.state_base(groups, total);
        let mut overflow = false;
        let mut offset = 0;
        for (a, call) in self.spec.aggregates.iter().enumerate() {
            let function = make_aggregate_function(call.function);
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
            let states = Self::states(base, stride, offset, seen);
            offset += function.state_words();
            let numbers = numbers.as_ref().map(Buffer::as_slice::<u32>);
            if let Some(set) = at_mut!(distinct, a) {
                // Only the rows with values their group hasn't had, as if the
                // batch were only those.
                let (values, firsts, rows) = first_values(context, set, &loaded, numbers)?;
                let kept = at!(values.as_slice::<i64>(), ..rows);
                let values = AggregateInput { values: kept, valid: None, rows };
                let firsts = at!(firsts.as_slice::<u32>(), ..rows);
                match numbers {
                    Some(_) => function.update(&values, firsts, states, &mut overflow),
                    None => function.update_group(&values, 0, states, &mut overflow),
                }
                continue;
            }
            match numbers {
                Some(numbers) => function.update(&loaded, numbers, states, &mut overflow),
                None => function.update_group(&loaded, 0, states, &mut overflow),
            }
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
            let function = make_aggregate_function(call.function);
            // Without keys there is always one group, even with no rows. In
            // that case it has no value.
            let seen = if function.tracks_seen() {
                at_mut!(seen, a).get(groups, !*consumed)?
            } else {
                None
            };
            let states = Self::states(base, stride, offset, seen);
            offset += function.state_words();
            let output_type = call.function.output_type();
            let column = finalize_to_buffer(context, function, output_type, states, numbers)?;
            let Ok(()) = output.push_column(column) else { failed!() };
        }
        state.emitted += count;
        Ok(state.emitted < groups)
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::String;
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
        let mut words = Buffer::allocate(&Heap, values.len() * 8).unwrap();
        let mut bits = Buffer::allocate(&Heap, values.len().div_ceil(8)).unwrap();
        bits.as_mut_slice::<u8>().fill(0);
        for (i, value) in values.iter().enumerate() {
            words.as_mut_slice::<i64>()[i] = value.unwrap_or(0);
            bits.as_mut_slice::<u8>()[i / 8] |= u8::from(value.is_some()) << (i % 8);
        }
        let validity = values.iter().any(Option::is_none).then_some(bits);
        ColumnView::new(&mut Context::new(&Heap), DataType::Int64, words, validity).unwrap()
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

    /// A string column of `values`, null where `None`.
    fn strings(values: &[Option<String>]) -> ColumnView {
        let mut views = Buffer::allocate(&Heap, values.len() * 8).unwrap();
        let mut bits = Buffer::allocate(&Heap, values.len().div_ceil(8)).unwrap();
        let mut bytes = Vec::new();
        for (i, value) in values.iter().enumerate() {
            let value = value.as_deref().unwrap_or_default();
            let view = [u32::try_from(bytes.len()).unwrap(), u32::try_from(value.len()).unwrap()];
            views.as_mut_slice::<u32>()[2 * i..2 * i + 2].copy_from_slice(&view);
            bytes.extend_from_slice(value.as_bytes());
        }
        for (i, value) in values.iter().enumerate() {
            bits.as_mut_slice::<u8>()[i / 8] |= u8::from(value.is_some()) << (i % 8);
        }
        let mut stored = Buffer::allocate(&Heap, bytes.len()).unwrap();
        stored.as_mut_slice::<u8>().copy_from_slice(&bytes);
        let validity = values.iter().any(Option::is_none).then_some(bits);
        ColumnView::strings(&mut Context::new(&Heap), views, stored, validity).unwrap()
    }

    /// `Rows`, with its key `k` as a string column, of each key's digits:
    /// flat, or, if there are dictionaries, as indices into its batch's.
    struct StringKeys(Rows, Vec<ColumnView>);

    impl Source for StringKeys {
        type State = usize;

        fn new_state(&self, context: &mut Context) -> Result<usize, Error> {
            self.0.new_state(context)
        }

        fn next<'s>(
            &'s self,
            context: &mut Context,
            i: &mut usize,
            batch: &mut RowBatch<'s>,
        ) -> Result<bool, Error> {
            if !self.0.next(context, i, batch)? {
                return Ok(false);
            }
            let keys = batch.column(0);
            let digits = |row: u32| format!("{}", keys.int64s()[row as usize]);
            let texts: Vec<_> = (0..keys.row_count())
                .map(|row| (!keys.is_null(row)).then(|| digits(row)))
                .collect();
            let Some(entries) = self.1.get(*i - 1) else {
                batch.set_column(0, strings(&texts));
                return Ok(true);
            };
            let found = |text: &Option<String>, e: u32| match text {
                None => entries.is_null(e),
                Some(text) => {
                    !entries.is_null(e)
                        && entries.values().strings().get(e as usize) == text.as_bytes()
                }
            };
            let mut indices = Buffer::allocate(&Heap, texts.len() * 4).unwrap();
            for (index, text) in indices.as_mut_slice::<u32>().iter_mut().zip(&texts) {
                *index = (0..entries.row_count()).find(|&e| found(text, e)).unwrap();
            }
            batch.set_column(0, ColumnView::dictionary(context, entries, indices).unwrap());
            Ok(true)
        }
    }

    /// By the columns at `keys`, `COUNT(*)`, then `COUNT`, `SUM`, `MIN` and
    /// `MAX` of the column at 1.
    fn spec(keys: &[u32]) -> HashAggregateSpec {
        let functions =
            [Function::CountStar, Function::Count, Function::Sum, Function::Min, Function::Max];
        let aggregates = functions.into_iter().map(|function| AggregateCall {
            function,
            column: 1,
            distinct: false,
        });
        HashAggregateSpec {
            key_columns: SlowVec::fixed_from(&Heap, keys.iter().copied()).unwrap(),
            string_keys: 0,
            aggregates: SlowVec::fixed_from(&Heap, aggregates).unwrap(),
        }
    }

    /// Each row out, its columns as integers, or floats' bits, or strings'
    /// digits, or `None` if null, sorted.
    fn run<S: Source + 'static>(
        rows: S,
        spec: HashAggregateSpec,
    ) -> Result<Vec<Vec<Option<i64>>>, Error> {
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
                        DataType::Float64 => {
                            column.values().float64s()[row as usize].to_bits().cast_signed()
                        }
                        DataType::String => {
                            let digits = column.values().strings().get(row as usize);
                            core::str::from_utf8(digits).unwrap().parse().unwrap()
                        }
                        DataType::Int64 => column.int64s()[row as usize],
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
    fn groups_by_strings_as_by_the_integers_they_spell() {
        let rows = || Rows {
            batches: alloc::vec![
                alloc::vec![(Some(1), Some(1)), (Some(12), Some(2)), (None, Some(1))],
                alloc::vec![(Some(12), Some(4)), (None, None), (Some(-3), Some(1))],
                Vec::new(),
                alloc::vec![(Some(12), Some(7)), (Some(-3), Some(2)), (None, Some(5))],
            ],
            dropped: alloc::vec![1],
        };
        // The first three batches share a dictionary, and the last has its
        // own, with entries in another order.
        let first = strings(&[Some("1".into()), Some("12".into()), None, Some("-3".into())]);
        let last = strings(&[Some("-3".into()), None, Some("12".into()), Some("x".into())]);
        let dictionaries = [Vec::new(), alloc::vec![first.clone(), first.clone(), first, last]];
        for (keys, groups) in [(&[0][..], 4), (&[0, 1], 6), (&[1, 0], 6)] {
            let expected = run(rows(), spec(keys)).unwrap();
            assert_eq!(expected.len(), groups);
            for dictionaries in &dictionaries {
                let rows = StringKeys(rows(), dictionaries.clone());
                let mut by_strings = spec(keys);
                by_strings.string_keys = 1 << keys.iter().position(|&key| key == 0).unwrap();
                assert_eq!(run(rows, by_strings).unwrap(), expected);
            }
        }
        let mut by_strings = spec(&[0]);
        by_strings.string_keys = 1;
        let none = StringKeys(Rows { batches: Vec::new(), dropped: Vec::new() }, Vec::new());
        assert!(run(none, by_strings).unwrap().is_empty());
    }

    #[test]
    fn tables_sharing_a_pool_number_their_own_groups() {
        let mut context = Context::new(&Heap);
        let mut tables = [(); 2].map(|()| GroupTable::new(&Heap, 1, 1, 0).unwrap());
        let column = |texts: &[Option<&str>]| {
            strings(&texts.iter().map(|text| text.map(String::from)).collect::<Vec<_>>())
        };
        let mut groups = [0; 5];
        let first = column(&[Some("a"), Some("b"), Some("c")]);
        tables[0].find_or_create(&mut context, &[&first], &[], &mut groups[..3]).unwrap();
        assert_eq!(groups[..3], [0, 1, 2]);
        // The pool numbered "a" and "c" for the first table, and numbers "d"
        // after them.
        let second = column(&[Some("c"), None, Some("a"), Some("c"), Some("d")]);
        tables[1].find_or_create(&mut context, &[&second], &[], &mut groups).unwrap();
        assert_eq!(groups, [0, 1, 2, 0, 3]);
        let keys = tables[1].write_keys(&mut context, 0, 0..4).unwrap();
        let key =
            |row: u32| (!keys.is_null(row)).then(|| keys.values().strings().get(row as usize));
        assert_eq!(
            (0..4).map(key).collect::<Vec<_>>(),
            [Some(&b"c"[..]), None, Some(b"a"), Some(b"d")]
        );
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

    #[test]
    fn averages_exactly_past_int64() {
        // Keyed and not, with a null, a dropped row, and a sum past `Int64`
        // whose mean isn't.
        let big = i64::MAX - 1;
        for keys in [&[][..], &[0]] {
            let rows = Rows {
                batches: alloc::vec![
                    alloc::vec![(Some(1), Some(big)), (Some(1), Some(7)), (Some(1), None)],
                    alloc::vec![(Some(1), Some(big))],
                ],
                dropped: alloc::vec![1],
            };
            let functions = [Function::Avg, Function::CountStar];
            let aggregates = functions.into_iter().map(|function| AggregateCall {
                function,
                column: 1,
                distinct: false,
            });
            let spec = HashAggregateSpec {
                key_columns: SlowVec::fixed_from(&Heap, keys.iter().copied()).unwrap(),
                string_keys: 0,
                aggregates: SlowVec::fixed_from(&Heap, aggregates).unwrap(),
            };
            let out = run(rows, spec).unwrap();
            let mean = out[0][keys.len()].map(|bits| f64::from_bits(bits.cast_unsigned()));
            #[expect(clippy::cast_precision_loss, reason = "the mean expected")]
            let expected = big as f64;
            assert_eq!((mean, out[0][keys.len() + 1]), (Some(expected), Some(3)));
        }
        // With no rows, the mean is null.
        let rows = Rows { batches: Vec::new(), dropped: Vec::new() };
        let call = AggregateCall { function: Function::Avg, column: 1, distinct: false };
        let spec = HashAggregateSpec {
            key_columns: SlowVec::fixed_from(&Heap, [].into_iter()).unwrap(),
            string_keys: 0,
            aggregates: SlowVec::fixed_from(&Heap, [call].into_iter()).unwrap(),
        };
        assert_eq!(run(rows, spec).unwrap(), [alloc::vec![None]]);
    }

    #[test]
    fn reads_each_groups_distinct_values() {
        // `COUNT(DISTINCT x)`, `SUM(DISTINCT x)`, `COUNT(*)` and `SUM(x)`, by
        // the key and not.
        let calls = [
            (Function::Count, true),
            (Function::Sum, true),
            (Function::CountStar, false),
            (Function::Sum, false),
        ];
        let spec = |keys: &[u32]| {
            let calls =
                calls.map(|(function, distinct)| AggregateCall { function, column: 1, distinct });
            HashAggregateSpec {
                key_columns: SlowVec::fixed_from(&Heap, keys.iter().copied()).unwrap(),
                string_keys: 0,
                aggregates: SlowVec::fixed_from(&Heap, calls.into_iter()).unwrap(),
            }
        };
        // Repeats within and across batches, a null, row 3 of each batch
        // dropped, and key 2's only value null.
        let rows = || Rows {
            batches: alloc::vec![
                alloc::vec![
                    (Some(1), Some(5)),
                    (Some(1), Some(5)),
                    (Some(1), None),
                    (Some(1), Some(9))
                ],
                alloc::vec![
                    (Some(1), Some(7)),
                    (Some(1), Some(5)),
                    (Some(2), None),
                    (Some(3), Some(5))
                ],
            ],
            dropped: alloc::vec![3],
        };
        assert_eq!(
            run(rows(), spec(&[])).unwrap(),
            [alloc::vec![Some(2), Some(12), Some(6), Some(22)]]
        );
        assert_eq!(
            run(rows(), spec(&[0])).unwrap(),
            [
                alloc::vec![Some(1), Some(2), Some(12), Some(5), Some(22)],
                alloc::vec![Some(2), Some(0), None, Some(1), None],
            ]
        );
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
