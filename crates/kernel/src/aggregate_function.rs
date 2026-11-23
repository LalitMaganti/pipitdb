//! `AggregateFunction`: what an aggregate function, such as `SUM`, keeps for
//! each group, and how it updates and outputs it, a batch at a time, as
//! Perfetto's do, for the operators that aggregate.

use crate::bitmap::BitSlice;
use core::cmp::Ordering;

use crate::aggregate::Function;
use crate::allocator::AllocError;
use crate::blocks::{Blocks, address};
use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType, Form, Validity};
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;
use crate::group_table::write_strings;
use crate::row_batch::BATCH_ROWS_MAX;
use crate::string_pool::Id;

/// What an aggregate reads of one batch: a value a row, contiguous, and
/// which rows hold one.
#[derive(Clone, Copy)]
pub(crate) struct AggregateInput<'a> {
    /// A value a row: none for `COUNT(*)`.
    pub(crate) values: &'a [i64],
    /// Bit r set if row r holds a value, or `None` if every row does.
    pub(crate) valid: Option<&'a [u64]>,
    /// How many rows there are.
    pub(crate) rows: usize,
}

impl AggregateInput<'_> {
    /// Whether row `row` holds a value.
    pub(crate) fn is_valid(&self, row: usize) -> bool {
        self.valid.is_none_or(|valid| BitSlice::of(valid).get(row))
    }

    /// How many rows hold a value.
    fn count_valid(&self) -> usize {
        let Some(valid) = self.valid else { return self.rows };
        let words = at!(valid, ..BitSlice::<u64>::words_for(self.rows)).iter().enumerate();
        let counts = words.map(|(w, &word)| {
            let n = (self.rows - w * 64).min(64);
            let mask = if n == 64 { u64::MAX } else { (1 << n) - 1 };
            (word & mask).count_ones()
        });
        counts.sum::<u32>() as usize
    }
}

/// Loads a batch's column at the rows it keeps into the shape
/// `AggregateInput` describes, pointing at the column's own values when it
/// keeps every row and copying them otherwise.
#[derive(Default)]
pub(crate) struct AggregateInputLoader {
    /// The values, if copied.
    values: Option<Buffer>,
    /// The rows' validity, from bit 0.
    valid: Option<Buffer>,
}

impl AggregateInputLoader {
    /// Loads the first `rows` rows `select` keeps of `column`, a flat
    /// `Int64`, or the first `rows` of it if `select` is empty.
    pub(crate) fn load<'a>(
        &'a mut self,
        context: &mut Context,
        column: &'a ColumnView,
        select: &[u16],
        rows: usize,
    ) -> Result<AggregateInput<'a>, Error> {
        let AggregateInputLoader { values, valid } = self;
        let values = gathered(context, select, column.int64s(), values)?;
        *valid = validity(context, column, select, rows)?;
        Ok(AggregateInput { values, valid: valid.as_ref().map(Buffer::as_slice), rows })
    }
}

/// Where each group's state for one aggregate is: group g's record is in
/// block `g >> BLOCK_SHIFT`, `stride` words a record, and the state starts
/// `offset` words in. For a function that tracks seen, group g's byte in the
/// blocks `seen` starts is 1 once it has added a value, or `seen` is `None`
/// if every group has: see `SeenBytes`.
pub(crate) struct GroupStates<'a> {
    pub(crate) blocks: &'a [*mut i64],
    pub(crate) stride: usize,
    pub(crate) offset: usize,
    pub(crate) seen: Option<&'a [*mut u8]>,
}

impl GroupStates<'_> {
    /// Where `group`'s state starts. Callers pass only groups that exist.
    #[inline]
    fn state(&self, group: u32) -> *mut i64 {
        // SAFETY: the group exists, so its record does, with its state in it.
        unsafe { address(self.blocks, self.stride, group as usize).add(self.offset) }
    }

    /// The first word of `group`'s state.
    fn word(&self, group: u32) -> i64 {
        // SAFETY: as `state` promises.
        unsafe { *self.state(group) }
    }

    fn word_mut(&mut self, group: u32) -> &mut i64 {
        // SAFETY: as `state` promises.
        unsafe { &mut *self.state(group) }
    }

    /// Whether `group` has seen a value.
    fn seen(&self, group: u32) -> bool {
        // SAFETY: the group exists, so its byte does.
        self.seen.is_none_or(|seen| unsafe { *address(seen, 1, group as usize) } != 0)
    }

    /// Marks `group` seen if `valid`.
    fn mark(&mut self, group: u32, valid: bool) {
        if let Some(seen) = self.seen {
            // SAFETY: the group exists, so its byte does.
            unsafe { *address(seen, 1, group as usize) |= u8::from(valid) };
        }
    }
}

/// Keeps `GroupStates::seen` for one aggregate. A group exists only because
/// a row of it arrived, so while no input row has been null every group has
/// seen a value and nothing is kept. From the first null, a byte a group:
/// a byte rather than a bit, so neighbouring groups' updates don't wait on
/// each other.
pub(crate) struct SeenBytes {
    /// A byte a group, once tracking.
    bytes: Blocks<u8>,
    /// Whether an input row has been null.
    tracking: bool,
    /// How many groups there were when last asked.
    groups: usize,
}

impl SeenBytes {
    pub(crate) fn new(env: &Env) -> Result<SeenBytes, AllocError> {
        Ok(SeenBytes { bytes: Blocks::new(env, 1)?, tracking: false, groups: 0 })
    }

    /// What to pass as `GroupStates::seen` for a batch with `groups` groups
    /// in all, whose input holds nulls if `nulls`.
    pub(crate) fn get(
        &mut self,
        env: &Env,
        groups: usize,
        nulls: bool,
    ) -> Result<Option<&[*mut u8]>, AllocError> {
        if !self.tracking && !nulls {
            self.groups = groups;
            return Ok(None);
        }
        // New groups' bytes start zeroed.
        self.bytes.reserve(env, groups)?;
        if !self.tracking {
            // Every group so far added only values.
            self.tracking = true;
            for group in 0..self.groups {
                // SAFETY: room for the group was reserved above.
                unsafe { *self.bytes.record_ptr(group) = 1 };
            }
        }
        self.groups = groups;
        Ok(Some(self.bytes.starts()))
    }
}

/// An aggregate function, e.g. `SUM`. Each operator keeps a state for each
/// of its groups, starting zeroed, and the function says how many words it
/// needs and how to update and read it. Every method takes a whole batch, so
/// a call costs once a batch rather than once a row.
pub(crate) trait AggregateFunction {
    /// How many words of state each group keeps.
    fn state_words(&self) -> usize;
    /// Whether `update` reads an input column: `COUNT(*)` doesn't.
    fn reads_input(&self) -> bool;
    /// Whether the function needs `GroupStates::seen`.
    fn tracks_seen(&self) -> bool;
    /// Adds each row of `input` into the state of its group `groups[row]`.
    fn update(
        &self,
        input: &AggregateInput,
        groups: &[u32],
        states: GroupStates,
        overflow: &mut bool,
    );
    /// As `update`, with every row of `input` in group `group`, as without
    /// keys. A function can then fold the batch before touching the state.
    fn update_group(
        &self,
        input: &AggregateInput,
        group: u32,
        states: GroupStates,
        overflow: &mut bool,
    );
    /// Writes the result of each of `groups` to `values`, and to `valid`, a
    /// bit each, whether it holds one.
    fn finalize(&self, states: GroupStates, groups: &[u32], values: &mut [i64], valid: &mut [u8]);
}

/// The function that computes `function`.
pub(crate) fn make_aggregate_function(function: Function) -> &'static dyn AggregateFunction {
    match function {
        Function::CountStar => &CountStar,
        Function::Count => &Count,
        Function::Sum => &Sum,
        // Flips every bit but the sign: the least value is the greatest word.
        Function::Min => &Extreme { mask: !(1 << 63) },
        // Flips the sign: words order as the values do, the least being zero.
        Function::Max => &Extreme { mask: 1 << 63 },
        Function::Avg => &Avg,
    }
}

/// `function`'s result, of `data_type`, for each of `groups`, in buffers from
/// `context`.
pub(crate) fn finalize_to_buffer(
    context: &mut Context,
    function: &dyn AggregateFunction,
    data_type: DataType,
    states: GroupStates,
    groups: &[u32],
) -> Result<ColumnView, Error> {
    let rows = groups.len();
    let mut values = context.values_buffer(rows * 8)?;
    let mut valid = context.small_buffer(BitSlice::<u8>::words_for(rows))?;
    let bits = valid.as_mut_slice::<u8>();
    function.finalize(states, groups, values.as_mut_slice(), bits);
    let (whole, last) = (rows / 8, rows % 8);
    let all = at!(bits, ..whole).iter().all(|&byte| byte == u8::MAX)
        && (last == 0 || !*at!(bits, whole) & ((1 << last) - 1) == 0);
    Ok(ColumnView::new(context, data_type, values, (!all).then_some(valid))?)
}

/// Writes to `valid` a bit for each of `groups`, set if it's seen a value.
fn write_seen(valid: &mut [u8], groups: &[u32], states: &GroupStates) {
    if states.seen.is_none() {
        valid.fill(u8::MAX);
        return;
    }
    for (byte, groups) in valid.iter_mut().zip(groups.chunks(8)) {
        let bits = groups.iter().enumerate();
        *byte = bits.fold(0, |byte, (i, &group)| byte | u8::from(states.seen(group)) << i);
    }
}

/// Calls `add` with the first `words` words of the state of each row's
/// group, its value, and whether it holds one, and marks the group seen if it
/// does: the loop each function's `update` shares. Its bounds are checked
/// once, so the loop reads without checks.
#[inline]
fn scatter(
    input: &AggregateInput,
    groups: &[u32],
    states: GroupStates,
    words: usize,
    mut add: impl FnMut(&mut [i64], i64, bool),
) {
    let rows = input.rows;
    let mut states = states;
    let seen = states.seen.take();
    let values = if input.values.is_empty() { None } else { Some(input.values) };
    check!(groups.len() >= rows && values.is_none_or(|values| values.len() >= rows));
    check!(input.valid.is_none_or(|valid| valid.len() >= BitSlice::<u64>::words_for(rows)));
    for row in 0..rows {
        // SAFETY: the bounds are checked above, and each number is a group's,
        // whose state is in `states`, and whose byte is in `seen`.
        unsafe {
            let valid = input.valid.is_none_or(|valid| BitSlice::of(valid).get_unchecked(row));
            let group = *groups.get_unchecked(row);
            let value = values.map_or(0, |values| *values.get_unchecked(row));
            add(core::slice::from_raw_parts_mut(states.state(group), words), value, valid);
            if let Some(seen) = seen {
                *address(seen, 1, group as usize) |= u8::from(valid);
            }
        }
    }
}

/// The sum of `input`'s values that aren't null, exactly. The high and low
/// 32 bits of the values are summed apart, which can't overflow for a batch
/// and vectorizes, unlike checking each addition.
fn batch_sum(input: &AggregateInput) -> i128 {
    check!(input.rows <= BATCH_ROWS_MAX as usize);
    let values = at!(input.values, ..input.rows);
    let (mut high, mut low) = (0_i64, 0_u64);
    match input.valid {
        None => {
            for &value in values {
                high += value >> 32;
                low += value.cast_unsigned() & 0xffff_ffff;
            }
        }
        // A word of validity at a time, rather than a bit for each row.
        Some(valid) => {
            for (values, &valid) in values.chunks(64).zip(valid) {
                for (i, &value) in values.iter().enumerate() {
                    let value = value & mask(valid >> i & 1 != 0);
                    high += value >> 32;
                    low += value.cast_unsigned() & 0xffff_ffff;
                }
            }
        }
    }
    (i128::from(high) << 32) + i128::from(low)
}

/// A word that's 0 if not `valid`, and all ones if so.
fn mask(valid: bool) -> i64 {
    -i64::from(valid)
}

/// `COUNT(*)`: how many rows a group holds.
struct CountStar;

impl AggregateFunction for CountStar {
    fn state_words(&self) -> usize {
        1
    }

    fn reads_input(&self) -> bool {
        false
    }

    fn tracks_seen(&self) -> bool {
        false
    }

    fn update(&self, _: &AggregateInput, groups: &[u32], states: GroupStates, _: &mut bool) {
        for &group in groups {
            // SAFETY: each number is a group's, whose state exists.
            unsafe { *states.state(group) += 1 };
        }
    }

    fn update_group(
        &self,
        input: &AggregateInput,
        group: u32,
        mut states: GroupStates,
        _: &mut bool,
    ) {
        *states.word_mut(group) += i64::try_from(input.rows).unwrap_or(0);
    }

    fn finalize(&self, states: GroupStates, groups: &[u32], values: &mut [i64], valid: &mut [u8]) {
        for (value, &group) in values.iter_mut().zip(groups) {
            *value = states.word(group);
        }
        valid.fill(u8::MAX);
    }
}

/// `COUNT(column)`: how many of a group's rows hold a value. Reads as
/// `COUNT(*)` does.
struct Count;

impl AggregateFunction for Count {
    fn state_words(&self) -> usize {
        1
    }

    fn reads_input(&self) -> bool {
        true
    }

    fn tracks_seen(&self) -> bool {
        false
    }

    fn update(&self, input: &AggregateInput, groups: &[u32], states: GroupStates, _: &mut bool) {
        scatter(input, groups, states, 1, |count, _, valid| *at_mut!(count, 0) += i64::from(valid));
    }

    fn update_group(
        &self,
        input: &AggregateInput,
        group: u32,
        mut states: GroupStates,
        _: &mut bool,
    ) {
        *states.word_mut(group) += i64::try_from(input.count_valid()).unwrap_or(0);
    }

    fn finalize(&self, states: GroupStates, groups: &[u32], values: &mut [i64], valid: &mut [u8]) {
        CountStar.finalize(states, groups, values, valid);
    }
}

/// `SUM`: the total of a group's values, null if it holds none.
struct Sum;

impl AggregateFunction for Sum {
    fn state_words(&self) -> usize {
        1
    }

    fn reads_input(&self) -> bool {
        true
    }

    fn tracks_seen(&self) -> bool {
        true
    }

    fn update(
        &self,
        input: &AggregateInput,
        groups: &[u32],
        states: GroupStates,
        overflow: &mut bool,
    ) {
        let mut ovf = false;
        scatter(input, groups, states, 1, |total, value, valid| {
            // A null row's value is added too, masked out, which avoids a branch.
            let (sum, o) = at!(total, 0).overflowing_add(value & mask(valid));
            (*at_mut!(total, 0), ovf) = (sum, ovf | o);
        });
        *overflow |= ovf;
    }

    fn update_group(
        &self,
        input: &AggregateInput,
        group: u32,
        mut states: GroupStates,
        overflow: &mut bool,
    ) {
        let sum = i64::try_from(batch_sum(input));
        let (total, o) = states.word(group).overflowing_add(sum.unwrap_or(0));
        *states.word_mut(group) = total;
        *overflow |= sum.is_err() || o;
        states.mark(group, input.count_valid() != 0);
    }

    fn finalize(&self, states: GroupStates, groups: &[u32], values: &mut [i64], valid: &mut [u8]) {
        for (value, &group) in values.iter_mut().zip(groups) {
            *value = states.word(group);
        }
        write_seen(valid, groups, &states);
    }
}

/// `MIN` or `MAX`: a group's extreme value, null if it holds none. The state
/// is the value, its bits flipped by a mask chosen so that, as unsigned words, the
/// larger state is the value wanted and zero is below every value: a zeroed
/// state holds nothing, and states merge by unsigned maximum.
struct Extreme {
    mask: u64,
}

impl Extreme {
    /// `value` as a state, or zero if not `valid`.
    fn encode(&self, value: i64, valid: bool) -> u64 {
        (value.cast_unsigned() ^ self.mask) & mask(valid).cast_unsigned()
    }
}

impl AggregateFunction for Extreme {
    fn state_words(&self) -> usize {
        1
    }

    fn reads_input(&self) -> bool {
        true
    }

    fn tracks_seen(&self) -> bool {
        true
    }

    fn update(&self, input: &AggregateInput, groups: &[u32], states: GroupStates, _: &mut bool) {
        scatter(input, groups, states, 1, |state, value, valid| {
            let state = at_mut!(state, 0);
            *state = state.cast_unsigned().max(self.encode(value, valid)).cast_signed();
        });
    }

    fn update_group(
        &self,
        input: &AggregateInput,
        group: u32,
        mut states: GroupStates,
        _: &mut bool,
    ) {
        let mut best = states.word(group).cast_unsigned();
        for (row, &value) in at!(input.values, ..input.rows).iter().enumerate() {
            best = best.max(self.encode(value, input.is_valid(row)));
        }
        *states.word_mut(group) = best.cast_signed();
        states.mark(group, input.count_valid() != 0);
    }

    fn finalize(&self, states: GroupStates, groups: &[u32], values: &mut [i64], valid: &mut [u8]) {
        for (value, &group) in values.iter_mut().zip(groups) {
            *value = (states.word(group).cast_unsigned() ^ self.mask).cast_signed();
        }
        write_seen(valid, groups, &states);
    }
}

/// `MIN` or `MAX` of strings: a group's least or greatest, by their bytes,
/// null if it holds none. The state is the string's id in the run's pool, 0
/// for none, so it needs no seen bytes. While a batch is read, it's instead
/// `BATCH_ROW` and the row of the batch's best so far, and only each group's
/// best of the batch is interned, so strings beaten within it never are.
pub(crate) struct StringExtreme {
    /// Whether the greatest string is wanted, rather than the least.
    pub(crate) max: bool,
}

/// Marks a `StringExtreme` state that holds a row of the batch being read.
const BATCH_ROW: i64 = 1 << 32;

impl StringExtreme {
    /// How many words of state each group keeps.
    pub(crate) const STATE_WORDS: usize = 1;

    /// Adds the first `rows` rows that `select` keeps of `column`, all if
    /// it's empty, into the state of their group, `groups[row]`, or of
    /// group 0 if `None`.
    pub(crate) fn update(
        &self,
        context: &mut Context,
        column: &ColumnView,
        select: &[u16],
        rows: usize,
        groups: Option<&[u32]>,
        mut states: GroupStates,
    ) -> Result<(), Error> {
        check!(column.data_type() == DataType::String);
        let values = column.values();
        let strings = values.strings();
        let form = column.form();
        // The string at batch row `row`, or `None` if it's null.
        let string = |row: usize| {
            let value = match form {
                Form::Flat => row,
                Form::Dictionary(indices) => *at!(indices, row) as usize,
                Form::Constant | Form::Sequence(_) => failed!(),
            };
            #[expect(clippy::cast_possible_truncation, reason = "a batch's value")]
            (!values.is_null(value as u32)).then(|| strings.get(value))
        };
        let wanted = if self.max { Ordering::Greater } else { Ordering::Less };
        let mut touched = context.indices_buffer(rows * 4)?;
        let touched = touched.as_mut_slice::<u32>();
        let mut count = 0;
        let env = context.env();
        let pool = context.string_pool()?;
        for i in 0..rows {
            let row = if select.is_empty() { i } else { usize::from(*at!(select, i)) };
            let Some(candidate) = string(row) else { continue };
            let group = groups.map_or(0, |groups| *at!(groups, i));
            let state = *states.word_mut(group);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a row or an id"
            )]
            let best = if state & BATCH_ROW != 0 {
                string((state & !BATCH_ROW) as usize)
            } else {
                pool.get(Id::raw(state as u32))
            };
            let better = best.is_none_or(|best| candidate.cmp(best) == wanted);
            if better {
                if state & BATCH_ROW == 0 {
                    *at_mut!(touched, count) = group;
                    count += 1;
                }
                #[expect(clippy::cast_possible_wrap, reason = "a batch's row")]
                let row = row as i64;
                *states.word_mut(group) = BATCH_ROW | row;
            }
        }
        for &group in at!(touched, ..count) {
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a row")]
            let row = (states.word(group) & !BATCH_ROW) as usize;
            *states.word_mut(group) = i64::from(pool.intern_string(env, string(row))?.raw_id());
        }
        Ok(())
    }

    /// The string of each of `groups`, as a string column.
    pub(crate) fn finalize(
        context: &mut Context,
        states: &GroupStates,
        groups: &[u32],
    ) -> Result<ColumnView, Error> {
        let mut ids = context.values_buffer(groups.len() * 8)?;
        for (id, &group) in ids.as_mut_slice::<i64>().iter_mut().zip(groups) {
            *id = states.word(group);
        }
        write_strings(context, at!(ids.as_slice::<i64>(), ..groups.len()))
    }
}

/// `AVG`: the mean of a group's values, a float, null if it holds none. The
/// state is the sum, in two words, low then high, so it can't overflow, then
/// how many values there are: a mean fits where the sum it's of may not, as
/// with `AVG(UserID)` over ClickBench's 100M rows.
struct Avg;

/// Where the count is in an `Avg` state.
const AVG_COUNT: usize = 2;

/// The two words of a sum, low then high, as one.
fn int128(state: &[i64]) -> i128 {
    i128::from(*at!(state, 1)) << 64 | i128::from(at!(state, 0).cast_unsigned())
}

/// Writes `sum` to its two words.
fn set_int128(state: &mut [i64], sum: i128) {
    #[expect(clippy::cast_possible_truncation, reason = "split in two")]
    at_mut!(state, ..2).copy_from_slice(&[sum as i64, (sum >> 64) as i64]);
}

impl Avg {
    /// Group `group`'s state.
    fn state<'s>(states: &'s GroupStates, group: u32) -> &'s [i64] {
        // SAFETY: the group exists, and an `Avg` state is `AVG_COUNT + 1` words.
        unsafe { core::slice::from_raw_parts(states.state(group), AVG_COUNT + 1) }
    }
}

impl AggregateFunction for Avg {
    fn state_words(&self) -> usize {
        AVG_COUNT + 1
    }

    fn reads_input(&self) -> bool {
        true
    }

    fn tracks_seen(&self) -> bool {
        false
    }

    fn update(&self, input: &AggregateInput, groups: &[u32], states: GroupStates, _: &mut bool) {
        scatter(input, groups, states, AVG_COUNT + 1, |state, value, valid| {
            set_int128(state, int128(state) + i128::from(value & mask(valid)));
            *at_mut!(state, AVG_COUNT) += i64::from(valid);
        });
    }

    fn update_group(&self, input: &AggregateInput, group: u32, states: GroupStates, _: &mut bool) {
        // SAFETY: the group exists, and an `Avg` state is `AVG_COUNT + 1` words.
        let state = unsafe { core::slice::from_raw_parts_mut(states.state(group), AVG_COUNT + 1) };
        set_int128(state, int128(state) + batch_sum(input));
        *at_mut!(state, AVG_COUNT) += i64::try_from(input.count_valid()).unwrap_or(0);
    }

    fn finalize(&self, states: GroupStates, groups: &[u32], values: &mut [i64], valid: &mut [u8]) {
        for (value, &group) in values.iter_mut().zip(groups) {
            let state = Avg::state(&states, group);
            let (low, high) = (*at!(state, 0), *at!(state, 1));
            // i128 to f64 is a slow library call, and few sums need it.
            #[expect(clippy::cast_precision_loss, reason = "means are approximate")]
            let sum = if high == low >> 63 { low as f64 } else { int128(state) as f64 };
            #[expect(clippy::cast_precision_loss, reason = "means are approximate")]
            let mean = sum / *at!(state, AVG_COUNT) as f64;
            *value = mean.to_bits().cast_signed();
        }
        for (byte, groups) in valid.iter_mut().zip(groups.chunks(8)) {
            let held = |group| *at!(Avg::state(&states, group), AVG_COUNT) != 0;
            *byte =
                (0..).zip(groups).fold(0, |byte, (i, &group)| byte | u8::from(held(group)) << i);
        }
    }
}

/// Whether `row` isn't null by `validity`.
pub(crate) fn is_valid(validity: Validity, row: usize) -> bool {
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
    let mut words = context.small_buffer(BitSlice::<u64>::words_for(rows) * 8)?;
    words.as_mut_slice::<u64>().fill(0);
    let out = BitSlice::of_mut(words.as_mut_slice::<u64>());
    for i in 0..rows {
        let row = if select.is_empty() { i } else { usize::from(*at!(select, i)) };
        out.set_if(i, is_valid(validity, row));
    }
    Ok(Some(words))
}

/// The words of `column` at the rows `select` keeps: all of them if it's
/// empty, or else a copy, in `copy`.
pub(crate) fn gathered<'a>(
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
