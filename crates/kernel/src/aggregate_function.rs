//! `AggregateFunction`: what an aggregate function, such as `SUM`, keeps for
//! each group, and how it updates and outputs it, a batch at a time, as
//! Perfetto's do, for the operators that aggregate.

use crate::aggregate::Function;
use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::column::{Bounds, ColumnView, DataType, Validity};
use crate::context::Context;
use crate::error::Error;
use crate::group_table::GROUPS_MAX;
use crate::row_batch::BATCH_ROWS_MAX;
use crate::slow_vec::SlowVec;

/// What an aggregate reads of one batch: a value a row, contiguous, and
/// which rows hold one.
#[derive(Clone, Copy)]
pub(crate) struct AggregateInput<'a> {
    /// A value a row, an integer or a float's bits: none for `COUNT(*)`.
    pub(crate) values: &'a [i64],
    /// Bit r set if row r holds a value, or `None` if every row does.
    pub(crate) valid: Option<&'a [u64]>,
    /// How many rows there are.
    pub(crate) rows: usize,
    /// The least and greatest value, if known.
    pub(crate) bounds: Option<Bounds>,
}

impl AggregateInput<'_> {
    /// How many rows hold a value.
    fn count_valid(&self) -> usize {
        let Some(valid) = self.valid else { return self.rows };
        let words = at!(valid, ..self.rows.div_ceil(64)).iter().enumerate();
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
    /// `Int64` or `Float64`, or the first `rows` of it if `select` is empty.
    pub(crate) fn load<'a>(
        &'a mut self,
        context: &mut Context,
        column: &'a ColumnView,
        select: &[u16],
        rows: usize,
    ) -> Result<AggregateInput<'a>, Error> {
        let AggregateInputLoader { values, valid } = self;
        let values = gathered(context, select, words(column), values)?;
        *valid = validity(context, column, select, rows)?;
        let valid = valid.as_ref().map(Buffer::as_slice);
        Ok(AggregateInput { values, valid, rows, bounds: column.bounds() })
    }
}

/// Where each group's state for one aggregate is: group g's words start at
/// `g * stride` of `words`. For a function that tracks seen, byte g of
/// `seen` is 1 once group g has added a value, or `seen` is `None` if every
/// group has: see `SeenBytes`.
pub(crate) struct GroupStates<'a> {
    pub(crate) words: &'a mut [i64],
    pub(crate) stride: usize,
    pub(crate) seen: Option<&'a mut [u8]>,
}

impl GroupStates<'_> {
    /// The first word of `group`'s state.
    fn word(&self, group: u32) -> i64 {
        *at!(self.words, group as usize * self.stride)
    }

    fn word_mut(&mut self, group: u32) -> &mut i64 {
        at_mut!(self.words, group as usize * self.stride)
    }

    /// Whether `group` has seen a value.
    fn seen(&self, group: u32) -> bool {
        self.seen.as_ref().is_none_or(|seen| *at!(seen, group as usize) != 0)
    }

    /// Marks `group` seen if `valid`.
    fn mark(&mut self, group: u32, valid: bool) {
        if let Some(seen) = &mut self.seen {
            *at_mut!(seen, group as usize) |= u8::from(valid);
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
    bytes: SlowVec<u8>,
    /// Whether an input row has been null.
    tracking: bool,
    /// How many groups there were when last asked.
    groups: usize,
}

impl SeenBytes {
    pub(crate) fn new(allocator: &dyn Allocator) -> Result<SeenBytes, AllocError> {
        Ok(SeenBytes { bytes: SlowVec::new(allocator, GROUPS_MAX)?, tracking: false, groups: 0 })
    }

    /// What to pass as `GroupStates::seen` for a batch with `groups` groups
    /// in all, whose input holds nulls if `nulls`.
    pub(crate) fn get(
        &mut self,
        groups: usize,
        nulls: bool,
    ) -> Result<Option<&mut [u8]>, AllocError> {
        if !self.tracking && !nulls {
            self.groups = groups;
            return Ok(None);
        }
        if !self.tracking {
            // Every group so far added only values.
            self.tracking = true;
            let ones = [1; 64];
            let mut left = self.groups;
            while left > 0 {
                let n = left.min(ones.len());
                self.bytes.extend_from_slice(at!(ones, ..n))?;
                left -= n;
            }
        }
        let new = groups - self.bytes.len();
        self.bytes.extend_zeroed(new)?;
        self.groups = groups;
        Ok(Some(&mut self.bytes))
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
    /// As `update_group` for each run of rows of one group: the rows from
    /// each of `starts` to the next, of the group `groups` says. `input` has
    /// no nulls.
    fn update_runs(
        &self,
        input: &AggregateInput,
        groups: &[u32],
        starts: &[u32],
        mut states: GroupStates,
        overflow: &mut bool,
    ) {
        for run in starts.windows(2) {
            let (start, end) = (*at!(run, 0) as usize, *at!(run, 1) as usize);
            let values = input.values.get(start..end).unwrap_or_default();
            let rows = end - start;
            let run_input = AggregateInput { values, valid: None, rows, bounds: input.bounds };
            let states = GroupStates {
                words: &mut *states.words,
                stride: states.stride,
                seen: states.seen.as_deref_mut(),
            };
            self.update_group(&run_input, *at!(groups, start), states, overflow);
        }
    }
    /// Writes the result of each of `groups` to `values`, and to `valid`, a
    /// bit each, whether it holds one.
    fn finalize(&self, states: GroupStates, groups: &[u32], values: &mut [i64], valid: &mut [u8]);
}

/// The function that computes `function` of a column of `data_type`.
pub(crate) fn make_aggregate_function(
    function: Function,
    data_type: DataType,
) -> &'static dyn AggregateFunction {
    // Flips the sign: words order as the values do, the least being zero.
    const MAX: u64 = 1 << 63;
    // Flips every bit but the sign: the least value is the greatest word.
    const MIN: u64 = !MAX;
    let float = data_type == DataType::Float64;
    match function {
        Function::CountStar => &CountStar,
        Function::Count => &Count,
        Function::Sum if float => &FloatSum,
        Function::Sum => &Sum,
        Function::Min if float => &Extreme { mask: MIN, float: MIN },
        Function::Min => &Extreme { mask: MIN, float: 0 },
        Function::Max if float => &Extreme { mask: MAX, float: MIN },
        Function::Max => &Extreme { mask: MAX, float: 0 },
        Function::Avg if float => &Avg { float: true },
        Function::Avg => &Avg { float: false },
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
    let mut valid = context.small_buffer(rows.div_ceil(8))?;
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
    let GroupStates { words: states, stride, mut seen } = states;
    let values = if input.values.is_empty() { None } else { Some(input.values) };
    check!(groups.len() >= rows && values.is_none_or(|values| values.len() >= rows));
    check!(input.valid.is_none_or(|valid| valid.len() >= rows.div_ceil(64)));
    if cfg!(debug_assertions) {
        let groups = at!(groups, ..rows).iter();
        let room = |group: usize| {
            group * stride + words <= states.len()
                && seen.as_ref().is_none_or(|seen| group < seen.len())
        };
        check!(groups.copied().all(|group| room(group as usize)));
    }
    let base = states.as_mut_ptr();
    for row in 0..rows {
        // SAFETY: the bounds are checked above, and each number is a group's,
        // whose state is in `states`, and whose byte is in `seen`.
        unsafe {
            let valid = input
                .valid
                .is_none_or(|valid| *valid.get_unchecked(row / 64) >> (row % 64) & 1 != 0);
            let group = *groups.get_unchecked(row) as usize;
            let value = values.map_or(0, |values| *values.get_unchecked(row));
            add(core::slice::from_raw_parts_mut(base.add(group * stride), words), value, valid);
            if let Some(seen) = &mut seen {
                *seen.get_unchecked_mut(group) |= u8::from(valid);
            }
        }
    }
}

/// Whether `rows` values within `bounds`, if known, can't sum past an
/// `Int64`.
fn fits(bounds: Option<Bounds>, rows: usize) -> bool {
    bounds.is_some_and(|bounds| {
        let largest = bounds.min.unsigned_abs().max(bounds.max.unsigned_abs());
        let reach = largest.checked_mul(rows as u64);
        reach.is_some_and(|reach| reach <= i64::MAX.cast_unsigned())
    })
}

/// The sum of `input`'s values that aren't null, exactly. The high and low
/// 32 bits of the values are summed apart, which can't overflow for a batch
/// and vectorizes, unlike checking each addition.
fn batch_sum(input: &AggregateInput) -> i128 {
    check!(input.rows <= BATCH_ROWS_MAX as usize);
    let values = at!(input.values, ..input.rows);
    // Where the bounds say the sum fits in one word, it's summed wrapping,
    // which vectorizes better still, nulls added as zeros.
    let fits = input.bounds.is_some_and(|bounds| {
        let largest = bounds.min.unsigned_abs().max(bounds.max.unsigned_abs());
        largest
            .checked_mul(values.len() as u64)
            .is_some_and(|reach| reach <= i64::MAX.cast_unsigned())
    });
    if fits {
        let mut sum = 0_i64;
        each(input, |value, valid| sum = sum.wrapping_add(value & mask(valid)));
        return i128::from(sum);
    }
    let (mut high, mut low) = (0_i64, 0_u64);
    each(input, |value, valid| {
        let value = value & mask(valid);
        high += value >> 32;
        low += value.cast_unsigned() & 0xffff_ffff;
    });
    (i128::from(high) << 32) + i128::from(low)
}

/// Calls `f` with each of `input`'s values and whether it holds one, a word
/// of validity at a time rather than a bit for each row, so the loop
/// vectorizes.
#[inline]
fn each(input: &AggregateInput, mut f: impl FnMut(i64, bool)) {
    let values = at!(input.values, ..input.rows);
    match input.valid {
        None => values.iter().for_each(|&value| f(value, true)),
        Some(valid) => {
            for (values, &valid) in values.chunks(64).zip(valid) {
                for (i, &value) in values.iter().enumerate() {
                    f(value, valid >> i & 1 != 0);
                }
            }
        }
    }
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
        let GroupStates { words, stride, .. } = states;
        if cfg!(debug_assertions) {
            check!(groups.iter().all(|&group| group as usize * stride < words.len()));
        }
        let words = words.as_mut_ptr();
        for &group in groups {
            // SAFETY: each number is a group's, whose state is in `words`.
            unsafe { *words.add(group as usize * stride) += 1 };
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

    fn update_runs(
        &self,
        _: &AggregateInput,
        groups: &[u32],
        starts: &[u32],
        mut states: GroupStates,
        _: &mut bool,
    ) {
        for run in starts.windows(2) {
            let (start, end) = (*at!(run, 0), *at!(run, 1));
            *states.word_mut(*at!(groups, start as usize)) += i64::from(end - start);
        }
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

    fn update_runs(
        &self,
        input: &AggregateInput,
        groups: &[u32],
        starts: &[u32],
        mut states: GroupStates,
        overflow: &mut bool,
    ) {
        // Each run is summed wrapping where the bounds say no run can
        // overflow, and checking each addition otherwise.
        let fits = fits(input.bounds, input.rows);
        let mut ovf = false;
        for run in starts.windows(2) {
            let (start, end) = (*at!(run, 0) as usize, *at!(run, 1) as usize);
            let values = at!(input.values, start..end).iter();
            let (sum, o) = if fits {
                (values.fold(0_i64, |sum, &value| sum.wrapping_add(value)), false)
            } else {
                values.fold((0_i64, false), |(sum, ovf), &value| {
                    let (sum, o) = sum.overflowing_add(value);
                    (sum, ovf | o)
                })
            };
            let group = *at!(groups, start);
            let (total, o2) = states.word(group).overflowing_add(sum);
            *states.word_mut(group) = total;
            states.mark(group, true);
            ovf |= o | o2;
        }
        *overflow |= ovf;
    }

    fn finalize(&self, states: GroupStates, groups: &[u32], values: &mut [i64], valid: &mut [u8]) {
        for (value, &group) in values.iter_mut().zip(groups) {
            *value = states.word(group);
        }
        write_seen(valid, groups, &states);
    }
}

/// `MIN` or `MAX`: a group's extreme value, null if it holds none. The state
/// is the value, its bits flipped by a mask chosen so that, as unsigned words,
/// the larger state is the value wanted and zero is below every value: a
/// zeroed state holds nothing, and states merge by unsigned maximum. A
/// negative float's bits but the sign are flipped first, so floats order as
/// integers do, and a NaN is above every number.
struct Extreme {
    /// What a value's bits are flipped by, once they order as integers do.
    mask: u64,
    /// The bits flipped of a negative value first: all but the sign for
    /// floats, none for integers.
    float: u64,
}

impl Extreme {
    /// `value`, an integer or a float's bits, as a state, or zero if not
    /// `valid`.
    fn encode(&self, value: i64, valid: bool) -> u64 {
        let bits = value.cast_unsigned() ^ ((value >> 63).cast_unsigned() & self.float);
        (bits ^ self.mask) & mask(valid).cast_unsigned()
    }

    fn decode(&self, state: i64) -> i64 {
        // The sign is as it was, so says which bits were flipped.
        let bits = state ^ self.mask.cast_signed();
        bits ^ ((bits >> 63).cast_unsigned() & self.float).cast_signed()
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
        each(input, |value, valid| best = best.max(self.encode(value, valid)));
        *states.word_mut(group) = best.cast_signed();
        states.mark(group, input.count_valid() != 0);
    }

    fn finalize(&self, states: GroupStates, groups: &[u32], values: &mut [i64], valid: &mut [u8]) {
        for (value, &group) in values.iter_mut().zip(groups) {
            *value = self.decode(states.word(group));
        }
        write_seen(valid, groups, &states);
    }
}

fn float(word: i64) -> f64 {
    f64::from_bits(word.cast_unsigned())
}

fn word(value: f64) -> i64 {
    value.to_bits().cast_signed()
}

/// `SUM` of floats: the total of a group's values, null if it holds none.
struct FloatSum;

impl AggregateFunction for FloatSum {
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
        scatter(input, groups, states, 1, |total, value, valid| {
            *at_mut!(total, 0) = word(float(*at!(total, 0)) + float(value & mask(valid)));
        });
    }

    fn update_group(
        &self,
        input: &AggregateInput,
        group: u32,
        mut states: GroupStates,
        _: &mut bool,
    ) {
        let mut total = float(states.word(group));
        each(input, |value, valid| total += float(value & mask(valid)));
        *states.word_mut(group) = word(total);
        states.mark(group, input.count_valid() != 0);
    }

    fn finalize(&self, states: GroupStates, groups: &[u32], values: &mut [i64], valid: &mut [u8]) {
        Sum.finalize(states, groups, values, valid);
    }
}

/// `AVG`: the mean of a group's values, a float, null if it holds none. The
/// state is the sum, then how many values there are. The sum of integers
/// takes two words, low then high, so it can't overflow: a mean fits where
/// the sum it's of may not.
struct Avg {
    /// Whether the values are floats.
    float: bool,
}

impl Avg {
    /// Where the count is in a state.
    fn count_at(&self) -> usize {
        if self.float { 1 } else { 2 }
    }

    /// Adds `sum` and `count` to `group`'s state: the sum of floats' bits,
    /// or of integers.
    fn add(&self, states: &mut GroupStates, group: u32, sum: i128, count: usize) {
        let start = group as usize * states.stride;
        let state = at_mut!(states.words, start..start + self.state_words());
        if self.float {
            #[expect(clippy::cast_possible_truncation, reason = "a float's bits")]
            let added = float(sum as i64);
            *at_mut!(state, 0) = word(float(*at!(state, 0)) + added);
        } else {
            set_int128(state, int128(state) + sum);
        }
        *at_mut!(state, self.count_at()) += i64::try_from(count).unwrap_or(0);
    }
}

/// The two words of an integer sum, low then high, as one.
fn int128(state: &[i64]) -> i128 {
    i128::from(*at!(state, 1)) << 64 | i128::from(at!(state, 0).cast_unsigned())
}

/// Writes `sum` to its two words.
fn set_int128(state: &mut [i64], sum: i128) {
    #[expect(clippy::cast_possible_truncation, reason = "split in two")]
    at_mut!(state, ..2).copy_from_slice(&[sum as i64, (sum >> 64) as i64]);
}

impl AggregateFunction for Avg {
    fn state_words(&self) -> usize {
        self.count_at() + 1
    }

    fn reads_input(&self) -> bool {
        true
    }

    fn tracks_seen(&self) -> bool {
        false
    }

    fn update(&self, input: &AggregateInput, groups: &[u32], states: GroupStates, _: &mut bool) {
        let at = self.count_at();
        scatter(input, groups, states, self.state_words(), |state, value, valid| {
            let value = value & mask(valid);
            if self.float {
                *at_mut!(state, 0) = word(float(*at!(state, 0)) + float(value));
            } else {
                set_int128(state, int128(state) + i128::from(value));
            }
            *at_mut!(state, at) += i64::from(valid);
        });
    }

    fn update_group(
        &self,
        input: &AggregateInput,
        group: u32,
        mut states: GroupStates,
        _: &mut bool,
    ) {
        let sum = if self.float {
            let mut total = 0.0;
            each(input, |value, valid| total += float(value & mask(valid)));
            i128::from(word(total))
        } else {
            batch_sum(input)
        };
        self.add(&mut states, group, sum, input.count_valid());
    }

    fn finalize(&self, states: GroupStates, groups: &[u32], values: &mut [i64], valid: &mut [u8]) {
        let (at, words) = (self.count_at(), self.state_words());
        let state = |group: u32| {
            let start = group as usize * states.stride;
            at!(states.words, start..start + words)
        };
        for (value, &group) in values.iter_mut().zip(groups) {
            let state = state(group);
            #[expect(clippy::cast_precision_loss, reason = "means are approximate")]
            let sum = if self.float { float(*at!(state, 0)) } else { int128(state) as f64 };
            #[expect(clippy::cast_precision_loss, reason = "means are approximate")]
            let count = *at!(state, at) as f64;
            *value = word(sum / count);
        }
        for (byte, groups) in valid.iter_mut().zip(groups.chunks(8)) {
            let bits = groups.iter().enumerate();
            *byte =
                bits.fold(0, |byte, (i, &group)| byte | u8::from(*at!(state(group), at) != 0) << i);
        }
    }
}

/// A flat column's values as words: integers, or floats' bits.
fn words(column: &ColumnView) -> &[i64] {
    // SAFETY: any bits are an `i64`.
    let (before, words, after) = unsafe { column.value_bytes().align_to::<i64>() };
    check!(before.is_empty() && after.is_empty());
    words
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
    let mut words = context.small_buffer(rows.div_ceil(64) * 8)?;
    let out = words.as_mut_slice::<u64>();
    if select.is_empty() {
        check!(rows == validity.row_count() as usize);
        validity.copy_to(out);
        return Ok(Some(words));
    }
    check!(select.len() >= rows);
    // Rows kept are in increasing order, so the last is the greatest.
    check!(
        select.get(rows.wrapping_sub(1)).is_none_or(|&row| u32::from(row) < validity.row_count())
    );
    for (word, select) in out.iter_mut().zip(at!(select, ..rows).chunks(64)) {
        let bits = select.iter().enumerate();
        *word = bits.fold(0, |word, (i, &row)| {
            // SAFETY: every row kept is below the greatest, checked above.
            let valid = unsafe { validity.is_valid_unchecked(u32::from(row)) };
            word | u64::from(valid) << i
        });
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
