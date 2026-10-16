//! `AggregateFunction`: what an aggregate function, such as `SUM`, keeps for
//! each group, and how it updates and outputs it, a batch at a time, as
//! Perfetto's do, for the operators that aggregate.

use crate::aggregate::Function;
use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType, Validity};
use crate::context::Context;
use crate::error::Error;
use crate::group_table::GROUPS_MAX;
use crate::row_batch::BATCH_ROWS_MAX;
use crate::slow_vec::SlowVec;

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
    fn is_valid(&self, row: usize) -> bool {
        self.valid.is_none_or(|valid| *at!(valid, row / 64) >> (row % 64) & 1 != 0)
    }

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
    }
}

/// `function`'s result for each of `groups`, in buffers from `context`.
pub(crate) fn finalize_to_buffer(
    context: &mut Context,
    function: &dyn AggregateFunction,
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
    Ok(ColumnView::new(context, DataType::Int64, values, (!all).then_some(valid))?)
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
        // The high and low 32 bits of the values are summed apart, which can't
        // overflow for a batch and vectorizes, unlike checking each addition.
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
        // The sum is high * 2^32 + low, which fits in 64 bits if, once low's
        // carry is in high, high fits in 32.
        high += (low >> 32).cast_signed();
        let fits = i32::try_from(high).is_ok();
        let sum = (high.cast_unsigned() << 32 | low & 0xffff_ffff).cast_signed();
        let (total, o) = states.word(group).overflowing_add(sum);
        *states.word_mut(group) = total;
        *overflow |= !fits || o;
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
    out.fill(0);
    for i in 0..rows {
        let row = if select.is_empty() { i } else { usize::from(*at!(select, i)) };
        *at_mut!(out, i / 64) |= u64::from(is_valid(validity, row)) << (i % 64);
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
