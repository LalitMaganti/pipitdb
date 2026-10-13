//! `AggregateOp`: aggregates, such as `COUNT(*)` and `SUM(x)`, over each
//! group of its child's rows with the same keys, or over all of them if it
//! has none; and `hash_aggregate`, the breaker that runs it.

use crate::allocator::{AllocError, Allocator};
use crate::boxed::Box;
use crate::column::{ColumnView, DataType, Form, Forms, Values};
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

    /// Keys in any form but lazy: it groups by their codes where it can,
    /// looking each up once. What it aggregates, flat.
    fn accepts(&self, column: ColumnId) -> Forms {
        if self.aggregates.iter().any(|aggregate| aggregate.input == Some(column)) {
            Forms::FLAT
        } else {
            Forms::FLAT | Forms::CONSTANT | Forms::DICTIONARY
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

/// The most groups an aggregation can hold.
const GROUPS_MAX: usize = 1 << 26;

/// The most codes a batch's keys can have together to be grouped by their
/// codes directly, without hashing: each takes 4 bytes, cleared whenever
/// the keys' spaces change.
const DIRECT_MAX: u64 = 1 << 16;

/// The most codes a key's range can have: its values between its bounds,
/// and null.
const RANGE_MAX: u64 = 1 << 16;

/// Groups rows by what's known of their keys. Where each key's values in a
/// batch are codes into a space, as a dictionary's indices, a constant's one
/// value, or a range its bounds keep narrow, a row's codes together are one,
/// whose group is looked up once while batches share the spaces: directly,
/// if there are few, or else by hashing that one code. Otherwise each row's
/// keys are hashed. Either way, it then adds each aggregate's column to the
/// rows' groups a column at a time.
struct HashAggregate {
    /// What it groups by and aggregates.
    grouping: Grouping,
    /// A new group's totals.
    empty: SlowVec<Total>,
}

/// The groups so far.
struct Groups {
    /// How many groups there are.
    len: usize,
    /// Each group's keys, one after another, each group's after a word of
    /// which are null, a bit each.
    keys: SlowVec<i64>,
    /// Each group's totals, one for each aggregate.
    totals: SlowVec<Total>,
    /// A power of two many, at most half full: each the index of a group
    /// plus one, or 0 if empty.
    slots: SlowVec<u32>,
    /// Each key's space of codes, as the last batch grouped by code had it.
    spaces: SlowVec<Space>,
    /// If those spaces together have few codes, each one's group plus one,
    /// or 0 if not looked up yet; else empty.
    direct: SlowVec<u32>,
    /// If not, the codes looked up, hashed to a slot of two words each: the
    /// code, and its group plus one, or 0 if the slot is empty. A power of
    /// two many slots, at most half full.
    hashed: SlowVec<u64>,
    /// How many slots of `hashed` are full.
    coded: usize,
    /// Room to work on a batch in, taken while one is.
    scratch: Option<Box<Scratch>>,
    /// How many groups have been output.
    served: usize,
}

/// Room to work on a batch in, kept from batch to batch so it isn't
/// cleared for each.
struct Scratch {
    /// 0, 1, 2 and so on: a batch's rows, if all are kept.
    all: [u16; ROWS],
    /// What finding the rows' groups works in.
    work: Work,
}

/// Room to find a batch's rows' groups in.
struct Work {
    /// Each row's group.
    ids: Ids,
    /// Each row's keys' codes, as one.
    codes: [u64; ROWS],
    /// A group's keys, after a word of which are null.
    keys: [i64; KEYS_MAX + 1],
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

impl Total {
    /// The total of no values for `function`: what its first value replaces,
    /// as `f64::min` takes the other of a NaN. A greatest is kept as the
    /// least, with the order reversed.
    fn empty(function: Function) -> Total {
        match function {
            Function::Min | Function::Max => Total { count: 0, int: i128::MAX, float: f64::NAN },
            _ => Total::default(),
        }
    }

    /// Adds `other`, a total of other values for `function`.
    fn add(&mut self, function: Function, other: &Total) {
        if other.count == 0 {
            return;
        }
        (self.int, self.float) = match function {
            Function::Min | Function::Max => (self.int.min(other.int), self.float.min(other.float)),
            _ => (self.int + other.int, self.float + other.float),
        };
        self.count += other.count;
    }
}

/// The codes of a key's values in a batch.
#[derive(Clone)]
enum Space {
    /// Indices into the values of a dictionary, a view of them.
    Dictionary(ColumnView),
    /// Code `i` is `min + i`, and the last of `len` is null.
    Range { min: i64, len: u32 },
    /// One code, of a value or null.
    Constant(Option<i64>),
}

impl Space {
    /// The space of `column`'s codes, if it has one small enough.
    fn of(column: &ColumnView) -> Option<Space> {
        match column.form() {
            Form::Dictionary(_) => Some(Space::Dictionary(column.clone())),
            Form::Constant => Some(Space::Constant(value(column.values(), 0))),
            Form::Flat => {
                let bounds = column.bounds()?;
                let len = bounds.max.abs_diff(bounds.min).checked_add(2)?;
                let len = u32::try_from(len).ok().filter(|&len| u64::from(len) <= RANGE_MAX)?;
                Some(Space::Range { min: bounds.min, len })
            }
        }
    }

    fn len(&self) -> u32 {
        match self {
            Space::Dictionary(column) => column.values().len(),
            Space::Range { len, .. } => *len,
            Space::Constant(_) => 1,
        }
    }

    /// Whether codes in this and `other` mean the same values.
    fn same(&self, other: &Space) -> bool {
        match (self, other) {
            (Space::Dictionary(mine), Space::Dictionary(theirs)) => mine.shares_values(theirs),
            (Space::Range { min, len }, Space::Range { min: other_min, len: other_len }) => {
                (min, len) == (other_min, other_len)
            }
            (Space::Constant(mine), Space::Constant(theirs)) => mine == theirs,
            _ => false,
        }
    }

    /// The value of `code`, or `None` if it's null.
    fn value(&self, code: u32) -> Option<i64> {
        match self {
            Space::Dictionary(column) => value(column.values(), code),
            Space::Range { min, len } => (code + 1 < *len).then(|| min.wrapping_add(code.into())),
            Space::Constant(value) => *value,
        }
    }
}

/// A batch's group for each row.
type Ids = [u32; ROWS];

/// Rows are below this, so masking a row by it keeps it in a batch's
/// arrays without a check.
const ROWS: usize = BATCH_ROWS_MAX as usize;

/// `column`'s value in `row`, or `None` if it's null.
fn key_of(column: &ColumnView, row: u16) -> Option<i64> {
    let index = match column.form() {
        Form::Flat => row.into(),
        Form::Constant => 0,
        Form::Dictionary(indices) => *at!(indices, usize::from(row)),
    };
    value(column.values(), index)
}

/// Value `i` of `values`, or `None` if it's null.
fn value(values: Values, i: u32) -> Option<i64> {
    (!values.is_null(i)).then(|| *at!(values.int64s(), i as usize))
}

impl HashAggregate {
    fn new(allocator: &dyn Allocator, grouping: Grouping) -> Result<HashAggregate, AllocError> {
        check!(grouping.keys.len() <= KEYS_MAX);
        let empty = grouping.aggregates.iter().map(|aggregate| Total::empty(aggregate.function));
        let empty = SlowVec::fixed_from(allocator, empty)?;
        Ok(HashAggregate { grouping, empty })
    }

    /// The group with `keys`, after a word of which are null, added if new.
    fn group(
        &self,
        allocator: &dyn Allocator,
        groups: &mut Groups,
        keys: &[i64],
    ) -> Result<u32, Error> {
        let mask = groups.slots.len() - 1;
        #[expect(clippy::cast_possible_truncation, reason = "masked")]
        let mut slot = hash(keys) as usize & mask;
        loop {
            let group = *at!(groups.slots, slot);
            if group == 0 {
                break;
            }
            let found = group as usize - 1;
            if at!(groups.keys, found * keys.len()..(found + 1) * keys.len()) == keys {
                return Ok(group - 1);
            }
            slot = (slot + 1) & mask;
        }
        if groups.len == GROUPS_MAX {
            return Err(Error::OutOfMemory);
        }
        groups.keys.extend_from_slice(keys)?;
        groups.totals.extend_from_slice(&self.empty)?;
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let found = groups.len as u32;
        groups.len += 1;
        *at_mut!(groups.slots, slot) = found + 1;
        if groups.len * 2 > groups.slots.len() {
            self.grow(allocator, groups)?;
        }
        Ok(found)
    }

    /// Doubles `groups`' slots, putting each group back.
    fn grow(&self, allocator: &dyn Allocator, groups: &mut Groups) -> Result<(), Error> {
        let mut slots = SlowVec::zeroed(allocator, groups.slots.len() * 2)?;
        let mask = slots.len() - 1;
        let width = self.grouping.keys.len() + 1;
        for (index, keys) in (1..).zip(groups.keys.chunks(width)) {
            #[expect(clippy::cast_possible_truncation, reason = "masked")]
            let mut slot = hash(keys) as usize & mask;
            while *at!(slots, slot) != 0 {
                slot = (slot + 1) & mask;
            }
            *at_mut!(slots, slot) = index;
        }
        groups.slots = slots;
        Ok(())
    }

    /// Writes the group of each of `rows` of `input` in `ids`, or returns
    /// it if it's every row's.
    fn group_rows(
        &self,
        allocator: &dyn Allocator,
        groups: &mut Groups,
        input: &RowBatch,
        rows: &[u16],
        work: &mut Work,
    ) -> Result<Option<u32>, Error> {
        let mut columns = self.grouping.keys.iter().map(|&position| input.column(position));
        if columns.all(|column| column.is_in(Forms::CONSTANT)) {
            return Ok(Some(self.row_group(allocator, groups, input, &mut work.keys, 0)?));
        }
        if !self.code(allocator, groups, input, rows, work)? {
            for &row in rows {
                *at_mut!(work.ids, usize::from(row) % ROWS) =
                    self.row_group(allocator, groups, input, &mut work.keys, row)?;
            }
        }
        Ok(None)
    }

    /// The group of `row` of `input`, by its keys, added if new.
    fn row_group(
        &self,
        allocator: &dyn Allocator,
        groups: &mut Groups,
        input: &RowBatch,
        keys: &mut [i64; KEYS_MAX + 1],
        row: u16,
    ) -> Result<u32, Error> {
        let keys = at_mut!(keys, ..=self.grouping.keys.len());
        let mut nulls = 0;
        for (k, (key, &position)) in
            keys.iter_mut().skip(1).zip(self.grouping.keys.iter()).enumerate()
        {
            let value = key_of(input.column(position), row);
            nulls |= u64::from(value.is_none()) << k;
            *key = value.unwrap_or(0);
        }
        *at_mut!(keys, 0) = nulls.cast_signed();
        self.group(allocator, groups, keys)
    }

    /// Writes the group of each of `rows` of `input` in `ids` by its keys'
    /// codes, if each key has a space of them, returning whether they did.
    fn code(
        &self,
        allocator: &dyn Allocator,
        groups: &mut Groups,
        input: &RowBatch,
        rows: &[u16],
        work: &mut Work,
    ) -> Result<bool, Error> {
        let (mut len, mut same) = (1_u64, groups.spaces.len() == self.grouping.keys.len());
        for (k, &position) in self.grouping.keys.iter().enumerate() {
            let Some(space) = Space::of(input.column(position)) else { return Ok(false) };
            let Some(product) = len.checked_mul(u64::from(space.len())) else { return Ok(false) };
            len = product;
            same &= groups.spaces.get(k).is_some_and(|known| known.same(&space));
        }
        if !same {
            groups.spaces.retain(|_| false);
            for &position in self.grouping.keys.iter() {
                let Some(space) = Space::of(input.column(position)) else {
                    crate::check::check_failed(line!());
                };
                groups.spaces.push(space).map_err(|_| AllocError)?;
            }
            #[expect(clippy::cast_possible_truncation, reason = "at most `DIRECT_MAX`")]
            let (direct, hashed) = if len > DIRECT_MAX { (0, 128) } else { (len as usize, 0) };
            groups.direct = SlowVec::zeroed(allocator, direct)?;
            (groups.hashed, groups.coded) = (SlowVec::zeroed(allocator, hashed)?, 0);
        }
        let Work { ids, codes, keys } = work;
        // Each row's codes, as one: the first key's, then each next's after.
        // A constant's one code leaves it as it is, and the first key that
        // isn't one scales what was there by 0, so it needn't be cleared.
        let mut first = true;
        for (space, &position) in groups.spaces.iter().zip(self.grouping.keys.iter()) {
            let column = input.column(position);
            if column.is_in(Forms::CONSTANT) {
                continue;
            }
            let len = u64::from(space.len());
            let scale = if first { 0 } else { len };
            first = false;
            match (column.form(), space) {
                (Form::Dictionary(indices), _) => {
                    for &row in rows {
                        let code = at_mut!(codes, usize::from(row) % ROWS);
                        *code = *code * scale + u64::from(*at!(indices, usize::from(row)));
                    }
                }
                (Form::Flat, Space::Range { min, .. }) => {
                    let (values, validity) = (column.int64s(), column.validity());
                    for &row in rows {
                        let mut value = at!(values, usize::from(row)).wrapping_sub(*min);
                        // SAFETY: `row` is one of the column's.
                        if !validity.is_none_or(|validity| unsafe {
                            validity.is_valid_unchecked(row.into())
                        }) {
                            value = (len - 1).cast_signed();
                        }
                        let code = at_mut!(codes, usize::from(row) % ROWS);
                        *code = *code * scale + value.cast_unsigned();
                    }
                }
                _ => crate::check::check_failed(line!()),
            }
        }
        for &row in rows {
            let code = *at!(codes, usize::from(row) % ROWS);
            #[expect(clippy::cast_possible_truncation, reason = "below `DIRECT_MAX` if direct")]
            let group = groups.direct.get(code as usize).copied().unwrap_or(0);
            let group =
                if group == 0 { self.coded(allocator, groups, keys, code)? } else { group - 1 };
            *at_mut!(ids, usize::from(row) % ROWS) = group;
        }
        Ok(true)
    }

    /// The group of `code`, of the keys' spaces together, added if new,
    /// with `keys` to decode it into.
    #[inline]
    fn coded(
        &self,
        allocator: &dyn Allocator,
        groups: &mut Groups,
        keys: &mut [i64; KEYS_MAX + 1],
        code: u64,
    ) -> Result<u32, Error> {
        if groups.hashed.is_empty() {
            let group = self.decode(allocator, groups, keys, code)?;
            #[expect(clippy::cast_possible_truncation, reason = "below `DIRECT_MAX`")]
            let slot = at_mut!(groups.direct, code as usize);
            *slot = group + 1;
            return Ok(group);
        }
        let mask = groups.hashed.len() / 2 - 1;
        #[expect(clippy::cast_possible_truncation, reason = "masked")]
        let mut slot = hash(&[code.cast_signed()]) as usize & mask;
        loop {
            let group = *at!(groups.hashed, slot * 2 + 1);
            if group == 0 {
                break;
            }
            #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
            if *at!(groups.hashed, slot * 2) == code {
                return Ok(group as u32 - 1);
            }
            slot = (slot + 1) & mask;
        }
        let group = self.decode(allocator, groups, keys, code)?;
        at_mut!(groups.hashed, slot * 2..slot * 2 + 2)
            .copy_from_slice(&[code, u64::from(group) + 1]);
        groups.coded += 1;
        if groups.coded * 2 > mask + 1 {
            let full =
                core::mem::replace(&mut groups.hashed, SlowVec::zeroed(allocator, (mask + 1) * 4)?);
            let mask = mask * 2 + 1;
            for pair in full.chunks(2).filter(|pair| *at!(pair, 1) != 0) {
                #[expect(clippy::cast_possible_truncation, reason = "masked")]
                let mut slot = hash(&[at!(pair, 0).cast_signed()]) as usize & mask;
                while *at!(groups.hashed, slot * 2 + 1) != 0 {
                    slot = (slot + 1) & mask;
                }
                at_mut!(groups.hashed, slot * 2..slot * 2 + 2).copy_from_slice(pair);
            }
        }
        Ok(group)
    }

    /// The group of `code`, decoded into `keys`, added if new.
    fn decode(
        &self,
        allocator: &dyn Allocator,
        groups: &mut Groups,
        keys: &mut [i64; KEYS_MAX + 1],
        code: u64,
    ) -> Result<u32, Error> {
        let keys = at_mut!(keys, ..=self.grouping.keys.len());
        let (mut rest, mut nulls) = (code, 0);
        let spaces = groups.spaces.iter();
        for (k, (key, space)) in keys.iter_mut().skip(1).zip(spaces).enumerate().rev() {
            let len = u64::from(space.len());
            #[expect(clippy::cast_possible_truncation, reason = "below a `u32`")]
            let value = space.value((rest % len) as u32);
            nulls |= u64::from(value.is_none()) << k;
            *key = value.unwrap_or(0);
            rest /= len;
        }
        *at_mut!(keys, 0) = nulls.cast_signed();
        self.group(allocator, groups, keys)
    }

    /// Adds `input`'s rows to their groups, with room in `scratch`.
    fn add_batch(
        &self,
        context: &mut Context,
        groups: &mut Groups,
        input: &RowBatch,
        scratch: &mut Scratch,
    ) -> Result<(), Error> {
        let Scratch { all, work } = scratch;
        let rows = match input.selection().kept() {
            Kept::All => at!(all, ..input.row_count() as usize),
            Kept::Select(rows) => rows,
            Kept::None => return Ok(()),
        };
        if rows.is_empty() {
            return Ok(());
        }
        // Without keys, every row is in group 0.
        let one = if self.grouping.keys.is_empty() {
            Some(0)
        } else {
            self.group_rows(context.allocator(), groups, input, rows, work)?
        };
        let inputs = self.grouping.aggregates.iter().filter_map(|aggregate| aggregate.input);
        let input = Flat::new(context, input, inputs.map(|(position, _)| position))?;
        for a in 0..self.grouping.aggregates.len() {
            let ids = &work.ids;
            let target = Aggregated { ids, one, count: self.grouping.aggregates.len(), a };
            self.add(&input, rows, target, &mut groups.totals);
        }
        Ok(())
    }

    /// Adds the values of an aggregate in `rows` of `input` to their
    /// groups' `totals`, as `target` says.
    fn add(&self, input: &Flat, rows: &[u16], target: Aggregated, totals: &mut [Total]) {
        let aggregate = *at!(self.grouping.aggregates, target.a);
        let Some((position, data_type)) = aggregate.input else {
            if let Some(group) = target.one {
                at_mut!(totals, target.at(group)).count += rows.len() as u64;
                return;
            }
            for &row in rows {
                at_mut!(totals, target.at(*at!(target.ids, usize::from(row) % ROWS))).count += 1;
            }
            return;
        };
        let column = input.column(position);
        let int = data_type == DataType::Int64;
        let (ints, floats) =
            if int { (column.int64s(), &[][..]) } else { (&[][..], column.float64s()) };
        let int = |row: usize| i128::from(*at!(ints, row));
        let float = |row: usize| *at!(floats, row);
        // The greatest is the least with the order reversed: of each
        // integer's bits flipped, or each float negated. Integers' sums are
        // the most common, and get their own loop.
        let (flip, sign) = if aggregate.function == Function::Max { (-1, -1.0) } else { (0, 1.0) };
        match (aggregate.function, data_type) {
            (Function::Sum | Function::Avg, DataType::Int64) => {
                target.fold(rows, column, totals, aggregate.function, |total, row| {
                    total.int += int(row);
                });
            }
            (function, data_type) => {
                target.fold(rows, column, totals, aggregate.function, |total, row| {
                    match (function, data_type) {
                        (Function::Count, _) => {}
                        (Function::Min | Function::Max, DataType::Int64) => {
                            total.int = total.int.min(int(row) ^ flip);
                        }
                        (Function::Min | Function::Max, _) => {
                            total.float = total.float.min(float(row) * sign);
                        }
                        _ => total.float += float(row),
                    }
                });
            }
        }
    }

    /// Column `c` of the output, of its keys and then its aggregates, for
    /// groups `first..first + rows`.
    fn output(
        &self,
        context: &mut Context,
        groups: &Groups,
        c: usize,
        first: usize,
        rows: usize,
    ) -> Result<ColumnView, Error> {
        let (width, count) = (self.grouping.keys.len() + 1, self.grouping.aggregates.len());
        // An aggregate's, or `None` for a key's.
        let keys = self.grouping.keys.len();
        let aggregate = c.checked_sub(keys).map(|a| (a, *at!(self.grouping.aggregates, a)));
        let data_type = aggregate.and_then(|(_, aggregate)| aggregate.input).map(|(_, t)| t);
        let output_type = match aggregate {
            None => Some(DataType::Int64),
            Some((_, aggregate)) => aggregate.function.output_type(data_type),
        };
        let Some(output_type) = output_type else { crate::check::check_failed(line!()) };
        let mut values = context.values_buffer(rows * 8)?;
        let mut validity = context.small_buffer(rows.div_ceil(8))?;
        validity.as_mut_slice::<u8>().fill(0);
        let mut any_null = false;
        for (i, value) in values.as_mut_slice::<i64>().iter_mut().enumerate() {
            let group = first + i;
            let word = match aggregate {
                None => (*at!(groups.keys, group * width) >> c & 1 == 0)
                    .then(|| *at!(groups.keys, group * width + 1 + c)),
                Some((a, aggregate)) => {
                    let total = *at!(groups.totals, group * count + a);
                    result(aggregate.function, data_type, total)?
                }
            };
            *value = word.unwrap_or(0);
            any_null |= word.is_none();
            *at_mut!(validity.as_mut_slice::<u8>(), i / 8) |= u8::from(word.is_some()) << (i % 8);
        }
        Ok(ColumnView::new(context, output_type, values, any_null.then_some(validity))?)
    }
}

/// What `function` makes of `total`, of values of `data_type`, as a word, or
/// `None` if null.
#[inline]
fn result(
    function: Function,
    data_type: Option<DataType>,
    mut total: Total,
) -> Result<Option<i64>, Error> {
    if function == Function::Max {
        (total.int, total.float) = (!total.int, -total.float);
    }
    Ok(match (function, data_type) {
        (Function::Count, _) => Some(i64::try_from(total.count).map_err(|_| Error::Unsupported)?),
        _ if total.count == 0 => None,
        #[expect(clippy::cast_precision_loss, reason = "averages are approximate")]
        (Function::Avg, Some(DataType::Int64)) => {
            Some((total.int as f64 / total.count as f64).to_bits().cast_signed())
        }
        #[expect(clippy::cast_precision_loss, reason = "averages are approximate")]
        (Function::Avg, _) => Some((total.float / total.count as f64).to_bits().cast_signed()),
        // A sum that doesn't fit in an `Int64` would need a wider type.
        (_, Some(DataType::Int64)) => {
            Some(i64::try_from(total.int).map_err(|_| Error::Unsupported)?)
        }
        _ => Some(total.float.to_bits().cast_signed()),
    })
}

/// Where the totals of one aggregate are, for the rows of a batch.
#[derive(Clone, Copy)]
struct Aggregated<'b> {
    /// Each row's group.
    ids: &'b Ids,
    /// The group of every row, if they're all in one.
    one: Option<u32>,
    /// How many aggregates each group has totals for.
    count: usize,
    /// Which aggregate's.
    a: usize,
}

impl Aggregated<'_> {
    /// Where `group`'s total is among every group's.
    fn at(self, group: u32) -> usize {
        group as usize * self.count + self.a
    }

    /// Calls `f` with each of `rows` where `column` isn't null, and its
    /// group's total in `totals`, of `function`, which then counts it.
    #[inline]
    fn fold(
        self,
        rows: &[u16],
        column: &ColumnView,
        totals: &mut [Total],
        function: Function,
        mut f: impl FnMut(&mut Total, usize),
    ) {
        let validity = column.validity();
        let valid = |row: u16| {
            // SAFETY: `row` is one of the column's.
            validity.is_none_or(|validity| unsafe { validity.is_valid_unchecked(row.into()) })
        };
        // Every row, none null: no need to look at each.
        let dense = validity.is_none() && rows.len() == column.row_count() as usize;
        if let Some(group) = self.one {
            let mut one = Total::empty(function);
            if dense {
                (0..rows.len()).for_each(|row| f(&mut one, row));
                one.count = rows.len() as u64;
            } else {
                for &row in rows.iter().filter(|&&row| valid(row)) {
                    f(&mut one, row.into());
                    one.count += 1;
                }
            }
            at_mut!(totals, self.at(group)).add(function, &one);
            return;
        }
        let mut add = |row: usize| {
            let total = at_mut!(totals, self.at(*at!(self.ids, row % ROWS)));
            f(total, row);
            total.count += 1;
        };
        if dense {
            (0..rows.len()).for_each(add);
        } else {
            rows.iter().filter(|&&row| valid(row)).for_each(|&row| add(row.into()));
        }
    }
}

/// A batch whose columns this reads are flat: they're in whatever form the
/// first op to read them takes, so those that aren't are made flat.
struct Flat<'b, 'r> {
    /// The batch, whose flat columns are read as they are.
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

/// `words`, hashed.
fn hash(words: &[i64]) -> u64 {
    let mix =
        |hash: u64, word: u64| (hash ^ word).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29);
    words.iter().fold(0, |hash, &word| mix(hash, word.cast_unsigned()))
}

impl Breaker for HashAggregate {
    type State = Groups;

    fn new_state(&self, context: &mut Context) -> Result<Groups, Error> {
        let allocator = context.allocator();
        let width = self.grouping.keys.len() + 1;
        let count = self.grouping.aggregates.len().max(1);
        let mut groups = Groups {
            len: 0,
            keys: SlowVec::new(allocator, (GROUPS_MAX * width).next_power_of_two())?,
            totals: SlowVec::new(allocator, (GROUPS_MAX * count).next_power_of_two())?,
            slots: SlowVec::zeroed(allocator, 16)?,
            spaces: SlowVec::new(allocator, KEYS_MAX)?,
            direct: SlowVec::new(allocator, 1)?,
            hashed: SlowVec::new(allocator, 1)?,
            coded: 0,
            scratch: None,
            served: 0,
        };
        #[expect(clippy::cast_possible_truncation, reason = "a batch's rows")]
        let all = core::array::from_fn(|row| row as u16);
        let work = Work { ids: [0; ROWS], codes: [0; ROWS], keys: [0; KEYS_MAX + 1] };
        let scratch = Scratch { all, work };
        groups.scratch = Some(Box::new(allocator, scratch)?);
        // All rows are one group, which is there even if there are none.
        if self.grouping.keys.is_empty() {
            self.group(allocator, &mut groups, &[0])?;
        }
        Ok(groups)
    }

    fn consume(
        &self,
        context: &mut Context,
        groups: &mut Groups,
        input: &RowBatch,
    ) -> Result<(), Error> {
        let Some(mut scratch) = groups.scratch.take() else {
            crate::check::check_failed(line!());
        };
        let added = self.add_batch(context, groups, input, &mut scratch);
        groups.scratch = Some(scratch);
        added
    }

    fn merge(
        &self,
        context: &mut Context,
        groups: &mut Groups,
        other: Groups,
    ) -> Result<(), Error> {
        let allocator = context.allocator();
        let width = self.grouping.keys.len() + 1;
        let count = self.grouping.aggregates.len();
        for (found, keys) in other.keys.chunks(width).enumerate() {
            let group = self.group(allocator, groups, keys)? as usize;
            let theirs = at!(other.totals, found * count..(found + 1) * count);
            let ours = at_mut!(groups.totals, group * count..(group + 1) * count);
            for ((aggregate, ours), theirs) in self.grouping.aggregates.iter().zip(ours).zip(theirs)
            {
                ours.add(aggregate.function, theirs);
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
        let rows = (groups.len - first).min(BATCH_ROWS_MAX as usize);
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
        output.reset(rows as u32);
        for c in 0..self.grouping.keys.len() + self.grouping.aggregates.len() {
            let column = self.output(context, groups, c, first, rows)?;
            let Ok(()) = output.push_column(column) else { crate::check::check_failed(line!()) };
        }
        groups.served += rows;
        Ok(groups.served < groups.len)
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
        assert_eq!(first.len, 2000);
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

    /// Each group's keys and `COUNT(*)`, sorted, of `batches`, each a list
    /// of key columns, consumed by two states in turn and then merged.
    fn counted(batches: &[Vec<ColumnView>]) -> Vec<(Vec<Option<i64>>, i64)> {
        let keys = 0..u32::try_from(batches[0].len()).unwrap();
        let aggregates = [Lowered { function: Function::Count, input: None }].into_iter();
        let grouping = Grouping {
            keys: SlowVec::fixed_from(&Heap, keys).unwrap(),
            aggregates: SlowVec::fixed_from(&Heap, aggregates).unwrap(),
        };
        let breaker = HashAggregate::new(&Heap, grouping).unwrap();
        let mut context = Context::new(&Heap);
        let mut states: Vec<_> =
            (0..2).map(|_| Breaker::new_state(&breaker, &mut context).unwrap()).collect();
        for (i, columns) in batches.iter().enumerate() {
            let mut batch = RowBatch::new();
            batch.reset(columns[0].row_count());
            for column in columns {
                assert!(batch.push_column(column.clone()).is_ok());
            }
            breaker.consume(&mut context, &mut states[i % 2], &batch).unwrap();
        }
        let second = states.pop().unwrap();
        let mut first = states.pop().unwrap();
        breaker.merge(&mut context, &mut first, second).unwrap();
        let mut output = RowBatch::new();
        let mut counts = Vec::new();
        loop {
            let more = breaker.serve(&mut context, &mut first, &mut output).unwrap();
            for row in 0..output.row_count() {
                let keys = (0..output.column_count() - 1).map(|k| {
                    let column = output.column(k);
                    (!column.is_null(row)).then(|| column.int64s()[row as usize])
                });
                let count = output.column(output.column_count() - 1).int64s()[row as usize];
                counts.push((keys.collect(), count));
            }
            if !more {
                break;
            }
        }
        counts.sort();
        counts
    }

    /// A dictionary column of `indices` into `values`.
    fn dictionary(values: &ColumnView, indices: &[u32]) -> ColumnView {
        let mut buffer = Buffer::allocate(&Heap, indices.len() * 4).unwrap();
        buffer.as_mut_slice::<u32>().copy_from_slice(indices);
        ColumnView::dictionary(&mut Context::new(&Heap), values, buffer).unwrap()
    }

    #[test]
    fn groups_keys_in_every_form() {
        // 10, 20, and a null, as Parquet's dictionaries end.
        let values = column(DataType::Int64, &[Some(10), Some(20), None]);
        let constant = |value: &ColumnView, rows| {
            ColumnView::constant(&mut Context::new(&Heap), value, rows).unwrap()
        };
        let bounds = Bounds { min: 10, max: 30 };
        let batches = [
            // Two batches sharing a dictionary.
            dictionary(&values, &[0, 1, 2, 0]),
            dictionary(&values, &[1, 1]),
            constant(&values.slice(0, 1), 3),
            constant(&column(DataType::Int64, &[None]), 2),
            column(DataType::Int64, &[Some(10), None, Some(30)]).with_bounds(bounds),
            // No bounds: hashed.
            column(DataType::Int64, &[Some(20), Some(-5)]),
        ];
        let batches: Vec<_> = batches.into_iter().map(|column| alloc::vec![column]).collect();
        let expected = [(None, 4), (Some(-5), 1), (Some(10), 6), (Some(20), 4), (Some(30), 1)];
        let expected: Vec<_> = expected.into_iter().map(|(k, n)| (alloc::vec![k], n)).collect();
        assert_eq!(counted(&batches), expected);
    }

    #[test]
    fn groups_keys_by_codes_hashed_if_many() {
        // 300 values each, so 90,000 codes together: too many for a slot each.
        let values: Vec<_> = (0..300).map(Some).collect();
        let values = column(DataType::Int64, &values);
        let batch = |rows: core::ops::Range<u32>| {
            let indices: Vec<_> = rows.map(|row| row % 300).collect();
            alloc::vec![dictionary(&values, &indices), dictionary(&values, &indices)]
        };
        // And one whose first key is a constant: 300 codes, a slot each.
        let zero = ColumnView::constant(&mut Context::new(&Heap), &values.slice(0, 1), 2).unwrap();
        let mixed = alloc::vec![zero, dictionary(&values, &[0, 0])];
        let count = |k| match k {
            0 => 6,
            1..100 => 4,
            _ => 3,
        };
        let expected: Vec<_> =
            (0..300).map(|k| (alloc::vec![Some(k), Some(k)], count(k))).collect();
        assert_eq!(counted(&[batch(0..500), batch(500..1000), mixed]), expected);
    }

    #[test]
    fn aggregates_floats() {
        let x = Some((0, DataType::Float64));
        let functions = [Function::Min, Function::Max, Function::Sum, Function::Avg];
        let aggregates = functions.into_iter().map(|function| Lowered { function, input: x });
        let grouping = Grouping {
            keys: SlowVec::fixed(&Heap, 0).unwrap(),
            aggregates: SlowVec::fixed_from(&Heap, aggregates).unwrap(),
        };
        let breaker = HashAggregate::new(&Heap, grouping).unwrap();
        let mut context = Context::new(&Heap);
        let mut state = Breaker::new_state(&breaker, &mut context).unwrap();
        let values = [Some(1.5), Some(-2.0), None, Some(4.0)];
        let bits: Vec<_> =
            values.iter().map(|value| value.map(|v: f64| v.to_bits().cast_signed())).collect();
        let mut batch = RowBatch::new();
        batch.reset(4);
        assert!(batch.push_column(column(DataType::Float64, &bits)).is_ok());
        breaker.consume(&mut context, &mut state, &batch).unwrap();
        let mut output = RowBatch::new();
        assert!(!breaker.serve(&mut context, &mut state, &mut output).unwrap());
        let row: Vec<_> = (0..4).map(|c| output.column(c).float64s()[0]).collect();
        assert_eq!(row, [-2.0, 4.0, 3.5, 3.5 / 3.0]);
    }
}
