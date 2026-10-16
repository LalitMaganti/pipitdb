//! `GroupTable`: numbers rows by their keys, as Perfetto's does, for the
//! operators that group rows.

use crate::aggregate_function::{gathered, is_valid};
use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType};
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::BATCH_ROWS_MAX;
use crate::slow_vec::SlowVec;

/// The most keys a grouping can have: one bit each says which are null.
pub(crate) const KEYS_MAX: usize = 64;

/// The most groups a table can hold.
pub(crate) const GROUPS_MAX: usize = 1 << 26;

/// The top bits of a hash, which a slot keeps.
fn salt(hash: u64) -> u64 {
    hash >> 48 << 48
}

/// Hash table that assigns a number to each distinct key, in the order keys
/// are first seen. As in DuckDB's aggregate hash table, each slot stores the
/// top bits of the group's hash, and keys are only compared when those match.
/// Nulls are equal to each other and to nothing else.
pub(crate) struct GroupTable {
    /// The hash table. An empty slot is 0. A used slot holds the top 16 bits
    /// of the group's hash and the group number plus one.
    slots: SlowVec<u64>,
    /// The hash of each group.
    hashes: SlowVec<u64>,
    /// Each group's null bits, then its keys, then its payload, `stride`
    /// words each. Keeping them together means comparing a group reads one
    /// cache line.
    records: SlowVec<i64>,
    /// How many words each group takes.
    stride: usize,
    /// How many keys there are.
    keys: usize,
}

impl GroupTable {
    /// A table of `keys` keys, with `payload_words` zeroed words after each
    /// group's keys for the caller to store per-group data.
    pub(crate) fn new(
        allocator: &dyn Allocator,
        keys: usize,
        payload_words: usize,
    ) -> Result<GroupTable, AllocError> {
        check!(keys <= KEYS_MAX);
        let stride = 1 + keys + payload_words;
        Ok(GroupTable {
            slots: SlowVec::new(allocator, 1)?,
            hashes: SlowVec::new(allocator, GROUPS_MAX)?,
            // A `SlowVec` holds at most 2^31 values.
            records: SlowVec::new(
                allocator,
                (GROUPS_MAX * stride).next_power_of_two().min(1 << 31),
            )?,
            stride,
            keys,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.hashes.len()
    }

    /// The payloads: group g's starts at `g * row_stride()`.
    pub(crate) fn payloads(&mut self) -> &mut [i64] {
        let start = 1 + self.keys;
        self.records.get_mut(start..).unwrap_or_default()
    }

    /// How many words apart consecutive groups' payloads are.
    pub(crate) fn row_stride(&self) -> usize {
        self.stride
    }

    /// Writes the group of each row `select` keeps of `columns`, all if it's
    /// empty, to `groups`, with `columns` as the key. New keys get new
    /// groups. Each column is a flat `Int64`.
    pub(crate) fn find_or_create(
        &mut self,
        context: &mut Context,
        columns: &[&ColumnView],
        select: &[u16],
        groups: &mut [u32],
    ) -> Result<(), Error> {
        check!(columns.len() == self.keys);
        let count = groups.len();
        // Each key as a word a row, 0 where null, and a word a row of which
        // keys are null, unless none are.
        let mut copies = [const { None }; KEYS_MAX];
        let mut nulls: Option<Buffer> = None;
        for (k, (column, copy)) in columns.iter().zip(copies.iter_mut()).enumerate() {
            load_key_words(context, column, select, count, k, copy, &mut nulls)?;
        }
        let mut keys: [&[i64]; KEYS_MAX] = [&[]; KEYS_MAX];
        for ((keys, copy), column) in keys.iter_mut().zip(&copies).zip(columns) {
            *keys = match copy {
                Some(copy) => copy.as_slice(),
                None => at!(column.int64s(), ..count),
            };
        }
        let keys = at!(keys, ..self.keys);
        let nulls = nulls.as_ref().map(Buffer::as_slice);
        self.grow(context.allocator(), self.len() + count)?;
        if let ([keys], None) = (keys, nulls) {
            return self.find_values(keys, groups);
        }
        self.find_hashed(keys, nulls, groups)
    }

    /// As `find_or_create`, given each key as a word a row, and each row's
    /// null word unless none are null.
    #[inline(never)]
    fn find_hashed(
        &mut self,
        keys: &[&[i64]],
        nulls: Option<&[u64]>,
        groups: &mut [u32],
    ) -> Result<(), Error> {
        let count = groups.len();
        check!(keys.iter().all(|keys| keys.len() >= count));
        check!(nulls.is_none_or(|nulls| nulls.len() >= count));
        check!(self.records.len() == self.len() * self.stride && self.stride > keys.len());
        // SAFETY: checked above.
        let null = |row: usize| nulls.map_or(0, |nulls| unsafe { *nulls.get_unchecked(row) });
        // SAFETY: as above, and `stored` is a group's record.
        let equal = |stored: *const i64, row: usize| unsafe {
            (*stored).cast_unsigned() == null(row)
                && keys
                    .iter()
                    .enumerate()
                    .all(|(k, keys)| *stored.add(1 + k) == *keys.get_unchecked(row))
        };
        for (row, group) in groups.iter_mut().enumerate() {
            // SAFETY: each key has a word for each row, checked above.
            let hash = unsafe { hash_row(keys, row, null(row)) };
            let (records, stride) = (self.records.as_ptr(), self.stride);
            let found = probe(&self.slots, hash, |group| {
                // SAFETY: a group in a slot has a record.
                equal(unsafe { records.add(group * stride) }, row)
            });
            *group = match found {
                Ok(group) => group,
                Err(at) => {
                    let mut key = [0_i64; 1 + KEYS_MAX];
                    *at_mut!(key, 0) = null(row).cast_signed();
                    for (word, keys) in at_mut!(key, 1..).iter_mut().zip(keys) {
                        *word = *at!(keys, row);
                    }
                    self.add(at, hash, at!(key, ..=self.keys))?
                }
            };
        }
        Ok(())
    }

    /// As `find_or_create`, for a single key with no nulls: compares the key
    /// values directly.
    #[inline(never)]
    fn find_values(&mut self, keys: &[i64], groups: &mut [u32]) -> Result<(), Error> {
        check!(self.records.len() == self.len() * self.stride && self.stride >= 2);
        for (&key, group) in keys.iter().zip(groups) {
            let hash = fmix(key.cast_unsigned());
            let (records, stride) = (self.records.as_ptr(), self.stride);
            // A stored group is its null bits, zero here, then its key.
            // SAFETY: a group in a slot has a record, of at least two words.
            let equal = |group: usize| unsafe {
                let stored = records.add(group * stride);
                *stored == 0 && *stored.add(1) == key
            };
            *group = match probe(&self.slots, hash, equal) {
                Ok(group) => group,
                Err(at) => self.add(at, hash, &[0, key])?,
            };
        }
        Ok(())
    }

    /// Adds a group for `key`, its null word and keys, hashed as `hash`, in
    /// slot `at`, and returns its number. Out of line, so the loops that find
    /// groups keep what they read in registers.
    #[cold]
    #[inline(never)]
    fn add(&mut self, at: usize, hash: u64, key: &[i64]) -> Result<u32, Error> {
        let group = self.len();
        if group == GROUPS_MAX {
            return Err(Error::OutOfMemory);
        }
        self.hashes.push(hash)?;
        let start = self.records.len();
        self.records.extend_zeroed(self.stride)?;
        at_mut!(self.records, start..start + key.len()).copy_from_slice(key);
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let group = group as u32;
        *at_mut!(self.slots, at) = salt(hash) | (u64::from(group) + 1);
        Ok(group)
    }

    /// Doubles the slots until at most two thirds are used by `groups`
    /// groups, and puts each group back in its slot.
    fn grow(&mut self, allocator: &dyn Allocator, groups: usize) -> Result<(), Error> {
        let mut capacity =
            if self.slots.is_empty() { 2 * BATCH_ROWS_MAX as usize } else { self.slots.len() };
        while groups * 3 > capacity * 2 {
            capacity *= 2;
        }
        if capacity == self.slots.len() {
            return Ok(());
        }
        self.slots = SlowVec::zeroed(allocator, capacity)?;
        let mask = capacity - 1;
        for (group, &hash) in (1..).zip(self.hashes.iter()) {
            #[expect(clippy::cast_possible_truncation, reason = "masked")]
            let mut at = hash as usize & mask;
            while *at!(self.slots, at) != 0 {
                at = (at + 1) & mask;
            }
            *at_mut!(self.slots, at) = salt(hash) | group;
        }
        Ok(())
    }

    /// Key `k` of the groups in `range`, as an `Int64` column.
    pub(crate) fn write_keys(
        &self,
        context: &mut Context,
        k: usize,
        range: core::ops::Range<usize>,
    ) -> Result<ColumnView, Error> {
        let stride = self.stride;
        let records = at!(self.records, range.start * stride..range.end * stride);
        let rows = range.len();
        let mut values = context.values_buffer(rows * 8)?;
        let out = values.as_mut_slice::<i64>().iter_mut();
        let mut nulls = false;
        for (out, record) in out.zip(records.chunks_exact(stride)) {
            *out = *at!(record, 1 + k);
            nulls |= *at!(record, 0) >> k & 1 != 0;
        }
        if !nulls {
            return Ok(ColumnView::new(context, DataType::Int64, values, None)?);
        }
        let mut validity = context.small_buffer(rows.div_ceil(8))?;
        let records = records.chunks(8 * stride);
        for (bits, records) in validity.as_mut_slice::<u8>().iter_mut().zip(records) {
            let records = records.chunks_exact(stride).enumerate();
            *bits = records
                .fold(0, |bits, (i, record)| bits | u8::from(*at!(record, 0) >> k & 1 == 0) << i);
        }
        Ok(ColumnView::new(context, DataType::Int64, values, Some(validity))?)
    }
}

/// Loads `count` rows that `select` keeps of key column `column`, all if
/// it's empty, as words: copied to `copy` unless every row is kept and none
/// is null, and 0 where null. Sets bit `k` of each null row's word in
/// `nulls`, made when the first column with nulls is.
fn load_key_words(
    context: &mut Context,
    column: &ColumnView,
    select: &[u16],
    count: usize,
    k: usize,
    copy: &mut Option<Buffer>,
    nulls: &mut Option<Buffer>,
) -> Result<(), Error> {
    let values = column.int64s();
    let Some(validity) = column.validity() else {
        gathered(context, select, values, copy)?;
        return Ok(());
    };
    if nulls.is_none() {
        let mut words = context.values_buffer(count * 8)?;
        words.as_mut_slice::<u64>().fill(0);
        *nulls = Some(words);
    }
    let Some(nulls) = nulls else { failed!() };
    let out = copy.insert(context.values_buffer(count * 8)?).as_mut_slice::<i64>();
    for (i, (out, null)) in out.iter_mut().zip(nulls.as_mut_slice::<u64>()).enumerate() {
        let row = if select.is_empty() { i } else { usize::from(*at!(select, i)) };
        let valid = is_valid(validity, row);
        *out = if valid { *at!(values, row) } else { 0 };
        *null |= u64::from(!valid) << k;
    }
    Ok(())
}

/// The group `equal` says is the one hashed as `hash`, or the empty slot to
/// put it in. `slots` are a power of two many.
#[inline]
fn probe(slots: &[u64], hash: u64, equal: impl Fn(usize) -> bool) -> Result<u32, usize> {
    check!(slots.len().is_power_of_two());
    let mask = slots.len() - 1;
    #[expect(clippy::cast_possible_truncation, reason = "masked")]
    let mut at = hash as usize & mask;
    loop {
        // SAFETY: masked below the slots' count, a power of two.
        let slot = unsafe { *slots.get_unchecked(at) };
        if slot == 0 {
            return Err(at);
        }
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let group = (slot & !salt(slot)) as u32 - 1;
        if salt(slot) == salt(hash) && equal(group as usize) {
            return Ok(group);
        }
        at = (at + 1) & mask;
    }
}

/// `fmix64` from Murmur hash 3, as Perfetto hashes a key.
#[inline]
fn fmix(word: u64) -> u64 {
    let mut h = word ^ word >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ h >> 33
}

/// What Perfetto's `MurmurHashCombiner` starts from.
const SEED: u64 = 0xe17a_1465;

/// `hash` with `word` combined in, as Perfetto's `MurmurHashCombiner` does.
#[inline]
fn combine(hash: u64, word: u64) -> u64 {
    hash ^ fmix(word).wrapping_add(0x9e37_79b9).wrapping_add(hash << 6).wrapping_add(hash >> 2)
}

/// The hash of row `row` of `keys`, whose null word is `null`, as Perfetto's
/// `GroupTable::Hash`.
///
/// # Safety
///
/// Each of `keys` must have a word for row `row`.
#[inline]
unsafe fn hash_row(keys: &[&[i64]], row: usize, null: u64) -> u64 {
    // SAFETY: as the caller promises.
    let key = |keys: &[i64]| unsafe { keys.get_unchecked(row) }.cast_unsigned();
    if let ([keys], 0) = (keys, null) {
        return fmix(key(keys));
    }
    let hash = keys.iter().fold(SEED, |hash, keys| combine(hash, key(keys)));
    if null == 0 { hash } else { combine(hash, null) }
}
