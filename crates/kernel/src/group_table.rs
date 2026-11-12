//! `GroupTable`: numbers rows by their keys, as Perfetto's does, for the
//! operators that group rows. A string key is grouped by its id in the
//! run's `StringPool`.

use crate::aggregate_function::{gathered, is_valid};
use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType, Form};
use crate::context::Context;
use crate::error::Error;
use crate::flat_hash_map::prefetch;
use crate::murmur_hash::mix;
use crate::row_batch::BATCH_ROWS_MAX;
use crate::slow_vec::SlowVec;
use crate::string_pool::{CACHED_SLOTS, Id, NO_NUMBER};

/// The most keys a grouping can have: one bit each says which are null.
pub(crate) const KEYS_MAX: usize = 64;

/// The most groups a table can hold.
pub(crate) const GROUPS_MAX: usize = 1 << 26;

/// The top bits of a hash, which a slot keeps.
fn salt(hash: u64) -> u64 {
    hash >> GROUP_BITS << GROUP_BITS
}

/// How many of a slot's low bits hold the group number plus one.
const GROUP_BITS: u32 = 27;

/// The slot that `hash` starts probing from, out of `slots`, a power of two.
/// It's picked by the hash's top bits, which a slot stores, so growing the
/// table doesn't need to hash any key again.
/// How many rows ahead `find_values` fetches a row's slot, and its group's
/// record.
const SLOT_AHEAD: usize = 16;
const RECORD_AHEAD: usize = 8;

#[inline]
fn slot_of(hash: u64, slots: usize) -> usize {
    #[expect(clippy::cast_possible_truncation, reason = "below `slots`")]
    let at = (hash >> (slots.leading_zeros() + 1)) as usize;
    at
}

/// Hash table that assigns a number to each distinct key, in the order keys
/// are first seen. As in DuckDB's aggregate hash table, each slot stores the
/// top bits of the group's hash, and keys are only compared when those match.
/// Nulls are equal to each other and to nothing else.
pub(crate) struct GroupTable {
    /// The hash table. An empty slot is 0. A used slot holds the top bits
    /// of the group's hash, then the group number plus one in the low
    /// `GROUP_BITS` bits.
    slots: SlowVec<u64>,
    /// Each group's null bits, then its keys, then its payload, `stride`
    /// words each. Keeping them together means comparing a group reads one
    /// cache line.
    records: SlowVec<i64>,
    /// How many words each group takes.
    stride: usize,
    /// How many keys there are.
    keys: usize,
    /// Bit `k` set if key `k` is a string, stored as its id, which is 0 if
    /// null: as in Perfetto, a null string is a key, not a null one.
    string_keys: u64,
    /// For each key, the last dictionary it came in, if it's a string key.
    dictionaries: SlowVec<Option<Interned>>,
    /// With a single key, a string, the group of each string by its number
    /// in the run's pool, `NONE` until it has one, so a row's group is found
    /// without hashing again.
    number_groups: SlowVec<u32>,
    /// With a single key, a string, the null string's group, once it has one.
    null_group: u32,
}

/// No number, or no group.
const NONE: u32 = u32::MAX;

/// A string key's dictionary, and the ids of its entries: kept while
/// batches share it, as a chunk's do, so each entry is interned once.
struct Interned {
    /// A column of the dictionary, kept so its values aren't freed, and
    /// their address reused, while the ids are.
    dictionary: ColumnView,
    /// Each entry's id, and its number above, or 0 until a row with it is
    /// found.
    ids: SlowVec<u64>,
}

impl GroupTable {
    /// A table of `keys` keys, those in `string_keys` strings, with
    /// `payload_words` zeroed words after each group's keys for the caller
    /// to store per-group data.
    pub(crate) fn new(
        allocator: &dyn Allocator,
        keys: usize,
        string_keys: u64,
        payload_words: usize,
    ) -> Result<GroupTable, AllocError> {
        check!(keys <= KEYS_MAX);
        let stride = 1 + keys + payload_words;
        Ok(GroupTable {
            slots: SlowVec::new(allocator, 1)?,
            // A `SlowVec` holds at most 2^31 values.
            records: SlowVec::new(
                allocator,
                (GROUPS_MAX * stride).next_power_of_two().min(1 << 31),
            )?,
            stride,
            keys,
            string_keys,
            dictionaries: SlowVec::fixed_from(allocator, (0..keys).map(|_| None))?,
            number_groups: SlowVec::new(allocator, 1 << 31)?,
            null_group: NONE,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.records.len() / self.stride
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
    /// groups. Each column is flat, or a dictionary for a string key, of
    /// strings for a string key, else of `Int64`s.
    pub(crate) fn find_or_create(
        &mut self,
        context: &mut Context,
        columns: &[&ColumnView],
        select: &[u16],
        groups: &mut [u32],
    ) -> Result<(), Error> {
        check!(columns.len() == self.keys);
        let count = groups.len();
        if (self.keys, self.string_keys) == (1, 1) {
            let (mut ids, mut numbers) = (None, context.indices_buffer(count * 4)?);
            let (interned, column) = (at_mut!(self.dictionaries, 0), at!(columns, 0));
            let out = Some(numbers.as_mut_slice());
            load_string_ids(context, interned, column, select, count, &mut ids, out)?;
            let Some(ids) = ids else { failed!() };
            return self.find_numbered(ids.as_slice(), numbers.as_slice(), groups);
        }
        // Each key as a word a row, 0 where null, and a word a row of which
        // keys are null, unless none are.
        let mut copies = [const { None }; KEYS_MAX];
        let mut nulls: Option<Buffer> = None;
        for (k, (column, copy)) in columns.iter().zip(copies.iter_mut()).enumerate() {
            if self.string_keys >> k & 1 != 0 {
                let interned = at_mut!(self.dictionaries, k);
                load_string_ids(context, interned, column, select, count, copy, None)?;
            } else {
                load_key_words(context, column, select, count, k, copy, &mut nulls)?;
            }
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
                Err(at) => self.add_row(at, hash, keys, row, null(row))?,
            };
        }
        Ok(())
    }

    /// As `find_or_create`, for a single key, a string, given each row's id
    /// and number in the run's pool: a string's group is found by its
    /// number, without hashing it again.
    #[inline(never)]
    fn find_numbered(
        &mut self,
        ids: &[i64],
        numbers: &[u32],
        groups: &mut [u32],
    ) -> Result<(), Error> {
        for ((&id, &number), group) in ids.iter().zip(numbers).zip(groups.iter_mut()) {
            // Other tables share the pool, so numbers can be skipped.
            while number != NO_NUMBER && number as usize >= self.number_groups.len() {
                self.number_groups.push(NONE)?;
            }
            let known = match number {
                NO_NUMBER => self.null_group,
                _ => *at!(self.number_groups, number as usize),
            };
            *group = if known == NONE { self.add_numbered(number, id)? } else { known };
        }
        Ok(())
    }

    /// Adds a group for the string with id `id` and number `number`, and
    /// returns its number.
    #[cold]
    #[inline(never)]
    fn add_numbered(&mut self, number: u32, id: i64) -> Result<u32, Error> {
        let group = self.push(&[0, id])?;
        match number {
            NO_NUMBER => self.null_group = group,
            _ => *at_mut!(self.number_groups, number as usize) = group,
        }
        Ok(group)
    }

    /// As `find_or_create`, for a single key with no nulls: compares the key
    /// values directly.
    fn find_values(&mut self, keys: &[i64], groups: &mut [u32]) -> Result<(), Error> {
        check!(self.records.len() == self.len() * self.stride && self.stride >= 2);
        // `prefetch` does nothing on other targets, so they don't look ahead.
        if cfg!(target_arch = "x86_64") && self.slots.len() > CACHED_SLOTS {
            self.find_values_in::<true>(keys, groups)
        } else {
            self.find_values_in::<false>(keys, groups)
        }
    }

    /// As `find_values`. A table too big for the cache misses it twice a
    /// row, for its slot and then its group's record, so if `AHEAD`, both
    /// are fetched ahead.
    #[inline(never)]
    fn find_values_in<const AHEAD: bool>(
        &mut self,
        keys: &[i64],
        groups: &mut [u32],
    ) -> Result<(), Error> {
        for (i, (&key, group)) in keys.iter().zip(groups).enumerate() {
            if AHEAD {
                if let Some(&ahead) = keys.get(i + SLOT_AHEAD) {
                    let at = slot_of(mix(ahead.cast_unsigned()), self.slots.len());
                    prefetch(self.slots.as_ptr().wrapping_add(at));
                }
                if let Some(&ahead) = keys.get(i + RECORD_AHEAD) {
                    let at = slot_of(mix(ahead.cast_unsigned()), self.slots.len());
                    let slot = *at!(self.slots, at);
                    #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
                    let group = (slot & !salt(slot)) as usize;
                    let record = group.wrapping_sub(1) * self.stride;
                    prefetch(self.records.as_ptr().wrapping_add(record));
                }
            }
            let hash = mix(key.cast_unsigned());
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

    /// Adds a group for row `row` of `keys`, with null word `null`, hashed as
    /// `hash`, in slot `at`, and returns its number. Out of line, so the loops
    /// that find groups keep what they read in registers.
    #[cold]
    #[inline(never)]
    fn add_row(
        &mut self,
        at: usize,
        hash: u64,
        keys: &[&[i64]],
        row: usize,
        null: u64,
    ) -> Result<u32, Error> {
        let group = self.len();
        if group == GROUPS_MAX {
            return Err(Error::OutOfMemory);
        }
        let start = self.records.len();
        self.records.extend_zeroed(self.stride)?;
        *at_mut!(self.records, start) = null.cast_signed();
        for (word, keys) in at_mut!(self.records, start + 1..).iter_mut().zip(keys) {
            *word = *at!(keys, row);
        }
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let group = group as u32;
        *at_mut!(self.slots, at) = salt(hash) | (u64::from(group) + 1);
        Ok(group)
    }

    /// As `add_row`, given the group's null word and keys together in `key`.
    fn add(&mut self, at: usize, hash: u64, key: &[i64]) -> Result<u32, Error> {
        let group = self.push(key)?;
        *at_mut!(self.slots, at) = salt(hash) | (u64::from(group) + 1);
        Ok(group)
    }

    /// Appends a group of `key`, its null word and keys, and returns its
    /// number.
    fn push(&mut self, key: &[i64]) -> Result<u32, Error> {
        let group = self.len();
        if group == GROUPS_MAX {
            return Err(Error::OutOfMemory);
        }
        let start = self.records.len();
        self.records.extend_zeroed(self.stride)?;
        at_mut!(self.records, start..start + key.len()).copy_from_slice(key);
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        Ok(group as u32)
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
        let mut slots = SlowVec::zeroed(allocator, capacity)?;
        let mask = capacity - 1;
        // No need to hash the keys again: `slot_of` only uses bits a slot has.
        for &slot in self.slots.iter().filter(|&&slot| slot != 0) {
            let mut at = slot_of(slot, capacity);
            while *at!(slots, at) != 0 {
                at = (at + 1) & mask;
            }
            *at_mut!(slots, at) = slot;
        }
        self.slots = slots;
        Ok(())
    }

    /// Key `k` of the groups in `range`, as a column of its type.
    pub(crate) fn write_keys(
        &self,
        context: &mut Context,
        k: usize,
        range: core::ops::Range<usize>,
    ) -> Result<ColumnView, Error> {
        let stride = self.stride;
        let records = at!(self.records, range.start * stride..range.end * stride);
        let rows = range.len();
        if self.string_keys >> k & 1 != 0 {
            return write_strings(context, records, stride, k);
        }
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

/// Loads `count` rows that `select` keeps of string key column `column`, flat
/// or a dictionary, all if `select` is empty, to `copy` as the ids the run's
/// string pool has for them, and their numbers to `numbers` if given, `NO_NUMBER`
/// for null. A dictionary's entries are each interned once,
/// when a row with it is first found, and their ids kept in `interned` while
/// batches share the dictionary.
fn load_string_ids(
    context: &mut Context,
    interned: &mut Option<Interned>,
    column: &ColumnView,
    select: &[u16],
    count: usize,
    copy: &mut Option<Buffer>,
    mut numbers: Option<&mut [u32]>,
) -> Result<(), Error> {
    check!(column.data_type() == DataType::String);
    let values = column.values();
    let strings = values.strings();
    let mut ids = context.values_buffer(count * 8)?;
    let allocator = context.allocator();
    let pool = context.string_pool()?;
    let row_of = |i: usize| if select.is_empty() { i } else { usize::from(*at!(select, i)) };
    let out = ids.as_mut_slice::<i64>().iter_mut().enumerate();
    match column.form() {
        Form::Flat => {
            let validity = column.validity();
            let string = |i: usize| {
                let row = row_of(i);
                validity.is_none_or(|validity| is_valid(validity, row)).then(|| strings.get(row))
            };
            let ids = ids.as_mut_slice::<i64>();
            pool.intern_strings(allocator, count, string, |i, id, number| {
                *at_mut!(ids, i) = i64::from(id.raw_id());
                if let Some(numbers) = numbers.as_deref_mut() {
                    *at_mut!(numbers, i) = number;
                }
            })?;
        }
        Form::Dictionary(indices) => {
            if !interned.as_ref().is_some_and(|interned| interned.dictionary.shares_values(column))
            {
                let ids = SlowVec::zeroed(allocator, values.len() as usize)?;
                *interned = Some(Interned { dictionary: column.clone(), ids });
            }
            let Some(Interned { ids: entries, .. }) = interned else { failed!() };
            for (i, id) in out {
                let entry = *at!(indices, row_of(i));
                let known = at_mut!(entries, entry as usize);
                if *known == 0 {
                    *known = if values.is_null(entry) {
                        u64::from(NO_NUMBER) << 32
                    } else {
                        let (found, number) =
                            pool.intern_numbered(allocator, strings.get(entry as usize))?;
                        u64::from(found.raw_id()) | u64::from(number) << 32
                    };
                }
                #[expect(clippy::cast_possible_truncation, reason = "the low half")]
                let found = *known as u32;
                *id = i64::from(found);
                if let Some(numbers) = numbers.as_deref_mut() {
                    *at_mut!(numbers, i) = (*known >> 32) as u32;
                }
            }
        }
        Form::Constant => failed!(),
    }
    *copy = Some(ids);
    Ok(())
}

/// Key `k`, a string key, of `records`, `stride` words each, as a string
/// column.
#[inline(never)]
fn write_strings(
    context: &mut Context,
    records: &[i64],
    stride: usize,
    k: usize,
) -> Result<ColumnView, Error> {
    let rows = records.len() / stride;
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "an id")]
    let id = |row: usize| Id::raw(*at!(records, row * stride + 1 + k) as u32);
    let mut views = context.values_buffer(rows * 8)?;
    let mut validity = context.small_buffer(rows.div_ceil(8))?;
    let valid = validity.as_mut_slice::<u8>();
    valid.fill(0);
    let (views_out, _) = views.as_mut_slice::<u32>().as_chunks_mut::<2>();
    let pool = context.string_pool()?;
    let (mut bytes, mut nulls) = (0_u32, false);
    for (row, view) in views_out.iter_mut().enumerate() {
        let string = pool.get(id(row));
        nulls |= string.is_none();
        *at_mut!(valid, row / 8) |= u8::from(string.is_some()) << (row % 8);
        let len = u32::try_from(string.map_or(0, <[u8]>::len)).map_err(|_| Error::OutOfMemory)?;
        *view = [bytes, len];
        bytes = bytes.checked_add(len).ok_or(Error::OutOfMemory)?;
    }
    let mut stored = context.bytes_buffer(bytes as usize)?;
    let pool = context.string_pool()?;
    let out = stored.as_mut_slice::<u8>();
    for (row, &[start, len]) in views_out.iter().enumerate() {
        let string = pool.get(id(row)).unwrap_or_default();
        at_mut!(out, start as usize..(start + len) as usize).copy_from_slice(string);
    }
    Ok(ColumnView::strings(context, views, stored, nulls.then_some(validity))?)
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
    let mut at = slot_of(hash, slots.len());
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

/// What Perfetto's `MurmurHashCombiner` starts from.
const SEED: u64 = 0xe17a_1465;

/// `hash` with `word` combined in, as Perfetto's `MurmurHashCombiner` does.
#[inline]
fn combine(hash: u64, word: u64) -> u64 {
    hash ^ mix(word).wrapping_add(0x9e37_79b9).wrapping_add(hash << 6).wrapping_add(hash >> 2)
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
        return mix(key(keys));
    }
    let hash = keys.iter().fold(SEED, |hash, keys| combine(hash, key(keys)));
    if null == 0 { hash } else { combine(hash, null) }
}
