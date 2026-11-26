//! `GroupTable`: numbers rows by their keys, as Perfetto's does, for the
//! operators that group rows. A string key is grouped by its id in the
//! run's `StringPool`.

#![allow(unsafe_code)]

use crate::aggregate_function::{gathered, is_valid};
use crate::allocator::AllocError;
use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType, Form};
use crate::containers::{Array, BitSlice, Blocks, SmallVec, address};
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;
use crate::flat_hash_map::prefetch;
use crate::murmur_hash::{SEED, combine};
use crate::row_batch::BATCH_ROWS_MAX;
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

/// How many rows ahead `fetch_ahead` fetches a row's slot, and its group's
/// record.
const SLOT_AHEAD: usize = 16;
const RECORD_AHEAD: usize = 8;

/// The slot that `hash` starts probing from, out of `slots`, a power of two.
/// It's picked by the hash's top bits, which a slot stores, so growing the
/// table doesn't need to hash any key again.
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
    slots: Array<u64>,
    /// Each group's null bits, then its keys, then its payload, `stride`
    /// words each. Keeping them together means comparing a group reads one
    /// cache line.
    records: Blocks<i64>,
    /// How many words each group takes.
    stride: usize,
    /// Where a record's keys start: after its null word, if a key can be
    /// null, else first.
    first: usize,
    /// Bit `k` set if key `k` isn't a string and can be null. Others' null
    /// bits are never kept, even if their batches carry them.
    nullable: u64,
    /// How many keys there are.
    keys: usize,
    /// Bit `k` set if key `k` is a string, stored as its word in the run's
    /// pool: as in Perfetto, a null string is a key, not a null one.
    string_keys: u64,
    /// For each key, the last dictionary it came in, if it's a string key.
    dictionaries: SmallVec<Option<Interned>>,
    /// With a single key, a string, each string's group plus one, 0 until it
    /// has one, at its number in the run's pool plus one, and the null
    /// string's at 0: so a row's group is found without hashing again.
    number_groups: SmallVec<u32>,
    /// Whether a key not found gets no group, `NONE`, rather than a new
    /// one: a join's build side, once built.
    frozen: bool,
    /// Whether it freezes, rather than fail, when it can't grow: a grouping
    /// that spills the rows of keys it hasn't.
    #[cfg(feature = "spill")]
    freezes_full: bool,
    /// The bytes each group takes outside it, which its owner allocates as
    /// groups are made, so room for them is checked for too.
    #[cfg(feature = "spill")]
    group_extra: usize,
}

/// No group.
pub(crate) const NONE: u32 = u32::MAX;

/// A string key's dictionary, and the words of its entries: kept while
/// batches share it, as a chunk's do, so each entry is interned once.
struct Interned {
    /// A column of the dictionary, kept so its values aren't freed, and
    /// their address reused, while the words are.
    dictionary: ColumnView,
    /// Each entry's word, or 0 until a row with it is found.
    words: Array<u64>,
}

impl GroupTable {
    /// A table of `keys` keys, those in `string_keys` strings and those in
    /// `nullable_keys` possibly null, with `payload_words` zeroed words after
    /// each group's keys for the caller to store per-group data.
    pub(crate) fn new(
        env: &Env,
        keys: usize,
        string_keys: u64,
        nullable_keys: u64,
        payload_words: usize,
    ) -> Result<GroupTable, AllocError> {
        check!(keys <= KEYS_MAX);
        // A null string is a key of its own, so only other keys need the word.
        let all = if keys == KEYS_MAX { u64::MAX } else { (1 << keys) - 1 };
        let nullable = nullable_keys & !string_keys & all;
        let first = usize::from(nullable != 0);
        let stride = first + keys + payload_words;
        Ok(GroupTable {
            slots: Array::zeroed(env, 0)?,
            records: Blocks::new(env, stride)?,
            stride,
            first,
            nullable,
            keys,
            string_keys,
            dictionaries: SmallVec::fixed_from(env, (0..keys).map(|_| None))?,
            number_groups: SmallVec::new(env, 1 << 31)?,
            frozen: false,
            #[cfg(feature = "spill")]
            freezes_full: false,
            #[cfg(feature = "spill")]
            group_extra: 0,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.records.len()
    }

    /// Group `group`'s first payload word.
    pub(crate) fn payload(&self, group: u32) -> i64 {
        *at!(self.records.record(group as usize), self.first + self.keys)
    }

    /// Sets group `group`'s first payload word to `value`.
    pub(crate) fn set_payload(&mut self, group: u32, value: i64) {
        check!((group as usize) < self.len() && self.stride > self.first + self.keys);
        // SAFETY: the group has a record, of `stride` words, checked above.
        unsafe { *self.records.record_ptr(group as usize).add(self.first + self.keys) = value };
    }

    /// From now on, `find_or_create` finds groups only, and gives a key not
    /// found `NONE`.
    pub(crate) fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Where each block of records starts, as `Blocks::starts`, and where a
    /// record's payload starts in it.
    pub(crate) fn payloads(&mut self) -> (&[*mut i64], usize) {
        (self.records.starts(), self.first + self.keys)
    }

    /// How many words apart consecutive groups' payloads are.
    pub(crate) fn row_stride(&self) -> usize {
        self.stride
    }

    /// Writes the group of each row `select` keeps of `columns`, all if it's
    /// empty, to `groups`, with `columns` as the key. New keys get new
    /// groups. Each column is flat, or a dictionary or a constant for a string
    /// key, of strings for a string key, else of `Int64`s.
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
            self.records.reserve(context.env(), self.len() + count)?;
            let (interned, column) = (at_mut!(self.dictionaries, 0), at!(columns, 0));
            let words = load_string_words(context, interned, column, select, count)?;
            return self.find_numbered(at!(words.as_slice(), ..count), groups);
        }
        // Each key as a word a row, 0 where null, and a word a row of which
        // keys are null, unless none are.
        let mut copies = [const { None }; KEYS_MAX];
        let mut nulls: Option<Buffer> = None;
        for (k, (column, copy)) in columns.iter().zip(copies.iter_mut()).enumerate() {
            if self.string_keys >> k & 1 != 0 {
                let interned = at_mut!(self.dictionaries, k);
                *copy = Some(load_string_words(context, interned, column, select, count)?);
            } else {
                let nulls = if self.nullable >> k & 1 != 0 { Some(&mut nulls) } else { None };
                load_key_words(context, column, select, count, k, copy, nulls)?;
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
        // Only a table with null words takes keys that can be null.
        check!(self.first == 1 || nulls.is_none());
        // A frozen table makes no groups, so needs no room for them, once it
        // has slots to look in.
        let full = self.frozen && !self.slots.is_empty();
        let room = if full { Ok(()) } else { self.grow(context.env(), self.len() + count) };
        self.or_freeze(room)?;
        // Rows often come in runs of one key, as a session's do: then only
        // each run's first row is found, and the rest take its group.
        let mut starts = context.indices_buffer(count * 4)?;
        let runs = run_starts(keys, nulls, at_mut!(starts.as_mut_slice::<u32>(), ..count));
        if runs * 4 > count * 3 {
            return self.find_rows(context, keys, nulls, groups);
        }
        let starts = at!(starts.as_slice::<u32>(), ..runs);
        let mut firsts = [const { None }; KEYS_MAX];
        for (first, keys) in firsts.iter_mut().zip(keys) {
            let mut words = context.values_buffer(runs * 8)?;
            gather(words.as_mut_slice(), keys, starts);
            *first = Some(words);
        }
        let mut first_nulls = None;
        if let Some(nulls) = nulls {
            let mut words = context.values_buffer(runs * 8)?;
            gather(words.as_mut_slice(), as_signed(nulls), starts);
            first_nulls = Some(words);
        }
        let mut first_keys: [&[i64]; KEYS_MAX] = [&[]; KEYS_MAX];
        for (keys, first) in first_keys.iter_mut().zip(&firsts) {
            *keys = first.as_ref().map_or(&[], |first| at!(first.as_slice(), ..runs));
        }
        let first_nulls = first_nulls.as_ref().map(|nulls| at!(nulls.as_slice(), ..runs));
        let mut run_groups = context.indices_buffer(runs * 4)?;
        let run_groups = at_mut!(run_groups.as_mut_slice::<u32>(), ..runs);
        let first_keys = at!(first_keys, ..self.keys);
        self.find_rows(context, first_keys, first_nulls, run_groups)?;
        for (i, (&start, &group)) in starts.iter().zip(run_groups.iter()).enumerate() {
            let end = starts.get(i + 1).map_or(count, |&end| end as usize);
            at_mut!(groups, start as usize..end).fill(group);
        }
        Ok(())
    }

    /// As `find_or_create`, given each key as a word a row, and each row's
    /// null word unless none are null.
    fn find_rows(
        &mut self,
        context: &mut Context,
        keys: &[&[i64]],
        nulls: Option<&[u64]>,
        groups: &mut [u32],
    ) -> Result<(), Error> {
        let count = groups.len();
        let mut hashes = context.values_buffer(count * 8)?;
        let hashes = at_mut!(hashes.as_mut_slice(), ..count);
        hash_rows(hashes, keys, nulls);
        match (keys, nulls) {
            (&[a], None) => self.find_values(hashes, [a], groups),
            (&[a, b], None) => self.find_values(hashes, [a, b], groups),
            _ => self.find_hashed(hashes, keys, nulls, groups),
        }
    }

    /// As `find_or_create`, given each key as a word a row, and each row's
    /// null word unless none are null, and each row's hash.
    #[inline(never)]
    fn find_hashed(
        &mut self,
        hashes: &[u64],
        keys: &[&[i64]],
        nulls: Option<&[u64]>,
        groups: &mut [u32],
    ) -> Result<(), Error> {
        let count = groups.len();
        check!(keys.iter().all(|keys| keys.len() >= count));
        check!(nulls.is_none_or(|nulls| nulls.len() >= count));
        check!(hashes.len() == count);
        check!(self.stride > keys.len());
        // SAFETY: checked above.
        let null = |row: usize| nulls.map_or(0, |nulls| unsafe { *nulls.get_unchecked(row) });
        let ahead = cfg!(target_arch = "x86_64") && self.slots.len() > CACHED_SLOTS;
        let hash_of = |i: usize| hashes.get(i).copied();
        for (row, group) in groups.iter_mut().enumerate() {
            if ahead {
                fetch_ahead(&self.slots, self.records.starts(), self.stride, hash_of, row);
            }
            let hash = *at!(hashes, row);
            let first = self.first;
            // SAFETY: a group in a slot has a record, and each key a word for
            // each row, checked above.
            let equal = |group: usize| unsafe {
                let stored = self.records.record_ptr(group).cast_const();
                (first == 0 || (*stored).cast_unsigned() == null(row))
                    && keys
                        .iter()
                        .enumerate()
                        .all(|(k, keys)| *stored.add(first + k) == *keys.get_unchecked(row))
            };
            *group = match probe(&self.slots, hash, equal) {
                Ok(group) => group,
                Err(_) if self.frozen => NONE,
                Err(at) => self.add_row(at, hash, keys, row, null(row))?,
            };
        }
        Ok(())
    }

    /// As `find_or_create`, for a single key, a string, given each row's
    /// word in the run's pool: a string's group is found by its number,
    /// without hashing it again.
    #[inline(never)]
    fn find_numbered(&mut self, words: &[i64], groups: &mut [u32]) -> Result<(), Error> {
        // `NO_NUMBER` plus one wraps to 0, the null string's.
        let place = |word: i64| ((word.cast_unsigned() >> 32) as u32).wrapping_add(1) as usize;
        let most = words.iter().map(|&word| place(word)).max().unwrap_or(0);
        // Other tables share the pool, so numbers can be skipped. Saturating,
        // as on 32-bit targets `most` can be `usize::MAX`: past the cap,
        // extending fails.
        let len = most.saturating_add(1);
        if len > self.number_groups.len() {
            self.number_groups.extend_zeroed(len - self.number_groups.len())?;
        }
        for (&word, group) in words.iter().zip(groups.iter_mut()) {
            let place = place(word);
            *group = match at!(self.number_groups, place).checked_sub(1) {
                Some(known) => known,
                None if self.frozen => NONE,
                None => self.add_numbered(place, word)?,
            };
        }
        Ok(())
    }

    /// Adds a group for the string with word `word`, whose place in
    /// `number_groups` is `place`, and returns its number.
    fn add_numbered(&mut self, place: usize, word: i64) -> Result<u32, Error> {
        // A string key is never null, so there's no null word.
        let group = push(&mut self.records, 0, [word])?;
        *at_mut!(self.number_groups, place) = group + 1;
        Ok(group)
    }

    /// As `find_or_create`, for `N` keys with no nulls, given each row's
    /// hash: compares the key values directly. `N` is a constant, so the
    /// loop compares them in registers.
    fn find_values<const N: usize>(
        &mut self,
        hashes: &[u64],
        keys: [&[i64]; N],
        groups: &mut [u32],
    ) -> Result<(), Error> {
        let first = self.first;
        check!(self.stride >= first + N);
        // Masks a record's first word to its null bits, if it has them.
        let nulls = if first == 0 { 0 } else { -1 };
        let count = groups.len();
        check!(hashes.len() == count && keys.iter().all(|keys| keys.len() >= count));
        // `prefetch` does nothing on other targets, so they don't look ahead.
        let ahead = cfg!(target_arch = "x86_64") && self.slots.len() > CACHED_SLOTS;
        let hash_of = |i: usize| hashes.get(i).copied();
        // Locals, so the loop keeps them in registers across its writes.
        let GroupTable { slots, records, stride, frozen, .. } = self;
        let (slots, stride): (&mut [u64], usize) = (slots, *stride);
        // `grow` made room for every row, so no block moves while adding.
        let starts: *const [*mut i64] = records.starts();
        for (i, (&hash, group)) in hashes.iter().zip(groups).enumerate() {
            // SAFETY: as above.
            let starts = unsafe { &*starts };
            if ahead {
                fetch_ahead(slots, starts, stride, hash_of, i);
            }
            // A stored group is its null bits, zero here, if it has them, then
            // its key.
            // SAFETY: each key has a word for each row, checked above.
            let key = keys.map(|keys| unsafe { *keys.get_unchecked(i) });
            // SAFETY: a group in a slot has a record, with its keys.
            let equal = |group: usize| unsafe {
                let stored = address(starts, stride, group).cast_const();
                *stored & nulls == 0 && *stored.add(first).cast::<[i64; N]>() == key
            };
            *group = match probe(slots, hash, equal) {
                Ok(group) => group,
                Err(_) if *frozen => NONE,
                Err(at) => {
                    let group = push(records, first, key)?;
                    *at_mut!(slots, at) = salt(hash) | (u64::from(group) + 1);
                    group
                }
            };
        }
        Ok(())
    }

    /// Adds a group for row `row` of `keys`, with null word `null`, hashed as
    /// `hash`, in slot `at`, and returns its number.
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
        let first = self.first;
        let record = self.records.push();
        if first == 1 {
            *at_mut!(record, 0) = null.cast_signed();
        }
        for (word, keys) in at_mut!(record, first..).iter_mut().zip(keys) {
            *word = *at!(keys, row);
        }
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let group = group as u32;
        *at_mut!(self.slots, at) = salt(hash) | (u64::from(group) + 1);
        Ok(group)
    }

    /// `room`, the result of making room for more groups, unless it ran out
    /// of memory and the table freezes when full and has groups: then it
    /// freezes instead.
    #[inline]
    #[cfg_attr(not(feature = "spill"), expect(clippy::unused_self, reason = "it freezes to spill"))]
    fn or_freeze(&mut self, room: Result<(), Error>) -> Result<(), Error> {
        match room {
            #[cfg(feature = "spill")]
            Err(Error::OutOfMemory) if self.freezes_full && self.len() > 0 => {
                self.frozen = true;
                Ok(())
            }
            room => room,
        }
    }

    /// Fails, if it freezes when full, unless `bytes` more leave `env` room,
    /// so it freezes while what runs after it still has some.
    #[cfg(feature = "spill")]
    fn check_room(&self, env: &Env, bytes: usize) -> Result<(), Error> {
        if self.freezes_full && !env.leaves_room(bytes) {
            return Err(Error::OutOfMemory);
        }
        Ok(())
    }

    /// Makes it freeze, rather than fail, when it can't grow, its owner
    /// taking `group_extra` bytes a group besides.
    #[cfg(feature = "spill")]
    pub(crate) fn freeze_when_full(&mut self, group_extra: usize) {
        self.freezes_full = true;
        self.group_extra = group_extra;
    }

    /// Whether a key not found gets no group.
    #[cfg(feature = "spill")]
    pub(crate) fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Doubles the slots until at most two thirds are used by `groups`
    /// groups, and puts each group back in its slot.
    fn grow(&mut self, env: &Env, groups: usize) -> Result<(), Error> {
        let records = groups.min(GROUPS_MAX);
        #[cfg(feature = "spill")]
        // What the owner takes for groups is at most this, as it grows as
        // the records do.
        #[cfg(feature = "spill")]
        self.check_room(
            env,
            self.records.reserve_bytes(records) + records.next_power_of_two() * self.group_extra,
        )?;
        self.records.reserve(env, records)?;
        let mut capacity =
            if self.slots.is_empty() { 2 * BATCH_ROWS_MAX as usize } else { self.slots.len() };
        while groups * 3 > capacity * 2 {
            capacity *= 2;
        }
        if capacity == self.slots.len() {
            return Ok(());
        }
        #[cfg(feature = "spill")]
        self.check_room(env, capacity * 8)?;
        let mut slots = Array::zeroed(env, capacity)?;
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

    /// Key `k` of `group`, an integer key's value.
    #[cfg(feature = "spill")]
    pub(crate) fn key(&self, group: u32, k: usize) -> i64 {
        *at!(self.records.record(group as usize), self.first + k)
    }

    /// Key `k` of the groups in `range`, as a column of its type.
    pub(crate) fn write_keys(
        &self,
        context: &mut Context,
        k: usize,
        range: core::ops::Range<usize>,
    ) -> Result<ColumnView, Error> {
        let rows = range.len();
        let mut values = if rows > BATCH_ROWS_MAX as usize {
            context.bytes_buffer(rows * 8)?
        } else {
            context.values_buffer(rows * 8)?
        };
        let out = values.as_mut_slice::<i64>().iter_mut();
        let mut nulls = false;
        for (out, group) in out.zip(range.clone()) {
            let record = self.records.record(group);
            *out = *at!(record, self.first + k);
            nulls |= self.first == 1 && *at!(record, 0) >> k & 1 != 0;
        }
        if self.string_keys >> k & 1 != 0 {
            return write_strings(context, at!(values.as_slice::<i64>(), ..rows));
        }
        if !nulls {
            return Ok(ColumnView::new(context, DataType::Int64, values, None)?);
        }
        let mut validity = context.small_buffer(BitSlice::<u8>::words_for(rows))?;
        let starts = range.clone().step_by(8);
        for (bits, start) in validity.as_mut_slice::<u8>().iter_mut().zip(starts) {
            let groups = (start..range.end.min(start + 8)).enumerate();
            *bits = groups.fold(0, |bits, (i, group)| {
                bits | u8::from(*at!(self.records.record(group), 0) >> k & 1 == 0) << i
            });
        }
        Ok(ColumnView::new(context, DataType::Int64, values, Some(validity))?)
    }
}

/// Loads `count` rows that `select` keeps of string key column `column`, flat,
/// a dictionary or a constant, all if `select` is empty, as their words in
/// the run's string pool. A dictionary's entries are each interned once, when
/// a row with it is first found, and their words kept in `interned` while
/// batches share the dictionary. A constant is interned once a batch.
fn load_string_words(
    context: &mut Context,
    interned: &mut Option<Interned>,
    column: &ColumnView,
    select: &[u16],
    count: usize,
) -> Result<Buffer, Error> {
    check!(column.data_type() == DataType::String);
    let values = column.values();
    let strings = values.strings();
    let mut words = context.values_buffer(count * 8)?;
    let env = context.env();
    let pool = context.string_pool()?;
    let row_of = |i: usize| if select.is_empty() { i } else { usize::from(*at!(select, i)) };
    let out = words.as_mut_slice::<u64>();
    match column.form() {
        Form::Flat => {
            let validity = column.validity();
            let string = |i: usize| {
                let row = row_of(i);
                validity.is_none_or(|validity| is_valid(validity, row)).then(|| strings.get(row))
            };
            pool.intern_strings(env, count, string, |i, id, number| {
                *at_mut!(out, i) = word(id, number);
            })?;
        }
        Form::Dictionary(indices) => {
            if !interned.as_ref().is_some_and(|interned| interned.dictionary.shares_values(column))
            {
                let words = Array::zeroed(env, values.len() as usize)?;
                *interned = Some(Interned { dictionary: column.clone(), words });
            }
            let Some(Interned { words: entries, .. }) = interned else { failed!() };
            for (i, out) in at_mut!(out, ..count).iter_mut().enumerate() {
                let entry = *at!(indices, row_of(i));
                let known = at_mut!(entries, entry as usize);
                if *known == 0 {
                    *known = if values.is_null(entry) {
                        word(Id::NULL, NO_NUMBER)
                    } else {
                        let (id, number) =
                            pool.intern_numbered(env, strings.get(entry as usize))?;
                        word(id, number)
                    };
                }
                *out = *known;
            }
        }
        Form::Constant => {
            let known = if values.is_null(0) {
                word(Id::NULL, NO_NUMBER)
            } else {
                let (id, number) = pool.intern_numbered(env, strings.get(0))?;
                word(id, number)
            };
            at_mut!(out, ..count).fill(known);
        }
        Form::Sequence(_) => failed!(),
    }
    Ok(words)
}

/// A string's word: its number in the run's pool above its id, so the word
/// is a key on its own, and its low half is its id. Never 0.
fn word(id: Id, number: u32) -> u64 {
    u64::from(id.raw_id()) | u64::from(number) << 32
}

/// The strings of `ids`, ids in the run's pool in their low halves, 0 for
/// null, as a string column: of more rows than a batch's, its buffers are
/// bytes.
#[inline(never)]
pub(crate) fn write_strings(context: &mut Context, ids: &[i64]) -> Result<ColumnView, Error> {
    let rows = ids.len();
    let large = rows > BATCH_ROWS_MAX as usize;
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "an id")]
    let id = |row: usize| Id::raw(*at!(ids, row) as u32);
    let mut validity = None;
    if (0..rows).any(|row| id(row).is_null()) {
        let size = BitSlice::<u8>::words_for(rows);
        let mut bits =
            if large { context.bytes_buffer(size)? } else { context.small_buffer(size)? };
        for (start, bits) in bits.as_mut_slice::<u8>().iter_mut().enumerate() {
            let row = start * 8;
            let valid = |i: usize| u8::from(row + i < rows && !id(row + i).is_null()) << i;
            *bits = (0..8).fold(0, |bits, i| bits | valid(i));
        }
        validity = Some(bits);
    }
    let mut views =
        if large { context.bytes_buffer(rows * 8)? } else { context.values_buffer(rows * 8)? };
    let (views_out, _) = views.as_mut_slice::<u32>().as_chunks_mut::<2>();
    let pool = context.string_pool()?;
    let mut bytes = 0_u32;
    for (row, view) in views_out.iter_mut().enumerate() {
        let len = pool.get(id(row)).map_or(0, <[u8]>::len);
        let len = u32::try_from(len).map_err(|_| Error::OutOfMemory)?;
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
    Ok(ColumnView::strings(context, views, stored, validity)?)
}

/// Loads `count` rows that `select` keeps of key column `column`, all if
/// it's empty, as words: copied to `copy` unless every row is kept and none
/// is null, and 0 where null. Sets bit `k` of each null row's word in
/// `nulls`, made when the first column with nulls is, unless it's `None`, for
/// a key that can't be null: then its validity, if any, isn't read.
fn load_key_words(
    context: &mut Context,
    column: &ColumnView,
    select: &[u16],
    count: usize,
    k: usize,
    copy: &mut Option<Buffer>,
    nulls: Option<&mut Option<Buffer>>,
) -> Result<(), Error> {
    let values = column.int64s();
    // A key that can't be null is read without its validity, if it has one.
    let (Some(validity), Some(nulls)) = (column.validity(), nulls) else {
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

/// Appends to `records` a group of keys `key` that aren't null, from word
/// `first`, and returns its number.
fn push<const N: usize>(
    records: &mut Blocks<i64>,
    first: usize,
    key: [i64; N],
) -> Result<u32, Error> {
    let group = records.len();
    if group == GROUPS_MAX {
        return Err(Error::OutOfMemory);
    }
    // A record starts zeroed, so its null word says the keys aren't null.
    at_mut!(records.push(), first..first + N).copy_from_slice(&key);
    #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
    Ok(group as u32)
}

/// Fetches the slot of the row `SLOT_AHEAD` rows after row `i`, and the
/// record of the group in the first slot of the row `RECORD_AHEAD` after,
/// given each row's hash by `hash_of`, and where each block of records of
/// `stride` words starts. A table too big for the cache misses it twice a
/// row, for its slot and then its group's record.
#[inline]
fn fetch_ahead(
    slots: &[u64],
    starts: &[*mut i64],
    stride: usize,
    hash_of: impl Fn(usize) -> Option<u64>,
    i: usize,
) {
    if let Some(next) = hash_of(i + SLOT_AHEAD) {
        prefetch(slots.as_ptr().wrapping_add(slot_of(next, slots.len())));
    }
    if let Some(next) = hash_of(i + RECORD_AHEAD) {
        let slot = *at!(slots, slot_of(next, slots.len()));
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let group = (slot & !salt(slot)) as usize;
        if group != 0 {
            // SAFETY: a slot holds its group plus one, and a group in a slot
            // has a record, so its block exists.
            prefetch(unsafe { address(starts, stride, group - 1) }.cast_const());
        }
    }
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

/// Writes the rows of `keys`, whose null words are `nulls` unless none are
/// null, that start a run of one key to `starts`, and returns how many there
/// are. Marks each row a key at a time, so each pass is a tight loop.
fn run_starts(keys: &[&[i64]], nulls: Option<&[u64]>, starts: &mut [u32]) -> usize {
    let count = starts.len();
    check!(count > 0);
    check!(keys.iter().all(|keys| keys.len() >= count));
    check!(nulls.is_none_or(|nulls| nulls.len() >= count));
    // First whether each row differs from the one before, in `starts`.
    starts.fill(0);
    let mut differ = |words: &[i64]| {
        let pairs = at!(words, ..count - 1).iter().zip(at!(words, 1..count));
        for (start, (before, word)) in at_mut!(starts, 1..).iter_mut().zip(pairs) {
            *start |= u32::from(before != word);
        }
    };
    for keys in keys {
        differ(keys);
    }
    if let Some(nulls) = nulls {
        differ(as_signed(nulls));
    }
    *at_mut!(starts, 0) = 1;
    let mut runs = 0;
    for row in 0..count {
        let start = *at!(starts, row) != 0;
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's rows")]
        let row_index = row as u32;
        // SAFETY: `runs` is at most `row`, which is below `count`.
        unsafe { *starts.get_unchecked_mut(runs) = row_index };
        runs += usize::from(start);
    }
    runs
}

/// `words` as signed words.
fn as_signed(words: &[u64]) -> &[i64] {
    // SAFETY: of one size and alignment, with every bit pattern valid.
    unsafe { core::slice::from_raw_parts(words.as_ptr().cast(), words.len()) }
}

/// Writes the words of `words` at `rows` to `out`.
fn gather(out: &mut [i64], words: &[i64], rows: &[u32]) {
    for (out, &row) in out.iter_mut().zip(rows) {
        *out = *at!(words, row as usize);
    }
}

/// The hash of each row of `keys`, whose null words are `nulls` unless none
/// are null, to `hashes`, as Perfetto's `GroupTable::Hash`: a key at a time,
/// so each pass is a tight loop.
fn hash_rows(hashes: &mut [u64], keys: &[&[i64]], nulls: Option<&[u64]>) {
    fn words(keys: &[i64]) -> impl Iterator<Item = u64> + '_ {
        keys.iter().map(|&key| key.cast_unsigned())
    }
    check!(keys.iter().all(|keys| keys.len() >= hashes.len()));
    check!(nulls.is_none_or(|nulls| nulls.len() >= hashes.len()));
    hashes.fill(SEED);
    for keys in keys {
        for (hash, key) in hashes.iter_mut().zip(words(keys)) {
            *hash = combine(*hash, key);
        }
    }
    for (hash, &null) in hashes.iter_mut().zip(nulls.unwrap_or_default()) {
        if null != 0 {
            *hash = combine(*hash, null);
        }
    }
}
