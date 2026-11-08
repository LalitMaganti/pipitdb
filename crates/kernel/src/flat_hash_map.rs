//! `FlatHashMap`: a map of keys to values, ported from Perfetto's
//! `FlatHashMapV2`, a Swiss table as absl's `flat_hash_map` is.
//!
//! Each slot has a control byte: free, a tombstone where a key was erased,
//! or the low 7 bits of its key's hash. A lookup compares a group of control
//! bytes at once with its key's, 16 with SSE and 8 in a word without, and
//! compares keys only where those match. The first `CLONES` control bytes
//! are copied after the last, so a group read near the end needn't wrap.

use core::alloc::Layout;
use core::marker::PhantomData;
use core::mem;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, Buffer};
use crate::murmur_hash;

/// A key: compared, and hashed as Perfetto's `MurmurHash` hashes it.
pub trait Key: Copy + Eq {
    fn hash(&self) -> u64;
}

impl Key for i64 {
    #[inline]
    fn hash(&self) -> u64 {
        murmur_hash::mix(self.cast_unsigned())
    }
}

impl Key for &[u8] {
    #[inline]
    fn hash(&self) -> u64 {
        murmur_hash::bytes(self)
    }
}

/// The control byte of a free slot.
const FREE_SLOT: u8 = 0x80;

/// The control byte of a slot whose key was erased.
const TOMBSTONE: u8 = 0xFE;

/// How full a table gets before it grows, as a percentage.
const LOAD_LIMIT_PERCENT: usize = 75;

/// The fewest slots a table has.
const CAPACITY_MIN: usize = 128;

/// How many control bytes are copied after the last.
const CLONES: usize = group::SIZE - 1;

/// Which slot a probe starts at, before masking.
#[inline]
fn h1(hash: u64) -> usize {
    (hash >> 7) as usize
}

/// What a used slot's control byte is.
#[inline]
fn h2(hash: u64) -> u8 {
    (hash & 0x7f) as u8
}

/// Positions in a group: each set bit of the mask, `STRIDE` bits apart.
struct Matches(group::Mask);

impl Iterator for Matches {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let at = self.0.trailing_zeros() as usize / group::STRIDE;
        self.0 &= self.0 - 1;
        Some(at)
    }
}

/// A group of 16 control bytes, compared with SSE, as Perfetto's x64 builds
/// do.
#[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
mod group {
    use core::arch::x86_64::{
        __m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8, _mm_sign_epi8,
    };

    use super::Matches;

    pub(super) const SIZE: usize = 16;
    pub(super) const STRIDE: usize = 1;
    pub(super) type Mask = u16;

    pub(super) struct Group(__m128i);

    impl Group {
        /// # Safety
        ///
        /// `ctrl` must have `SIZE` bytes from `at`.
        #[inline]
        pub(super) unsafe fn load(ctrl: &[u8], at: usize) -> Group {
            #[expect(clippy::cast_ptr_alignment, reason = "an unaligned load")]
            // SAFETY: as the caller promises, and the build has SSE2.
            Group(unsafe { _mm_loadu_si128(ctrl.as_ptr().add(at).cast::<__m128i>()) })
        }

        /// The bytes that are `h2`.
        #[inline]
        pub(super) fn matching(&self, h2: u8) -> Matches {
            // SAFETY: the build has SSE2.
            mask(unsafe { _mm_cmpeq_epi8(self.0, _mm_set1_epi8(h2.cast_signed())) })
        }

        /// The bytes of free slots: the only bytes `sign` leaves negative.
        #[inline]
        pub(super) fn free(&self) -> Matches {
            // SAFETY: the build has SSSE3.
            mask(unsafe { _mm_sign_epi8(self.0, self.0) })
        }

        /// The bytes of free slots and tombstones: those with their top bit
        /// set.
        #[inline]
        pub(super) fn free_or_tombstone(&self) -> Matches {
            mask(self.0)
        }
    }

    #[inline]
    fn mask(bytes: __m128i) -> Matches {
        // SAFETY: the build has SSE2.
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "16 bits")]
        Matches(unsafe { _mm_movemask_epi8(bytes) } as u16)
    }
}

/// A group of 8 control bytes, compared in a word, as Perfetto's builds
/// without SSE do.
#[cfg(not(all(target_arch = "x86_64", target_feature = "ssse3")))]
mod group {
    use super::Matches;

    pub(super) const SIZE: usize = 8;
    pub(super) const STRIDE: usize = 8;
    pub(super) type Mask = u64;

    const LSBS: u64 = 0x0101_0101_0101_0101;
    const MSBS: u64 = 0x8080_8080_8080_8080;

    /// The bytes, little-endian.
    pub(super) struct Group(u64);

    impl Group {
        /// # Safety
        ///
        /// `ctrl` must have `SIZE` bytes from `at`.
        #[inline]
        pub(super) unsafe fn load(ctrl: &[u8], at: usize) -> Group {
            // SAFETY: as the caller promises.
            let word = unsafe { ctrl.as_ptr().add(at).cast::<u64>().read_unaligned() };
            Group(u64::from_le(word))
        }

        /// The bytes that may be `h2`: maybe more, but never fewer.
        #[inline]
        pub(super) fn matching(&self, h2: u8) -> Matches {
            let x = self.0 ^ (LSBS * u64::from(h2));
            Matches(x.wrapping_sub(LSBS) & !x & MSBS)
        }

        /// The bytes of free slots: of those with their top bit set, the ones
        /// whose bit 1 is clear.
        #[inline]
        pub(super) fn free(&self) -> Matches {
            Matches(self.0 & !(self.0 << 6) & MSBS)
        }

        /// The bytes of free slots and tombstones: of those with their top
        /// bit set, the ones whose bit 0 is clear.
        #[inline]
        pub(super) fn free_or_tombstone(&self) -> Matches {
            Matches(self.0 & !(self.0 << 7) & MSBS)
        }
    }
}

use group::Group;

/// Asks for the cache line at `at` to be loaded, where the target can.
#[inline]
fn prefetch<T>(at: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch never faults, wherever it points.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(at.cast());
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = at;
}

/// What doesn't depend on a map's key and value types, compiled once for
/// every map, as Perfetto's `FlatHashMapV2Base` is: its memory, its counts,
/// and finding free slots.
struct Base {
    /// A control byte for each slot, then copies of the first `CLONES`, then
    /// the slots: owned here, and reached through `ctrl` and `slots`.
    #[expect(dead_code, reason = "owns the memory")]
    storage: Buffer,
    /// Where the control bytes start in `storage`, kept as Perfetto keeps
    /// `ctrl_`, so lookups needn't find them.
    ctrl: NonNull<u8>,
    /// Where the slots start in `storage`.
    slots: NonNull<u8>,
    /// How many slots there are: a power of two.
    capacity: usize,
    /// How many keys there are.
    len: usize,
    /// How many more free slots can be filled before it grows.
    growth_left: usize,
    /// Whether any slot is a tombstone.
    has_tombstones: bool,
}

impl Base {
    /// An empty table of `capacity` slots, a power of two, or of 128 if more,
    /// each laid out as `slot`. Out of line, as is all of `Base` that isn't
    /// on a lookup's path, as Perfetto's is in `flat_hash_map.cc`.
    #[inline(never)]
    fn new(allocator: &dyn Allocator, capacity: usize, slot: Layout) -> Result<Base, AllocError> {
        check!(capacity.is_power_of_two() && slot.align() <= BUFFER_ALIGNMENT_BYTES);
        let capacity = capacity.max(CAPACITY_MIN);
        let slots_at = (capacity + CLONES).next_multiple_of(slot.align());
        let bytes = slot.size().checked_mul(capacity).and_then(|size| size.checked_add(slots_at));
        // SAFETY: the control bytes are written below, and a slot only once
        // a key is put in it, before it's read.
        let storage = unsafe { Buffer::allocate_uninit(allocator, bytes.ok_or(AllocError)?)? };
        let ctrl = storage.as_non_null();
        // SAFETY: within the storage.
        let slots = unsafe { ctrl.add(slots_at) };
        let mut base =
            Base { storage, ctrl, slots, capacity, len: 0, growth_left: 0, has_tombstones: false };
        base.reset();
        Ok(base)
    }

    /// Empties it, keeping its memory.
    #[inline(never)]
    fn reset(&mut self) {
        let capacity = self.capacity;
        // SAFETY: the control bytes are within the storage.
        unsafe { self.ctrl.write_bytes(FREE_SLOT, capacity + CLONES) };
        self.len = 0;
        self.growth_left = capacity * LOAD_LIMIT_PERCENT / 100;
        self.has_tombstones = false;
    }

    /// The control bytes, with the copies after the last.
    #[inline]
    fn ctrl(&self) -> &[u8] {
        // SAFETY: the control bytes are written when it's made.
        unsafe { core::slice::from_raw_parts(self.ctrl.as_ptr(), self.capacity + CLONES) }
    }

    /// Where slot `at` is, laid out as `T`.
    #[inline]
    fn slot<T>(&self, at: usize) -> *mut T {
        // SAFETY: slots are within the storage, and `at` is masked by callers.
        unsafe { self.slots.as_ptr().cast::<T>().add(at) }
    }

    /// The first free slot or tombstone the probe for `hash` reaches.
    #[inline(never)]
    fn first_free_or_tombstone(&self, hash: u64) -> usize {
        self.first_free_or_tombstone_inline(hash)
    }

    #[inline]
    fn first_free_or_tombstone_inline(&self, hash: u64) -> usize {
        let mask = self.capacity - 1;
        let mut offset = h1(hash) & mask;
        let mut probe = 0;
        loop {
            // SAFETY: as in `FlatHashMap::find_slot`.
            let group = unsafe { Group::load(self.ctrl(), offset) };
            if let Some(i) = group.free_or_tombstone().next() {
                return (offset + i) & mask;
            }
            probe += group::SIZE;
            offset = (offset + probe) & mask;
        }
    }

    /// Sets slot `at`'s control byte, and its copy, if it has one.
    #[inline]
    fn set_ctrl(&mut self, at: usize, ctrl: u8) {
        check!(at < self.capacity);
        // SAFETY: a slot's control byte, and its copy, within the storage.
        unsafe {
            self.ctrl.add(at).write(ctrl);
            if at < CLONES {
                self.ctrl.add(self.capacity + at).write(ctrl);
            }
        }
    }

    /// Replaces this with an empty table of twice the slots, laid out as
    /// `slot`, and returns the old one, whose keys the caller moves over.
    #[inline(never)]
    fn grow(&mut self, allocator: &dyn Allocator, slot: Layout) -> Result<Base, AllocError> {
        let grown = Base::new(allocator, self.capacity * 2, slot)?;
        Ok(mem::replace(self, grown))
    }

    /// Counts the `len` keys moved over by `grow`.
    fn finish_grow(&mut self, len: usize) {
        check!(self.growth_left >= len);
        self.len = len;
        self.growth_left -= len;
    }
}

/// Whether a control byte is a used slot's.
#[inline]
fn is_used(ctrl: u8) -> bool {
    ctrl != FREE_SLOT && ctrl != TOMBSTONE
}

/// Keys and values are copied in and out, never dropped.
pub struct FlatHashMap<K, V> {
    base: Base,
    slots: PhantomData<(K, V)>,
}

impl<K: Key, V: Copy> FlatHashMap<K, V> {
    /// How a slot is laid out.
    const SLOT: Layout = Layout::new::<(K, V)>();

    /// An empty map of `capacity` slots, a power of two, or 128 if more.
    pub fn new(allocator: &dyn Allocator, capacity: usize) -> Result<Self, AllocError> {
        Ok(FlatHashMap { base: Base::new(allocator, capacity, Self::SLOT)?, slots: PhantomData })
    }

    /// An empty map with room for `len` keys before it grows.
    pub fn for_len(allocator: &dyn Allocator, len: usize) -> Result<Self, AllocError> {
        FlatHashMap::new(allocator, (len * 100 / LOAD_LIMIT_PERCENT + 1).next_power_of_two())
    }

    /// How many keys it has.
    pub fn len(&self) -> usize {
        self.base.len
    }

    pub fn is_empty(&self) -> bool {
        self.base.len == 0
    }

    /// How many slots it has.
    pub fn capacity(&self) -> usize {
        self.base.capacity
    }

    /// The value of `key`, if it has it. Out of line, as Perfetto's is, so
    /// each call site is a call: it's compiled once for each map type.
    #[inline(never)]
    pub fn find(&self, key: K) -> Option<&V> {
        let at = self.find_slot::<false>(key.hash(), |&held| held == key).ok()?;
        Some(&self.slot(at).1)
    }

    /// The value of the key hashed as `hash` that `equal` is true for, if it
    /// has one: for finding keys by what they stand for, as Perfetto's maps
    /// do with a transparent hasher.
    #[inline(never)]
    pub fn find_by(&self, hash: u64, equal: impl Fn(&K) -> bool) -> Option<&V> {
        let at = self.find_slot::<false>(hash, equal).ok()?;
        Some(&self.slot(at).1)
    }

    /// Adds `key` with `value`, unless it has it. Returns its value, and
    /// whether it was added. Grows, allocating from `allocator`, if full.
    #[inline]
    pub fn insert(
        &mut self,
        allocator: &dyn Allocator,
        key: K,
        value: V,
    ) -> Result<(&mut V, bool), AllocError> {
        let (at, added) = self.find_or_prepare_insert(allocator, key)?;
        let slot = self.base.slot::<(K, V)>(at);
        if added {
            // SAFETY: a slot of this map, claimed for `key`.
            unsafe { slot.write((key, value)) };
        }
        // SAFETY: the slot holds a key and value, and the map is borrowed
        // mutably for as long as the value is.
        Ok((unsafe { &mut (*slot).1 }, added))
    }

    /// Removes `key`, returning whether it had it.
    #[inline(never)]
    pub fn erase(&mut self, key: K) -> bool {
        let Ok(at) = self.find_slot::<false>(key.hash(), |&held| held == key) else {
            return false;
        };
        self.base.set_ctrl(at, TOMBSTONE);
        self.base.len -= 1;
        self.base.has_tombstones = true;
        true
    }

    /// Removes every key, keeping the memory.
    pub fn clear(&mut self) {
        self.base.reset();
    }

    /// Its keys and values, in no order.
    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        let used = (0..self.capacity()).filter(|&at| is_used(*at!(self.base.ctrl(), at)));
        used.map(|at| self.slot(at))
    }

    /// Slot `at`, which holds a key.
    #[inline]
    fn slot(&self, at: usize) -> &(K, V) {
        // SAFETY: a used slot holds a key and value.
        unsafe { &*self.base.slot::<(K, V)>(at) }
    }

    /// The slot holding the key hashed as `hash` that `equal` is true for,
    /// or else, if `FOR_INSERT`, the first free slot its probe reaches.
    /// Tombstones are passed over.
    #[inline]
    fn find_slot<const FOR_INSERT: bool>(
        &self,
        hash: u64,
        equal: impl Fn(&K) -> bool,
    ) -> Result<usize, usize> {
        let ctrl = self.base.ctrl();
        let mask = self.capacity() - 1;
        let mut offset = h1(hash) & mask;
        let mut probe = 0;
        prefetch(ctrl.as_ptr().wrapping_add(offset));
        loop {
            prefetch(self.base.slot::<(K, V)>(offset));
            // SAFETY: `offset` is a slot, and the control bytes have a group
            // from each, with the copies after the last.
            let group = unsafe { Group::load(ctrl, offset) };
            // Every byte that matches is a used slot's: a free slot's and a
            // tombstone's have their top bit set, and `h2`'s doesn't.
            for i in group.matching(h2(hash)) {
                let at = (offset + i) & mask;
                if equal(&self.slot(at).0) {
                    return Ok(at);
                }
            }
            if let Some(i) = group.free().next() {
                return Err(if FOR_INSERT { (offset + i) & mask } else { 0 });
            }
            // The load limit leaves free slots, which ends every probe.
            probe += group::SIZE;
            offset = (offset + probe) & mask;
        }
    }

    /// The slot holding `key`, and false, or else a slot claimed for it, and
    /// true. Out of line, as Perfetto's is: compiled once for each map type.
    #[inline(never)]
    fn find_or_prepare_insert(
        &mut self,
        allocator: &dyn Allocator,
        key: K,
    ) -> Result<(usize, bool), AllocError> {
        let hash = key.hash();
        let mut at = match self.find_slot::<true>(hash, |&held| held == key) {
            Ok(at) => return Ok((at, false)),
            Err(at) => at,
        };
        if self.base.growth_left == 0 {
            self.grow_and_rehash(allocator)?;
            at = self.base.first_free_or_tombstone(hash);
        }
        let mut free = true;
        if self.base.has_tombstones {
            at = self.base.first_free_or_tombstone(hash);
            free = *at!(self.base.ctrl(), at) != TOMBSTONE;
        }
        self.base.set_ctrl(at, h2(hash));
        self.base.len += 1;
        if free {
            self.base.growth_left -= 1;
        }
        Ok((at, true))
    }

    /// Moves the keys to a table twice the size, which has no tombstones.
    /// Allocating it is shared by every map, in `Base::grow`: only moving the
    /// keys over is compiled for each map type.
    #[inline(never)]
    fn grow_and_rehash(&mut self, allocator: &dyn Allocator) -> Result<(), AllocError> {
        let old = self.base.grow(allocator, Self::SLOT)?;
        for at in 0..old.capacity {
            if is_used(*at!(old.ctrl(), at)) {
                // SAFETY: a used slot holds a key and value.
                let slot = unsafe { old.slot::<(K, V)>(at).read() };
                let hash = slot.0.hash();
                let to = self.base.first_free_or_tombstone_inline(hash);
                // SAFETY: a free slot of this map.
                unsafe { self.base.slot::<(K, V)>(to).write(slot) };
                self.base.set_ctrl(to, h2(hash));
            }
        }
        self.base.finish_grow(old.len);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::{Heap, LimitAllocator};

    #[test]
    fn finds_what_it_holds_as_it_grows() {
        let mut map = FlatHashMap::<i64, i64>::new(&Heap, 1).unwrap();
        // Enough to grow several times, with keys whose low bits are the same.
        for key in (-500..500).map(|i| i * 1024) {
            assert_eq!(map.insert(&Heap, key, key + 1).unwrap(), (&mut (key + 1), true));
        }
        assert_eq!(map.insert(&Heap, 0, 7).unwrap(), (&mut 1, false));
        assert_eq!(map.len(), 1000);
        assert!((-500..500).all(|i| map.find(i * 1024) == Some(&(i * 1024 + 1))));
        assert!((1..1000).all(|i| map.find(i).is_none()));
        assert_eq!(map.find_by(Key::hash(&1024), |&key| key == 1024), Some(&1025));
        assert!(map.find_by(Key::hash(&1024), |&key| key == 2048).is_none());
        assert_eq!(map.iter().map(|&(key, _)| key).sum::<i64>(), -500 * 1024);
    }

    #[test]
    fn reuses_the_slots_of_erased_keys() {
        let mut map = FlatHashMap::<i64, ()>::new(&Heap, 128).unwrap();
        // Many times what fits, but never more than a few at once. As in
        // Perfetto's, erasing doesn't give back room to grow into: tombstones
        // are filled again only where a probe reaches them first.
        for key in 0..10_000 {
            map.insert(&Heap, key, ()).unwrap();
            if key >= 8 {
                assert!(map.erase(key - 8));
            }
        }
        assert!(!map.erase(0));
        assert_eq!(map.len(), 8);
        assert!((9992..10_000).all(|key| map.find(key).is_some()));
        assert!(map.find(9991).is_none());
        map.clear();
        assert!(map.is_empty() && map.find(9999).is_none());
    }

    #[test]
    fn keys_strings_by_their_bytes() {
        let owned = b"ab".to_vec();
        let mut map = FlatHashMap::<&[u8], ()>::for_len(&Heap, 4).unwrap();
        for key in [&b""[..], &b"a"[..], &b"ab"[..], &owned[..]] {
            map.insert(&Heap, key, ()).unwrap();
        }
        assert_eq!(map.len(), 3);
        assert!(map.find(b"ab").is_some() && map.find(b"").is_some());
        assert!(map.find(b"b").is_none());
    }

    #[test]
    fn fails_to_grow_past_its_allocator() {
        let limited = LimitAllocator::new(&Heap, 4096);
        let mut map = FlatHashMap::<i64, ()>::new(&limited, 128).unwrap();
        assert!((0..1000).any(|key| map.insert(&limited, key, ()).is_err()));
        // What it held before is still there.
        let held: Vec<i64> = map.iter().map(|&(key, ())| key).collect();
        assert!(!held.is_empty() && held.iter().all(|&key| map.find(key).is_some()));
    }
}
