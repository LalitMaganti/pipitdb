//! `StringPool`: interns strings, handing out compact ids that give each one
//! back in O(1), ported from Perfetto's.

use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::flat_hash_map::{FlatHashMap, Key, prefetch};
use crate::murmur_hash;
use crate::row_batch::BATCH_ROWS_MAX;
use crate::slow_vec::SlowVec;

/// How many bits of an id say which block its string is in.
const BLOCK_INDEX_BITS: u32 = 9;

/// How many bits of an id say where in its block its string is.
const BLOCK_OFFSET_BITS: u32 = 22;

const LARGE_STRING_FLAG_MASK: u32 = 1 << 31;
const BLOCK_OFFSET_MASK: u32 = (1 << BLOCK_OFFSET_BITS) - 1;
const BLOCK_INDEX_MASK: u32 = !LARGE_STRING_FLAG_MASK & !BLOCK_OFFSET_MASK;

/// 4 MB.
const BLOCK_BYTES: usize = BLOCK_OFFSET_MASK as usize + 1;

const BLOCKS_MAX: usize = 1 << BLOCK_INDEX_BITS;

/// Strings this long or longer aren't put in a block, but stored apart.
const LARGE_STRING_BYTES_MIN: usize = BLOCK_BYTES / 4;

/// The bytes a string in a block takes besides its own: its size, and a nul
/// after it.
const METADATA_BYTES: usize = 5;

/// The most slots an index has while it's taken to fit in cache: 512 KB of
/// them, as a core's L2.
const CACHED_SLOTS: usize = 1 << 15;

/// How many strings a batch is interned in at a time: each step's loads
/// are started for all of them before the next step waits for any.
const CHUNK_ROWS: usize = 512;

/// How many slots the index starts with.
const INDEX_CAPACITY: usize = 4096;

/// A string's id. If its top bit is set, the rest are an index into the
/// large strings. Otherwise the next 9 bits are the index of the block it's
/// in, and the last 22 bits where in the block it starts:
///
/// ```text
/// [31] [30:22] [21:0]
///  |      |       +---- offset in block (or LSB of large string index).
///  |      +------------ block index (or MSB of large string index).
///  +------------------- 1: large string, 0: string in a block.
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Id(u32);

impl Id {
    /// The null string's id.
    pub const NULL: Id = Id(0);

    #[inline]
    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn is_large_string(self) -> bool {
        self.0 & LARGE_STRING_FLAG_MASK != 0
    }

    #[inline]
    pub fn block_offset(self) -> u32 {
        self.0 & BLOCK_OFFSET_MASK
    }

    #[inline]
    pub fn block_index(self) -> u32 {
        (self.0 & BLOCK_INDEX_MASK) >> BLOCK_OFFSET_BITS
    }

    #[inline]
    pub fn large_string_index(self) -> u32 {
        check!(self.is_large_string());
        self.0 & !LARGE_STRING_FLAG_MASK
    }

    #[inline]
    pub fn raw_id(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn raw(raw: u32) -> Id {
        Id(raw)
    }

    fn large_string(index: usize) -> Id {
        let index = u32::try_from(index).unwrap_or(u32::MAX);
        check!(index & LARGE_STRING_FLAG_MASK == 0);
        Id(LARGE_STRING_FLAG_MASK | index)
    }

    #[inline]
    fn block_string(index: usize, offset: u32) -> Id {
        #[expect(clippy::cast_possible_truncation, reason = "below `BLOCKS_MAX`")]
        let index = index as u32;
        Id(!LARGE_STRING_FLAG_MASK & (index << BLOCK_OFFSET_BITS | offset & BLOCK_OFFSET_MASK))
    }
}

pub struct StringPool {
    /// Where the strings are.
    store: Store,
    /// The id of each string, found by its hash and then its bytes.
    string_index: FlatHashMap<Interned, ()>,
}

/// Where a pool's strings are, apart from its index, so a string can be
/// stored while its slot in the index is held.
struct Store {
    /// The blocks, of `BLOCK_BYTES` each, written up to `block_end` in the
    /// last.
    blocks: SlowVec<Buffer>,
    /// How many bytes of the last block are written.
    block_end: usize,
    /// Each string too large for a block, with a nul after it.
    large_strings: SlowVec<Buffer>,
}

/// A string's hash, which the index takes as already hashed, and its id.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Interned {
    hash: u64,
    id: Id,
}

impl Key for Interned {
    #[inline]
    fn hash(&self) -> u64 {
        self.hash
    }
}

impl StringPool {
    /// A pool holding only the null string. `allocator` must outlive it.
    pub fn new(allocator: &dyn Allocator) -> Result<StringPool, AllocError> {
        let mut store = Store {
            blocks: SlowVec::new(allocator, BLOCKS_MAX)?,
            block_end: 0,
            large_strings: SlowVec::new(allocator, 1 << 31)?,
        };
        store.add_block(allocator)?;
        // The null string's slot, so no string's id is 0.
        store.insert_in_current_block(&[]);
        Ok(StringPool { store, string_index: FlatHashMap::new(allocator, INDEX_CAPACITY)? })
    }

    /// Interns `str` and returns its id: the same each time it's interned.
    /// After an error, the pool mustn't be used again.
    #[inline]
    pub fn intern_string(
        &mut self,
        allocator: &dyn Allocator,
        str: Option<&[u8]>,
    ) -> Result<Id, AllocError> {
        let Some(str) = str else { return Ok(Id::NULL) };
        self.intern_hashed(allocator, str, murmur_hash::bytes(str))
    }

    /// Interns `count` strings, `string(i)` for each, giving each one's id to
    /// `out(i, id)`: as `intern_string` for each, but a batch at a time, so
    /// the memory each needs is loaded for all of them at once rather than
    /// waited for one by one.
    pub fn intern_strings<'s>(
        &mut self,
        allocator: &dyn Allocator,
        count: usize,
        string: impl Fn(usize) -> Option<&'s [u8]>,
        mut out: impl FnMut(usize, Id),
    ) -> Result<(), AllocError> {
        check!(count <= BATCH_ROWS_MAX as usize);
        // While the index fits in cache, there's no wait to hide.
        if self.string_index.capacity() <= CACHED_SLOTS {
            for i in 0..count {
                out(i, self.intern_string(allocator, string(i))?);
            }
            return Ok(());
        }
        for first in (0..count).step_by(CHUNK_ROWS) {
            let rows = CHUNK_ROWS.min(count - first);
            self.intern_chunk(allocator, first, rows, &string, &mut out)?;
        }
        Ok(())
    }

    /// As `intern_strings`, for the `rows` strings from `first`: their
    /// slots are loaded, then the strings they hold, then they're compared.
    fn intern_chunk<'s>(
        &mut self,
        allocator: &dyn Allocator,
        first: usize,
        rows: usize,
        string: &impl Fn(usize) -> Option<&'s [u8]>,
        out: &mut impl FnMut(usize, Id),
    ) -> Result<(), AllocError> {
        let mut strs: [Option<&[u8]>; CHUNK_ROWS] = [None; CHUNK_ROWS];
        let mut hashes = [0_u64; CHUNK_ROWS];
        let mut found = [Id::NULL; CHUNK_ROWS];
        let strs = at_mut!(strs, ..rows);
        for (i, (str, hash)) in strs.iter_mut().zip(&mut hashes).enumerate() {
            *str = string(first + i);
            if let Some(str) = str {
                *hash = murmur_hash::bytes(str);
                self.string_index.prefetch(*hash);
            }
        }
        for ((str, &hash), found) in strs.iter().zip(&hashes).zip(&mut found) {
            if str.is_some() {
                let held = self.string_index.find_key_by(hash, |held| held.hash == hash);
                *found = held.map_or(Id::NULL, |held| held.id);
                self.store.prefetch(*found);
            }
        }
        for (i, ((str, &hash), &found)) in strs.iter().zip(&hashes).zip(&found).enumerate() {
            let id = match *str {
                None => Id::NULL,
                Some(str) if !found.is_null() && self.store.equals(found, str) => found,
                Some(str) => self.intern_hashed(allocator, str, hash)?,
            };
            out(first + i, id);
        }
        Ok(())
    }

    /// As `intern_string`, for `str`, hashed as `hash`.
    #[inline]
    fn intern_hashed(
        &mut self,
        allocator: &dyn Allocator,
        str: &[u8],
        hash: u64,
    ) -> Result<Id, AllocError> {
        // Inserts the null id just to see if the string is already in, and
        // if it isn't, overwrites it with the string's. Strings with the
        // same hash are told apart by their bytes, unlike in Perfetto.
        let store = &self.store;
        let equal = |held: &Interned| held.hash == hash && store.equals(held.id, str);
        let key = Interned { hash, id: Id::NULL };
        let ((held, ()), inserted) =
            self.string_index.insert_by(allocator, hash, equal, key, ())?;
        if inserted {
            held.id = self.store.insert_string(allocator, str)?;
        }
        Ok(held.id)
    }

    /// The string whose id is `id`, or `None` for the null string's.
    #[inline]
    pub fn get(&self, id: Id) -> Option<&[u8]> {
        self.store.get(id)
    }

    /// How many strings it has, besides the null string.
    pub fn size(&self) -> usize {
        self.string_index.len()
    }
}

impl Store {
    /// Whether the string whose id is `id`, not the null string's, is `str`.
    #[inline]
    fn equals(&self, id: Id, str: &[u8]) -> bool {
        if id.is_large_string() {
            return self.get_large_string(id) == str;
        }
        let block = at!(self.blocks, id.block_index() as usize).as_slice::<u8>();
        let offset = id.block_offset() as usize;
        let Some(size) = at!(block, offset..).first_chunk::<4>() else { failed!() };
        let start = offset + 4;
        u32::from_ne_bytes(*size) as usize == str.len()
            && block.get(start..start + str.len()) == Some(str)
    }

    /// Starts loading the string whose id is `id`, unless it's null or large.
    #[inline]
    fn prefetch(&self, id: Id) {
        if !id.is_null()
            && !id.is_large_string()
            && let Some(block) = self.blocks.get(id.block_index() as usize)
        {
            prefetch(block.as_slice::<u8>().as_ptr().wrapping_add(id.block_offset() as usize));
        }
    }

    #[inline]
    fn get(&self, id: Id) -> Option<&[u8]> {
        if id.is_null() {
            return None;
        }
        if id.is_large_string() {
            return Some(self.get_large_string(id));
        }
        let block = at!(self.blocks, id.block_index() as usize).as_slice::<u8>();
        let offset = id.block_offset() as usize;
        let Some(size) = at!(block, offset..).first_chunk::<4>() else { failed!() };
        let start = offset + 4;
        Some(at!(block, start..start + u32::from_ne_bytes(*size) as usize))
    }

    #[inline(never)]
    fn add_block(&mut self, allocator: &dyn Allocator) -> Result<(), AllocError> {
        // SAFETY: a block's bytes are read only below `block_end`, which
        // `insert_in_current_block` writes up to.
        self.blocks.push(unsafe { Buffer::allocate_uninit(allocator, BLOCK_BYTES) }?)?;
        self.block_end = 0;
        Ok(())
    }

    fn insert_string(&mut self, allocator: &dyn Allocator, str: &[u8]) -> Result<Id, AllocError> {
        if str.len() >= LARGE_STRING_BYTES_MIN {
            return self.insert_large_string(allocator, str);
        }
        if self.block_end + str.len() + METADATA_BYTES > BLOCK_BYTES {
            self.add_block(allocator)?;
        }
        Ok(Id::block_string(self.blocks.len() - 1, self.insert_in_current_block(str)))
    }

    #[inline(never)]
    fn insert_large_string(
        &mut self,
        allocator: &dyn Allocator,
        str: &[u8],
    ) -> Result<Id, AllocError> {
        // Zeroed, so the nul is already there.
        let mut buffer = Buffer::allocate(allocator, str.len() + 1)?;
        at_mut!(buffer.as_mut_slice::<u8>(), ..str.len()).copy_from_slice(str);
        self.large_strings.push(buffer)?;
        Ok(Id::large_string(self.large_strings.len() - 1))
    }

    /// Writes `str` at the end of the last block, which has room for it, and
    /// returns where it starts.
    #[inline(never)]
    fn insert_in_current_block(&mut self, str: &[u8]) -> u32 {
        let start = self.block_end;
        let Some(block) = self.blocks.last_mut() else { failed!() };
        let bytes = at_mut!(block.as_mut_slice::<u8>(), start..start + str.len() + METADATA_BYTES);
        let (size, rest) = bytes.split_at_mut(4);
        #[expect(clippy::cast_possible_truncation, reason = "below `LARGE_STRING_BYTES_MIN`")]
        size.copy_from_slice(&(str.len() as u32).to_ne_bytes());
        let (chars, nul) = rest.split_at_mut(str.len());
        chars.copy_from_slice(str);
        *at_mut!(nul, 0) = 0;
        self.block_end = start + str.len() + METADATA_BYTES;
        #[expect(clippy::cast_possible_truncation, reason = "below `BLOCK_BYTES`")]
        let start = start as u32;
        start
    }

    #[inline(never)]
    fn get_large_string(&self, id: Id) -> &[u8] {
        let bytes = at!(self.large_strings, id.large_string_index() as usize).as_slice::<u8>();
        at!(bytes, ..bytes.len() - 1)
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::{Heap, LimitAllocator};

    #[test]
    fn interns_strings() {
        let mut pool = StringPool::new(&Heap).unwrap();
        let large = alloc::vec![b'x'; LARGE_STRING_BYTES_MIN];
        let strings: [Option<&[u8]>; 7] =
            [Some(b"b"), None, Some(b""), Some(b"a"), Some(b"b"), Some(&large), Some(b"a")];
        let ids: Vec<Id> =
            strings.iter().map(|&str| pool.intern_string(&Heap, str).unwrap()).collect();
        assert_eq!(ids[1], Id::NULL);
        assert_eq!([ids[4], ids[6]], [ids[0], ids[3]]);
        assert!(ids[5].is_large_string() && !ids[0].is_large_string());
        for (&id, str) in ids.iter().zip(strings) {
            assert_eq!(pool.get(id), str);
        }
        assert_eq!(pool.size(), 4);
        // Enough to fill more than a block, each found again.
        let mut ids = Vec::new();
        for i in 0..400_000_u32 {
            let str = format!("{i:08}");
            ids.push(pool.intern_string(&Heap, Some(str.as_bytes())).unwrap());
        }
        assert!(ids.last().unwrap().block_index() > 0);
        for (i, &id) in ids.iter().enumerate() {
            let str = format!("{i:08}");
            assert_eq!(pool.get(id), Some(str.as_bytes()));
            assert_eq!(pool.intern_string(&Heap, Some(str.as_bytes())).unwrap(), id);
        }
    }

    #[test]
    fn interns_a_batch_as_one_at_a_time() {
        // Small enough to intern a row at a time, then past `CACHED_SLOTS`,
        // when a batch is looked up at once.
        let mut pool = StringPool::new(&Heap).unwrap();
        let mut one_at_a_time = StringPool::new(&Heap).unwrap();
        let mut ids = alloc::vec![Id::NULL; BATCH_ROWS_MAX as usize];
        for batch in 0..40_u32 {
            let strings: Vec<Option<Vec<u8>>> = (0..BATCH_ROWS_MAX)
                .map(|i| {
                    (i % 9 != 4).then(|| format!("{}", (batch * 1000 + i) % 30_000).into_bytes())
                })
                .collect();
            let string = |i: usize| strings[i].as_deref();
            pool.intern_strings(&Heap, strings.len(), string, |i, id| ids[i] = id).unwrap();
            for (i, &id) in ids.iter().enumerate() {
                assert_eq!(pool.get(id), string(i));
                let expected = one_at_a_time.intern_string(&Heap, string(i)).unwrap();
                assert_eq!(pool.get(id), one_at_a_time.get(expected));
            }
        }
        assert!(pool.string_index.capacity() > CACHED_SLOTS);
        assert_eq!(pool.size(), one_at_a_time.size());
    }

    #[test]
    fn fails_past_its_allocator() {
        let limited = LimitAllocator::new(&Heap, BLOCK_BYTES + (1 << 20));
        let mut pool = StringPool::new(&limited).unwrap();
        let long = alloc::vec![b'y'; LARGE_STRING_BYTES_MIN - 2];
        let ids: Vec<_> = (0..4_u8)
            .map(|i| pool.intern_string(&limited, Some(&[&long[..], &[i]].concat())))
            .collect();
        assert!(ids[..3].iter().all(Result::is_ok));
        assert!(ids[3].is_err());
    }
}
