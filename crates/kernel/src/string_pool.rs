//! `StringPool`: interns strings, handing out compact ids that give each one
//! back in O(1), ported from Perfetto's.

use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::flat_hash_map::{FlatHashMap, Key};
use crate::murmur_hash;
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
    /// The id of each string, by its hash.
    string_index: FlatHashMap<StringHash, Id>,
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

/// A string's hash, which the index takes as already hashed.
#[derive(Clone, Copy, PartialEq, Eq)]
struct StringHash(u64);

impl Key for StringHash {
    #[inline]
    fn hash(&self) -> u64 {
        self.0
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
    /// Strings are told apart by their hash alone.
    #[inline]
    pub fn intern_string(
        &mut self,
        allocator: &dyn Allocator,
        str: Option<&[u8]>,
    ) -> Result<Id, AllocError> {
        let Some(str) = str else { return Ok(Id::NULL) };
        // Inserts the null id just to see if the string is already in, and
        // if it isn't, overwrites it with the string's.
        let hash = StringHash(murmur_hash::bytes(str));
        let (id, inserted) = self.string_index.insert(allocator, hash, Id::NULL)?;
        if inserted {
            match self.store.insert_string(allocator, str) {
                Ok(new) => *id = new,
                Err(error) => {
                    self.string_index.erase(hash);
                    return Err(error);
                }
            }
            return Ok(*id);
        }
        let id = *id;
        if cfg!(debug_assertions) {
            check!(self.get(id) == Some(str));
        }
        Ok(id)
    }

    /// The string whose id is `id`, or `None` for the null string's.
    #[inline]
    pub fn get(&self, id: Id) -> Option<&[u8]> {
        if id.is_null() {
            return None;
        }
        if id.is_large_string() {
            return Some(self.store.get_large_string(id));
        }
        let block = at!(self.store.blocks, id.block_index() as usize).as_slice::<u8>();
        let offset = id.block_offset() as usize;
        let Some(size) = at!(block, offset..).first_chunk::<4>() else { failed!() };
        let start = offset + 4;
        Some(at!(block, start..start + u32::from_ne_bytes(*size) as usize))
    }

    /// How many strings it has, besides the null string.
    pub fn size(&self) -> usize {
        self.string_index.len()
    }
}

impl Store {
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
