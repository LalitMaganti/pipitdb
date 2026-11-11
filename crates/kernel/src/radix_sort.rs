//! Sorting rows' words in place, most significant digit first, partitioning
//! whole blocks at a time, and sorting strings by their bytes.

use crate::allocator::{AllocError, Allocator};
use crate::slow_vec::SlowVec;

/// How many bits one partition splits on.
const DIGIT_BITS: u32 = 8;

/// How many buckets one partition splits into.
const BUCKETS: usize = 1 << DIGIT_BITS;

/// How many words a block holds, the unit partitioning moves.
const BLOCK: usize = 128;

/// Up to this many words are sorted by insertion.
const INSERTION_MAX: usize = 24;

/// Up to this many words are sorted by copying them to their buckets, which
/// costs far less per call than partitioning in place.
const SMALL: usize = 4096;

/// What sorting words works in, one after another: by bucket, a block
/// being filled then any block that had nowhere to go, which `sort_small`
/// also copies through; the block being carried to its bucket; and by block
/// of the input, whether it's been moved. Partitioning is done before
/// the buckets are sorted, so one is enough for every level.
pub(crate) struct Scratch(SlowVec<u64>);

impl Scratch {
    /// For sorting up to `count` words.
    pub(crate) fn new(allocator: &dyn Allocator, count: usize) -> Result<Scratch, AllocError> {
        let len =
            if count <= SMALL { count } else { BUCKETS * 2 * BLOCK + BLOCK + count / BLOCK + 2 };
        Ok(Scratch(SlowVec::zeroed(allocator, len)?))
    }
}

/// The bits to split `words`, which all differ, on: the highest that differ.
fn top_digit(words: &[u64]) -> u32 {
    let first = *at!(words, 0);
    let differ = words.iter().fold(0, |differ, &word| differ | (word ^ first));
    differ.ilog2().saturating_sub(DIGIT_BITS - 1)
}

/// Sorts up to `SMALL` `words`, which all differ, through `spare`.
fn sort_small(words: &mut [u64], spare: &mut [u64]) {
    if words.len() <= INSERTION_MAX {
        insertion_sort(words);
        return;
    }
    let shift = top_digit(words);
    #[expect(clippy::cast_possible_truncation, reason = "masked to a digit")]
    let digit = |word: u64| (word >> shift) as usize & (BUCKETS - 1);
    let mut start = [0_u32; BUCKETS + 1];
    for &word in words.iter() {
        *at_mut!(start, digit(word) + 1) += 1;
    }
    for b in 0..BUCKETS {
        *at_mut!(start, b + 1) += *at!(start, b);
    }
    let copy = at_mut!(spare, ..words.len());
    copy.copy_from_slice(words);
    let mut next = start;
    for &word in copy.iter() {
        let at = at_mut!(next, digit(word));
        *at_mut!(words, *at as usize) = word;
        *at += 1;
    }
    for b in 0..BUCKETS {
        let bucket = at_mut!(words, *at!(start, b) as usize..*at!(start, b + 1) as usize);
        if bucket.len() > 1 {
            sort_small(bucket, spare);
        }
    }
}

fn insertion_sort(words: &mut [u64]) {
    for i in 1..words.len() {
        let word = *at!(words, i);
        let mut j = i;
        while j > 0 && word < *at!(words, j - 1) {
            *at_mut!(words, j) = *at!(words, j - 1);
            j -= 1;
        }
        *at_mut!(words, j) = word;
    }
}

/// Puts `words` into blocks of one bucket each, written back over the words
/// already read, and returns how many words are left in each bucket's
/// buffer, how many each bucket has, and how many blocks were written.
fn fill_blocks(
    words: &mut [u64],
    digit: impl Fn(u64) -> usize,
    buffers: &mut [u64],
) -> ([usize; BUCKETS], [usize; BUCKETS], usize) {
    let (mut filled, mut total, mut written) = ([0_usize; BUCKETS], [0_usize; BUCKETS], 0);
    for i in 0..words.len() {
        let word = *at!(words, i);
        let bucket = digit(word);
        *at_mut!(buffers, bucket * 2 * BLOCK + *at!(filled, bucket)) = word;
        *at_mut!(filled, bucket) += 1;
        *at_mut!(total, bucket) += 1;
        if *at!(filled, bucket) == BLOCK {
            let block = at!(buffers, bucket * 2 * BLOCK..bucket * 2 * BLOCK + BLOCK);
            at_mut!(words, written..written + BLOCK).copy_from_slice(block);
            written += BLOCK;
            *at_mut!(filled, bucket) = 0;
        }
    }
    (filled, total, written / BLOCK)
}

/// Splits `words` in place by the digit at `shift`, and returns where each
/// bucket starts, then their end. Whole blocks are moved to the blocks
/// inside their bucket, and what's left fills each bucket's ends.
fn partition(words: &mut [u64], shift: u32, scratch: &mut [u64]) -> [usize; BUCKETS + 1] {
    #[expect(clippy::cast_possible_truncation, reason = "masked to a digit")]
    let digit = |word: u64| (word >> shift) as usize & (BUCKETS - 1);
    let n = words.len();
    let (buffers, rest) = scratch.split_at_mut(BUCKETS * 2 * BLOCK);
    let (carried, moved) = rest.split_at_mut(BLOCK);
    let (filled, total, full) = fill_blocks(words, digit, buffers);
    // A bucket's slots are the blocks inside it: from `next`, the next to
    // fill, to `end`.
    let mut start = [0_usize; BUCKETS + 1];
    let (mut next, mut end) = ([0_usize; BUCKETS], [0_usize; BUCKETS]);
    for b in 0..BUCKETS {
        *at_mut!(start, b + 1) = *at!(start, b) + *at!(total, b);
        *at_mut!(next, b) = at!(start, b).div_ceil(BLOCK);
        *at_mut!(end, b) = (*at!(next, b)).max(*at!(start, b + 1) / BLOCK);
    }
    at_mut!(moved, ..n / BLOCK + 2).fill(0);
    for g in 0..full {
        if *at!(moved, g) != 0 {
            continue;
        }
        carried.copy_from_slice(at!(words, g * BLOCK..(g + 1) * BLOCK));
        *at_mut!(moved, g) = 1;
        loop {
            let b = digit(*at!(carried, 0));
            if *at!(next, b) == *at!(end, b) {
                // At most one block of a bucket's has no slot inside it.
                let at = b * 2 * BLOCK + *at!(filled, b);
                at_mut!(buffers, at..at + BLOCK).copy_from_slice(carried);
                break;
            }
            let target = *at!(next, b);
            *at_mut!(next, b) += 1;
            let range = target * BLOCK..(target + 1) * BLOCK;
            let free = target >= full || *at!(moved, target) != 0;
            *at_mut!(moved, target) = 1;
            if free {
                at_mut!(words, range).copy_from_slice(carried);
                break;
            }
            at_mut!(words, range).swap_with_slice(carried);
        }
    }
    // Each bucket's words before and after its slots are the ones left in
    // its buffer.
    for b in 0..BUCKETS {
        let (low, high) = (*at!(start, b), *at!(start, b + 1));
        let from = (low.div_ceil(BLOCK) * BLOCK).min(high);
        let to = (*at!(end, b) * BLOCK).min(high);
        let left = at!(buffers, b * 2 * BLOCK..);
        at_mut!(words, low..from).copy_from_slice(at!(left, ..from - low));
        at_mut!(words, to..high).copy_from_slice(at!(left, from - low..from - low + high - to));
    }
    start
}

/// Sorts `words`, which must all differ, in `scratch`, made for at least as
/// many: each level partitions on the highest bits that differ among them,
/// so a level always makes progress and few words are ever left to
/// insertion sort.
pub(crate) fn sort_words(words: &mut [u64], scratch: &mut Scratch) {
    if words.len() <= SMALL {
        sort_small(words, &mut scratch.0);
        return;
    }
    let start = partition(words, top_digit(words), &mut scratch.0);
    let mut rest = words;
    for b in 0..BUCKETS {
        let (bucket, tail) = rest.split_at_mut(*at!(start, b + 1) - *at!(start, b));
        rest = tail;
        if bucket.len() > 1 {
            sort_words(bucket, scratch);
        }
    }
}

/// Merges the sorted `source[start..mid]` and `source[mid..end]` into
/// `dest[start..end]`.
pub(crate) fn merge(source: &[u64], dest: &mut [u64], start: usize, mid: usize, end: usize) {
    let (mut a, mut b) = (start, mid);
    for out in at_mut!(dest, start..end) {
        let left = b == end || (a < mid && at!(source, a) <= at!(source, b));
        *out = if left { *at!(source, a) } else { *at!(source, b) };
        if left { a += 1 } else { b += 1 }
    }
}

/// Below this many, a bucket of strings is sorted by insertion.
const MSD_INSERTION_SORT: usize = 24;

/// A bucket of `msd_radix_sort`'s still to sort: `elements[start..end]`, by
/// their keys' bytes from `depth` on.
#[derive(Clone, Copy)]
struct Bucket {
    start: usize,
    end: usize,
    depth: usize,
}

/// Sorts `elements` by `key`, a string of bytes, with a most significant
/// digit first radix sort, as Perfetto's `MsdRadixSort` does: the elements
/// are split into buckets by their key's byte at a depth, each then by the
/// next, from a stack rather than by recursion. Equal keys end up in any
/// order. `scratch` must be at least as long.
pub(crate) fn msd_radix_sort<'k, T: Copy>(
    allocator: &dyn Allocator,
    elements: &mut [T],
    scratch: &mut [T],
    key: impl Fn(&T) -> &'k [u8],
) -> Result<(), AllocError> {
    if elements.len() <= 1 {
        return Ok(());
    }
    let suffix = |element: &T, depth: usize| key(element).get(depth..).unwrap_or_default();
    // The byte of `key` at `depth`, plus one, or 0 for keys that short.
    let digit = |e: &T, depth| suffix(e, depth).first().map_or(0, |&b| usize::from(b) + 1);
    let mut stack = SlowVec::new(allocator, 1 << 24)?;
    stack.push(Bucket { start: 0, end: elements.len(), depth: 0 })?;
    while let Some(bucket) = stack.pop() {
        let (depth, items) = (bucket.depth, at_mut!(elements, bucket.start..bucket.end));
        if items.len() <= MSD_INSERTION_SORT {
            for i in 1..items.len() {
                let mut j = i;
                while j > 0 && suffix(at!(items, j - 1), depth) > suffix(at!(items, j), depth) {
                    items.swap(j - 1, j);
                    j -= 1;
                }
            }
            continue;
        }
        let copy = at_mut!(scratch, bucket.start..bucket.end);
        copy.copy_from_slice(items);
        let mut counts = [0_usize; 257];
        for element in copy.iter() {
            *at_mut!(counts, digit(element, depth)) += 1;
        }
        let mut total = 0;
        for c in &mut counts {
            (*c, total) = (total, total + *c);
        }
        for element in copy.iter() {
            let at = at_mut!(counts, digit(element, depth));
            *at_mut!(items, *at) = *element;
            *at += 1;
        }
        // Each count is now where the next digit's bucket starts. Keys that
        // end at this depth are all equal: only longer ones go on.
        for d in (0..256).rev() {
            let (start, end) = (bucket.start + *at!(counts, d), bucket.start + *at!(counts, d + 1));
            if end - start > 1 {
                stack.push(Bucket { start, end, depth: depth + 1 })?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;

    #[test]
    #[cfg_attr(miri, ignore = "slow under Miri")]
    fn sorts_words_of_any_spread() {
        for (count, mask) in
            [(100_000_u64, u64::MAX), (50_000, 0xFF << 24), (9000, 0), (10, u64::MAX)]
        {
            let mut seed = 0x2545_f491_4f6c_dd1d_u64;
            let mut words: Vec<u64> = (0..count)
                .map(|row| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed & mask) << 24 >> 24 & !0xFF_FFFF | row
                })
                .collect();
            let mut expected = words.clone();
            expected.sort_unstable();
            let mut scratch = Scratch::new(&Heap, words.len()).unwrap();
            sort_words(&mut words, &mut scratch);
            assert_eq!(words, expected);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "slow under Miri")]
    fn sorts_strings_by_their_bytes() {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let strings: Vec<Vec<u8>> = (0..2000)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                // Short strings of a few bytes, so many share prefixes or tie.
                let len = (seed % 6) as usize;
                (0..len).map(|i| b"ab\xffz"[((seed >> (8 * i)) % 4) as usize]).collect()
            })
            .collect();
        let mut order: Vec<u32> = (0..2000).collect();
        let mut scratch = order.clone();
        msd_radix_sort(&Heap, &mut order, &mut scratch, |&i| strings[i as usize].as_slice())
            .unwrap();
        let sorted: Vec<&Vec<u8>> = order.iter().map(|&i| &strings[i as usize]).collect();
        let mut expected: Vec<&Vec<u8>> = strings.iter().collect();
        expected.sort();
        assert_eq!(sorted, expected);
    }
}
