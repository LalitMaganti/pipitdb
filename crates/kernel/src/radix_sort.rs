//! Sorting `KeyedIndex`es by their keys, as Perfetto's `core/util/sort.h`
//! does: by an LSD radix sort where that's estimated to be cheaper, and by
//! comparing them otherwise.

use crate::allocator::{AllocError, Allocator};
use crate::buffer::Primitive;
use crate::slow_vec::SlowVec;

/// What's sorted: a key, the index which breaks ties, and a value carried
/// along.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
#[repr(C)]
pub(crate) struct KeyedIndex {
    pub(crate) key: u64,
    pub(crate) index: u32,
    pub(crate) value: u32,
}

// SAFETY: any bits are a valid `KeyedIndex`, which has no padding.
unsafe impl Primitive for KeyedIndex {}

/// How a key is split into digits for a radix sort.
#[derive(Clone, Copy)]
struct RadixDigits {
    passes: u32,
    bits: u32,
}

/// Splits a key of `key_bits` bits into digits of at most 16 bits, all the
/// same width. Each pass scatters every element, which costs several times
/// clearing and summing one count, so the split minimizes
/// `passes * (4 * size + counts)`: the fewest passes for many elements, and
/// narrower digits, with more passes, for few.
fn radix_digits(key_bits: u32, size: usize) -> RadixDigits {
    let fewest = key_bits.div_ceil(16);
    if fewest == 0 {
        return RadixDigits { passes: 0, bits: 0 };
    }
    let mut best = RadixDigits { passes: 0, bits: 0 };
    let mut best_cost = u64::MAX;
    for passes in fewest..=fewest + 2 {
        let bits = key_bits.div_ceil(passes);
        let cost = u64::from(passes) * (4 * size as u64 + (1 << bits));
        if cost < best_cost {
            (best, best_cost) = (RadixDigits { passes, bits }, cost);
        }
    }
    best
}

/// The number of counts `radix_sort` needs for `size` keys of `key_bits`
/// bits.
fn radix_sort_counts_size(key_bits: u32, size: usize) -> usize {
    let digits = radix_digits(key_bits, size);
    (digits.passes as usize) << digits.bits
}

/// Sorts `elements` by key with a stable radix sort, least significant digit
/// first. Only the low `key_bits` bits of the keys are sorted on: any above
/// them must be the same in every key. Each pass is a stable counting sort
/// on one digit into the other buffer, and the two then swap roles. The
/// counts of every digit are taken in one read, and a pass is skipped when
/// every element has the same digit. Returns whether the sorted elements are
/// in `scratch`, at least as long, rather than `elements`.
fn radix_sort(
    elements: &mut [KeyedIndex],
    scratch: &mut [KeyedIndex],
    counts: &mut [u32],
    key_bits: u32,
) -> bool {
    let n = elements.len();
    if n <= 1 || key_bits == 0 {
        return false;
    }
    let digits = radix_digits(key_bits, n);
    let buckets = 1_usize << digits.bits;
    let mask = buckets as u64 - 1;
    let counts = at_mut!(counts, ..digits.passes as usize * buckets);
    counts.fill(0);
    for element in elements.iter() {
        for p in 0..digits.passes {
            #[expect(clippy::cast_possible_truncation, reason = "under 2^16")]
            let digit = ((element.key >> (p * digits.bits)) & mask) as usize;
            *at_mut!(counts, p as usize * buckets + digit) += 1;
        }
    }
    let (mut source, mut dest) = (elements, at_mut!(scratch, ..n));
    let mut in_scratch = false;
    for (p, count) in counts.chunks_exact_mut(buckets).enumerate() {
        #[expect(clippy::cast_possible_truncation, reason = "at most 4 passes")]
        let shift = p as u32 * digits.bits;
        #[expect(clippy::cast_possible_truncation, reason = "under 2^16")]
        let digit = |element: &KeyedIndex| ((element.key >> shift) & mask) as usize;
        // Every element has the same digit: this pass can't move any.
        if *at!(count, digit(at!(source, 0))) as usize == n {
            continue;
        }
        let mut total = 0;
        for c in count.iter_mut() {
            (*c, total) = (total, total + *c);
        }
        for element in source.iter() {
            let at = at_mut!(count, digit(element));
            *at_mut!(dest, *at as usize) = *element;
            *at += 1;
        }
        core::mem::swap(&mut source, &mut dest);
        in_scratch = !in_scratch;
    }
    in_scratch
}

/// Whether radix sorting `size` keys of `key_bits` bits is estimated to cost
/// less than comparing them. Each pass of `radix_sort` reads and writes
/// every element, and clears and sums its counts.
fn radix_sort_is_cheaper(size: usize, key_bits: u32) -> bool {
    let digits = radix_digits(key_bits, size);
    let radix = u64::from(digits.passes) * (3 * size as u64 + (1 << digits.bits));
    let log2_size = u64::from(usize::BITS - size.leading_zeros());
    radix < 2 * size as u64 * log2_size
}

/// Stably sorts `elements` by key, ties by index, where only the low
/// `key_bits` bits of the keys may differ: by a radix sort when that's
/// estimated to be cheaper, as for many elements, and by comparing them
/// otherwise. Returns whether the sorted elements are in `scratch`, at least
/// as long, rather than `elements`.
pub(crate) fn stable_sort_keyed_indices(
    allocator: &dyn Allocator,
    elements: &mut [KeyedIndex],
    scratch: &mut [KeyedIndex],
    key_bits: u32,
) -> Result<bool, AllocError> {
    let size = elements.len();
    if !radix_sort_is_cheaper(size, key_bits) {
        return Ok(sort_keyed_indices(elements, scratch));
    }
    let mut counts = SlowVec::zeroed(allocator, radix_sort_counts_size(key_bits, size))?;
    Ok(radix_sort(elements, scratch, &mut counts, key_bits))
}

/// How many elements an insertion sort sorts before runs are merged.
const RUN: usize = 16;

/// Sorts `elements` by key, then index: runs of `RUN` by insertion, then
/// merged, back and forth with `scratch`. Returns whether the sorted
/// elements are in `scratch`.
fn sort_keyed_indices(elements: &mut [KeyedIndex], scratch: &mut [KeyedIndex]) -> bool {
    let n = elements.len();
    for run in elements.chunks_mut(RUN) {
        insertion_sort(run);
    }
    let less = |a: &KeyedIndex, b: &KeyedIndex| (a.key, a.index) < (b.key, b.index);
    let (mut source, mut dest) = (elements, at_mut!(scratch, ..n));
    let mut in_scratch = false;
    let mut width = RUN;
    while width < n {
        for start in (0..n).step_by(2 * width) {
            let (mid, end) = ((start + width).min(n), (start + 2 * width).min(n));
            let (mut a, mut b) = (start, mid);
            for out in at_mut!(dest, start..end) {
                let left = b == end || (a < mid && !less(at!(source, b), at!(source, a)));
                *out = if left { *at!(source, a) } else { *at!(source, b) };
                if left { a += 1 } else { b += 1 }
            }
        }
        core::mem::swap(&mut source, &mut dest);
        in_scratch = !in_scratch;
        width *= 2;
    }
    in_scratch
}

/// Sorts `elements` by key, then index, in place.
pub(crate) fn insertion_sort(elements: &mut [KeyedIndex]) {
    for i in 1..elements.len() {
        let t = *at!(elements, i);
        let mut j = i;
        while j > 0 && (at!(elements, j - 1).key, at!(elements, j - 1).index) > (t.key, t.index) {
            *at_mut!(elements, j) = *at!(elements, j - 1);
            j -= 1;
        }
        *at_mut!(elements, j) = t;
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
    // The byte of `key` at `depth`, plus one, or 0 for keys that short.
    let digit =
        |element: &T, depth: usize| key(element).get(depth).map_or(0, |&b| usize::from(b) + 1);
    let mut stack = SlowVec::new(allocator, 1 << 24)?;
    stack.push(Bucket { start: 0, end: elements.len(), depth: 0 })?;
    while let Some(bucket) = stack.pop() {
        let items = at_mut!(elements, bucket.start..bucket.end);
        if items.len() <= MSD_INSERTION_SORT {
            for i in 1..items.len() {
                let mut j = i;
                while j > 0
                    && at!(key(at!(items, j - 1)), bucket.depth.min(key(at!(items, j - 1)).len())..)
                        > at!(key(at!(items, j)), bucket.depth.min(key(at!(items, j)).len())..)
                {
                    items.swap(j - 1, j);
                    j -= 1;
                }
            }
            continue;
        }
        let copy = at_mut!(scratch, bucket.start..bucket.end);
        copy.copy_from_slice(items);
        let mut counts = [0_usize; 258];
        for element in copy.iter() {
            *at_mut!(counts, digit(element, bucket.depth) + 1) += 1;
        }
        for d in 1..counts.len() {
            *at_mut!(counts, d) += *at!(counts, d - 1);
        }
        let starts = counts;
        for element in copy.iter() {
            let at = at_mut!(counts, digit(element, bucket.depth));
            *at_mut!(items, *at) = *element;
            *at += 1;
        }
        // Keys that end at this depth are all equal: only longer ones go on.
        for d in (1..257).rev() {
            let (start, end) = (*at!(starts, d), *at!(starts, d + 1));
            if end - start > 1 {
                let start = bucket.start + start;
                stack.push(Bucket { start, end: bucket.start + end, depth: bucket.depth + 1 })?;
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

    /// `count` elements whose keys are of `key_bits` bits, many of them tied.
    fn elements(count: u32, key_bits: u32) -> Vec<KeyedIndex> {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mask = if key_bits == 64 { u64::MAX } else { (1 << key_bits) - 1 };
        (0..count)
            .map(|index| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                // A few values, spread across every digit.
                let key = (seed % 50).wrapping_mul(0x9e37_79b9_7f4a_7c15) & mask;
                KeyedIndex { key, index, value: index * 3 }
            })
            .collect()
    }

    fn sorted(
        mut elements: Vec<KeyedIndex>,
        sort: impl Fn(&mut [KeyedIndex], &mut [KeyedIndex]) -> bool,
    ) -> Vec<KeyedIndex> {
        let mut scratch = alloc::vec![KeyedIndex::default(); elements.len()];
        if sort(&mut elements, &mut scratch) { scratch } else { elements }
    }

    #[test]
    #[cfg_attr(miri, ignore = "slow under Miri, and there's no unsafe code to check")]
    fn sorts_stably_by_radix_and_by_comparing() {
        for (count, key_bits) in
            [(3000, 64), (3000, 20), (300, 9), (17, 64), (3, 8), (1, 8), (0, 8)]
        {
            let elements = elements(count, key_bits);
            let mut expected = elements.clone();
            expected.sort_by_key(|e| (e.key, e.index));
            let counts = radix_sort_counts_size(key_bits, elements.len());
            let radix = sorted(elements.clone(), |e, s| {
                radix_sort(e, s, &mut alloc::vec![0; counts], key_bits)
            });
            assert_eq!(radix, expected, "radix, {count} of {key_bits} bits");
            let compared = sorted(elements.clone(), sort_keyed_indices);
            assert_eq!(compared, expected, "comparing, {count} of {key_bits} bits");
            let either =
                sorted(elements, |e, s| stable_sort_keyed_indices(&Heap, e, s, key_bits).unwrap());
            assert_eq!(either, expected, "either, {count} of {key_bits} bits");
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

    #[test]
    fn radix_sorts_many_keys_and_compares_few() {
        assert!(radix_sort_is_cheaper(30_000, 64));
        assert!(!radix_sort_is_cheaper(300, 64));
        let digits = radix_digits(64, 1 << 20);
        assert_eq!((digits.passes, digits.bits), (4, 16));
        let digits = radix_digits(20, 100);
        assert_eq!((digits.passes, digits.bits), (3, 7));
    }
}
