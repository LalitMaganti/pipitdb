//! Sorting words by keys in their high bits, as Perfetto's `core/util/sort.h`
//! sorts `KeyedIndex`es: by an LSD radix sort where that's estimated to be
//! cheaper, and by comparing them otherwise.

use crate::allocator::{AllocError, Allocator};
use crate::slow_vec::SlowVec;

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

/// Sorts `words` by key with a stable radix sort, least significant digit
/// first. Each pass is a stable counting sort on one digit into the other
/// buffer, and the two then swap roles. The counts of every digit are taken
/// in one read, and a pass is skipped when every word has the same digit.
/// Returns whether the sorted words are in `scratch`, at least as long,
/// rather than `words`.
fn radix_sort(
    words: &mut [u64],
    scratch: &mut [u64],
    counts: &mut [u32],
    low: u32,
    key_bits: u32,
) -> bool {
    let n = words.len();
    if n <= 1 || key_bits == 0 {
        return false;
    }
    let digits = radix_digits(key_bits, n);
    let buckets = 1_usize << digits.bits;
    let mask = buckets as u64 - 1;
    let counts = at_mut!(counts, ..digits.passes as usize * buckets);
    counts.fill(0);
    for &word in words.iter() {
        for p in 0..digits.passes {
            #[expect(clippy::cast_possible_truncation, reason = "under 2^16")]
            let digit = ((word >> (low + p * digits.bits)) & mask) as usize;
            *at_mut!(counts, p as usize * buckets + digit) += 1;
        }
    }
    let (mut source, mut dest) = (words, at_mut!(scratch, ..n));
    let mut in_scratch = false;
    for (p, count) in counts.chunks_exact_mut(buckets).enumerate() {
        #[expect(clippy::cast_possible_truncation, reason = "at most 4 passes")]
        let shift = low + p as u32 * digits.bits;
        #[expect(clippy::cast_possible_truncation, reason = "under 2^16")]
        let digit = |word: u64| ((word >> shift) & mask) as usize;
        // Every word has the same digit: this pass can't move any.
        if *at!(count, digit(*at!(source, 0))) as usize == n {
            continue;
        }
        let mut total = 0;
        for c in count.iter_mut() {
            (*c, total) = (total, total + *c);
        }
        for &word in source.iter() {
            let at = at_mut!(count, digit(word));
            *at_mut!(dest, *at as usize) = word;
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

/// Sorts `words` whose keys are their `key_bits` bits from `low`: any bits
/// above must be the same in every word, and words with equal keys must be
/// in order already, which they stay in. By a radix sort when that's
/// estimated to be cheaper, as for many words, and by comparing them
/// otherwise. Returns whether the sorted words are in `scratch`, at least as
/// long, rather than `words`.
pub(crate) fn sort_words(
    allocator: &dyn Allocator,
    words: &mut [u64],
    scratch: &mut [u64],
    low: u32,
    key_bits: u32,
) -> Result<bool, AllocError> {
    let size = words.len();
    if !radix_sort_is_cheaper(size, key_bits) {
        return Ok(sort_by_comparing(words, scratch));
    }
    let mut counts = SlowVec::zeroed(allocator, radix_sort_counts_size(key_bits, size))?;
    Ok(radix_sort(words, scratch, &mut counts, low, key_bits))
}

/// How many words an insertion sort sorts before runs are merged.
const RUN: usize = 16;

/// Sorts `words`: runs of `RUN` by insertion, then merged, back and forth
/// with `scratch`. Returns whether the sorted words are in `scratch`.
fn sort_by_comparing(words: &mut [u64], scratch: &mut [u64]) -> bool {
    let n = words.len();
    for run in words.chunks_mut(RUN) {
        insertion_sort(run);
    }
    let (mut source, mut dest) = (words, at_mut!(scratch, ..n));
    let mut in_scratch = false;
    let mut width = RUN;
    while width < n {
        for start in (0..n).step_by(2 * width) {
            merge(source, dest, start, (start + width).min(n), (start + 2 * width).min(n));
        }
        core::mem::swap(&mut source, &mut dest);
        in_scratch = !in_scratch;
        width *= 2;
    }
    in_scratch
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

pub(crate) fn insertion_sort(words: &mut [u64]) {
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

    /// `count` words of keys of `key_bits` bits, many of them tied, above
    /// their index's 20 bits.
    fn words(count: u32, key_bits: u32) -> Vec<u64> {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mask = (1 << key_bits) - 1;
        (0..count)
            .map(|index| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                // A few values, spread across every digit.
                let key = (seed % 50).wrapping_mul(0x9e37_79b9_7f4a_7c15) & mask;
                key << 20 | u64::from(index)
            })
            .collect()
    }

    fn sorted(mut words: Vec<u64>, sort: impl Fn(&mut [u64], &mut [u64]) -> bool) -> Vec<u64> {
        let mut scratch = alloc::vec![0; words.len()];
        if sort(&mut words, &mut scratch) { scratch } else { words }
    }

    #[test]
    #[cfg_attr(miri, ignore = "slow under Miri, and there's no unsafe code to check")]
    fn sorts_stably_by_radix_and_by_comparing() {
        for (count, key_bits) in
            [(3000, 44), (3000, 20), (300, 9), (17, 44), (3, 8), (1, 8), (0, 8)]
        {
            let words = words(count, key_bits);
            let mut expected = words.clone();
            expected.sort_unstable();
            let counts = radix_sort_counts_size(key_bits, words.len());
            let radix = sorted(words.clone(), |w, s| {
                radix_sort(w, s, &mut alloc::vec![0; counts], 20, key_bits)
            });
            assert_eq!(radix, expected, "radix, {count} of {key_bits} bits");
            let compared = sorted(words.clone(), sort_by_comparing);
            assert_eq!(compared, expected, "comparing, {count} of {key_bits} bits");
            let either = sorted(words, |w, s| sort_words(&Heap, w, s, 20, key_bits).unwrap());
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
