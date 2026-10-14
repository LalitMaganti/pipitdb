//! `sort_by_key`: sorts row numbers by fixed-width key bytes, stably: by an
//! LSD radix sort, a 16-bit digit at a time, as Perfetto's does, or by a
//! merge sort where there are too few rows for a radix sort to pay.

use crate::allocator::{AllocError, Allocator};
use crate::slow_vec::SlowVec;

/// The bits of a digit, so its counts fit in a 256 KB table.
const DIGIT_BITS: u32 = 16;

/// Sorts `order`, row numbers in increasing order, by each row's key, the
/// `width` bytes at `width * row` of `keys`, stably: rows with equal keys
/// stay in increasing order. Memory for a radix sort comes from
/// `allocator`.
pub(crate) fn sort_by_key(
    allocator: &dyn Allocator,
    order: &mut [u32],
    keys: &[u8],
    width: usize,
) -> Result<(), AllocError> {
    let rows = order.len();
    check!(keys.len() >= rows * width);
    let key = |row: u32| at!(keys, row as usize * width..(row as usize + 1) * width);
    if !radix_pays(rows, width) {
        return merge_sort(allocator, order, key);
    }
    radix(allocator, order, keys, width)
}

/// How many rows an insertion sort sorts before runs are merged.
const RUN_ROWS: usize = 16;

/// Sorts `order` by `key` stably: runs of `RUN_ROWS` by insertion, then
/// merged, back and forth with scratch. Small, where the core library's
/// sorts take kilobytes for each key they're given.
fn merge_sort<'k>(
    allocator: &dyn Allocator,
    order: &mut [u32],
    key: impl Fn(u32) -> &'k [u8],
) -> Result<(), AllocError> {
    let rows = order.len();
    for run in order.chunks_mut(RUN_ROWS) {
        for i in 1..run.len() {
            let mut j = i;
            while j > 0 && key(*at!(run, j - 1)) > key(*at!(run, j)) {
                run.swap(j - 1, j);
                j -= 1;
            }
        }
    }
    if rows <= RUN_ROWS {
        return Ok(());
    }
    let mut scratch = SlowVec::zeroed(allocator, rows)?;
    let (mut from, mut to): (&mut [u32], &mut [u32]) = (order, &mut scratch);
    let mut swapped = false;
    let mut width = RUN_ROWS;
    while width < rows {
        for start in (0..rows).step_by(2 * width) {
            let (mid, end) = ((start + width).min(rows), (start + 2 * width).min(rows));
            let (mut a, mut b) = (start, mid);
            for out in at_mut!(to, start..end) {
                // The left run's first on ties, so equal keys keep their order.
                let left = b == end || (a < mid && key(*at!(from, a)) <= key(*at!(from, b)));
                *out = if left { *at!(from, a) } else { *at!(from, b) };
                if left { a += 1 } else { b += 1 }
            }
        }
        (from, to) = (to, from);
        swapped = !swapped;
        width *= 2;
    }
    if swapped {
        to.copy_from_slice(from);
    }
    Ok(())
}

/// `sort_by_key`, by a radix sort, whatever the number of rows.
fn radix(
    allocator: &dyn Allocator,
    order: &mut [u32],
    keys: &[u8],
    width: usize,
) -> Result<(), AllocError> {
    let rows = order.len();
    let key = |row: u32| at!(keys, row as usize * width..(row as usize + 1) * width);
    let mut scratch = SlowVec::zeroed(allocator, rows)?;
    let mut counts = SlowVec::<u32>::zeroed(allocator, 1 << DIGIT_BITS)?;
    // Least significant digit first, back and forth between `order` and
    // `scratch`: a digit of two bytes, then a last one of the first byte if
    // the width is odd.
    let (mut from, mut to): (&mut [u32], &mut [u32]) = (order, &mut scratch);
    let mut swapped = false;
    let mut end = width;
    while end > 0 {
        let start = end.saturating_sub(2);
        pass(from, to, &mut counts, |row| {
            let key = key(row);
            if end - start == 2 {
                u16::from_be_bytes([*at!(key, start), *at!(key, start + 1)]) as usize
            } else {
                usize::from(*at!(key, start))
            }
        });
        (from, to) = (to, from);
        swapped = !swapped;
        end = start;
    }
    // The sorted rows are in `from`: if that's the scratch, copy them back.
    if swapped {
        to.copy_from_slice(from);
    }
    Ok(())
}

/// Whether a radix sort of `rows` keys of `width` bytes takes fewer steps
/// than comparing them: each pass reads the rows twice and the counts
/// once, where a comparison sort compares about `rows × log2(rows)` times.
fn radix_pays(rows: usize, width: usize) -> bool {
    let passes = width.div_ceil(2) as u64;
    let rows = rows as u64;
    let log = u64::from(rows.max(1).ilog2());
    passes * (4 * rows + (1 << DIGIT_BITS)) < 2 * rows * log
}

/// One stable counting pass: `from`'s rows into `to`, ordered by `digit`.
fn pass(from: &[u32], to: &mut [u32], counts: &mut [u32], digit: impl Fn(u32) -> usize) {
    counts.fill(0);
    for &row in from {
        *at_mut!(counts, digit(row)) += 1;
    }
    let mut total = 0;
    for count in counts.iter_mut() {
        (*count, total) = (total, total + *count);
    }
    for &row in from {
        let at = at_mut!(counts, digit(row));
        *at_mut!(to, *at as usize) = row;
        *at += 1;
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;

    /// `rows` keys of `width` bytes, from a few distinct values so many tie.
    fn keys(rows: usize, width: usize) -> Vec<u8> {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut keys = Vec::new();
        for _ in 0..rows {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let value = (seed % 50).to_be_bytes();
            keys.extend_from_slice(&value[8 - width.min(8)..]);
            keys.resize(keys.len() + width.saturating_sub(8), 7);
        }
        keys
    }

    #[test]
    #[cfg_attr(miri, ignore = "slow under Miri, and there's no unsafe code to check")]
    fn sorts_stably_by_radix_and_by_comparing() {
        for (rows, width, by_radix) in
            [(300, 8, false), (300, 8, true), (300, 9, true), (3, 1, false)]
        {
            let keys = keys(rows, width);
            let rows = u32::try_from(rows).unwrap();
            let mut order: Vec<u32> = (0..rows).collect();
            if by_radix {
                radix(&Heap, &mut order, &keys, width).unwrap();
            } else {
                sort_by_key(&Heap, &mut order, &keys, width).unwrap();
            }
            let key = |row: u32| &keys[row as usize * width..(row as usize + 1) * width];
            let mut expected: Vec<u32> = (0..rows).collect();
            expected.sort_by(|&a, &b| key(a).cmp(key(b)));
            assert_eq!(order, expected);
        }
        assert!(radix_pays(30_000, 8) && !radix_pays(300, 8));
    }
}
