//! `sort_row_layout`: stably sorts rows laid out by a `RowLayout`, as
//! Perfetto's `core/util/row_layout_sort.cc` does. It sorts on the bytes
//! which differ between rows, the first 8 of them packed into one integer
//! per row so that sorting doesn't read the rows again. Rows which agree on
//! those 8 bytes are then sorted on the next 8, and so on, with the same
//! sort.

use core::cmp::Ordering;

use crate::allocator::{AllocError, Allocator};
use crate::radix_sort::{KeyedIndex, insertion_sort, stable_sort_keyed_indices};
use crate::slow_vec::SlowVec;

/// Up to this many ascending runs, of this many rows on average, are merged
/// rather than sorted.
const MAX_MERGED_RUNS: usize = 32;
const ROWS_PER_MERGED_RUN: usize = 256;

/// The most bytes packed into a key.
const MAX_KEY_BYTES: usize = 8;

/// Below this size, a group of tied rows is sorted by insertion.
const INSERTION_SORT_TIES: usize = 16;

/// Rows of `stride` bytes each, one after another.
#[derive(Clone, Copy)]
struct Rows<'a> {
    data: &'a [u8],
    stride: usize,
    count: u32,
}

impl<'a> Rows<'a> {
    fn get(self, row: u32) -> &'a [u8] {
        let at = row as usize * self.stride;
        at!(self.data, at..at + self.stride)
    }
}

/// The order rows are already in, as far as it helps.
struct ExistingOrder {
    sorted: bool,
    strictly_descending: bool,
    /// Where each ascending run starts, then the row count, if there are few:
    /// the first `runs_len`.
    runs: [u32; MAX_MERGED_RUNS + 2],
    runs_len: usize,
}

/// Stops as soon as the rows are neither in a few runs nor strictly
/// descending, which for most input is within a few rows.
fn find_existing_order(rows: Rows) -> ExistingOrder {
    let max_runs = MAX_MERGED_RUNS.min(rows.count as usize / ROWS_PER_MERGED_RUN);
    let mut order = ExistingOrder {
        sorted: false,
        strictly_descending: false,
        runs: [0; MAX_MERGED_RUNS + 2],
        runs_len: 1,
    };
    let (mut few_runs, mut descending) = (true, true);
    let mut r = 1;
    while r < rows.count && (few_runs || descending) {
        let cmp = rows.get(r - 1).cmp(rows.get(r));
        descending &= cmp == Ordering::Greater;
        if cmp == Ordering::Greater && few_runs {
            few_runs = order.runs_len < max_runs;
            *at_mut!(order.runs, order.runs_len) = r;
            order.runs_len += 1;
        }
        r += 1;
    }
    order.sorted = few_runs && order.runs_len == 1;
    order.strictly_descending = descending;
    if few_runs {
        *at_mut!(order.runs, order.runs_len) = rows.count;
        order.runs_len += 1;
    } else {
        order.runs_len = 0;
    }
    order
}

/// The positions of the bytes which differ between any two rows.
fn find_varying_bytes(allocator: &dyn Allocator, rows: Rows) -> Result<SlowVec<u32>, AllocError> {
    let stride = rows.stride;
    let mut differs = SlowVec::<u8>::zeroed(allocator, stride)?;
    if stride >= 8 {
        // A word at a time, the last of which may overlap the one before it.
        let words = stride.div_ceil(8);
        let word_at = |w: usize| (w * 8).min(stride - 8);
        let word = |row: &[u8], w: usize| {
            let mut bytes = [0; 8];
            bytes.copy_from_slice(at!(row, word_at(w)..word_at(w) + 8));
            u64::from_ne_bytes(bytes)
        };
        let first = rows.get(0);
        let mut diff = SlowVec::<u64>::zeroed(allocator, words)?;
        // XOR each row with the first and gather the differences. A byte is
        // the same in every row exactly when none of its bits are set.
        for r in 0..rows.count {
            let row = rows.get(r);
            for (w, d) in diff.iter_mut().enumerate() {
                *d |= word(row, w) ^ word(first, w);
            }
        }
        for (w, d) in diff.iter().enumerate() {
            for (b, byte) in d.to_ne_bytes().into_iter().enumerate() {
                *at_mut!(differs, word_at(w) + b) |= u8::from(byte != 0);
            }
        }
    } else {
        for r in 0..rows.count {
            for (b, (byte, first)) in rows.get(r).iter().zip(rows.get(0)).enumerate() {
                *at_mut!(differs, b) |= u8::from(byte != first);
            }
        }
    }
    let count = differs.iter().filter(|&&d| d != 0).count();
    let mut varying = SlowVec::fixed(allocator, count)?;
    for (b, &d) in differs.iter().enumerate() {
        if d != 0 {
            #[expect(clippy::cast_possible_truncation, reason = "a stride is at most 65,535")]
            varying.push(b as u32)?;
        }
    }
    Ok(varying)
}

/// Packs the bytes of `row` at `varying`, most significant first, into an
/// integer which orders as they do.
fn pack_key(row: &[u8], varying: &[u32]) -> u64 {
    varying.iter().fold(0, |key, &b| (key << 8) | u64::from(*at!(row, b as usize)))
}

/// Merges pairs of the runs `tokens` starts with at `runs`, earlier run
/// first so ties keep their order, until one is left. Returns whether the
/// result is in `scratch` rather than `tokens`.
fn merge_runs(
    tokens: &mut [KeyedIndex],
    scratch: &mut [KeyedIndex],
    runs: &mut ExistingOrder,
) -> bool {
    let (mut source, mut dest) = (tokens, scratch);
    let mut in_scratch = false;
    let runs_len = &mut runs.runs_len;
    let runs = &mut runs.runs;
    while *runs_len > 2 {
        let mut kept = 0;
        let mut r = 0;
        while r + 2 < *runs_len {
            let (start, mid, end) = (*at!(runs, r), *at!(runs, r + 1), *at!(runs, r + 2));
            let (mut a, mut b) = (start as usize, mid as usize);
            let (mid, end) = (mid as usize, end as usize);
            for out in at_mut!(dest, start as usize..end) {
                let left = b == end || (a < mid && at!(source, b).key >= at!(source, a).key);
                *out = if left { *at!(source, a) } else { *at!(source, b) };
                if left { a += 1 } else { b += 1 }
            }
            *at_mut!(runs, kept) = start;
            kept += 1;
            r += 2;
        }
        if r + 1 < *runs_len {
            // An odd last run has no partner: it's carried into the next pass.
            let (start, end) = (*at!(runs, r) as usize, *at!(runs, r + 1) as usize);
            at_mut!(dest, start..end).copy_from_slice(at!(source, start..end));
            *at_mut!(runs, kept) = *at!(runs, r);
            kept += 1;
        }
        *at_mut!(runs, kept) = *at!(runs, *runs_len - 1);
        *runs_len = kept + 1;
        core::mem::swap(&mut source, &mut dest);
        in_scratch = !in_scratch;
    }
    in_scratch
}

/// Sorts each run of tokens with equal keys on the varying bytes from
/// `from` on, 8 at a time, with the same sort as the first 8.
fn sort_ties(
    allocator: &dyn Allocator,
    tokens: &mut [KeyedIndex],
    rows: Rows,
    varying: &[u32],
    from: usize,
    spare: &mut [KeyedIndex],
) -> Result<(), AllocError> {
    let next = at!(varying, from..(from + MAX_KEY_BYTES).min(varying.len()));
    let mut i = 0;
    while i < tokens.len() {
        let mut j = i + 1;
        while j < tokens.len() && at!(tokens, j).key == at!(tokens, i).key {
            j += 1;
        }
        if j - i > 1 {
            let tied = at_mut!(tokens, i..j);
            for t in tied.iter_mut() {
                t.key = pack_key(rows.get(t.index), next);
            }
            let spare = at_mut!(spare, i..j);
            if tied.len() < INSERTION_SORT_TIES {
                insertion_sort(tied);
            } else {
                #[expect(clippy::cast_possible_truncation, reason = "at most 64 bits")]
                let key_bits = next.len() as u32 * 8;
                if stable_sort_keyed_indices(allocator, tied, spare, key_bits)? {
                    tied.copy_from_slice(spare);
                }
            }
            if varying.len() > from + MAX_KEY_BYTES {
                sort_ties(allocator, tied, rows, varying, from + MAX_KEY_BYTES, spare)?;
            }
        }
        i = j;
    }
    Ok(())
}

/// Stably sorts `indices` by the rows of `stride` bytes in `row_layout`,
/// the `i`th index's row the `i`th.
pub(crate) fn sort_row_layout(
    allocator: &dyn Allocator,
    row_layout: &[u8],
    stride: u32,
    indices: &mut [u32],
) -> Result<(), AllocError> {
    let Ok(count) = u32::try_from(indices.len()) else { failed!() };
    if count <= 1 {
        return Ok(());
    }
    let rows = Rows { data: row_layout, stride: stride as usize, count };
    check!(row_layout.len() >= count as usize * rows.stride);
    // 1. Reuse the order the rows are already in, where it helps: keep where
    // runs start, to merge them, if there are few.
    let mut order = find_existing_order(rows);
    if order.sorted {
        return Ok(());
    }
    if order.strictly_descending {
        // No two rows are equal, so reversing keeps ties in order.
        indices.reverse();
        return Ok(());
    }
    // 2. Pack the first 8 varying bytes into keys, so sorting them doesn't
    // read rows through their offsets again and again.
    let varying = find_varying_bytes(allocator, rows)?;
    let first = at!(varying, ..varying.len().min(MAX_KEY_BYTES));
    let mut tokens = SlowVec::<KeyedIndex>::zeroed(allocator, indices.len())?;
    for (r, (token, &value)) in (0..count).zip(tokens.iter_mut().zip(indices.iter())) {
        *token = KeyedIndex { key: pack_key(rows.get(r), first), index: r, value };
    }
    let mut spare = SlowVec::<KeyedIndex>::zeroed(allocator, indices.len())?;
    // 3. Merge the runs, or sort, keeping ties in input order.
    let in_spare = if order.runs_len > 0 {
        merge_runs(&mut tokens, &mut spare, &mut order)
    } else {
        #[expect(clippy::cast_possible_truncation, reason = "at most 64 bits")]
        let key_bits = first.len() as u32 * 8;
        stable_sort_keyed_indices(allocator, &mut tokens, &mut spare, key_bits)?
    };
    let (sorted, free) =
        if in_spare { (&mut spare, &mut tokens) } else { (&mut tokens, &mut spare) };
    // 4. If the keys left out varying bytes, equal keys may still be
    // different rows: sort each tied group on the bytes after the first 8.
    if varying.len() > MAX_KEY_BYTES {
        sort_ties(allocator, sorted, rows, &varying, MAX_KEY_BYTES, free)?;
    }
    for (index, token) in indices.iter_mut().zip(sorted.iter()) {
        *index = token.value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;

    /// `count` rows of `stride` bytes, of a few values so many tie, with
    /// `fixed` bytes the same in every row.
    fn rows(count: u32, stride: usize, fixed: &[usize]) -> Vec<u8> {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut rows = Vec::new();
        for _ in 0..count {
            for b in 0..stride {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                rows.push(if fixed.contains(&b) { 7 } else { (seed % 3) as u8 });
            }
        }
        rows
    }

    fn expect_sorted(rows: &[u8], stride: usize) {
        let count = u32::try_from(rows.len() / stride).unwrap();
        let row = |i: u32| &rows[i as usize * stride..(i as usize + 1) * stride];
        // The indices are the rows' own, shifted, to see values carried.
        let mut indices: Vec<u32> = (0..count).map(|i| i + 100).collect();
        sort_row_layout(&Heap, rows, u32::try_from(stride).unwrap(), &mut indices).unwrap();
        let mut expected: Vec<u32> = (0..count).collect();
        expected.sort_by(|&a, &b| row(a).cmp(row(b)));
        let expected: Vec<u32> = expected.iter().map(|i| i + 100).collect();
        assert_eq!(indices, expected, "{count} rows of {stride} bytes");
    }

    #[test]
    #[cfg_attr(miri, ignore = "slow under Miri")]
    fn sorts_rows_stably() {
        // Short and long rows, rows needing ties sorted on later bytes, few
        // rows, and many, sorted by radix.
        for (count, stride, fixed) in [
            (40, 3, &[][..]),
            (40, 9, &[0, 4]),
            (500, 21, &[1, 2, 3, 20]),
            (5000, 9, &[]),
            (2, 8, &[]),
        ] {
            expect_sorted(&rows(count, stride, fixed), stride);
        }
    }

    #[test]
    fn uses_the_order_rows_are_in() {
        // Sorted, strictly descending, and a few ascending runs to merge.
        let ascending: Vec<u8> = (0..=255).collect();
        expect_sorted(&ascending, 1);
        let descending: Vec<u8> = (0..=255).rev().collect();
        expect_sorted(&descending, 1);
        let mut runs: Vec<u8> = Vec::new();
        for run in 0..3_u8 {
            runs.extend((0..=255_u8).map(|i| i / 2 + run));
        }
        expect_sorted(&runs, 1);
    }
}
