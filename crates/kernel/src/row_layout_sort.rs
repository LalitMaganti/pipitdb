//! `sort_row_layout`: stably sorts rows laid out by a `RowLayout`, as
//! Perfetto's `core/util/row_layout_sort.cc` does. It sorts on the bytes
//! which differ between rows, as many as fit packed above the row's number
//! in one word, so that sorting doesn't read the rows again. Rows which
//! agree on those bytes are then sorted on the next, and so on, with the
//! same sort.

use core::cmp::Ordering;

use crate::allocator::{AllocError, Allocator};
use crate::radix_sort::{insertion_sort, merge, sort_words};
use crate::slow_vec::SlowVec;

/// Up to this many ascending runs, of this many rows on average, are merged
/// rather than sorted.
const MAX_MERGED_RUNS: usize = 32;
const ROWS_PER_MERGED_RUN: usize = 256;

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

/// How a row's bytes and number are packed into a word.
#[derive(Clone, Copy)]
struct Words {
    /// The low bits, which hold the row's number.
    row_bits: u32,
    /// How many of its varying bytes a word holds above them.
    key_bytes: usize,
}

impl Words {
    /// For at least 2 rows.
    fn of(count: u32) -> Words {
        let row_bits = u32::BITS - (count - 1).leading_zeros();
        Words { row_bits, key_bytes: ((u64::BITS - row_bits) / 8) as usize }
    }

    /// The word of row `r`, `row`, holding its bytes at `varying`, at most
    /// `key_bytes`.
    fn word(self, row: &[u8], varying: &[u32], r: u32) -> u64 {
        pack_key(row, varying) << self.row_bits | u64::from(r)
    }

    fn key(self, word: u64) -> u64 {
        word >> self.row_bits
    }

    fn row(self, word: u64) -> u32 {
        #[expect(clippy::cast_possible_truncation, reason = "at most 31 bits")]
        let row = (word & ((1 << self.row_bits) - 1)) as u32;
        row
    }

    /// Sorts `words`, those with equal keys in order already, on their keys
    /// of `key_bytes` bytes. Returns whether they're in `scratch` instead.
    fn sort(
        self,
        allocator: &dyn Allocator,
        words: &mut [u64],
        scratch: &mut [u64],
        key_bytes: usize,
    ) -> Result<bool, AllocError> {
        #[expect(clippy::cast_possible_truncation, reason = "at most 64 bits")]
        let key_bits = key_bytes as u32 * 8;
        sort_words(allocator, words, scratch, self.row_bits, key_bits)
    }
}

/// Merges pairs of the runs `words` starts with at `runs` until one is
/// left. Returns whether the result is in `scratch` rather than `words`.
fn merge_runs(words: &mut [u64], scratch: &mut [u64], order: &mut ExistingOrder) -> bool {
    let (mut source, mut dest) = (words, scratch);
    let mut in_scratch = false;
    let ExistingOrder { runs, runs_len, .. } = order;
    while *runs_len > 2 {
        let mut kept = 0;
        let mut r = 0;
        while r + 2 < *runs_len {
            let [start, mid, end] = [r, r + 1, r + 2].map(|i| *at!(runs, i) as usize);
            merge(source, dest, start, mid, end);
            *at_mut!(runs, kept) = *at!(runs, r);
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

/// Sorts each run of words with equal keys on the varying bytes from
/// `from` on, as many at a time as a word holds, with the same sort.
fn sort_ties(
    allocator: &dyn Allocator,
    words: &mut [u64],
    rows: Rows,
    varying: &[u32],
    from: usize,
    packing: Words,
    spare: &mut [u64],
) -> Result<(), AllocError> {
    let next = at!(varying, from..(from + packing.key_bytes).min(varying.len()));
    let mut i = 0;
    while i < words.len() {
        let mut j = i + 1;
        while j < words.len() && packing.key(*at!(words, j)) == packing.key(*at!(words, i)) {
            j += 1;
        }
        if j - i > 1 {
            let tied = at_mut!(words, i..j);
            for word in tied.iter_mut() {
                let r = packing.row(*word);
                *word = packing.word(rows.get(r), next, r);
            }
            let spare = at_mut!(spare, i..j);
            if tied.len() < INSERTION_SORT_TIES {
                insertion_sort(tied);
            } else if packing.sort(allocator, tied, spare, next.len())? {
                tied.copy_from_slice(spare);
            }
            if varying.len() > from + packing.key_bytes {
                let from = from + packing.key_bytes;
                sort_ties(allocator, tied, rows, varying, from, packing, spare)?;
            }
        }
        i = j;
    }
    Ok(())
}

/// Stably sorts `order`, the numbers of the rows of `stride` bytes in
/// `row_layout` in the order they're laid out. Frees the rows once sorting
/// no longer needs them, before the memory the rest of the sort takes.
pub(crate) fn sort_row_layout(
    allocator: &dyn Allocator,
    row_layout: &mut SlowVec<u8>,
    stride: u32,
    order: &mut [u32],
) -> Result<(), AllocError> {
    let Ok(count) = u32::try_from(order.len()) else { failed!() };
    if count <= 1 {
        return Ok(());
    }
    let rows = Rows { data: row_layout, stride: stride as usize, count };
    check!(row_layout.len() >= count as usize * rows.stride);
    // 1. Reuse the order the rows are already in, where it helps: keep where
    // runs start, to merge them, if there are few.
    let mut existing = find_existing_order(rows);
    if existing.sorted {
        return Ok(());
    }
    if existing.strictly_descending {
        // No two rows are equal, so reversing keeps ties in order.
        order.reverse();
        return Ok(());
    }
    // 2. Pack the varying bytes a word holds above each row's number, so
    // sorting them doesn't read rows through their offsets again and again,
    // and the words of rows with equal bytes are in order.
    let varying = find_varying_bytes(allocator, rows)?;
    let packing = Words::of(count);
    let first = at!(varying, ..varying.len().min(packing.key_bytes));
    let words = (0..count).map(|r| packing.word(rows.get(r), first, r));
    let mut words = SlowVec::fixed_from(allocator, words)?;
    if varying.len() <= packing.key_bytes {
        // Every byte that orders the rows is in their words.
        row_layout.free();
    }
    let mut spare = SlowVec::<u64>::zeroed(allocator, order.len())?;
    // 3. Merge the runs, or sort.
    let in_spare = if existing.runs_len > 0 {
        merge_runs(&mut words, &mut spare, &mut existing)
    } else {
        packing.sort(allocator, &mut words, &mut spare, first.len())?
    };
    let (sorted, free) = if in_spare { (&mut spare, &mut words) } else { (&mut words, &mut spare) };
    // 4. If the words left out varying bytes, equal keys may still be
    // different rows: sort each tied group on the bytes after them.
    if varying.len() > packing.key_bytes {
        let rows = Rows { data: row_layout, stride: stride as usize, count };
        sort_ties(allocator, sorted, rows, &varying, packing.key_bytes, packing, free)?;
    }
    for (row, &word) in order.iter_mut().zip(sorted.iter()) {
        *row = packing.row(word);
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
        let mut indices: Vec<u32> = (0..count).collect();
        let mut row_layout = SlowVec::new(&Heap, 1 << 20).unwrap();
        row_layout.extend_from_slice(rows).unwrap();
        let stride = u32::try_from(stride).unwrap();
        sort_row_layout(&Heap, &mut row_layout, stride, &mut indices).unwrap();
        let mut expected: Vec<u32> = (0..count).collect();
        expected.sort_by(|&a, &b| row(a).cmp(row(b)));
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
