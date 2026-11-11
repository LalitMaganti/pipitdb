//! `sort_row_layout`: stably sorts rows laid out by a `RowLayout`, as
//! Perfetto's `core/util/row_layout_sort.cc` does. It sorts on the bytes
//! which differ between rows, as many as fit packed above the row's number
//! in one word, so that sorting doesn't read the rows again. Rows which
//! agree on those bytes are then sorted on the next, and so on, with the
//! same sort.

use core::cmp::Ordering;

use crate::allocator::{AllocError, Allocator};
use crate::radix_sort::{Scratch, merge, sort_words};
use crate::row_batch::BATCH_ROWS_MAX;
use crate::row_blocks::RowBlocks;
use crate::row_layout::Slot;
use crate::slow_vec::SlowVec;

/// Up to this many ascending runs, of this many rows on average, are merged
/// rather than sorted.
const MAX_MERGED_RUNS: usize = 32;
const ROWS_PER_MERGED_RUN: usize = 256;

/// Rows of `stride` bytes each.
#[derive(Clone, Copy)]
struct Rows<'a> {
    blocks: &'a RowBlocks,
    stride: usize,
    count: u32,
}

impl<'a> Rows<'a> {
    fn get(self, row: u32) -> &'a [u8] {
        self.blocks.get(row)
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
pub(crate) struct Words {
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

    pub(crate) fn row(self, word: u64) -> u32 {
        #[expect(clippy::cast_possible_truncation, reason = "at most 31 bits")]
        let row = (word & ((1 << self.row_bits) - 1)) as u32;
        row
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
    words: &mut [u64],
    rows: Rows,
    varying: &[u32],
    from: usize,
    packing: Words,
    scratch: &mut Scratch,
) {
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
            sort_words(tied, scratch);
            if varying.len() > from + packing.key_bytes {
                let from = from + packing.key_bytes;
                sort_ties(tied, rows, varying, from, packing, scratch);
            }
        }
        i = j;
    }
}

/// Rows in order.
pub(crate) enum Sorted {
    /// The `count` rows in the order they're held, or reversed.
    Held { reverse: bool, count: u32 },
    /// Each row's word, in order. If they hold every byte that varies, at
    /// `varying`, `fixed` is the bytes of the first row, as every other byte
    /// is in every row.
    Words { words: SlowVec<u64>, packing: Words, varying: SlowVec<u32>, fixed: Option<SlowVec<u8>> },
}

impl Sorted {
    /// The row `i`th in order.
    pub(crate) fn row(&self, i: u32) -> u32 {
        match self {
            Sorted::Held { reverse: false, .. } => i,
            Sorted::Held { reverse: true, count } => count - 1 - i,
            Sorted::Words { words, packing, .. } => packing.row(*at!(words, i as usize)),
        }
    }

    /// The words, and how to read `slot`, of 8-byte values of floats if
    /// `float`, from them, if they hold every key.
    pub(crate) fn slot_in_words(&self, slot: Slot, float: bool) -> Option<(&[u64], SlotInWords)> {
        let Sorted::Words { words, packing, varying, fixed: Some(fixed) } = self else {
            return None;
        };
        Some((words, SlotInWords::new(*packing, varying, fixed, slot, float)))
    }
}

/// How to read a slot of 8-byte values from the words of rows whose every
/// key they hold: its bytes are `fixed`'s, but for those the word's key
/// holds.
pub(crate) struct SlotInWords {
    packing: Words,
    slot: Slot,
    float: bool,
    /// The value's bytes, big-endian, where the key doesn't hold them, and
    /// its flag's.
    value: u64,
    flag: u8,
    /// Where the flag is in the key, and its bits there: none if it doesn't
    /// vary.
    flag_shift: u32,
    flag_mask: u8,
    /// Each run of the value's bytes the key holds: where it is in the key,
    /// where in the value, and its bits.
    runs: [(u32, u32, u64); 8],
    len: usize,
}

impl SlotInWords {
    pub(crate) fn new(
        packing: Words,
        varying: &[u32],
        fixed: &[u8],
        slot: Slot,
        float: bool,
    ) -> SlotInWords {
        let flag_at = usize::from(slot.offset);
        let start = flag_at + usize::from(slot.nullable);
        let mut bytes = [0; 8];
        bytes.copy_from_slice(at!(fixed, start..start + 8));
        let (mut value, flag) =
            (u64::from_be_bytes(bytes), if slot.nullable { *at!(fixed, flag_at) } else { 0 });
        let runs = [(0, 0, 0); 8];
        let (flag_shift, flag_mask) = (0, 0);
        let mut in_words = SlotInWords {
            packing,
            slot,
            float,
            value: 0,
            flag,
            flag_shift,
            flag_mask,
            runs,
            len: 0,
        };
        #[expect(clippy::cast_possible_truncation, reason = "at most 8 bytes")]
        let count = varying.len() as u32;
        for (v, &b) in (0_u32..).zip(varying) {
            let (b, from) = (b as usize, (count - 1 - v) * 8);
            if slot.nullable && b == flag_at {
                (in_words.flag_shift, in_words.flag_mask) = (from, 0xff);
            }
            if !(start..start + 8).contains(&b) {
                continue;
            }
            #[expect(clippy::cast_possible_truncation, reason = "within a value")]
            let to = (7 - (b - start) as u32) * 8;
            value &= !(0xff << to);
            // A byte after the last of a run, in both, extends it.
            let last = in_words.len.checked_sub(1).map(|r| at_mut!(in_words.runs, r));
            match last {
                Some((run_from, run_to, mask)) if *run_from == from + 8 && *run_to == to + 8 => {
                    (*run_from, *run_to, *mask) = (from, to, *mask << 8 | 0xff);
                }
                _ => {
                    *at_mut!(in_words.runs, in_words.len) = (from, to, 0xff);
                    in_words.len += 1;
                }
            }
        }
        in_words.value = value;
        in_words
    }

    /// Writes the value of each of `words`' rows into `out`, if none is null
    /// and their bytes are one run in the key, returning whether it did.
    pub(crate) fn decode_all(&self, words: &[u64], out: &mut [i64]) -> bool {
        let invert = if self.slot.descending { u64::MAX } else { 0 };
        #[expect(clippy::cast_possible_truncation, reason = "a byte of all ones or none")]
        let present = !self.slot.nullable || self.flag ^ invert as u8 != 0;
        if self.flag_mask != 0 || !present || self.len != 1 {
            return false;
        }
        let ((from, to, mask), float) = (*at!(self.runs, 0), self.float);
        for (out, &word) in out.iter_mut().zip(words) {
            let value = self.value | (self.packing.key(word) >> from & mask) << to;
            let bits = (value ^ invert).cast_signed() ^ i64::MIN;
            let flip = if float { ((bits >> 63).cast_unsigned() >> 1).cast_signed() } else { 0 };
            *out = bits ^ flip;
        }
        true
    }

    /// The value `word`'s row has in the slot, an integer or a float's
    /// bits; `None` if none.
    #[inline]
    pub(crate) fn decode(&self, word: u64) -> Option<i64> {
        let key = self.packing.key(word);
        let mut value = self.value;
        if self.len == 1 {
            let (from, to, mask) = *at!(self.runs, 0);
            value |= (key >> from & mask) << to;
        } else {
            for &(from, to, mask) in at!(self.runs, ..self.len) {
                value |= (key >> from & mask) << to;
            }
        }
        #[expect(clippy::cast_possible_truncation, reason = "a byte")]
        let flag = self.flag & !self.flag_mask | (key >> self.flag_shift) as u8 & self.flag_mask;
        let invert = if self.slot.descending { u64::MAX } else { 0 };
        #[expect(clippy::cast_possible_truncation, reason = "a byte of all ones or none")]
        if self.slot.nullable && flag ^ invert as u8 == 0 {
            return None;
        }
        let bits = (value ^ invert).cast_signed() ^ i64::MIN;
        if !self.float {
            return Some(bits);
        }
        // A negative float's bits but its sign were flipped by `Word::encode`.
        Some(bits ^ ((bits >> 63).cast_unsigned() >> 1).cast_signed())
    }
}

/// Stably sorts the rows laid out in `row_layout`. Frees them once sorting
/// no longer needs them, before the memory the rest of the sort takes.
pub(crate) fn sort_row_layout(
    allocator: &dyn Allocator,
    row_layout: &mut RowBlocks,
) -> Result<Sorted, AllocError> {
    let count = row_layout.len();
    let rows = Rows { blocks: row_layout, stride: row_layout.stride(), count };
    // 1. Reuse the order the rows are already in, where it helps: keep where
    // runs start, to merge them, if there are few.
    let mut existing = find_existing_order(rows);
    if count <= 1 || existing.sorted || existing.strictly_descending {
        // No two rows are equal when strictly descending, so reversing keeps
        // ties in order.
        return Ok(Sorted::Held { reverse: existing.strictly_descending, count });
    }
    // 2. Pack the varying bytes a word holds above each row's number, so
    // sorting them doesn't read rows through their offsets again and again,
    // and the words of rows with equal bytes are in order.
    let varying = find_varying_bytes(allocator, rows)?;
    let packing = Words::of(count);
    let first = at!(varying, ..varying.len().min(packing.key_bytes));
    // If every byte that orders the rows is in their words, the rows aren't
    // read again: each block is let go of once its words are written.
    let in_words = varying.len() <= packing.key_bytes;
    let mut fixed = None;
    if in_words {
        fixed = Some(SlowVec::fixed_from(allocator, rows.get(0).iter().copied())?);
    }
    let mut words = SlowVec::fixed(allocator, count as usize)?;
    for from in (0..count).step_by(BATCH_ROWS_MAX as usize) {
        let to = count.min(from + BATCH_ROWS_MAX);
        words.extend_within((from..to).map(|r| packing.word(row_layout.get(r), first, r)));
        if in_words {
            row_layout.let_go_below(to, true);
        }
    }
    if in_words {
        row_layout.free();
    }
    let mut scratch = Scratch::new(allocator, count as usize)?;
    // 3. Merge the runs, or sort.
    if existing.runs_len > 0 {
        let mut other = SlowVec::<u64>::zeroed(allocator, count as usize)?;
        if merge_runs(&mut words, &mut other, &mut existing) {
            words.copy_from_slice(&other);
        }
    } else {
        sort_words(&mut words, &mut scratch);
    }
    // 4. If the words left out varying bytes, equal keys may still be
    // different rows: sort each tied group on the bytes after them.
    if varying.len() > packing.key_bytes {
        let rows = Rows { blocks: row_layout, stride: row_layout.stride(), count };
        sort_ties(&mut words, rows, &varying, packing.key_bytes, packing, &mut scratch);
    }
    Ok(Sorted::Words { words, packing, varying, fixed })
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;
    use crate::context::Context;

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
        let mut context = Context::new(&Heap);
        let mut row_layout = RowBlocks::new(&context, stride).unwrap();
        row_layout
            .extend(&mut context, count, |first, bytes| {
                let at = first as usize * stride;
                bytes.copy_from_slice(&rows[at..at + bytes.len()]);
            })
            .unwrap();
        let sorted = sort_row_layout(&Heap, &mut row_layout).unwrap();
        let indices: Vec<u32> = (0..count).map(|i| sorted.row(i)).collect();
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
