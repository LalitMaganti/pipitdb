//! `RowStore`: the rows a breaker holds on to, to hand back later in any
//! order, as Perfetto's `core/exec/row_store` holds them.

#![allow(unsafe_code)]

use crate::allocator::AllocError;
use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType, Form, GatherSource};
use crate::containers::{Array, BitSlice, SmallVec};
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::selection::Kept;
use core::mem::MaybeUninit;
use core::ops::Range;
use core::ptr;

/// The bits of a row's place that say which row of its batch it is. A
/// dense row's place is its index, so a full batch must be `1 << ROW_BITS`
/// rows.
const ROW_BITS: u32 = BATCH_ROWS_MAX.trailing_zeros();
const _: () = assert!(BATCH_ROWS_MAX.is_power_of_two());

/// Keeps the bits of a row's place that say which row of its batch it is.
const ROW_MASK: u32 = (1 << ROW_BITS) - 1;

/// The most batches a store holds, as a place is 32 bits: with full
/// batches, about 4 billion rows.
const BATCHES_MAX: usize = 1 << (32 - ROW_BITS);

/// `scatter` puts this many output rows in order at a time, so a column's
/// values for them stay in cache.
const REGION_BITS: u32 = 16;

/// `scatter` copies only between these many rows, the most whose numbers
/// fit the bits of a word below `POSITION_SHIFT`, and only once it can copy
/// this many columns.
const SCATTER_ROWS: Range<usize> = 1 << 16..1 << POSITION_SHIFT;
const SCATTER_COLUMNS_MIN: usize = 3;

/// Where `scatter` keeps, in each row's word, where that row goes, and in
/// each slot's word, where in its region the value that arrived in the slot
/// goes.
const POSITION_SHIFT: u32 = 24;
const OFFSET_SHIFT: u32 = 48;
const POSITION_MASK: u64 = (1 << POSITION_SHIFT) - 1;

/// Rows are numbered from 0, in the order they were appended. Every column
/// appended must be flat, as planning makes the columns a breaker holds.
///
/// Holding rows never costs much more than the rows themselves. A batch
/// which keeps at least half a batch's worth of rows, or whose columns all
/// borrow what they view, is held by sharing its buffers: nothing is
/// copied. The rows any other batch keeps are packed, a full batch at a
/// time, into buffers from the run's `Context`, so a few rows never keep
/// whole buffers alive.
pub(crate) struct RowStore {
    /// Each batch held, by its columns.
    held: SmallVec<GatherSource>,
    /// How many batches are held.
    batches: u32,
    /// How many columns each batch held has.
    width: u32,
    /// How many rows, from the first, are in batches held whole: each full
    /// but maybe the last, and keeping every row. A row's place is its number.
    dense: u32,
    /// Where each row held after the `dense` ones is: its batch, shifted up
    /// `ROW_BITS`, and its row in it.
    places: SmallVec<u32>,
    /// By column, the rows being packed, until there are a batch's worth or a
    /// batch is shared after them.
    packing: SmallVec<Packing>,
    /// How many rows are being packed.
    packed: u32,
    /// By column, a bit for whether any batch held has nulls in it; the
    /// last bit is for every column from there on.
    nullable: u64,
    /// By column, a bit for whether it isn't held, as whoever gathers it
    /// gives its values.
    dropped: u64,
    /// By column dropped, a batch's worth of words never read, held in its
    /// place.
    placeholders: SmallVec<ColumnView>,
    /// By column, a bit for whether `scatter` copied it.
    scattered: u64,
    /// The columns `scatter` copied, in output order: by column copied, a
    /// view of each batch's worth.
    copies: SmallVec<ColumnView>,
}

impl RowStore {
    pub(crate) fn new(env: &Env) -> Result<RowStore, AllocError> {
        Ok(RowStore {
            held: SmallVec::new(env, 1 << 31)?,
            batches: 0,
            width: 0,
            dense: 0,
            places: SmallVec::new(env, 1 << 31)?,
            packing: SmallVec::new(env, 1 << 6)?,
            packed: 0,
            nullable: 0,
            dropped: 0,
            placeholders: SmallVec::new(env, 1 << 6)?,
            scattered: 0,
            copies: SmallVec::new(env, 1 << 24)?,
        })
    }

    /// How many rows are held.
    #[inline(never)]
    pub(crate) fn len(&self) -> u32 {
        #[expect(clippy::cast_possible_truncation, reason = "at most 2^31")]
        let len = self.dense + self.places.len() as u32 + self.packed;
        len
    }

    /// Holds the rows `batch` keeps. Every batch must have the same columns.
    pub(crate) fn append(&mut self, context: &mut Context, batch: &RowBatch) -> Result<(), Error> {
        let count = batch.selection().len();
        if count == 0 {
            return Ok(());
        }
        check!((0..batch.column_count()).all(|c| matches!(batch.column(c).form(), Form::Flat)));
        if self.len().checked_add(count).is_none_or(|len| len > 1 << 31) {
            return Err(Error::OutOfMemory);
        }
        if shares(batch) {
            // Rows packed before come first.
            self.flush(context)?;
            return self.hold(batch);
        }
        let columns = (0..batch.column_count()).map(|c| batch.column(c));
        let kept = batch.selection().kept();
        let mut done = 0;
        while done < count {
            let take = (count - done).min(BATCH_ROWS_MAX - self.packed);
            if self.packing.is_empty() {
                for column in columns.clone() {
                    self.packing.push(Packing::new(context, column.data_type())?)?;
                }
            }
            for (packing, column) in self.packing.iter_mut().zip(columns.clone()) {
                packing.append(column, kept, done..done + take)?;
            }
            (self.packed, done) = (self.packed + take, done + take);
            if self.packed == BATCH_ROWS_MAX {
                self.flush(context)?;
            }
        }
        Ok(())
    }

    /// Stops holding column `column` of the rows appended from now on, which
    /// must be all of them: gathering it needs its values given.
    pub(crate) fn drop_column(&mut self, context: &mut Context, column: u32) -> Result<(), Error> {
        check!(column < 63 && self.len() == 0 && self.dropped >> column & 1 == 0);
        // Of any type, as it's never read.
        let nothing = context.values_buffer(BATCH_ROWS_MAX as usize * 8)?;
        self.placeholders.push(ColumnView::new(context, DataType::Int64, nothing, None)?)?;
        self.dropped |= 1 << column;
        Ok(())
    }

    /// Whether any row held in column `column` is null.
    pub(crate) fn has_nulls(&self, column: u32) -> bool {
        self.nullable >> column.min(63) & 1 != 0
    }

    /// Whether column `column` is in output order.
    pub(crate) fn is_scattered(&self, column: u32) -> bool {
        column < 63 && self.scattered >> column & 1 != 0
    }

    /// Which column dropped column `column` is, if it is.
    fn dropped(&self, column: usize) -> Option<usize> {
        (column < 63 && self.dropped >> column & 1 != 0)
            .then(|| (self.dropped & ((1 << column) - 1)).count_ones() as usize)
    }

    /// Holds `batch` as it is, sharing its buffers.
    fn hold(&mut self, batch: &RowBatch) -> Result<(), Error> {
        let held = self.batches;
        if held as usize == BATCHES_MAX {
            return Err(Error::OutOfMemory);
        }
        self.width = batch.column_count();
        for c in 0..batch.column_count() {
            let column = batch.column(c);
            let placeholder = self.dropped(c as usize).map(|k| at!(self.placeholders, k));
            self.held.push(GatherSource::of(placeholder.unwrap_or(column)))?;
            self.nullable |= u64::from(column.has_nulls()) << c.min(63);
        }
        let kept = batch.selection().kept();
        let whole = self.dense.is_multiple_of(BATCH_ROWS_MAX) && matches!(kept, Kept::Prefix);
        if whole && self.places.is_empty() {
            self.dense += batch.selection().len();
        } else {
            for i in 0..batch.selection().len() {
                self.places.push(held << ROW_BITS | kept.row(i))?;
            }
        }
        self.batches += 1;
        Ok(())
    }

    /// Holds the rows packed so far as one batch, so every row held is in a
    /// batch.
    pub(crate) fn flush(&mut self, context: &mut Context) -> Result<(), Error> {
        if self.packed == 0 {
            return Ok(());
        }
        let mut batch = RowBatch::new();
        batch.reset(self.packed);
        for packing in self.packing.iter() {
            let Ok(()) = batch.push_column(packing.take(context)?) else { failed!() };
        }
        while self.packing.pop().is_some() {}
        self.packed = 0;
        self.hold(&batch)
    }

    /// The string of the held row `row` in column `column`, a string column,
    /// if not null. Rows packed must be flushed first.
    pub(crate) fn string(&self, column: u32, row: u32) -> Option<&[u8]> {
        check!(self.packed == 0);
        let (held, row) = self.find(self.place(row));
        let values = at!(held, column as usize).view().values();
        (!values.is_null(row)).then(|| values.strings().get(row as usize))
    }

    /// Fills `out` with up to a batch's worth of `rows`, in that order and
    /// repeats included, returning how many. Only the rows `kept` keeps are
    /// read and written, but a string not kept is empty. Rows held one
    /// after another in one batch are viewed where they are held; any
    /// others are copied into buffers from `context`. `at` is where `rows`
    /// start in the order `scatter` was given, if it was.
    /// `given(context, column)` makes a dropped column's values.
    pub(crate) fn gather(
        &mut self,
        context: &mut Context,
        out: &mut RowBatch,
        (rows, kept): (&[u32], Kept),
        at: u32,
        given: &mut dyn FnMut(&mut Context, u32) -> Result<ColumnView, Error>,
    ) -> Result<u32, Error> {
        #[expect(clippy::cast_possible_truncation, reason = "at most a batch's")]
        let count = rows.len().min(BATCH_ROWS_MAX as usize) as u32;
        let rows = at!(rows, ..count as usize);
        out.reset(count);
        let Some(&first) = rows.first() else { return Ok(0) };
        self.flush(context)?;
        // The rows kept, by their place in `rows`.
        let mut every = [MaybeUninit::<u16>::uninit(); BATCH_ROWS_MAX as usize];
        let kept = match kept {
            Kept::Prefix => {
                let every = at_mut!(every, ..count as usize);
                for (r, slot) in (0..).zip(every.iter_mut()) {
                    slot.write(r);
                }
                // SAFETY: each was just written.
                unsafe { &*(ptr::from_ref(every) as *const [u16]) }
            }
            Kept::None => &[],
            Kept::Select(kept) => kept,
        };
        check!(kept.last().is_none_or(|&r| u32::from(r) < count));
        // Found once for every column, by row: a row held whole is its own
        // place.
        let mut found = [0_u32; BATCH_ROWS_MAX as usize];
        let places = if self.places.is_empty() {
            rows
        } else {
            for &r in kept {
                *at_mut!(found, usize::from(r)) = self.place(*at!(rows, usize::from(r)));
            }
            at!(found, ..count as usize)
        };
        // A run's first, or any held, as rows not kept may be of none.
        let first_place = kept.first().map_or(0, |&r| *at!(places, usize::from(r)));
        let (held, start) = self.find(first_place);
        let scattered = |c: usize| self.copied(c, at, count);
        // Every row, from `start` on in one batch, not running past its last.
        let run = kept.len() == count as usize
            && start + count <= 1 << ROW_BITS
            && (0..count)
                .zip(rows.iter().zip(places))
                .all(|(r, (&row, &place))| row == first + r && place == first_place + r);
        if run {
            for (c, held) in (0..).zip(held) {
                let column = match scattered(c as usize) {
                    Some(column) => column,
                    None if self.dropped(c as usize).is_some() => given(context, c)?,
                    None => held.view().slice(start, count),
                };
                let Ok(()) = out.push_column(column) else { failed!() };
            }
            return Ok(count);
        }
        // So every place, of a batch held, has its sources: `gather` relies on it.
        let width = self.width as usize;
        check!(self.held.len() == self.batches as usize * width);
        for c in 0..width {
            let column = match scattered(c) {
                Some(column) => column,
                #[expect(clippy::cast_possible_truncation, reason = "a column")]
                None if self.dropped(c).is_some() => given(context, c as u32)?,
                None => {
                    let nullable = self.nullable >> c.min(63) & 1 != 0;
                    let data_type = at!(held, c).view().data_type();
                    let source = (&*self.held, width, c);
                    let one = self.batches == 1;
                    gather(context, data_type, source, (places, kept), (nullable, one))?
                }
            };
            let Ok(()) = out.push_column(column) else { failed!() };
        }
        Ok(count)
    }

    /// Column `column`'s `count` values from `at` in the order `scatter` was
    /// given, if it copied the column.
    #[inline(never)]
    fn copied(&self, column: usize, at: u32, count: u32) -> Option<ColumnView> {
        if column >= 63 || self.scattered >> column & 1 == 0 {
            return None;
        }
        let k = (self.scattered & ((1 << column) - 1)).count_ones() as usize;
        check!(at.is_multiple_of(BATCH_ROWS_MAX));
        let batch = k * self.batches as usize + (at / BATCH_ROWS_MAX) as usize;
        Some(at!(self.copies, batch).slice(0, count))
    }

    /// Copies each column it can, of words with no nulls, into the order the
    /// rows will be gathered in, letting go of each held batch's values once
    /// they're copied. `words` are the rows in that order, each word's row
    /// `row_of` it, in bits below `POSITION_SHIFT`; it uses the bits above.
    /// Rows are appended to the region of output rows they go to, a held
    /// batch at a time, which reads and writes in order rather than at
    /// random; then each region, which fits in cache, is put in order.
    /// It only pays for many rows and a few columns. If `decode` is given,
    /// dropped columns are copied too, column `c`'s value of a row
    /// `decode(c, word)` of its word, before the words' bits are used; if
    /// any has nulls, nothing is copied.
    pub(crate) fn scatter(
        &mut self,
        context: &mut Context,
        words: &mut [u64],
        row_of: impl Fn(u64) -> u32,
        (by_word, mut decode): (bool, Option<impl FnMut(usize, u64) -> i64>),
    ) -> Result<(), Error> {
        self.flush(context)?;
        let (width, rows) = (self.width as usize, words.len());
        let env = context.env();
        // A bit for each column to copy, and each column's type.
        let (mut copy, mut types) = (0_u64, [DataType::Int64; 63]);
        for c in 0..width.min(63) {
            *at_mut!(types, c) = at!(self.held, c).view().data_type();
            if (self.nullable | self.dropped) >> c & 1 == 0 && !at!(types, c).has_bytes() {
                copy |= 1 << c;
            }
        }
        let dropped = if decode.is_some() { self.dropped } else { 0 };
        // Decoded by word before the rest, or by row with the rest.
        let (decoded, by_row) = if by_word { (dropped, 0) } else { (0, dropped) };
        let columns = |mask: u64| (0..63).filter(move |c| mask >> c & 1 != 0);
        // Every row is then held in order, a full batch at a time.
        let dense = self.dense as usize == rows && self.len() as usize == rows;
        if !dense
            || !SCATTER_ROWS.contains(&rows)
            || ((copy | self.dropped).count_ones() as usize) < SCATTER_COLUMNS_MIN
            || dropped & self.nullable != 0
        {
            return Ok(());
        }
        // 0. Columns decoded from the words, in order already: a bit for
        // each of the columns copied.
        let moved = copy | dropped;
        let mut copies = Copies::new(env, moved.count_ones() as usize, rows)?;
        let mut in_order = 0_u64;
        if let Some(decode) = decode.as_mut() {
            for (k, c) in columns(moved).enumerate().filter(|&(_, c)| decoded >> c & 1 != 0) {
                in_order |= 1 << k;
                copies.decode(context, k, |at| decode(c, *at!(words, at)))?;
            }
        }
        // 1. Where each row goes, in its own word.
        mark_positions(words, row_of);
        // 2. Each held batch's rows to the next slots of their regions, then
        // its values are let go of. A region's slots fill from its start, so
        // an output batch's buffer is taken when its first slot is: by then
        // about as many held batches are let go of, whose buffers it reuses.
        let mut next: [usize; 256] = core::array::from_fn(|r| r << REGION_BITS);
        // What's held in place of a column copied, which is never read.
        let nothing = context.values_buffer(BATCH_ROWS_MAX as usize * 8)?;
        let empty = ColumnView::new(context, DataType::Int64, nothing, None)?;
        let mut slots = [0_usize; BATCH_ROWS_MAX as usize];
        for b in 0..rows.div_ceil(BATCH_ROWS_MAX as usize) {
            let first = b * BATCH_ROWS_MAX as usize;
            let count = (rows - first).min(BATCH_ROWS_MAX as usize);
            for (slot, row) in slots.iter_mut().zip(first..first + count) {
                let position = (*at!(words, row) >> POSITION_SHIFT & POSITION_MASK) as usize;
                let region = position >> REGION_BITS;
                *slot = *at!(next, region);
                // So every slot is written once.
                check!(*slot < rows.min((region + 1) << REGION_BITS));
                *at_mut!(next, region) += 1;
                if slot.is_multiple_of(BATCH_ROWS_MAX as usize) {
                    copies.take(context, *slot)?;
                }
                let offset = (position & ((1 << REGION_BITS) - 1)) as u64;
                let word = at_mut!(words, *slot);
                *word = *word & !(u64::MAX << OFFSET_SHIFT) | offset << OFFSET_SHIFT;
            }
            for (k, c) in columns(moved).enumerate().filter(|&(_, c)| decoded >> c & 1 == 0) {
                if let (Some(decode), true) = (decode.as_mut(), by_row >> c & 1 != 0) {
                    for (row, &slot) in (first as u64..).zip(at!(slots, ..count)) {
                        copies.write(k, slot, decode(c, row));
                    }
                    continue;
                }
                let source = at!(self.held, b * width + c);
                for (row, &slot) in (0..).zip(at!(slots, ..count)) {
                    copies.write(k, slot, source.word(row));
                }
                *at_mut!(self.held, b * width + c) = GatherSource::of(&empty);
            }
        }
        // 3. Each region in order.
        copies.put_in_order(env, words, in_order)?;
        self.copies = copies.views(context, columns(moved).map(|c| *at!(types, c)))?;
        self.scattered = moved;
        Ok(())
    }

    /// Where row `row` is: its batch, shifted up `ROW_BITS`, and its row in it.
    fn place(&self, row: u32) -> u32 {
        if row < self.dense { row } else { *at!(self.places, (row - self.dense) as usize) }
    }

    /// The columns of the batch held at `place`, and its row there.
    fn find(&self, place: u32) -> (&[GatherSource], u32) {
        let width = self.width as usize;
        let first = (place >> ROW_BITS) as usize * width;
        (at!(self.held, first..first + width), place & ROW_MASK)
    }
}

/// Writes where in `words`' order each row is into its own word, above
/// its row's number.
fn mark_positions(words: &mut [u64], row_of: impl Fn(u64) -> u32) {
    for at in 0..words.len() {
        let word = at_mut!(words, row_of(*at!(words, at)) as usize);
        *word = *word & !(POSITION_MASK << POSITION_SHIFT) | (at as u64) << POSITION_SHIFT;
    }
}

/// What `scatter` copies columns into: by column copied, then batch of
/// output rows, a buffer once its first row is taken.
struct Copies {
    buffers: SmallVec<Option<Buffer>>,
    rows: usize,
    batches: usize,
}

impl Copies {
    fn new(env: &Env, columns: usize, rows: usize) -> Result<Copies, Error> {
        let batches = rows.div_ceil(BATCH_ROWS_MAX as usize);
        let buffers = SmallVec::fixed_from(env, (0..columns * batches).map(|_| None))?;
        Ok(Copies { buffers, rows, batches })
    }

    /// Takes every column's buffer for the batch output row `first`, a
    /// batch's first, is in, if not taken.
    #[inline(never)]
    fn take(&mut self, context: &mut Context, first: usize) -> Result<(), Error> {
        let batch = first / BATCH_ROWS_MAX as usize;
        let len = (self.rows - first).min(BATCH_ROWS_MAX as usize);
        for k in 0..self.buffers.len() / self.batches {
            let buffer = at_mut!(self.buffers, k * self.batches + batch);
            if buffer.is_none() {
                *buffer = Some(context.values_buffer(len * 8)?);
            }
        }
        Ok(())
    }

    /// Where column `k` holds output row `at`, whose batch's buffer is taken.
    fn slot(&self, k: usize, at: usize) -> *mut i64 {
        check!(at < self.rows);
        let Some(buffer) = at!(self.buffers, k * self.batches + at / BATCH_ROWS_MAX as usize)
        else {
            failed!()
        };
        // SAFETY: a batch's buffer holds a word for each row from its first
        // to the last, or a batch's.
        unsafe { buffer.as_non_null().cast::<i64>().as_ptr().add(at % BATCH_ROWS_MAX as usize) }
    }

    fn write(&self, k: usize, at: usize, value: i64) {
        // SAFETY: as in `slot`, and the buffer isn't shared yet.
        unsafe { self.slot(k, at).write(value) };
    }

    /// Takes column `k`'s buffers, and writes each output row `at` of it as
    /// `value(at)`.
    fn decode(
        &mut self,
        context: &mut Context,
        k: usize,
        mut value: impl FnMut(usize) -> i64,
    ) -> Result<(), Error> {
        for b in 0..self.batches {
            let len = (self.rows - b * BATCH_ROWS_MAX as usize).min(BATCH_ROWS_MAX as usize);
            *at_mut!(self.buffers, k * self.batches + b) = Some(context.values_buffer(len * 8)?);
        }
        for at in 0..self.rows {
            self.write(k, at, value(at));
        }
        Ok(())
    }

    /// Puts each region of output rows in order, through a copy that stays
    /// in cache: each slot's word in `words` says where in its region the
    /// value that arrived in it goes.
    /// Columns with a bit in `in_order` are left as they are.
    fn put_in_order(&self, env: &Env, words: &[u64], in_order: u64) -> Result<(), Error> {
        let mut values = Array::<i64>::zeroed(env, 1 << REGION_BITS)?;
        for first in (0..self.rows).step_by(1 << REGION_BITS) {
            let end = self.rows.min(first + (1 << REGION_BITS));
            for k in (0..self.buffers.len() / self.batches).filter(|k| in_order >> k & 1 == 0) {
                let values = at_mut!(values, ..end - first);
                for (value, at) in values.iter_mut().zip(first..) {
                    *value = self.read(k, at);
                }
                for (&value, &word) in values.iter().zip(at!(words, first..end)) {
                    self.write(k, first + (word >> OFFSET_SHIFT) as usize, value);
                }
            }
        }
        Ok(())
    }

    /// Each column, of the `types` given, as views of each output batch.
    fn views(
        &mut self,
        context: &mut Context,
        types: impl Iterator<Item = DataType>,
    ) -> Result<SmallVec<ColumnView>, Error> {
        let mut views = SmallVec::fixed(context.env(), self.buffers.len())?;
        for (data_type, buffers) in types.zip(self.buffers.chunks_mut(self.batches)) {
            for buffer in buffers {
                let Some(buffer) = buffer.take() else { failed!() };
                views.push(ColumnView::new(context, data_type, buffer, None)?)?;
            }
        }
        Ok(views)
    }

    /// Output row `at` of column `k`, once written.
    fn read(&self, k: usize, at: usize) -> i64 {
        // SAFETY: as in `slot`.
        unsafe { *self.slot(k, at) }
    }
}

/// Whether `batch` is held by sharing its buffers rather than packed: if it
/// keeps at least half a batch's worth of rows, or its columns all borrow
/// what they view, and each column's buffer has a batch's worth of words
/// from its first, as gathering reads.
fn shares(batch: &RowBatch) -> bool {
    let columns = 0..batch.column_count();
    (batch.selection().len() * 2 >= BATCH_ROWS_MAX || columns.clone().all(|c| batch.is_borrowed(c)))
        && columns.into_iter().all(|c| GatherSource::fits(batch.column(c)))
}

/// A column being packed: its values, up to a batch's, and nulls.
struct Packing {
    /// What type the column is.
    data_type: DataType,
    /// Each row's value: a word, or a string's view into `bytes`.
    values: Buffer,
    /// A bit for each row, set if it isn't null.
    validity: Buffer,
    /// Whether any row is null.
    nulls: bool,
    /// The strings' bytes, for a string column.
    bytes: Option<SmallVec<u8>>,
    /// How many rows are packed.
    rows: u32,
}

impl Packing {
    fn new(context: &mut Context, data_type: DataType) -> Result<Packing, AllocError> {
        let values = context.values_buffer(BATCH_ROWS_MAX as usize * 8)?;
        let mut validity =
            context.small_buffer(BitSlice::<u8>::words_for(BATCH_ROWS_MAX as usize))?;
        validity.as_mut_slice::<u8>().fill(0);
        let mut bytes = None;
        if data_type.has_bytes() {
            // An empty string first, which every null and empty string is.
            let empty = bytes.insert(SmallVec::new(context.env(), 1 << 31)?);
            empty.extend_from_slice(&[0; 4])?;
        }
        Ok(Packing { data_type, values, validity, nulls: false, bytes, rows: 0 })
    }

    /// Packs `column`'s rows `rows`, after those packed.
    fn append(&mut self, column: &ColumnView, kept: Kept, rows: Range<u32>) -> Result<(), Error> {
        let values = column.values();
        for i in rows {
            let row = kept.row(i);
            let at = self.rows as usize;
            let valid = !values.is_null(row);
            BitSlice::of_mut(self.validity.as_mut_slice::<u8>()).set_if(at, valid);
            self.nulls |= !valid;
            match &mut self.bytes {
                None => {
                    *at_mut!(self.values.as_mut_slice::<i64>(), at) =
                        *at!(values.words(), row as usize);
                }
                Some(bytes) => {
                    let string = if valid { values.strings().get(row as usize) } else { &[] };
                    *at_mut!(self.values.as_mut_slice::<u64>(), at) = append(bytes, string)?;
                }
            }
            self.rows += 1;
        }
        Ok(())
    }

    /// The rows packed, as a flat column.
    fn take(&self, context: &mut Context) -> Result<ColumnView, Error> {
        let values = self.values.clone();
        let validity = self.nulls.then(|| self.validity.clone());
        let column = match &self.bytes {
            None => ColumnView::new(context, self.data_type, values, validity)?,
            Some(packed) => {
                let mut bytes = context.bytes_buffer(packed.len())?;
                bytes.as_mut_slice::<u8>().copy_from_slice(packed);
                // SAFETY: `append` wrote each word where its string is.
                unsafe { ColumnView::strings_unchecked(context, values, bytes, validity)? }
            }
        };
        Ok(column.slice(0, self.rows))
    }
}

/// Appends `string` to `bytes`, which start with an empty string, after its
/// length, returning its word: the empty string's if it's empty.
fn append(bytes: &mut SmallVec<u8>, string: &[u8]) -> Result<u64, Error> {
    if string.is_empty() {
        return Ok(0);
    }
    let (Ok(at), Ok(len)) = (u32::try_from(bytes.len()), u32::try_from(string.len())) else {
        return Err(Error::OutOfMemory);
    };
    bytes.extend_from_slice(&len.to_le_bytes())?;
    bytes.extend_from_slice(string)?;
    Ok(u64::from(at))
}

/// A flat column of `data_type` holding, for each row `kept`, the row at
/// its place in `places` of its batch's column in `held`: the store's
/// columns, how many each batch has, and which column. `kept` increase,
/// the last below `places`' length. Other rows are unspecified, but a
/// string's is one of the column's. `nullable` is whether any has nulls,
/// and `one` whether there's one batch held, whose strings' bytes are then
/// shared rather than copied.
fn gather(
    context: &mut Context,
    data_type: DataType,
    (held, width, column): (&[GatherSource], usize, usize),
    (places, kept): (&[u32], &[u16]),
    (nullable, one): (bool, bool),
) -> Result<ColumnView, Error> {
    let rows = places.len();
    // This column of every batch held is every `width`th from here.
    let held = at!(held, column..);
    let found = |r: u16| {
        // SAFETY: as `kept` says.
        let place = *unsafe { places.get_unchecked(usize::from(r)) };
        let at = (place >> ROW_BITS) as usize * width;
        // SAFETY: a place is of a batch held, each with `width` sources.
        (unsafe { held.get_unchecked(at) }, place & ROW_MASK)
    };
    let mut values = context.values_buffer(rows * 8)?;
    let mut validity = None;
    if nullable {
        let mut buffer = context.small_buffer(BitSlice::<u8>::words_for(rows))?;
        let bits = buffer.as_mut_slice::<u8>();
        bits.fill(0);
        let mut nulls = false;
        for &r in kept {
            let (source, row) = found(r);
            let valid = source.is_valid(row);
            BitSlice::of_mut(&mut *bits).set_if(usize::from(r), valid);
            nulls |= !valid;
        }
        validity = nulls.then_some(buffer);
    }
    let words = at_mut!(values.as_mut_slice::<i64>(), ..rows);
    // A row not kept takes the first kept row's string, which is one.
    if kept.len() < rows && data_type.has_bytes() {
        let first = kept.first().map_or(0, |&r| {
            let (source, row) = found(r);
            source.word(row)
        });
        words.fill(first);
    }
    for &r in kept {
        let (source, row) = found(r);
        // SAFETY: as `kept` says.
        *unsafe { words.get_unchecked_mut(usize::from(r)) } = source.word(row);
    }
    if !data_type.has_bytes() {
        return Ok(ColumnView::new(context, data_type, values, validity)?);
    }
    if let (true, Some(bytes), false) = (one, at!(held, 0).bytes(), kept.is_empty()) {
        // SAFETY: each word is one of the batch's, valid when it was made.
        return Ok(unsafe {
            ColumnView::strings_unchecked(context, values, bytes.clone(), validity)?
        });
    }
    let words = at_mut!(values.as_mut_slice::<u64>(), ..rows);
    let string = |r: u16| {
        let (source, _) = found(r);
        source.string(*at!(words, usize::from(r)))
    };
    // An empty string first, which every other row is.
    let total = kept.iter().fold(4_u64, |total, &r| total + 4 + string(r).len() as u64);
    // So no word's offset is past 32 bits.
    let Ok(total) = u32::try_from(total) else { return Err(Error::OutOfMemory) };
    let mut bytes = context.bytes_buffer(total as usize)?;
    let out = bytes.as_mut_slice::<u8>();
    at_mut!(out, ..4).fill(0);
    let mut at = 4;
    let mut rebased = [0_u64; BATCH_ROWS_MAX as usize];
    for &r in kept {
        let string = string(r);
        let len = string.len();
        #[expect(clippy::cast_possible_truncation, reason = "below `total`")]
        at_mut!(out, at..at + 4).copy_from_slice(&(len as u32).to_le_bytes());
        at_mut!(out, at + 4..at + 4 + len).copy_from_slice(string);
        *at_mut!(rebased, usize::from(r)) = at as u64;
        at += 4 + len;
    }
    for (word, &rebased) in words.iter_mut().zip(&rebased) {
        *word = rebased;
    }
    // SAFETY: each word was written where its string is, just above.
    Ok(unsafe { ColumnView::strings_unchecked(context, values, bytes, validity)? })
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::vec::Vec;

    use super::*;

    /// The value row `row` of the batch from `base` holds, `None` if null.
    fn value(base: i64, row: u32) -> Option<i64> {
        (row % 7 != 3).then_some(base + i64::from(row))
    }

    /// A batch of a full batch's rows, an integer and a string column, of
    /// `value`s, keeping the rows `keep` says.
    fn batch(context: &mut Context, base: i64, keep: impl FnMut(u16) -> bool) -> RowBatch<'static> {
        let rows = BATCH_ROWS_MAX;
        let mut ints = context.values_buffer(rows as usize * 8).unwrap();
        let mut validity = context.small_buffer(BitSlice::<u8>::words_for(rows as usize)).unwrap();
        validity.as_mut_slice::<u8>().fill(0);
        let mut strings = Vec::new();
        for row in 0..rows {
            let value = value(base, row);
            ints.as_mut_slice::<i64>()[row as usize] = value.unwrap_or(0);
            BitSlice::of_mut(validity.as_mut_slice::<u8>()).set_if(row as usize, value.is_some());
            strings.push(value.map(|value| format!("v{value}")));
        }
        let ints = ColumnView::new(context, DataType::Int64, ints, Some(validity)).unwrap();
        let string = |row: usize| strings[row].as_deref().map(str::as_bytes);
        let strings = ColumnView::from_strings(context, rows as usize, string).unwrap();
        let mut batch = RowBatch::new();
        batch.reset(rows);
        batch.push_column(ints).ok().unwrap();
        batch.push_column(strings).ok().unwrap();
        batch.selection_mut().retain(keep);
        batch
    }

    /// The rows of `out`, as `value`s and their strings.
    fn read(out: &RowBatch) -> Vec<(Option<i64>, Vec<u8>)> {
        let (ints, strings) = (out.column(0), out.column(1));
        (0..out.row_count())
            .map(|r| {
                let int = (!ints.is_null(r)).then(|| ints.int64s()[r as usize]);
                (int, strings.string_values().get(r as usize).to_vec())
            })
            .collect()
    }

    /// A full batch of the two columns `batch` makes, with no nulls.
    fn batch_without_nulls(context: &mut Context, base: i64) -> RowBatch<'static> {
        let rows = BATCH_ROWS_MAX as usize;
        let mut ints = context.values_buffer(rows * 8).unwrap();
        let mut strings = Vec::new();
        for row in 0..rows {
            let value = base + i64::try_from(row).unwrap();
            ints.as_mut_slice::<i64>()[row] = value;
            strings.push(format!("v{value}"));
        }
        let ints = ColumnView::new(context, DataType::Int64, ints, None).unwrap();
        let strings =
            ColumnView::from_strings(context, rows, |row| Some(strings[row].as_bytes())).unwrap();
        let mut batch = RowBatch::new();
        batch.reset(BATCH_ROWS_MAX);
        batch.push_column(ints).ok().unwrap();
        batch.push_column(strings).ok().unwrap();
        batch
    }

    /// Which of a batch's rows to keep.
    type Keep = fn(u16) -> bool;

    #[test]
    fn hands_back_rows_shared_and_packed_in_any_order() {
        let env = Env::current();
        let mut context = Context::new(env, None);
        let mut store = RowStore::new(env).unwrap();
        // Shared, packed, shared after packed rows, and packed again.
        let keeps: [(i64, Keep); 4] = [
            (0, |_| true),
            (10_000, |row| row % 10 == 0),
            (20_000, |row| row < 1500),
            (30_000, |row| row % 30 == 1),
        ];
        let mut expected = Vec::new();
        let mut first = None;
        for (base, keep) in keeps {
            let batch = batch(&mut context, base, keep);
            first.get_or_insert_with(|| batch.column(0).clone());
            for row in (0..BATCH_ROWS_MAX).filter(|&row| keep(u16::try_from(row).unwrap())) {
                let value = value(base, row);
                let string = value.map_or_else(Vec::new, |value| format!("v{value}").into_bytes());
                expected.push((value, string));
            }
            store.append(&mut context, &batch).unwrap();
        }
        assert_eq!(store.len() as usize, expected.len());

        let mut out = RowBatch::new();
        let all: Vec<u32> = (0..store.len()).collect();
        let mut got = Vec::new();
        for chunk in all.chunks(BATCH_ROWS_MAX as usize) {
            assert_eq!(
                store
                    .gather(&mut context, &mut out, (chunk, Kept::Prefix), 0, &mut |_, _| failed!())
                    .unwrap() as usize,
                chunk.len()
            );
            if chunk[0] == 0 {
                // The first batch's rows, in order, are viewed where held.
                assert!(out.column(0).shares_values(first.as_ref().unwrap()));
            }
            got.extend(read(&out));
        }
        assert_eq!(got, expected);

        let backwards: Vec<u32> = (0..store.len()).rev().step_by(3).collect();
        let mut got = Vec::new();
        for chunk in backwards.chunks(BATCH_ROWS_MAX as usize) {
            store
                .gather(&mut context, &mut out, (chunk, Kept::Prefix), 0, &mut |_, _| failed!())
                .unwrap();
            got.extend(read(&out));
        }
        let expected: Vec<_> = backwards.iter().map(|&r| expected[r as usize].clone()).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn gathers_across_batches_with_and_without_nulls() {
        let env = Env::current();
        let mut context = Context::new(env, None);
        let mut store = RowStore::new(env).unwrap();
        let with = batch(&mut context, 0, |_| true);
        store.append(&mut context, &with).unwrap();
        let without = batch_without_nulls(&mut context, 10_000);
        store.append(&mut context, &without).unwrap();

        let last = BATCH_ROWS_MAX * 2 - 1;
        let rows = [last, 3, BATCH_ROWS_MAX, 0];
        let mut out = RowBatch::new();
        store
            .gather(&mut context, &mut out, (&rows, Kept::Prefix), 0, &mut |_, _| failed!())
            .unwrap();
        let int = |value: i64| (Some(value), format!("v{value}").into_bytes());
        let null = (None, Vec::new());
        let newest = 10_000 + i64::from(BATCH_ROWS_MAX) - 1;
        assert_eq!(read(&out), [int(newest), null, int(10_000), int(0)]);
    }

    #[test]
    fn gathers_only_the_rows_kept() {
        let env = Env::current();
        let mut context = Context::new(env, None);
        let mut store = RowStore::new(env).unwrap();
        // Half the rows, shared: row `i` held is the batch's `2 * i`.
        let one = batch(&mut context, 0, |row| row % 2 == 0);
        store.append(&mut context, &one).unwrap();
        // Rows 0 and 2 aren't kept, and their rows held aren't any.
        let rows = [u32::MAX, 4, u32::MAX, 2];
        let mut out = RowBatch::new();
        let kept = (&rows[..], Kept::Select(&[1, 3]));
        store.gather(&mut context, &mut out, kept, 0, &mut |_, _| failed!()).unwrap();
        // One batch held: its strings' bytes are shared.
        assert!(core::ptr::eq(
            out.column(1).string_values().get(1).as_ptr(),
            one.column(1).string_values().get(8).as_ptr()
        ));
        let int = |value: i64| (Some(value), format!("v{value}").into_bytes());
        // Only the kept rows: the others' are unspecified.
        let (ints, strings) = (out.column(0), out.column(1));
        let row = |r: u32| {
            let int = (!ints.is_null(r)).then(|| ints.int64s()[r as usize]);
            (int, strings.string_values().get(r as usize).to_vec())
        };
        assert_eq!([row(1), row(3)], [int(8), int(4)]);
    }

    #[test]
    #[cfg_attr(miri, ignore = "slow under Miri")]
    fn scatters_columns_into_an_order() {
        let env = Env::current();
        let mut context = Context::new(env, None);
        let mut store = RowStore::new(env).unwrap();
        let batches = 40;
        for b in 0..batches {
            let mut batch = batch_without_nulls(&mut context, i64::from(b) * 10_000);
            // Two more integer columns, so there are enough to copy.
            for k in 1..3 {
                let column = batch.column(0).clone();
                let mut values = context.values_buffer(BATCH_ROWS_MAX as usize * 8).unwrap();
                for (out, &value) in values.as_mut_slice::<i64>().iter_mut().zip(column.int64s()) {
                    *out = value * 10 + k;
                }
                let view = ColumnView::new(&mut context, DataType::Int64, values, None).unwrap();
                batch.push_column(view).ok().unwrap();
            }
            store.append(&mut context, &batch).unwrap();
        }
        let n = store.len();
        // Every row once, out of order.
        let order = |i: u32| u32::try_from(u64::from(i) * 48_271 % u64::from(n)).unwrap();
        let rows: Vec<u32> = (0..n).map(order).collect();
        let mut sorted = rows.clone();
        sorted.sort_unstable();
        assert!(sorted.iter().copied().eq(0..n));
        let mut words: Vec<u64> = rows.iter().map(|&row| u64::from(row) | 0xdead << 30).collect();
        let no_decode = (false, None::<fn(usize, u64) -> i64>);
        store
            .scatter(&mut context, &mut words, |word| (word & 0xff_ffff) as u32, no_decode)
            .unwrap();
        assert_eq!(store.scattered.count_ones(), 3);

        let mut out = RowBatch::new();
        let value =
            |row: u32| i64::from(row / BATCH_ROWS_MAX) * 10_000 + i64::from(row % BATCH_ROWS_MAX);
        for (chunk, at) in
            rows.chunks(BATCH_ROWS_MAX as usize).zip((0..).step_by(BATCH_ROWS_MAX as usize))
        {
            store
                .gather(&mut context, &mut out, (chunk, Kept::Prefix), at, &mut |_, _| failed!())
                .unwrap();
            for (r, &row) in chunk.iter().enumerate() {
                assert_eq!(out.column(0).int64s()[r], value(row));
                assert_eq!(out.column(2).int64s()[r], value(row) * 10 + 1);
                assert_eq!(out.column(3).int64s()[r], value(row) * 10 + 2);
                let string = out.column(1).string_values().get(r).to_vec();
                assert_eq!(string, format!("v{}", value(row)).into_bytes());
            }
        }
    }
}
