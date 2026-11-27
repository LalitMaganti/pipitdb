//! How a sort spills: it keeps the memory sorting what it holds takes
//! reserved, and when it's spilled, sorts in that memory and writes the
//! rows in order as a run. Once every row is in, it merges the runs.

#![allow(unsafe_code)]

use core::cmp::Ordering;

use crate::column::{ColumnView, DataType};
use crate::containers::{BitSlice, SmallVec};
use crate::context::Context;
use crate::env::{Env, Reservation};
use crate::error::Error;
use crate::ordered::ordered;
use crate::radix_sort::msd_stack_bytes;
use crate::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use crate::row_layout_sort;
use crate::sorted_rows::{SortKey, SortedRows};
use crate::spill::{LogId, Rows, SpillStore, read_batch, write_batch};

/// The most runs merged at once, each compared for every row.
const FAN_IN_MAX: usize = 16;

/// The runs a sort's rows are written to, and the memory to sort them in.
pub(crate) struct Runs {
    /// What sorting the rows held takes, while there's a store to spill to.
    reservation: Reservation,
    /// The store the runs are in, once any is.
    store: Option<&'static dyn SpillStore>,
    /// Each run. Once a level of merging is done, they're in the order
    /// their rows came, so ties keep that order; while it's under way, the
    /// first `left` are the level's still to merge, which come after the
    /// rest.
    written: SmallVec<Run>,
    /// How many of `written`'s first runs the level of merging under way
    /// has still to merge, if any.
    left: usize,
    /// How many runs are merged at once: fewer once a merge doesn't fit.
    fan_in: usize,
    /// The runs being merged into the output, once all are written.
    merge: Option<Merge>,
}

/// A run's log, of batches in sorted order.
#[derive(Clone, Copy)]
struct Run {
    /// Its log.
    log: LogId,
    /// How many batches it holds.
    batches: u32,
}

impl Drop for Runs {
    fn drop(&mut self) {
        let Some(store) = self.store else { return };
        for run in self.written.iter() {
            store.delete(run.log);
        }
    }
}

/// The most bytes `SortedRows::sort` allocates at once for `count` rows,
/// with keys `stride` bytes long, `string_keys` of them strings, each of
/// which it ranks first.
fn sorting_bytes(count: usize, stride: usize, string_keys: usize) -> usize {
    if stride == 0 {
        return 0;
    }
    let ranking = if string_keys > 0 { 8 * count + msd_stack_bytes(count) } else { 0 };
    ranking.max(row_layout_sort::sorting_bytes(count, stride))
}

impl Runs {
    pub(crate) fn new(env: &Env) -> Result<Runs, Error> {
        Ok(Runs {
            reservation: Reservation::new(env),
            store: None,
            written: SmallVec::new(env, 1 << 24)?,
            left: 0,
            fan_in: FAN_IN_MAX,
            merge: None,
        })
    }

    /// Reserves what sorting `rows` with `count` more takes, and room to
    /// write them as a run, if there's a store to spill to: while
    /// preparing a batch.
    pub(crate) fn prepare(
        &mut self,
        env: &Env,
        rows: &SortedRows,
        count: u32,
    ) -> Result<(), Error> {
        if env.spill().is_none() {
            return Ok(());
        }
        let count = rows.len() as usize + count as usize;
        let bytes = sorting_bytes(count, rows.stride(), rows.string_keys());
        self.reservation.grow_to(bytes)?;
        Ok(self.written.reserve(1)?)
    }

    /// Runs `f`, sorting `rows`, in what's reserved, if anything is; then
    /// reserves nothing.
    fn sorting<R>(&self, env: &Env, f: impl FnOnce() -> R) -> R {
        let sorted = if env.spill().is_some() { self.reservation.draw(f) } else { f() };
        self.reservation.release();
        sorted
    }

    /// Sorts `rows` and writes them as the next run, then holds none,
    /// allocating nothing but what's reserved: `Breaker::spill`.
    pub(crate) fn spill(&mut self, env: &Env, rows: &mut SortedRows) -> Result<bool, Error> {
        let Some(store) = env.spill() else { return Ok(false) };
        if rows.len() == 0 {
            return Ok(false);
        }
        self.store = Some(store);
        let written = &mut self.written;
        self.reservation.draw(|| write_run(env, store, written, rows))?;
        self.reservation.release();
        Ok(true)
    }

    /// Sorts the rows once every one is in: if it spilled, writes those
    /// held as the last run and starts merging the runs.
    pub(crate) fn finalize(
        &mut self,
        context: &mut Context,
        rows: &mut SortedRows,
        keys: &[SortKey],
    ) -> Result<(), Error> {
        let env = context.env();
        if self.written.is_empty() {
            // Packed rows are read quicker held, but holding them may not
            // fit once nothing can be spilled.
            if env.spill().is_none() {
                rows.flush(context)?;
            }
            self.sorting(env, || rows.sort(env))?;
            return rows.scatter(context);
        }
        self.spill(env, rows)?;
        self.reservation.release();
        self.merge_runs(context, keys)
    }

    /// Whether it's merging runs, which `serve` serves.
    pub(crate) fn merging(&self) -> bool {
        self.merge.is_some()
    }

    /// Merges the runs, as many at once as `fan_in`, into runs after the
    /// rest, a level at a time, until they're few enough to merge into the
    /// output. A merge that doesn't fit is tried again with half as many.
    fn merge_runs(&mut self, context: &mut Context, keys: &[SortKey]) -> Result<(), Error> {
        let Some(store) = self.store else { failed!() };
        loop {
            if self.left == 0 && self.written.len() <= self.fan_in {
                match Merge::new(context, store, &self.written) {
                    Ok(merge) => {
                        self.merge = Some(merge);
                        return Ok(());
                    }
                    Err(Error::OutOfMemory) if self.fan_in > 2 => self.fan_in /= 2,
                    Err(error) => return Err(error),
                }
                continue;
            }
            if self.left == 0 {
                self.left = self.written.len();
            }
            let group = self.left.min(self.fan_in);
            match self.merge_group(context, store, keys, group) {
                Ok(()) => self.left -= group,
                Err(Error::OutOfMemory) if self.fan_in > 2 => self.fan_in /= 2,
                Err(error) => return Err(error),
            }
        }
    }

    /// Merges the first `group` runs into a run after the rest, and deletes
    /// them; if it fails, deletes what it wrote and keeps them.
    fn merge_group(
        &mut self,
        context: &mut Context,
        store: &'static dyn SpillStore,
        keys: &[SortKey],
        group: usize,
    ) -> Result<(), Error> {
        let runs = at!(self.written, ..group);
        let Some(&first) = runs.first() else { failed!() };
        if group == 1 {
            self.written.retain(|run| run.log != first.log);
            return Ok(self.written.push(first)?);
        }
        let mut merge = Merge::new(context, store, runs)?;
        start_run(store, &mut self.written)?;
        let written = write_merged(context, store, keys, &mut merge, &mut self.written);
        drop(merge);
        if let Err(error) = written {
            if let Some(run) = self.written.pop() {
                store.delete(run.log);
            }
            return Err(error);
        }
        let mut i = 0;
        self.written.retain(|run| {
            i += 1;
            if i <= group {
                store.delete(run.log);
            }
            i > group
        });
        Ok(())
    }

    /// Fills `output` with the next rows in order, returning whether any
    /// are left after them.
    pub(crate) fn serve(
        &mut self,
        context: &mut Context,
        keys: &[SortKey],
        output: &mut RowBatch,
    ) -> Result<bool, Error> {
        let (Some(merge), Some(store)) = (&mut self.merge, self.store) else { failed!() };
        if !merge.fill(context, store, keys, output)? {
            return Ok(false);
        }
        Ok(merge.inputs.iter().any(Input::left))
    }
}

/// Starts a run, the last of `written`, held from the start so it's
/// deleted even if writing it fails.
fn start_run(store: &dyn SpillStore, written: &mut SmallVec<Run>) -> Result<(), Error> {
    let log = store.create()?;
    if let Err(error) = written.push(Run { log, batches: 0 }) {
        store.delete(log);
        return Err(error.into());
    }
    Ok(())
}

/// Sorts `rows`, writes them as a run after `written`, and holds none.
fn write_run(
    env: &Env,
    store: &dyn SpillStore,
    written: &mut SmallVec<Run>,
    rows: &mut SortedRows,
) -> Result<(), Error> {
    rows.sort(env)?;
    start_run(store, written)?;
    let Some(run) = written.last_mut() else { failed!() };
    let mut batch = rows.in_order();
    loop {
        write_batch(store, run.log, &batch)?;
        run.batches += 1;
        if !batch.next() {
            break;
        }
    }
    store.seal(run.log)?;
    rows.clear(env)
}

/// Writes the rows `merge` merges to the last of `written`, and seals it.
fn write_merged(
    context: &mut Context,
    store: &dyn SpillStore,
    keys: &[SortKey],
    merge: &mut Merge,
    written: &mut SmallVec<Run>,
) -> Result<(), Error> {
    let Some(run) = written.last_mut() else { failed!() };
    let mut batch = RowBatch::new();
    while merge.fill(context, store, keys, &mut batch)? {
        write_batch(store, run.log, &Batch(&batch))?;
        run.batches += 1;
    }
    batch.reset(0);
    store.seal(run.log)
}

/// The next rows of each run merged.
struct Merge {
    inputs: SmallVec<Input>,
}

/// A run being merged: its batch being read, and the next row of it.
struct Input {
    /// The run.
    run: Run,
    /// How many of its batches were read.
    read: u32,
    /// Where its next batch starts.
    at: u64,
    /// The batch last read.
    batch: RowBatch<'static>,
    /// Its next row to merge.
    row: u32,
}

impl Merge {
    /// Reads the first batch of each of `runs`.
    fn new(context: &mut Context, store: &dyn SpillStore, runs: &[Run]) -> Result<Merge, Error> {
        let mut inputs = SmallVec::fixed(context.env(), runs.len())?;
        for &run in runs {
            inputs.push(Input { run, read: 0, at: 0, batch: RowBatch::new(), row: 0 })?;
        }
        for input in inputs.iter_mut() {
            input.load(context, store)?;
        }
        Ok(Merge { inputs })
    }

    /// Fills `output` with the next rows of the runs in order, up to a
    /// batch's, or until a run's batch is used up, returning whether there
    /// were any. A tie goes to the earlier run.
    fn fill(
        &mut self,
        context: &mut Context,
        store: &dyn SpillStore,
        keys: &[SortKey],
        output: &mut RowBatch,
    ) -> Result<bool, Error> {
        output.reset(0);
        for input in self.inputs.iter_mut() {
            input.load(context, store)?;
        }
        let mut from = [0_u8; BATCH_ROWS_MAX as usize];
        let mut rows = [0_u16; BATCH_ROWS_MAX as usize];
        let mut count = 0;
        while count < BATCH_ROWS_MAX as usize {
            let mut best: Option<usize> = None;
            for (i, input) in self.inputs.iter().enumerate() {
                if input.row == input.batch.row_count() {
                    continue;
                }
                let before = |b: usize| compare(keys, input, at!(self.inputs, b)) == Ordering::Less;
                if best.is_none_or(before) {
                    best = Some(i);
                }
            }
            let Some(best) = best else { break };
            let input = at_mut!(self.inputs, best);
            #[expect(clippy::cast_possible_truncation, reason = "at most `FAN_IN_MAX`")]
            let best_u8 = best as u8;
            *at_mut!(from, count) = best_u8;
            #[expect(clippy::cast_possible_truncation, reason = "a row of a batch")]
            let row = input.row as u16;
            *at_mut!(rows, count) = row;
            count += 1;
            input.row += 1;
            // Its next batch would replace the rows chosen from this one.
            if input.row == input.batch.row_count() && input.read < input.run.batches {
                break;
            }
        }
        if count == 0 {
            return Ok(false);
        }
        let Some(first) = self.inputs.iter().find(|input| input.batch.column_count() > 0) else {
            failed!()
        };
        let width = first.batch.column_count();
        #[expect(clippy::cast_possible_truncation, reason = "a batch's rows")]
        output.reset(count as u32);
        for c in 0..width {
            let column = |i: usize| {
                let input = at!(self.inputs, usize::from(*at!(from, i)));
                (input.batch.column(c), u32::from(*at!(rows, i)))
            };
            let column = gather(context, count, column)?;
            let Ok(()) = output.push_column(column) else { failed!() };
        }
        Ok(true)
    }
}

impl Input {
    /// Reads its next batch, if its last one is used up and it has more.
    fn load(&mut self, context: &mut Context, store: &dyn SpillStore) -> Result<(), Error> {
        if self.row < self.batch.row_count() || self.read == self.run.batches {
            return Ok(());
        }
        // Freed before the next is read.
        self.batch.reset(0);
        let mut columns = [const { None }; BATCH_COLUMNS_MAX as usize];
        let mut c = 0;
        let (rows, next) = read_batch(context, store, self.run.log, self.at, |_, column| {
            *at_mut!(columns, c) = Some(column);
            c += 1;
            Ok(())
        })?;
        (self.at, self.read, self.row) = (next, self.read + 1, 0);
        self.batch.reset(rows);
        for column in at_mut!(columns, ..c) {
            let Some(column) = column.take() else { failed!() };
            let Ok(()) = self.batch.push_column(column) else { failed!() };
        }
        Ok(())
    }

    /// Whether it has rows not yet merged.
    fn left(&self) -> bool {
        self.row < self.batch.row_count() || self.read < self.run.batches
    }
}

/// How the next row of `ours` orders against the next of `theirs` by
/// `keys`. Nulls sort first ascending, last descending, and strings by
/// their bytes, as `SortedRows` sorts them.
fn compare(keys: &[SortKey], ours: &Input, theirs: &Input) -> Ordering {
    for key in keys {
        let (x, y) = (ours.batch.column(key.column), theirs.batch.column(key.column));
        let order = match (x.is_null(ours.row), y.is_null(theirs.row)) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => {
                let (i, j) = (ours.row as usize, theirs.row as usize);
                let (xs, ys) = (x.values(), y.values());
                match x.data_type() {
                    DataType::Int64 => at!(xs.words(), i).cmp(at!(ys.words(), j)),
                    DataType::Float64 => {
                        ordered(*at!(xs.words(), i)).cmp(&ordered(*at!(ys.words(), j)))
                    }
                    DataType::String => xs.strings().get(i).cmp(ys.strings().get(j)),
                }
            }
        };
        let order = if key.descending { order.reverse() } else { order };
        if order != Ordering::Equal {
            return order;
        }
    }
    Ordering::Equal
}

/// A column of `count` rows, each the row `row(i)` gives of a flat column,
/// all of one type.
fn gather<'c>(
    context: &mut Context,
    count: usize,
    row: impl Fn(usize) -> (&'c ColumnView, u32),
) -> Result<ColumnView, Error> {
    let data_type = row(0).0.data_type();
    let null = |i: usize| {
        let (column, row) = row(i);
        column.is_null(row)
    };
    let mut validity = None;
    if (0..count).any(null) {
        let mut bits = context.small_buffer(BitSlice::<u8>::words_for(count))?;
        bits.as_mut_slice::<u8>().fill(0);
        let bitmap = BitSlice::of_mut(bits.as_mut_slice::<u8>());
        for i in 0..count {
            bitmap.set_if(i, !null(i));
        }
        validity = Some(bits);
    }
    let mut words = context.values_buffer(count * 8)?;
    if !data_type.has_bytes() {
        for (i, word) in at_mut!(words.as_mut_slice::<i64>(), ..count).iter_mut().enumerate() {
            let (column, row) = row(i);
            *word =
                if column.is_null(row) { 0 } else { *at!(column.values().words(), row as usize) };
        }
        return Ok(ColumnView::new(context, data_type, words, validity)?);
    }
    let string = |i: usize| {
        let (column, row) = row(i);
        if column.is_null(row) { &[][..] } else { column.values().strings().get(row as usize) }
    };
    let total = (0..count).fold(4_usize, |total, i| {
        let len = string(i).len();
        total + if len == 0 { 0 } else { 4 + len }
    });
    // So no word's offset is past 32 bits.
    if u32::try_from(total).is_err() {
        return Err(Error::OutOfMemory);
    }
    let mut bytes = context.bytes_buffer(total)?;
    let out = bytes.as_mut_slice::<u8>();
    at_mut!(out, ..4).fill(0);
    let mut at = 4;
    for (i, word) in at_mut!(words.as_mut_slice::<u64>(), ..count).iter_mut().enumerate() {
        let string = string(i);
        if string.is_empty() {
            *word = 0;
            continue;
        }
        #[expect(clippy::cast_possible_truncation, reason = "below `total`")]
        at_mut!(out, at..at + 4).copy_from_slice(&(string.len() as u32).to_le_bytes());
        at_mut!(out, at + 4..at + 4 + string.len()).copy_from_slice(string);
        *word = at as u64;
        at += 4 + string.len();
    }
    // SAFETY: each word was written where its string is, just above.
    Ok(unsafe { ColumnView::strings_unchecked(context, words, bytes, validity)? })
}

/// A batch of flat columns, all its rows kept, as `write_batch` writes it.
struct Batch<'b>(&'b RowBatch<'b>);

impl Batch<'_> {
    #[expect(clippy::cast_possible_truncation, reason = "a column")]
    fn of(&self, column: usize) -> &ColumnView {
        self.0.column(column as u32)
    }
}

impl Rows for Batch<'_> {
    fn count(&self) -> u32 {
        self.0.row_count()
    }

    fn width(&self) -> usize {
        self.0.column_count() as usize
    }

    fn column(&self, column: usize) -> (DataType, bool) {
        let column = self.of(column);
        (column.data_type(), column.has_nulls())
    }

    fn is_null(&self, column: usize, row: u32) -> bool {
        self.of(column).is_null(row)
    }

    fn word(&self, column: usize, row: u32) -> u64 {
        at!(self.of(column).values().words(), row as usize).cast_unsigned()
    }

    fn string(&self, column: usize, row: u32) -> &[u8] {
        let column = self.of(column);
        if column.is_null(row) { &[] } else { column.values().strings().get(row as usize) }
    }
}
