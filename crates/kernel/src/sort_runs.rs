//! `Runs`: a sort's rows written out in sorted runs once it can't hold more,
//! then merged back in order.

use core::cmp::Ordering;

use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType};
use crate::containers::{BitSlice, SmallVec};
use crate::context::Context;
use crate::error::Error;
use crate::ordered::ordered;
use crate::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use crate::selection::Kept;
use crate::sorted_rows::{SortKey, SortedRows};
use crate::spill::{LogId, SpillStore, read_batch, write_batch};

/// The most runs merged at once, each compared for every row.
pub(crate) const FAN_IN_MAX: usize = 16;

pub(crate) struct Runs {
    /// The store the runs are in.
    store: &'static dyn SpillStore,
    /// Each run, in the order its rows came, so ties keep that order.
    written: SmallVec<Run>,
    /// How many bytes of rows a run holds before it's written.
    capacity: usize,
    /// How many bytes the sort can hold: a merge holds a batch a run.
    available: usize,
    /// The most bytes a batch written took.
    batch_bytes: usize,
    /// How many columns the rows have.
    width: usize,
    /// The runs being merged into the output, once all are written.
    merge: Option<Merge>,
}

/// A run's log, of batches in sorted order.
#[derive(Clone, Copy)]
struct Run {
    /// The log.
    log: LogId,
    /// How many batches it holds.
    batches: u32,
}

/// The next rows of each run merged.
struct Merge {
    inputs: SmallVec<Input>,
}

/// A run being merged: its batch being read, and the next row of it.
struct Input {
    run: Run,
    /// How many of its batches were read.
    read: u32,
    /// Where its next batch starts.
    at: u64,
    batch: RowBatch<'static>,
    row: u32,
}

impl Drop for Runs {
    fn drop(&mut self) {
        for run in self.written.iter() {
            self.store.delete(run.log);
        }
    }
}

impl Runs {
    /// Writes `rows`, which held `capacity` bytes when they could hold no
    /// more, as the first run, of a sort that can hold `available` bytes.
    pub(crate) fn new(
        context: &mut Context,
        store: &'static dyn SpillStore,
        rows: &mut SortedRows,
        capacity: usize,
        available: usize,
    ) -> Result<Runs, Error> {
        let written = SmallVec::new(context.env(), 1 << 24)?;
        let (batch_bytes, width, merge) = (0, 0, None);
        let mut runs = Runs { store, written, capacity, available, batch_bytes, width, merge };
        runs.write(context, rows)?;
        Ok(runs)
    }

    /// How many bytes of rows a run holds before it's written.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Sorts `rows` and writes them as the next run.
    pub(crate) fn write(
        &mut self,
        context: &mut Context,
        rows: &mut SortedRows,
    ) -> Result<(), Error> {
        rows.sort(context)?;
        rows.scatter(context)?;
        self.start_run()?;
        let mut batch = RowBatch::new();
        loop {
            batch.reset(0);
            let more = rows.serve(context, &mut batch)?;
            self.append(&batch)?;
            if !more {
                break;
            }
        }
        self.seal_run()
    }

    /// Starts a run, the last, held from the start so it's deleted even if
    /// writing it fails.
    fn start_run(&mut self) -> Result<(), Error> {
        let log = self.store.create()?;
        if let Err(error) = self.written.push(Run { log, batches: 0 }) {
            self.store.delete(log);
            return Err(error.into());
        }
        Ok(())
    }

    /// Ends the last run.
    fn seal_run(&self) -> Result<(), Error> {
        let Some(run) = self.written.last() else { failed!() };
        self.store.seal(run.log)
    }

    /// Appends `batch`'s kept rows to the last run.
    fn append(&mut self, batch: &RowBatch) -> Result<(), Error> {
        let count = batch.selection().len();
        if count == 0 {
            return Ok(());
        }
        let mut columns = [batch.column(0); BATCH_COLUMNS_MAX as usize];
        for (c, column) in columns.iter_mut().enumerate().take(batch.column_count() as usize) {
            #[expect(clippy::cast_possible_truncation, reason = "at most 64")]
            let c = c as u32;
            *column = batch.column(c);
        }
        self.width = batch.column_count() as usize;
        let columns = at!(columns, ..self.width);
        let select = match batch.selection().kept() {
            Kept::Select(select) => select,
            Kept::Prefix | Kept::None => &[],
        };
        let Some(run) = self.written.last_mut() else { failed!() };
        write_batch(self.store, run.log, columns, select, count)?;
        run.batches += 1;
        self.batch_bytes = self.batch_bytes.max(batch_bytes(batch));
        Ok(())
    }

    /// Merges the runs, in groups as many as can be held a batch each, with
    /// the batches read and made from them, until they're few enough to
    /// merge into the output.
    pub(crate) fn finish(&mut self, context: &mut Context, keys: &[SortKey]) -> Result<(), Error> {
        let fan_in = (self.available / (3 * self.batch_bytes.max(1))).clamp(2, FAN_IN_MAX);
        // A level at a time, each group of runs, from the first, is merged
        // into a run after the rest, which keeps their order.
        while self.written.len() > fan_in {
            let mut left = self.written.len();
            while left > 0 {
                let group = left.min(fan_in);
                let mut merge = Merge::new(context, at!(self.written, ..group))?;
                self.start_run()?;
                let mut batch = RowBatch::new();
                while merge.fill(context, self.store, self.width, keys, &mut batch)? {
                    self.append(&batch)?;
                }
                self.seal_run()?;
                for run in at!(self.written, ..group) {
                    self.store.delete(run.log);
                }
                let mut i = 0;
                self.written.retain(|_| {
                    i += 1;
                    i > group
                });
                left -= group;
            }
        }
        self.merge = Some(Merge::new(context, &self.written)?);
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
        let Some(merge) = &mut self.merge else { failed!() };
        if !merge.fill(context, self.store, self.width, keys, output)? {
            return Ok(false);
        }
        Ok(merge.inputs.iter().any(Input::left))
    }
}

impl Merge {
    fn new(context: &Context, runs: &[Run]) -> Result<Merge, Error> {
        let mut inputs = SmallVec::fixed(context.env(), runs.len())?;
        for &run in runs {
            inputs.push(Input { run, read: 0, at: 0, batch: RowBatch::new(), row: 0 })?;
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
        width: usize,
        keys: &[SortKey],
        output: &mut RowBatch,
    ) -> Result<bool, Error> {
        for input in self.inputs.iter_mut() {
            input.load(context, store, width)?;
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
        #[expect(clippy::cast_possible_truncation, reason = "a batch's rows")]
        output.reset(count as u32);
        for c in 0..width {
            #[expect(clippy::cast_possible_truncation, reason = "at most 64")]
            let c = c as u32;
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
    fn load(
        &mut self,
        context: &mut Context,
        store: &dyn SpillStore,
        width: usize,
    ) -> Result<(), Error> {
        if self.row < self.batch.row_count() || self.read == self.run.batches {
            return Ok(());
        }
        let mut columns = [const { None }; BATCH_COLUMNS_MAX as usize];
        let mut c = 0;
        self.at = read_batch(context, store, self.run.log, self.at, width, |column| {
            *at_mut!(columns, c) = Some(column);
            c += 1;
            Ok(())
        })?;
        self.read += 1;
        self.row = 0;
        let Some(first) = at!(columns, 0) else { failed!() };
        self.batch.reset(first.row_count());
        for column in at_mut!(columns, ..width) {
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

/// How the next row of `ours` orders against the next of `theirs` by `keys`. Nulls
/// sort first ascending, last descending, and strings by their bytes, as
/// `SortedRows` sorts them.
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
                    DataType::Int64 => at!(xs.int64s(), i).cmp(at!(ys.int64s(), j)),
                    DataType::Float64 => {
                        let bits = |v: f64| ordered(v.to_bits().cast_signed());
                        bits(*at!(xs.float64s(), i)).cmp(&bits(*at!(ys.float64s(), j)))
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
    let validity = if (0..count).any(|i| {
        let (column, row) = row(i);
        column.is_null(row)
    }) {
        let mut bits = context.small_buffer(BitSlice::<u8>::words_for(count))?;
        bits.as_mut_slice::<u8>().fill(0);
        let bitmap = BitSlice::of_mut(bits.as_mut_slice::<u8>());
        for i in 0..count {
            let (column, row) = row(i);
            bitmap.set_if(i, !column.is_null(row));
        }
        Some(bits)
    } else {
        None
    };
    if data_type != DataType::String {
        let mut words = context.values_buffer(count * 8)?;
        for (i, word) in words.as_mut_slice::<i64>().iter_mut().enumerate() {
            let (column, row) = row(i);
            let values = column.values();
            *word = match data_type {
                _ if column.is_null(row) => 0,
                DataType::Float64 => at!(values.float64s(), row as usize).to_bits().cast_signed(),
                _ => *at!(values.int64s(), row as usize),
            };
        }
        return Ok(ColumnView::new(context, data_type, words, validity)?);
    }
    let string = |i: usize| {
        let (column, row) = row(i);
        if column.is_null(row) { &[][..] } else { column.values().strings().get(row as usize) }
    };
    let total: usize = (0..count).map(|i| string(i).len()).sum();
    let mut views = context.values_buffer(count * 8)?;
    let mut bytes: Buffer = context.bytes_buffer(total)?;
    let (views_out, bytes_out) = (views.as_mut_slice::<u32>(), bytes.as_mut_slice::<u8>());
    let mut at = 0;
    for i in 0..count {
        let string = string(i);
        at_mut!(bytes_out, at..at + string.len()).copy_from_slice(string);
        let Ok(start) = u32::try_from(at) else { return Err(Error::OutOfMemory) };
        let Ok(len) = u32::try_from(string.len()) else { return Err(Error::OutOfMemory) };
        *at_mut!(views_out, 2 * i) = start;
        *at_mut!(views_out, 2 * i + 1) = len;
        at += string.len();
    }
    Ok(ColumnView::strings(context, views, bytes, validity)?)
}

/// Roughly how many bytes `batch`'s columns hold: a word a row each, and a
/// string column's bytes.
pub(crate) fn batch_bytes(batch: &RowBatch) -> usize {
    let rows = batch.row_count() as usize;
    (0..batch.column_count())
        .map(|c| {
            let column = batch.column(c);
            let bytes = if column.data_type() == DataType::String {
                column.values().strings().bytes().len()
            } else {
                0
            };
            rows * 8 + bytes
        })
        .sum()
}
