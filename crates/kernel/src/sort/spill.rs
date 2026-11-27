//! How a sort spills: it sorts the rows it holds in the room it made for
//! that as they came, writes them out in order as a run and forgets them,
//! and once every row is in, merges the runs back in order.

use core::cmp::Ordering;

use crate::column::{ColumnView, DataType};
use crate::containers::{BitSlice, SmallVec};
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;
use crate::ordered::ordered;
use crate::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use crate::sorted_rows::{SortKey, SortedRows};
use crate::spill::{LogId, Rows, SpillStore, read_batch, write_batch};

use super::SortState;

/// The most runs merged at once, each compared for every row.
const FAN_IN: usize = 16;

/// The runs a sort's rows are written in.
pub(super) struct Runs {
    /// Each run, in the order its rows came, so ties keep that order.
    written: SmallVec<Run>,
    /// The store they're in, once any is.
    store: Option<&'static dyn SpillStore>,
    /// How many columns the rows have.
    width: usize,
    /// Each column's type.
    types: [DataType; BATCH_COLUMNS_MAX as usize],
    /// The runs being merged into the output, once all are written.
    merge: Option<Merge>,
}

/// A run's log, of batches in sorted order.
#[derive(Clone, Copy)]
struct Run {
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

impl Runs {
    pub(super) fn new(env: &Env) -> Runs {
        let written = SmallVec::none(env);
        let types = [DataType::Int64; BATCH_COLUMNS_MAX as usize];
        Runs { written, store: None, width: 0, types, merge: None }
    }

    /// Whether any run is written.
    pub(super) fn any(&self) -> bool {
        !self.written.is_empty()
    }

    /// Makes room for one more run, if rows can spill.
    pub(super) fn reserve(&mut self, env: &Env) -> Result<(), Error> {
        if env.spill().is_none() {
            return Ok(());
        }
        if self.written.capacity() == 0 {
            self.written = SmallVec::new(env, 1 << 24)?;
        }
        Ok(self.written.reserve(1)?)
    }

    /// Starts a run, the last, held from the start so it's deleted even if
    /// writing it fails. There must be room for it.
    fn start(&mut self, store: &'static dyn SpillStore) -> Result<(), Error> {
        self.store = Some(store);
        let log = store.create()?;
        let Ok(()) = self.written.push(Run { log, batches: 0 }) else { failed!() };
        Ok(())
    }

    /// Appends `rows` to the last run.
    fn append(&mut self, rows: &dyn Rows) -> Result<(), Error> {
        let (Some(store), Some(run)) = (self.store, self.written.last_mut()) else { failed!() };
        write_batch(store, run.log, rows)?;
        run.batches += 1;
        Ok(())
    }

    /// Ends the last run.
    fn seal(&self) -> Result<(), Error> {
        let (Some(store), Some(run)) = (self.store, self.written.last()) else { failed!() };
        store.seal(run.log)
    }

    /// Merges the runs, as many at a time as fit, up to `FAN_IN`, until
    /// they're few enough to merge into the output. Each group, from the
    /// first, is merged into a run after the rest, a level at a time,
    /// which keeps their order. Fewer are merged at once after merging as
    /// many fails for want of memory.
    pub(super) fn finish(&mut self, context: &mut Context, keys: &[SortKey]) -> Result<(), Error> {
        let mut fan_in = FAN_IN;
        loop {
            while self.written.len() > fan_in {
                let mut left = self.written.len();
                while left > 0 {
                    let group = left.min(fan_in);
                    match self.merge_first(context, keys, group) {
                        Ok(()) => left -= group,
                        Err(Error::OutOfMemory) if fan_in > 2 => fan_in /= 2,
                        Err(error) => return Err(error),
                    }
                }
            }
            let (Some(store), width) = (self.store, self.width) else { failed!() };
            let mut merge = Merge::new(context, &self.written)?;
            match merge.load(context, store, width) {
                Ok(()) => {
                    self.merge = Some(merge);
                    return Ok(());
                }
                Err(Error::OutOfMemory) if self.written.len() > 2 => {
                    fan_in = (self.written.len() / 2).max(2);
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Merges the first `group` runs into one after the rest.
    fn merge_first(
        &mut self,
        context: &mut Context,
        keys: &[SortKey],
        group: usize,
    ) -> Result<(), Error> {
        let Some(store) = self.store else { failed!() };
        // A group of one is carried after the rest as it is.
        let carried = (group == 1).then(|| *at!(self.written, 0));
        if carried.is_none() {
            let mut merge = Merge::new(context, at!(self.written, ..group))?;
            self.written.reserve(1)?;
            self.start(store)?;
            let written = self.merge_into_last(context, keys, &mut merge);
            if let Err(error) = written {
                if let Some(run) = self.written.pop() {
                    store.delete(run.log);
                }
                return Err(error);
            }
        }
        for run in at!(self.written, ..group).iter().filter(|_| carried.is_none()) {
            store.delete(run.log);
        }
        let mut i = 0;
        self.written.retain(|_| {
            i += 1;
            i > group
        });
        if let Some(run) = carried {
            let Ok(()) = self.written.push(run) else { failed!() };
        }
        Ok(())
    }

    /// Writes what `merge` merges to the last run, and seals it.
    fn merge_into_last(
        &mut self,
        context: &mut Context,
        keys: &[SortKey],
        merge: &mut Merge,
    ) -> Result<(), Error> {
        let (Some(store), width, types) = (self.store, self.width, self.types) else { failed!() };
        loop {
            merge.load(context, store, width)?;
            let Some(picks) = merge.pick(keys, &types, width) else { break };
            self.append(&picks)?;
        }
        self.seal()
    }

    /// Fills `output` with the next rows in order, returning whether any
    /// are left after them.
    pub(super) fn serve(
        &mut self,
        context: &mut Context,
        keys: &[SortKey],
        output: &mut RowBatch,
    ) -> Result<bool, Error> {
        let (Some(store), Some(merge)) = (self.store, &mut self.merge) else { failed!() };
        merge.load(context, store, self.width)?;
        let Some(picks) = merge.pick(keys, &self.types, self.width) else {
            output.reset(0);
            return Ok(false);
        };
        fill(context, &picks, output)?;
        Ok(merge.inputs.iter().any(Input::left))
    }
}

/// Sorts the rows `state` holds that batches were applied to, writes them
/// as the last run and forgets them, allocating nothing, returning whether
/// it held any: `Breaker::spill`.
pub(super) fn write_run(env: &Env, state: &mut SortState) -> Result<bool, Error> {
    let count = state.applied;
    let Some(store) = env.spill() else { return Ok(false) };
    if count == 0 {
        return Ok(false);
    }
    let rows = &mut state.rows;
    rows.sort_held(env, count);
    let runs = &mut state.runs;
    runs.width = rows.width() as usize;
    for (c, data_type) in runs.types.iter_mut().enumerate().take(runs.width) {
        #[expect(clippy::cast_possible_truncation, reason = "a batch's columns")]
        let column = c as u32;
        *data_type = rows.column_type(column).0;
    }
    runs.start(store)?;
    for start in (0..count).step_by(BATCH_ROWS_MAX as usize) {
        let count = (count - start).min(BATCH_ROWS_MAX);
        runs.append(&Held { rows, start, count })?;
    }
    runs.seal()?;
    *rows = SortedRows::new(env)?;
    state.applied = 0;
    Ok(true)
}

/// Rows held, from the `start`th in order, as `write_batch` writes them.
struct Held<'r> {
    rows: &'r SortedRows,
    start: u32,
    count: u32,
}

impl Held<'_> {
    fn row(&self, row: u32) -> u32 {
        self.rows.row_in_order(self.start + row)
    }
}

impl Rows for Held<'_> {
    fn count(&self) -> u32 {
        self.count
    }

    fn width(&self) -> usize {
        self.rows.width() as usize
    }

    #[expect(clippy::cast_possible_truncation, reason = "a batch's columns")]
    fn column(&self, column: usize) -> (DataType, bool) {
        self.rows.column_type(column as u32)
    }

    #[expect(clippy::cast_possible_truncation, reason = "a batch's columns")]
    fn is_null(&self, column: usize, row: u32) -> bool {
        let (column, row) = (column as u32, self.row(row));
        match self.rows.column_type(column).0 {
            DataType::String => self.rows.string(column, row).is_none(),
            _ => self.rows.value(column, row).is_none(),
        }
    }

    #[expect(clippy::cast_possible_truncation, reason = "a batch's columns")]
    fn word(&self, column: usize, row: u32) -> u64 {
        self.rows.value(column as u32, self.row(row)).unwrap_or(0).cast_unsigned()
    }

    #[expect(clippy::cast_possible_truncation, reason = "a batch's columns")]
    fn string(&self, column: usize, row: u32) -> &[u8] {
        self.rows.string(column as u32, self.row(row)).unwrap_or_default()
    }
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

impl Input {
    /// Whether it has rows not yet merged.
    fn left(&self) -> bool {
        self.row < self.batch.row_count() || self.read < self.run.batches
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

    /// Reads the next batch of each input whose batch is used up, if it
    /// has more.
    fn load(
        &mut self,
        context: &mut Context,
        store: &dyn SpillStore,
        width: usize,
    ) -> Result<(), Error> {
        for input in self.inputs.iter_mut() {
            if input.row < input.batch.row_count() || input.read == input.run.batches {
                continue;
            }
            let mut columns = [const { None }; BATCH_COLUMNS_MAX as usize];
            let mut c = 0;
            let (rows, at) = read_batch(context, store, input.run.log, input.at, |_, column| {
                *at_mut!(columns, c) = Some(column);
                c += 1;
                Ok(())
            })?;
            check!(c == width);
            (input.at, input.read, input.row) = (at, input.read + 1, 0);
            input.batch.reset(rows);
            for column in at_mut!(columns, ..width) {
                let Some(column) = column.take() else { failed!() };
                let Ok(()) = input.batch.push_column(column) else { failed!() };
            }
        }
        Ok(())
    }

    /// The next rows of the inputs in order, up to a batch's, or until an
    /// input's batch is used up, if there are any. A tie goes to the
    /// earlier input.
    fn pick<'m>(
        &'m mut self,
        keys: &[SortKey],
        types: &'m [DataType],
        width: usize,
    ) -> Option<Picks<'m>> {
        let mut picks = Picks {
            inputs: &[],
            types: at!(types, ..width),
            from: [0; BATCH_ROWS_MAX as usize],
            rows: [0; BATCH_ROWS_MAX as usize],
            count: 0,
        };
        while picks.count < BATCH_ROWS_MAX {
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
            let at = picks.count as usize;
            #[expect(clippy::cast_possible_truncation, reason = "at most `FAN_IN`")]
            let from = best as u8;
            *at_mut!(picks.from, at) = from;
            #[expect(clippy::cast_possible_truncation, reason = "a row of a batch")]
            let row = input.row as u16;
            *at_mut!(picks.rows, at) = row;
            picks.count += 1;
            input.row += 1;
            // Its next batch would replace the rows picked from this one.
            if input.row == input.batch.row_count() && input.read < input.run.batches {
                break;
            }
        }
        picks.inputs = &self.inputs;
        (picks.count > 0).then_some(picks)
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

/// Rows picked from the inputs of a merge, in order.
struct Picks<'m> {
    inputs: &'m [Input],
    /// Each column's type.
    types: &'m [DataType],
    /// Which input each row is from.
    from: [u8; BATCH_ROWS_MAX as usize],
    /// Which row of its input's batch each is.
    rows: [u16; BATCH_ROWS_MAX as usize],
    count: u32,
}

impl Picks<'_> {
    /// Column `column` of row `row`'s batch, and which row of it it is.
    fn at(&self, column: usize, row: u32) -> (&ColumnView, u32) {
        let input = at!(self.inputs, usize::from(*at!(self.from, row as usize)));
        #[expect(clippy::cast_possible_truncation, reason = "a batch's columns")]
        let column = input.batch.column(column as u32);
        (column, u32::from(*at!(self.rows, row as usize)))
    }
}

impl Rows for Picks<'_> {
    fn count(&self) -> u32 {
        self.count
    }

    fn width(&self) -> usize {
        self.types.len()
    }

    fn column(&self, column: usize) -> (DataType, bool) {
        (*at!(self.types, column), true)
    }

    fn is_null(&self, column: usize, row: u32) -> bool {
        let (column, row) = self.at(column, row);
        column.is_null(row)
    }

    fn word(&self, column: usize, row: u32) -> u64 {
        let (column, row) = self.at(column, row);
        at!(column.values().words(), row as usize).cast_unsigned()
    }

    fn string(&self, column: usize, row: u32) -> &[u8] {
        let (column, row) = self.at(column, row);
        if column.is_null(row) { &[] } else { column.values().strings().get(row as usize) }
    }
}

/// Fills `output` with `rows`, in buffers from `context`.
fn fill(context: &mut Context, rows: &dyn Rows, output: &mut RowBatch) -> Result<(), Error> {
    let count = rows.count();
    output.reset(count);
    for c in 0..rows.width() {
        let (data_type, _) = rows.column(c);
        let column = if data_type.has_bytes() {
            let string = |i: usize| {
                #[expect(clippy::cast_possible_truncation, reason = "a batch's row")]
                let row = i as u32;
                (!rows.is_null(c, row)).then(|| rows.string(c, row))
            };
            ColumnView::from_strings(context, count as usize, string)?
        } else {
            let mut values = context.values_buffer(count as usize * 8)?;
            let mut validity = context.small_buffer(BitSlice::<u8>::words_for(count as usize))?;
            let (out, bits) = (values.as_mut_slice::<u64>(), validity.as_mut_slice::<u8>());
            bits.fill(0);
            let mut nulls = false;
            for row in 0..count {
                let null = rows.is_null(c, row);
                *at_mut!(out, row as usize) = if null { 0 } else { rows.word(c, row) };
                BitSlice::of_mut(&mut *bits).set_if(row as usize, !null);
                nulls |= null;
            }
            ColumnView::new(context, data_type, values, nulls.then_some(validity))?
        };
        let Ok(()) = output.push_column(column) else { failed!() };
    }
    Ok(())
}
