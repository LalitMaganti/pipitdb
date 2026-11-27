//! `Runs`: a sort's rows, sorted a chunk at a time into runs as they come,
//! each held as batches in order in memory of its own, apart from batches'
//! column buffers, so spilling writes it as it is, freeing memory any
//! allocation can take. Once every row is in, runs held are sorted again as
//! one, and runs written are merged in order.

#![allow(unsafe_code)]

use core::cmp::Ordering;

use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType};
use crate::containers::{BitSlice, SmallVec};
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;
use crate::ordered::ordered;
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::sorted_rows::{SortKey, SortedRows};
use crate::spill::{LogId, Rows, SpillStore, read_batch, write_batch};

/// The most rows a sort holds before it sorts them into a run: few enough
/// that sorting them needs little memory.
pub(crate) const CHUNK_ROWS: u32 = BATCH_ROWS_MAX;

/// The most runs written that are merged at once, each a batch read.
const FAN_IN: usize = 16;

/// No run, in a merge's tree.
const NONE: u32 = u32::MAX;

pub(crate) struct Runs {
    /// Each run, oldest first, so ties keep the order rows came in.
    list: SmallVec<Run>,
    /// How many columns the rows have.
    width: usize,
    /// The store runs are written to, once one is.
    store: Option<&'static dyn SpillStore>,
    /// How many runs written are merged at once: fewer once a merge of
    /// more didn't fit.
    fan_in: usize,
    /// Whether the rows last given to `hold` are sorted, but not yet held.
    sorted: bool,
    /// The runs being merged into the output, once every row is in.
    merge: Option<Merge>,
}

enum Source {
    /// Held: each batch's columns, and how many rows each batch has.
    Held { columns: SmallVec<Option<Held>>, counts: SmallVec<u32> },
    /// Written to a sealed log, as batches.
    Written { log: LogId, batches: u32 },
    /// Merged into another run, or into the output.
    Merged,
}

struct Run {
    /// Where its rows are.
    source: Source,
    /// How many of its batches are loaded.
    next: u32,
    /// Where its next batch starts, if written.
    at: u64,
}

/// A column of a batch of a run held, flat.
struct Held {
    /// What type its values are.
    data_type: DataType,
    /// A word a row: a value, or a string's offset into `bytes`, 0 if empty.
    values: Buffer,
    /// A bit a row, set if it isn't null, if any is.
    validity: Option<Buffer>,
    /// Strings, each a 4-byte little-endian length and then its bytes,
    /// after an empty one.
    bytes: Option<Buffer>,
}

/// Merges runs: the batch loaded of each, and a tree of which run's next
/// row comes first.
struct Merge {
    /// Each run's batch loaded.
    loaded: SmallVec<Loaded>,
    /// The runs' leaves, a power of two, at least one.
    leaves: usize,
    /// Each node's run, `NONE` if it has no rows: node `j`'s children are
    /// `2j` and `2j + 1`, and run `i`'s leaf is `leaves + i`.
    tree: SmallVec<u32>,
}

/// A run's batch being merged.
struct Loaded {
    /// Its columns, flat.
    columns: SmallVec<ColumnView>,
    /// How many rows it has.
    rows: u32,
    /// The next row to merge.
    row: u32,
}

impl Run {
    /// Whether it has batches not yet loaded.
    fn more(&self) -> bool {
        match &self.source {
            Source::Held { counts, .. } => (self.next as usize) < counts.len(),
            Source::Written { batches, .. } => self.next < *batches,
            Source::Merged => false,
        }
    }

    /// Loads its next batch from `store` into `loaded` once the one there
    /// is merged, if it has more, deleting its log once it has none. Fails,
    /// if it does, with none loaded, to load again.
    fn load(
        &mut self,
        context: &mut Context,
        store: &dyn SpillStore,
        loaded: &mut Loaded,
    ) -> Result<(), Error> {
        if loaded.row < loaded.rows {
            return Ok(());
        }
        while loaded.columns.pop().is_some() {}
        (loaded.rows, loaded.row) = (0, 0);
        if !self.more() {
            if let Source::Written { log, .. } = self.source {
                store.delete(log);
            }
            self.source = Source::Merged;
            return Ok(());
        }
        match &mut self.source {
            Source::Written { log, .. } => {
                let columns = &mut loaded.columns;
                let (rows, at) = read_batch(context, store, *log, self.at, |_, column| {
                    Ok(columns.push(column)?)
                })?;
                (loaded.rows, self.at) = (rows, at);
            }
            // Only runs written are merged.
            Source::Held { .. } | Source::Merged => failed!(),
        }
        self.next += 1;
        Ok(())
    }
}

impl Held {
    /// `column`'s rows, flat, copied into buffers of no batch's, each with
    /// room for a batch's rows, as a sort holding a view of them shares it.
    fn of(context: &mut Context, column: &ColumnView) -> Result<Held, Error> {
        let (data_type, count) = (column.data_type(), column.row_count() as usize);
        let room = BATCH_ROWS_MAX as usize;
        let mut validity = None;
        if column.has_nulls() {
            let mut bits = context.bytes_buffer(BitSlice::<u8>::words_for(room))?;
            bits.as_mut_slice::<u8>().fill(0);
            #[expect(clippy::cast_possible_truncation, reason = "a row of a batch")]
            for row in 0..count {
                BitSlice::of_mut(bits.as_mut_slice::<u8>())
                    .set_if(row, !column.is_null(row as u32));
            }
            validity = Some(bits);
        }
        let mut values = context.bytes_buffer(room * 8)?;
        let (rows, rest) = values.as_mut_slice::<i64>().split_at_mut(count);
        rest.fill(0);
        if !data_type.has_bytes() {
            rows.copy_from_slice(at!(column.values().words(), ..count));
            return Ok(Held { data_type, values, validity, bytes: None });
        }
        #[expect(clippy::cast_possible_truncation, reason = "a row of a batch")]
        let string = |row: usize| {
            if column.is_null(row as u32) { &[][..] } else { column.values().strings().get(row) }
        };
        let total = (0..count).fold(4_u64, |total, row| total + strings_bytes(string(row)));
        // So no word's offset is past 32 bits.
        let Ok(total) = u32::try_from(total) else { return Err(Error::OutOfMemory) };
        let mut bytes = context.bytes_buffer(total as usize)?;
        let out = bytes.as_mut_slice::<u8>();
        at_mut!(out, ..4).fill(0);
        let mut at = 4;
        for (row, word) in at_mut!(values.as_mut_slice::<u64>(), ..count).iter_mut().enumerate() {
            let string = string(row);
            *word = if string.is_empty() { 0 } else { at as u64 };
            if !string.is_empty() {
                #[expect(clippy::cast_possible_truncation, reason = "below `total`")]
                at_mut!(out, at..at + 4).copy_from_slice(&(string.len() as u32).to_le_bytes());
                at_mut!(out, at + 4..at + 4 + string.len()).copy_from_slice(string);
                at += 4 + string.len();
            }
        }
        Ok(Held { data_type, values, validity, bytes: Some(bytes) })
    }

    /// It as a column of `count` rows, sharing its buffers.
    fn view(&self, context: &mut Context, count: u32) -> Result<ColumnView, Error> {
        let (values, validity) = (self.values.clone(), self.validity.clone());
        let column = match &self.bytes {
            // SAFETY: `of` wrote each word where its string is.
            Some(bytes) => unsafe {
                ColumnView::strings_unchecked(context, values, bytes.clone(), validity)?
            },
            None => ColumnView::new(context, self.data_type, values, validity)?,
        };
        Ok(column.slice(0, count))
    }

    fn is_null(&self, row: u32) -> bool {
        let Some(bits) = &self.validity else { return false };
        !BitSlice::of(bits.as_slice::<u8>()).get(row as usize)
    }

    fn word(&self, row: u32) -> u64 {
        *at!(self.values.as_slice::<u64>(), row as usize)
    }

    fn string(&self, row: u32) -> &[u8] {
        let (Some(bytes), Ok(at)) = (&self.bytes, usize::try_from(self.word(row))) else {
            failed!()
        };
        if at == 0 {
            return &[];
        }
        let bytes = bytes.as_slice::<u8>();
        let Some(&len) = at!(bytes, at..at + 4).first_chunk::<4>() else { failed!() };
        at!(bytes, at + 4..at + 4 + u32::from_le_bytes(len) as usize)
    }
}

/// The bytes `string` takes among a held column's: none if it's empty.
fn strings_bytes(string: &[u8]) -> u64 {
    if string.is_empty() { 0 } else { 4 + string.len() as u64 }
}

/// A batch of a run held, as `write_batch` writes it.
struct Batch<'h> {
    count: u32,
    columns: &'h [Option<Held>],
}

impl Batch<'_> {
    fn of(&self, column: usize) -> &Held {
        let Some(held) = at!(self.columns, column) else { failed!() };
        held
    }
}

impl Rows for Batch<'_> {
    fn count(&self) -> u32 {
        self.count
    }

    fn width(&self) -> usize {
        self.columns.len()
    }

    fn column(&self, column: usize) -> (DataType, bool) {
        let held = self.of(column);
        (held.data_type, held.validity.is_some())
    }

    fn is_null(&self, column: usize, row: u32) -> bool {
        self.of(column).is_null(row)
    }

    fn word(&self, column: usize, row: u32) -> u64 {
        self.of(column).word(row)
    }

    fn string(&self, column: usize, row: u32) -> &[u8] {
        self.of(column).string(row)
    }
}

/// A merged batch's columns, flat, as `write_batch` writes them.
struct Merged<'b>(&'b RowBatch<'b>);

impl Merged<'_> {
    fn of(&self, column: usize) -> &ColumnView {
        #[expect(clippy::cast_possible_truncation, reason = "a column")]
        self.0.column(column as u32)
    }
}

impl Rows for Merged<'_> {
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

impl Drop for Runs {
    fn drop(&mut self) {
        let Some(store) = self.store else { return };
        for run in self.list.iter() {
            if let Source::Written { log, .. } = run.source {
                store.delete(log);
            }
        }
    }
}

impl Runs {
    pub(crate) fn new(env: &Env) -> Result<Runs, Error> {
        let list = SmallVec::new(env, 1 << 24)?;
        Ok(Runs { list, width: 0, store: None, fan_in: FAN_IN, sorted: false, merge: None })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Whether it's merging runs into the output, rather than having
    /// sorted every row again.
    pub(crate) fn is_merging(&self) -> bool {
        self.merge.is_some()
    }

    /// Sorts `rows` and holds them as the newest run, leaving `rows` empty.
    /// Fails, if it does, leaving them as they were, but maybe sorted.
    pub(crate) fn hold(
        &mut self,
        context: &mut Context,
        rows: &mut SortedRows,
    ) -> Result<(), Error> {
        let env = context.env();
        self.list.reserve(1)?;
        if !self.sorted {
            rows.sort(context)?;
            self.sorted = true;
        }
        self.width = rows.width();
        let batches = rows.len().div_ceil(BATCH_ROWS_MAX) as usize;
        let mut columns = SmallVec::fixed(env, batches * self.width)?;
        let mut counts = SmallVec::fixed(env, batches)?;
        let mut at = 0;
        while at < rows.len() {
            let mut batch = RowBatch::new();
            let count = rows.gather(context, &mut batch, at)?;
            for c in 0..batch.column_count() {
                columns.push(Some(Held::of(context, batch.column(c))?))?;
            }
            counts.push(count)?;
            at += count;
        }
        *rows = SortedRows::new(env)?;
        self.sorted = false;
        let run = Run { source: Source::Held { columns, counts }, next: 0, at: 0 };
        let Ok(()) = self.list.push(run) else { failed!() };
        Ok(())
    }

    /// Writes each run held to `env`'s store, from its batches not yet
    /// loaded, and lets go of them, allocating nothing. Returns whether any
    /// was.
    pub(crate) fn spill(&mut self, env: &Env) -> Result<bool, Error> {
        let Some(store) = env.spill() else { return Ok(false) };
        let mut spilled = false;
        for run in self.list.iter_mut() {
            let Source::Held { columns, counts } = &run.source else { continue };
            if !run.more() {
                continue;
            }
            self.store = Some(store);
            let log = store.create()?;
            let width = columns.len() / counts.len();
            let write = || {
                for b in run.next as usize..counts.len() {
                    let columns = at!(columns, b * width..(b + 1) * width);
                    write_batch(store, log, &Batch { count: *at!(counts, b), columns })?;
                }
                store.seal(log)
            };
            write().inspect_err(|_| store.delete(log))?;
            #[expect(clippy::cast_possible_truncation, reason = "as `next` is")]
            let batches = counts.len() as u32 - run.next;
            *run = Run { source: Source::Written { log, batches }, next: 0, at: 0 };
            spilled = true;
        }
        Ok(spilled)
    }

    /// Holds `rows`, the last, as a run. If none was written, sorts every
    /// row again into `rows`, as cheaply as a sort that never spills. If any
    /// was, or that doesn't fit, makes ready to merge every run: each held
    /// is written too, and those written are merged until few enough are
    /// left to merge at once.
    pub(crate) fn finish(
        &mut self,
        context: &mut Context,
        keys: &[SortKey],
        rows: &mut SortedRows,
    ) -> Result<(), Error> {
        while rows.len() > 0 {
            match self.hold(context, rows) {
                Err(Error::OutOfMemory) if self.spill(context.env())? => {}
                held => held?,
            }
        }
        if self.store.is_none() {
            match self.sort_held(context, keys) {
                Ok(all) => {
                    *rows = all;
                    while self.list.pop().is_some() {}
                    return Ok(());
                }
                Err(Error::OutOfMemory) => {}
                Err(error) => return Err(error),
            }
        }
        self.spill(context.env())?;
        let Some(store) = self.store else { failed!() };
        loop {
            while self.list.len() > self.fan_in {
                self.merge_level(context, keys)?;
            }
            let mut merge = Merge::new(context.env(), self.list.len(), self.width)?;
            match merge.load(context, store, &mut self.list) {
                // Every run is written, so can be read again.
                Err(Error::OutOfMemory) if self.fan_in > 2 => {
                    for run in self.list.iter_mut() {
                        (run.next, run.at) = (0, 0);
                    }
                    self.fan_in /= 2;
                }
                loaded => {
                    loaded?;
                    self.merge = Some(merge);
                    return Ok(());
                }
            }
        }
    }

    /// Every row of the runs, all held, sorted as one, sharing their buffers.
    fn sort_held(&self, context: &mut Context, keys: &[SortKey]) -> Result<SortedRows, Error> {
        let mut all = SortedRows::new(context.env())?;
        let mut batch = RowBatch::new();
        for run in self.list.iter() {
            let Source::Held { columns, counts } = &run.source else { failed!() };
            let width = columns.len() / counts.len();
            for (b, &count) in counts.iter().enumerate() {
                batch.reset(count);
                for held in at!(columns, b * width..(b + 1) * width) {
                    let Some(held) = held else { failed!() };
                    let Ok(()) = batch.push_column(held.view(context, count)?) else { failed!() };
                }
                all.append(context, &batch, keys)?;
            }
        }
        all.sort(context)?;
        all.scatter(context)?;
        Ok(all)
    }

    /// Merges each `fan_in` runs, from the first, into one, which keeps them
    /// in order, merging fewer at once from then on if they don't fit.
    fn merge_level(&mut self, context: &mut Context, keys: &[SortKey]) -> Result<(), Error> {
        let mut first = 0;
        while first < self.list.len() {
            let end = self.list.len().min(first + self.fan_in);
            match self.merge_group(context, keys, first..end) {
                Err(Error::OutOfMemory) if self.fan_in > 2 => self.fan_in /= 2,
                merged => {
                    merged?;
                    first += 1;
                }
            }
        }
        Ok(())
    }

    /// Merges runs `group`, all written, into one written in their place.
    /// Fails, if it does, leaving them as they were.
    fn merge_group(
        &mut self,
        context: &mut Context,
        keys: &[SortKey],
        group: core::ops::Range<usize>,
    ) -> Result<(), Error> {
        if group.len() < 2 {
            return Ok(());
        }
        let Some(store) = self.store else { failed!() };
        let log = store.create()?;
        let first = group.start;
        let width = self.width;
        let runs = at_mut!(self.list, group);
        let mut merge = || {
            let mut merge = Merge::new(context.env(), runs.len(), width)?;
            let (mut batch, mut batches) = (RowBatch::new(), 0);
            while merge.fill(context, store, keys, runs, &mut batch)? {
                write_batch(store, log, &Merged(&batch))?;
                batches += 1;
            }
            store.seal(log)?;
            Ok::<_, Error>(batches)
        };
        let batches = merge().inspect_err(|_| {
            store.delete(log);
            for run in runs.iter_mut() {
                (run.next, run.at) = (0, 0);
            }
        })?;
        for run in runs.iter_mut() {
            if let Source::Written { log, .. } = run.source {
                store.delete(log);
            }
            run.source = Source::Merged;
        }
        *at_mut!(self.list, first) =
            Run { source: Source::Written { log, batches }, next: 0, at: 0 };
        self.list.retain(|run| !matches!(run.source, Source::Merged));
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
        let filled = merge.fill(context, store, keys, &mut self.list, output)?;
        let mut left = self.list.iter().zip(merge.loaded.iter());
        Ok(filled && left.any(|(run, loaded)| run.more() || loaded.row < loaded.rows))
    }
}

impl Merge {
    fn new(env: &Env, runs: usize, width: usize) -> Result<Merge, Error> {
        let mut loaded = SmallVec::fixed(env, runs)?;
        for _ in 0..runs {
            loaded.push(Loaded { columns: SmallVec::fixed(env, width)?, rows: 0, row: 0 })?;
        }
        let leaves = runs.max(1).next_power_of_two();
        let tree = SmallVec::fixed_from(env, (0..2 * leaves).map(|_| NONE))?;
        Ok(Merge { loaded, leaves, tree })
    }

    /// Loads each of `runs`' next batch once the one loaded is merged.
    fn load(
        &mut self,
        context: &mut Context,
        store: &dyn SpillStore,
        runs: &mut [Run],
    ) -> Result<(), Error> {
        for (run, loaded) in runs.iter_mut().zip(self.loaded.iter_mut()) {
            run.load(context, store, loaded)?;
        }
        Ok(())
    }

    /// Fills `output` with the next rows of `runs` in order, up to a batch's,
    /// or until a run's batch loaded is used up, returning whether there
    /// were any. A tie goes to the older run. Fails, if it does, with no
    /// row taken.
    fn fill(
        &mut self,
        context: &mut Context,
        store: &dyn SpillStore,
        keys: &[SortKey],
        runs: &mut [Run],
        output: &mut RowBatch,
    ) -> Result<bool, Error> {
        self.load(context, store, runs)?;
        let leaves = self.leaves;
        for (i, loaded) in self.loaded.iter().enumerate() {
            #[expect(clippy::cast_possible_truncation, reason = "below 2^24")]
            let leaf = if loaded.row < loaded.rows { i as u32 } else { NONE };
            *at_mut!(self.tree, leaves + i) = leaf;
        }
        for j in (1..leaves).rev() {
            *at_mut!(self.tree, j) = winner(keys, &self.loaded, &self.tree, j);
        }
        let mut from = [0_u32; BATCH_ROWS_MAX as usize];
        let mut rows = [0_u16; BATCH_ROWS_MAX as usize];
        let mut count = 0;
        while count < BATCH_ROWS_MAX as usize {
            let best = *at!(self.tree, 1);
            if best == NONE {
                break;
            }
            let loaded = at_mut!(self.loaded, best as usize);
            *at_mut!(from, count) = best;
            #[expect(clippy::cast_possible_truncation, reason = "a row of a batch")]
            let row = loaded.row as u16;
            *at_mut!(rows, count) = row;
            count += 1;
            loaded.row += 1;
            if loaded.row == loaded.rows {
                // Its next batch would be loaded over the rows taken from it.
                if at!(runs, best as usize).more() {
                    break;
                }
                *at_mut!(self.tree, leaves + best as usize) = NONE;
            }
            let leaf = leaves + best as usize;
            let mut j = leaf / 2;
            while j > 0 {
                *at_mut!(self.tree, j) = winner(keys, &self.loaded, &self.tree, j);
                j /= 2;
            }
        }
        if count == 0 {
            return Ok(false);
        }
        let (from, rows) = (at!(from, ..count), at!(rows, ..count));
        let filled = output_rows(context, &self.loaded, (from, rows), output);
        if filled.is_err() {
            // Each run's first row taken is its next again.
            for (&run, &row) in from.iter().zip(rows).rev() {
                at_mut!(self.loaded, run as usize).row = u32::from(row);
            }
        }
        filled.map(|()| true)
    }
}

/// Fills `output` with the rows of `loaded`'s batches `from` and `rows`
/// say: each row's run, and its row of that run's batch.
fn output_rows(
    context: &mut Context,
    loaded: &[Loaded],
    (from, rows): (&[u32], &[u16]),
    output: &mut RowBatch,
) -> Result<(), Error> {
    #[expect(clippy::cast_possible_truncation, reason = "a batch's rows")]
    let count = from.len() as u32;
    let first = at!(loaded, *at!(from, 0) as usize);
    output.reset(count);
    // Rows of one run only follow each other, and are viewed where loaded.
    if from.iter().all(|&run| run == *at!(from, 0)) {
        let start = u32::from(*at!(rows, 0));
        for column in first.columns.iter() {
            let Ok(()) = output.push_column(column.slice(start, count)) else { failed!() };
        }
        return Ok(());
    }
    for c in 0..first.columns.len() {
        let column = |i: usize| {
            let loaded = at!(loaded, *at!(from, i) as usize);
            (at!(loaded.columns, c), u32::from(*at!(rows, i)))
        };
        let column = gather(context, from.len(), column)?;
        let Ok(()) = output.push_column(column) else { failed!() };
    }
    Ok(())
}

/// Which of node `j`'s children's runs has the row that comes first: the
/// older, the left, on a tie.
fn winner(keys: &[SortKey], loaded: &[Loaded], tree: &[u32], j: usize) -> u32 {
    let (left, right) = (*at!(tree, 2 * j), *at!(tree, 2 * j + 1));
    if left == NONE || right == NONE {
        return left.min(right);
    }
    let (ours, theirs) = (at!(loaded, left as usize), at!(loaded, right as usize));
    if compare(keys, ours, theirs) == Ordering::Greater { right } else { left }
}

/// How the next row of `ours` orders against the next of `theirs` by
/// `keys`. Nulls sort first ascending, last descending, and strings by
/// their bytes, as `SortedRows` sorts them.
fn compare(keys: &[SortKey], ours: &Loaded, theirs: &Loaded) -> Ordering {
    let (i, j) = (ours.row, theirs.row);
    for key in keys {
        let column = key.column as usize;
        let (x, y) = (at!(ours.columns, column), at!(theirs.columns, column));
        let order = match (x.is_null(i), y.is_null(j)) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => {
                let (xs, ys, i, j) = (x.values(), y.values(), i as usize, j as usize);
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
/// all of one type, in buffers from `context`.
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
        for i in 0..count {
            BitSlice::of_mut(bits.as_mut_slice::<u8>()).set_if(i, !null(i));
        }
        validity = Some(bits);
    }
    let mut words = context.values_buffer(count * 8)?;
    if !data_type.has_bytes() {
        for (i, word) in words.as_mut_slice::<i64>().iter_mut().enumerate() {
            let (column, row) = row(i);
            *word = *at!(column.values().words(), row as usize);
        }
        return Ok(ColumnView::new(context, data_type, words, validity)?);
    }
    let string = |i: usize| {
        let (column, row) = row(i);
        if column.is_null(row) { &[][..] } else { column.values().strings().get(row as usize) }
    };
    let total = (0..count).fold(4_u64, |total, i| total + strings_bytes(string(i)));
    // So no word's offset is past 32 bits.
    let Ok(total) = u32::try_from(total) else { return Err(Error::OutOfMemory) };
    let mut bytes = context.bytes_buffer(total as usize)?;
    let out = bytes.as_mut_slice::<u8>();
    at_mut!(out, ..4).fill(0);
    let mut at = 4;
    for (i, word) in at_mut!(words.as_mut_slice::<u64>(), ..count).iter_mut().enumerate() {
        let string = string(i);
        *word = if string.is_empty() { 0 } else { at as u64 };
        if !string.is_empty() {
            #[expect(clippy::cast_possible_truncation, reason = "below `total`")]
            at_mut!(out, at..at + 4).copy_from_slice(&(string.len() as u32).to_le_bytes());
            at_mut!(out, at + 4..at + 4 + string.len()).copy_from_slice(string);
            at += 4 + string.len();
        }
    }
    // SAFETY: each word was written where its string is, just above.
    Ok(unsafe { ColumnView::strings_unchecked(context, words, bytes, validity)? })
}
