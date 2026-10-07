//! `ParquetTable`: Parquet files with the same columns, scanned as one table.

use pipit_kernel::allocator::{AllocError, Allocator};
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::column::{ColumnView, DataType, Forms};
use pipit_kernel::condition::{CONDITIONS_MAX, Condition};
use pipit_kernel::context::Context;
use pipit_kernel::filter::Entries;
use pipit_kernel::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::Scannable;
use pipit_kernel::selection::{Kept, Selection};
use pipit_kernel::slow_vec::SlowVec;
use pipit_kernel::step::{DynSource, Source};

use crate::chunk::ChunkReader;
use crate::footer::ParquetFile;
use crate::{Codec, Error, bounds};

/// The most files a table can have.
const FILES_MAX: usize = 1 << 16;

/// A batch keeps few rows if fewer than one in this many: then `load`
/// decodes only those.
const KEPT_FEW: usize = 8;

/// The most row groups a pruned scan reads, in all files.
const ROW_GROUPS_MAX: usize = 1 << 24;

pub struct ParquetTable<'a> {
    files: SlowVec<File<'a>>,
    codecs: &'a dyn Codec,
}

struct File<'a> {
    source: &'a dyn ByteSource,
    footer: ParquetFile,
}

/// Where a scan is: the columns read, the next file and row group in it,
/// and the rows of the one started.
pub struct ScanState<'a> {
    columns: &'a [(u32, Forms)],
    file: usize,
    group: usize,
    rows: Rows<'a>,
}

/// Reads a row group a batch at a time, once started: a reader for each
/// column read, which for a lazy column only counts its rows, how many rows
/// are left, and which file and row group it is, for lazy columns' handles.
struct Rows<'a> {
    readers: SlowVec<ChunkReader<'a>>,
    started: bool,
    left: u64,
    file: u32,
    group: u32,
}

impl<'a> Rows<'a> {
    fn new(allocator: &dyn Allocator, columns: usize) -> Result<Rows<'a>, Error> {
        let readers = SlowVec::fixed(allocator, columns)?;
        Ok(Rows { readers, started: false, left: 0, file: 0, group: 0 })
    }

    /// Starts reading `columns` of row group `group` of file `file` of
    /// `table`.
    fn start(
        &mut self,
        table: &ParquetTable<'a>,
        columns: &[(u32, Forms)],
        file: usize,
        group: usize,
    ) -> Result<(), Error> {
        let (index, file) = (file, at!(table.files, file));
        for &(column, _) in columns {
            let c = column as usize;
            let chunk = file.footer.chunk(group, c);
            let column = *at!(file.footer.columns(), c);
            let reader = ChunkReader::new(file.source, table.codecs, column, chunk)?;
            self.readers.push(reader).map_err(|_| Error::OutOfMemory)?;
        }
        (self.started, self.left) = (true, file.footer.group_rows(group));
        self.file = u32::try_from(index).map_err(|_| Error::Unsupported)?;
        self.group = u32::try_from(group).map_err(|_| Error::Unsupported)?;
        Ok(())
    }

    /// Resets `batch` and fills it with the next rows of the row group
    /// started, or stops and returns false when none are left.
    fn next(
        &mut self,
        context: &mut Context,
        columns: &[(u32, Forms)],
        batch: &mut RowBatch,
    ) -> Result<bool, Error> {
        let rows = self.next_batch_rows(context)?;
        if rows == 0 {
            return Ok(false);
        }
        batch.reset(rows);
        self.fill(context, columns, rows, batch)?;
        Ok(true)
    }

    /// As `next`, but of the rows meeting `conditions`, each on the reader
    /// at its place after `outputs`', tested in turn. Once a batch has no
    /// rows left, the rest aren't tested and the batch is passed over.
    fn next_filtered(
        &mut self,
        context: &mut Context,
        outputs: &[(u32, Forms)],
        conditions: &[Condition],
        states: &mut [ConditionState],
        batch: &mut RowBatch,
    ) -> Result<bool, Error> {
        loop {
            let rows = self.next_batch_rows(context)?;
            if rows == 0 {
                return Ok(false);
            }
            batch.reset(rows);
            let tested = self.readers.iter_mut().skip(outputs.len());
            for ((condition, state), reader) in conditions.iter().zip(states.iter_mut()).zip(tested)
            {
                if batch.selection().is_empty() {
                    reader.skip(context, rows as usize)?;
                } else {
                    state.narrow(context, reader, condition, batch.selection_mut())?;
                }
            }
            if !batch.selection().is_empty() {
                self.fill(context, outputs, rows, batch)?;
                return Ok(true);
            }
            for (reader, &(_, forms)) in self.readers.iter_mut().zip(outputs) {
                if forms.contains(Forms::LAZY) {
                    reader.pass(rows as usize);
                } else {
                    reader.skip(context, rows as usize)?;
                }
            }
            self.left -= u64::from(rows);
        }
    }

    /// How many rows the next batch has: up to the first end of a page of
    /// any column, so pages are read whole. None, and the row group ended,
    /// once none are left.
    fn next_batch_rows(&mut self, context: &mut Context) -> Result<u32, Error> {
        let mut rows = u32::try_from(self.left).unwrap_or(u32::MAX).min(BATCH_ROWS_MAX);
        for reader in self.readers.iter_mut() {
            rows = rows.min(u32::try_from(reader.page_left(context)?).unwrap_or(u32::MAX));
        }
        if rows == 0 {
            if self.left > 0 {
                return Err(Error::Corrupt);
            }
            self.readers.retain(|_| false);
            self.started = false;
        }
        Ok(rows)
    }

    /// Fills `batch`, reset to `rows` rows, with `columns`' next `rows` rows.
    fn fill(
        &mut self,
        context: &mut Context,
        columns: &[(u32, Forms)],
        rows: u32,
        batch: &mut RowBatch,
    ) -> Result<(), Error> {
        for (reader, &(column, forms)) in self.readers.iter_mut().zip(columns) {
            let view = if forms.contains(Forms::LAZY) {
                // Where its rows are, for `load`: only counted here, so no
                // page body is read.
                let (header, row) = reader.place();
                let row = u32::try_from(row).map_err(|_| Error::Corrupt)?;
                let handle = Handle { file: self.file, group: self.group, column, row, header };
                reader.pass(rows as usize);
                ColumnView::lazy(context, reader.data_type(), &handle.bytes(), rows)?
            } else {
                reader.read(context, rows as usize, forms)?
            };
            let Ok(()) = batch.push_column(view) else {
                pipit_kernel::check::check_failed(line!());
            };
        }
        self.left -= u64::from(rows);
        Ok(())
    }
}

/// What applying a condition keeps between batches.
struct ConditionState {
    /// The dictionary entries its comparisons found, as a filter's.
    entries: SlowVec<Entries>,
    /// The last dictionary whose entries it tested.
    dictionary: Option<ColumnView>,
    /// A bit for each of `dictionary`'s entries: whether it keeps it.
    kept_entries: SlowVec<u64>,
}

impl ConditionState {
    /// Narrows `selection`, of the next rows of `reader`, to those meeting
    /// `condition`, passing them: tested from a dictionary's runs where the
    /// page allows and no row is dropped yet, or else read and then tested.
    fn narrow(
        &mut self,
        context: &mut Context,
        reader: &mut ChunkReader,
        condition: &Condition,
        selection: &mut Selection,
    ) -> Result<(), Error> {
        let rows = selection.rows() as usize;
        // From runs only while every row is kept: the first condition, which
        // drops most.
        let dictionary = match selection.kept() {
            Kept::All => reader.page_dictionary(context)?,
            _ => None,
        };
        if let Some(values) = dictionary {
            if !self.dictionary.as_ref().is_some_and(|tested| tested.shares_values(&values)) {
                let words = core::iter::repeat_n(0, (values.row_count() as usize).div_ceil(64));
                self.kept_entries = SlowVec::fixed_from(context.allocator(), words)?;
                let kept_entries = &mut self.kept_entries;
                condition.kept_entries(context, &mut self.entries, &values, kept_entries)?;
                self.dictionary = Some(values);
            }
            if reader.select_kept_rows(context, rows, &self.kept_entries, selection)? {
                return Ok(());
            }
        }
        let column = reader.read(context, rows, condition.accepts())?;
        Ok(condition.select_column(context, &mut self.entries, &column, selection)?)
    }
}

/// A scan of a `ParquetTable` given conditions: it reads only the row
/// groups whose statistics don't rule a condition out, and only the rows
/// meeting those it applies.
struct Within<'a> {
    table: &'a ParquetTable<'a>,
    /// The columns it reads: the `output_count` it outputs, then the column
    /// of each of `applied`, read only to test it.
    columns: SlowVec<(u32, Forms)>,
    output_count: usize,
    /// The row groups it reads, by file and row group in it.
    groups: SlowVec<(usize, usize)>,
    /// The conditions it applies.
    applied: SlowVec<Condition>,
}

impl<'a> Source for Within<'a> {
    /// How many of `groups` are started, the rows of the last, and what
    /// applying each of `applied` keeps.
    type State = (usize, Rows<'a>, SlowVec<ConditionState>);

    fn new_state(&self, context: &mut Context) -> Result<Self::State, Error> {
        let mut states = SlowVec::fixed(context.allocator(), self.applied.len())?;
        for condition in self.applied.iter() {
            let entries = condition.new_entries(context)?;
            let kept_entries = SlowVec::fixed(context.allocator(), 0)?;
            let state = ConditionState { entries, dictionary: None, kept_entries };
            states.push(state).map_err(|_| Error::OutOfMemory)?;
        }
        Ok((0, Rows::new(context.allocator(), self.columns.len())?, states))
    }

    fn next<'s>(
        &'s self,
        context: &mut Context,
        (started, rows, states): &mut Self::State,
        batch: &mut RowBatch<'s>,
    ) -> Result<bool, Error> {
        let outputs = at!(self.columns, ..self.output_count);
        loop {
            if !rows.started {
                let Some(&(file, group)) = self.groups.get(*started) else { return Ok(false) };
                rows.start(self.table, &self.columns, file, group)?;
                *started += 1;
            }
            let more = if self.applied.is_empty() {
                rows.next(context, outputs, batch)?
            } else {
                rows.next_filtered(context, outputs, &self.applied, states, batch)?
            };
            if more {
                return Ok(true);
            }
        }
    }
}

impl<'a> ParquetTable<'a> {
    /// The files in `sources`, which must have the same columns, in the same
    /// order, of types the kernel has. Their pages are decompressed with
    /// `codecs`.
    pub fn open(
        allocator: &dyn Allocator,
        codecs: &'a dyn Codec,
        sources: &[&'a dyn ByteSource],
    ) -> Result<ParquetTable<'a>, Error> {
        let mut files: SlowVec<File<'a>> = SlowVec::new(allocator, FILES_MAX)?;
        for &source in sources {
            let footer = ParquetFile::open(allocator, source)?;
            if let Some(first) = files.first() {
                check_same_columns(&first.footer, &footer)?;
            } else if footer.columns().iter().any(|column| column.data_type().is_none()) {
                return Err(Error::Unsupported);
            }
            files.push(File { source, footer }).map_err(|_| Error::OutOfMemory)?;
        }
        if files.is_empty() {
            return Err(Error::Unsupported);
        }
        Ok(ParquetTable { files, codecs })
    }

    fn first(&self) -> &ParquetFile {
        &at!(self.files, 0).footer
    }
}

/// Fails unless `file` has the columns `first` has.
fn check_same_columns(first: &ParquetFile, file: &ParquetFile) -> Result<(), Error> {
    let (a, b) = (first.columns(), file.columns());
    let same = a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            (a.physical, a.optional) == (b.physical, b.optional) && first.name(a) == file.name(b)
        });
    if same { Ok(()) } else { Err(Error::Unsupported) }
}

/// Where a lazy column's rows are: column `column` of row group `group` of
/// file `file`, from row `row` of the page whose header starts at `header`.
struct Handle {
    file: u32,
    group: u32,
    column: u32,
    row: u32,
    header: u64,
}

impl Handle {
    const BYTES: usize = 24;

    fn bytes(&self) -> [u8; Handle::BYTES] {
        let mut bytes = [0; Handle::BYTES];
        let words = [self.file, self.group, self.column, self.row];
        for (to, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            to.copy_from_slice(&word.to_le_bytes());
        }
        at_mut!(bytes, 16..).copy_from_slice(&self.header.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Handle, Error> {
        let Some((words, header)) = bytes.split_first_chunk::<16>() else {
            return Err(Error::Corrupt);
        };
        let header = u64::from_le_bytes(header.try_into().map_err(|_| Error::Corrupt)?);
        let (words, _) = words.as_chunks::<4>();
        let [file, group, column, row] =
            core::array::from_fn(|i| u32::from_le_bytes(*at!(words, i)));
        Ok(Handle { file, group, column, row, header })
    }
}

/// What loading lazy columns keeps between batches: a reader for each
/// column loaded, with the file and row group it reads.
pub struct Loader<'a> {
    readers: SlowVec<((u32, u32, u32), ChunkReader<'a>)>,
}

impl<'a> Scannable for ParquetTable<'a> {
    type State<'s>
        = ScanState<'s>
    where
        Self: 's;
    type Loader = Loader<'a>;

    fn column_count(&self) -> u32 {
        let Ok(count) = u32::try_from(self.first().columns().len()) else {
            pipit_kernel::check::check_failed(line!());
        };
        count
    }

    fn column_name(&self, column: u32) -> &str {
        let first = self.first();
        first.name(at!(first.columns(), column as usize))
    }

    fn column_type(&self, column: u32) -> DataType {
        let Some(data_type) = at!(self.first().columns(), column as usize).data_type() else {
            pipit_kernel::check::check_failed(line!());
        };
        data_type
    }

    /// Constant where a page's run of one entry covers a batch, dictionary
    /// for strings' dictionary pages, and lazy, loaded a batch at a time.
    fn forms(&self, _: u32) -> Forms {
        Forms::FLAT | Forms::CONSTANT | Forms::DICTIONARY | Forms::LAZY
    }

    fn new_loader(&self, context: &mut Context) -> Result<Loader<'a>, Error> {
        Ok(Loader { readers: SlowVec::fixed(context.allocator(), BATCH_COLUMNS_MAX as usize)? })
    }

    /// Reads the rows of `lazy` `selection` keeps, keeping a reader for each
    /// column from batch to batch. Only those rows' values are decoded when
    /// few are kept; all are otherwise, which is faster per row.
    fn load(
        &self,
        context: &mut Context,
        loader: &mut Loader<'a>,
        lazy: &ColumnView,
        selection: &Selection,
        forms: Forms,
    ) -> Result<ColumnView, Error> {
        let (bytes, start) = lazy.handle();
        let handle = Handle::from_bytes(bytes)?;
        let key = (handle.column, handle.file, handle.group);
        let found = loader.readers.iter().position(|((column, ..), _)| *column == handle.column);
        let at = match found {
            Some(at) if at!(loader.readers, at).0 == key => at,
            _ => {
                let file = self.files.get(handle.file as usize).ok_or(Error::Corrupt)?;
                let (group, c) = (handle.group as usize, handle.column as usize);
                if group >= file.footer.row_groups() || c >= file.footer.columns().len() {
                    return Err(Error::Corrupt);
                }
                let column = *at!(file.footer.columns(), c);
                let chunk = file.footer.chunk(group, c);
                let mut reader = ChunkReader::new(file.source, self.codecs, column, chunk)?;
                // Past the dictionary page, if any, so data pages can be
                // gone to.
                reader.page_left(context)?;
                if let Some(at) = found {
                    *at_mut!(loader.readers, at) = (key, reader);
                    at
                } else {
                    loader.readers.push((key, reader)).map_err(|_| Error::OutOfMemory)?;
                    loader.readers.len() - 1
                }
            }
        };
        let reader = &mut at_mut!(loader.readers, at).1;
        reader.go_to(context, handle.header, handle.row as usize + start as usize)?;
        let rows = lazy.row_count() as usize;
        match selection.kept() {
            Kept::Select(kept) if kept.len() < rows / KEPT_FEW => {
                reader.read_kept(context, rows, kept, forms)
            }
            _ => reader.read(context, rows, forms),
        }
    }

    fn open<'s>(
        &'s self,
        context: &mut Context,
        columns: &'s [(u32, Forms)],
    ) -> Result<ScanState<'s>, Error> {
        check!(columns.len() <= BATCH_COLUMNS_MAX as usize);
        check!(columns.iter().all(|&(column, _)| column < self.column_count()));
        let rows = Rows::new(context.allocator(), columns.len())?;
        Ok(ScanState { columns, file: 0, group: 0, rows })
    }

    fn next<'s>(
        &'s self,
        context: &mut Context,
        state: &mut ScanState<'s>,
        batch: &mut RowBatch,
    ) -> Result<bool, Error> {
        loop {
            if !state.rows.started {
                let Some(file) = self.files.get(state.file) else { return Ok(false) };
                if state.group == file.footer.row_groups() {
                    (state.file, state.group) = (state.file + 1, 0);
                    continue;
                }
                state.rows.start(self, state.columns, state.file, state.group)?;
                state.group += 1;
            }
            if state.rows.next(context, state.columns, batch)? {
                return Ok(true);
            }
        }
    }

    /// Skips the row groups whose footer statistics rule out a condition.
    fn scan_within<'s>(
        &'s self,
        allocator: &dyn Allocator,
        columns: &[(u32, Forms)],
        conditions: &[Condition],
    ) -> Result<Option<DynSource<'s>>, AllocError> {
        let may_keep = |file: &File, group: usize| {
            conditions.iter().all(|condition| {
                let c = condition.column() as usize;
                let chunk = file.footer.chunk(group, c);
                let bounds = bounds::of_chunk(*at!(file.footer.columns(), c), chunk);
                bounds.is_none_or(|bounds| condition.may_keep(bounds))
            })
        };
        let mut groups = SlowVec::new(allocator, ROW_GROUPS_MAX)?;
        let mut skipped = false;
        for (f, file) in self.files.iter().enumerate() {
            for group in 0..file.footer.row_groups() {
                if may_keep(file, group) {
                    groups.push((f, group))?;
                } else {
                    skipped = true;
                }
            }
        }
        let applied_count = conditions.iter().filter(|condition| condition.is_applied()).count();
        if !skipped && applied_count == 0 {
            return Ok(None);
        }
        // The columns read: those asked for, then those only tested, each
        // with a reader of its own, even if it's also asked for.
        let mut read = SlowVec::new(allocator, BATCH_COLUMNS_MAX as usize + CONDITIONS_MAX)?;
        let mut applied = SlowVec::fixed(allocator, applied_count)?;
        for &column in columns {
            read.push(column)?;
        }
        for condition in conditions.iter().filter(|condition| condition.is_applied()) {
            read.push((condition.column(), Forms::FLAT))?;
            applied.push(condition.copy(allocator)?)?;
        }
        let output_count = columns.len();
        let within = Within { table: self, columns: read, output_count, groups, applied };
        Ok(Some(DynSource::new(allocator, within)?))
    }

    /// Every one: it tests them as it reads, from a dictionary's runs where
    /// it can, and otherwise reads their columns and then tests them.
    fn applies(&self, _: &Condition) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::string::{String, ToString};
    use std::vec::Vec;

    use pipit_kernel::allocator::Heap;
    use pipit_kernel::filter::{Comparison, Value};
    use pipit_kernel::pipeline::Pipeline;
    use pipit_kernel::predicate::{Leaf, Node, Predicate};
    use pipit_kernel::query_allocators::QueryAllocators;
    use pipit_kernel::scannable::DynScannable;

    use super::*;
    use crate::Uncompressed;

    const SMALL: &[u8] = include_bytes!("../tests/data/small.parquet");
    const DICT: &[u8] = include_bytes!("../tests/data/dict.parquet");
    const NULLS: &[u8] = include_bytes!("../tests/data/nulls.parquet");

    /// Every batch's rows of `columns`: each row's `id`s, or -1 for nulls.
    fn scan(table: &ParquetTable, columns: &[u32]) -> (usize, Vec<i64>) {
        let mut context = Context::new(&Heap);
        let read: Vec<_> = columns.iter().map(|&column| (column, Forms::FLAT)).collect();
        let mut state = table.open(&mut context, &read).unwrap();
        let mut batch = RowBatch::new();
        let (mut rows, mut ids) = (0, Vec::new());
        while table.next(&mut context, &mut state, &mut batch).unwrap() {
            assert_eq!(batch.column_count() as usize, columns.len());
            rows += batch.row_count() as usize;
            if let Some((i, _)) = (0..).zip(columns).find(|&(_, &c)| c == 0) {
                let column = batch.column(i);
                ids.extend((0..column.row_count()).map(|row| {
                    if column.is_null(row) { -1 } else { column.int64s()[row as usize] }
                }));
            }
        }
        (rows, ids)
    }

    /// Each row of `column`, as text, `-` for nulls.
    fn cells(context: &mut Context, column: &ColumnView) -> Vec<String> {
        let mut flat = column.clone();
        flat.make_in(context, Forms::FLAT).unwrap();
        (0..flat.row_count())
            .map(|row| match flat.data_type() {
                _ if flat.is_null(row) => "-".into(),
                DataType::Int64 => flat.int64s()[row as usize].to_string(),
                DataType::Float64 => flat.float64s()[row as usize].to_string(),
                DataType::String => {
                    String::from_utf8_lossy(flat.string_values().get(row as usize)).into()
                }
            })
            .collect()
    }

    #[test]
    fn loads_lazy_columns_as_read_eagerly() {
        let table = ParquetTable::open(&Heap, &Uncompressed, &[&NULLS, &NULLS]).unwrap();
        let mut context = Context::new(&Heap);
        let batches = |forms: Forms, context: &mut Context| {
            let read = [(2, forms), (0, forms), (1, forms)];
            let mut state = table.open(context, &read).unwrap();
            let (mut batch, mut batches) = (RowBatch::new(), Vec::new());
            while table.next(context, &mut state, &mut batch).unwrap() {
                batches.push((0..3).map(|i| batch.column(i).clone()).collect::<Vec<_>>());
            }
            batches
        };
        let eager = batches(Forms::FLAT, &mut context);
        let lazy = batches(Forms::FLAT | Forms::LAZY, &mut context);
        assert_eq!(lazy.len(), eager.len());
        assert!(lazy.iter().flatten().all(ColumnView::is_lazy));
        // Every other batch, skipping some, then the rest backwards, going
        // back within pages and to earlier row groups, and one twice.
        let n = lazy.len();
        let order = (0..n).step_by(2).chain((0..n).rev().filter(|b| b % 2 == 1)).chain([0]);
        let mut loader = table.new_loader(&mut context).unwrap();
        for b in order {
            for (lazy, eager) in lazy[b].iter().zip(&eager[b]) {
                let all = Selection::all(lazy.row_count());
                let column =
                    table.load(&mut context, &mut loader, lazy, &all, Forms::FLAT).unwrap();
                assert_eq!(cells(&mut context, &column), cells(&mut context, eager), "batch {b}");
            }
        }
    }

    #[test]
    fn scans_files_as_one_table() {
        let table = ParquetTable::open(&Heap, &Uncompressed, &[&NULLS, &NULLS]).unwrap();
        assert_eq!(table.column_count(), 3);
        assert_eq!(table.column_name(2), "name");
        assert!(table.column_type(2) == DataType::String);
        let expected: Vec<i64> = (0..5000).map(|i| if i % 7 == 3 { -1 } else { i }).collect();
        let (rows, ids) = scan(&table, &[2, 0]);
        assert_eq!(rows, 10_000);
        assert_eq!((&ids[..5000], &ids[5000..]), (&expected[..], &expected[..]));
        // With no columns, only rows are counted.
        assert_eq!(scan(&table, &[]), (10_000, Vec::new()));
    }

    #[test]
    fn files_must_have_the_same_columns() {
        let opened = ParquetTable::open(&Heap, &Uncompressed, &[&NULLS, &SMALL]);
        assert_eq!(opened.err(), Some(Error::Unsupported));
    }

    /// 5000 rows: `a` is the row less 2500, a 32-bit integer; `u` 4e9 and
    /// the row, an unsigned one; `b` three times the row, 64 bits.
    const TYPES: &[u8] = include_bytes!("../tests/data/types.parquet");

    /// Ten decimals, stored as 32-bit integers.
    const DECIMAL: &[u8] = include_bytes!("../tests/data/decimal.parquet");

    #[test]
    fn bounds_columns_by_their_row_groups_statistics() {
        let table = ParquetTable::open(&Heap, &Uncompressed, &[&TYPES]).unwrap();
        let mut context = Context::new(&Heap);
        let read = [(0, Forms::FLAT), (1, Forms::FLAT), (2, Forms::FLAT)];
        let mut state = table.open(&mut context, &read).unwrap();
        let mut batch = RowBatch::new();
        assert!(table.next(&mut context, &mut state, &mut batch).unwrap());
        let bounds = |c: u32| batch.column(c).bounds().map(|b| (b.min, b.max));
        assert_eq!(bounds(0), Some((-2500, -453)));
        assert_eq!(bounds(1), Some((4_000_000_000, 4_000_002_047)));
        assert_eq!(bounds(2), Some((0, 6141)));
    }

    #[test]
    fn reads_unsigned_integers_above_31_bits() {
        let table = ParquetTable::open(&Heap, &Uncompressed, &[&TYPES]).unwrap();
        let mut context = Context::new(&Heap);
        let mut state = table.open(&mut context, &[(1, Forms::FLAT)]).unwrap();
        let mut batch = RowBatch::new();
        let mut values = Vec::new();
        while table.next(&mut context, &mut state, &mut batch).unwrap() {
            let mut column = batch.column(0).clone();
            column.make_in(&mut context, Forms::FLAT).unwrap();
            values.extend_from_slice(column.int64s());
        }
        assert!(values.iter().copied().eq((0..5000).map(|i| 4_000_000_000 + i)));
    }

    #[test]
    fn doesnt_read_decimals_as_integers() {
        let opened = ParquetTable::open(&Heap, &Uncompressed, &[&DECIMAL]);
        assert_eq!(opened.err(), Some(Error::Unsupported));
    }

    /// `min <= column 0 AND column 0 <= max`.
    fn between(min: i64, max: i64) -> Predicate {
        let compare = |comparison, value| {
            Node::Leaf(Leaf::Compare { column: 0, comparison, value: Value::Int64(value) })
        };
        let nodes = [
            compare(Comparison::GreaterEqual, min),
            compare(Comparison::LessEqual, max),
            Node::And(0, 1),
        ];
        Predicate::new(SlowVec::fixed_from(&Heap, nodes.into_iter()).unwrap())
    }

    /// `id`s and `name`s of the rows of `file` meeting `id = 3`, applied by
    /// the scan, with `id` read as well if `read_id`.
    fn filtered_rows(file: &[u8], read_id: bool) -> Vec<(Option<i64>, String)> {
        let table = ParquetTable::open(&Heap, &Uncompressed, &[&file]).unwrap();
        let mut context = Context::new(&Heap);
        let mut loader = table.new_loader(&mut context).unwrap();
        let mut read = Vec::from([(2, Forms::FLAT | Forms::LAZY)]);
        if read_id {
            read.push((0, Forms::FLAT));
        }
        let mut condition = Condition::new(0, between(3, 3));
        condition.set_applied();
        let conditions = [condition];
        let source = table.scan_within(&Heap, &read, &conditions).unwrap().unwrap();
        let pipeline = Pipeline::new(source, SlowVec::fixed(&Heap, 0).unwrap());
        let query = QueryAllocators::new(&Heap);
        let mut execution = pipeline.start(&query).unwrap();
        let (mut batch, mut rows) = (RowBatch::new(), Vec::new());
        while execution.next(&mut batch).unwrap() {
            let mut names = batch.column(0).clone();
            if names.is_lazy() {
                let selection = batch.selection();
                names =
                    table.load(&mut context, &mut loader, &names, selection, Forms::FLAT).unwrap();
            }
            let kept: Vec<u16> = match batch.selection().kept() {
                Kept::All => (0..u16::try_from(names.row_count()).unwrap()).collect(),
                Kept::None => Vec::new(),
                Kept::Select(kept) => kept.to_vec(),
            };
            for row in kept {
                let id = read_id.then(|| batch.column(1).int64s()[usize::from(row)]);
                let name = names.string_values().get(usize::from(row));
                rows.push((id, String::from_utf8_lossy(name).into_owned()));
            }
        }
        rows
    }

    #[test]
    fn applies_conditions_as_it_reads() {
        // `id` is plain in `small`, the row; and in `dict` a dictionary of
        // the row mod 10, null every 7th row from the 3rd.
        let in_dict = (0..5000).filter(|i| i % 10 == 3 && i % 7 != 3).count();
        for (file, count) in [(SMALL, 1), (DICT, in_dict)] {
            // Tested only, from its runs where it can be.
            let tested = filtered_rows(file, false);
            assert_eq!(tested.len(), count);
            // Output too, from a reader of its own: the same rows.
            let passed = filtered_rows(file, true);
            assert!(passed.iter().all(|(id, _)| *id == Some(3)));
            let names: Vec<_> = passed.into_iter().map(|(_, name)| (None, name)).collect();
            assert_eq!(names, tested);
        }
        assert_eq!(filtered_rows(SMALL, false), [(None, "row3".to_string())]);
    }

    #[test]
    fn applies_conditions_from_dictionary_runs() {
        // `id` is the row; `kind` is 0 in runs of 900 rows of each 1000, then
        // the row mod 7; `other` is the row mod 3: dictionary pages, without
        // nulls. The first condition keeps whole runs; the second narrows
        // the rows the first kept.
        const RUNS: &[u8] = include_bytes!("../tests/data/runs.parquet");
        let table = ParquetTable::open(&Heap, &Uncompressed, &[&RUNS]).unwrap();
        let mut conditions = [Condition::new(1, between(1, 6)), Condition::new(2, between(1, 1))];
        conditions.iter_mut().for_each(Condition::set_applied);
        let source = table.scan_within(&Heap, &[(0, Forms::FLAT)], &conditions).unwrap().unwrap();
        let pipeline = Pipeline::new(source, SlowVec::fixed(&Heap, 0).unwrap());
        let query = QueryAllocators::new(&Heap);
        let mut execution = pipeline.start(&query).unwrap();
        let (mut batch, mut ids) = (RowBatch::new(), Vec::new());
        while execution.next(&mut batch).unwrap() {
            let column = batch.column(0);
            match batch.selection().kept() {
                Kept::All => ids.extend(column.int64s()),
                Kept::None => panic!("a batch keeping none is passed over"),
                Kept::Select(kept) => {
                    ids.extend(kept.iter().map(|&row| column.int64s()[usize::from(row)]));
                }
            }
        }
        let meets = |i: &i64| i % 1000 >= 900 && i % 7 != 0 && i % 3 == 1;
        assert_eq!(ids, (0..5000).filter(meets).collect::<Vec<_>>());
    }

    #[test]
    fn skips_row_groups_statistics_rule_out() {
        let table = ParquetTable::open(&Heap, &Uncompressed, &[&SMALL, &SMALL]).unwrap();
        let table = DynScannable::new(&Heap, table).unwrap();
        // Each file's row groups have `id`s 0 to 2047, 2048 to 4095 and 4096
        // to 4999.
        let ids = |min, max| {
            // Not applied: it only skips.
            let conditions = [Condition::new(0, between(min, max))];
            let source = table.scan(&Heap, [(0, Forms::FLAT)].into_iter(), &conditions).unwrap();
            let pipeline = Pipeline::new(source, SlowVec::fixed(&Heap, 0).unwrap());
            let query = QueryAllocators::new(&Heap);
            let mut execution = pipeline.start(&query).unwrap();
            let (mut batch, mut ids) = (RowBatch::new(), Vec::new());
            while execution.next(&mut batch).unwrap() {
                let column = batch.column(0);
                ids.extend((0..column.row_count()).map(|row| column.int64s()[row as usize]));
            }
            ids
        };
        let middle: Vec<i64> = (2048..4096).collect();
        assert_eq!(ids(3000, 3000), [&middle[..], &middle[..]].concat());
        assert_eq!(ids(5000, i64::MAX), []);
        assert_eq!(ids(0, 4999).len(), 10000);
    }

    #[test]
    fn makes_dictionaries_where_allowed() {
        let table = ParquetTable::open(&Heap, &Uncompressed, &[&DICT]).unwrap();
        // `name`, a string column of dictionary pages.
        assert!(table.forms(2).contains(Forms::DICTIONARY));
        let first = |forms: Forms| {
            let mut context = Context::new(&Heap);
            let read = [(2, forms)];
            let mut state = table.open(&mut context, &read).unwrap();
            let mut batch = RowBatch::new();
            assert!(table.next(&mut context, &mut state, &mut batch).unwrap());
            let column = batch.column(0).clone();
            let dictionary = matches!(column.form(), pipit_kernel::column::Form::Dictionary(_));
            let mut flat = column;
            flat.make_in(&mut context, Forms::FLAT).unwrap();
            let strings = flat.string_values();
            let values: Vec<Vec<u8>> = (0..3).map(|row| strings.get(row).to_vec()).collect();
            (dictionary, values)
        };
        let (kept, values) = first(Forms::FLAT | Forms::DICTIONARY);
        assert!(kept);
        assert_eq!(first(Forms::FLAT), (false, values));
    }
}
