//! `ParquetTable`: Parquet files with the same columns, scanned as one table.

use pipit_kernel::allocator::{AllocError, Allocator};
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::column::{Bounds, DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::Scannable;
use pipit_kernel::slow_vec::SlowVec;
use pipit_kernel::step::{DynSource, Source};

use crate::chunk::ChunkReader;
use crate::footer::ParquetFile;
use crate::{Codec, Error, bounds};

/// The most files a table can have.
const FILES_MAX: usize = 1 << 16;

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
/// column read, and how many rows are left.
struct Rows<'a> {
    readers: SlowVec<ChunkReader<'a>>,
    started: bool,
    left: u64,
}

impl<'a> Rows<'a> {
    fn new(allocator: &dyn Allocator, columns: usize) -> Result<Rows<'a>, Error> {
        Ok(Rows { readers: SlowVec::fixed(allocator, columns)?, started: false, left: 0 })
    }

    /// Starts reading `columns` of row group `group` of `file`.
    fn start(
        &mut self,
        file: &File<'a>,
        codecs: &'a dyn Codec,
        columns: &[(u32, Forms)],
        group: usize,
    ) -> Result<(), Error> {
        for &(column, _) in columns {
            let c = column as usize;
            let chunk = file.footer.chunk(group, c);
            let column = *at!(file.footer.columns(), c);
            let reader = ChunkReader::new(file.source, codecs, column, chunk)?;
            self.readers.push(reader).map_err(|_| Error::OutOfMemory)?;
        }
        (self.started, self.left) = (true, file.footer.group_rows(group));
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
        // A batch stays within each column's page, so pages are read whole.
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
            return Ok(false);
        }
        batch.reset(rows);
        for (reader, &(_, forms)) in self.readers.iter_mut().zip(columns) {
            let Ok(()) = batch.push_column(reader.read(context, rows as usize, forms)?) else {
                pipit_kernel::check::check_failed(line!());
            };
        }
        self.left -= u64::from(rows);
        Ok(true)
    }
}

/// A scan of a `ParquetTable` that reads only `groups`, by file and row
/// group in it: those whose statistics don't rule out the values kept.
struct Pruned<'a> {
    table: &'a ParquetTable<'a>,
    columns: SlowVec<(u32, Forms)>,
    groups: SlowVec<(usize, usize)>,
}

impl<'a> Source for Pruned<'a> {
    /// How many of `groups` are started, and the rows of the last.
    type State = (usize, Rows<'a>);

    fn new_state(&self, context: &mut Context) -> Result<(usize, Rows<'a>), Error> {
        Ok((0, Rows::new(context.allocator(), self.columns.len())?))
    }

    fn next<'s>(
        &'s self,
        context: &mut Context,
        (started, rows): &mut (usize, Rows<'a>),
        batch: &mut RowBatch<'s>,
    ) -> Result<bool, Error> {
        loop {
            if !rows.started {
                let Some(&(file, group)) = self.groups.get(*started) else { return Ok(false) };
                let file = at!(self.table.files, file);
                rows.start(file, self.table.codecs, &self.columns, group)?;
                *started += 1;
            }
            if rows.next(context, &self.columns, batch)? {
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

impl Scannable for ParquetTable<'_> {
    type State<'s>
        = ScanState<'s>
    where
        Self: 's;
    type Loader = ();

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

    /// Constant where a page's run of one entry covers a batch, and
    /// dictionary for strings' dictionary pages.
    fn forms(&self, _: u32) -> Forms {
        Forms::FLAT | Forms::CONSTANT | Forms::DICTIONARY
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
                state.rows.start(file, self.codecs, state.columns, state.group)?;
                state.group += 1;
            }
            if state.rows.next(context, state.columns, batch)? {
                return Ok(true);
            }
        }
    }

    /// Skips the row groups whose footer statistics rule out a range.
    fn scan_within<'s>(
        &'s self,
        allocator: &dyn Allocator,
        columns: &[(u32, Forms)],
        ranges: &[(u32, Bounds)],
    ) -> Result<Option<DynSource<'s>>, AllocError> {
        let may_keep = |file: &File, group: usize| {
            ranges.iter().all(|&(column, range)| {
                let c = column as usize;
                let chunk = file.footer.chunk(group, c);
                let bounds = bounds::of_chunk(*at!(file.footer.columns(), c), chunk);
                bounds.is_none_or(|bounds| bounds.overlaps(range))
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
        if !skipped {
            return Ok(None);
        }
        let columns = SlowVec::fixed_from(allocator, columns.iter().copied())?;
        Ok(Some(DynSource::new(allocator, Pruned { table: self, columns, groups })?))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec::Vec;

    use pipit_kernel::allocator::Heap;
    use pipit_kernel::pipeline::Pipeline;
    use pipit_kernel::query_allocators::QueryAllocators;
    use pipit_kernel::scannable::DynScannable;

    use super::*;
    use crate::Uncompressed;

    const SMALL: &[u8] = include_bytes!("../tests/data/small.parquet");
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

    #[test]
    fn skips_row_groups_statistics_rule_out() {
        let table = ParquetTable::open(&Heap, &Uncompressed, &[&SMALL, &SMALL]).unwrap();
        let table = DynScannable::new(&Heap, table).unwrap();
        // Each file's row groups have `id`s 0 to 2047, 2048 to 4095 and 4096
        // to 4999.
        let ids = |min, max| {
            let ranges = [(0, Bounds { min, max })];
            let source = table.scan(&Heap, [(0, Forms::FLAT)].into_iter(), &ranges).unwrap();
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
        const DICT: &[u8] = include_bytes!("../tests/data/dict.parquet");
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
