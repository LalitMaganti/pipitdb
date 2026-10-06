//! `ParquetTable`: Parquet files with the same columns, scanned as one table.

use pipit_kernel::allocator::Allocator;
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::column::{Bounds, DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::Scannable;
use pipit_kernel::slow_vec::SlowVec;

use crate::chunk::ChunkReader;
use crate::footer::ParquetFile;
use crate::{Codec, Error, bounds};

/// The most files a table can have.
const FILES_MAX: usize = 1 << 16;

/// The most row groups a table can have, in all its files.
const ROW_GROUPS_MAX: usize = 1 << 24;

pub struct ParquetTable<'a> {
    files: SlowVec<File<'a>>,
    /// Each row group, numbered across the files in order: its file, and
    /// which of the file's it is.
    row_groups: SlowVec<(u32, u32)>,
    codecs: &'a dyn Codec,
}

struct File<'a> {
    source: &'a dyn ByteSource,
    footer: ParquetFile,
}

/// Where a scan is: how many of the row groups it reads are read, and,
/// once one is started, a reader for each column read and how many rows are
/// left.
pub struct ScanState<'a> {
    columns: &'a [(u32, Forms)],
    row_groups: Option<&'a [u32]>,
    read: usize,
    readers: SlowVec<ChunkReader<'a>>,
    started: bool,
    left: u64,
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
        let mut row_groups = SlowVec::new(allocator, ROW_GROUPS_MAX)?;
        for (file, footer) in (0..).zip(files.iter().map(|file| &file.footer)) {
            for group in 0..footer.row_groups() {
                let group = u32::try_from(group).map_err(|_| Error::Unsupported)?;
                row_groups.push((file, group)).map_err(|_| Error::Unsupported)?;
            }
        }
        Ok(ParquetTable { files, row_groups, codecs })
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

    #[expect(clippy::cast_possible_truncation, reason = "at most `ROW_GROUPS_MAX`")]
    fn row_group_count(&self) -> u32 {
        self.row_groups.len() as u32
    }

    /// From the footer's statistics, and the column's type.
    fn bounds(&self, row_group: u32, column: u32) -> Option<Bounds> {
        let &(file, group) = at!(self.row_groups, row_group as usize);
        let footer = &at!(self.files, file as usize).footer;
        let chunk = footer.chunk(group as usize, column as usize);
        bounds::of_chunk(*at!(footer.columns(), column as usize), chunk)
    }

    fn open<'s>(
        &'s self,
        context: &mut Context,
        columns: &'s [(u32, Forms)],
        row_groups: Option<&'s [u32]>,
    ) -> Result<ScanState<'s>, Error> {
        check!(columns.len() <= BATCH_COLUMNS_MAX as usize);
        check!(columns.iter().all(|&(column, _)| column < self.column_count()));
        let readers = SlowVec::fixed(context.allocator(), columns.len())?;
        Ok(ScanState { columns, row_groups, read: 0, readers, started: false, left: 0 })
    }

    fn next<'s>(
        &'s self,
        context: &mut Context,
        state: &mut ScanState<'s>,
        batch: &mut RowBatch,
    ) -> Result<bool, Error> {
        loop {
            if !state.started {
                let next = match state.row_groups {
                    Some(row_groups) => row_groups.get(state.read).map(|&group| group as usize),
                    None => Some(state.read),
                };
                let Some(&(file, group)) = next.and_then(|next| self.row_groups.get(next)) else {
                    return Ok(false);
                };
                state.read += 1;
                let (file, group) = (at!(self.files, file as usize), group as usize);
                for &(column, _) in state.columns {
                    let c = column as usize;
                    let chunk = file.footer.chunk(group, c);
                    let column = *at!(file.footer.columns(), c);
                    let reader = ChunkReader::new(file.source, self.codecs, column, chunk)?;
                    state.readers.push(reader).map_err(|_| Error::OutOfMemory)?;
                }
                (state.started, state.left) = (true, file.footer.group_rows(group));
            }
            // A batch stays within each column's page, so pages are read
            // whole.
            let mut rows = u32::try_from(state.left).unwrap_or(u32::MAX).min(BATCH_ROWS_MAX);
            for reader in state.readers.iter_mut() {
                rows = rows.min(u32::try_from(reader.page_left(context)?).unwrap_or(u32::MAX));
            }
            if rows == 0 {
                if state.left > 0 {
                    return Err(Error::Corrupt);
                }
                state.readers.retain(|_| false);
                state.started = false;
                continue;
            }
            batch.reset(rows);
            for (reader, &(_, forms)) in state.readers.iter_mut().zip(state.columns) {
                let Ok(()) = batch.push_column(reader.read(context, rows as usize, forms)?) else {
                    pipit_kernel::check::check_failed(line!());
                };
            }
            state.left -= u64::from(rows);
            return Ok(true);
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec::Vec;

    use pipit_kernel::allocator::Heap;

    use super::*;
    use crate::Uncompressed;

    const SMALL: &[u8] = include_bytes!("../tests/data/small.parquet");
    const NULLS: &[u8] = include_bytes!("../tests/data/nulls.parquet");

    /// Every batch's rows of `columns`: each row's `id`s, or -1 for nulls.
    fn scan(table: &ParquetTable, columns: &[u32]) -> (usize, Vec<i64>) {
        let mut context = Context::new(&Heap);
        let read: Vec<_> = columns.iter().map(|&column| (column, Forms::FLAT)).collect();
        let mut state = table.open(&mut context, &read, None).unwrap();
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
        let mut state = table.open(&mut context, &read, None).unwrap();
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
        let mut state = table.open(&mut context, &[(1, Forms::FLAT)], None).unwrap();
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
    fn makes_dictionaries_where_allowed() {
        const DICT: &[u8] = include_bytes!("../tests/data/dict.parquet");
        let table = ParquetTable::open(&Heap, &Uncompressed, &[&DICT]).unwrap();
        // `name`, a string column of dictionary pages.
        assert!(table.forms(2).contains(Forms::DICTIONARY));
        let first = |forms: Forms| {
            let mut context = Context::new(&Heap);
            let read = [(2, forms)];
            let mut state = table.open(&mut context, &read, None).unwrap();
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

    #[test]
    fn numbers_row_groups_across_files_and_reads_those_given() {
        let table = ParquetTable::open(&Heap, &Uncompressed, &[&TYPES, &TYPES]).unwrap();
        let per_file = table.row_group_count() / 2;
        assert!(per_file >= 1 && table.row_group_count() == 2 * per_file);
        // From the footer: the same for the same row group of the same file.
        assert!(table.bounds(0, 0).is_some());
        assert_eq!(table.bounds(per_file, 0), table.bounds(0, 0));
        let read = [(0, Forms::FLAT)];
        let rows = |row_groups: Option<&[u32]>| {
            let mut context = Context::new(&Heap);
            let mut state = table.open(&mut context, &read, row_groups).unwrap();
            let mut batch = RowBatch::new();
            let mut rows = 0;
            while table.next(&mut context, &mut state, &mut batch).unwrap() {
                rows += batch.row_count();
            }
            rows
        };
        let each: u32 = (0..table.row_group_count()).map(|group| rows(Some(&[group]))).sum();
        assert_eq!(each, rows(None));
        assert_eq!(rows(Some(&[])), 0);
    }
}
