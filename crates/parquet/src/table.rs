//! `ParquetTable`: Parquet files with the same columns, scanned as one table.

use pipit_kernel::allocator::Allocator;
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::column::DataType;
use pipit_kernel::context::Context;
use pipit_kernel::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::Scannable;
use pipit_kernel::slow_vec::SlowVec;

use crate::Error;
use crate::chunk::ChunkReader;
use crate::footer::ParquetFile;

/// The most files a table can have.
const FILES_MAX: usize = 1 << 16;

pub struct ParquetTable<'a> {
    files: SlowVec<File<'a>>,
}

struct File<'a> {
    source: &'a dyn ByteSource,
    footer: ParquetFile,
}

/// Where a scan is: a file, a row group in it, and, once the group is
/// started, a reader for each column read and how many rows are left.
pub struct ScanState<'a> {
    file: usize,
    group: usize,
    readers: SlowVec<ChunkReader<'a>>,
    started: bool,
    left: u64,
}

impl<'a> ParquetTable<'a> {
    /// The files in `sources`, which must have the same columns, in the same
    /// order, of types the kernel has.
    pub fn open(
        allocator: &dyn Allocator,
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
        Ok(ParquetTable { files })
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

impl<'a> Scannable for ParquetTable<'a> {
    type State = ScanState<'a>;

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

    fn new_state(&self, context: &mut Context) -> Result<ScanState<'a>, Error> {
        let readers = SlowVec::fixed(context.allocator(), BATCH_COLUMNS_MAX as usize)?;
        Ok(ScanState { file: 0, group: 0, readers, started: false, left: 0 })
    }

    fn next(
        &self,
        columns: &[u32],
        context: &mut Context,
        state: &mut ScanState<'a>,
        batch: &mut RowBatch,
    ) -> Result<bool, Error> {
        loop {
            if !state.started {
                let Some(file) = self.files.get(state.file) else { return Ok(false) };
                if state.group == file.footer.row_groups() {
                    (state.file, state.group) = (state.file + 1, 0);
                    continue;
                }
                for &column in columns {
                    let c = column as usize;
                    let chunk = file.footer.chunk(state.group, c);
                    let column = *at!(file.footer.columns(), c);
                    let reader = ChunkReader::new(file.source, column, chunk)?;
                    state.readers.push(reader).map_err(|_| Error::OutOfMemory)?;
                }
                (state.started, state.left) = (true, file.footer.group_rows(state.group));
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
                (state.started, state.group) = (false, state.group + 1);
                continue;
            }
            batch.reset(rows);
            for reader in state.readers.iter_mut() {
                let Ok(()) = batch.push_column(reader.read(context, rows as usize)?) else {
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

    const SMALL: &[u8] = include_bytes!("../tests/data/small.parquet");
    const NULLS: &[u8] = include_bytes!("../tests/data/nulls.parquet");

    /// Every batch's rows of `columns`: each row's `id`s, or -1 for nulls.
    fn scan(table: &ParquetTable, columns: &[u32]) -> (usize, Vec<i64>) {
        let mut context = Context::new(&Heap);
        let mut state = table.new_state(&mut context).unwrap();
        let mut batch = RowBatch::new();
        let (mut rows, mut ids) = (0, Vec::new());
        while table.next(columns, &mut context, &mut state, &mut batch).unwrap() {
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
        let table = ParquetTable::open(&Heap, &[&NULLS, &NULLS]).unwrap();
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
        let opened = ParquetTable::open(&Heap, &[&NULLS, &SMALL]);
        assert_eq!(opened.err(), Some(Error::Unsupported));
    }
}
