//! `SpillStore`: where state that doesn't fit in memory goes, such as a
//! grouping's partitions or a sort's runs. It can be local disk, object
//! storage, or both in tiers, so its calls suit the strictest, S3: a log is
//! appended to, then sealed, and only then read back.
//!
//! Calls block. A store that's remote does its uploads and fetches on its
//! own threads, behind them.

use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpillError {
    /// The store couldn't write or read, such as from a disk or network
    /// failure.
    Io,
    /// There was no memory to read into.
    OutOfMemory,
}

impl From<AllocError> for SpillError {
    fn from(_: AllocError) -> SpillError {
        SpillError::OutOfMemory
    }
}

/// A log in a store, which the store names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LogId(pub u64);

/// Where bytes appended to a log are.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Block {
    pub offset: u64,
    pub len: u64,
}

pub trait SpillStore {
    /// A new, empty log.
    fn create(&self) -> Result<LogId, SpillError>;

    /// Appends `bytes` to `log`, which isn't sealed. Each block starts where
    /// the one before ended.
    fn append(&self, log: LogId, bytes: &[u8]) -> Result<Block, SpillError>;

    /// Ends writing to `log`. Only a sealed log can be read.
    fn seal(&self, log: LogId) -> Result<(), SpillError>;

    /// Says `blocks` of `log` will be read soon, so a remote store can fetch
    /// them together. By default, nothing is done.
    fn prefetch(&self, log: LogId, blocks: &[Block]) {
        let _ = (log, blocks);
    }

    /// Reads `block` of `log`, which is sealed, into `into`, which is its
    /// length.
    fn read(&self, log: LogId, block: Block, into: &mut [u8]) -> Result<(), SpillError>;

    /// Deletes `log`, sealed or not.
    fn delete(&self, log: LogId);
}

/// A column written to a log: what's needed to read it back.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SpilledColumn {
    pub data_type: DataType,
    pub row_count: u32,
    /// Its values, or if its type has offsets, them, starting at 0.
    pub values: Block,
    /// If its type has offsets, the bytes they're into.
    pub bytes: Option<Block>,
    /// Its null bitmap, if it may have nulls.
    pub validity: Option<Block>,
}

/// How many rows' null bits `write_column` appends at once.
const CHUNK_ROWS: u32 = 512;

/// Appends `column` to `log`: its values, or its offsets and then bytes,
/// then its null bitmap, if any.
pub fn write_column(
    store: &dyn SpillStore,
    log: LogId,
    column: &ColumnView,
) -> Result<SpilledColumn, SpillError> {
    let (values, bytes) = if column.data_type().has_offsets() {
        let strings = column.string_values();
        (write_offsets(store, log, strings.offsets())?, Some(store.append(log, strings.bytes())?))
    } else {
        (store.append(log, column.value_bytes())?, None)
    };
    let validity = match column.validity() {
        None => None,
        Some(validity) => {
            // Bits from the first row, a chunk at a time, so a view that
            // starts mid-byte needs no memory to shift them in.
            let mut chunk = [0_u8; CHUNK_ROWS as usize / 8];
            let mut block: Option<Block> = None;
            let row_count = validity.row_count();
            for first in (0..row_count).step_by(CHUNK_ROWS as usize) {
                let rows = (row_count - first).min(CHUNK_ROWS);
                let bytes = at_mut!(chunk, ..rows.div_ceil(8) as usize);
                bytes.fill(0);
                for row in 0..rows {
                    // SAFETY: `first + row` is below `row_count`.
                    let valid = unsafe { validity.is_valid_unchecked(first + row) };
                    *at_mut!(bytes, row as usize / 8) |= u8::from(valid) << (row % 8);
                }
                let appended = store.append(log, bytes)?;
                block = Some(match block {
                    None => appended,
                    Some(block) => Block { offset: block.offset, len: block.len + appended.len },
                });
            }
            block
        }
    };
    let (data_type, row_count) = (column.data_type(), column.row_count());
    Ok(SpilledColumn { data_type, row_count, values, bytes, validity })
}

/// Appends `offsets`, less the first, so they start at 0, a chunk at a time,
/// as one block.
fn write_offsets(store: &dyn SpillStore, log: LogId, offsets: &[u32]) -> Result<Block, SpillError> {
    let first = *at!(offsets, 0);
    let mut chunk = [0_u8; CHUNK_ROWS as usize * 4];
    let mut block: Option<Block> = None;
    for part in offsets.chunks(CHUNK_ROWS as usize) {
        let bytes = at_mut!(chunk, ..part.len() * 4);
        for (to, &offset) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(part) {
            *to = (offset - first).to_le_bytes();
        }
        let appended = store.append(log, bytes)?;
        block = Some(match block {
            None => appended,
            Some(block) => Block { offset: block.offset, len: block.len + appended.len },
        });
    }
    block.ok_or(SpillError::Io)
}

/// Reads a column `write_column` wrote, into memory from `allocator`.
pub fn read_column(
    allocator: &dyn Allocator,
    store: &dyn SpillStore,
    log: LogId,
    spilled: &SpilledColumn,
) -> Result<ColumnView, SpillError> {
    let has_offsets = spilled.data_type.has_offsets();
    let size_bytes =
        (spilled.row_count as usize + usize::from(has_offsets)) * spilled.data_type.width_bytes();
    check!(spilled.values.len == size_bytes as u64);
    let mut values = Buffer::allocate(allocator, size_bytes)?;
    store.read(log, spilled.values, values.as_mut_slice::<u8>())?;
    if has_offsets {
        // Offsets were written little-endian.
        for offset in values.as_mut_slice::<u32>() {
            *offset = u32::from_le(*offset);
        }
    }
    let validity = match spilled.validity {
        None => None,
        Some(block) => {
            let len = spilled.row_count.div_ceil(8);
            check!(block.len == u64::from(len));
            let mut bits = Buffer::allocate(allocator, len as usize)?;
            store.read(log, block, bits.as_mut_slice::<u8>())?;
            Some(bits)
        }
    };
    let Some(block) = spilled.bytes.filter(|_| has_offsets) else {
        return Ok(ColumnView::new(spilled.data_type, values, validity));
    };
    let len = usize::try_from(block.len).map_err(|_| SpillError::OutOfMemory)?;
    let mut bytes = Buffer::allocate(allocator, len)?;
    store.read(log, block, bytes.as_mut_slice::<u8>())?;
    Ok(ColumnView::strings(values, bytes, validity))
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec as StdVec;
    use core::cell::RefCell;

    use super::*;
    use crate::allocator::Heap;

    /// Behaves as S3 does, in memory: appends are uploaded in parts of at
    /// least `PART` bytes, except the last; nothing can be read before its
    /// log is sealed. Counts the requests a remote store would make.
    #[derive(Default)]
    struct Strict {
        logs: RefCell<StdVec<Option<Log>>>,
        requests: core::cell::Cell<u32>,
    }

    #[derive(Default)]
    struct Log {
        uploaded: StdVec<u8>,
        pending: StdVec<u8>,
        sealed: bool,
    }

    const PART: usize = 64;

    fn index(log: LogId) -> usize {
        usize::try_from(log.0).unwrap()
    }

    impl Strict {
        fn upload(&self, log: &mut Log) {
            self.requests.set(self.requests.get() + 1);
            log.uploaded.append(&mut log.pending);
        }
    }

    impl SpillStore for Strict {
        fn create(&self) -> Result<LogId, SpillError> {
            let mut logs = self.logs.borrow_mut();
            logs.push(Some(Log::default()));
            Ok(LogId(logs.len() as u64 - 1))
        }

        fn append(&self, log: LogId, bytes: &[u8]) -> Result<Block, SpillError> {
            let mut logs = self.logs.borrow_mut();
            let log = logs[index(log)].as_mut().ok_or(SpillError::Io)?;
            assert!(!log.sealed);
            let offset = (log.uploaded.len() + log.pending.len()) as u64;
            log.pending.extend_from_slice(bytes);
            if log.pending.len() >= PART {
                self.upload(log);
            }
            Ok(Block { offset, len: bytes.len() as u64 })
        }

        fn seal(&self, log: LogId) -> Result<(), SpillError> {
            let mut logs = self.logs.borrow_mut();
            let log = logs[index(log)].as_mut().ok_or(SpillError::Io)?;
            if !log.pending.is_empty() {
                self.upload(log);
            }
            log.sealed = true;
            Ok(())
        }

        fn read(&self, log: LogId, block: Block, into: &mut [u8]) -> Result<(), SpillError> {
            let logs = self.logs.borrow();
            let log = logs[index(log)].as_ref().ok_or(SpillError::Io)?;
            if !log.sealed {
                return Err(SpillError::Io);
            }
            self.requests.set(self.requests.get() + 1);
            let start = usize::try_from(block.offset).unwrap();
            into.copy_from_slice(&log.uploaded[start..start + into.len()]);
            Ok(())
        }

        fn delete(&self, log: LogId) {
            self.logs.borrow_mut()[index(log)] = None;
        }
    }

    fn column(values: &[i64], nulls: &[u32]) -> ColumnView {
        let mut buffer = Buffer::allocate(&Heap, values.len() * 8).unwrap();
        buffer.as_mut_slice::<i64>().copy_from_slice(values);
        let validity = (!nulls.is_empty()).then(|| {
            let mut bits = Buffer::allocate(&Heap, values.len().div_ceil(8)).unwrap();
            bits.as_mut_slice::<u8>().fill(0xff);
            for &row in nulls {
                bits.as_mut_slice::<u8>()[row as usize / 8] &= !(1 << (row % 8));
            }
            bits
        });
        ColumnView::new(DataType::Int64, buffer, validity)
    }

    fn rows(column: &ColumnView) -> StdVec<Option<i64>> {
        let values = column.int64s();
        (0..column.row_count()).map(|r| (!column.is_null(r)).then(|| values[r as usize])).collect()
    }

    #[test]
    fn columns_come_back_as_they_were_written() {
        let store = Strict::default();
        let values: StdVec<i64> = (0..1000).collect();
        let whole = column(&values, &[3, 500, 999]);
        // A slice starting mid-byte of the null bitmap, and one without nulls.
        let columns = [whole.clone(), whole.slice(5, 700), column(&[7, 8, 9], &[])];

        let log = store.create().unwrap();
        let spilled: StdVec<SpilledColumn> =
            columns.iter().map(|c| write_column(&store, log, c).unwrap()).collect();
        store.seal(log).unwrap();
        for (column, spilled) in columns.iter().zip(&spilled) {
            let read = read_column(&Heap, &store, log, spilled).unwrap();
            assert_eq!(rows(&read), rows(column));
        }
        store.delete(log);
    }

    #[test]
    fn string_columns_come_back_as_they_were_written() {
        let store = Strict::default();
        let words: StdVec<StdVec<u8>> =
            (0..700).map(|i| alloc::format!("word{i}").into_bytes()).collect();
        let mut offsets = Buffer::allocate(&Heap, (words.len() + 1) * 4).unwrap();
        let mut bytes = Buffer::allocate(&Heap, words.iter().map(StdVec::len).sum()).unwrap();
        let mut at = 0;
        for (i, word) in words.iter().enumerate() {
            bytes.as_mut_slice::<u8>()[at..at + word.len()].copy_from_slice(word);
            at += word.len();
            offsets.as_mut_slice::<u32>()[i + 1] = u32::try_from(at).unwrap();
        }
        let mut validity = Buffer::allocate(&Heap, words.len().div_ceil(8)).unwrap();
        validity.as_mut_slice::<u8>().fill(0b1110_1111);
        let whole = ColumnView::strings(offsets, bytes, Some(validity));
        let columns = [whole.clone(), whole.slice(3, 600)];

        let log = store.create().unwrap();
        let spilled: StdVec<SpilledColumn> =
            columns.iter().map(|c| write_column(&store, log, c).unwrap()).collect();
        store.seal(log).unwrap();
        for (column, spilled) in columns.iter().zip(&spilled) {
            let read = read_column(&Heap, &store, log, spilled).unwrap();
            let (got, want) = (read.string_values(), column.string_values());
            assert_eq!(got.len(), want.len());
            for row in 0..read.row_count() {
                assert_eq!(got.get(row as usize), want.get(row as usize));
                assert_eq!(read.is_null(row), column.is_null(row));
            }
        }
    }

    #[test]
    fn logs_are_read_only_once_sealed() {
        let store = Strict::default();
        let log = store.create().unwrap();
        let spilled = write_column(&store, log, &column(&[1, 2, 3], &[1])).unwrap();
        assert_eq!(read_column(&Heap, &store, log, &spilled).err(), Some(SpillError::Io));
        store.seal(log).unwrap();
        assert!(read_column(&Heap, &store, log, &spilled).is_ok());
    }

    #[test]
    fn small_appends_are_uploaded_together() {
        let store = Strict::default();
        let log = store.create().unwrap();
        // Ten columns of two rows each: 160 bytes, in 64-byte parts.
        for _ in 0..10 {
            write_column(&store, log, &column(&[1, 2], &[])).unwrap();
        }
        store.seal(log).unwrap();
        assert_eq!(store.requests.get(), 3);
    }
}
