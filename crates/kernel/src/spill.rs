//! `SpillStore`: where state that doesn't fit in memory goes, such as a
//! grouping's partitions or a sort's runs. It can be local disk, object
//! storage, or both in tiers, so its calls suit the strictest, S3: a log is
//! appended to, then sealed, and only then read back.
//!
//! Calls block. A store that's remote does its uploads and fetches on its
//! own threads, behind them.

use crate::bytes::ByteRange;
use crate::column::{ColumnView, DataType};
use crate::containers::BitSlice;
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX};

/// A log in a store, which the store names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LogId(pub u64);

pub trait SpillStore {
    /// A new, empty log.
    fn create(&self) -> Result<LogId, Error>;

    /// Appends `bytes` to `log`, which isn't sealed. Each block starts where
    /// the one before ended.
    fn append(&self, log: LogId, bytes: &[u8]) -> Result<ByteRange, Error>;

    /// Ends writing to `log`. Only a sealed log can be read.
    fn seal(&self, log: LogId) -> Result<(), Error>;

    /// Says `blocks` of `log` will be read soon, so a remote store can fetch
    /// them together. By default, nothing is done.
    fn prefetch(&self, log: LogId, blocks: &[ByteRange]) {
        let _ = (log, blocks);
    }

    /// Reads `block` of `log`, which is sealed, into `into`, which is its
    /// length.
    fn read(&self, log: LogId, block: ByteRange, into: &mut [u8]) -> Result<(), Error>;

    /// Deletes `log`, sealed or not.
    fn delete(&self, log: LogId);
}

/// A column written to a log: what's needed to read it back.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SpilledColumn {
    /// What type its values are.
    pub data_type: DataType,
    /// How many rows it has.
    pub row_count: u32,
    /// Its values, or if its type has bytes, views into them.
    pub values: ByteRange,
    /// If its type has bytes, them, one string after another.
    pub bytes: Option<ByteRange>,
    /// Its null bitmap, if it may have nulls.
    pub validity: Option<ByteRange>,
}

/// How many bytes `Appender` appends at once.
const CHUNK_BYTES: usize = 4096;

/// Appends `column`, which must be flat, to `log`: its values, or its
/// strings' views and then bytes, then its null bitmap, if any.
pub fn write_column(
    store: &dyn SpillStore,
    log: LogId,
    column: &ColumnView,
) -> Result<SpilledColumn, Error> {
    write_rows(store, log, column, &[], column.row_count())
}

/// As `write_column`, of the `count` rows of `column` that `select` lists,
/// or its first `count` if it lists none.
pub fn write_rows(
    store: &dyn SpillStore,
    log: LogId,
    column: &ColumnView,
    select: &[u16],
    count: u32,
) -> Result<SpilledColumn, Error> {
    let row_of = |i: u32| if select.is_empty() { i } else { u32::from(*at!(select, i as usize)) };
    let (values, bytes) = if column.data_type().has_bytes() {
        // Each view says where its string starts among those written after.
        let strings = column.string_values();
        let (mut views, mut start) = (Appender::new(store, log), 0_u32);
        for i in 0..count {
            #[expect(clippy::cast_possible_truncation, reason = "a view's length is a `u32`")]
            let len = strings.get(row_of(i) as usize).len() as u32;
            views.push(&start.to_le_bytes())?;
            views.push(&len.to_le_bytes())?;
            start += len;
        }
        let views = views.end()?;
        let mut bytes = Appender::new(store, log);
        for i in 0..count {
            bytes.push(strings.get(row_of(i) as usize))?;
        }
        (views, Some(bytes.end()?))
    } else {
        let (words, mut values) = (column.value_bytes(), Appender::new(store, log));
        for i in 0..count {
            let at = row_of(i) as usize * 8;
            values.push(at!(words, at..at + 8))?;
        }
        (values.end()?, None)
    };
    let validity = if column.validity().is_none() {
        None
    } else {
        let (mut bits, mut byte) = (Appender::new(store, log), 0_u8);
        for i in 0..count {
            byte |= u8::from(!column.is_null(row_of(i))) << (i % 8);
            if i % 8 == 7 || i + 1 == count {
                bits.push(&[byte])?;
                byte = 0;
            }
        }
        Some(bits.end()?)
    };
    let data_type = column.data_type();
    Ok(SpilledColumn { data_type, row_count: count, values, bytes, validity })
}

/// The types a batch's header names, by their number.
const TYPES: [DataType; 3] = [DataType::Int64, DataType::Float64, DataType::String];

/// The bytes of the header of a batch of `columns` columns: its rows, then
/// a word for each column.
fn header_bytes(columns: usize) -> usize {
    8 * (1 + columns)
}

/// Appends the `count` rows of `columns`, each flat, that `select` lists, or
/// their first `count` if it lists none, to `log`, after a header saying
/// how long each is, so `read_batch` needs only where it starts.
pub fn write_batch(
    store: &dyn SpillStore,
    log: LogId,
    columns: &[&ColumnView],
    select: &[u16],
    count: u32,
) -> Result<(), Error> {
    check!(columns.len() <= BATCH_COLUMNS_MAX as usize);
    let mut header = [0_u8; 8 * (1 + BATCH_COLUMNS_MAX as usize)];
    at_mut!(header, ..8).copy_from_slice(&u64::from(count).to_le_bytes());
    for (word, column) in at_mut!(header, 8..).chunks_exact_mut(8).zip(columns) {
        let mut bytes = 0_u64;
        if column.data_type().has_bytes() {
            let strings = column.string_values();
            for i in 0..count {
                let row = if select.is_empty() {
                    i as usize
                } else {
                    usize::from(*at!(select, i as usize))
                };
                bytes += strings.get(row).len() as u64;
            }
        }
        let Some(kind) = TYPES.iter().position(|&kind| kind == column.data_type()) else {
            failed!()
        };
        let nullable = u64::from(column.validity().is_some());
        word.copy_from_slice(&(bytes << 8 | nullable << 4 | kind as u64).to_le_bytes());
    }
    store.append(log, at!(header, ..header_bytes(columns.len())))?;
    for column in columns {
        write_rows(store, log, column, select, count)?;
    }
    Ok(())
}

/// Reads the batch of `width` columns `write_batch` wrote at `at` of `log`,
/// which is sealed, giving each column to `column`, and returns where the
/// next batch starts.
pub fn read_batch(
    context: &mut Context,
    store: &dyn SpillStore,
    log: LogId,
    at: u64,
    width: usize,
    mut column: impl FnMut(ColumnView) -> Result<(), Error>,
) -> Result<u64, Error> {
    check!(width <= BATCH_COLUMNS_MAX as usize);
    let mut header = [0_u8; 8 * (1 + BATCH_COLUMNS_MAX as usize)];
    let header = at_mut!(header, ..header_bytes(width));
    store.read(log, ByteRange { offset: at, len: header.len() as u64 }, header)?;
    let mut words = header.as_chunks::<8>().0.iter().map(|&word| u64::from_le_bytes(word));
    let rows = words.next().unwrap_or(0);
    let Ok(row_count) = u32::try_from(rows) else { return Err(Error::Io) };
    let mut next = at + header.len() as u64;
    let mut take = |len: u64| {
        let block = ByteRange { offset: next, len };
        next += len;
        block
    };
    for word in words {
        let Some(&data_type) = TYPES.get((word & 0xf) as usize) else { return Err(Error::Io) };
        let values = take(rows * 8);
        let bytes = data_type.has_bytes().then(|| take(word >> 8));
        let validity = (word >> 4 & 1 != 0).then(|| take(rows.div_ceil(8)));
        let spilled = SpilledColumn { data_type, row_count, values, bytes, validity };
        column(read_column(context, store, log, &spilled)?)?;
    }
    Ok(next)
}

/// Appends bytes to a log a chunk at a time, as one block.
struct Appender<'s> {
    /// The store the log is in.
    store: &'s dyn SpillStore,
    /// The log.
    log: LogId,
    /// The bytes not yet appended.
    chunk: [u8; CHUNK_BYTES],
    /// How many of `chunk`'s are.
    filled: usize,
    /// What's been appended.
    block: Option<ByteRange>,
}

impl<'s> Appender<'s> {
    fn new(store: &'s dyn SpillStore, log: LogId) -> Appender<'s> {
        Appender { store, log, chunk: [0; CHUNK_BYTES], filled: 0, block: None }
    }

    fn push(&mut self, bytes: &[u8]) -> Result<(), Error> {
        if self.filled + bytes.len() > CHUNK_BYTES {
            self.flush()?;
        }
        if bytes.len() > CHUNK_BYTES {
            let appended = self.store.append(self.log, bytes)?;
            self.grow(appended);
        } else {
            at_mut!(self.chunk, self.filled..self.filled + bytes.len()).copy_from_slice(bytes);
            self.filled += bytes.len();
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Error> {
        let filled = core::mem::take(&mut self.filled);
        let appended = self.store.append(self.log, at!(self.chunk, ..filled))?;
        self.grow(appended);
        Ok(())
    }

    /// Adds `appended`, which follows it, to the block.
    fn grow(&mut self, appended: ByteRange) {
        self.block = Some(match self.block {
            None => appended,
            Some(block) => ByteRange { offset: block.offset, len: block.len + appended.len },
        });
    }

    /// The block of every byte pushed, appended.
    fn end(mut self) -> Result<ByteRange, Error> {
        if self.filled > 0 || self.block.is_none() {
            self.flush()?;
        }
        let Some(block) = self.block else { failed!() };
        Ok(block)
    }
}

/// Reads a column `write_column` wrote, of at most a batch's rows, into
/// buffers from `context`.
pub fn read_column(
    context: &mut Context,
    store: &dyn SpillStore,
    log: LogId,
    spilled: &SpilledColumn,
) -> Result<ColumnView, Error> {
    check!(spilled.row_count <= BATCH_ROWS_MAX);
    let has_bytes = spilled.data_type.has_bytes();
    let size_bytes = spilled.row_count as usize * spilled.data_type.width_bytes();
    check!(spilled.values.len == size_bytes as u64);
    // Reading a block writes every byte of each of these buffers.
    let mut values = context.values_buffer(size_bytes)?;
    store.read(log, spilled.values, values.as_mut_slice::<u8>())?;
    if has_bytes {
        // Views were written little-endian.
        for half in values.as_mut_slice::<u32>() {
            *half = u32::from_le(*half);
        }
    }
    let validity = match spilled.validity {
        None => None,
        Some(block) => {
            let len = BitSlice::<u8>::words_for(spilled.row_count as usize);
            check!(block.len == len as u64);
            let mut bits = context.small_buffer(len)?;
            store.read(log, block, bits.as_mut_slice::<u8>())?;
            Some(bits)
        }
    };
    let Some(block) = spilled.bytes.filter(|_| has_bytes) else {
        return Ok(ColumnView::new(context, spilled.data_type, values, validity)?);
    };
    let len = usize::try_from(block.len).map_err(|_| Error::OutOfMemory)?;
    let mut bytes = context.bytes_buffer(len)?;
    store.read(log, block, bytes.as_mut_slice::<u8>())?;
    Ok(ColumnView::strings(context, values, bytes, validity)?)
}

#[cfg(test)]
pub(crate) mod tests {
    use alloc::vec::Vec as StdVec;
    use core::cell::RefCell;

    use super::*;

    use crate::buffer::Buffer;
    use crate::context::Context;
    use crate::env::Env;

    /// Behaves as S3 does, in memory: appends are uploaded in parts of at
    /// least `PART` bytes, except the last; nothing can be read before its
    /// log is sealed. Counts the requests a remote store would make.
    #[derive(Default)]
    pub(crate) struct Strict {
        pub(crate) logs: RefCell<StdVec<Option<Log>>>,
        requests: core::cell::Cell<u32>,
    }

    #[derive(Default)]
    pub(crate) struct Log {
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
        fn create(&self) -> Result<LogId, Error> {
            let mut logs = self.logs.borrow_mut();
            logs.push(Some(Log::default()));
            Ok(LogId(logs.len() as u64 - 1))
        }

        fn append(&self, log: LogId, bytes: &[u8]) -> Result<ByteRange, Error> {
            let mut logs = self.logs.borrow_mut();
            let log = logs[index(log)].as_mut().ok_or(Error::Io)?;
            assert!(!log.sealed);
            let offset = (log.uploaded.len() + log.pending.len()) as u64;
            log.pending.extend_from_slice(bytes);
            if log.pending.len() >= PART {
                self.upload(log);
            }
            Ok(ByteRange { offset, len: bytes.len() as u64 })
        }

        fn seal(&self, log: LogId) -> Result<(), Error> {
            let mut logs = self.logs.borrow_mut();
            let log = logs[index(log)].as_mut().ok_or(Error::Io)?;
            if !log.pending.is_empty() {
                self.upload(log);
            }
            log.sealed = true;
            Ok(())
        }

        fn read(&self, log: LogId, block: ByteRange, into: &mut [u8]) -> Result<(), Error> {
            let logs = self.logs.borrow();
            let log = logs[index(log)].as_ref().ok_or(Error::Io)?;
            if !log.sealed {
                return Err(Error::Io);
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
        let null = |row: usize| nulls.iter().any(|&null| null as usize == row);
        let cells = values.iter().enumerate().map(|(row, &value)| (!null(row)).then_some(value));
        ColumnView::from_int64s(&mut Context::new(Env::current(), None), cells).unwrap()
    }

    fn rows(column: &ColumnView) -> StdVec<Option<i64>> {
        let values = column.int64s();
        (0..column.row_count()).map(|r| (!column.is_null(r)).then(|| values[r as usize])).collect()
    }

    #[test]
    fn columns_come_back_as_they_were_written() {
        let env = Env::current();
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
            let read = read_column(&mut Context::new(env, None), &store, log, spilled).unwrap();
            assert_eq!(rows(&read), rows(column));
        }
        store.delete(log);
    }

    #[test]
    fn string_columns_come_back_as_they_were_written() {
        let env = Env::current();
        let store = Strict::default();
        let words: StdVec<StdVec<u8>> =
            (0..700).map(|i| alloc::format!("word{i}").into_bytes()).collect();
        let mut views = Buffer::allocate(env, words.len() * 8).unwrap();
        let mut bytes = Buffer::allocate(env, words.iter().map(StdVec::len).sum()).unwrap();
        let mut at = 0;
        for (i, word) in words.iter().enumerate() {
            bytes.as_mut_slice::<u8>()[at..at + word.len()].copy_from_slice(word);
            let len = u32::try_from(word.len()).unwrap();
            views.as_mut_slice::<u32>()[2 * i..2 * i + 2]
                .copy_from_slice(&[u32::try_from(at).unwrap(), len]);
            at += word.len();
        }
        let mut validity = Buffer::allocate(env, BitSlice::<u8>::words_for(words.len())).unwrap();
        validity.as_mut_slice::<u8>().fill(0b1110_1111);
        let whole = ColumnView::strings(&mut Context::new(env, None), views, bytes, Some(validity))
            .unwrap();
        let columns = [whole.clone(), whole.slice(3, 600)];

        let log = store.create().unwrap();
        let spilled: StdVec<SpilledColumn> =
            columns.iter().map(|c| write_column(&store, log, c).unwrap()).collect();
        store.seal(log).unwrap();
        for (column, spilled) in columns.iter().zip(&spilled) {
            let read = read_column(&mut Context::new(env, None), &store, log, spilled).unwrap();
            let (got, want) = (read.string_values(), column.string_values());
            assert_eq!(got.len(), want.len());
            for row in 0..read.row_count() {
                assert_eq!(got.get(row as usize), want.get(row as usize));
                assert_eq!(read.is_null(row), column.is_null(row));
            }
        }
    }

    #[test]
    fn strings_come_back_with_only_their_bytes() {
        let env = Env::current();
        let store = Strict::default();
        // Longer than a chunk of appends, so they're written in parts.
        let text: StdVec<u8> =
            (0..3000_u32).map(|i| b'a' + u8::try_from(i % 26).unwrap()).collect();
        let mut page = Buffer::allocate(env, text.len()).unwrap();
        page.as_mut_slice::<u8>().copy_from_slice(&text);
        let spans: StdVec<(u32, u32)> = (0..600).map(|i| (i * 3 % 2000, i % 7)).collect();
        let mut views = Buffer::allocate(env, spans.len() * 8).unwrap();
        for (view, &(start, len)) in views.as_mut_slice::<u32>().chunks_mut(2).zip(&spans) {
            view.copy_from_slice(&[start, len]);
        }
        let column = ColumnView::strings(&mut Context::new(env, None), views, page, None).unwrap();

        let log = store.create().unwrap();
        let spilled = write_column(&store, log, &column).unwrap();
        store.seal(log).unwrap();
        let read = read_column(&mut Context::new(env, None), &store, log, &spilled).unwrap();
        let (got, want) = (read.string_values(), column.string_values());
        // The strings overlapped, from a buffer of 3,000 bytes; written out,
        // each has its own.
        assert_eq!(spilled.bytes.unwrap().len, (0..600).map(|i| i % 7).sum::<u64>());
        assert_eq!(got.len(), want.len());
        for row in 0..got.len() {
            assert_eq!(got.get(row), want.get(row));
        }
    }

    #[test]
    fn logs_are_read_only_once_sealed() {
        let env = Env::current();
        let store = Strict::default();
        let log = store.create().unwrap();
        let spilled = write_column(&store, log, &column(&[1, 2, 3], &[1])).unwrap();
        assert_eq!(
            read_column(&mut Context::new(env, None), &store, log, &spilled).err(),
            Some(Error::Io)
        );
        store.seal(log).unwrap();
        assert!(read_column(&mut Context::new(env, None), &store, log, &spilled).is_ok());
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
