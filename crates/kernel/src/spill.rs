//! `SpillStore`: where state that doesn't fit in memory goes, such as a
//! grouping's partitions or a sort's runs. It can be local disk, object
//! storage, or both in tiers, so its calls suit the strictest, S3: a log is
//! appended to, then sealed, and only then read back.
//!
//! Calls block. A store that's remote does its uploads and fetches on its
//! own threads, behind them.

use crate::bytes::ByteRange;
use crate::column::{self, ColumnView, DataType, Strings, Validity};
use crate::containers::BitSlice;
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::BATCH_ROWS_MAX;

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

/// How many rows' null bits, or strings' words, `write_column` appends at
/// once.
const CHUNK_ROWS: u32 = 512;

/// The bytes of `CHUNK_ROWS` rows' null bits.
const CHUNK_BYTES: usize = CHUNK_ROWS as usize / 8;

/// Appends `column`, which must be flat, to `log`: its values, or its
/// strings' words and then bytes, then its null bitmap, if any.
pub fn write_column(
    store: &dyn SpillStore,
    log: LogId,
    column: &ColumnView,
) -> Result<SpilledColumn, Error> {
    let (values, bytes) = if column.data_type().has_bytes() {
        write_strings(store, log, column.string_values())?
    } else {
        (store.append(log, column.value_bytes())?, None)
    };
    let validity = match column.validity() {
        None => None,
        Some(validity) => Some(write_validity(store, log, validity)?),
    };
    let (data_type, row_count) = (column.data_type(), column.row_count());
    Ok(SpilledColumn { data_type, row_count, values, bytes, validity })
}

/// Appends `validity`'s bits from the first row, shifted a chunk at a time
/// if the view starts mid-byte, so they need no memory.
fn write_validity(
    store: &dyn SpillStore,
    log: LogId,
    validity: Validity,
) -> Result<ByteRange, Error> {
    let (bits, shift) = validity.bits();
    let len = BitSlice::<u8>::words_for(validity.row_count() as usize);
    let mut chunk = [0_u8; CHUNK_BYTES];
    let mut block = None;
    for first in (0..len).step_by(CHUNK_BYTES) {
        let count = (len - first).min(CHUNK_BYTES);
        for (i, byte) in at_mut!(chunk, ..count).iter_mut().enumerate() {
            let next = bits.get(first + i + 1).copied().unwrap_or(0);
            let pair = u16::from(next) << 8 | u16::from(*at!(bits, first + i));
            #[expect(clippy::cast_possible_truncation, reason = "a byte of the pair")]
            let shifted = (pair >> shift) as u8;
            *byte = shifted;
        }
        block = Some(grow(block, store.append(log, at!(chunk, ..count))?));
    }
    let Some(block) = block else { failed!() };
    Ok(block)
}

/// Appends `strings`' words, then their bytes, one after another, each
/// after its length, with the words pointing into them there, each a chunk
/// at a time, as a block each. The bytes start with an empty string, which
/// every empty string is.
fn write_strings(
    store: &dyn SpillStore,
    log: LogId,
    strings: Strings<'_>,
) -> Result<(ByteRange, Option<ByteRange>), Error> {
    let mut chunk = [0_u8; CHUNK_ROWS as usize * 8];
    let (mut words, mut filled, mut start) = (None, 0, 4_u64);
    for row in 0..strings.len() {
        let len = strings.len_of(row) as u64;
        let word = if len == 0 { 0 } else { start };
        at_mut!(chunk, filled..filled + 8).copy_from_slice(&word.to_le_bytes());
        filled += 8;
        start += if len == 0 { 0 } else { 4 + len };
        if filled == chunk.len() {
            words = Some(grow(words, store.append(log, &chunk)?));
            filled = 0;
        }
    }
    words = Some(grow(words, store.append(log, at!(chunk, ..filled))?));
    // So no word's offset is past 32 bits.
    if u32::try_from(start).is_err() {
        return Err(Error::OutOfMemory);
    }
    let mut bytes = Some(store.append(log, &[0; 4])?);
    filled = 0;
    for row in 0..strings.len() {
        let value = strings.get(row);
        if value.is_empty() {
            continue;
        }
        if filled + 4 + value.len() > chunk.len() {
            bytes = Some(grow(bytes, store.append(log, at!(chunk, ..filled))?));
            filled = 0;
        }
        #[expect(clippy::cast_possible_truncation, reason = "checked above")]
        let len = (value.len() as u32).to_le_bytes();
        if 4 + value.len() > chunk.len() {
            bytes = Some(grow(bytes, store.append(log, &len)?));
            bytes = Some(grow(bytes, store.append(log, value)?));
        } else {
            at_mut!(chunk, filled..filled + 4).copy_from_slice(&len);
            at_mut!(chunk, filled + 4..filled + 4 + value.len()).copy_from_slice(value);
            filled += 4 + value.len();
        }
    }
    bytes = Some(grow(bytes, store.append(log, at!(chunk, ..filled))?));
    let Some(words) = words else { failed!() };
    Ok((words, bytes))
}

/// `appended` added to `block`, which it follows.
fn grow(block: Option<ByteRange>, appended: ByteRange) -> ByteRange {
    match block {
        None => appended,
        Some(block) => ByteRange { offset: block.offset, len: block.len + appended.len },
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
        // Words were written little-endian.
        for word in values.as_mut_slice::<u64>() {
            *word = u64::from_le(*word);
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
    // What a store gives back may be damaged: strings are read unchecked.
    if !column::strings_valid(values.as_slice(), bytes.as_slice()) {
        return Err(Error::Corrupt);
    }
    Ok(ColumnView::strings(context, values, bytes, validity)?)
}

#[cfg(test)]
mod tests {
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
        // Every fifth null.
        let word = |i: usize| (i % 8 != 4).then(|| words[i].as_slice());
        let whole =
            ColumnView::from_strings(&mut Context::new(env, None), words.len(), word).unwrap();
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
        // Strings between other bytes, longer than a chunk of appends in
        // all, so they're written in parts, each read by several rows.
        let mut text = StdVec::new();
        let mut at = StdVec::new();
        for i in 0..200_u32 {
            text.push(b'.');
            at.push(u64::try_from(text.len()).unwrap());
            let len = i % 7;
            text.extend_from_slice(&len.to_le_bytes());
            text.extend((0..len).map(|j| b'a' + u8::try_from((i + j) % 26).unwrap()));
        }
        let mut page = Buffer::allocate(env, text.len()).unwrap();
        page.as_mut_slice::<u8>().copy_from_slice(&text);
        let mut words = Buffer::allocate(env, 600 * 8).unwrap();
        for (i, word) in words.as_mut_slice::<u64>().iter_mut().enumerate() {
            *word = at[i * 3 % 200];
        }
        let column = ColumnView::strings(&mut Context::new(env, None), words, page, None).unwrap();

        let log = store.create().unwrap();
        let spilled = write_column(&store, log, &column).unwrap();
        store.seal(log).unwrap();
        let read = read_column(&mut Context::new(env, None), &store, log, &spilled).unwrap();
        let (got, want) = (read.string_values(), column.string_values());
        // Rows shared strings, between other bytes; written out, each has its
        // own, but for the empty ones, which share the first.
        let len = |i: u64| i * 3 % 200 % 7;
        let taken = (0..600).map(|i| if len(i) == 0 { 0 } else { 4 + len(i) }).sum::<u64>();
        assert_eq!(spilled.bytes.unwrap().len, 4 + taken);
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
