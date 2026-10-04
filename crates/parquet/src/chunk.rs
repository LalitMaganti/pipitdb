//! `ChunkReader`: a column chunk's values, a page at a time, decoded into
//! columns a batch at a time.

use pipit_kernel::buffer::Buffer;
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::column::{ColumnView, DataType};
use pipit_kernel::context::Context;
use pipit_kernel::row_batch::BATCH_ROWS_MAX;

use crate::Error;
use crate::footer::{Chunk, Column, Physical};
use crate::hybrid::Hybrid;
use crate::thrift::{Cursor, Value};

/// Page headers are read in windows of this size, larger if need be.
const HEADER_BYTES: usize = 256;

pub struct ChunkReader<'s> {
    source: &'s dyn ByteSource,
    column: Column,
    data_type: DataType,
    /// Where the chunk ends.
    end: u64,
    position: Position,
    /// The body of the page last read, and where it starts in the file.
    page: Option<(u64, Buffer)>,
    /// Where the chunk's dictionary page's body is, how long it is and how
    /// many values it has, once its header is read.
    dictionary_page: Option<(u64, u64, usize)>,
    /// The dictionary's values, once a page needs them.
    dictionary: Option<Dictionary>,
}

/// Where reading is in a chunk. It's small and copied, so a lazy column can
/// keep one and read its rows from it later, with `seek`.
#[derive(Clone, Copy)]
pub struct Position {
    /// Where the next page's header starts.
    next: u64,
    /// The page being read: where its body starts and how long it is, how
    /// many rows it has left, and whether its values are indices into the
    /// dictionary.
    body: u64,
    len: u64,
    left: usize,
    indexed: bool,
    /// Within the body, once it's been read from: where an optional
    /// column's definition levels are, and how far they've been read; where
    /// the next plain value is, or where the indices start and how far
    /// they've been read.
    started: bool,
    levels: (usize, usize),
    level: Hybrid,
    value: usize,
    index: Hybrid,
}

/// A chunk's dictionary: words for fixed-width types, or offsets into bytes
/// for strings.
#[derive(Clone)]
struct Dictionary {
    values: Buffer,
    bytes: Option<Buffer>,
    count: usize,
}

/// Where a page's values come from, kept separately from the page so it can
/// be copied to read ahead.
#[derive(Clone, Copy)]
enum Values<'p> {
    /// Plain values, from `at`.
    Plain { bytes: &'p [u8], at: usize },
    /// Indices into `dictionary`.
    Indexed { indices: &'p [u8], index: Hybrid, dictionary: &'p Dictionary },
}

impl<'s> ChunkReader<'s> {
    pub fn new(
        source: &'s dyn ByteSource,
        column: Column,
        chunk: &Chunk,
    ) -> Result<ChunkReader<'s>, Error> {
        let data_type = column.data_type().ok_or(Error::Unsupported)?;
        if chunk.codec != 0 {
            return Err(Error::Unsupported);
        }
        let position = Position {
            next: chunk.start,
            body: 0,
            len: 0,
            left: 0,
            indexed: false,
            started: false,
            levels: (0, 0),
            level: Hybrid::default(),
            value: 0,
            index: Hybrid::default(),
        };
        let end = chunk.start.checked_add(chunk.len).ok_or(Error::Corrupt)?;
        let (page, dictionary_page, dictionary) = (None, None, None);
        Ok(ChunkReader {
            source,
            column,
            data_type,
            end,
            position,
            page,
            dictionary_page,
            dictionary,
        })
    }

    /// How many rows are left in the page being read, moving to the next
    /// page if that one's done: 0 once the chunk is. Only reads the next
    /// page's header, not its body.
    pub fn page_left(&mut self, context: &mut Context) -> Result<usize, Error> {
        while self.position.left == 0 && self.position.next < self.end {
            self.next_page(context)?;
        }
        Ok(self.position.left)
    }

    /// Where reading is now.
    pub fn position(&self) -> Position {
        self.position
    }

    /// Goes back, or on, to `position`, which this reader gave.
    pub fn seek(&mut self, position: Position) {
        self.position = position;
    }

    /// Passes over the next `rows` rows, which must be in the page being
    /// read, without decoding their values. Passing over the rest of a page
    /// that hasn't been read from doesn't read it at all.
    pub fn skip(&mut self, context: &mut Context, rows: usize) -> Result<(), Error> {
        check!(rows <= self.position.left);
        if rows == self.position.left && !self.position.started {
            self.position.left = 0;
            return Ok(());
        }
        let (page, dictionary) = self.body(context)?;
        let page = page.as_slice::<u8>();
        let mut valid = rows;
        if self.column.optional {
            let levels = page.get(self.position.levels.0..self.position.levels.1);
            let levels = levels.ok_or(Error::Corrupt)?;
            valid = 0;
            for _ in 0..rows {
                valid += usize::from(self.position.level.next(levels).ok_or(Error::Corrupt)? == 1);
            }
        }
        let mut values = self.values(page, dictionary.as_ref())?;
        match (&mut values, self.data_type) {
            (Values::Plain { at, .. }, DataType::Int64 | DataType::Float64) => {
                *at += valid * width(self.column.physical);
            }
            _ => {
                for _ in 0..valid {
                    values.pass(self.data_type, self.column.physical)?;
                }
            }
        }
        self.advance(values);
        self.position.left -= rows;
        Ok(())
    }

    /// The next `rows` rows, which must be in the page being read.
    pub fn read(&mut self, context: &mut Context, rows: usize) -> Result<ColumnView, Error> {
        check!(rows <= self.position.left);
        let (page, dictionary) = self.body(context)?;
        let page = page.as_slice::<u8>();
        let validity = self.validity(context, page, rows)?;
        let values = self.values(page, dictionary.as_ref())?;
        let (data_type, physical) = (self.data_type, self.column.physical);
        let bits = validity.as_ref().map(Buffer::as_slice::<u8>);
        let (values, buffer, bytes) = decode(context, data_type, physical, values, rows, bits)?;
        self.advance(values);
        self.position.left -= rows;
        Ok(match bytes {
            Some(bytes) => ColumnView::strings(buffer, bytes, validity),
            None => ColumnView::new(data_type, buffer, validity),
        })
    }

    /// Where the page's next values come from.
    fn values<'p>(
        &self,
        page: &'p [u8],
        dictionary: Option<&'p Dictionary>,
    ) -> Result<Values<'p>, Error> {
        let at = self.position.value;
        Ok(match dictionary {
            None => Values::Plain { bytes: page, at },
            Some(dictionary) => {
                let indices = page.get(at..).ok_or(Error::Corrupt)?;
                Values::Indexed { indices, index: self.position.index, dictionary }
            }
        })
    }

    /// Notes how far `values` have been read.
    fn advance(&mut self, values: Values<'_>) {
        match values {
            Values::Plain { at, .. } => self.position.value = at,
            Values::Indexed { index, .. } => self.position.index = index,
        }
    }

    /// The next `rows` rows' validity, from their definition levels, or
    /// `None` if they can't be null or none are.
    fn validity(
        &mut self,
        context: &mut Context,
        page: &[u8],
        rows: usize,
    ) -> Result<Option<Buffer>, Error> {
        if !self.column.optional {
            return Ok(None);
        }
        check!(rows <= BATCH_ROWS_MAX as usize);
        let levels = page.get(self.position.levels.0..self.position.levels.1);
        let levels = levels.ok_or(Error::Corrupt)?;
        let mut bits = [0_u8; BATCH_ROWS_MAX as usize / 8];
        let mut nulls = 0;
        for row in 0..rows {
            let valid = self.position.level.next(levels).ok_or(Error::Corrupt)? == 1;
            *at_mut!(bits, row / 8) |= u8::from(valid) << (row % 8);
            nulls += usize::from(!valid);
        }
        if nulls == 0 {
            return Ok(None);
        }
        let mut validity = context.column_buffer(rows.div_ceil(8))?;
        validity.as_mut_slice::<u8>().copy_from_slice(at!(bits, ..rows.div_ceil(8)));
        Ok(Some(validity))
    }

    /// Moves to the next page, reading only its header.
    fn next_page(&mut self, context: &mut Context) -> Result<(), Error> {
        let (header, body) = self.header(context)?;
        let (len, left) = (header.len, header.values);
        let next = body.checked_add(len).ok_or(Error::Corrupt)?;
        self.position.next = next;
        let indexed = match (header.kind, header.encoding) {
            // Data pages, version 1, of plain values or dictionary indices.
            (0, 0) => false,
            (0, 2 | 8) => true,
            // A dictionary page, of plain values, first in the chunk.
            (2, 0 | 2) if self.dictionary_page.is_none() => {
                self.dictionary_page = Some((body, len, left));
                return Ok(());
            }
            (0 | 2 | 3, _) => return Err(Error::Unsupported),
            // Index pages are skipped.
            _ => return Ok(()),
        };
        if indexed && self.dictionary_page.is_none() {
            return Err(Error::Corrupt);
        }
        self.position = Position {
            next,
            body,
            len,
            left,
            indexed,
            started: false,
            levels: (0, 0),
            level: Hybrid::default(),
            value: 0,
            index: Hybrid::default(),
        };
        Ok(())
    }

    /// The body of the page being read, read if it isn't already, and its
    /// levels and values found if they haven't been; and the dictionary, if
    /// its values are indices into it.
    fn body(&mut self, context: &mut Context) -> Result<(Buffer, Option<Dictionary>), Error> {
        let dictionary = if self.position.indexed { Some(self.dictionary(context)?) } else { None };
        let (body, len) = (self.position.body, self.position.len);
        let page = match &self.page {
            Some((at, page)) if *at == body => page.clone(),
            _ => {
                let page = self.read_bytes(context, body, len)?;
                self.page = Some((body, page.clone()));
                page
            }
        };
        if !self.position.started {
            let bytes = page.as_slice::<u8>();
            // An optional column's definition levels come first, after their
            // length.
            if self.column.optional {
                let len = length(bytes, 0)?;
                (self.position.levels, self.position.value) = ((4, 4 + len), 4 + len);
            }
            // Indices come after their width in bits.
            if self.position.indexed {
                let width = u32::from(*bytes.get(self.position.value).ok_or(Error::Corrupt)?);
                if width > 32 {
                    return Err(Error::Corrupt);
                }
                (self.position.index, self.position.value) =
                    (Hybrid::new(width), self.position.value + 1);
            }
            (self.position.level, self.position.started) = (Hybrid::new(1), true);
        }
        Ok((page, dictionary))
    }

    /// The chunk's dictionary, decoded the first time it's needed.
    fn dictionary(&mut self, context: &mut Context) -> Result<Dictionary, Error> {
        if let Some(dictionary) = &self.dictionary {
            return Ok(dictionary.clone());
        }
        let (body, len, count) = self.dictionary_page.ok_or(Error::Corrupt)?;
        let page = self.read_bytes(context, body, len)?;
        let values = Values::Plain { bytes: page.as_slice::<u8>(), at: 0 };
        let (data_type, physical) = (self.data_type, self.column.physical);
        let (_, values, bytes) = decode(context, data_type, physical, values, count, None)?;
        Ok(self.dictionary.insert(Dictionary { values, bytes, count }).clone())
    }

    /// `len` bytes of the chunk from `at`.
    fn read_bytes(&self, context: &Context, at: u64, len: u64) -> Result<Buffer, Error> {
        let len = usize::try_from(len).map_err(|_| Error::Corrupt)?;
        let mut bytes = Buffer::allocate(context.allocator(), len)?;
        self.source.read(at, bytes.as_mut_slice::<u8>())?;
        Ok(bytes)
    }

    /// The header of the page at `next`, and where its body starts.
    fn header(&self, context: &Context) -> Result<(PageHeader, u64), Error> {
        let next = self.position.next;
        let mut window = HEADER_BYTES as u64;
        loop {
            let len = (self.end - next).min(window);
            let bytes = self.read_bytes(context, next, len)?;
            let mut c = Cursor::new(bytes.as_slice::<u8>());
            if let Some(header) = page_header(&mut c) {
                return Ok((header, next + c.pos as u64));
            }
            if len < window {
                return Err(Error::Corrupt);
            }
            window *= 4;
        }
    }
}

impl<'p> Values<'p> {
    /// The next value of a fixed-width type, as a word.
    fn word(&mut self, physical: Physical) -> Result<i64, Error> {
        match self {
            Values::Plain { bytes, at } => {
                let width = width(physical);
                let value = bytes.get(*at..*at + width).ok_or(Error::Corrupt)?;
                *at += width;
                plain(physical, value).ok_or(Error::Corrupt)
            }
            Values::Indexed { indices, index, dictionary } => {
                let i = next_index(indices, index, dictionary)?;
                Ok(*at!(dictionary.values.as_slice::<i64>(), i))
            }
        }
    }

    /// The next string.
    fn string(&mut self) -> Result<&'p [u8], Error> {
        match self {
            Values::Plain { bytes, at } => {
                let len = length(bytes, *at)?;
                let value = bytes.get(*at + 4..*at + 4 + len).ok_or(Error::Corrupt)?;
                *at += 4 + len;
                Ok(value)
            }
            Values::Indexed { indices, index, dictionary } => {
                let i = next_index(indices, index, dictionary)?;
                let offsets = dictionary.values.as_slice::<u32>();
                let (start, end) = (*at!(offsets, i) as usize, *at!(offsets, i + 1) as usize);
                let Some(bytes) = &dictionary.bytes else { return Err(Error::Corrupt) };
                Ok(at!(bytes.as_slice::<u8>(), start..end))
            }
        }
    }

    /// Passes over the next value.
    fn pass(&mut self, data_type: DataType, physical: Physical) -> Result<(), Error> {
        match data_type {
            DataType::String => self.string().map(|_| ()),
            DataType::Int64 | DataType::Float64 => self.word(physical).map(|_| ()),
        }
    }
}

/// The next index into `dictionary`, checked.
fn next_index(indices: &[u8], index: &mut Hybrid, dictionary: &Dictionary) -> Result<usize, Error> {
    let i = index.next(indices).ok_or(Error::Corrupt)? as usize;
    if i >= dictionary.count {
        return Err(Error::Corrupt);
    }
    Ok(i)
}

/// Decodes `rows` rows from `values` into column buffers: words, or offsets
/// and bytes for strings. Rows that `validity` says are null have no value.
/// Returns where `values` got to as well.
fn decode<'p>(
    context: &mut Context,
    data_type: DataType,
    physical: Physical,
    mut values: Values<'p>,
    rows: usize,
    validity: Option<&[u8]>,
) -> Result<(Values<'p>, Buffer, Option<Buffer>), Error> {
    let valid = |row: usize| validity.is_none_or(|bits| *at!(bits, row / 8) >> (row % 8) & 1 != 0);
    match data_type {
        DataType::String => {
            // Walked once to size the bytes, then again to copy them.
            let (mut sizing, mut total) = (values, 0);
            for _ in (0..rows).filter(|&row| valid(row)) {
                total += sizing.string()?.len();
            }
            let mut offsets = context.column_buffer((rows + 1) * 4)?;
            let mut bytes = context.column_buffer(total)?;
            let (ends, out) = (offsets.as_mut_slice::<u32>(), bytes.as_mut_slice::<u8>());
            let mut to = 0;
            *at_mut!(ends, 0) = 0;
            for row in 0..rows {
                if valid(row) {
                    let value = values.string()?;
                    at_mut!(out, to..to + value.len()).copy_from_slice(value);
                    to += value.len();
                }
                *at_mut!(ends, row + 1) = u32::try_from(to).map_err(|_| Error::Unsupported)?;
            }
            Ok((values, offsets, Some(bytes)))
        }
        DataType::Int64 | DataType::Float64 => {
            let mut out = context.column_buffer(rows * 8)?;
            for (row, word) in out.as_mut_slice::<i64>().iter_mut().enumerate() {
                *word = if valid(row) { values.word(physical)? } else { 0 };
            }
            Ok((values, out, None))
        }
    }
}

/// How many bytes a plain fixed-width value takes.
fn width(physical: Physical) -> usize {
    if matches!(physical, Physical::Int32 | Physical::Float) { 4 } else { 8 }
}

/// A plain value of `physical` from its bytes, as a word: an integer, or a
/// float's bits.
fn plain(physical: Physical, bytes: &[u8]) -> Option<i64> {
    Some(match physical {
        Physical::Int32 => i64::from(i32::from_le_bytes(bytes.try_into().ok()?)),
        Physical::Float => {
            f64::from(f32::from_le_bytes(bytes.try_into().ok()?)).to_bits().cast_signed()
        }
        _ => i64::from_le_bytes(bytes.try_into().ok()?),
    })
}

struct PageHeader {
    kind: i64,
    /// How many bytes the page's body takes, as stored.
    len: u64,
    values: usize,
    encoding: i64,
}

/// A `PageHeader`: its type, stored size, and its data or dictionary page
/// header's value count and encoding.
fn page_header(c: &mut Cursor<'_>) -> Option<PageHeader> {
    let mut header = [Value::Missing; 4];
    c.fields(&[1, 3, 5, 7], &mut header)?;
    let mut data = [Value::Missing; 2];
    if let Some(at) = header[2].at().or(header[3].at()) {
        c.at(at).fields(&[1, 2], &mut data)?;
    }
    Some(PageHeader {
        kind: header[0].int()?,
        len: u64::try_from(header[1].int()?).ok()?,
        values: usize::try_from(data[0].int().unwrap_or(0)).ok()?,
        encoding: data[1].int().unwrap_or(0),
    })
}

/// The little-endian `u32` length at `at`.
fn length(bytes: &[u8], at: usize) -> Result<usize, Error> {
    let len = bytes.get(at..at + 4).ok_or(Error::Corrupt)?;
    Ok(u32::from_le_bytes([len[0], len[1], len[2], len[3]]) as usize)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::string::String;
    use std::vec::Vec;

    use pipit_kernel::allocator::Heap;
    use pipit_kernel::bytes::ReadError;

    use super::*;
    use crate::footer::ParquetFile;

    /// 5000 rows: `id` is the row, `half` half of it, `name` "row" and it,
    /// in row groups of 2048, as DuckDB writes them.
    const SMALL: &[u8] = include_bytes!("../tests/data/small.parquet");

    /// As `small`, but `id` is null every 7th row from the 3rd, `quarter` a
    /// quarter of the row, null every 5th from the 1st, and `name` "name"
    /// and the row, null every 11th from the 2nd.
    const NULLS: &[u8] = include_bytes!("../tests/data/nulls.parquet");

    /// As `nulls`, but with few distinct values, so DuckDB stores them in a
    /// dictionary: `id` is the row mod 10, `quarter` a quarter of the row mod
    /// 6, and `name` "name" and the row mod 13.
    const DICT: &[u8] = include_bytes!("../tests/data/dict.parquet");

    #[derive(PartialEq, Debug)]
    enum Cell {
        Int(i64),
        Float(f64),
        Text(String),
        Null,
    }

    /// Row `i` of column `c` of `nulls`.
    #[expect(clippy::cast_precision_loss, reason = "small test values")]
    fn nulls(c: usize, i: i64) -> Cell {
        match c {
            0 if i % 7 != 3 => Cell::Int(i),
            1 if i % 5 != 1 => Cell::Float(i as f64 * 0.25),
            2 if i % 11 != 2 => Cell::Text(std::format!("name{i}")),
            _ => Cell::Null,
        }
    }

    /// Row `i` of column `c` of `dict`.
    #[expect(clippy::cast_precision_loss, reason = "small test values")]
    fn dict(c: usize, i: i64) -> Cell {
        match c {
            0 if i % 7 != 3 => Cell::Int(i % 10),
            1 if i % 5 != 1 => Cell::Float((i % 6) as f64 * 0.25),
            2 if i % 11 != 2 => Cell::Text(std::format!("name{}", i % 13)),
            _ => Cell::Null,
        }
    }

    fn cells(view: &ColumnView) -> Vec<Cell> {
        (0..view.row_count())
            .map(|row| {
                let r = row as usize;
                match view.data_type() {
                    _ if view.is_null(row) => Cell::Null,
                    DataType::Int64 => Cell::Int(view.int64s()[r]),
                    DataType::Float64 => Cell::Float(view.float64s()[r]),
                    DataType::String => {
                        Cell::Text(String::from_utf8(view.string_values().get(r).to_vec()).unwrap())
                    }
                }
            })
            .collect()
    }

    /// Every row of column `c` of `bytes`, read in batches of at most 1000.
    fn read_all(mut bytes: &[u8], c: usize) -> Vec<Cell> {
        let source: &dyn ByteSource = &mut bytes;
        let file = ParquetFile::open(&Heap, source).unwrap();
        let mut context = Context::new(&Heap);
        let column = file.columns()[c];
        let mut all = Vec::new();
        for group in 0..file.row_groups() {
            let mut reader = ChunkReader::new(source, column, file.chunk(group, c)).unwrap();
            loop {
                let rows = reader.page_left(&mut context).unwrap().min(1000);
                if rows == 0 {
                    break;
                }
                all.extend(cells(&reader.read(&mut context, rows).unwrap()));
            }
        }
        all
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under Miri")]
    #[expect(clippy::cast_precision_loss, reason = "small test values")]
    fn reads_plain_values() {
        let rows = 0..5000_i64;
        let ids: Vec<_> = rows.clone().map(Cell::Int).collect();
        let halves: Vec<_> = rows.clone().map(|i| Cell::Float(i as f64 * 0.5)).collect();
        let names: Vec<_> = rows.map(|i| Cell::Text(std::format!("row{i}"))).collect();
        assert_eq!(read_all(SMALL, 0), ids);
        assert_eq!(read_all(SMALL, 1), halves);
        assert_eq!(read_all(SMALL, 2), names);
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under Miri")]
    fn reads_nulls() {
        for c in 0..3 {
            let expected: Vec<_> = (0..5000).map(|i| nulls(c, i)).collect();
            assert_eq!(read_all(NULLS, c), expected);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under Miri")]
    fn reads_dictionaries() {
        for c in 0..3 {
            let expected: Vec<_> = (0..5000).map(|i| dict(c, i)).collect();
            assert_eq!(read_all(DICT, c), expected);
        }
    }

    #[test]
    fn skips_and_seeks() {
        skips_and_seeks_in(NULLS, nulls);
        skips_and_seeks_in(DICT, dict);
    }

    /// Skips and reads by turns through each column's first chunk of
    /// `bytes`, whose row `i` of column `c` is `expected(c, i)`.
    fn skips_and_seeks_in(mut bytes: &[u8], expected: fn(usize, i64) -> Cell) {
        let source: &dyn ByteSource = &mut bytes;
        let file = ParquetFile::open(&Heap, source).unwrap();
        let mut context = Context::new(&Heap);
        for c in 0..3 {
            let mut reader = ChunkReader::new(source, file.columns()[c], file.chunk(0, c)).unwrap();
            let mut row = 0;
            loop {
                let left = reader.page_left(&mut context).unwrap();
                if left == 0 {
                    break;
                }
                let skip = left.min(300);
                reader.skip(&mut context, skip).unwrap();
                let rows = (left - skip).min(50);
                let start = reader.position();
                let read = cells(&reader.read(&mut context, rows).unwrap());
                let from = i64::try_from(row + skip).unwrap();
                let expected: Vec<_> =
                    (from..from + 50).take(rows).map(|i| expected(c, i)).collect();
                assert_eq!(read, expected);
                reader.seek(start);
                assert_eq!(cells(&reader.read(&mut context, rows).unwrap()), expected);
                row += skip + rows;
            }
            assert_eq!(row, 2048);
        }
    }

    /// Counts the bytes read from it.
    struct Counting<'a>(&'a [u8], core::cell::Cell<u64>);

    impl ByteSource for Counting<'_> {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), ReadError> {
            self.1.set(self.1.get() + into.len() as u64);
            self.0.read(offset, into)
        }
    }

    #[test]
    fn skipping_pages_reads_only_headers() {
        let source = Counting(SMALL, core::cell::Cell::new(0));
        let file = ParquetFile::open(&Heap, &source).unwrap();
        let mut context = Context::new(&Heap);
        let chunk = file.chunk(0, 0);
        let mut reader = ChunkReader::new(&source, file.columns()[0], chunk).unwrap();
        source.1.set(0);
        loop {
            let left = reader.page_left(&mut context).unwrap();
            if left == 0 {
                break;
            }
            reader.skip(&mut context, left).unwrap();
        }
        assert!(source.1.get() * 10 < chunk.len);
    }

    #[test]
    fn decodes_the_dictionary_only_when_values_are_read() {
        let mut bytes = DICT;
        let source: &dyn ByteSource = &mut bytes;
        let file = ParquetFile::open(&Heap, source).unwrap();
        let mut context = Context::new(&Heap);
        let mut reader = ChunkReader::new(source, file.columns()[2], file.chunk(0, 2)).unwrap();
        let left = reader.page_left(&mut context).unwrap();
        reader.skip(&mut context, left).unwrap();
        assert!(reader.dictionary.is_none());
        let mut reader = ChunkReader::new(source, file.columns()[2], file.chunk(1, 2)).unwrap();
        reader.page_left(&mut context).unwrap();
        assert_eq!(cells(&reader.read(&mut context, 1).unwrap()), [dict(2, 2048)]);
        assert!(reader.dictionary.is_some());
    }
}
