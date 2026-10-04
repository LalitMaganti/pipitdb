//! `ChunkReader`: a column chunk's values, a page at a time, decoded into
//! columns a batch at a time.

use pipit_kernel::allocator::Allocator;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::column::{ColumnView, DataType};

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
    /// Where the next page starts, and where the chunk ends.
    next: u64,
    end: u64,
    /// The page being read, and how many of its values are left.
    page: Option<Buffer>,
    left: usize,
    /// Where an optional column's definition levels are in the page, and
    /// how far they've been read.
    levels: (usize, usize),
    level: Hybrid,
    /// Where the next value is in the page.
    value: usize,
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
        Ok(ChunkReader {
            source,
            column,
            data_type,
            next: chunk.start,
            end: chunk.start.checked_add(chunk.len).ok_or(Error::Corrupt)?,
            page: None,
            left: 0,
            levels: (0, 0),
            level: Hybrid::default(),
            value: 0,
        })
    }

    /// How many rows are left in the page being read, reading the next page
    /// if that one's done: 0 once the chunk is.
    pub fn page_left(&mut self, allocator: &dyn Allocator) -> Result<usize, Error> {
        if self.left == 0 && self.next < self.end {
            self.read_page(allocator)?;
        }
        Ok(self.left)
    }

    /// The next `rows` rows, which must be in the page being read.
    pub fn read(&mut self, allocator: &dyn Allocator, rows: usize) -> Result<ColumnView, Error> {
        check!(rows <= self.left);
        let Some(page) = self.page.clone() else { return Err(Error::Corrupt) };
        let page = page.as_slice::<u8>();
        let validity = self.validity(allocator, page, rows)?;
        let valid = |row: usize| {
            validity
                .as_ref()
                .is_none_or(|bits| *at!(bits.as_slice::<u8>(), row / 8) >> (row % 8) & 1 != 0)
        };
        let values = page.get(self.value..).ok_or(Error::Corrupt)?;
        let (column, used) = match self.data_type {
            DataType::String => {
                // Lengths and bytes alternate: walked once to size the bytes,
                // then copied.
                let (mut at, mut total) = (0, 0);
                for _ in (0..rows).filter(|&row| valid(row)) {
                    let len = length(values, at)?;
                    total += len;
                    at += 4 + len;
                }
                let mut offsets = Buffer::allocate(allocator, (rows + 1) * 4)?;
                let mut bytes = Buffer::allocate(allocator, total)?;
                let (ends, out) = (offsets.as_mut_slice::<u32>(), bytes.as_mut_slice::<u8>());
                let (mut from, mut to) = (0, 0);
                for row in 0..rows {
                    if valid(row) {
                        let len = length(values, from)?;
                        let value = values.get(from + 4..from + 4 + len).ok_or(Error::Corrupt)?;
                        at_mut!(out, to..to + len).copy_from_slice(value);
                        (from, to) = (from + 4 + len, to + len);
                    }
                    *at_mut!(ends, row + 1) = u32::try_from(to).map_err(|_| Error::Unsupported)?;
                }
                (ColumnView::strings(offsets, bytes, validity), from)
            }
            DataType::Int64 | DataType::Float64 => {
                let physical = self.column.physical;
                let width =
                    if matches!(physical, Physical::Int32 | Physical::Float) { 4 } else { 8 };
                let mut out = Buffer::allocate(allocator, rows * 8)?;
                let mut at = 0;
                for (row, word) in out.as_mut_slice::<i64>().iter_mut().enumerate() {
                    if valid(row) {
                        let bytes = values.get(at..at + width).ok_or(Error::Corrupt)?;
                        *word = plain(physical, bytes);
                        at += width;
                    }
                }
                (ColumnView::new(self.data_type, out, validity), at)
            }
        };
        self.value += used;
        self.left -= rows;
        Ok(column)
    }

    /// The next `rows` rows' validity, from their definition levels, or
    /// `None` if they can't be null or none are.
    fn validity(
        &mut self,
        allocator: &dyn Allocator,
        page: &[u8],
        rows: usize,
    ) -> Result<Option<Buffer>, Error> {
        if !self.column.optional {
            return Ok(None);
        }
        let levels = page.get(self.levels.0..self.levels.1).ok_or(Error::Corrupt)?;
        let mut bits = Buffer::allocate(allocator, rows.div_ceil(8))?;
        let mut nulls = 0;
        for row in 0..rows {
            let valid = self.level.next(levels).ok_or(Error::Corrupt)? == 1;
            *at_mut!(bits.as_mut_slice::<u8>(), row / 8) |= u8::from(valid) << (row % 8);
            nulls += usize::from(!valid);
        }
        Ok((nulls > 0).then_some(bits))
    }

    fn read_page(&mut self, allocator: &dyn Allocator) -> Result<(), Error> {
        let (header, body) = self.header(allocator)?;
        self.next = body.checked_add(header.len).ok_or(Error::Corrupt)?;
        match header.kind {
            // A data page, version 1, of plain values.
            0 if header.encoding == 0 => {}
            // Other data pages, and dictionary pages.
            0 | 2 | 3 => return Err(Error::Unsupported),
            // Index pages are skipped.
            _ => return self.page_left(allocator).map(|_| ()),
        }
        let len = usize::try_from(header.len).map_err(|_| Error::Corrupt)?;
        let mut page = Buffer::allocate(allocator, len)?;
        self.source.read(body, page.as_mut_slice::<u8>())?;
        // An optional column's definition levels come first, after their
        // length.
        let mut value = 0;
        if self.column.optional {
            let len = length(page.as_slice::<u8>(), 0)?;
            self.levels = (4, 4 + len);
            value = 4 + len;
        }
        (self.page, self.left, self.level, self.value) =
            (Some(page), header.values, Hybrid::new(1), value);
        Ok(())
    }

    /// The header of the page at `next`, and where its body starts.
    fn header(&self, allocator: &dyn Allocator) -> Result<(PageHeader, u64), Error> {
        let mut window = HEADER_BYTES as u64;
        loop {
            let len = (self.end - self.next).min(window);
            let size = usize::try_from(len).map_err(|_| Error::Corrupt)?;
            let mut bytes = Buffer::allocate(allocator, size)?;
            self.source.read(self.next, bytes.as_mut_slice::<u8>())?;
            let mut c = Cursor::new(bytes.as_slice::<u8>());
            if let Some(header) = page_header(&mut c) {
                return Ok((header, self.next + c.pos as u64));
            }
            if len < window {
                return Err(Error::Corrupt);
            }
            window *= 4;
        }
    }
}

/// A plain value of `physical` from its bytes, as a word: an integer, or a
/// float's bits.
fn plain(physical: Physical, bytes: &[u8]) -> i64 {
    match physical {
        Physical::Int32 => i64::from(i32::from_le_bytes(array(bytes))),
        Physical::Float => f64::from(f32::from_le_bytes(array(bytes))).to_bits().cast_signed(),
        _ => i64::from_le_bytes(array(bytes)),
    }
}

/// The first `N` of `bytes`, which has at least that many.
fn array<const N: usize>(bytes: &[u8]) -> [u8; N] {
    let mut array = [0; N];
    for (to, &from) in array.iter_mut().zip(at!(bytes, ..N)) {
        *to = from;
    }
    array
}

struct PageHeader {
    kind: i64,
    /// How many bytes the page's body takes, as stored.
    len: u64,
    values: usize,
    encoding: i64,
}

/// A `PageHeader`: its type, stored size, and its data page header's value
/// count and encoding.
fn page_header(c: &mut Cursor<'_>) -> Option<PageHeader> {
    let mut header = [Value::Missing; 3];
    c.fields(&[1, 3, 5], &mut header)?;
    let mut data = [Value::Missing; 2];
    if let Some(at) = header[2].at() {
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

    use super::*;
    use crate::footer::ParquetFile;

    /// 5000 rows: `id` is the row, `half` half of it, `name` "row" and it,
    /// in row groups of 2048, as DuckDB writes them.
    const SMALL: &[u8] = include_bytes!("../tests/data/small.parquet");

    /// As `small`, but `id` is null every 7th row from the 3rd, `quarter` a
    /// quarter of the row, null every 5th from the 1st, and `name` "name"
    /// and the row, null every 11th from the 2nd.
    const NULLS: &[u8] = include_bytes!("../tests/data/nulls.parquet");

    #[derive(PartialEq, Debug)]
    enum Cell {
        Int(i64),
        Float(f64),
        Text(String),
        Null,
    }

    /// Every row of column `c` of `bytes`, read in batches of at most 1000.
    fn read_all(mut bytes: &[u8], c: usize) -> Vec<Cell> {
        let source: &dyn ByteSource = &mut bytes;
        let file = ParquetFile::open(&Heap, source).unwrap();
        let column = file.columns()[c];
        let mut cells = Vec::new();
        for group in 0..file.row_groups() {
            let mut reader = ChunkReader::new(source, column, file.chunk(group, c)).unwrap();
            loop {
                let rows = reader.page_left(&Heap).unwrap().min(1000);
                if rows == 0 {
                    break;
                }
                let view = reader.read(&Heap, rows).unwrap();
                for row in 0..view.row_count() {
                    let r = row as usize;
                    cells.push(match view.data_type() {
                        _ if view.is_null(row) => Cell::Null,
                        DataType::Int64 => Cell::Int(view.int64s()[r]),
                        DataType::Float64 => Cell::Float(view.float64s()[r]),
                        DataType::String => Cell::Text(
                            String::from_utf8(view.string_values().get(r).to_vec()).unwrap(),
                        ),
                    });
                }
            }
        }
        cells
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
    #[expect(clippy::cast_precision_loss, reason = "small test values")]
    fn reads_nulls() {
        let rows = 0..5000_i64;
        let or_null = |null: bool, cell: Cell| if null { Cell::Null } else { cell };
        let ids: Vec<_> = rows.clone().map(|i| or_null(i % 7 == 3, Cell::Int(i))).collect();
        let quarters: Vec<_> =
            rows.clone().map(|i| or_null(i % 5 == 1, Cell::Float(i as f64 * 0.25))).collect();
        let names: Vec<_> =
            rows.map(|i| or_null(i % 11 == 2, Cell::Text(std::format!("name{i}")))).collect();
        assert_eq!(read_all(NULLS, 0), ids);
        assert_eq!(read_all(NULLS, 1), quarters);
        assert_eq!(read_all(NULLS, 2), names);
    }
}
