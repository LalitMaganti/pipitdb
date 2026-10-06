//! `ChunkReader`: a column chunk's values, a page at a time, decoded into
//! columns a batch at a time.

use pipit_kernel::buffer::Buffer;
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::column::{Bounds, ColumnView, DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::row_batch::BATCH_ROWS_MAX;

use crate::footer::{Chunk, Column};
use crate::hybrid::{Hybrid, Run};
use crate::thrift::Cursor;
use crate::{Codec, Error};
use crate::{bits, bounds, page, plain};

/// Page headers are read in windows of this size, larger if need be.
const HEADER_BYTES: usize = 256;

pub struct ChunkReader<'s> {
    /// Where the file's bytes come from.
    source: &'s dyn ByteSource,
    /// What decompresses pages.
    codecs: &'s dyn Codec,
    /// The chunk's codec, or 0 if it isn't compressed.
    codec: u8,
    /// The column the chunk is of, as the footer describes it.
    column: Column,
    /// What the column reads as.
    data_type: DataType,
    /// Where the chunk ends.
    end: u64,
    /// Where reading is.
    position: Position,
    /// The body of the page last read, and where it starts in the file.
    page: Option<(u64, Buffer)>,
    /// The chunk's dictionary page's body and how many values it has, once
    /// its header is read.
    dictionary_page: Option<(Body, usize)>,
    /// The dictionary's values, once a page needs them.
    dictionary: Option<Dictionary>,
    /// Room to decode a batch's levels or indices, made the first time it's
    /// needed.
    scratch: Option<Buffer>,
    /// Bounds of the chunk's values, if known, from their type and
    /// statistics.
    bounds: Option<Bounds>,
}

/// A batch's worth of `u32`s, for levels or indices.
const SCRATCH_BYTES: usize = BATCH_ROWS_MAX as usize * 4;

/// Where reading is in a chunk. It's small and copied, so a lazy column can
/// keep one and read its rows from it later, with `seek`.
#[derive(Clone, Copy)]
pub struct Position {
    /// Where the next page's header starts.
    next: u64,
    /// The body of the page being read.
    body: Body,
    /// How many rows the page has left.
    left: usize,
    /// Whether the page's values are indices into the dictionary.
    indexed: bool,
    /// Whether the body has been read from, which sets the fields below.
    started: bool,
    /// Where an optional column's definition levels are in the body.
    levels: (usize, usize),
    /// How far the levels have been read.
    level: Hybrid,
    /// Where the next plain value is, or where the indices start.
    value: usize,
    /// How far the indices have been read.
    index: Hybrid,
}

/// Where a page's body is.
#[derive(Clone, Copy)]
struct Body {
    /// Where it starts.
    at: u64,
    /// How many bytes it takes, as stored.
    len: u64,
    /// How many bytes it takes once decompressed.
    size: u64,
}

/// A chunk's dictionary: words for fixed-width types, or views into its page
/// for strings.
#[derive(Clone)]
struct Dictionary {
    /// The values, as a column, with their bounds if they're integers.
    values: ColumnView,
    /// Bounds of the values, if they're integers: their least and greatest.
    bounds: Option<Bounds>,
    /// For a fixed-width type, the values as words.
    words: Option<Buffer>,
    /// The values and then a null, for null rows to point at.
    nullable: ColumnView,
    /// How many values there are.
    count: usize,
}

impl<'s> ChunkReader<'s> {
    /// A reader of `chunk`, of `column`, which decompresses pages with
    /// `codecs`.
    pub fn new(
        source: &'s dyn ByteSource,
        codecs: &'s dyn Codec,
        column: Column,
        chunk: &Chunk,
    ) -> Result<ChunkReader<'s>, Error> {
        let data_type = column.data_type().ok_or(Error::Unsupported)?;
        let position = Position {
            next: chunk.start,
            body: Body { at: 0, len: 0, size: 0 },
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
            codecs,
            codec: chunk.codec,
            bounds: bounds::of_chunk(column, chunk),
            column,
            data_type,
            end,
            position,
            page,
            dictionary_page,
            dictionary,
            scratch: None,
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
        let (page, _) = self.body(context)?;
        let page = page.as_slice::<u8>();
        let mut valid = rows;
        if self.column.optional {
            let levels = self.levels(page)?;
            let mut scratch = self.take_scratch(context)?;
            let decoded = scratch.as_mut_slice::<u32>();
            valid = 0;
            for start in (0..rows).step_by(decoded.len()) {
                let chunk = at_mut!(decoded, ..(rows - start).min(BATCH_ROWS_MAX as usize));
                self.position.level.take(levels, chunk).ok_or(Error::Corrupt)?;
                valid += chunk.iter().filter(|&&level| level == 1).count();
            }
            self.scratch = Some(scratch);
        }
        let at = self.position.value;
        match (self.position.indexed, self.data_type) {
            (true, _) => {
                let indices = page.get(at..).ok_or(Error::Corrupt)?;
                self.position.index.skip(indices, valid).ok_or(Error::Corrupt)?;
            }
            (false, DataType::String) => {
                let mut at = at;
                for _ in 0..valid {
                    at += 4 + plain::length(page, at)?;
                }
                self.position.value = at;
            }
            (false, _) => self.position.value += valid * plain::width(self.column.physical),
        }
        self.position.left -= rows;
        Ok(())
    }

    /// The next `rows` rows, which must be in the page being read and are at
    /// most `BATCH_ROWS_MAX`: a column in one of `forms`.
    pub fn read(
        &mut self,
        context: &mut Context,
        rows: usize,
        forms: Forms,
    ) -> Result<ColumnView, Error> {
        check!(rows <= self.position.left && rows <= BATCH_ROWS_MAX as usize);
        let (body, dictionary) = self.body(context)?;
        let page = body.as_slice::<u8>();
        let mut scratch = self.take_scratch(context)?;
        let decoded = scratch.as_mut_slice::<u32>();
        let (validity, valid) = self.validity(context, page, rows, decoded)?;
        if let Some(dictionary) = &dictionary {
            let mut column =
                self.indexed(context, page, dictionary, rows, valid, validity, decoded)?;
            // Strings' dictionaries are kept, and a run of one entry is a
            // constant, unless `forms` rules them out.
            column.make_in(context, forms)?;
            self.scratch = Some(scratch);
            self.position.left -= rows;
            return Ok(column);
        }
        let bits = validity.as_ref().map(Buffer::as_slice::<u8>);
        let column = match self.data_type {
            DataType::String => {
                // Views into the page, so no string is copied: the column
                // holds the page.
                let mut views = context.values_buffer(rows * 8)?;
                let at = self.position.value;
                self.position.value = plain::views(page, at, views.as_mut_slice::<u32>(), valid)?;
                if let Some(bits) = bits {
                    bits::spread(views.as_mut_slice::<i64>(), valid, bits, 0);
                }
                ColumnView::strings(context, views, body.clone(), validity)?
            }
            DataType::Int64 | DataType::Float64 => {
                let mut out = context.values_buffer(rows * 8)?;
                let words = out.as_mut_slice::<i64>();
                let (column, at) = (self.column, self.position.value);
                self.position.value = plain::words(column, page, at, at_mut!(words, ..valid))?;
                if let Some(bits) = bits {
                    bits::spread(words, valid, bits, 0);
                }
                bounds::bounded(
                    ColumnView::new(context, self.data_type, out, validity)?,
                    self.bounds,
                )
            }
        };
        self.scratch = Some(scratch);
        self.position.left -= rows;
        Ok(column)
    }

    /// The scratch memory, to give back once used.
    fn take_scratch(&mut self, context: &Context) -> Result<Buffer, Error> {
        match self.scratch.take() {
            Some(scratch) => Ok(scratch),
            None => Ok(Buffer::allocate(context.allocator(), SCRATCH_BYTES)?),
        }
    }

    /// The page's definition levels.
    fn levels<'p>(&self, page: &'p [u8]) -> Result<&'p [u8], Error> {
        page.get(self.position.levels.0..self.position.levels.1).ok_or(Error::Corrupt)
    }

    /// The next `rows` rows of a page of indices into `dictionary`, `valid`
    /// of them not null as `validity` says: a constant, if every row has one
    /// value; else, for strings, a dictionary column, which copies none; else
    /// flat values, looked up once here so operators read them plainly, as
    /// DuckDB's reader does. Indices are decoded into `decoded`, a run at a
    /// time.
    #[expect(clippy::too_many_arguments, reason = "a step of `read`")]
    fn indexed(
        &mut self,
        context: &mut Context,
        page: &[u8],
        dictionary: &Dictionary,
        rows: usize,
        valid: usize,
        validity: Option<Buffer>,
        decoded: &mut [u32],
    ) -> Result<ColumnView, Error> {
        let bytes = page.get(self.position.value..).ok_or(Error::Corrupt)?;
        let rows32 = u32::try_from(rows).map_err(|_| Error::Corrupt)?;
        let count = dictionary.count;
        if valid == 0 {
            let null = dictionary.nullable.slice(dictionary.nullable.row_count() - 1, 1);
            return Ok(ColumnView::constant(context, &null, rows32)?);
        }
        let mut run = self.position.index.next_run(bytes, at_mut!(decoded, ..valid));
        // One value for every row.
        if let Some(Run::Repeat(i, n)) = run
            && n == rows
            && (i as usize) < count
        {
            return Ok(ColumnView::constant(context, &dictionary.values.slice(i, 1), rows32)?);
        }
        let bits = validity.as_ref().map(Buffer::as_slice::<u8>);
        let null = u32::try_from(count).map_err(|_| Error::Unsupported)?;
        let mut at = 0;
        if let Some(words) = &dictionary.words {
            let entries = at!(words.as_slice::<i64>(), ..count);
            let entry = |i: u32| entries.get(i as usize).copied().ok_or(Error::Corrupt);
            let mut out = context.values_buffer(rows * 8)?;
            let values = out.as_mut_slice::<i64>();
            loop {
                match run.ok_or(Error::Corrupt)? {
                    Run::Repeat(i, n) => at_mut!(values, at..at + n).fill(entry(i)?),
                    Run::Packed(n) => {
                        let indices = at!(decoded, ..n);
                        // Checked together, so each lookup needn't be.
                        if indices.iter().fold(0, |max, &i| max.max(i)) as usize >= count {
                            return Err(Error::Corrupt);
                        }
                        for (value, &i) in at_mut!(values, at..at + n).iter_mut().zip(indices) {
                            // SAFETY: every index is below `count`, the number
                            // of `entries`, checked just above.
                            *value = unsafe { *entries.get_unchecked(i as usize) };
                        }
                    }
                }
                at += run.map_or(0, Run::len);
                if at == valid {
                    break;
                }
                run = self.position.index.next_run(bytes, at_mut!(decoded, ..valid - at));
            }
            if let Some(bits) = bits {
                bits::spread(values, valid, bits, 0);
            }
            let column = ColumnView::new(context, self.data_type, out, validity)?;
            return Ok(bounds::bounded(column, dictionary.bounds));
        }
        let mut indices = context.indices_buffer(rows * 4)?;
        let out = indices.as_mut_slice::<u32>();
        loop {
            match run.ok_or(Error::Corrupt)? {
                Run::Repeat(i, n) => at_mut!(out, at..at + n).fill(i),
                Run::Packed(n) => at_mut!(out, at..at + n).copy_from_slice(at!(decoded, ..n)),
            }
            at += run.map_or(0, Run::len);
            if at == valid {
                break;
            }
            run = self.position.index.next_run(bytes, at_mut!(decoded, ..valid - at));
        }
        if at!(out, ..valid).iter().any(|&i| i >= null) {
            return Err(Error::Corrupt);
        }
        Ok(match bits {
            None => ColumnView::dictionary(context, &dictionary.values, indices)?,
            Some(bits) => {
                bits::spread(out, valid, bits, null);
                ColumnView::dictionary(context, &dictionary.nullable, indices)?
            }
        })
    }

    /// The next `rows` rows' validity, from their definition levels, or
    /// `None` if they can't be null or none are; and how many aren't null.
    /// Levels are decoded into `decoded`.
    fn validity(
        &mut self,
        context: &mut Context,
        page: &[u8],
        rows: usize,
        decoded: &mut [u32],
    ) -> Result<(Option<Buffer>, usize), Error> {
        if !self.column.optional {
            return Ok((None, rows));
        }
        let levels = self.levels(page)?;
        let mut bits = [0_u8; BATCH_ROWS_MAX as usize / 8];
        let (mut row, mut valid) = (0, 0);
        while row < rows {
            let left = at_mut!(decoded, ..rows - row);
            match self.position.level.next_run(levels, left).ok_or(Error::Corrupt)? {
                Run::Repeat(level, n) => {
                    if level == 1 {
                        bits::set(&mut bits, row, n);
                        valid += n;
                    }
                    row += n;
                }
                Run::Packed(n) => {
                    for &level in at!(decoded, ..n) {
                        *at_mut!(bits, row / 8) |= u8::from(level == 1) << (row % 8);
                        (row, valid) = (row + 1, valid + usize::from(level == 1));
                    }
                }
            }
        }
        if valid == rows {
            return Ok((None, rows));
        }
        let mut validity = context.small_buffer(rows.div_ceil(8))?;
        validity.as_mut_slice::<u8>().copy_from_slice(at!(bits, ..rows.div_ceil(8)));
        Ok((Some(validity), valid))
    }

    /// Moves to the next page, reading only its header.
    fn next_page(&mut self, context: &mut Context) -> Result<(), Error> {
        let (header, at) = self.header(context)?;
        let (len, left) = (header.len, header.values);
        let body = Body { at, len, size: header.size };
        let next = at.checked_add(len).ok_or(Error::Corrupt)?;
        self.position.next = next;
        // Levels encoded otherwise than as the hybrid, as very old writers
        // did, would be misread.
        if header.kind == 0 && self.column.optional && header.levels != page::RLE {
            return Err(Error::Unsupported);
        }
        let indexed = match (header.kind, header.encoding) {
            // Data pages, version 1, of plain values or dictionary indices.
            (0, 0) => false,
            (0, 2 | 8) => true,
            // A dictionary page, of plain values, first in the chunk.
            (2, 0 | 2) if self.dictionary_page.is_none() => {
                self.dictionary_page = Some((body, left));
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
        let body = self.position.body;
        let page = match &self.page {
            Some((at, page)) if *at == body.at => page.clone(),
            _ => {
                let page = self.read_body(context, body)?;
                self.page = Some((body.at, page.clone()));
                page
            }
        };
        if !self.position.started {
            let bytes = page.as_slice::<u8>();
            // An optional column's definition levels come first, after their
            // length.
            if self.column.optional {
                let len = plain::length(bytes, 0)?;
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
        let (body, count) = self.dictionary_page.ok_or(Error::Corrupt)?;
        let body = self.read_body(context, body)?;
        let page = body.as_slice::<u8>();
        let count32 = u32::try_from(count).map_err(|_| Error::Unsupported)?;
        // The values, then a null.
        let mut nulls = context.bytes_buffer((count + 1).div_ceil(8))?;
        let bits = nulls.as_mut_slice::<u8>();
        bits.fill(0xff);
        *at_mut!(bits, count / 8) &= !(1 << (count % 8));
        let (values, nullable, words, mut bounds) = match self.data_type {
            DataType::String => {
                // Views into the page, then an empty string, for nulls.
                let mut views = context.bytes_buffer((count + 1) * 8)?;
                plain::views(page, 0, views.as_mut_slice::<u32>(), count)?;
                *at_mut!(views.as_mut_slice::<i64>(), count) = 0;
                let values = ColumnView::strings(context, views.clone(), body.clone(), None)?;
                (values, ColumnView::strings(context, views, body, Some(nulls))?, None, None)
            }
            DataType::Int64 | DataType::Float64 => {
                let mut values = context.bytes_buffer((count + 1) * 8)?;
                let words = values.as_mut_slice::<i64>();
                plain::words(self.column, page, 0, at_mut!(words, ..count))?;
                *at_mut!(words, count) = 0;
                // The values' own least and greatest are their tightest bounds.
                let entries = at!(words, ..count);
                let bounds = (self.data_type == DataType::Int64 && count > 0).then(|| Bounds {
                    min: entries.iter().fold(i64::MAX, |min, &w| min.min(w)),
                    max: entries.iter().fold(i64::MIN, |max, &w| max.max(w)),
                });
                let all = ColumnView::new(context, self.data_type, values.clone(), None)?;
                let nullable =
                    ColumnView::new(context, self.data_type, values.clone(), Some(nulls))?;
                (all, nullable, Some(values), bounds)
            }
        };
        let values = bounds::bounded(values.slice(0, count32), bounds);
        let nullable = bounds::bounded(nullable, bounds);
        bounds = bounds.or(self.bounds);
        let dictionary = Dictionary { values, bounds, words, nullable, count };
        Ok(self.dictionary.insert(dictionary).clone())
    }

    /// A page's body, decompressed if it's stored compressed.
    fn read_body(&self, context: &Context, body: Body) -> Result<Buffer, Error> {
        let stored = self.read_bytes(context, body.at, body.len)?;
        if self.codec == 0 {
            return Ok(stored);
        }
        let size = usize::try_from(body.size).map_err(|_| Error::Corrupt)?;
        // SAFETY: decompressing writes every byte, or fails.
        let mut page = unsafe { Buffer::allocate_uninit(context.allocator(), size)? };
        self.codecs.decompress(self.codec, stored.as_slice(), page.as_mut_slice())?;
        Ok(page)
    }

    /// `len` bytes of the chunk from `at`.
    fn read_bytes(&self, context: &Context, at: u64, len: u64) -> Result<Buffer, Error> {
        let len = usize::try_from(len).map_err(|_| Error::Corrupt)?;
        let mut bytes = Buffer::allocate(context.allocator(), len)?;
        self.source.read(at, bytes.as_mut_slice::<u8>())?;
        Ok(bytes)
    }

    /// The header of the page at `next`, and where its body starts.
    fn header(&self, context: &Context) -> Result<(page::Header, u64), Error> {
        let next = self.position.next;
        let mut window = HEADER_BYTES as u64;
        loop {
            let len = (self.end - next).min(window);
            let bytes = self.read_bytes(context, next, len)?;
            let mut c = Cursor::new(bytes.as_slice::<u8>());
            if let Some(header) = page::header(&mut c) {
                return Ok((header, next + c.pos as u64));
            }
            if len < window {
                return Err(Error::Corrupt);
            }
            window *= 4;
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::string::String;
    use std::vec::Vec;

    use pipit_kernel::allocator::Heap;
    use pipit_kernel::bytes::ReadError;
    use pipit_kernel::column::Form;

    use super::*;
    use crate::Uncompressed;

    /// Every form a reader can read in.
    const ANY: Forms = Forms::FLAT.union(Forms::CONSTANT).union(Forms::DICTIONARY);
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
        let mut view = view.clone();
        view.make_in(&mut Context::new(&Heap), Forms::FLAT).unwrap();
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
            let mut reader =
                ChunkReader::new(source, &Uncompressed, column, file.chunk(group, c)).unwrap();
            loop {
                let rows = reader.page_left(&mut context).unwrap().min(1000);
                if rows == 0 {
                    break;
                }
                all.extend(cells(&reader.read(&mut context, rows, ANY).unwrap()));
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
    fn reads_dictionaries_of_strings_as_dictionary_columns() {
        let mut bytes = DICT;
        let source: &dyn ByteSource = &mut bytes;
        let file = ParquetFile::open(&Heap, source).unwrap();
        let mut context = Context::new(&Heap);
        for c in 0..3 {
            let column = file.columns()[c];
            let mut reader =
                ChunkReader::new(source, &Uncompressed, column, file.chunk(0, c)).unwrap();
            reader.page_left(&mut context).unwrap();
            let view = reader.read(&mut context, 100, ANY).unwrap();
            if c == 2 {
                // Strings: none copied out of the dictionary.
                assert!(matches!(view.form(), Form::Dictionary(_)));
                assert!(view.values().len() <= 14);
            } else {
                assert!(matches!(view.form(), Form::Flat));
            }
            // An integer dictionary's own least and greatest bound its rows.
            if c == 0 {
                assert_eq!(view.bounds().map(|b| (b.min, b.max)), Some((0, 9)));
            }
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
            let mut reader =
                ChunkReader::new(source, &Uncompressed, file.columns()[c], file.chunk(0, c))
                    .unwrap();
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
                let read = cells(&reader.read(&mut context, rows, ANY).unwrap());
                let from = i64::try_from(row + skip).unwrap();
                let expected: Vec<_> =
                    (from..from + 50).take(rows).map(|i| expected(c, i)).collect();
                assert_eq!(read, expected);
                reader.seek(start);
                assert_eq!(cells(&reader.read(&mut context, rows, ANY).unwrap()), expected);
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
        let mut reader =
            ChunkReader::new(&source, &Uncompressed, file.columns()[0], chunk).unwrap();
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
        let mut reader =
            ChunkReader::new(source, &Uncompressed, file.columns()[2], file.chunk(0, 2)).unwrap();
        let left = reader.page_left(&mut context).unwrap();
        reader.skip(&mut context, left).unwrap();
        assert!(reader.dictionary.is_none());
        let mut reader =
            ChunkReader::new(source, &Uncompressed, file.columns()[2], file.chunk(1, 2)).unwrap();
        reader.page_left(&mut context).unwrap();
        assert_eq!(cells(&reader.read(&mut context, 1, ANY).unwrap()), [dict(2, 2048)]);
        assert!(reader.dictionary.is_some());
    }
}
