//! `ChunkReader`: a column chunk's values, a page at a time, decoded into
//! columns a batch at a time.

use pipit_kernel::bitmap::BitSlice;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::column::{Bounds, ColumnView, DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::env::MemoryKind;
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
    /// Where the body of the page last read starts in the file, and its
    /// length.
    page: Option<(u64, usize)>,
    /// The buffer it's in, which can be longer: reused for the next page,
    /// and the next chunk's, unless a column views it.
    body: Option<Buffer>,
    /// The buffer a compressed page's bytes were last read into, reused
    /// likewise.
    stored: Option<Buffer>,
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
    /// Where the page being read's header starts.
    header: u64,
    /// How many rows the page being read has.
    rows: usize,
    /// The body of the page being read.
    body: Body,
    /// How many rows the page has left.
    left: usize,
    /// Whether the page's values are indices into the dictionary.
    indexed: bool,
    /// Whether the body has been read from, which sets the fields below.
    started: bool,
    /// How many rows were skipped before it was, to pass over once it is.
    deferred: usize,
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
    #[inline(never)]
    pub fn new(
        source: &'s dyn ByteSource,
        codecs: &'s dyn Codec,
        column: Column,
        chunk: &Chunk,
    ) -> Result<ChunkReader<'s>, Error> {
        let data_type = column.data_type().ok_or(Error::Unsupported)?;
        let position = Position {
            next: chunk.start,
            header: chunk.start,
            rows: 0,
            body: Body { at: 0, len: 0, size: 0 },
            left: 0,
            indexed: false,
            started: false,
            deferred: 0,
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
            body: None,
            stored: None,
            scratch: None,
        })
    }

    /// As `new`, keeping this reader's buffers for the new chunk's pages.
    pub fn restart(
        &mut self,
        source: &'s dyn ByteSource,
        codecs: &'s dyn Codec,
        column: Column,
        chunk: &Chunk,
    ) -> Result<(), Error> {
        let old = core::mem::replace(self, ChunkReader::new(source, codecs, column, chunk)?);
        (self.body, self.stored, self.scratch) = (old.body, old.stored, old.scratch);
        Ok(())
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

    /// What the column reads as.
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Where reading is now.
    pub fn position(&self) -> Position {
        self.position
    }

    /// Goes back, or on, to `position`, which this reader gave.
    pub fn seek(&mut self, position: Position) {
        self.position = position;
    }

    /// Where reading is, to come back to with `go_to`: where the page being
    /// read's header starts, and how many of its rows are read or passed.
    pub fn place(&self) -> (u64, usize) {
        (self.position.header, self.position.rows - self.position.left)
    }

    /// Goes to row `row` of the page whose header starts at `header`, which
    /// `place` gave, on from where reading is if that's in the same page and
    /// not past it. Reads the page's header again otherwise.
    pub fn go_to(&mut self, context: &mut Context, header: u64, row: usize) -> Result<(), Error> {
        let (at, read) = self.place();
        if at != header || read > row {
            if header >= self.end {
                return Err(Error::Corrupt);
            }
            (self.position.next, self.position.left) = (header, 0);
            self.next_page(context)?;
            if self.position.header != header || self.position.left == 0 {
                return Err(Error::Corrupt);
            }
        }
        let (_, read) = self.place();
        if row - read > self.position.left {
            return Err(Error::Corrupt);
        }
        self.skip(context, row - read)
    }

    /// Passes over the next `rows` rows of the page being read, which
    /// hasn't been read from, only counting them: it can't be read from
    /// after, but can be gone back to.
    pub fn pass(&mut self, rows: usize) {
        check!(rows <= self.position.left && !self.position.started);
        self.position.left -= rows;
    }

    /// Passes over the next `rows` rows, which must be in the page being
    /// read, without decoding their values. A page that hasn't been read from
    /// isn't read until one of its rows is, so skipping all its rows doesn't
    /// read it at all.
    pub fn skip(&mut self, context: &mut Context, rows: usize) -> Result<(), Error> {
        check!(rows <= self.position.left);
        self.position.left -= rows;
        if !self.position.started {
            self.position.deferred += rows;
            return Ok(());
        }
        self.body(context)?;
        let (Some((_, len)), Some(body)) = (self.page, &self.body) else { pipit_kernel::failed!() };
        let page = at!(body.as_slice::<u8>(), ..len);
        let (column, data_type) = (self.column, self.data_type);
        let position = &mut self.position;
        position.advance(context, column, data_type, &mut self.scratch, page, rows)
    }

    /// The next `rows` rows, as `read` gives them, but with values only for
    /// the rows `kept` lists, in increasing order: the others hold any value.
    /// The indices or values of the rows between are passed over, not
    /// decoded. Strings are read whole.
    pub fn read_kept(
        &mut self,
        context: &mut Context,
        rows: usize,
        kept: &[u16],
        forms: Forms,
    ) -> Result<ColumnView, Error> {
        check!(rows <= self.position.left && rows <= BATCH_ROWS_MAX as usize);
        check!(kept.windows(2).all(|pair| pair[0] < pair[1]));
        check!(kept.last().is_none_or(|&row| usize::from(row) < rows));
        if self.data_type == DataType::String {
            return self.read(context, rows, forms);
        }
        self.body(context)?;
        let (Some((_, len)), Some(body)) = (self.page, &self.body) else { pipit_kernel::failed!() };
        let page = at!(body.as_slice::<u8>(), ..len);
        let position = &mut self.position;
        let decoded = scratch(context, &mut self.scratch)?.as_mut_slice();
        let (validity, valid) =
            position.validity(context, self.column.optional, page, rows, decoded)?;
        let bits = validity.as_ref().map(Buffer::as_slice::<u8>);
        let dictionary = if position.indexed { self.dictionary.as_ref() } else { None };
        let bounds = match dictionary {
            Some(dictionary) => dictionary.bounds,
            None => self.bounds,
        };
        // The rows not kept hold a value within the bounds, so they're still
        // true of every value.
        let mut out = context.values_buffer(rows * 8)?;
        let values = out.as_mut_slice::<i64>();
        at_mut!(values, ..rows).fill(bounds.map_or(0, |bounds| bounds.min));
        // Each kept row's value: how many rows before it aren't null.
        let (mut row, mut before) = (0, 0);
        let mut value_of = |kept: usize| {
            if let Some(bits) = bits {
                before += (row..kept).filter(|&r| BitSlice::of(bits).get(r)).count();
                row = kept;
                BitSlice::of(bits).get(kept).then_some(before)
            } else {
                Some(kept)
            }
        };
        let mut passed = 0;
        if let Some(dictionary) = dictionary {
            let Some(words) = &dictionary.words else { return Err(Error::Corrupt) };
            let entries = at!(words.as_slice::<i64>(), ..dictionary.count);
            let indices = page.get(position.value..).ok_or(Error::Corrupt)?;
            let index = &mut position.index;
            let mut one = [0];
            for &kept in kept {
                let kept = usize::from(kept);
                let Some(value) = value_of(kept) else { continue };
                index.skip(indices, value - passed).ok_or(Error::Corrupt)?;
                index.take(indices, &mut one).ok_or(Error::Corrupt)?;
                *at_mut!(values, kept) = *entries.get(one[0] as usize).ok_or(Error::Corrupt)?;
                passed = value + 1;
            }
            index.skip(indices, valid - passed).ok_or(Error::Corrupt)?;
        } else {
            let (width, at) = (plain::width(self.column.physical), position.value);
            for &kept in kept {
                let kept = usize::from(kept);
                let Some(value) = value_of(kept) else { continue };
                plain::words(self.column, page, at + value * width, at_mut!(values, kept..=kept))?;
            }
            position.value = at + valid * width;
        }
        position.left -= rows;
        let column = ColumnView::new(context, self.data_type, out, validity)?;
        Ok(bounds::bounded(column, bounds))
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
        self.body(context)?;
        let (Some((_, len)), Some(body)) = (self.page, &self.body) else { pipit_kernel::failed!() };
        let page = at!(body.as_slice::<u8>(), ..len);
        let position = &mut self.position;
        let decoded = scratch(context, &mut self.scratch)?.as_mut_slice();
        let (validity, valid) =
            position.validity(context, self.column.optional, page, rows, decoded)?;
        let data_type = self.data_type;
        let column = if position.indexed {
            let Some(dictionary) = &self.dictionary else { pipit_kernel::failed!() };
            let mut column = position
                .indexed(context, data_type, page, dictionary, rows, valid, validity, decoded)?;
            // Strings' dictionaries are kept, and a run of one entry is a
            // constant, unless `forms` rules them out.
            column.make_in(context, forms)?;
            column
        } else {
            let bits = validity.as_ref().map(Buffer::as_slice::<u8>);
            match data_type {
                DataType::String => {
                    // Views into the page, so no string is copied: the column
                    // holds the page.
                    let mut views = context.values_buffer(rows * 8)?;
                    let at = position.value;
                    position.value = plain::views(page, at, views.as_mut_slice::<u32>(), valid)?;
                    if let Some(bits) = bits {
                        bits::spread(views.as_mut_slice::<i64>(), valid, bits, 0);
                    }
                    ColumnView::strings(context, views, body.clone(), validity)?
                }
                DataType::Int64 | DataType::Float64 => {
                    let mut out = context.values_buffer(rows * 8)?;
                    let words = out.as_mut_slice::<i64>();
                    let (column, at) = (self.column, position.value);
                    position.value = plain::words(column, page, at, at_mut!(words, ..valid))?;
                    if let Some(bits) = bits {
                        bits::spread(words, valid, bits, 0);
                    }
                    bounds::bounded(
                        ColumnView::new(context, data_type, out, validity)?,
                        self.bounds,
                    )
                }
            }
        };
        position.left -= rows;
        Ok(column)
    }

    /// Moves to the next page, reading only its header.
    fn next_page(&mut self, context: &mut Context) -> Result<(), Error> {
        let header_at = self.position.next;
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
            header: header_at,
            rows: left,
            body,
            left,
            indexed,
            started: false,
            deferred: 0,
            levels: (0, 0),
            level: Hybrid::default(),
            value: 0,
            index: Hybrid::default(),
        };
        Ok(())
    }

    /// Reads the body of the page being read, if it isn't already, and finds
    /// its levels and values, if they haven't been; and the dictionary, if
    /// its values are indices into it.
    fn body(&mut self, context: &mut Context) -> Result<(), Error> {
        if self.position.indexed && self.dictionary.is_none() {
            self.dictionary(context)?;
        }
        let body = self.position.body;
        if !matches!(self.page, Some((at, _)) if at == body.at) {
            self.page = None;
            let into = self.body.take();
            let (buffer, len) = self.read_body(context, body, into)?;
            (self.page, self.body) = (Some((body.at, len)), Some(buffer));
        }
        if !self.position.started {
            let (Some((_, len)), Some(page)) = (self.page, &self.body) else {
                pipit_kernel::failed!()
            };
            let bytes = at!(page.as_slice::<u8>(), ..len);
            let position = &mut self.position;
            // An optional column's definition levels come first, after their
            // length.
            if self.column.optional {
                let len = plain::length(bytes, 0)?;
                (position.levels, position.value) = ((4, 4 + len), 4 + len);
            }
            // Indices come after their width in bits.
            if position.indexed {
                let width = u32::from(*bytes.get(position.value).ok_or(Error::Corrupt)?);
                if width > 32 {
                    return Err(Error::Corrupt);
                }
                (position.index, position.value) = (Hybrid::new(width), position.value + 1);
            }
            (position.level, position.started) = (Hybrid::new(1), true);
            let deferred = core::mem::take(&mut position.deferred);
            if deferred > 0 {
                let (column, data_type) = (self.column, self.data_type);
                position.advance(context, column, data_type, &mut self.scratch, bytes, deferred)?;
            }
        }
        Ok(())
    }

    /// Decodes the chunk's dictionary.
    fn dictionary(&mut self, context: &mut Context) -> Result<(), Error> {
        let (body, count) = self.dictionary_page.ok_or(Error::Corrupt)?;
        let (body, len) = self.read_body(context, body, None)?;
        let page = at!(body.as_slice::<u8>(), ..len);
        let count32 = u32::try_from(count).map_err(|_| Error::Unsupported)?;
        // The values, then a null.
        let mut nulls = context.bytes_buffer(BitSlice::<u8>::words_for(count + 1))?;
        let bits = nulls.as_mut_slice::<u8>();
        bits.fill(0xff);
        BitSlice::of_mut(&mut *bits).clear(count);
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
        self.dictionary = Some(Dictionary { values, bounds, words, nullable, count });
        Ok(())
    }

    /// A page's body, decompressed if it's stored compressed, in `into` if
    /// it can be reused, and its length.
    fn read_body(
        &mut self,
        context: &Context,
        body: Body,
        into: Option<Buffer>,
    ) -> Result<(Buffer, usize), Error> {
        let len = usize::try_from(body.len).map_err(|_| Error::Corrupt)?;
        if self.codec == 0 {
            let mut page = reusable(context, into, len)?;
            self.source.read(body.at, at_mut!(page.as_mut_slice::<u8>(), ..len))?;
            return Ok((page, len));
        }
        let mut stored = reusable(context, self.stored.take(), len)?;
        self.source.read(body.at, at_mut!(stored.as_mut_slice::<u8>(), ..len))?;
        let size = usize::try_from(body.size).map_err(|_| Error::Corrupt)?;
        let mut page = reusable(context, into, size)?;
        let (from, to) = (at!(stored.as_slice(), ..len), at_mut!(page.as_mut_slice(), ..size));
        self.codecs.decompress(self.codec, from, to)?;
        self.stored = Some(stored);
        Ok((page, size))
    }

    /// `len` bytes of the chunk from `at`.
    fn read_bytes(&self, context: &Context, at: u64, len: u64) -> Result<Buffer, Error> {
        let len = usize::try_from(len).map_err(|_| Error::Corrupt)?;
        // SAFETY: reading fills every byte, or fails.
        let mut bytes = unsafe { Buffer::allocate_uninit(context.env(), MemoryKind::Base, len)? };
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

/// `buffer` if nothing else holds it and it has `len` bytes, else a new one
/// of `len`.
fn reusable(context: &Context, buffer: Option<Buffer>, len: usize) -> Result<Buffer, Error> {
    match buffer {
        Some(buffer) if buffer.is_unique() && buffer.size_bytes() >= len => Ok(buffer),
        // SAFETY: what's read from it is written first, or fails.
        _ => Ok(unsafe { Buffer::allocate_uninit(context.env(), MemoryKind::Base, len)? }),
    }
}

/// The scratch memory in `slot`, allocated the first time it's needed.
fn scratch<'b>(context: &Context, slot: &'b mut Option<Buffer>) -> Result<&'b mut Buffer, Error> {
    match slot {
        Some(scratch) => Ok(scratch),
        None => Ok(slot.insert(Buffer::allocate(context.env(), SCRATCH_BYTES)?)),
    }
}

impl Position {
    /// The page's definition levels.
    fn levels<'p>(&self, page: &'p [u8]) -> Result<&'p [u8], Error> {
        page.get(self.levels.0..self.levels.1).ok_or(Error::Corrupt)
    }

    /// Moves past the next `rows` rows of `page`, the body being read, of
    /// `column`, without decoding their values. Levels are decoded into the
    /// scratch memory in `slot`.
    fn advance(
        &mut self,
        context: &Context,
        column: Column,
        data_type: DataType,
        slot: &mut Option<Buffer>,
        page: &[u8],
        rows: usize,
    ) -> Result<(), Error> {
        let mut valid = rows;
        if column.optional {
            let levels = self.levels(page)?;
            let decoded = scratch(context, slot)?.as_mut_slice::<u32>();
            valid = 0;
            for start in (0..rows).step_by(decoded.len()) {
                let chunk = at_mut!(decoded, ..(rows - start).min(BATCH_ROWS_MAX as usize));
                self.level.take(levels, chunk).ok_or(Error::Corrupt)?;
                valid += chunk.iter().filter(|&&level| level == 1).count();
            }
        }
        let at = self.value;
        match (self.indexed, data_type) {
            (true, _) => {
                let indices = page.get(at..).ok_or(Error::Corrupt)?;
                self.index.skip(indices, valid).ok_or(Error::Corrupt)?;
            }
            (false, DataType::String) => {
                let mut at = at;
                for _ in 0..valid {
                    at += 4 + plain::length(page, at)?;
                }
                self.value = at;
            }
            (false, _) => self.value += valid * plain::width(column.physical),
        }
        Ok(())
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
        data_type: DataType,
        page: &[u8],
        dictionary: &Dictionary,
        rows: usize,
        valid: usize,
        validity: Option<Buffer>,
        decoded: &mut [u32],
    ) -> Result<ColumnView, Error> {
        let bytes = page.get(self.value..).ok_or(Error::Corrupt)?;
        let rows32 = u32::try_from(rows).map_err(|_| Error::Corrupt)?;
        let count = dictionary.count;
        if valid == 0 {
            let null = dictionary.nullable.slice(dictionary.nullable.row_count() - 1, 1);
            return Ok(ColumnView::constant(context, &null, rows32)?);
        }
        let mut run = self.index.next_run(bytes, at_mut!(decoded, ..valid));
        // One value for every row.
        if let Some(Run::Repeat(i, n)) = run
            && n == rows
            && (i as usize) < count
        {
            return Ok(ColumnView::constant(context, &dictionary.values.slice(i, 1), rows32)?);
        }
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
                run = self.index.next_run(bytes, at_mut!(decoded, ..valid - at));
            }
            if let Some(bits) = &validity {
                bits::spread(values, valid, bits.as_slice(), 0);
            }
            let column = ColumnView::new(context, data_type, out, validity)?;
            return Ok(bounds::bounded(column, dictionary.bounds));
        }
        let mut indices = context.indices_buffer(rows * 4)?;
        let out = indices.as_mut_slice::<u32>();
        loop {
            match run.ok_or(Error::Corrupt)? {
                Run::Repeat(i, n) => at_mut!(out, at..at + n).fill(i),
                // Only the first run is in `decoded`: the rest are unpacked
                // in place.
                Run::Packed(n) if at == 0 => at_mut!(out, ..n).copy_from_slice(at!(decoded, ..n)),
                Run::Packed(_) => {}
            }
            at += run.map_or(0, Run::len);
            if at == valid {
                break;
            }
            run = self.index.next_run(bytes, at_mut!(out, at..valid));
        }
        if at!(out, ..valid).iter().any(|&i| i >= null) {
            return Err(Error::Corrupt);
        }
        Ok(match &validity {
            None => ColumnView::dictionary(context, &dictionary.values, indices)?,
            Some(bits) => {
                bits::spread(out, valid, bits.as_slice(), null);
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
        optional: bool,
        page: &[u8],
        rows: usize,
        decoded: &mut [u32],
    ) -> Result<(Option<Buffer>, usize), Error> {
        if !optional {
            return Ok((None, rows));
        }
        let levels = self.levels(page)?;
        // Most pages have no nulls, so their levels are runs of 1s.
        let mut row = 0;
        let mut run = loop {
            if row == rows {
                return Ok((None, rows));
            }
            let left = at_mut!(decoded, ..rows - row);
            let run = self.level.next_run(levels, left).ok_or(Error::Corrupt)?;
            if run != Run::Repeat(1, run.len()) {
                break run;
            }
            row += run.len();
        };
        let mut validity = context.small_buffer(BitSlice::<u8>::words_for(rows))?;
        let bits = validity.as_mut_slice::<u8>();
        bits.fill(0);
        BitSlice::of_mut(&mut *bits).set_run(0, row);
        let mut valid = row;
        loop {
            match run {
                Run::Repeat(level, n) => {
                    if level == 1 {
                        BitSlice::of_mut(&mut *bits).set_run(row, n);
                        valid += n;
                    }
                    row += n;
                }
                Run::Packed(n) => {
                    for &level in at!(decoded, ..n) {
                        BitSlice::of_mut(&mut *bits).set_if(row, level == 1);
                        (row, valid) = (row + 1, valid + usize::from(level == 1));
                    }
                }
            }
            if row == rows {
                break;
            }
            let left = at_mut!(decoded, ..rows - row);
            run = self.level.next_run(levels, left).ok_or(Error::Corrupt)?;
        }
        if valid == rows {
            return Ok((None, rows));
        }
        Ok((Some(validity), valid))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::string::String;
    use std::vec::Vec;

    use pipit_kernel::bytes::ReadError;
    use pipit_kernel::column::Form;

    use super::*;
    use crate::Uncompressed;
    use pipit_kernel::env::Env;

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

    #[derive(Clone, PartialEq, Debug)]
    enum Cell {
        Int(i64),
        Float(f64),
        Text(String),
        Null,
    }

    /// Row `i` of column `c` of `small`.
    #[expect(clippy::cast_precision_loss, reason = "small test values")]
    fn small(c: usize, i: i64) -> Cell {
        match c {
            0 => Cell::Int(i),
            1 => Cell::Float(i as f64 * 0.5),
            _ => Cell::Text(std::format!("row{i}")),
        }
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
        let env = Env::current();
        let mut view = view.clone();
        view.make_in(&mut Context::new(env, None), Forms::FLAT).unwrap();
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
        let env = Env::current();
        let source: &dyn ByteSource = &mut bytes;
        let file = ParquetFile::open(env, source).unwrap();
        let mut context = Context::new(env, None);
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
        let env = Env::current();
        let mut bytes = DICT;
        let source: &dyn ByteSource = &mut bytes;
        let file = ParquetFile::open(env, source).unwrap();
        let mut context = Context::new(env, None);
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
        let env = Env::current();
        let source: &dyn ByteSource = &mut bytes;
        let file = ParquetFile::open(env, source).unwrap();
        let mut context = Context::new(env, None);
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

    #[test]
    fn reads_kept_rows_as_read_does() {
        reads_kept_rows_in(SMALL, small);
        reads_kept_rows_in(NULLS, nulls);
        reads_kept_rows_in(DICT, dict);
    }

    /// Reads batches of each column's first chunk of `bytes` keeping some
    /// rows, then the next batch whole, which must start where they ended.
    fn reads_kept_rows_in(mut bytes: &[u8], expected: fn(usize, i64) -> Cell) {
        let env = Env::current();
        let source: &dyn ByteSource = &mut bytes;
        let file = ParquetFile::open(env, source).unwrap();
        let mut context = Context::new(env, None);
        let patterns: [&[u16]; 4] = [&[], &[0], &[3, 4, 5, 90, 299], &[1, 7, 150, 151, 152]];
        for c in 0..3 {
            let mut reader =
                ChunkReader::new(source, &Uncompressed, file.columns()[c], file.chunk(0, c))
                    .unwrap();
            let mut row = 0;
            for kept in patterns {
                let rows = reader.page_left(&mut context).unwrap().min(300);
                let read = cells(&reader.read_kept(&mut context, rows, kept, ANY).unwrap());
                let got: Vec<_> = kept.iter().map(|&r| read[usize::from(r)].clone()).collect();
                let from = i64::try_from(row).unwrap();
                let want: Vec<_> = kept.iter().map(|&r| expected(c, from + i64::from(r))).collect();
                assert_eq!(got, want, "column {c} from {row}");
                row += rows;
                let rows = reader.page_left(&mut context).unwrap().min(10);
                let from = i64::try_from(row).unwrap();
                let want: Vec<_> = (from..).take(rows).map(|i| expected(c, i)).collect();
                assert_eq!(cells(&reader.read(&mut context, rows, ANY).unwrap()), want);
                row += rows;
            }
        }
    }

    #[test]
    fn goes_to_places_passed() {
        goes_to_places_passed_in(NULLS, nulls);
        goes_to_places_passed_in(DICT, dict);
    }

    /// Passes over each column's first chunk of `bytes` noting places, as a
    /// scan of lazy columns does, then reads from them out of order, as
    /// loads do: on, back within a page, and again.
    fn goes_to_places_passed_in(mut bytes: &[u8], expected: fn(usize, i64) -> Cell) {
        let env = Env::current();
        let source: &dyn ByteSource = &mut bytes;
        let file = ParquetFile::open(env, source).unwrap();
        let mut context = Context::new(env, None);
        for c in 0..3 {
            let reader = || {
                ChunkReader::new(source, &Uncompressed, file.columns()[c], file.chunk(0, c))
                    .unwrap()
            };
            let (mut passing, mut places, mut row) = (reader(), Vec::new(), 0);
            while passing.page_left(&mut context).unwrap() > 0 {
                let rows = passing.page_left(&mut context).unwrap().min(300);
                places.push((passing.place(), row, rows.min(50)));
                passing.pass(rows);
                row += rows;
            }
            assert_eq!(row, 2048);
            assert!(places.len() > 2);
            let mut loading = reader();
            loading.page_left(&mut context).unwrap();
            let order = [2, 0, 1, 1, places.len() - 1, 0];
            for &((header, at), row, rows) in order.iter().map(|&i| &places[i]) {
                loading.go_to(&mut context, header, at).unwrap();
                let from = i64::try_from(row).unwrap();
                let expected: Vec<_> = (from..).take(rows).map(|i| expected(c, i)).collect();
                assert_eq!(cells(&loading.read(&mut context, rows, ANY).unwrap()), expected);
            }
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
        let env = Env::current();
        let source = Counting(SMALL, core::cell::Cell::new(0));
        let file = ParquetFile::open(env, &source).unwrap();
        let mut context = Context::new(env, None);
        let chunk = file.chunk(0, 0);
        let mut reader =
            ChunkReader::new(&source, &Uncompressed, file.columns()[0], chunk).unwrap();
        source.1.set(0);
        loop {
            let left = reader.page_left(&mut context).unwrap();
            if left == 0 {
                break;
            }
            // In two parts, as a scan skips a page a batch at a time.
            reader.skip(&mut context, left / 2).unwrap();
            reader.skip(&mut context, left - left / 2).unwrap();
        }
        assert!(source.1.get() * 10 < chunk.len);
    }

    #[test]
    fn decodes_the_dictionary_only_when_values_are_read() {
        let env = Env::current();
        let mut bytes = DICT;
        let source: &dyn ByteSource = &mut bytes;
        let file = ParquetFile::open(env, source).unwrap();
        let mut context = Context::new(env, None);
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
