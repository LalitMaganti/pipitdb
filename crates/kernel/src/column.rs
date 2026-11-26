//! `ColumnView`: a range of rows over a column's buffers.

#![allow(unsafe_code)]

use crate::containers::BitSlice;
use core::cell::Cell;
use core::ptr::NonNull;

use crate::allocator::AllocError;
use crate::buffer::{Buffer, Primitive};
use crate::context::Context;
use crate::row_batch::BATCH_ROWS_MAX;

/// A set of the forms a column may come in: what a plan lets a producer
/// make a column in, and what a consumer takes it in. Every producer can
/// make flat columns and every consumer takes them, so every set a plan
/// gives holds `FLAT`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Forms(u8);

impl Forms {
    /// Row `i` is value `i`.
    pub const FLAT: Forms = Forms(1);
    /// Not read yet: read when loaded, for the rows still kept.
    pub const LAZY: Forms = Forms(2);
    /// Every row the same value.
    pub const CONSTANT: Forms = Forms(4);
    /// Each row an index into a column of values.
    pub const DICTIONARY: Forms = Forms(8);
    /// Each row's value its own number, from a first: an id column, stored
    /// as nothing.
    pub const SEQUENCE: Forms = Forms(16);

    /// Whether every form in `other` is in this.
    pub fn contains(self, other: Forms) -> bool {
        self.0 & other.0 == other.0
    }

    /// The forms in this or `other`, as `|` gives, for constants.
    pub const fn union(self, other: Forms) -> Forms {
        Forms(self.0 | other.0)
    }
}

impl core::ops::BitOr for Forms {
    type Output = Forms;

    fn bitor(self, other: Forms) -> Forms {
        self.union(other)
    }
}

impl core::ops::BitAnd for Forms {
    type Output = Forms;

    fn bitand(self, other: Forms) -> Forms {
        Forms(self.0 & other.0)
    }
}

/// A column's type. Strings are bytes, as stored: usually UTF-8 text, but
/// not checked to be, as nothing reads them as text. They're compared,
/// hashed and grouped byte by byte; what shows them to people decides what
/// to do with bytes that aren't UTF-8.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int64,
    Float64,
    String,
}

impl DataType {
    /// How many bytes a value takes: a word, or for a string, a view: where
    /// its bytes start, and how many there are.
    pub fn width_bytes(self) -> usize {
        match self {
            DataType::Int64 | DataType::Float64 | DataType::String => 8,
        }
    }

    /// Whether a column's values are views into a buffer of bytes.
    pub fn has_bytes(self) -> bool {
        matches!(self, DataType::String)
    }
}

/// The least and greatest values of a column of integers, so operators can
/// pick narrower arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Bounds {
    pub min: i64,
    pub max: i64,
}

impl Bounds {
    /// How many integers lie within them.
    pub fn count(self) -> u64 {
        self.max.abs_diff(self.min).saturating_add(1)
    }
}

/// Rows of values in Arrow's layout: values, and an optional validity bitmap
/// with one bit per value, set if it isn't null. As DuckDB's vectors, rows
/// map to values in one of three forms: each to its own (flat), all to one
/// (constant), or each through an index (dictionary). Or the values aren't
/// read yet (lazy): a handle says where they are, for whoever wrote it to
/// load them.
///
/// It's a handle, small to move: the buffers and what they hold are in a
/// header it shares with its clones and slices.
///
/// A view in a batch has the batch's rows, at most `BATCH_ROWS_MAX`. One an
/// integration keeps for itself, such as a chunk's dictionary, may have more.
#[derive(Clone)]
pub struct ColumnView {
    /// A `Header`, written in it.
    header: Buffer,
    /// Where the view's rows start among the header's.
    start: u32,
    /// How many rows the view has.
    row_count: u32,
}

/// A column's buffers and what they hold, shared by its views. Its values
/// are `len` of `values`, and of `bytes` and `validity` if any, from
/// `value_start`. A flat header's rows are its values, a dictionary one's its
/// `indices`, and a constant or lazy one's as many as its views say. A lazy
/// header's `values` hold its handle, `len` bytes of it.
struct Header {
    data_type: DataType,
    kind: Kind,
    values: Buffer,
    bytes: Option<Buffer>,
    bytes_len: usize,
    others: Option<Others>,
    validity: Option<Buffer>,
    indices: Option<Buffer>,
    value_start: u32,
    len: u32,
    bounds: Cell<Bounds>,
}

impl Header {
    /// The bytes of its own that its strings' words read.
    fn own_bytes(&self) -> &[u8] {
        self.bytes.as_ref().map_or(&[][..], |bytes| at!(bytes.as_slice::<u8>(), ..self.bytes_len))
    }

    /// What its strings are in, for another column to share.
    fn strings(&self) -> StringsIn {
        StringsIn { bytes: self.bytes.clone(), len: self.bytes_len, others: self.others.clone() }
    }
}

/// A flat view, with where its values are, flat enough to read in a loop
/// as Perfetto's `GatherSource`: row `row` is the word at `row` and the bit
/// at `row`.
///
/// It keeps the view, so the buffers its pointers are into stay alive, and
/// they aren't written once shared. Their bytes don't move with it. Each
/// is a cache line, so finding one is a shift and reading one a line.
#[repr(align(64))]
pub(crate) struct GatherSource {
    /// The view.
    view: ColumnView,
    /// Its values, a word each: an integer, a float's bits, or a string's
    /// word.
    words: NonNull<[i64; BATCH_ROWS_MAX as usize]>,
    /// Which values aren't null, if any are.
    validity: Option<NonNull<[u8; BATCH_ROWS_MAX as usize / 8]>>,
    /// What a string column's words whose high half is 0 are into, or
    /// none.
    bytes: NonNull<[u8]>,
}

impl GatherSource {
    /// `view`'s, which must `fit`.
    pub(crate) fn of(view: &ColumnView) -> GatherSource {
        check!(GatherSource::fits(view));
        let header = view.header();
        let start = (header.value_start + view.start) as usize;
        let at = |buffer: &Buffer, at: usize| buffer.as_ptr::<u8>().wrapping_add(at).cast_mut();
        let validity = header.validity.as_ref().map(|bits| at(bits, start / 8));
        let Some(words) = NonNull::new(at(&header.values, start * 8)) else { failed!() };
        GatherSource {
            view: view.clone(),
            words: words.cast(),
            validity: validity.and_then(NonNull::new).map(NonNull::cast),
            bytes: NonNull::from(header.own_bytes()),
        }
    }

    /// Whether `view` has a batch's worth of words and of bits from its
    /// first, its first bit the first of a byte.
    #[inline(never)]
    pub(crate) fn fits(view: &ColumnView) -> bool {
        let header = view.header();
        let start = (header.value_start + view.start) as usize;
        let rows = BATCH_ROWS_MAX as usize;
        header.values.size_bytes() >= (start + rows) * 8
            && header.validity.as_ref().is_none_or(|bits| {
                start.is_multiple_of(8) && bits.size_bytes() >= (start + rows) / 8
            })
    }

    pub(crate) fn view(&self) -> &ColumnView {
        &self.view
    }

    /// The word of row `row`, a row of a batch.
    #[inline]
    pub(crate) fn word(&self, row: u32) -> i64 {
        // SAFETY: made from the view's words, which it keeps alive.
        let words = unsafe { self.words.as_ref() };
        *at!(words, row as usize % BATCH_ROWS_MAX as usize)
    }

    /// Whether row `row`, a row of a batch, isn't null.
    #[inline]
    pub(crate) fn is_valid(&self, row: u32) -> bool {
        let row = row as usize % BATCH_ROWS_MAX as usize;
        self.validity.is_none_or(|bits| {
            // SAFETY: made from the view's validity, which it keeps alive.
            let bits = unsafe { bits.as_ref() };
            BitSlice::of(bits).get(row)
        })
    }

    /// The string at `word` of a string column.
    #[inline]
    pub(crate) fn string(&self, word: u64) -> &[u8] {
        if word >> 32 != 0 {
            return self.view.values().strings().of(word);
        }
        // SAFETY: made from the view's bytes, which it keeps alive, and
        // `word` is one of its words, valid when it was made.
        unsafe { string_at(self.bytes.as_ref(), word) }
    }

    /// What a string column's strings are in, for a column that shares
    /// them: its own bytes, how many words read, and the others.
    pub(crate) fn shared_strings(&self) -> StringsIn {
        self.view.header().strings()
    }
}

/// What a string column's words point into: its own bytes, of which words
/// read the first `len`, as their owner may append past them, and other
/// buffers, by a word's high half less one.
pub(crate) struct StringsIn {
    pub(crate) bytes: Option<Buffer>,
    pub(crate) len: usize,
    pub(crate) others: Option<Others>,
}

impl StringsIn {
    /// None, as a column of numbers has.
    pub(crate) const NONE: StringsIn = StringsIn { bytes: None, len: 0, others: None };

    /// All of `bytes`, and no others.
    pub(crate) fn own(bytes: Buffer) -> StringsIn {
        StringsIn { len: bytes.size_bytes(), bytes: Some(bytes), others: None }
    }
}

/// A buffer of strings a column's words point into, besides its own bytes,
/// and how many of its bytes words read.
#[derive(Clone)]
pub(crate) struct Slot {
    buffer: Buffer,
    len: usize,
}

impl Slot {
    pub(crate) fn new(buffer: Buffer, len: usize) -> Slot {
        check!(len <= buffer.size_bytes());
        Slot { buffer, len }
    }

    /// It as a column's own bytes.
    pub(crate) fn into_own(self) -> StringsIn {
        StringsIn { bytes: Some(self.buffer), len: self.len, others: None }
    }

    /// The bytes words read.
    #[inline]
    fn bytes(&self) -> &[u8] {
        // SAFETY: the first `len` bytes are in the buffer, and its owner
        // appends only past them.
        unsafe { core::slice::from_raw_parts(self.buffer.as_non_null().as_ptr(), self.len) }
    }
}

/// The other buffers a string column's words point into, held in a buffer
/// of `Slot`s shared by the columns that share them, and let go of with
/// the last.
#[derive(Clone)]
pub(crate) struct Others(Buffer);

impl Others {
    /// `slots`, as many as `count`.
    pub(crate) fn of(
        context: &mut Context,
        count: usize,
        slots: impl Iterator<Item = Slot>,
    ) -> Result<Others, AllocError> {
        let size = count * size_of::<Slot>();
        let mut held =
            if size <= 256 { context.small_buffer(size)? } else { context.bytes_buffer(size)? };
        let at = held.as_mut_non_null().cast::<Slot>();
        let mut written = 0;
        for slot in slots.take(count) {
            // SAFETY: the buffer has room for `count` slots, aligned for any.
            unsafe { at.add(written).write(slot) };
            written += 1;
        }
        check!(written == count);
        Ok(Others(held))
    }

    #[inline]
    pub(crate) fn slots(&self) -> &[Slot] {
        let count = self.0.size_bytes() / size_of::<Slot>();
        // SAFETY: `of` wrote `count` slots, which live until the last clone.
        unsafe { core::slice::from_raw_parts(self.0.as_non_null().cast::<Slot>().as_ptr(), count) }
    }
}

impl Drop for Others {
    fn drop(&mut self) {
        if self.0.is_unique() {
            let count = self.0.size_bytes() / size_of::<Slot>();
            let at = self.0.as_non_null().cast::<Slot>();
            // SAFETY: the last holder drops the slots `of` wrote, once.
            unsafe {
                core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(at.as_ptr(), count));
            }
        }
    }
}

/// Each is its bit in `Forms`.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Kind {
    Flat = Forms::FLAT.0,
    Lazy = Forms::LAZY.0,
    Constant = Forms::CONSTANT.0,
    Dictionary = Forms::DICTIONARY.0,
    Sequence = Forms::SEQUENCE.0,
}

/// Bounds no value is within, for when they aren't known.
const UNKNOWN: Bounds = Bounds { min: 1, max: 0 };

/// How a column's rows map to its `values`.
#[derive(Clone, Copy)]
pub enum Form<'a> {
    /// Row `i` is value `i`.
    Flat,
    /// Every row is value 0.
    Constant,
    /// Row `i` is value `indices[i]`.
    Dictionary(&'a [u32]),
    /// Row `i` is the integer `first + i`, with no values.
    Sequence(i64),
}

/// The values a column's rows map to.
#[derive(Clone, Copy)]
pub struct Values<'a> {
    header: &'a Header,
    start: u32,
    len: u32,
}

impl ColumnView {
    /// A flat view of every value in `values`.
    pub fn new(
        context: &mut Context,
        data_type: DataType,
        values: Buffer,
        validity: Option<Buffer>,
    ) -> Result<ColumnView, AllocError> {
        check!(!data_type.has_bytes());
        check!(values.size_bytes().is_multiple_of(data_type.width_bytes()));
        let Ok(len) = u32::try_from(values.size_bytes() / data_type.width_bytes()) else {
            failed!();
        };
        ColumnView::flat(context, data_type, values, StringsIn::NONE, validity, len)
    }

    /// `Int64` values, each null if `None`, in buffers of their own: for
    /// tests and tools, which make columns from values they hold.
    pub fn from_int64s(
        context: &mut Context,
        values: impl ExactSizeIterator<Item = Option<i64>>,
    ) -> Result<ColumnView, AllocError> {
        let env = context.env();
        let mut words = Buffer::allocate(env, values.len() * 8)?;
        let mut bits = Buffer::allocate(env, BitSlice::<u8>::words_for(values.len()))?;
        let mut nulls = false;
        let (out, valid) = (words.as_mut_slice::<i64>(), bits.as_mut_slice::<u8>());
        for (row, value) in values.enumerate() {
            *at_mut!(out, row) = value.unwrap_or(0);
            BitSlice::of_mut(&mut *valid).set_if(row, value.is_some());
            nulls |= value.is_none();
        }
        ColumnView::new(context, DataType::Int64, words, nulls.then_some(bits))
    }

    /// Strings as words: value `i` is the string at `words[i]` of `bytes`, a
    /// 4-byte little-endian length and then that many bytes, as a Parquet
    /// page holds them. The strings needn't be in order or next to each
    /// other, so they can stay where they were read, between other bytes. A
    /// null's word is that of a string too, such as an empty one. Each word
    /// is checked, so strings are read without checks.
    pub fn strings(
        context: &mut Context,
        words: Buffer,
        bytes: Buffer,
        validity: Option<Buffer>,
    ) -> Result<ColumnView, AllocError> {
        check!(strings_valid(words.as_slice(), bytes.as_slice()));
        // SAFETY: each word was checked just above.
        unsafe { ColumnView::strings_in(context, words, StringsIn::own(bytes), validity) }
    }

    /// As `strings`, for words already known to be valid, as whoever wrote
    /// them checked or made them so.
    ///
    /// # Safety
    ///
    /// As `strings_in`'s, of words into `bytes`.
    pub unsafe fn strings_unchecked(
        context: &mut Context,
        words: Buffer,
        bytes: Buffer,
        validity: Option<Buffer>,
    ) -> Result<ColumnView, AllocError> {
        // SAFETY: as the caller promises.
        unsafe { ColumnView::strings_in(context, words, StringsIn::own(bytes), validity) }
    }

    /// As `strings`, of words into `bytes`, if any, and into the other
    /// buffers `strings` says, already known to be valid, as whoever wrote
    /// them checked or made them so.
    ///
    /// # Safety
    ///
    /// Each word's low half, a null's too, must be where a 4-byte
    /// little-endian length is in the buffer its high half names, followed
    /// by that many bytes, all among those its column reads: strings are
    /// read without checks.
    pub(crate) unsafe fn strings_in(
        context: &mut Context,
        words: Buffer,
        strings: StringsIn,
        validity: Option<Buffer>,
    ) -> Result<ColumnView, AllocError> {
        check!(words.size_bytes().is_multiple_of(8));
        let Ok(len) = u32::try_from(words.size_bytes() / 8) else {
            failed!();
        };
        ColumnView::flat(context, DataType::String, words, strings, validity, len)
    }

    /// `count` strings, string `i` being `string(i)`, null if `None`, in
    /// buffers of their own from `context`.
    pub fn from_strings<'s>(
        context: &mut Context,
        count: usize,
        string: impl Fn(usize) -> Option<&'s [u8]>,
    ) -> Result<ColumnView, AllocError> {
        let mut bits = context.small_buffer(BitSlice::<u8>::words_for(count))?;
        let valid = bits.as_mut_slice::<u8>();
        valid.fill(0);
        let mut nulls = false;
        for i in 0..count {
            let value = string(i).is_some();
            BitSlice::of_mut(&mut *valid).set_if(i, value);
            nulls |= !value;
        }
        let built = string_buffers(context, count, |i| string(i).unwrap_or_default())?;
        ColumnView::built_strings(context, built, nulls.then_some(bits))
    }

    /// The strings `string_buffers` built.
    pub(crate) fn built_strings(
        context: &mut Context,
        Built(words, bytes): Built,
        validity: Option<Buffer>,
    ) -> Result<ColumnView, AllocError> {
        // SAFETY: `string_buffers` wrote each word where its string is.
        unsafe { ColumnView::strings_unchecked(context, words, bytes, validity) }
    }

    /// `rows` rows of `text`, each a word of one copy of it.
    pub(crate) fn repeated_string(
        context: &mut Context,
        rows: usize,
        text: &[u8],
        validity: Option<Buffer>,
    ) -> Result<ColumnView, AllocError> {
        let mut bytes = context.bytes_buffer(4 + text.len())?;
        let Ok(len) = u32::try_from(text.len()) else { return Err(AllocError) };
        let (length, rest) = bytes.as_mut_slice::<u8>().split_at_mut(4);
        length.copy_from_slice(&len.to_le_bytes());
        rest.copy_from_slice(text);
        let mut words = context.values_buffer(rows * 8)?;
        words.as_mut_slice::<u64>().fill(0);
        // SAFETY: every word is 0, where the string just written is.
        unsafe { ColumnView::strings_unchecked(context, words, bytes, validity) }
    }

    fn flat(
        context: &mut Context,
        data_type: DataType,
        values: Buffer,
        StringsIn { bytes, len: bytes_len, others }: StringsIn,
        validity: Option<Buffer>,
        len: u32,
    ) -> Result<ColumnView, AllocError> {
        if let Some(validity) = &validity {
            check!(validity.size_bytes() >= BitSlice::<u8>::words_for(len as usize));
        }
        let (indices, bounds) = (None, Cell::new(UNKNOWN));
        let kind = Kind::Flat;
        let header = Header {
            data_type,
            kind,
            values,
            bytes,
            bytes_len,
            others,
            validity,
            indices,
            value_start: 0,
            len,
            bounds,
        };
        ColumnView::with_header(context, header, len)
    }

    /// `row_count` rows of the one value of `value`, a flat view.
    pub fn constant(
        context: &mut Context,
        value: &ColumnView,
        row_count: u32,
    ) -> Result<ColumnView, AllocError> {
        check!(value.header().kind == Kind::Flat && value.row_count == 1);
        let header = Header { kind: Kind::Constant, len: 1, ..value.shared(None) };
        ColumnView::with_header(context, header, row_count)
    }

    /// A row for each of `indices`, a `u32` each, into the values of
    /// `dictionary`, a flat view.
    pub fn dictionary(
        context: &mut Context,
        dictionary: &ColumnView,
        indices: Buffer,
    ) -> Result<ColumnView, AllocError> {
        check!(dictionary.header().kind == Kind::Flat);
        let Ok(row_count) = u32::try_from(indices.size_bytes() / 4) else {
            failed!();
        };
        // Each index is checked when it's read.
        let header = Header { kind: Kind::Dictionary, ..dictionary.shared(Some(indices)) };
        ColumnView::with_header(context, header, row_count)
    }

    /// `row_count` integers from `first`, one more each row, such as ids:
    /// with no values, so they cost nothing to read, and their bounds known.
    pub fn sequence(
        context: &mut Context,
        first: i64,
        row_count: u32,
    ) -> Result<ColumnView, AllocError> {
        let mut values = context.small_buffer(8)?;
        *at_mut!(values.as_mut_slice::<i64>(), 0) = first;
        let last = first.checked_add(i64::from(row_count) - 1);
        let bounds = match last {
            Some(last) if row_count > 0 => Bounds { min: first, max: last },
            _ => UNKNOWN,
        };
        let header = Header {
            data_type: DataType::Int64,
            kind: Kind::Sequence,
            values,
            bytes: None,
            bytes_len: 0,
            others: None,
            validity: None,
            indices: None,
            value_start: 0,
            len: row_count,
            bounds: Cell::new(bounds),
        };
        ColumnView::with_header(context, header, row_count)
    }

    /// `row_count` rows of `data_type` whose values aren't read yet: a copy
    /// of `handle`, which says where they are, for whoever wrote it to load
    /// them.
    pub fn lazy(
        context: &mut Context,
        data_type: DataType,
        handle: &[u8],
        row_count: u32,
    ) -> Result<ColumnView, AllocError> {
        let Ok(len) = u32::try_from(handle.len()) else { return Err(AllocError) };
        let mut values = context.small_buffer(handle.len())?;
        values.as_mut_slice::<u8>().copy_from_slice(handle);
        let header = Header {
            data_type,
            kind: Kind::Lazy,
            values,
            bytes: None,
            bytes_len: 0,
            others: None,
            validity: None,
            indices: None,
            value_start: 0,
            len,
            bounds: Cell::new(UNKNOWN),
        };
        ColumnView::with_header(context, header, row_count)
    }

    /// Whether the view's values aren't read yet. Nothing but loading them
    /// reads a lazy view.
    pub fn is_lazy(&self) -> bool {
        self.header().kind == Kind::Lazy
    }

    /// A lazy view's handle, and where its rows start among the handle's.
    pub fn handle(&self) -> (&[u8], u32) {
        let header = self.header();
        check!(header.kind == Kind::Lazy);
        (header.values.as_slice::<u8>(), self.start)
    }

    /// A header for a new view of this one's values: those of its rows, for
    /// a flat one, or all of them.
    fn shared(&self, indices: Option<Buffer>) -> Header {
        let header = self.header();
        let values = self.values();
        Header {
            data_type: header.data_type,
            kind: header.kind,
            values: header.values.clone(),
            bytes: header.bytes.clone(),
            bytes_len: header.bytes_len,
            others: header.others.clone(),
            validity: header.validity.clone(),
            indices,
            value_start: values.start,
            len: values.len,
            bounds: Cell::new(header.bounds.get()),
        }
    }

    /// A view of all `row_count` rows of `header`, written into a column
    /// buffer.
    fn with_header(
        context: &mut Context,
        header: Header,
        row_count: u32,
    ) -> Result<ColumnView, AllocError> {
        let mut block = context.small_buffer(size_of::<Header>())?;
        // SAFETY: the block is the size of a `Header`, aligned for any, and
        // only this view has it; what it held was a header already dropped,
        // or nothing.
        unsafe { block.as_mut_non_null().cast::<Header>().write(header) };
        Ok(ColumnView { header: block, start: 0, row_count })
    }

    fn header(&self) -> &Header {
        // SAFETY: a `Header` was written in the block when the view was made,
        // and is dropped only with the last view.
        unsafe { header_in(&self.header).as_ref() }
    }

    /// This view, said to have every value that isn't null within `bounds`,
    /// which whoever says so must be sure of. Views made from it keep them.
    pub fn with_bounds(self, bounds: Bounds) -> ColumnView {
        check!(self.data_type() == DataType::Int64 && bounds.min <= bounds.max);
        check!(!self.is_lazy());
        self.header().bounds.set(bounds);
        self
    }

    /// Bounds every value that isn't null is within, if known.
    pub fn bounds(&self) -> Option<Bounds> {
        let bounds = self.header().bounds.get();
        (bounds.min <= bounds.max).then_some(bounds)
    }

    /// How rows map to `values`.
    pub fn form(&self) -> Form<'_> {
        let header = self.header();
        match header.kind {
            Kind::Flat => Form::Flat,
            Kind::Constant => Form::Constant,
            Kind::Dictionary => {
                let Some(indices) = &header.indices else { failed!() };
                let start = self.start as usize;
                Form::Dictionary(at!(
                    indices.as_slice::<u32>(),
                    start..start + self.row_count as usize
                ))
            }
            Kind::Sequence => {
                let first = *at!(header.values.as_slice::<i64>(), 0);
                Form::Sequence(first.wrapping_add(i64::from(self.start)))
            }
            Kind::Lazy => failed!(),
        }
    }

    /// The values rows map to: for a flat view, its rows' own. A sequence
    /// has none.
    pub fn values(&self) -> Values<'_> {
        let header = self.header();
        match header.kind {
            Kind::Flat => {
                Values { header, start: header.value_start + self.start, len: self.row_count }
            }
            Kind::Constant | Kind::Dictionary => {
                Values { header, start: header.value_start, len: header.len }
            }
            Kind::Lazy | Kind::Sequence => failed!(),
        }
    }

    /// Makes this view one of `forms`, which holds `FLAT`: as it is if its
    /// form is one, or else flat, copying values. A lazy view can't be made
    /// flat without loading, so `forms` must hold `LAZY` for it.
    #[inline]
    pub fn make_in(&mut self, context: &mut Context, forms: Forms) -> Result<(), AllocError> {
        if !self.is_in(forms) {
            *self = self.flatten(context)?;
        }
        Ok(())
    }

    /// Whether this view's values are `other`'s: the same values of the same
    /// buffer, as two batches of a dictionary column from one chunk share.
    pub fn shares_values(&self, other: &ColumnView) -> bool {
        let (mine, theirs) = (self.header(), other.header());
        core::ptr::eq(mine.values.as_ptr::<u8>(), theirs.values.as_ptr::<u8>())
            && (mine.value_start, mine.len) == (theirs.value_start, theirs.len)
    }

    /// Whether this view's form is one of `forms`.
    #[inline]
    pub fn is_in(&self, forms: Forms) -> bool {
        forms.contains(self.forms())
    }

    /// The form this view is in, as a set of one.
    pub fn forms(&self) -> Forms {
        Forms(self.header().kind as u8)
    }

    /// A flat copy of this view. Out of line, so `make_in` stays small
    /// where, as usually, the view is in a form allowed.
    #[cold]
    #[inline(never)]
    fn flatten(&self, context: &mut Context) -> Result<ColumnView, AllocError> {
        let form = self.form();
        if let Form::Sequence(first) = form {
            let mut out = context.values_buffer(self.row_count as usize * 8)?;
            for (value, row) in out.as_mut_slice::<i64>().iter_mut().zip(0..) {
                *value = first.wrapping_add(row);
            }
            let flat = ColumnView::new(context, DataType::Int64, out, None)?;
            flat.header().bounds.set(self.header().bounds.get());
            return Ok(flat);
        }
        let index = |row: u32| match form {
            Form::Dictionary(indices) => *at!(indices, row as usize),
            Form::Flat | Form::Constant | Form::Sequence(_) => 0,
        };
        let (rows, values) = (self.row_count as usize, self.values());
        let validity = match values.validity() {
            None => None,
            Some(valid) => {
                let mut bits = context.small_buffer(BitSlice::<u8>::words_for(rows))?;
                let out = bits.as_mut_slice::<u8>();
                out.fill(0);
                for row in 0..self.row_count {
                    let i = index(row);
                    check!(i < values.len);
                    // SAFETY: `i` is one of the values, checked just above.
                    let valid = unsafe { valid.is_valid_unchecked(i) };
                    BitSlice::of_mut(&mut *out).set_if(row as usize, valid);
                }
                Some(bits)
            }
        };
        // A string is a word too: the words are gathered, and keep their
        // bytes, so no string is copied.
        let words = values.slice_of::<i64>();
        let mut out = context.values_buffer(rows * 8)?;
        let gathered = at_mut!(out.as_mut_slice::<i64>(), ..rows);
        match form {
            Form::Dictionary(indices) => gather(gathered, words, at!(indices, ..rows)),
            Form::Flat | Form::Constant | Form::Sequence(_) => gathered.fill(*at!(words, 0)),
        }
        if values.header.data_type == DataType::String {
            let strings = values.header.strings();
            // SAFETY: each word is one of the view's, valid when it was made.
            return unsafe { ColumnView::strings_in(context, out, strings, validity) };
        }
        let flat = ColumnView::new(context, self.data_type(), out, validity)?;
        flat.header().bounds.set(self.header().bounds.get());
        Ok(flat)
    }

    /// The strings of a flat view.
    pub fn string_values(&self) -> Strings<'_> {
        self.check_flat();
        self.values().strings()
    }

    pub fn data_type(&self) -> DataType {
        self.header().data_type
    }

    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    /// Rows `start..start + row_count` of this view.
    pub fn slice(&self, start: u32, row_count: u32) -> ColumnView {
        check!(start.checked_add(row_count).is_some_and(|end| end <= self.row_count));
        ColumnView { header: self.header.clone(), start: self.start + start, row_count }
    }

    /// The integers of a flat view.
    pub fn int64s(&self) -> &[i64] {
        self.check_flat();
        self.values().int64s()
    }

    /// The floats of a flat view.
    pub fn float64s(&self) -> &[f64] {
        self.check_flat();
        self.values().float64s()
    }

    /// Whether any row may be null.
    pub fn has_nulls(&self) -> bool {
        check!(!self.is_lazy());
        self.header().validity.is_some()
    }

    /// Which rows of a flat view aren't null, or `None` if none are, for
    /// reading in a loop.
    pub fn validity(&self) -> Option<Validity<'_>> {
        self.check_flat();
        self.values().validity()
    }

    pub fn is_null(&self, row: u32) -> bool {
        check!(row < self.row_count);
        let value = match self.form() {
            Form::Flat => row,
            Form::Constant => 0,
            Form::Dictionary(indices) => *at!(indices, row as usize),
            Form::Sequence(_) => return false,
        };
        self.values().is_null(value)
    }

    /// A flat view's values, as bytes.
    pub(crate) fn value_bytes(&self) -> &[u8] {
        self.check_flat();
        self.values().slice_of::<u8>()
    }

    fn check_flat(&self) {
        check!(self.header().kind == Kind::Flat);
    }
}

impl Drop for ColumnView {
    fn drop(&mut self) {
        if self.header.is_unique() {
            // SAFETY: this is the header's last view, so its fields are
            // dropped once, before its block is let go.
            unsafe { core::ptr::drop_in_place(header_in(&self.header).as_ptr()) }
        }
    }
}

/// Where the `Header` in `block` is: buffers are aligned for any.
fn header_in(block: &Buffer) -> NonNull<Header> {
    block.as_non_null().cast()
}

/// Writes the values at `indices` to `out`: checked together, so each row's
/// gather needn't be.
pub(crate) fn gather<T: Copy>(out: &mut [T], values: &[T], indices: &[u32]) {
    let max = indices.iter().fold(0, |max, &i| max.max(i));
    check!(indices.is_empty() || (max as usize) < values.len());
    for (out, &i) in out.iter_mut().zip(indices) {
        // SAFETY: every index is below `values.len()`, checked above.
        *out = unsafe { *values.get_unchecked(i as usize) };
    }
}

impl<'a> Values<'a> {
    pub fn len(self) -> u32 {
        self.len
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    pub fn int64s(self) -> &'a [i64] {
        check!(self.header.data_type == DataType::Int64);
        self.slice_of()
    }

    pub fn float64s(self) -> &'a [f64] {
        check!(self.header.data_type == DataType::Float64);
        self.slice_of()
    }

    /// The values of a column of integers or floats, as words: integers, or
    /// floats' bits.
    pub(crate) fn words(self) -> &'a [i64] {
        check!(!self.header.data_type.has_bytes());
        self.slice_of()
    }

    pub fn strings(self) -> Strings<'a> {
        check!(self.header.data_type == DataType::String);
        let bytes = self.header.own_bytes();
        let others = self.header.others.as_ref().map_or(&[][..], Others::slots);
        Strings { words: self.slice_of::<u64>(), bytes, others }
    }

    /// Which values aren't null, or `None` if none are, for reading in a loop.
    pub fn validity(self) -> Option<Validity<'a>> {
        let bits = self.header.validity.as_ref()?.as_slice::<u8>();
        Some(Validity { bits, start: self.start, row_count: self.len })
    }

    pub fn is_null(self, value: u32) -> bool {
        check!(value < self.len);
        let Some(validity) = &self.header.validity else { return false };
        let bit = (self.start + value) as usize;
        !BitSlice::of(validity.as_slice::<u8>()).get(bit)
    }

    fn slice_of<T: Primitive>(self) -> &'a [T] {
        let width = self.header.data_type.width_bytes() / size_of::<T>();
        let (start, len) = (self.start as usize * width, self.len as usize * width);
        at!(self.header.values.as_slice::<T>(), start..start + len)
    }
}

/// A string column's values: words into bytes, each where a string's length
/// is, its bytes after it.
#[derive(Clone, Copy)]
pub struct Strings<'a> {
    /// Where each value's length is, in its low half, in its column's
    /// `bytes` or, if its high half isn't 0, in that of `others` less one.
    words: &'a [u64],
    /// What the words whose high half is 0 are into.
    bytes: &'a [u8],
    /// The other buffers the column's words point into.
    others: &'a [Slot],
}

impl<'a> Strings<'a> {
    pub fn len(self) -> usize {
        self.words.len()
    }

    pub fn is_empty(self) -> bool {
        self.words.is_empty()
    }

    /// The length of string `row`.
    #[inline]
    pub fn len_of(self, row: usize) -> usize {
        let word = *at!(self.words, row);
        // SAFETY: the column's words were valid when it was made.
        unsafe { length_at(self.bytes_of(word), word) }
    }

    /// As `len_of`, without checking `row` is one of the strings.
    ///
    /// # Safety
    ///
    /// `row` must be below `len()`.
    #[inline]
    pub unsafe fn len_of_unchecked(self, row: usize) -> usize {
        // SAFETY: guaranteed by the caller.
        let word = unsafe { *self.words.get_unchecked(row) };
        // SAFETY: the column's words were valid when it was made.
        unsafe { length_at(self.bytes_of(word), word) }
    }

    #[inline]
    pub fn get(self, row: usize) -> &'a [u8] {
        let word = *at!(self.words, row);
        // SAFETY: the column's words were valid when it was made.
        unsafe { string_at(self.bytes_of(word), word) }
    }

    /// What the strings whose words' high half is 0 are in: the column's
    /// own bytes.
    pub fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// The string at `word`, one of the column's.
    pub(crate) fn of(self, word: u64) -> &'a [u8] {
        // SAFETY: `word` is one of the column's, valid when it was made.
        unsafe { string_at(self.bytes_of(word), word) }
    }

    /// What `word` points into.
    #[inline]
    fn bytes_of(self, word: u64) -> &'a [u8] {
        match (word >> 32) as u32 {
            0 => self.bytes,
            index => at!(self.others, index as usize - 1).bytes(),
        }
    }

    /// The strings, if every one is in its column's own bytes, as a scan's
    /// are: read without asking which buffer each is in.
    #[inline]
    pub(crate) fn own(self) -> Option<OwnStrings<'a>> {
        self.others.is_empty().then_some(OwnStrings { words: self.words, bytes: self.bytes })
    }
}

/// The length of the string at `word` of `bytes`, unchecked, as filters
/// read many.
///
/// # Safety
///
/// `word` must be valid, as `ColumnView::strings_in` says.
#[inline]
unsafe fn length_at(bytes: &[u8], word: u64) -> usize {
    #[expect(clippy::cast_possible_truncation, reason = "an offset is its low half")]
    let at = word as u32 as usize;
    // SAFETY: the 4 bytes from `at` are in `bytes`, as the caller promises.
    let length = unsafe { bytes.as_ptr().add(at).cast::<[u8; 4]>().read_unaligned() };
    u32::from_le_bytes(length) as usize
}

/// Whether each of `words` is where a string is in `bytes`, its length and
/// then its bytes, with nothing in its high half, as strings are read
/// without checks.
pub fn strings_valid(words: &[u64], bytes: &[u8]) -> bool {
    words.iter().all(|&word| {
        let Ok(at) = u32::try_from(word) else { return false };
        let at = at as usize;
        let Some(&length) = bytes.get(at..).and_then(<[u8]>::first_chunk::<4>) else {
            return false;
        };
        let end = (at + 4).checked_add(u32::from_le_bytes(length) as usize);
        end.is_some_and(|end| end <= bytes.len())
    })
}

/// Words, and the bytes where each one's string is, as `string_buffers`
/// writes them: valid, so read without checks.
pub(crate) struct Built(Buffer, Buffer);

/// The words and bytes of `count` strings, string `i` being `string(i)`, in
/// buffers from `context`. Every empty string is one, first in the bytes.
pub(crate) fn string_buffers<'s>(
    context: &mut Context,
    count: usize,
    string: impl Fn(usize) -> &'s [u8],
) -> Result<Built, AllocError> {
    let total = (0..count).fold(4, |total, i| {
        let len = string(i).len();
        total + if len == 0 { 0 } else { 4 + len }
    });
    // So no word's offset is past 32 bits.
    if u32::try_from(total).is_err() {
        return Err(AllocError);
    }
    let mut bytes = context.bytes_buffer(total)?;
    let mut words = if count > BATCH_ROWS_MAX as usize {
        context.bytes_buffer(count * 8)?
    } else {
        context.values_buffer(count * 8)?
    };
    let out = bytes.as_mut_slice::<u8>();
    at_mut!(out, ..4).fill(0);
    let mut at = 4;
    for (i, word) in words.as_mut_slice::<u64>().iter_mut().enumerate() {
        let value = string(i);
        *word = 0;
        if !value.is_empty() {
            #[expect(clippy::cast_possible_truncation, reason = "below `total`")]
            let len = value.len() as u32;
            at_mut!(out, at..at + 4).copy_from_slice(&len.to_le_bytes());
            at_mut!(out, at + 4..at + 4 + value.len()).copy_from_slice(value);
            *word = at as u64;
            at += 4 + value.len();
        }
    }
    Ok(Built(words, bytes))
}

/// The string at `word` of `bytes`, unchecked.
///
/// # Safety
///
/// `word` must be valid, as `ColumnView::strings_in` says.
#[inline]
pub(crate) unsafe fn string_at(bytes: &[u8], word: u64) -> &[u8] {
    #[expect(clippy::cast_possible_truncation, reason = "an offset is its low half")]
    let first = word as u32 as usize + 4;
    // SAFETY: the caller promises the length and the bytes after it are in
    // `bytes`.
    unsafe { core::slice::from_raw_parts(bytes.as_ptr().add(first), length_at(bytes, word)) }
}

/// A string column's values that are all in its own bytes.
#[derive(Clone, Copy)]
pub(crate) struct OwnStrings<'a> {
    words: &'a [u64],
    bytes: &'a [u8],
}

impl<'a> OwnStrings<'a> {
    /// The length of string `row`, without checking `row` is one.
    ///
    /// # Safety
    ///
    /// `row` must be below the number of strings.
    #[inline]
    pub(crate) unsafe fn len_of_unchecked(self, row: usize) -> usize {
        // SAFETY: guaranteed by the caller.
        // SAFETY: as the caller promises, and the column's words were valid
        // when it was made.
        unsafe { length_at(self.bytes, *self.words.get_unchecked(row)) }
    }

    #[inline]
    pub(crate) fn get(self, row: usize) -> &'a [u8] {
        // SAFETY: the column's words were valid when it was made.
        unsafe { string_at(self.bytes, *at!(self.words, row)) }
    }
}

/// A view's bitmap of the rows that aren't null.
#[derive(Clone, Copy)]
pub struct Validity<'a> {
    bits: &'a [u8],
    start: u32,
    row_count: u32,
}

impl<'a> Validity<'a> {
    pub fn row_count(self) -> u32 {
        self.row_count
    }

    /// The bitmap from the byte holding the first row's bit, and which bit
    /// of it that is.
    pub(crate) fn bits(self) -> (&'a [u8], u32) {
        (at!(self.bits, self.start as usize / 8..), self.start % 8)
    }

    /// Whether `row` isn't null, without checking it's a row.
    ///
    /// # Safety
    ///
    /// `row` must be below `row_count`.
    #[inline]
    pub unsafe fn is_valid_unchecked(self, row: u32) -> bool {
        let bit = (self.start + row) as usize;
        // SAFETY: `ColumnView::new` checked the bitmap covers every row, and
        // `slice` keeps rows within it.
        unsafe { BitSlice::of(self.bits).get_unchecked(bit) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::context::Context;
    use crate::env::Env;

    #[test]
    fn reads_values_and_nulls() {
        let env = Env::current();
        let mut values = Buffer::allocate(env, 4 * 8).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(&[1, 2, 3, 4]);
        let mut validity = Buffer::allocate(env, 1).unwrap();
        validity.as_mut_slice::<u8>()[0] = 0b1011;

        let column =
            ColumnView::new(&mut Context::new(env, None), DataType::Int64, values, Some(validity))
                .unwrap();
        assert_eq!(column.int64s(), [1, 2, 3, 4]);
        assert!(column.is_null(2));
        assert!(!column.is_null(3));

        let slice = column.slice(1, 2);
        assert_eq!(slice.int64s(), [2, 3]);
        assert!(slice.is_null(1));
    }

    #[test]
    #[should_panic(expected = "Int64")]
    fn reading_the_wrong_type_panics() {
        let env = Env::current();
        let values = Buffer::allocate(env, 8).unwrap();
        ColumnView::new(&mut Context::new(env, None), DataType::Float64, values, None)
            .unwrap()
            .int64s();
    }

    /// `values`, one after another in a buffer, as strings.
    fn strings(values: &[&str]) -> ColumnView {
        let value = |i: usize| Some(values[i].as_bytes());
        ColumnView::from_strings(&mut Context::new(Env::current(), None), values.len(), value)
            .unwrap()
    }

    #[test]
    fn strings_are_bytes() {
        // Not UTF-8: a lone continuation byte, and an overlong encoding.
        let values: [&[u8]; 2] = [&[0x80], &[0xc0, 0x80]];
        let context = &mut Context::new(Env::current(), None);
        let column = ColumnView::from_strings(context, 2, |i| Some(values[i])).unwrap();
        let values = column.string_values();
        assert_eq!((values.get(0), values.get(1)), (&[0x80][..], &[0xc0, 0x80][..]));
    }

    #[test]
    fn reads_strings_and_slices_of_them() {
        let column = strings(&["ab", "", "cde", "f"]);
        let values = column.string_values();
        assert_eq!(
            (values.len(), values.get(0), values.get(1), values.get(2)),
            (4, &b"ab"[..], &b""[..], &b"cde"[..])
        );

        let sliced = column.slice(2, 2);
        let slice = sliced.string_values();
        assert_eq!((slice.get(0), slice.get(1)), (&b"cde"[..], &b"f"[..]));
    }

    /// `words` into `bytes`, as strings.
    fn words(bytes: &[u8], words: &[u64], validity: Option<u8>) -> ColumnView {
        let env = Env::current();
        let mut page = Buffer::allocate(env, bytes.len()).unwrap();
        page.as_mut_slice::<u8>().copy_from_slice(bytes);
        let mut out = Buffer::allocate(env, words.len() * 8).unwrap();
        out.as_mut_slice::<u64>().copy_from_slice(words);
        let validity = validity.map(|bits| {
            let mut buffer = Buffer::allocate(env, 1).unwrap();
            buffer.as_mut_slice::<u8>()[0] = bits;
            buffer
        });
        ColumnView::strings(&mut Context::new(env, None), out, page, validity).unwrap()
    }

    #[test]
    fn reads_string_words_and_slices_of_them() {
        // Out of order, sharing a string, between other bytes, and a null,
        // which is the empty string.
        let page = b"..\x05\0\0\0hello.\x03\0\0\0llo\0\0\0\0";
        let column = words(page, &[2, 19, 12, 2], Some(0b1101));
        let values = column.string_values();
        assert_eq!(
            (values.len(), values.get(0), values.get(1), values.get(2), values.get(3)),
            (4, &b"hello"[..], &b""[..], &b"llo"[..], &b"hello"[..])
        );
        assert!(column.is_null(1));
        let sliced = column.slice(2, 2);
        let slice = sliced.string_values();
        assert_eq!((slice.len(), slice.get(0), slice.get(1)), (2, &b"llo"[..], &b"hello"[..]));
    }

    #[test]
    #[should_panic(expected = "has_bytes")]
    fn strings_need_their_bytes() {
        let env = Env::current();
        let _ = ColumnView::new(
            &mut Context::new(env, None),
            DataType::String,
            Buffer::allocate(env, 8).unwrap(),
            None,
        )
        .unwrap();
    }

    fn int64s(values: &[i64], nulls: &[usize]) -> ColumnView {
        let env = Env::current();
        let mut buffer = Buffer::allocate(env, values.len() * 8).unwrap();
        buffer.as_mut_slice::<i64>().copy_from_slice(values);
        let validity = (!nulls.is_empty()).then(|| {
            let mut bits = Buffer::allocate(env, BitSlice::<u8>::words_for(values.len())).unwrap();
            bits.as_mut_slice::<u8>().fill(0xff);
            for &row in nulls {
                BitSlice::of_mut(bits.as_mut_slice::<u8>()).clear(row);
            }
            bits
        });
        ColumnView::new(&mut Context::new(env, None), DataType::Int64, buffer, validity).unwrap()
    }

    fn indices(indices: &[u32]) -> Buffer {
        let env = Env::current();
        let mut buffer = Buffer::allocate(env, indices.len() * 4).unwrap();
        buffer.as_mut_slice::<u32>().copy_from_slice(indices);
        buffer
    }

    #[test]
    fn maps_rows_to_values_in_each_form() {
        let env = Env::current();
        let mut context = crate::context::Context::new(env, None);
        // A dictionary of 10, 20 and a null, as Parquet's nulls point at one.
        let bounds = Bounds { min: 10, max: 20 };
        let dictionary = int64s(&[10, 20, 0], &[2]).with_bounds(bounds);
        let column = ColumnView::dictionary(
            &mut Context::new(env, None),
            &dictionary,
            indices(&[1, 0, 2, 1, 1]),
        )
        .unwrap();
        // Bounds hold for whatever's made of a column.
        assert_eq!(column.bounds(), Some(bounds));
        assert_eq!(column.slice(1, 2).flatten(&mut context).unwrap().bounds(), Some(bounds));
        assert_eq!(
            ColumnView::constant(&mut Context::new(env, None), &dictionary.slice(0, 1), 3)
                .unwrap()
                .bounds(),
            Some(bounds)
        );
        assert!(matches!(column.form(), Form::Dictionary([1, 0, 2, 1, 1])));
        assert_eq!(column.row_count(), 5);
        assert_eq!(column.values().int64s(), [10, 20, 0]);
        assert_eq!((column.is_null(2), column.is_null(3)), (true, false));
        let flat = column.flatten(&mut context).unwrap();
        assert_eq!(flat.int64s(), [20, 10, 0, 20, 20]);
        assert_eq!((flat.is_null(2), flat.is_null(0)), (true, false));
        let sliced = column.slice(1, 3).flatten(&mut context).unwrap();
        assert_eq!(sliced.int64s(), [10, 0, 20]);

        let constant =
            ColumnView::constant(&mut Context::new(env, None), &int64s(&[7], &[]), 4).unwrap();
        assert!(matches!(constant.form(), Form::Constant));
        assert_eq!(constant.slice(1, 2).flatten(&mut context).unwrap().int64s(), [7, 7]);
        let nulls =
            ColumnView::constant(&mut Context::new(env, None), &int64s(&[0], &[0]), 3).unwrap();
        assert!((0..3).all(|row| nulls.is_null(row)));
    }

    #[test]
    fn keeps_a_lazy_views_handle() {
        let env = Env::current();
        let column =
            ColumnView::lazy(&mut Context::new(env, None), DataType::Int64, &[1, 2, 3], 10)
                .unwrap();
        assert!(column.is_lazy());
        assert_eq!(column.handle(), (&[1, 2, 3][..], 0));
        let sliced = column.slice(4, 3);
        assert_eq!((sliced.handle(), sliced.row_count()), ((&[1, 2, 3][..], 4), 3));
        assert!(!int64s(&[1], &[]).is_lazy());
    }

    #[test]
    #[should_panic(expected = "check failed")]
    fn lazy_views_have_no_values() {
        let env = Env::current();
        let column =
            ColumnView::lazy(&mut Context::new(env, None), DataType::Int64, &[1], 2).unwrap();
        let _ = column.values();
    }

    #[test]
    fn flattens_strings() {
        let env = Env::current();
        let mut context = crate::context::Context::new(env, None);
        let dictionary = strings(&["ab", "cde"]);
        let column =
            ColumnView::dictionary(&mut Context::new(env, None), &dictionary, indices(&[1, 1, 0]))
                .unwrap();
        let flat = column.flatten(&mut context).unwrap();
        let strings = flat.string_values();
        assert_eq!(
            (strings.get(0), strings.get(1), strings.get(2)),
            (&b"cde"[..], &b"cde"[..], &b"ab"[..])
        );
    }

    #[test]
    fn batches_borrow_views_without_a_reference() {
        let (a, b) = (int64s(&[1, 2], &[]), int64s(&[3, 4], &[]));
        let mut batch = crate::row_batch::RowBatch::new();
        batch.reset(2);
        batch.push_borrowed([&a, &b]);
        assert!(a.header.is_unique() && b.header.is_unique());
        // A clone holds a reference, so it can outlive the batch.
        let clone = batch.column(1).clone();
        assert!(!b.header.is_unique());
        // Replacing or dropping borrowed columns leaves the views alone.
        batch.set_column(0, int64s(&[5, 6], &[]));
        drop(batch);
        assert!(a.header.is_unique());
        drop(b);
        assert_eq!(clone.int64s(), [3, 4]);
    }
}
