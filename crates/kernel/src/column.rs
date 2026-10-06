//! `ColumnView`: a range of rows over a column's buffers.

use core::cell::Cell;
use core::ptr::NonNull;

use crate::allocator::AllocError;
use crate::buffer::{Buffer, Primitive};
use crate::context::Context;

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
    pub fn width_bytes(self) -> usize {
        match self {
            DataType::Int64 | DataType::Float64 => 8,
            DataType::String => 4,
        }
    }

    /// Whether a column's values are offsets into a buffer of bytes, with one
    /// more offset than there are rows.
    pub fn has_offsets(self) -> bool {
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

/// Rows of values in Arrow's layout: values, and an optional validity bitmap
/// with one bit per value, set if it isn't null. As DuckDB's vectors, rows
/// map to values in one of three forms: each to its own (flat), all to one
/// (constant), or each through an index (dictionary).
///
/// It's a handle, small to move: the buffers and what they hold are in a
/// header it shares with its clones and slices.
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
/// `indices`, and a constant one's as many as its views say.
struct Header {
    data_type: DataType,
    kind: Kind,
    values: Buffer,
    bytes: Option<Buffer>,
    validity: Option<Buffer>,
    indices: Option<Buffer>,
    value_start: u32,
    len: u32,
    bounds: Cell<Bounds>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Flat,
    Constant,
    Dictionary,
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
        check!(!data_type.has_offsets());
        check!(values.size_bytes().is_multiple_of(data_type.width_bytes()));
        let Ok(len) = u32::try_from(values.size_bytes() / data_type.width_bytes()) else {
            crate::check::check_failed(line!());
        };
        ColumnView::flat(context, data_type, values, None, validity, len)
    }

    /// Strings: value `i` is `bytes[offsets[i]..offsets[i + 1]]`, so
    /// `offsets` holds one more `u32` than there are values.
    pub fn strings(
        context: &mut Context,
        offsets: Buffer,
        bytes: Buffer,
        validity: Option<Buffer>,
    ) -> Result<ColumnView, AllocError> {
        let count = offsets.size_bytes() / 4;
        check!(count >= 1 && offsets.size_bytes().is_multiple_of(4));
        let Ok(len) = u32::try_from(count - 1) else {
            crate::check::check_failed(line!());
        };
        // Each value's range is checked when it's read.
        check!(*at!(offsets.as_slice::<u32>(), count - 1) as usize <= bytes.size_bytes());
        ColumnView::flat(context, DataType::String, offsets, Some(bytes), validity, len)
    }

    fn flat(
        context: &mut Context,
        data_type: DataType,
        values: Buffer,
        bytes: Option<Buffer>,
        validity: Option<Buffer>,
        len: u32,
    ) -> Result<ColumnView, AllocError> {
        if let Some(validity) = &validity {
            check!(validity.size_bytes() >= len.div_ceil(8) as usize);
        }
        let (indices, bounds) = (None, Cell::new(UNKNOWN));
        let kind = Kind::Flat;
        let header = Header {
            data_type,
            kind,
            values,
            bytes,
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
            crate::check::check_failed(line!());
        };
        // Each index is checked when it's read.
        let header = Header { kind: Kind::Dictionary, ..dictionary.shared(Some(indices)) };
        ColumnView::with_header(context, header, row_count)
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
                let Some(indices) = &header.indices else { crate::check::check_failed(line!()) };
                let start = self.start as usize;
                Form::Dictionary(at!(
                    indices.as_slice::<u32>(),
                    start..start + self.row_count as usize
                ))
            }
        }
    }

    /// The values rows map to: for a flat view, its rows' own.
    pub fn values(&self) -> Values<'_> {
        let header = self.header();
        match header.kind {
            Kind::Flat => {
                Values { header, start: header.value_start + self.start, len: self.row_count }
            }
            Kind::Constant | Kind::Dictionary => {
                Values { header, start: header.value_start, len: header.len }
            }
        }
    }

    /// A flat view of the same rows, copying values unless it's flat already.
    pub fn flatten(&self, context: &mut Context) -> Result<ColumnView, AllocError> {
        let form = self.form();
        if matches!(form, Form::Flat) {
            return Ok(self.clone());
        }
        let index = |row: u32| match form {
            Form::Dictionary(indices) => *at!(indices, row as usize),
            Form::Flat | Form::Constant => 0,
        };
        let (rows, values) = (self.row_count as usize, self.values());
        let validity = match values.validity() {
            None => None,
            Some(valid) => {
                let mut bits = context.small_buffer(rows.div_ceil(8))?;
                let out = bits.as_mut_slice::<u8>();
                out.fill(0);
                for row in 0..self.row_count {
                    let i = index(row);
                    check!(i < values.len);
                    // SAFETY: `i` is one of the values, checked just above.
                    let bit = u8::from(unsafe { valid.is_valid_unchecked(i) });
                    *at_mut!(out, row as usize / 8) |= bit << (row % 8);
                }
                Some(bits)
            }
        };
        if self.data_type().has_offsets() {
            let strings = values.strings();
            let total: usize =
                (0..self.row_count).map(|row| strings.get(index(row) as usize).len()).sum();
            let mut offsets = context.indices_buffer((rows + 1) * 4)?;
            let mut bytes = context.bytes_buffer(total)?;
            let (ends, out) = (offsets.as_mut_slice::<u32>(), bytes.as_mut_slice::<u8>());
            let mut to = 0;
            *at_mut!(ends, 0) = 0;
            for row in 0..self.row_count {
                let value = strings.get(index(row) as usize);
                at_mut!(out, to..to + value.len()).copy_from_slice(value);
                to += value.len();
                let Ok(end) = u32::try_from(to) else { return Err(AllocError) };
                *at_mut!(ends, row as usize + 1) = end;
            }
            return ColumnView::strings(context, offsets, bytes, validity);
        }
        let words = values.slice_of::<i64>();
        let mut out = context.values_buffer(rows * 8)?;
        for (row, word) in (0..).zip(out.as_mut_slice::<i64>()) {
            *word = *at!(words, index(row) as usize);
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

    pub fn strings(self) -> Strings<'a> {
        check!(self.header.data_type == DataType::String);
        let start = self.start as usize;
        let offsets = at!(self.header.values.as_slice::<u32>(), start..=start + self.len as usize);
        let Some(bytes) = &self.header.bytes else { crate::check::check_failed(line!()) };
        Strings { offsets, bytes: bytes.as_slice() }
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
        at!(validity.as_slice::<u8>(), bit / 8) & (1 << (bit % 8)) == 0
    }

    fn slice_of<T: Primitive>(self) -> &'a [T] {
        let width = self.header.data_type.width_bytes() / size_of::<T>();
        let (start, len) = (self.start as usize * width, self.len as usize * width);
        at!(self.header.values.as_slice::<T>(), start..start + len)
    }
}

/// A string column's values: offsets into bytes.
#[derive(Clone, Copy)]
pub struct Strings<'a> {
    offsets: &'a [u32],
    bytes: &'a [u8],
}

impl<'a> Strings<'a> {
    pub fn len(self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn get(self, row: usize) -> &'a [u8] {
        let (start, end) = (*at!(self.offsets, row), *at!(self.offsets, row + 1));
        at!(self.bytes, start as usize..end as usize)
    }

    /// The bytes the rows take, from the first's start to the last's end.
    pub fn bytes(self) -> &'a [u8] {
        let (first, last) = (*at!(self.offsets, 0), *at!(self.offsets, self.offsets.len() - 1));
        at!(self.bytes, first as usize..last as usize)
    }

    /// Each row's start in `bytes`, then the last's end, as offsets into the
    /// whole buffer: subtract the first to have them start at 0.
    pub fn offsets(self) -> &'a [u32] {
        self.offsets
    }
}

/// A view's bitmap of the rows that aren't null.
#[derive(Clone, Copy)]
pub struct Validity<'a> {
    bits: &'a [u8],
    start: u32,
    row_count: u32,
}

impl Validity<'_> {
    pub fn row_count(self) -> u32 {
        self.row_count
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
        (unsafe { *self.bits.get_unchecked(bit / 8) } >> (bit % 8)) & 1 != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;
    use crate::context::Context;

    #[test]
    fn reads_values_and_nulls() {
        let mut values = Buffer::allocate(&Heap, 4 * 8).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(&[1, 2, 3, 4]);
        let mut validity = Buffer::allocate(&Heap, 1).unwrap();
        validity.as_mut_slice::<u8>()[0] = 0b1011;

        let column =
            ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, Some(validity))
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
        let values = Buffer::allocate(&Heap, 8).unwrap();
        ColumnView::new(&mut Context::new(&Heap), DataType::Float64, values, None)
            .unwrap()
            .int64s();
    }

    fn strings(values: &[&str]) -> ColumnView {
        let mut offsets = Buffer::allocate(&Heap, (values.len() + 1) * 4).unwrap();
        let total: usize = values.iter().map(|v| v.len()).sum();
        let mut bytes = Buffer::allocate(&Heap, total.max(1)).unwrap();
        let mut at = 0;
        for (i, value) in values.iter().enumerate() {
            bytes.as_mut_slice::<u8>()[at..at + value.len()].copy_from_slice(value.as_bytes());
            at += value.len();
            offsets.as_mut_slice::<u32>()[i + 1] = u32::try_from(at).unwrap();
        }
        ColumnView::strings(&mut Context::new(&Heap), offsets, bytes, None).unwrap()
    }

    #[test]
    fn strings_are_bytes() {
        // Not UTF-8: a lone continuation byte, and an overlong encoding.
        let mut offsets = Buffer::allocate(&Heap, 3 * 4).unwrap();
        offsets.as_mut_slice::<u32>().copy_from_slice(&[0, 1, 3]);
        let mut bytes = Buffer::allocate(&Heap, 3).unwrap();
        bytes.as_mut_slice::<u8>().copy_from_slice(&[0x80, 0xc0, 0x80]);
        let column = ColumnView::strings(&mut Context::new(&Heap), offsets, bytes, None).unwrap();
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
        assert_eq!((slice.bytes(), slice.offsets()), (&b"cdef"[..], &[2, 5, 6][..]));
    }

    #[test]
    #[should_panic(expected = "has_offsets")]
    fn strings_need_their_bytes() {
        let _ = ColumnView::new(
            &mut Context::new(&Heap),
            DataType::String,
            Buffer::allocate(&Heap, 8).unwrap(),
            None,
        )
        .unwrap();
    }

    fn int64s(values: &[i64], nulls: &[usize]) -> ColumnView {
        let mut buffer = Buffer::allocate(&Heap, values.len() * 8).unwrap();
        buffer.as_mut_slice::<i64>().copy_from_slice(values);
        let validity = (!nulls.is_empty()).then(|| {
            let mut bits = Buffer::allocate(&Heap, values.len().div_ceil(8)).unwrap();
            bits.as_mut_slice::<u8>().fill(0xff);
            for &row in nulls {
                bits.as_mut_slice::<u8>()[row / 8] &= !(1 << (row % 8));
            }
            bits
        });
        ColumnView::new(&mut Context::new(&Heap), DataType::Int64, buffer, validity).unwrap()
    }

    fn indices(indices: &[u32]) -> Buffer {
        let mut buffer = Buffer::allocate(&Heap, indices.len() * 4).unwrap();
        buffer.as_mut_slice::<u32>().copy_from_slice(indices);
        buffer
    }

    #[test]
    fn maps_rows_to_values_in_each_form() {
        let mut context = crate::context::Context::new(&Heap);
        // A dictionary of 10, 20 and a null, as Parquet's nulls point at one.
        let bounds = Bounds { min: 10, max: 20 };
        let dictionary = int64s(&[10, 20, 0], &[2]).with_bounds(bounds);
        let column = ColumnView::dictionary(
            &mut Context::new(&Heap),
            &dictionary,
            indices(&[1, 0, 2, 1, 1]),
        )
        .unwrap();
        // Bounds hold for whatever's made of a column.
        assert_eq!(column.bounds(), Some(bounds));
        assert_eq!(column.slice(1, 2).flatten(&mut context).unwrap().bounds(), Some(bounds));
        assert_eq!(
            ColumnView::constant(&mut Context::new(&Heap), &dictionary.slice(0, 1), 3)
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
            ColumnView::constant(&mut Context::new(&Heap), &int64s(&[7], &[]), 4).unwrap();
        assert!(matches!(constant.form(), Form::Constant));
        assert_eq!(constant.slice(1, 2).flatten(&mut context).unwrap().int64s(), [7, 7]);
        let nulls = ColumnView::constant(&mut Context::new(&Heap), &int64s(&[0], &[0]), 3).unwrap();
        assert!((0..3).all(|row| nulls.is_null(row)));
    }

    #[test]
    fn flattens_strings() {
        let mut context = crate::context::Context::new(&Heap);
        let mut offsets = Buffer::allocate(&Heap, 3 * 4).unwrap();
        offsets.as_mut_slice::<u32>().copy_from_slice(&[0, 2, 5]);
        let mut bytes = Buffer::allocate(&Heap, 5).unwrap();
        bytes.as_mut_slice::<u8>().copy_from_slice(b"abcde");
        let dictionary =
            ColumnView::strings(&mut Context::new(&Heap), offsets, bytes, None).unwrap();
        let column =
            ColumnView::dictionary(&mut Context::new(&Heap), &dictionary, indices(&[1, 1, 0]))
                .unwrap();
        let flat = column.flatten(&mut context).unwrap();
        let strings = flat.string_values();
        assert_eq!(
            (strings.get(0), strings.get(1), strings.get(2)),
            (&b"cde"[..], &b"cde"[..], &b"ab"[..])
        );
    }
}
