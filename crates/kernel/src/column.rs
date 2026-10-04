//! `ColumnView`: a range of rows over a column's buffers.

use crate::buffer::{Buffer, Primitive};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int64,
    Float64,
    /// Bytes of any length, as Arrow's `Binary`: offsets into a buffer of
    /// them.
    String,
}

impl DataType {
    /// How wide each value is in a column's values buffer: for strings, an
    /// offset.
    pub fn width_bytes(self) -> usize {
        match self {
            DataType::Int64 | DataType::Float64 => 8,
            DataType::String => 4,
        }
    }
}

/// Values in Arrow's layout, and an optional validity bitmap with one bit per
/// row, set if the row is not null. Cloning shares the buffers.
#[derive(Clone)]
pub struct ColumnView {
    data_type: DataType,
    /// For strings, offsets into `bytes`, one more than there are rows.
    values: Buffer,
    bytes: Option<Buffer>,
    validity: Option<Buffer>,
    start: u32,
    row_count: u32,
}

impl ColumnView {
    /// A view of every row in `values`.
    pub fn new(data_type: DataType, values: Buffer, validity: Option<Buffer>) -> ColumnView {
        check!(data_type != DataType::String);
        check!(values.size_bytes().is_multiple_of(data_type.width_bytes()));
        let Ok(row_count) = u32::try_from(values.size_bytes() / data_type.width_bytes()) else {
            crate::check::check_failed(line!());
        };
        if let Some(validity) = &validity {
            check!(validity.size_bytes() >= row_count.div_ceil(8) as usize);
        }
        ColumnView { data_type, values, bytes: None, validity, start: 0, row_count }
    }

    /// Strings: row `i` is `bytes[offsets[i]..offsets[i + 1]]`, so `offsets`
    /// holds one more `u32` than there are rows.
    pub fn strings(offsets: Buffer, bytes: Buffer, validity: Option<Buffer>) -> ColumnView {
        let count = offsets.size_bytes() / 4;
        check!(count >= 1 && offsets.size_bytes().is_multiple_of(4));
        let Ok(row_count) = u32::try_from(count - 1) else {
            crate::check::check_failed(line!());
        };
        // Each row's range is checked when it's read.
        check!(*at!(offsets.as_slice::<u32>(), count - 1) as usize <= bytes.size_bytes());
        if let Some(validity) = &validity {
            check!(validity.size_bytes() >= row_count.div_ceil(8) as usize);
        }
        let bytes = Some(bytes);
        ColumnView {
            data_type: DataType::String,
            values: offsets,
            bytes,
            validity,
            start: 0,
            row_count,
        }
    }

    /// A string column's values.
    pub fn string_values(&self) -> Strings<'_> {
        check!(self.data_type == DataType::String);
        let start = self.start as usize;
        let offsets = at!(self.values.as_slice::<u32>(), start..=start + self.row_count as usize);
        let Some(bytes) = &self.bytes else { crate::check::check_failed(line!()) };
        Strings { offsets, bytes: bytes.as_slice() }
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    /// Rows `start..start + row_count` of this view.
    pub fn slice(&self, start: u32, row_count: u32) -> ColumnView {
        check!(start.checked_add(row_count).is_some_and(|end| end <= self.row_count));
        ColumnView { start: self.start + start, row_count, ..self.clone() }
    }

    pub fn int64s(&self) -> &[i64] {
        check!(self.data_type == DataType::Int64);
        self.values()
    }

    pub fn float64s(&self) -> &[f64] {
        check!(self.data_type == DataType::Float64);
        self.values()
    }

    /// Whether any row may be null.
    pub fn has_nulls(&self) -> bool {
        self.validity.is_some()
    }

    /// Which rows aren't null, or `None` if none are, for reading in a loop.
    pub fn validity(&self) -> Option<Validity<'_>> {
        let bits = self.validity.as_ref()?.as_slice::<u8>();
        Some(Validity { bits, start: self.start, row_count: self.row_count })
    }

    pub fn is_null(&self, row: u32) -> bool {
        check!(row < self.row_count);
        let Some(validity) = &self.validity else { return false };
        let bit = (self.start + row) as usize;
        at!(validity.as_slice::<u8>(), bit / 8) & (1 << (bit % 8)) == 0
    }

    /// The view's values, as bytes.
    pub(crate) fn value_bytes(&self) -> &[u8] {
        let width = self.data_type.width_bytes();
        let start = self.start as usize * width;
        at!(self.values.as_slice::<u8>(), start..start + self.row_count as usize * width)
    }

    fn values<T: Primitive>(&self) -> &[T] {
        let start = self.start as usize;
        at!(self.values.as_slice::<T>(), start..start + self.row_count as usize)
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

    /// Row `row`'s bytes.
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

    #[test]
    fn reads_values_and_nulls() {
        let mut values = Buffer::allocate(&Heap, 4 * 8).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(&[1, 2, 3, 4]);
        let mut validity = Buffer::allocate(&Heap, 1).unwrap();
        validity.as_mut_slice::<u8>()[0] = 0b1011;

        let column = ColumnView::new(DataType::Int64, values, Some(validity));
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
        ColumnView::new(DataType::Float64, values, None).int64s();
    }

    /// `values` as a string column.
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
        ColumnView::strings(offsets, bytes, None)
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
    #[should_panic(expected = "String")]
    fn strings_need_their_bytes() {
        let _ = ColumnView::new(DataType::String, Buffer::allocate(&Heap, 8).unwrap(), None);
    }
}
