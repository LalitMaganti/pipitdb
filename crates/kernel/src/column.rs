//! `ColumnView`: a range of rows over a column's buffers.

use crate::buffer::{Buffer, Primitive};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int64,
    Float64,
}

impl DataType {
    pub fn width_bytes(self) -> usize {
        match self {
            DataType::Int64 | DataType::Float64 => 8,
        }
    }
}

/// Values in Arrow's layout, and an optional validity bitmap with one bit per
/// row, set if the row is not null. Cloning shares the buffers.
#[derive(Clone)]
pub struct ColumnView {
    data_type: DataType,
    values: Buffer,
    validity: Option<Buffer>,
    start: u32,
    row_count: u32,
}

impl ColumnView {
    /// A view of every row in `values`.
    pub fn new(data_type: DataType, values: Buffer, validity: Option<Buffer>) -> ColumnView {
        check!(values.size_bytes().is_multiple_of(data_type.width_bytes()));
        let Ok(row_count) = u32::try_from(values.size_bytes() / data_type.width_bytes()) else {
            crate::check::check_failed(line!());
        };
        if let Some(validity) = &validity {
            check!(validity.size_bytes() >= row_count.div_ceil(8) as usize);
        }
        ColumnView { data_type, values, validity, start: 0, row_count }
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

    fn values<T: Primitive>(&self) -> &[T] {
        let start = self.start as usize;
        at!(self.values.as_slice::<T>(), start..start + self.row_count as usize)
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
}
