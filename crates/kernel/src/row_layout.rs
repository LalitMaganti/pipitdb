//! `RowLayout`: rows' sort keys laid out in fixed-width slots whose bytes
//! compare as their values do, as Perfetto's does, so a sort compares rows
//! as bytes. After Arrow's row format:
//! <https://arrow.apache.org/blog/2022/11/07/multi-column-sorts-in-arrow-rust-part-2/>

use crate::allocator::{AllocError, Allocator};
use crate::column::{ColumnView, DataType};
use crate::slow_vec::SlowVec;

/// A key column, as a layout needs to know it.
#[derive(Clone, Copy)]
pub(crate) struct KeyColumn {
    pub(crate) data_type: DataType,
    pub(crate) nullable: bool,
    pub(crate) descending: bool,
}

/// Where a key column's bytes are in a row.
#[derive(Clone, Copy)]
struct Slot {
    offset: u16,
    data_type: DataType,
    nullable: bool,
    descending: bool,
}

/// A slot for each key column, one after another in each row. A nullable
/// slot starts with 0xFF if the row has a value and 0 if not, so nulls sort
/// first ascending and last descending, as in SQLite. A descending slot's
/// bits are inverted.
pub(crate) struct RowLayout {
    slots: SlowVec<Slot>,
    stride: u16,
}

impl RowLayout {
    pub(crate) fn new(
        allocator: &dyn Allocator,
        columns: &[KeyColumn],
    ) -> Result<RowLayout, AllocError> {
        let mut slots = SlowVec::fixed(allocator, columns.len())?;
        let mut stride = 0_usize;
        for column in columns {
            check!(!column.data_type.has_bytes());
            let Ok(offset) = u16::try_from(stride) else { return Err(AllocError) };
            let KeyColumn { data_type, nullable, descending } = *column;
            slots.push(Slot { offset, data_type, nullable, descending })?;
            stride += usize::from(nullable) + 8;
        }
        let Ok(stride) = u16::try_from(stride) else { return Err(AllocError) };
        Ok(RowLayout { slots, stride })
    }

    /// How many bytes a row's keys take.
    pub(crate) fn stride(&self) -> usize {
        usize::from(self.stride)
    }

    /// Writes key `k` of the rows `select` keeps of `column`, all of them if
    /// empty, to the rows of `out`, one after another.
    pub(crate) fn write(&self, k: usize, column: &ColumnView, select: &[u16], out: &mut [u8]) {
        let slot = *at!(self.slots, k);
        let stride = self.stride();
        let words = words(column);
        let validity = column.validity();
        let rows = if select.is_empty() { column.row_count() as usize } else { select.len() };
        check!(out.len() >= rows * stride);
        let invert = if slot.descending { u8::MAX } else { 0 };
        for (i, row) in out.chunks_exact_mut(stride).take(rows).enumerate() {
            let at = if select.is_empty() { i } else { usize::from(*at!(select, i)) };
            let to = at_mut!(row, usize::from(slot.offset)..);
            #[expect(clippy::cast_possible_truncation, reason = "a batch's row")]
            let row32 = at as u32;
            let value = match validity {
                // SAFETY: `at` is one of the column's rows.
                Some(validity) if !unsafe { validity.is_valid_unchecked(row32) } => None,
                _ => Some(*at!(words, at)),
            };
            let to = if slot.nullable {
                *at_mut!(to, 0) = u8::from(value.is_some()).wrapping_neg() ^ invert;
                at_mut!(to, 1..)
            } else {
                to
            };
            let bytes = value.map_or([0; 8], |word| encode(slot.data_type, word));
            for (to, byte) in at_mut!(to, ..8).iter_mut().zip(bytes) {
                *to = byte ^ invert;
            }
        }
    }
}

/// `word`, an integer or a float's bits, as bytes that compare as values of
/// `data_type` do: big-endian, with the sign bit flipped, and a negative
/// float's other bits too, so they order backwards.
fn encode(data_type: DataType, word: i64) -> [u8; 8] {
    let bits = word.cast_unsigned();
    let bits = match data_type {
        DataType::Float64 if bits >> 63 == 1 => !bits,
        _ => bits ^ 1 << 63,
    };
    bits.to_be_bytes()
}

/// A flat column's values as words: integers, or floats' bits.
fn words(column: &ColumnView) -> &[i64] {
    // SAFETY: any bits are an `i64`.
    let (before, words, after) = unsafe { column.value_bytes().align_to::<i64>() };
    check!(before.is_empty() && after.is_empty());
    words
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::context::Context;

    fn column(data_type: DataType, words: &[i64], nulls: &[usize]) -> ColumnView {
        let mut values = Buffer::allocate(&Heap, words.len() * 8).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(words);
        let validity = (!nulls.is_empty()).then(|| {
            let mut bits = Buffer::allocate(&Heap, words.len().div_ceil(8)).unwrap();
            bits.as_mut_slice::<u8>().fill(0xff);
            for &row in nulls {
                bits.as_mut_slice::<u8>()[row / 8] &= !(1 << (row % 8));
            }
            bits
        });
        ColumnView::new(&mut Context::new(&Heap), data_type, values, validity).unwrap()
    }

    /// The order of `column`'s rows by their encoded keys.
    fn sorted(column: &ColumnView, nullable: bool, descending: bool) -> Vec<usize> {
        let key = KeyColumn { data_type: column.data_type(), nullable, descending };
        let layout = RowLayout::new(&Heap, &[key]).unwrap();
        let rows = column.row_count() as usize;
        let mut out = alloc::vec![0_u8; rows * layout.stride()];
        layout.write(0, column, &[], &mut out);
        let mut order: Vec<usize> = (0..rows).collect();
        order.sort_by_key(|&row| &out[row * layout.stride()..(row + 1) * layout.stride()]);
        order
    }

    #[test]
    fn bytes_compare_as_values_do() {
        let ints = column(DataType::Int64, &[3, -1, i64::MIN, 0, i64::MAX, -7], &[]);
        assert_eq!(sorted(&ints, false, false), [2, 5, 1, 3, 0, 4]);
        assert_eq!(sorted(&ints, false, true), [4, 0, 3, 1, 5, 2]);
        let floats = [1.5, -0.5, f64::NEG_INFINITY, 0.0, -2.0, f64::INFINITY];
        let words: Vec<i64> = floats.iter().map(|f| f.to_bits().cast_signed()).collect();
        assert_eq!(
            sorted(&column(DataType::Float64, &words, &[]), false, false),
            [2, 4, 1, 3, 0, 5]
        );
        // Nulls first ascending, last descending.
        let nullable = column(DataType::Int64, &[5, 0, -5], &[1]);
        assert_eq!(sorted(&nullable, true, false), [1, 2, 0]);
        assert_eq!(sorted(&nullable, true, true), [0, 2, 1]);
    }
}
