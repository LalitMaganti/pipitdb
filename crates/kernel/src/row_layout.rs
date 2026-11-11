//! `RowLayout`: rows laid out in fixed-width slots whose bytes compare as
//! their values do, as Perfetto's `core/common/row_layout.h` lays them out.
//! After Arrow's row format:
//! <https://arrow.apache.org/blog/2022/11/07/multi-column-sorts-in-arrow-rust-part-2/>

use crate::allocator::{AllocError, Allocator};
use crate::slow_vec::SlowVec;

/// What a slot holds.
#[derive(Clone, Copy)]
pub(crate) enum Type {
    /// A string's rank among the strings sorted.
    Uint32,
    Int64,
    Float64,
}

/// A column to lay out.
#[derive(Clone, Copy)]
pub(crate) struct Column {
    pub(crate) data_type: Type,
    pub(crate) nullable: bool,
    pub(crate) descending: bool,
}

/// Where a column is in each row. Small enough to pass in a register:
/// `write` takes it by value, so its fields stay in registers while row
/// bytes are written.
#[derive(Clone, Copy)]
pub(crate) struct Slot {
    pub(crate) offset: u16,
    pub(crate) stride: u16,
    pub(crate) nullable: bool,
    pub(crate) descending: bool,
}

/// A slot for each column, one after another in each row. A nullable slot
/// starts with 0xFF if the row has a value and 0 if not; a descending slot's
/// bits are inverted.
pub(crate) struct RowLayout {
    slots: SlowVec<Slot>,
    stride: u32,
}

impl RowLayout {
    pub(crate) fn new(
        allocator: &dyn Allocator,
        columns: &[Column],
    ) -> Result<RowLayout, AllocError> {
        let mut slots = SlowVec::fixed(allocator, columns.len())?;
        let mut stride = 0_u32;
        for column in columns {
            let Ok(offset) = u16::try_from(stride) else { return Err(AllocError) };
            let (nullable, descending) = (column.nullable, column.descending);
            slots.push(Slot { offset, stride: 0, nullable, descending })?;
            stride += u32::from(nullable) + value_size(column.data_type);
        }
        let Ok(slot_stride) = u16::try_from(stride) else { return Err(AllocError) };
        for slot in slots.iter_mut() {
            slot.stride = slot_stride;
        }
        Ok(RowLayout { slots, stride })
    }

    pub(crate) fn stride(&self) -> u32 {
        self.stride
    }

    pub(crate) fn slot(&self, column: usize) -> Slot {
        *at!(self.slots, column)
    }

    /// Writes `slot` of the first `count` rows of `rows`, each row `i`'s
    /// value `get(i)`, `None` if it has none.
    #[inline]
    pub(crate) fn write<T: Encode>(
        slot: Slot,
        count: u32,
        get: impl Fn(u32) -> Option<T>,
        rows: &mut [u8],
    ) {
        let (offset, stride) = (usize::from(slot.offset), usize::from(slot.stride));
        check!(count as usize * stride <= rows.len());
        let invert = if slot.descending { u64::MAX } else { 0 };
        for i in 0..count {
            let to = at_mut!(rows, i as usize * stride + offset..);
            if slot.nullable {
                let value = get(i);
                #[expect(clippy::cast_possible_truncation, reason = "a byte of all ones or none")]
                let byte = u8::from(value.is_some()).wrapping_neg() ^ invert as u8;
                *at_mut!(to, 0) = byte;
                let bits = value.map_or(0, |value| value.encode() ^ invert).to_be_bytes();
                at_mut!(to, 1..=T::SIZE).copy_from_slice(at!(bits, 8 - T::SIZE..));
            } else {
                let Some(value) = get(i) else { failed!() };
                let bits = (value.encode() ^ invert).to_be_bytes();
                at_mut!(to, ..T::SIZE).copy_from_slice(at!(bits, 8 - T::SIZE..));
            }
        }
    }
}

/// The word `write` laid out in `slot` of `row`, an integer or, if `float`,
/// a float's bits; `None` if it has none.
pub(crate) fn decode(slot: Slot, row: &[u8], float: bool) -> Option<i64> {
    let invert = if slot.descending { u64::MAX } else { 0 };
    let mut at = usize::from(slot.offset);
    if slot.nullable {
        #[expect(clippy::cast_possible_truncation, reason = "a byte of all ones or none")]
        let present = *at!(row, at) ^ invert as u8 != 0;
        if !present {
            return None;
        }
        at += 1;
    }
    Some(decode_value(slot, at!(row, at..at + 8), float))
}

/// The word `write` laid out in `slot` of a row from its value's `bytes`,
/// after any flag.
#[inline]
pub(crate) fn decode_value(slot: Slot, bytes: &[u8], float: bool) -> i64 {
    let invert = if slot.descending { u64::MAX } else { 0 };
    let mut be = [0; 8];
    be.copy_from_slice(bytes);
    let bits = (u64::from_be_bytes(be) ^ invert).cast_signed() ^ i64::MIN;
    // A negative float's bits but its sign were flipped by `Word::encode`.
    let flip = if float { ((bits >> 63).cast_unsigned() >> 1).cast_signed() } else { 0 };
    bits ^ flip
}

fn value_size(data_type: Type) -> u32 {
    match data_type {
        Type::Uint32 => 4,
        Type::Int64 | Type::Float64 => 8,
    }
}

/// A value as bits whose unsigned order is the value's, in the low `SIZE`
/// bytes.
pub(crate) trait Encode: Copy {
    const SIZE: usize;
    fn encode(self) -> u64;
}

impl Encode for u32 {
    const SIZE: usize = 4;
    fn encode(self) -> u64 {
        u64::from(self)
    }
}

impl Encode for i64 {
    const SIZE: usize = 8;
    fn encode(self) -> u64 {
        (self ^ i64::MIN).cast_unsigned()
    }
}

/// An integer, or a float's bits, as a column of either holds it.
#[derive(Clone, Copy)]
pub(crate) struct Word {
    pub(crate) bits: i64,
    pub(crate) float: bool,
}

impl Encode for Word {
    const SIZE: usize = 8;
    /// As `f64`'s if a float, and `i64`'s if not.
    fn encode(self) -> u64 {
        let negative_float = (self.bits >> 63).cast_unsigned() >> 1;
        let flip = if self.float { negative_float } else { 0 };
        (self.bits ^ flip.cast_signed()).encode()
    }
}

impl Encode for f64 {
    const SIZE: usize = 8;
    /// A negative float's bits but its sign are flipped too, so they order
    /// backwards.
    fn encode(self) -> u64 {
        let bits = self.to_bits().cast_signed();
        (bits ^ ((bits >> 63).cast_unsigned() >> 1).cast_signed()).encode()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::cmp::Ordering;

    use super::*;
    use crate::allocator::Heap;

    fn expect_ordered<T: Encode>(data_type: Type, ascending: &[Option<T>], nullable: bool) {
        let count = u32::try_from(ascending.len()).unwrap();
        for (descending, expected) in [(false, Ordering::Less), (true, Ordering::Greater)] {
            let layout =
                RowLayout::new(&Heap, &[Column { data_type, nullable, descending }]).unwrap();
            let mut rows = alloc::vec![0; layout.stride() as usize * ascending.len()];
            RowLayout::write(layout.slot(0), count, |i| ascending[i as usize], &mut rows);
            let rows: Vec<_> = rows.chunks(layout.stride() as usize).collect();
            for (i, pair) in rows.windows(2).enumerate() {
                assert_eq!(pair[0].cmp(pair[1]), expected, "{i}");
            }
        }
    }

    #[test]
    fn bytes_sort_as_values() {
        let ints = [i64::MIN, -(1 << 40), -1, 0, 1, 1 << 40, i64::MAX];
        expect_ordered(Type::Int64, &ints.map(Some), false);
        let floats = [f64::NEG_INFINITY, -1e300, -1.5, -0.0, 1e-300, 1.5, 1e300, f64::INFINITY];
        expect_ordered(Type::Float64, &floats.map(Some), false);
        expect_ordered(Type::Uint32, &[0, 1, 255, 256, u32::MAX].map(Some), true);
        // Nulls sort first.
        expect_ordered(Type::Int64, &[None, Some(i64::MIN), Some(0)], true);
    }

    #[test]
    fn slots_sit_one_after_another() {
        let columns = [
            Column { data_type: Type::Int64, nullable: false, descending: false },
            Column { data_type: Type::Float64, nullable: true, descending: false },
            Column { data_type: Type::Int64, nullable: true, descending: true },
        ];
        let layout = RowLayout::new(&Heap, &columns).unwrap();
        assert_eq!(layout.slot(0).offset, 0);
        assert_eq!(layout.slot(1).offset, 8);
        assert_eq!(layout.slot(2).offset, 17);
        assert_eq!(layout.stride(), 26);
        assert_eq!(layout.slot(1).stride, 26);
    }

    #[test]
    fn decodes_what_it_lays_out() {
        let values = [Some(-3.5_f64), None, Some(0.0), Some(f64::MAX), Some(-0.0)];
        for (descending, float) in [(false, true), (true, true), (false, false), (true, false)] {
            let data_type = if float { Type::Float64 } else { Type::Int64 };
            let column = Column { data_type, nullable: true, descending };
            let layout = RowLayout::new(&Heap, &[column]).unwrap();
            let stride = layout.stride() as usize;
            #[expect(clippy::cast_possible_truncation, reason = "whole numbers, or floats' bits")]
            let bits = |i: u32| {
                values[i as usize]
                    .map(|v: f64| if float { v.to_bits().cast_signed() } else { v as i64 })
            };
            let get = |i: u32| bits(i).map(|bits| Word { bits, float });
            let mut rows = alloc::vec![0; stride * values.len()];
            RowLayout::write(layout.slot(0), 5, get, &mut rows);
            for (i, row) in (0..).zip(rows.chunks(stride)) {
                assert_eq!(decode(layout.slot(0), row, float), bits(i), "{i}");
            }
        }
    }
}
