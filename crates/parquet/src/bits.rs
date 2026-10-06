//! Validity bitmaps: reading bits, setting runs of them, and moving values
//! to the rows that aren't null.

/// Whether bit `at` of `bits` is set.
pub(crate) fn get(bits: &[u8], at: usize) -> bool {
    *at!(bits, at / 8) >> (at % 8) & 1 == 1
}

/// Sets bits `start..start + count` of `bits`.
pub(crate) fn set(bits: &mut [u8], start: usize, count: usize) {
    let end = start + count;
    let mut bit = start;
    // Up to a whole byte, then whole bytes, then the rest.
    while bit < end && !bit.is_multiple_of(8) {
        *at_mut!(bits, bit / 8) |= 1 << (bit % 8);
        bit += 1;
    }
    let whole = (end - bit) / 8;
    at_mut!(bits, bit / 8..bit / 8 + whole).fill(0xff);
    bit += whole * 8;
    while bit < end {
        *at_mut!(bits, bit / 8) |= 1 << (bit % 8);
        bit += 1;
    }
}

/// Moves the first `valid` of `values` to the rows `validity` says aren't
/// null, in order, and sets the rest to `null`. From the end, so none is
/// written over before it's moved.
pub(crate) fn spread<T: Copy>(values: &mut [T], mut valid: usize, validity: &[u8], null: T) {
    for row in (0..values.len()).rev() {
        let value = if *at!(validity, row / 8) >> (row % 8) & 1 != 0 {
            valid -= 1;
            *at!(values, valid)
        } else {
            null
        };
        *at_mut!(values, row) = value;
    }
}
