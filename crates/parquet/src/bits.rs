//! Moving values to the rows a validity bitmap says aren't null.

use pipit_kernel::containers::BitSlice;

/// Moves the first `valid` of `values` to the rows `validity` says aren't
/// null, in order, and sets the rest to `null`. From the end, so none is
/// written over before it's moved.
pub(crate) fn spread<T: Copy>(values: &mut [T], mut valid: usize, validity: &[u8], null: T) {
    for row in (0..values.len()).rev() {
        let value = if BitSlice::of(validity).get(row) {
            valid -= 1;
            *at!(values, valid)
        } else {
            null
        };
        *at_mut!(values, row) = value;
    }
}
