//! Words that order as integers do as their values do: a double's bits, so
//! sorts, filters and aggregates compare doubles as integers, and a sort
//! key's, which orders unsigned.

/// A double's `bits`, with a negative's flipped but its sign, so they order
/// as integers as the doubles do. Undoes itself.
#[inline]
pub(crate) const fn ordered(bits: i64) -> i64 {
    bits ^ ((bits >> 63).cast_unsigned() >> 1).cast_signed()
}

/// As `ordered` if `float`, and `bits` as they are if not. Takes whether
/// they're a double's as a flag, so a loop over one type compiles to one
/// without the other's work.
#[inline]
pub(crate) const fn ordered_if(bits: i64, float: bool) -> i64 {
    if float { ordered(bits) } else { bits }
}

/// The value of sort key `key`, which orders unsigned once `invert`, all
/// ones for a descending key, is undone: an integer, or a double's bits if
/// `float`.
#[inline]
pub(crate) const fn key_value(key: u64, invert: u64, float: bool) -> i64 {
    ordered_if((key ^ invert).cast_signed() ^ i64::MIN, float)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders_doubles_as_integers() {
        let doubles = [f64::NEG_INFINITY, -2.5, -0.0, 0.0, 1.0, 3.5, f64::INFINITY];
        let words = doubles.map(|d| ordered(d.to_bits().cast_signed()));
        assert!(words.is_sorted());
        for (&word, d) in words.iter().zip(doubles) {
            assert_eq!(ordered(word).cast_unsigned(), d.to_bits());
        }
        assert_eq!(ordered_if(-5, false), -5);
    }
}
