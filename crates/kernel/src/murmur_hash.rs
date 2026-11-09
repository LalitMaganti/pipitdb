//! Hashes, as Perfetto's `MurmurHash` makes them.

/// `fmix64` from Murmur hash 3: Perfetto's hash of an integer.
#[inline]
pub(crate) fn mix(word: u64) -> u64 {
    let mut h = word ^ word >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ h >> 33
}

/// Murmur hash 2's 64-bit hash of `bytes`: Perfetto's hash of a string.
pub(crate) fn bytes(bytes: &[u8]) -> u64 {
    const SEED: u64 = 0xe17a_1465;
    const MULTIPLY: u64 = 0xc6a4_a793_5bd1_e995;
    const SHIFT: u32 = 47;
    let mut h = SEED ^ (bytes.len() as u64).wrapping_mul(MULTIPLY);
    let (words, tail) = bytes.as_chunks::<8>();
    for &word in words {
        let mut k = u64::from_le_bytes(word).wrapping_mul(MULTIPLY);
        k ^= k >> SHIFT;
        h ^= k.wrapping_mul(MULTIPLY);
        h = h.wrapping_mul(MULTIPLY);
    }
    if !tail.is_empty() {
        for (i, &byte) in tail.iter().enumerate() {
            h ^= u64::from(byte) << (8 * i);
        }
        h = h.wrapping_mul(MULTIPLY);
    }
    h ^= h >> SHIFT;
    h = h.wrapping_mul(MULTIPLY);
    h ^ h >> SHIFT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_as_perfetto_does() {
        // From a build of Perfetto's `MurmurHashMix` and `MurmurHashBytes`.
        assert_eq!(mix(1), 0xb456_bcfc_34c2_cb2c);
        assert_eq!(mix((-1_i64).cast_unsigned()), 0x64b5_720b_4b82_5f21);
        assert_eq!(bytes(b""), 0x9bfa_e0a4_e613_fc3c);
        assert_eq!(bytes(b"a"), 0x081c_cc83_1546_66a7);
        assert_eq!(bytes(b"hello world!"), 0x16b1_7bea_7cc4_027e);
        assert_eq!(bytes(b"0123456789abcdef"), 0xa56e_4ce8_3ead_c694);
    }
}
