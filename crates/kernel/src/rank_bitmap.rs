//! A bitmap that also says how many bits are set before any bit: its rank,
//! which numbers whatever the set bits stand for, in order, without gaps.

use crate::allocator::AllocError;
use crate::array::Array;
use crate::bitmap::Bitmap;
use crate::env::Env;

/// Bits, with how many are set before each word of them.
pub struct RankBitmap {
    bits: Bitmap,
    before: Array<u32>,
}

impl RankBitmap {
    /// `len` bits, of which those in `set` are set.
    pub fn new(
        env: &Env,
        len: usize,
        set: impl Iterator<Item = usize>,
    ) -> Result<RankBitmap, AllocError> {
        let mut bits = Bitmap::new(env, len)?;
        for at in set {
            bits.set(at);
        }
        let mut before = Array::zeroed(env, bits.words().len())?;
        let mut count = 0;
        for (before, word) in before.iter_mut().zip(bits.words()) {
            *before = count;
            count += word.count_ones();
        }
        Ok(RankBitmap { bits, before })
    }

    /// Whether bit `at` is set.
    #[inline]
    pub fn get(&self, at: usize) -> bool {
        self.bits.get(at)
    }

    /// How many bits are set before bit `at`.
    #[inline]
    pub fn rank(&self, at: usize) -> u32 {
        let word = *at!(self.bits.words(), at / 64);
        *at!(self.before, at / 64) + (word & ((1 << (at % 64)) - 1)).count_ones()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_set_bits_in_order() {
        let set = [3, 64, 65, 200];
        let bitmap = RankBitmap::new(Env::current(), 201, set.into_iter()).unwrap();
        for (rank, &at) in set.iter().enumerate() {
            assert!(bitmap.get(at));
            assert_eq!(bitmap.rank(at) as usize, rank);
        }
        assert!(!bitmap.get(4));
        assert_eq!(bitmap.rank(100), 3);
    }
}
