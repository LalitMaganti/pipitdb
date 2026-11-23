//! A bitmap that also says how many bits are set before any bit: its rank,
//! which numbers whatever the set bits stand for, in order, without gaps.

use crate::allocator::AllocError;
use crate::bitmap::Bitmap;
use crate::global::Global;
use crate::slow_vec::SlowVec;

/// Bits, with how many are set before each word of them.
pub struct RankBitmap {
    bits: Bitmap,
    before: SlowVec<u32>,
}

impl RankBitmap {
    /// `len` bits, of which those in `set` are set.
    pub fn new(
        global: &Global,
        len: usize,
        set: impl Iterator<Item = usize>,
    ) -> Result<RankBitmap, AllocError> {
        let mut bits = Bitmap::new(global, len)?;
        for at in set {
            bits.set(at);
        }
        let mut before = SlowVec::fixed_from(global, bits.words().iter().map(|_| 0))?;
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
        let bitmap = RankBitmap::new(Global::current(), 201, set.into_iter()).unwrap();
        for (rank, &at) in set.iter().enumerate() {
            assert!(bitmap.get(at));
            assert_eq!(bitmap.rank(at) as usize, rank);
        }
        assert!(!bitmap.get(4));
        assert_eq!(bitmap.rank(100), 3);
    }
}
