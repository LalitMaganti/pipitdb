//! Bitmaps: bits packed in words, the lowest first, as a column's validity
//! is, a byte at a time, and a join's matches are, a `u64` at a time.

use core::ops::{Deref, DerefMut};

use crate::allocator::AllocError;
use crate::env::Env;
use crate::small_vec::SmallVec;

/// A word bits are packed in.
pub trait Word: Copy {
    /// How many bits it holds.
    const BITS: usize;

    /// Whether bit `bit` is set.
    fn bit(self, bit: usize) -> bool;

    /// With bit `bit` set too if `value`.
    fn or(self, bit: usize, value: bool) -> Self;

    /// With bit `bit` cleared.
    fn without(self, bit: usize) -> Self;
}

macro_rules! word {
    ($($t:ty),*) => {$(
        impl Word for $t {
            const BITS: usize = <$t>::BITS as usize;

            #[inline]
            fn bit(self, bit: usize) -> bool {
                self >> bit & 1 != 0
            }

            #[inline]
            fn or(self, bit: usize, value: bool) -> Self {
                self | <$t>::from(value) << bit
            }

            #[inline]
            fn without(self, bit: usize) -> Self {
                self & !(1 << bit)
            }
        }
    )*};
}

word!(u8, u64);

/// Bits in words of `W`, borrowed, as a slice is a `Vec`'s.
#[repr(transparent)]
pub struct BitSlice<W: Word = u64>([W]);

impl<W: Word> BitSlice<W> {
    /// `words`, as bits.
    #[inline]
    pub fn of(words: &[W]) -> &BitSlice<W> {
        // SAFETY: `BitSlice` is `[W]`, by `repr(transparent)`.
        unsafe { &*(core::ptr::from_ref(words) as *const BitSlice<W>) }
    }

    /// As `of`, to change them.
    #[inline]
    pub fn of_mut(words: &mut [W]) -> &mut BitSlice<W> {
        // SAFETY: as in `of`, and the borrow is unique.
        unsafe { &mut *(core::ptr::from_mut(words) as *mut BitSlice<W>) }
    }

    /// How many words hold `bits` bits.
    #[inline]
    pub fn words_for(bits: usize) -> usize {
        bits.div_ceil(W::BITS)
    }

    /// Whether bit `at` is set.
    #[inline]
    pub fn get(&self, at: usize) -> bool {
        at!(self.0, at / W::BITS).bit(at % W::BITS)
    }

    /// As `get`, without checking `at` is within the bits.
    ///
    /// # Safety
    ///
    /// `at` must be within the bits.
    #[inline]
    pub unsafe fn get_unchecked(&self, at: usize) -> bool {
        // SAFETY: guaranteed by the caller.
        unsafe { self.0.get_unchecked(at / W::BITS) }.bit(at % W::BITS)
    }

    /// Sets bit `at`.
    #[inline]
    pub fn set(&mut self, at: usize) {
        self.set_if(at, true);
    }

    /// Sets bit `at` if `value`, so a clear bit comes to hold it.
    #[inline]
    pub fn set_if(&mut self, at: usize, value: bool) {
        let word = at_mut!(self.0, at / W::BITS);
        *word = word.or(at % W::BITS, value);
    }

    /// Clears bit `at`.
    #[inline]
    pub fn clear(&mut self, at: usize) {
        let word = at_mut!(self.0, at / W::BITS);
        *word = word.without(at % W::BITS);
    }

    /// The words the bits are in.
    #[inline]
    pub fn words(&self) -> &[W] {
        &self.0
    }
}

impl BitSlice<u8> {
    /// Sets bits `start..start + count`.
    pub fn set_run(&mut self, start: usize, count: usize) {
        let end = start + count;
        let mut bit = start;
        // Up to a whole byte, then whole bytes, then the rest.
        while bit < end && !bit.is_multiple_of(8) {
            self.set(bit);
            bit += 1;
        }
        let whole = (end - bit) / 8;
        at_mut!(self.0, bit / 8..bit / 8 + whole).fill(0xff);
        bit += whole * 8;
        while bit < end {
            self.set(bit);
            bit += 1;
        }
    }
}

/// Bits of its own, a `u64` at a time.
pub struct Bitmap {
    words: SmallVec<u64>,
}

impl Bitmap {
    /// `len` bits, all clear.
    pub fn new(env: &Env, len: usize) -> Result<Bitmap, AllocError> {
        Ok(Bitmap { words: SmallVec::zeroed(env, BitSlice::<u64>::words_for(len))? })
    }

    /// Clears every bit.
    pub fn clear_all(&mut self) {
        self.words.fill(0);
    }
}

impl Deref for Bitmap {
    type Target = BitSlice;

    fn deref(&self) -> &BitSlice {
        BitSlice::of(&self.words)
    }
}

impl DerefMut for Bitmap {
    fn deref_mut(&mut self) -> &mut BitSlice {
        BitSlice::of_mut(&mut self.words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sets_and_clears_bits_of_either_width() {
        let mut bytes = [0_u8; 3];
        let bits = BitSlice::of_mut(&mut bytes);
        bits.set_run(3, 15);
        bits.clear(10);
        bits.set_if(20, false);
        bits.set_if(21, true);
        let set: std::vec::Vec<usize> = (0..24).filter(|&at| bits.get(at)).collect();
        let expected: std::vec::Vec<usize> = (3..18).filter(|&at| at != 10).chain([21]).collect();
        assert_eq!(set, expected);

        let mut bitmap = Bitmap::new(Env::current(), 130).unwrap();
        bitmap.set(129);
        assert!(bitmap.get(129) && !bitmap.get(128));
        assert_eq!(bitmap.words().len(), 3);
        bitmap.clear_all();
        assert!(!bitmap.get(129));
    }
}
