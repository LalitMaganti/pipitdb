//! Parquet's RLE and bit-packed hybrid encoding, of definition levels and
//! dictionary indices: runs of one value, and groups of 8 values packed in
//! `width` bits each.

/// Where decoding is: its bytes are passed to `next`, so it can be kept
/// between calls without borrowing them.
#[derive(Clone, Copy, Default)]
pub struct Hybrid {
    pos: usize,
    width: u32,
    /// Values left in the run being read.
    left: usize,
    /// For a run of one value, the value; else where the packed run's next
    /// value starts, in bits from `pos`'s start.
    rle: Option<u32>,
    bit: usize,
    packed: usize,
}

impl Hybrid {
    pub fn new(width: u32) -> Hybrid {
        Hybrid { pos: 0, width, left: 0, rle: None, bit: 0, packed: 0 }
    }

    /// Fills `out` with the next values of `bytes`, or returns `None` if
    /// there aren't that many or they're damaged.
    #[expect(clippy::cast_possible_truncation, reason = "values are at most 32 bits")]
    pub fn take(&mut self, bytes: &[u8], out: &mut [u32]) -> Option<()> {
        let mut at = 0;
        while at < out.len() {
            if self.left == 0 {
                self.start_run(bytes)?;
            }
            let n = self.left.min(out.len() - at);
            let values = at_mut!(out, at..at + n);
            if let Some(value) = self.rle {
                values.fill(value);
            } else {
                let width = self.width as usize;
                let mask = (1_u64 << width) - 1;
                for value in values {
                    let (byte, shift) = (self.packed + self.bit / 8, self.bit % 8);
                    let word = match bytes.get(byte..byte + 8) {
                        Some(word) => u64::from_le_bytes(word.try_into().ok()?),
                        // Near the end, fewer bytes are left than a word.
                        None => (0..8).fold(0, |word, i| {
                            let byte = bytes.get(byte + i).copied().unwrap_or(0);
                            word | u64::from(byte) << (8 * i)
                        }),
                    };
                    *value = ((word >> shift) & mask) as u32;
                    self.bit += width;
                }
            }
            (self.left, at) = (self.left - n, at + n);
        }
        Some(())
    }

    /// Passes over the next `count` values of `bytes` without decoding them,
    /// or returns `None` if there aren't that many or they're damaged.
    pub fn skip(&mut self, bytes: &[u8], mut count: usize) -> Option<()> {
        while count > 0 {
            if self.left == 0 {
                self.start_run(bytes)?;
            }
            let n = self.left.min(count);
            if self.rle.is_none() {
                self.bit += n * self.width as usize;
            }
            (self.left, count) = (self.left - n, count - n);
        }
        Some(())
    }

    fn start_run(&mut self, bytes: &[u8]) -> Option<()> {
        let mut header = 0_u64;
        for shift in (0..64).step_by(7) {
            let byte = *bytes.get(self.pos)?;
            self.pos += 1;
            header |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
        }
        let count = usize::try_from(header >> 1).ok()?;
        if header & 1 == 0 {
            // A run of `count` copies of one value, in whole bytes.
            let len = self.width.div_ceil(8) as usize;
            let mut value = 0_u32;
            for i in 0..len {
                value |= u32::from(*bytes.get(self.pos + i)?) << (8 * i);
            }
            self.pos += len;
            (self.rle, self.left) = (Some(value), count);
        } else {
            // `count` groups of 8 values, packed, which must all be there.
            self.packed = self.pos;
            self.bit = 0;
            self.pos = self.pos.checked_add(count.checked_mul(self.width as usize)?)?;
            if self.pos > bytes.len() {
                return None;
            }
            (self.rle, self.left) = (None, count * 8);
        }
        if self.left == 0 { self.start_run(bytes) } else { Some(()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_runs_and_packed_groups() {
        // A run of three 1s, then a packed group of 8 values of 3 bits:
        // 0..8, then a run of two 5s.
        let bytes = [6, 1, 3, 0b1000_1000, 0b1100_0110, 0b1111_1010, 4, 5];
        let want = [1, 1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 5, 5];
        let mut hybrid = Hybrid::new(3);
        let mut values = [0; 13];
        assert_eq!(hybrid.take(&bytes, &mut values), Some(()));
        assert_eq!(values, want);
        assert_eq!(hybrid.take(&bytes, &mut [0]), None);
        // Taken a few at a time, or skipped, from the middle of a run.
        let mut hybrid = Hybrid::new(3);
        let (mut two, mut five) = ([0; 2], [0; 5]);
        assert_eq!(hybrid.take(&bytes, &mut two), Some(()));
        assert_eq!(hybrid.skip(&bytes, 3), Some(()));
        assert_eq!(hybrid.take(&bytes, &mut five), Some(()));
        assert_eq!((two, five), ([1, 1], [2, 3, 4, 5, 6]));
        assert_eq!(hybrid.skip(&bytes, 4), None);
    }
}
