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

    /// The next value of `bytes`, or `None` if there are no more or they're
    /// damaged.
    pub fn next(&mut self, bytes: &[u8]) -> Option<u32> {
        if self.left == 0 {
            self.start_run(bytes)?;
        }
        self.left -= 1;
        if let Some(value) = self.rle {
            return Some(value);
        }
        let width = self.width as usize;
        let (byte, shift) = (self.packed + self.bit / 8, self.bit % 8);
        let mut word = 0_u64;
        for i in 0..(shift + width).div_ceil(8) {
            word |= u64::from(*bytes.get(byte + i)?) << (8 * i);
        }
        self.bit += width;
        u32::try_from((word >> shift) & ((1 << width) - 1)).ok()
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
            // `count` groups of 8 values, packed.
            self.packed = self.pos;
            self.bit = 0;
            self.pos = self.pos.checked_add(count * self.width as usize)?;
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
        let mut hybrid = Hybrid::new(3);
        let values: [Option<u32>; 13] = core::array::from_fn(|_| hybrid.next(&bytes));
        let want = [1, 1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 5, 5].map(Some);
        assert_eq!(values, want);
        assert_eq!(hybrid.next(&bytes), None);
    }
}
