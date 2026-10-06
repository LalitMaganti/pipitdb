//! Parquet's RLE and bit-packed hybrid encoding, of definition levels and
//! dictionary indices: runs of one value, and groups of 8 values packed in
//! `width` bits each.

/// Values decoded together: `n` copies of one value, or `n` values unpacked
/// into the caller's memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Run {
    Repeat(u32, usize),
    Packed(usize),
}

impl Run {
    /// How many values it has.
    pub fn len(self) -> usize {
        match self {
            Run::Repeat(_, n) | Run::Packed(n) => n,
        }
    }
}

/// Where decoding is: its bytes are passed to `next_run`, so it can be kept
/// between calls without borrowing them.
#[derive(Clone, Copy, Default)]
pub struct Hybrid {
    /// Where the next run's header is.
    pos: usize,
    /// How many bits each value takes.
    width: u32,
    /// Values left in the run being read.
    left: usize,
    /// For a run of one value, the value.
    rle: Option<u32>,
    /// For a packed run, where its next value starts, in bits from where
    /// its values start.
    bit: usize,
    /// For a packed run, where its values start.
    packed: usize,
}

impl Hybrid {
    pub fn new(width: u32) -> Hybrid {
        Hybrid { pos: 0, width, left: 0, rle: None, bit: 0, packed: 0 }
    }

    /// The next values of `bytes`, at most `out.len()` of them: a run of one
    /// value, or values unpacked into `out`. `None` if there are no more or
    /// they're damaged.
    pub fn next_run(&mut self, bytes: &[u8], out: &mut [u32]) -> Option<Run> {
        if self.left == 0 {
            self.start_run(bytes)?;
        }
        let n = self.left.min(out.len());
        self.left -= n;
        if let Some(value) = self.rle {
            return Some(Run::Repeat(value, n));
        }
        self.unpack(bytes, at_mut!(out, ..n));
        Some(Run::Packed(n))
    }

    /// Fills `out` with the next values of `bytes`, or returns `None` if
    /// there aren't that many or they're damaged.
    pub fn take(&mut self, bytes: &[u8], out: &mut [u32]) -> Option<()> {
        let mut at = 0;
        while at < out.len() {
            let run = self.next_run(bytes, at_mut!(out, at..))?;
            if let Run::Repeat(value, n) = run {
                at_mut!(out, at..at + n).fill(value);
            }
            at += run.len();
        }
        Some(())
    }

    /// Fills `values` from the packed run, whose bytes `start_run` checked
    /// are there. The bits stream through a word, topped up 4 bytes at a
    /// time.
    #[expect(clippy::cast_possible_truncation, reason = "values are at most 32 bits")]
    fn unpack(&mut self, bytes: &[u8], values: &mut [u32]) {
        let width = self.width;
        let mask = (1_u64 << width) - 1;
        let (mut at, skip) = (self.packed + self.bit / 8, self.bit % 8);
        let (mut word, mut bits) = (0_u64, 0_u32);
        let mut top_up = |word: &mut u64, bits: &mut u32| {
            let next = match bytes.get(at..at + 4) {
                Some(next) => u32::from_le_bytes([next[0], next[1], next[2], next[3]]),
                // Near the end, fewer than 4 bytes may be left.
                None => (0..4).fold(0, |next, i| {
                    next | u32::from(bytes.get(at + i).copied().unwrap_or(0)) << (8 * i)
                }),
            };
            *word |= u64::from(next) << *bits;
            (*bits, at) = (*bits + 32, at + 4);
        };
        top_up(&mut word, &mut bits);
        (word, bits) = (word >> skip, bits - skip as u32);
        for value in values.iter_mut() {
            if bits < width {
                top_up(&mut word, &mut bits);
            }
            *value = (word & mask) as u32;
            (word, bits) = (word >> width, bits - width);
        }
        self.bit += values.len() * width as usize;
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
    extern crate std;

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

    /// `values`, a multiple of 8 of them, packed `width` bits each.
    fn packed(values: &[u32], width: u32) -> std::vec::Vec<u8> {
        let groups = values.len() / 8;
        let mut bytes = std::vec![u8::try_from(groups << 1 | 1).unwrap()];
        let mut bits = std::vec![0_u8; groups * width as usize];
        for (i, &value) in values.iter().enumerate() {
            for b in 0..width as usize {
                let bit = i * width as usize + b;
                bits[bit / 8] |= u8::from(value >> b & 1 == 1) << (bit % 8);
            }
        }
        bytes.extend(bits);
        bytes
    }

    #[test]
    fn unpacks_wide_values() {
        for width in [10, 17, 32] {
            let values: std::vec::Vec<u32> =
                (0..48_u32).map(|i| i.wrapping_mul(2_654_435_761) >> (32 - width)).collect();
            let bytes = packed(&values, width);
            let mut out = std::vec![0; 48];
            assert_eq!(Hybrid::new(width).take(&bytes, &mut out), Some(()));
            assert_eq!(out, values);
            // From the middle of a value's byte.
            let mut hybrid = Hybrid::new(width);
            let (mut three, mut rest) = ([0; 3], std::vec![0; 45]);
            assert_eq!(hybrid.take(&bytes, &mut three), Some(()));
            assert_eq!(hybrid.take(&bytes, &mut rest), Some(()));
            assert_eq!((&three[..], &rest[..]), (&values[..3], &values[3..]));
        }
    }
}
