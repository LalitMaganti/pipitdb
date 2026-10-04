//! `ByteSource`: bytes read by range, such as a file's, on local disk or in
//! object storage. Readers, such as Parquet's, read through it, so where the
//! bytes are is the source's business. Calls block, and readers own the
//! memory read into.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ReadError;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ByteRange {
    pub offset: u64,
    pub len: u64,
}

pub trait ByteSource {
    /// How many bytes there are.
    fn len(&self) -> u64;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Says `ranges` will be read soon, so a source that's remote can fetch
    /// them together. By default, nothing is done.
    fn prefetch(&self, ranges: &[ByteRange]) {
        let _ = ranges;
    }

    /// Fills `into` with the bytes from `offset`, or fails if there aren't
    /// that many.
    fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), ReadError>;
}

impl ByteSource for &[u8] {
    fn len(&self) -> u64 {
        <[u8]>::len(self) as u64
    }

    fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), ReadError> {
        let start = usize::try_from(offset).map_err(|_| ReadError)?;
        let end = start.checked_add(into.len()).ok_or(ReadError)?;
        into.copy_from_slice(self.get(start..end).ok_or(ReadError)?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_ranges_of_bytes_in_memory() {
        let bytes: &[u8] = b"pipitdb";
        let source: &dyn ByteSource = &bytes;
        let mut into = [0; 3];
        assert_eq!(source.len(), 7);
        assert_eq!(source.read(2, &mut into), Ok(()));
        assert_eq!(&into, b"pit");
        assert_eq!(source.read(5, &mut into), Err(ReadError));
    }
}
