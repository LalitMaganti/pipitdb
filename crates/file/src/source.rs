//! `FileSource`: a `ByteSource` reading a file on local disk.

use std::fs::File;
use std::path::Path;

use pipit_kernel::bytes::{ByteSource, ReadError};

use crate::read_at;

pub struct FileSource {
    file: File,
    len: u64,
}

impl FileSource {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<FileSource> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(FileSource { file, len })
    }
}

impl ByteSource for FileSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), ReadError> {
        let fits = offset.checked_add(into.len() as u64).is_some_and(|end| end <= self.len);
        if !fits {
            return Err(ReadError);
        }
        read_at(&self.file, into, offset).map_err(|_| ReadError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore = "Miri can't use real files")]
    fn reads_ranges_of_a_file() {
        let path = std::env::temp_dir().join(format!("pipitdb-source-{}", std::process::id()));
        std::fs::write(&path, b"pipitdb").unwrap();
        let source = FileSource::open(&path).unwrap();
        let mut into = [0; 3];
        assert_eq!(source.len(), 7);
        assert_eq!(source.read(4, &mut into), Ok(()));
        assert_eq!(&into, b"tdb");
        assert_eq!(source.read(5, &mut into), Err(ReadError));
        std::fs::remove_file(&path).unwrap();
    }
}
