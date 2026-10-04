//! `FileSpill`: a `SpillStore` on local disk, a file for each log.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use pipit_kernel::error::Error;
use pipit_kernel::spill::{Block, LogId, SpillStore};

use crate::read_at;

/// Logs are files in a directory, without names, so they go when they're
/// closed, even if the process ends early. As on S3, a log can only be read
/// once sealed.
pub struct FileSpill {
    dir: PathBuf,
    logs: Mutex<Vec<Option<Log>>>,
}

struct Log {
    file: File,
    len: u64,
    sealed: bool,
}

/// Names files uniquely, across stores in this process.
static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

impl FileSpill {
    /// A store with its logs in `dir`, which must exist.
    pub fn new(dir: impl Into<PathBuf>) -> FileSpill {
        FileSpill { dir: dir.into(), logs: Mutex::new(Vec::new()) }
    }

    /// Runs `f` on `log`, which must exist.
    fn with<T>(
        &self,
        log: LogId,
        f: impl FnOnce(&mut Log) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut logs = self.logs.lock().map_err(|_| Error::Io)?;
        let index = usize::try_from(log.0).map_err(|_| Error::Io)?;
        f(logs.get_mut(index).and_then(Option::as_mut).ok_or(Error::Io)?)
    }
}

impl SpillStore for FileSpill {
    fn create(&self) -> Result<LogId, Error> {
        let file = open_unnamed(&self.dir).map_err(|_| Error::Io)?;
        let log = Log { file, len: 0, sealed: false };
        let mut logs = self.logs.lock().map_err(|_| Error::Io)?;
        logs.push(Some(log));
        Ok(LogId(logs.len() as u64 - 1))
    }

    fn append(&self, log: LogId, bytes: &[u8]) -> Result<Block, Error> {
        self.with(log, |log| {
            if log.sealed {
                return Err(Error::Io);
            }
            log.file.write_all(bytes).map_err(|_| Error::Io)?;
            let block = Block { offset: log.len, len: bytes.len() as u64 };
            log.len += block.len;
            Ok(block)
        })
    }

    fn seal(&self, log: LogId) -> Result<(), Error> {
        self.with(log, |log| {
            log.sealed = true;
            Ok(())
        })
    }

    fn read(&self, log: LogId, block: Block, into: &mut [u8]) -> Result<(), Error> {
        self.with(log, |log| {
            let fits = block.offset.checked_add(block.len).is_some_and(|end| end <= log.len);
            if !log.sealed || !fits || block.len != into.len() as u64 {
                return Err(Error::Io);
            }
            read_at(&log.file, into, block.offset).map_err(|_| Error::Io)
        })
    }

    fn delete(&self, log: LogId) {
        let Ok(mut logs) = self.logs.lock() else { return };
        let Some(slot) = usize::try_from(log.0).ok().and_then(|i| logs.get_mut(i)) else { return };
        // Closing the file deletes it.
        drop(slot.take());
    }
}

/// `O_TMPFILE`, where it's known: a file in a directory that never has a
/// name.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const O_TMPFILE: i32 = 0o20_200_000;
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const O_TMPFILE: i32 = 0o20_040_000;

/// Opens a file in `dir` that's deleted when it's closed, even if the
/// process ends first. On Linux it never has a name; on other Unixes it has
/// one only until it's open, as SQLite does; on Windows, Windows deletes it.
fn open_unnamed(dir: &Path) -> std::io::Result<File> {
    #[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Not every filesystem has it; then a named file is used.
        let unnamed = OpenOptions::new().read(true).write(true).custom_flags(O_TMPFILE).open(dir);
        if let Ok(file) = unnamed {
            return Ok(file);
        }
    }
    let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("pipitdb-spill-{}-{id}", std::process::id()));
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;
        options.custom_flags(FILE_FLAG_DELETE_ON_CLOSE);
    }
    let file = options.open(&path)?;
    #[cfg(unix)]
    std::fs::remove_file(&path)?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use pipit_kernel::allocator::Heap;
    use pipit_kernel::buffer::Buffer;
    use pipit_kernel::column::{ColumnView, DataType};
    use pipit_kernel::spill::{read_column, write_column};

    use super::*;

    fn column(values: &[i64]) -> ColumnView {
        let mut buffer = Buffer::allocate(&Heap, values.len() * 8).unwrap();
        buffer.as_mut_slice::<i64>().copy_from_slice(values);
        let mut validity = Buffer::allocate(&Heap, values.len().div_ceil(8)).unwrap();
        validity.as_mut_slice::<u8>().fill(0b1111_1101);
        ColumnView::new(DataType::Int64, buffer, Some(validity))
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri can't use real files")]
    fn spills_columns_to_files_and_back() {
        let store = FileSpill::new(std::env::temp_dir());
        let values: Vec<i64> = (0..3000).collect();
        let columns = [column(&values), column(&values).slice(3, 2000)];
        // Two logs, written in turns, read back independently.
        let (a, b) = (store.create().unwrap(), store.create().unwrap());
        let spilled: Vec<_> = columns
            .iter()
            .map(|c| (write_column(&store, a, c).unwrap(), write_column(&store, b, c).unwrap()))
            .collect();
        store.seal(a).unwrap();
        store.seal(b).unwrap();
        for (column, (in_a, in_b)) in columns.iter().zip(&spilled) {
            for (log, spilled) in [(a, in_a), (b, in_b)] {
                let read = read_column(&Heap, &store, log, spilled).unwrap();
                assert_eq!(read.int64s(), column.int64s());
                let nulls = |c: &ColumnView| (0..c.row_count()).filter(|&r| c.is_null(r)).count();
                assert_eq!(nulls(&read), nulls(column));
            }
        }
        store.delete(a);
        assert_eq!(read_column(&Heap, &store, a, &spilled[0].0).err(), Some(Error::Io));
        store.delete(b);
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri can't use real files")]
    fn logs_are_read_only_once_sealed() {
        let store = FileSpill::new(std::env::temp_dir());
        let log = store.create().unwrap();
        let spilled = write_column(&store, log, &column(&[1, 2, 3])).unwrap();
        assert_eq!(read_column(&Heap, &store, log, &spilled).err(), Some(Error::Io));
        store.seal(log).unwrap();
        assert!(read_column(&Heap, &store, log, &spilled).is_ok());
        assert_eq!(store.append(log, &[0]).err(), Some(Error::Io));
        store.delete(log);
    }

    #[test]
    #[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[cfg_attr(miri, ignore = "Miri can't use real files")]
    fn opens_files_without_names_on_linux() {
        use std::os::unix::fs::OpenOptionsExt;
        // A wrong `O_TMPFILE` would quietly fall back to named files.
        let unnamed = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(O_TMPFILE)
            .open(std::env::temp_dir());
        assert!(unnamed.is_ok());
    }

    #[test]
    #[cfg(unix)]
    #[cfg_attr(miri, ignore = "Miri can't use real files")]
    fn leaves_no_files_behind() {
        let dir = std::env::temp_dir().join(format!("pipitdb-spill-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store = FileSpill::new(&dir);
        let log = store.create().unwrap();
        store.append(log, &[1, 2, 3]).unwrap();
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        store.delete(log);
        std::fs::remove_dir(&dir).unwrap();
    }
}
