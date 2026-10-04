//! `S3Spill`: a `SpillStore` on S3, or anything that speaks its API.
//!
//! Each log is a multipart upload. Appends are staged in a local file until
//! there's a part's worth, then streamed up from it, so parts never sit in
//! memory. Sealing completes the upload; reads are ranged gets.

use std::io::Read;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use pipit_file::spill::FileSpill;
use pipit_kernel::spill::{Block, LogId, SpillError, SpillStore};
use rusty_s3::actions::{CreateMultipartUpload, S3Action};
use rusty_s3::{Bucket, Credentials};

/// S3's smallest part, but for the last.
pub const PART_BYTES_MIN: u64 = 5 << 20;

/// How long a signed request stays valid.
const SIGNED_FOR: Duration = Duration::from_mins(15);

pub struct S3Spill {
    agent: ureq::Agent,
    bucket: Bucket,
    credentials: Credentials,
    /// What each log's key starts with.
    prefix: String,
    /// Where parts are staged before they're uploaded.
    staging: FileSpill,
    part_bytes: u64,
    logs: Mutex<Vec<Option<Log>>>,
}

struct Log {
    key: String,
    upload_id: String,
    /// Each part uploaded's `ETag`, in order.
    etags: Vec<String>,
    /// The part being staged, and its length so far.
    staged: Option<(LogId, u64)>,
    len: u64,
    sealed: bool,
}

/// Names logs uniquely, across stores in this process.
static NEXT_LOG: AtomicU64 = AtomicU64::new(0);

impl S3Spill {
    /// Logs in `bucket`, as `prefix` and a number, uploaded in parts of
    /// `part_bytes`, staged in `staging`. S3 needs parts of at least
    /// `PART_BYTES_MIN`; a store that allows less can be given less.
    pub fn new(
        bucket: Bucket,
        credentials: Credentials,
        prefix: impl Into<String>,
        staging: FileSpill,
        part_bytes: u64,
    ) -> S3Spill {
        S3Spill {
            agent: ureq::Agent::new_with_defaults(),
            bucket,
            credentials,
            prefix: prefix.into(),
            staging,
            part_bytes,
            logs: Mutex::new(Vec::new()),
        }
    }

    /// Runs `f` on `log`, which must exist.
    fn with<T>(
        &self,
        log: LogId,
        f: impl FnOnce(&mut Log) -> Result<T, SpillError>,
    ) -> Result<T, SpillError> {
        let mut logs = self.logs.lock().map_err(|_| SpillError::Io)?;
        let index = usize::try_from(log.0).map_err(|_| SpillError::Io)?;
        f(logs.get_mut(index).and_then(Option::as_mut).ok_or(SpillError::Io)?)
    }

    /// Uploads the part being staged, maybe empty, as the log's next part.
    fn upload(&self, log: &mut Log) -> Result<(), SpillError> {
        let (staged, len) = match log.staged.take() {
            Some(staged) => staged,
            None => (self.staging.create()?, 0),
        };
        let uploaded = self.upload_staged(log, staged, len);
        self.staging.delete(staged);
        log.etags.push(uploaded?);
        Ok(())
    }

    fn upload_staged(&self, log: &Log, staged: LogId, len: u64) -> Result<String, SpillError> {
        self.staging.seal(staged)?;
        let number = u16::try_from(log.etags.len() + 1).map_err(|_| SpillError::Io)?;
        let action =
            self.bucket.upload_part(Some(&self.credentials), &log.key, number, &log.upload_id);
        let mut reader = Staged { store: &self.staging, log: staged, offset: 0, len };
        let response = self
            .agent
            .put(action.sign(SIGNED_FOR).as_str())
            .header("content-length", len)
            .send(ureq::SendBody::from_reader(&mut reader))
            .map_err(|_| SpillError::Io)?;
        let etag = response.headers().get("etag").ok_or(SpillError::Io)?;
        Ok(etag.to_str().map_err(|_| SpillError::Io)?.to_owned())
    }

    fn complete(&self, log: &Log) -> Result<(), SpillError> {
        let action = self.bucket.complete_multipart_upload(
            Some(&self.credentials),
            &log.key,
            &log.upload_id,
            log.etags.iter().map(String::as_str),
        );
        let url = action.sign(SIGNED_FOR);
        self.agent.post(url.as_str()).send(action.body()).map_err(|_| SpillError::Io)?;
        Ok(())
    }
}

impl SpillStore for S3Spill {
    fn create(&self) -> Result<LogId, SpillError> {
        let n = NEXT_LOG.fetch_add(1, Ordering::Relaxed);
        let key = format!("{}{}-{n}", self.prefix, std::process::id());
        let action = self.bucket.create_multipart_upload(Some(&self.credentials), &key);
        let mut response = self
            .agent
            .post(action.sign(SIGNED_FOR).as_str())
            .send_empty()
            .map_err(|_| SpillError::Io)?;
        let body = response.body_mut().read_to_string().map_err(|_| SpillError::Io)?;
        let created = CreateMultipartUpload::parse_response(&body).map_err(|_| SpillError::Io)?;
        let log = Log {
            key,
            upload_id: created.upload_id().to_owned(),
            etags: Vec::new(),
            staged: None,
            len: 0,
            sealed: false,
        };
        let mut logs = self.logs.lock().map_err(|_| SpillError::Io)?;
        logs.push(Some(log));
        Ok(LogId(logs.len() as u64 - 1))
    }

    fn append(&self, log: LogId, bytes: &[u8]) -> Result<Block, SpillError> {
        self.with(log, |log| {
            if log.sealed {
                return Err(SpillError::Io);
            }
            let (staged, len) = match log.staged {
                Some(staged) => staged,
                None => (self.staging.create()?, 0),
            };
            self.staging.append(staged, bytes)?;
            let block = Block { offset: log.len, len: bytes.len() as u64 };
            log.len += block.len;
            log.staged = Some((staged, len + block.len));
            if len + block.len >= self.part_bytes {
                self.upload(log)?;
            }
            Ok(block)
        })
    }

    fn seal(&self, log: LogId) -> Result<(), SpillError> {
        self.with(log, |log| {
            // S3 needs a part, even an empty one.
            if log.staged.is_some() || log.etags.is_empty() {
                self.upload(log)?;
            }
            self.complete(log)?;
            log.sealed = true;
            Ok(())
        })
    }

    fn read(&self, log: LogId, block: Block, into: &mut [u8]) -> Result<(), SpillError> {
        self.with(log, |log| {
            let fits = block.offset.checked_add(block.len).is_some_and(|end| end <= log.len);
            if !log.sealed || !fits || block.len != into.len() as u64 {
                return Err(SpillError::Io);
            }
            if into.is_empty() {
                return Ok(());
            }
            let action = self.bucket.get_object(Some(&self.credentials), &log.key);
            let last = block.offset + block.len - 1;
            let mut response = self
                .agent
                .get(action.sign(SIGNED_FOR).as_str())
                .header("range", format!("bytes={}-{last}", block.offset))
                .call()
                .map_err(|_| SpillError::Io)?;
            response.body_mut().as_reader().read_exact(into).map_err(|_| SpillError::Io)
        })
    }

    fn delete(&self, log: LogId) {
        let Ok(mut logs) = self.logs.lock() else { return };
        let Some(slot) = usize::try_from(log.0).ok().and_then(|i| logs.get_mut(i)) else { return };
        let Some(log) = slot.take() else { return };
        if let Some((staged, _)) = log.staged {
            self.staging.delete(staged);
        }
        // Failing to delete leaves an object or upload behind, which a
        // bucket's lifecycle rules can clean up: nothing else is affected.
        let url = if log.sealed {
            self.bucket.delete_object(Some(&self.credentials), &log.key).sign(SIGNED_FOR)
        } else {
            let action = self.bucket.abort_multipart_upload(
                Some(&self.credentials),
                &log.key,
                &log.upload_id,
            );
            action.sign(SIGNED_FOR)
        };
        let _ = self.agent.delete(url.as_str()).call();
    }
}

/// Reads a staged part, for uploading it.
struct Staged<'s> {
    store: &'s FileSpill,
    log: LogId,
    offset: u64,
    len: u64,
}

impl Read for Staged<'_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = (self.len - self.offset).min(out.len() as u64);
        let block = Block { offset: self.offset, len: n };
        let n = usize::try_from(n).map_err(|_| std::io::ErrorKind::InvalidInput)?;
        self.store.read(self.log, block, &mut out[..n]).map_err(|_| std::io::ErrorKind::Other)?;
        self.offset += block.len;
        Ok(n)
    }
}

/// Tests against a real S3 API, such as Silo's: they run only when
/// `PIPIT_S3_ENDPOINT`, `PIPIT_S3_KEY` and `PIPIT_S3_SECRET` are set.
#[cfg(test)]
mod tests {
    use pipit_kernel::allocator::Heap;
    use pipit_kernel::buffer::Buffer;
    use pipit_kernel::column::{ColumnView, DataType};
    use pipit_kernel::spill::{read_column, write_column};
    use rusty_s3::UrlStyle;

    use super::*;

    /// A store in the test bucket, made if need be, or `None` if no S3 is
    /// configured.
    fn store(part_bytes: u64) -> Option<S3Spill> {
        let endpoint = std::env::var("PIPIT_S3_ENDPOINT").ok()?;
        let credentials = Credentials::new(
            std::env::var("PIPIT_S3_KEY").ok()?,
            std::env::var("PIPIT_S3_SECRET").ok()?,
        );
        let url = endpoint.parse().unwrap();
        let bucket = Bucket::new(url, UrlStyle::Path, "pipitdb-test", "us-east-1").unwrap();
        // Fails if it's there already, which is fine.
        let _ =
            ureq::put(bucket.create_bucket(&credentials).sign(SIGNED_FOR).as_str()).send_empty();
        let staging = FileSpill::new(std::env::temp_dir());
        Some(S3Spill::new(bucket, credentials, "spill-test-", staging, part_bytes))
    }

    fn column(rows: usize) -> ColumnView {
        let mut buffer = Buffer::allocate(&Heap, rows * 8).unwrap();
        for (o, v) in buffer.as_mut_slice::<i64>().iter_mut().zip(0..) {
            *o = v * 3;
        }
        ColumnView::new(DataType::Int64, buffer, None)
    }

    #[test]
    fn round_trips_columns_across_parts() {
        let Some(store) = store(PART_BYTES_MIN) else { return };
        // About 12 MiB: two full parts, and a last, smaller one.
        let columns = [column(600_000), column(600_000), column(400_000)];
        let log = store.create().unwrap();
        let spilled: Vec<_> =
            columns.iter().map(|c| write_column(&store, log, c).unwrap()).collect();
        store.seal(log).unwrap();
        for (column, spilled) in columns.iter().zip(&spilled) {
            let read = read_column(&Heap, &store, log, spilled).unwrap();
            assert_eq!(read.int64s(), column.int64s());
        }
        store.delete(log);
    }

    #[test]
    fn logs_are_read_only_once_sealed() {
        let Some(store) = store(PART_BYTES_MIN) else { return };
        let log = store.create().unwrap();
        let spilled = write_column(&store, log, &column(10)).unwrap();
        assert_eq!(read_column(&Heap, &store, log, &spilled).err(), Some(SpillError::Io));
        store.seal(log).unwrap();
        assert!(read_column(&Heap, &store, log, &spilled).is_ok());
        store.delete(log);
    }

    #[test]
    fn empty_logs_seal_and_unsealed_ones_are_aborted() {
        let Some(store) = store(PART_BYTES_MIN) else { return };
        let empty = store.create().unwrap();
        store.seal(empty).unwrap();
        store.delete(empty);
        let unsealed = store.create().unwrap();
        store.append(unsealed, &[1, 2, 3]).unwrap();
        store.delete(unsealed);
    }

    #[test]
    fn s3_rejects_parts_smaller_than_its_minimum() {
        // Checks the server behaves as S3 does: parts but the last must be
        // at least `PART_BYTES_MIN`.
        let Some(store) = store(1024) else { return };
        let log = store.create().unwrap();
        for _ in 0..3 {
            store.append(log, &[0; 1024]).unwrap();
        }
        assert_eq!(store.seal(log).err(), Some(SpillError::Io));
        store.delete(log);
    }
}
