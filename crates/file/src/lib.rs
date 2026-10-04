//! What pipitdb needs from the operating system's files. Unlike the kernel,
//! this uses `std`.

pub mod source;
pub mod spill;

use std::fs::File;

/// Fills `into` with `file`'s bytes from `offset`.
#[cfg(unix)]
pub(crate) fn read_at(file: &File, into: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, into, offset)
}

#[cfg(windows)]
pub(crate) fn read_at(file: &File, mut into: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    while !into.is_empty() {
        let read = std::os::windows::fs::FileExt::seek_read(file, into, offset)?;
        if read == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        into = &mut into[read..];
        offset += read as u64;
    }
    Ok(())
}

/// Elsewhere, such as Wasm, there are no files to read.
#[cfg(not(any(unix, windows)))]
pub(crate) fn read_at(_: &File, _: &mut [u8], _: u64) -> std::io::Result<()> {
    Err(std::io::ErrorKind::Unsupported.into())
}
