//! `Names`: names copied into memory of their own, each found by a `Name`.

use crate::allocator::{AllocError, Allocator};
use crate::vec::Vec;

pub struct Names {
    bytes: Vec<u8>,
}

/// Where a name is in its `Names`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Name {
    start: u32,
    len: u32,
}

impl Names {
    /// Room for up to `max_bytes` of names, a power of two.
    pub fn new<A: Allocator + Clone + 'static>(
        allocator: A,
        max_bytes: usize,
    ) -> Result<Names, AllocError> {
        Ok(Names { bytes: Vec::new(allocator, max_bytes)? })
    }

    /// Room for exactly `bytes` of names, which never grows.
    pub fn fixed<A: Allocator + Clone + 'static>(
        allocator: A,
        bytes: usize,
    ) -> Result<Names, AllocError> {
        Ok(Names { bytes: Vec::fixed(allocator, bytes)? })
    }

    /// Copies `name` in.
    pub fn add(&mut self, name: &str) -> Result<Name, AllocError> {
        let (Ok(start), Ok(len)) = (u32::try_from(self.bytes.len()), u32::try_from(name.len()))
        else {
            return Err(AllocError);
        };
        self.bytes.extend_from_slice(name.as_bytes())?;
        Ok(Name { start, len })
    }

    pub fn get(&self, name: Name) -> &str {
        let start = name.start as usize;
        let bytes = at!(self.bytes, start..start + name.len as usize);
        // SAFETY: each name's bytes were copied whole from a `str`.
        unsafe { core::str::from_utf8_unchecked(bytes) }
    }
}

#[cfg(test)]
mod tests {
    use crate::allocator::Heap;

    use super::*;

    #[test]
    fn keeps_names() {
        let mut names = Names::new(Heap, 16).unwrap();
        let a = names.add("ts").unwrap();
        let b = names.add("dur").unwrap();
        assert_eq!((names.get(a), names.get(b)), ("ts", "dur"));
        assert_eq!(names.add("too long for the rest").err(), Some(AllocError));
    }
}
