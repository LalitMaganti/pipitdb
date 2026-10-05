//! `Names`: names copied into memory of their own, each found by a `Name`.

use crate::allocator::{AllocError, Allocator};
use crate::slow_vec::SlowVec;

pub struct Names {
    bytes: SlowVec<u8>,
}

/// Where a name is in its `Names`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Name {
    start: u32,
    len: u32,
}

impl Names {
    /// Room for up to `max_bytes` of names, a power of two.
    pub fn new(allocator: &dyn Allocator, max_bytes: usize) -> Result<Names, AllocError> {
        Ok(Names { bytes: SlowVec::new(allocator, max_bytes)? })
    }

    /// Room for exactly `bytes` of names, which never grows.
    pub fn fixed(allocator: &dyn Allocator, bytes: usize) -> Result<Names, AllocError> {
        Ok(Names { bytes: SlowVec::fixed(allocator, bytes)? })
    }

    /// Copies `name` in.
    pub fn add(&mut self, name: &str) -> Result<Name, AllocError> {
        self.add_parts(&[name])
    }

    /// Copies in the name made of `parts`, one after another. Fails, adding
    /// none of it, if there isn't room.
    pub fn add_parts(&mut self, parts: &[&str]) -> Result<Name, AllocError> {
        let len: usize = parts.iter().map(|part| part.len()).sum();
        let (Ok(start), Ok(len32)) = (u32::try_from(self.bytes.len()), u32::try_from(len)) else {
            return Err(AllocError);
        };
        for part in parts {
            if self.bytes.extend_from_slice(part.as_bytes()).is_err() {
                while self.bytes.len() > start as usize {
                    self.bytes.pop();
                }
                return Err(AllocError);
            }
        }
        Ok(Name { start, len: len32 })
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
        let mut names = Names::new(&Heap, 16).unwrap();
        let a = names.add("ts").unwrap();
        let b = names.add("dur").unwrap();
        assert_eq!((names.get(a), names.get(b)), ("ts", "dur"));
        assert_eq!(names.add("too long for the rest").err(), Some(AllocError));
        let c = names.add_parts(&["f(", "x)"]).unwrap();
        assert_eq!(names.get(c), "f(x)");
        // What doesn't fit isn't added in part.
        assert_eq!(names.add_parts(&["abcd", "efgh"]).err(), Some(AllocError));
        assert_eq!(names.add("abc").map(|d| names.get(d) == "abc"), Ok(true));
    }
}
