//! Thrift's compact protocol, as Parquet's footer and page headers use it:
//! only what reading them needs.

pub const BINARY: u8 = 8;
pub const LIST: u8 = 9;
pub const STRUCT: u8 = 12;

/// How deep `skip` follows nested values before giving up, so a damaged
/// file can't exhaust the stack.
const DEPTH_MAX: u32 = 32;

/// A field `Cursor::fields` read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Value<'a> {
    /// The field isn't there.
    Missing,
    /// An integer, or a boolean as 0 or 1.
    Int(i64),
    /// A string or binary field's bytes.
    Bytes(&'a [u8]),
    /// A struct or list, to read from here.
    At(usize),
}

impl<'a> Value<'a> {
    pub fn int(self) -> Option<i64> {
        if let Value::Int(value) = self { Some(value) } else { None }
    }

    pub fn bytes(self) -> Option<&'a [u8]> {
        if let Value::Bytes(bytes) = self { Some(bytes) } else { None }
    }

    pub fn at(self) -> Option<usize> {
        if let Value::At(at) = self { Some(at) } else { None }
    }
}

pub struct Cursor<'a> {
    bytes: &'a [u8],
    pub pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(bytes: &'a [u8]) -> Cursor<'a> {
        Cursor { bytes, pos: 0 }
    }

    fn byte(&mut self) -> Option<u8> {
        let byte = *self.bytes.get(self.pos)?;
        self.pos += 1;
        Some(byte)
    }

    fn varint(&mut self) -> Option<u64> {
        let mut value = 0;
        for shift in (0..64).step_by(7) {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Some(value);
            }
        }
        None
    }

    /// An `i16`, `i32` or `i64`, which are all zigzag varints.
    pub fn int(&mut self) -> Option<i64> {
        let value = self.varint()?;
        Some((value >> 1).cast_signed() ^ -(value & 1).cast_signed())
    }

    pub fn binary(&mut self) -> Option<&'a [u8]> {
        let len = usize::try_from(self.varint()?).ok()?;
        let bytes = self.bytes.get(self.pos..self.pos.checked_add(len)?)?;
        self.pos += len;
        Some(bytes)
    }

    /// A list's elements' type, and how many there are.
    pub fn list(&mut self) -> Option<(u8, usize)> {
        let header = self.byte()?;
        let len = match header >> 4 {
            0xf => usize::try_from(self.varint()?).ok()?,
            len => usize::from(len),
        };
        Some((header & 0xf, len))
    }

    /// The next field of a struct: its id and type, or `None` at its end.
    /// `last` is the id before, which ids are written relative to.
    #[expect(clippy::option_option, reason = "damaged, or at a struct's end")]
    pub fn field(&mut self, last: &mut i16) -> Option<Option<(i16, u8)>> {
        let header = self.byte()?;
        if header == 0 {
            return Some(None);
        }
        let id = match header >> 4 {
            0 => i16::try_from(self.int()?).ok()?,
            delta => last.checked_add(i16::from(delta))?,
        };
        *last = id;
        Some(Some((id, header & 0xf)))
    }

    /// A cursor at `pos` of the same bytes, such as at a nested value
    /// `fields` found.
    pub fn at(&self, pos: usize) -> Cursor<'a> {
        Cursor { bytes: self.bytes, pos }
    }

    /// Reads a struct, putting each field whose id is in `ids` in the same
    /// place in `out`, and skipping the rest: so a struct is read by a list
    /// of ids, not code of its own.
    pub fn fields(&mut self, ids: &[i16], out: &mut [Value<'a>]) -> Option<()> {
        out.fill(Value::Missing);
        let mut last = 0;
        while let Some((id, kind)) = self.field(&mut last)? {
            let Some(i) = ids.iter().position(|&want| want == id) else {
                self.skip(kind)?;
                continue;
            };
            *out.get_mut(i)? = match kind {
                1 | 2 => Value::Int(i64::from(kind == 1)),
                3..=6 => Value::Int(self.int()?),
                BINARY => Value::Bytes(self.binary()?),
                LIST | STRUCT => {
                    let at = self.pos;
                    self.skip(kind)?;
                    Value::At(at)
                }
                _ => {
                    self.skip(kind)?;
                    Value::Missing
                }
            };
        }
        Some(())
    }

    pub fn skip(&mut self, kind: u8) -> Option<()> {
        self.skip_nested(kind, 0)
    }

    fn skip_nested(&mut self, kind: u8, depth: u32) -> Option<()> {
        if depth > DEPTH_MAX {
            return None;
        }
        match kind {
            // Booleans in fields carry their value in their type.
            1 | 2 => {}
            3 => {
                self.byte()?;
            }
            4..=6 => {
                self.varint()?;
            }
            7 => self.pos = self.pos.checked_add(8).filter(|&end| end <= self.bytes.len())?,
            BINARY => {
                self.binary()?;
            }
            LIST | 10 => {
                let (element, len) = self.list()?;
                for _ in 0..len {
                    // Booleans in lists are a byte each.
                    if element == 1 || element == 2 {
                        self.byte()?;
                    } else {
                        self.skip_nested(element, depth + 1)?;
                    }
                }
            }
            STRUCT => {
                let mut last = 0;
                while let Some((_, kind)) = self.field(&mut last)? {
                    self.skip_nested(kind, depth + 1)?;
                }
            }
            _ => return None,
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_fields_and_skips_what_it_doesnt_need() {
        // A struct of field 1, an i32 of -3; field 4, binary "ab"; field 5,
        // a list of two i32s; then its end.
        let bytes = [0x15, 0x05, 0x38, 0x02, b'a', b'b', 0x19, 0x25, 0x02, 0x04, 0x00];
        let mut cursor = Cursor::new(&bytes);
        let mut last = 0;
        assert_eq!(cursor.field(&mut last), Some(Some((1, 5))));
        assert_eq!(cursor.int(), Some(-3));
        assert_eq!(cursor.field(&mut last), Some(Some((4, BINARY))));
        assert_eq!(cursor.binary(), Some(&b"ab"[..]));
        assert_eq!(cursor.field(&mut last), Some(Some((5, LIST))));
        assert_eq!(cursor.skip(LIST), Some(()));
        assert_eq!(cursor.field(&mut last), Some(None));
        // Cut short, it fails rather than reading past the end.
        assert_eq!(Cursor::new(&bytes[..4]).skip(STRUCT), None);
    }
}
