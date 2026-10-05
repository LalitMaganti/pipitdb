//! `ParquetFile`: a file's footer, read once: its columns, and each row
//! group's size and column chunks.

use pipit_kernel::allocator::Allocator;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::column::DataType;
use pipit_kernel::names::{Name, Names};
use pipit_kernel::slow_vec::SlowVec;

use crate::Error;
use crate::thrift::{Cursor, Value};

/// How values are stored, as Parquet names them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Physical {
    Boolean,
    Int32,
    Int64,
    Int96,
    Float,
    Double,
    ByteArray,
    FixedLenByteArray,
}

/// What a column's values mean, beyond how they're stored, where that
/// changes how they're read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Logical {
    /// As stored: integers are signed.
    Plain,
    /// Unsigned integers.
    Unsigned,
    /// Something reading as stored would get wrong, such as decimals.
    Other,
}

#[derive(Clone, Copy, Debug)]
pub struct Column {
    pub name: Name,
    pub physical: Physical,
    pub logical: Logical,
    /// Whether rows can be null.
    pub optional: bool,
}

impl Column {
    /// What the column reads as, if this reader can read it. Unsigned 64-bit
    /// integers can't all be held.
    pub fn data_type(&self) -> Option<DataType> {
        match (self.physical, self.logical) {
            (_, Logical::Other)
            | (Physical::Int64, Logical::Unsigned)
            | (Physical::Boolean | Physical::Int96 | Physical::FixedLenByteArray, _) => None,
            (Physical::Int32 | Physical::Int64, _) => Some(DataType::Int64),
            (Physical::Float | Physical::Double, _) => Some(DataType::Float64),
            (Physical::ByteArray, _) => Some(DataType::String),
        }
    }
}

/// A column's values in a row group.
#[derive(Clone, Copy, Debug)]
pub struct Chunk {
    /// Where its first page starts, a dictionary page if it has one, and how
    /// many bytes its pages take.
    pub start: u64,
    pub len: u64,
    /// Parquet's number for how its pages are compressed: 0 for none.
    pub codec: u8,
    pub values: u64,
    /// The smallest and largest value, for integer columns that record them.
    pub min: Option<i64>,
    pub max: Option<i64>,
}

pub struct ParquetFile {
    names: Names,
    columns: SlowVec<Column>,
    group_rows: SlowVec<u64>,
    /// Row group by row group, a chunk for each column.
    chunks: SlowVec<Chunk>,
}

/// The magic bytes at the start and end of every Parquet file.
const MAGIC: &[u8; 4] = b"PAR1";

/// The most columns, row groups, and column chunks in all, a file can have.
const COLUMNS_MAX: usize = 1 << 12;
const GROUPS_MAX: usize = 1 << 20;
const CHUNKS_MAX: usize = 1 << 24;

impl ParquetFile {
    /// Reads `source`'s footer, with memory from `allocator`.
    pub fn open(allocator: &dyn Allocator, source: &dyn ByteSource) -> Result<ParquetFile, Error> {
        // The footer, then its length and the magic bytes again.
        let mut tail = [0; 8];
        let len = source.len();
        source.read(len.checked_sub(8).ok_or(Error::Corrupt)?, &mut tail)?;
        let footer_len = u64::from(u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]));
        if &tail[4..] != MAGIC || footer_len + 12 > len {
            return Err(Error::Corrupt);
        }
        let size = usize::try_from(footer_len).map_err(|_| Error::Corrupt)?;
        let mut footer = Buffer::allocate(allocator, size)?;
        source.read(len - 8 - footer_len, footer.as_mut_slice::<u8>())?;
        parse(allocator, footer.as_slice::<u8>()).ok_or(Error::Corrupt)?
    }

    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    pub fn name(&self, column: &Column) -> &str {
        self.names.get(column.name)
    }

    pub fn row_groups(&self) -> usize {
        self.group_rows.len()
    }

    pub fn group_rows(&self, group: usize) -> u64 {
        *at!(self.group_rows, group)
    }

    pub fn chunk(&self, group: usize, column: usize) -> &Chunk {
        at!(self.chunks, group * self.columns.len() + column)
    }
}

/// The footer's `FileMetaData`: `None` if it's damaged, and an error if it
/// can't be read.
fn parse(allocator: &dyn Allocator, footer: &[u8]) -> Option<Result<ParquetFile, Error>> {
    let (Ok(mut names), Ok(mut columns), Ok(mut group_rows), Ok(mut chunks)) = (
        Names::new(allocator, 1 << 16),
        SlowVec::new(allocator, COLUMNS_MAX),
        SlowVec::new(allocator, GROUPS_MAX),
        SlowVec::new(allocator, CHUNKS_MAX),
    ) else {
        return Some(Err(Error::OutOfMemory));
    };
    let c = Cursor::new(footer);
    // `FileMetaData`: its schema, and its row groups.
    let mut file = [Value::Missing; 2];
    c.at(0).fields(&[2, 4], &mut file)?;
    let mut schema = c.at(file[0].at()?);
    let (_, len) = schema.list()?;
    // `SchemaElement`s: its type, repetition, name, children's count, and
    // converted and logical types. The first is the root; flat schemas have
    // columns only under it.
    let mut element = [Value::Missing; 6];
    for i in 0..len {
        schema.fields(&[1, 3, 4, 5, 6, 10], &mut element)?;
        if i == 0 {
            continue;
        }
        let physical = match element[0].int() {
            Some(0) => Physical::Boolean,
            Some(1) => Physical::Int32,
            Some(2) => Physical::Int64,
            Some(3) => Physical::Int96,
            Some(4) => Physical::Float,
            Some(5) => Physical::Double,
            Some(6) => Physical::ByteArray,
            Some(7) => Physical::FixedLenByteArray,
            _ => return Some(Err(Error::Unsupported)),
        };
        // Nested and repeated columns need repetition levels.
        if element[3].int().unwrap_or(0) != 0 || element[1].int() == Some(2) {
            return Some(Err(Error::Unsupported));
        }
        let optional = element[1].int() == Some(1);
        let logical = logical(&c, element[4], element[5])?;
        let Ok(name) = names.add(core::str::from_utf8(element[2].bytes()?).ok()?) else {
            return Some(Err(Error::OutOfMemory));
        };
        if columns.push(Column { name, physical, logical, optional }).is_err() {
            return Some(Err(Error::Unsupported));
        }
    }
    let mut groups = c.at(file[1].at()?);
    let (_, len) = groups.list()?;
    // `RowGroup`s: their column chunks, and rows.
    let mut group = [Value::Missing; 2];
    for _ in 0..len {
        groups.fields(&[1, 3], &mut group)?;
        let mut list = c.at(group[0].at()?);
        if list.list()?.1 != columns.len() {
            return None;
        }
        for column in columns.iter() {
            if chunks.push(column_chunk(&mut list, *column)?).is_err() {
                return Some(Err(Error::Unsupported));
            }
        }
        let rows = u64::try_from(group[1].int()?).ok()?;
        if group_rows.push(rows).is_err() {
            return Some(Err(Error::Unsupported));
        }
    }
    Some(Ok(ParquetFile { names, columns, group_rows, chunks }))
}

/// A column's `Logical` kind, from its converted type, and its logical type
/// at `logical`, if any.
fn logical(c: &Cursor<'_>, converted: Value<'_>, logical: Value<'_>) -> Option<Logical> {
    // `ConvertedType`: 5 is a decimal, 11 to 14 unsigned integers.
    match converted.int() {
        Some(5) => return Some(Logical::Other),
        Some(11..=14) => return Some(Logical::Unsigned),
        _ => {}
    }
    let Some(at) = logical.at() else { return Some(Logical::Plain) };
    // `LogicalType`, a union: 5 is a decimal, 10 an integer, with whether
    // it's signed as its field 2.
    let mut kind = [Value::Missing; 2];
    c.at(at).fields(&[5, 10], &mut kind)?;
    if kind[0].at().is_some() {
        return Some(Logical::Other);
    }
    if let Some(at) = kind[1].at() {
        let mut signed = [Value::Missing];
        c.at(at).fields(&[2], &mut signed)?;
        if signed[0].int() == Some(0) {
            return Some(Logical::Unsigned);
        }
    }
    Some(Logical::Plain)
}

/// A `ColumnChunk`, at `c`, of `column`.
fn column_chunk(c: &mut Cursor<'_>, column: Column) -> Option<Chunk> {
    let mut chunk = [Value::Missing];
    c.fields(&[3], &mut chunk)?;
    // `ColumnMetaData`: its codec, values, size, the offsets of its data and
    // dictionary pages, and its statistics.
    let mut meta = [Value::Missing; 6];
    c.at(chunk[0].at()?).fields(&[4, 5, 7, 9, 11, 12], &mut meta)?;
    let unsigned = |value: Value<'_>| u64::try_from(value.int()?).ok();
    let data = unsigned(meta[3])?;
    let start = unsigned(meta[4]).map_or(data, |dictionary| dictionary.min(data));
    let (mut min, mut max) = (None, None);
    if let Some(at) = meta[5].at() {
        // `Statistics`: the current max and min are fields 5 and 6, ordered
        // as the column's values are. 1 and 2 are older and always ordered
        // as signed, so are only right for signed integers.
        let mut stats = [Value::Missing; 4];
        c.at(at).fields(&[5, 6, 1, 2], &mut stats)?;
        let integer = |value: Value<'_>| integer(value.bytes()?, column);
        let signed = column.logical == Logical::Plain;
        max = integer(stats[0]).or_else(|| integer(stats[2]).filter(|_| signed));
        min = integer(stats[1]).or_else(|| integer(stats[3]).filter(|_| signed));
    }
    Some(Chunk {
        start,
        len: unsigned(meta[2])?,
        codec: u8::try_from(meta[0].int().unwrap_or(0)).ok()?,
        values: unsigned(meta[1]).unwrap_or(0),
        min,
        max,
    })
}

/// A statistic of `column`, an integer column it can read, as its value.
fn integer(bytes: &[u8], column: Column) -> Option<i64> {
    column.data_type().filter(|&data_type| data_type == DataType::Int64)?;
    Some(match (column.physical, column.logical) {
        (Physical::Int32, Logical::Unsigned) => {
            i64::from(u32::from_le_bytes(bytes.try_into().ok()?))
        }
        (Physical::Int32, _) => i64::from(i32::from_le_bytes(bytes.try_into().ok()?)),
        _ => i64::from_le_bytes(bytes.try_into().ok()?),
    })
}

#[cfg(test)]
mod tests {
    use pipit_file::source::FileSource;
    use pipit_kernel::allocator::Heap;

    use super::*;

    /// 5000 rows of an `id`, `half` of it, and a `name`, in row groups of
    /// 2048, uncompressed, as DuckDB writes them.
    const SMALL: &[u8] = include_bytes!("../tests/data/small.parquet");

    fn small() -> ParquetFile {
        ParquetFile::open(&Heap, &SMALL).unwrap()
    }

    extern crate std;

    #[test]
    fn reads_the_schema() {
        let file = small();
        let columns: std::vec::Vec<_> =
            file.columns().iter().map(|c| (file.name(c), c.physical, c.optional)).collect();
        assert_eq!(
            columns,
            [
                ("id", Physical::Int64, true),
                ("half", Physical::Double, true),
                ("name", Physical::ByteArray, true),
            ]
        );
        let types = [DataType::Int64, DataType::Float64, DataType::String];
        assert!(file.columns().iter().zip(types).all(|(c, t)| c.data_type() == Some(t)));
    }

    #[test]
    fn reads_row_groups_and_their_statistics() {
        let file = small();
        let rows: std::vec::Vec<u64> = (0..file.row_groups()).map(|g| file.group_rows(g)).collect();
        assert_eq!(rows, [2048, 2048, 904]);
        let ids: std::vec::Vec<_> =
            (0..3).map(|g| (file.chunk(g, 0).min, file.chunk(g, 0).max)).collect();
        assert_eq!(
            ids,
            [(Some(0), Some(2047)), (Some(2048), Some(4095)), (Some(4096), Some(4999))]
        );
        // Only integer columns' statistics are kept.
        assert_eq!((file.chunk(0, 1).min, file.chunk(0, 2).max), (None, None));
        assert_eq!((file.chunk(0, 0).start, file.chunk(1, 0).start), (4, 54277));
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri can't use real files")]
    fn reads_through_a_file() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/small.parquet");
        let file = ParquetFile::open(&Heap, &FileSource::open(path).unwrap()).unwrap();
        assert_eq!(file.row_groups(), 3);
    }

    #[test]
    fn rejects_what_isnt_parquet() {
        let bytes: &[u8] = b"not parquet at all";
        assert_eq!(ParquetFile::open(&Heap, &bytes).err(), Some(Error::Corrupt));
    }
}
