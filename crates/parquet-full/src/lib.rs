//! What the small Parquet reader leaves out, such as compression codecs,
//! plugged into it as its `Codec`.

use pipit_parquet::{Codec, Error};

/// Parquet's codec numbers.
const SNAPPY: u8 = 1;

/// The codecs this crate has: Snappy, the one most Parquet files use.
pub struct Codecs;

impl Codec for Codecs {
    fn decompress(&self, codec: u8, input: &[u8], output: &mut [u8]) -> Result<(), Error> {
        match codec {
            SNAPPY if snappy(input, output) => Ok(()),
            SNAPPY => Err(Error::Corrupt),
            _ => Err(Error::Unsupported),
        }
    }
}

/// Decompresses Snappy's `input` into all of `output`, or says it couldn't.
#[cfg(not(feature = "snappy-cpp"))]
fn snappy(input: &[u8], output: &mut [u8]) -> bool {
    snap::raw::Decoder::new().decompress(input, output).is_ok_and(|len| len == output.len())
}

#[cfg(feature = "snappy-cpp")]
fn snappy(input: &[u8], output: &mut [u8]) -> bool {
    let mut len = output.len();
    // SAFETY: the C library reads `input` and writes at most `len` bytes of
    // `output`, as their lengths say.
    let status = unsafe {
        snappy_src::snappy_uncompress(
            input.as_ptr().cast(),
            input.len(),
            output.as_mut_ptr().cast(),
            &raw mut len,
        )
    };
    status == snappy_src::snappy_status_SNAPPY_OK && len == output.len()
}

#[cfg(test)]
mod tests {
    use pipit_kernel::allocator::Heap;
    use pipit_kernel::column::Forms;
    use pipit_kernel::context::Context;
    use pipit_kernel::row_batch::RowBatch;
    use pipit_kernel::scannable::Scannable;
    use pipit_parquet::Uncompressed;
    use pipit_parquet::table::ParquetTable;

    use super::*;

    /// 5000 rows: `id` is the row, null every 7th from the 3rd, and `name`
    /// "name" and the row mod 13, null every 11th from the 2nd, compressed
    /// with Snappy, plain and dictionary-encoded as DuckDB writes them.
    const SNAPPY_FILE: &[u8] = include_bytes!("../tests/data/snappy.parquet");

    #[test]
    #[cfg_attr(miri, ignore = "too slow under Miri")]
    fn reads_snappy_pages() {
        let table = ParquetTable::open(&Heap, &Codecs, &[&SNAPPY_FILE]).unwrap();
        let mut context = Context::new(&Heap);
        let mut state =
            table.open(&mut context, &[(0, Forms::FLAT), (1, Forms::FLAT)], None).unwrap();
        let mut batch = RowBatch::new();
        let mut row = 0;
        while table.next(&mut context, &mut state, &mut batch).unwrap() {
            let (mut ids, mut names) = (batch.column(0).clone(), batch.column(1).clone());
            ids.make_in(&mut context, Forms::FLAT).unwrap();
            names.make_in(&mut context, Forms::FLAT).unwrap();
            for r in 0..batch.row_count() {
                assert_eq!(ids.is_null(r), row % 7 == 3);
                if !ids.is_null(r) {
                    assert_eq!(ids.int64s()[r as usize], row);
                }
                assert_eq!(names.is_null(r), row % 11 == 2);
                if !names.is_null(r) {
                    let name = format!("name{}", row % 13);
                    assert_eq!(names.string_values().get(r as usize), name.as_bytes());
                }
                row += 1;
            }
        }
        assert_eq!(row, 5000);
    }

    #[test]
    fn needs_codecs_for_compressed_pages() {
        let table = ParquetTable::open(&Heap, &Uncompressed, &[&SNAPPY_FILE]).unwrap();
        let mut context = Context::new(&Heap);
        let mut state = table.open(&mut context, &[(0, Forms::FLAT)], None).unwrap();
        let mut batch = RowBatch::new();
        let read = table.next(&mut context, &mut state, &mut batch);
        assert_eq!(read, Err(Error::Unsupported));
    }
}
