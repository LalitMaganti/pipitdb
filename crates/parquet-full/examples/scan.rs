//! Scans Parquet files as one table, a column at a time, and prints each
//! column's non-null count and checksum: the sum of its integers, or of its
//! strings' lengths. For checking the reader against other engines.
//!
//! Usage: `cargo run --release -p pipitdb-parquet-full --example scan -- FILE...`

use std::time::Instant;

use pipit_file::source::FileSource;
use pipit_kernel::allocator::Heap;
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::column::DataType;
use pipit_kernel::context::Context;
use pipit_kernel::row_batch::RowBatch;
use pipit_kernel::scannable::Scannable;
use pipit_parquet::table::ParquetTable;
use pipit_parquet_full::Codecs;

#[expect(clippy::cast_possible_truncation, reason = "floats are summed roughly")]
fn main() -> Result<(), String> {
    let mut files = Vec::new();
    for path in std::env::args().skip(1) {
        files.push(FileSource::open(&path).map_err(|e| format!("{path}: {e}"))?);
    }
    let sources: Vec<&dyn ByteSource> = files.iter().map(|file| file as &dyn ByteSource).collect();
    let table = ParquetTable::open(&Heap, &Codecs, &sources).map_err(|e| format!("{e:?}"))?;
    let start = Instant::now();
    for column in 0..table.column_count() {
        let mut context = Context::new(&Heap);
        let mut state = table.new_state(&mut context).map_err(|e| format!("{e:?}"))?;
        let mut batch = RowBatch::new();
        let (mut count, mut sum) = (0_u64, 0_i128);
        while table
            .next(&[column], &mut context, &mut state, &mut batch)
            .map_err(|e| format!("{}: {e:?}", table.column_name(column)))?
        {
            let values = batch.column(0).flatten(&mut context).map_err(|e| format!("{e:?}"))?;
            for row in (0..values.row_count()).filter(|&row| !values.is_null(row)) {
                count += 1;
                let r = row as usize;
                sum += match table.column_type(column) {
                    DataType::Int64 => i128::from(values.int64s()[r]),
                    DataType::Float64 => values.float64s()[r] as i128,
                    DataType::String => values.string_values().get(r).len() as i128,
                };
            }
        }
        println!("{}\t{count}\t{sum}", table.column_name(column));
    }
    eprintln!("scanned in {:?}", start.elapsed());
    Ok(())
}
