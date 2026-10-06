//! Runs the ClickBench queries pipitdb can run over ClickBench's Parquet
//! files, as the table `hits`, printing each row of each result and, to
//! stderr, how long each took. `clickbench/check.py` compares the rows with
//! DuckDB's.
//!
//! Usage: `cargo run --profile speed -p pipitdb-bench --example clickbench -- FILE...`
//!
//! `QUERY` runs only that query, `REPEAT` runs each that many times and times
//! the last, and `QUIET` doesn't print rows.

use std::cell::Cell;
use std::time::Instant;

use pipit_file::source::FileSource;
use pipit_kernel::allocator::{Heap, LimitAllocator};
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::column::{ColumnView, DataType, Form};
use pipit_kernel::lower::lower;
use pipit_kernel::optimize::optimize;
use pipit_kernel::query_allocators::QueryAllocators;
use pipit_kernel::row_batch::RowBatch;
use pipit_kernel::scannable::{Catalog, DynScannable};
use pipit_kernel::selection::Kept;
use pipit_parquet::Codec;
use pipit_parquet::table::ParquetTable;
use pipit_parquet_full::Codecs;
use pipit_pipesql::compile::compile;
use pipit_pipesql::registry::Registry;

static REGISTRY: Registry = Registry::new(&[pipit_pipesql::stages::RELATIONAL]);

/// Each query's name, PipeSQL, if pipitdb can run it yet, and SQL, a line
/// each.
const QUERIES: &str = include_str!("../clickbench/queries.tsv");

/// Decompresses as `Codecs` does, counting the pages and the bytes they
/// decompress to, which tells how much a query read.
#[derive(Default)]
struct Counting {
    pages: Cell<u64>,
    bytes: Cell<u64>,
}

impl Codec for Counting {
    fn decompress(
        &self,
        codec: u8,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<(), pipit_parquet::Error> {
        self.pages.set(self.pages.get() + 1);
        self.bytes.set(self.bytes.get() + output.len() as u64);
        Codecs.decompress(codec, input, output)
    }
}

struct Hits<'a>(DynScannable<'a>);

impl Catalog for Hits<'_> {
    fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
        (name == "hits").then_some(&self.0)
    }
}

fn main() -> Result<(), String> {
    let mut files = Vec::new();
    for path in std::env::args().skip(1) {
        files.push(FileSource::open(&path).map_err(|e| format!("{path}: {e}"))?);
    }
    let sources: Vec<&dyn ByteSource> = files.iter().map(|file| file as &dyn ByteSource).collect();
    let codecs = Counting::default();
    let table = ParquetTable::open(&Heap, &codecs, &sources).map_err(|e| format!("{e:?}"))?;
    let hits = Hits(DynScannable::new(&Heap, table).map_err(|e| format!("{e:?}"))?);
    for line in QUERIES.lines() {
        let mut fields = line.split('\t');
        let (Some(number), Some(query)) = (fields.next(), fields.next()) else { continue };
        // A query with no PipeSQL is one pipitdb can't run yet; other engines
        // still run its SQL.
        if query.is_empty() {
            continue;
        }
        if std::env::var("QUERY").is_ok_and(|only| only != number) {
            continue;
        }
        let repeat = std::env::var("REPEAT").ok().and_then(|n| n.parse().ok()).unwrap_or(1);
        for _ in 1..repeat {
            run(&hits, query, &QueryAllocators::new(&Heap))?;
        }
        let limit = LimitAllocator::new(&Heap, usize::MAX);
        // Results hold blocks from it, so it outlives them.
        let allocators = QueryAllocators::new(&limit);
        codecs.pages.set(0);
        codecs.bytes.set(0);
        let start = Instant::now();
        match run(&hits, query, &allocators) {
            Ok(batches) => {
                // Timed up to the results, as other engines' times are; turning
                // them into text isn't.
                let elapsed = start.elapsed();
                // QUIET skips printing, for timing and profiling.
                let quiet = std::env::var("QUIET").is_ok();
                for (columns, kept) in batches.into_iter().filter(|_| !quiet) {
                    for row in kept {
                        let row: Vec<String> =
                            columns.iter().map(|column| text(column, row)).collect();
                        println!("{number}\t{}", row.join("\t"));
                    }
                }
                #[expect(clippy::cast_precision_loss, reason = "a report")]
                let peak = limit.peak() as f64 / 1e6;
                #[expect(clippy::cast_precision_loss, reason = "a report")]
                let read = codecs.bytes.get() as f64 / 1e6;
                let pages = codecs.pages.get();
                eprintln!(
                    "Q{number}: {elapsed:?} peak {peak:.1} MB read {read:.1} MB in {pages} pages"
                );
            }
            Err(error) => eprintln!("Q{number}: {error}"),
        }
    }
    Ok(())
}

/// A batch's columns, in the forms the plan made them, and its kept rows.
type Batch = (Vec<ColumnView>, Vec<u32>);

/// The batches `query` makes: their columns, as made, and kept rows. As
/// another engine's results, they're columns, read only for the rows kept.
/// Runs it with memory from `allocators`, which must outlive the batches.
fn run(hits: &Hits, query: &str, allocators: &QueryAllocators) -> Result<Vec<Batch>, String> {
    let mut plan =
        compile(&Heap, &REGISTRY, hits, query.as_bytes()).map_err(|e| format!("{e:?}"))?;
    optimize(&Heap, &mut plan).map_err(|e| format!("{e:?}"))?;
    let physical = lower(&Heap, &plan).map_err(|e| format!("{e:?}"))?;
    let mut execution = physical.pipeline().start(allocators).map_err(|e| format!("{e:?}"))?;
    let mut batch = RowBatch::new();
    let mut rows = Vec::new();
    while execution.next(&mut batch).map_err(|e| format!("{e:?}"))? {
        let kept: Vec<u32> = match batch.selection().kept() {
            Kept::All => (0..batch.row_count()).collect(),
            Kept::None => Vec::new(),
            Kept::Select(rows) => rows.iter().map(|&row| u32::from(row)).collect(),
        };
        let columns = physical.columns().iter().map(|c| batch.column(c.position).clone());
        rows.push((columns.collect(), kept));
    }
    Ok(rows)
}

fn text(column: &ColumnView, row: u32) -> String {
    let values = column.values();
    let value = match column.form() {
        Form::Flat => row as usize,
        Form::Constant => 0,
        Form::Dictionary(indices) => indices[row as usize] as usize,
    };
    match column.data_type() {
        _ if column.is_null(row) => "NULL".into(),
        DataType::Int64 => values.int64s()[value].to_string(),
        DataType::Float64 => values.float64s()[value].to_string(),
        DataType::String => String::from_utf8_lossy(values.strings().get(value)).into(),
    }
}
