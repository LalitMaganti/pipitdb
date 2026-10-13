//! Runs the queries of a suite, such as ClickBench or TPC-H, that pipitdb can
//! run over Parquet files, printing each row of each result and, to stderr,
//! how long each took. `check_queries.py` compares the rows with DuckDB's.
//!
//! Usage: `cargo run --profile speed -p pipitdb-bench --example suite -- [QUERIES] FILE...`
//!
//! `QUERIES` is a file of them, as `clickbench/queries.tsv` is, which is the
//! default. Each file is a table named after it, less a trailing `_N`, so
//! `hits_0.parquet` and `hits_1.parquet` are both `hits`.
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

static REGISTRY: Registry =
    Registry::new(&[pipit_pipesql::stages::RELATIONAL, pipit_pipesql::stages::AGGREGATION]);

/// ClickBench's queries: each one's name, PipeSQL, if pipitdb can run it yet,
/// and SQL, a line each.
const CLICKBENCH: &str = include_str!("../clickbench/queries.tsv");

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

/// The tables, by name.
struct Tables<'a>(Vec<(String, DynScannable<'a>)>);

impl Catalog for Tables<'_> {
    fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
        self.0.iter().find(|(table, _)| table == name).map(|(_, scannable)| scannable)
    }
}

/// The table `path` is part of: its file's name, less its extension and any
/// trailing `_N`.
fn table_name(path: &str) -> String {
    let stem = std::path::Path::new(path).file_stem().map_or(path.into(), |s| s.to_string_lossy());
    let trimmed = stem.trim_end_matches(|c: char| c.is_ascii_digit());
    match trimmed.strip_suffix('_') {
        Some(name) if trimmed.len() < stem.len() => name.into(),
        _ => stem.into(),
    }
}

/// The queries in `args`'s first, if it's a queries file, which it removes,
/// or else ClickBench's.
fn take_queries(args: &mut Vec<String>) -> Result<String, String> {
    let is_queries = args.first().is_some_and(|path| {
        std::path::Path::new(path).extension().is_some_and(|e| e.eq_ignore_ascii_case("tsv"))
    });
    if !is_queries {
        return Ok(CLICKBENCH.into());
    }
    let path = args.remove(0);
    std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))
}

/// The files at `paths`, opened, by the table each is part of.
fn open_files(paths: &[String]) -> Result<Vec<(String, Vec<FileSource>)>, String> {
    let mut files: Vec<(String, Vec<FileSource>)> = Vec::new();
    for path in paths {
        let file = FileSource::open(path).map_err(|e| format!("{path}: {e}"))?;
        let name = table_name(path);
        match files.iter_mut().find(|(table, _)| *table == name) {
            Some((_, table)) => table.push(file),
            None => files.push((name, vec![file])),
        }
    }
    Ok(files)
}

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let queries = take_queries(&mut args)?;
    let files = open_files(&args)?;
    let sources: Vec<(&str, Vec<&dyn ByteSource>)> = files
        .iter()
        .map(|(name, table)| (name.as_str(), table.iter().map(|f| f as &dyn ByteSource).collect()))
        .collect();
    let codecs = Counting::default();
    let mut tables = Vec::new();
    for (name, sources) in &sources {
        // A table pipitdb can't read is left out, so only queries reading it fail.
        match ParquetTable::open(&Heap, &codecs, sources) {
            Ok(table) => {
                let table = DynScannable::new(&Heap, table).map_err(|e| format!("{e:?}"))?;
                tables.push(((*name).to_string(), table));
            }
            Err(error) => eprintln!("{name}: {error:?}"),
        }
    }
    let tables = Tables(tables);
    for line in queries.lines() {
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
            run(&tables, query, &QueryAllocators::new(&Heap))?;
        }
        let limit = LimitAllocator::new(&Heap, usize::MAX);
        // Counts what the query holds at its peak, for the report.
        let allocators = QueryAllocators::new(&limit);
        codecs.pages.set(0);
        codecs.bytes.set(0);
        let start = Instant::now();
        match run(&tables, query, &allocators) {
            Ok(columns) => {
                // Timed up to the results, as other engines' times are; turning
                // them into text isn't.
                let elapsed = start.elapsed();
                // QUIET skips printing, for timing and profiling.
                let rows = if std::env::var("QUIET").is_ok() { 0 } else { columns[0].len() };
                for row in 0..rows {
                    let row: Vec<String> = columns.iter().map(|column| column.text(row)).collect();
                    println!("{number}\t{}", row.join("\t"));
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

/// A column of a result: its kept rows' values, as other engines' results
/// are, so it takes memory for them alone.
struct Output {
    /// What its rows hold: that of the columns they're from.
    data_type: DataType,
    /// Whether each row is null.
    nulls: Vec<bool>,
    /// Each row's integer, or float's bits.
    words: Vec<i64>,
    /// Each row's string, one after another.
    bytes: Vec<u8>,
    /// Where each row's string ends in `bytes`.
    ends: Vec<usize>,
}

impl Output {
    fn new() -> Output {
        let data_type = DataType::Int64;
        Output {
            data_type,
            nulls: Vec::new(),
            words: Vec::new(),
            bytes: Vec::new(),
            ends: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.nulls.len()
    }

    /// Adds row `row` of `column`, reading its value through its form.
    fn push(&mut self, column: &ColumnView, row: u32) {
        self.data_type = column.data_type();
        let null = column.is_null(row);
        self.nulls.push(null);
        let values = column.values();
        let value = match column.form() {
            Form::Flat => row as usize,
            Form::Constant => 0,
            Form::Dictionary(indices) => indices[row as usize] as usize,
        };
        match self.data_type {
            DataType::Int64 => self.words.push(if null { 0 } else { values.int64s()[value] }),
            DataType::Float64 => {
                let float = if null { 0.0 } else { values.float64s()[value] };
                self.words.push(float.to_bits().cast_signed());
            }
            DataType::String => {
                if !null {
                    self.bytes.extend_from_slice(values.strings().get(value));
                }
                self.ends.push(self.bytes.len());
            }
        }
    }

    fn text(&self, row: usize) -> String {
        match self.data_type {
            _ if self.nulls[row] => "NULL".into(),
            DataType::Int64 => self.words[row].to_string(),
            DataType::Float64 => f64::from_bits(self.words[row].cast_unsigned()).to_string(),
            DataType::String => {
                let start = if row == 0 { 0 } else { self.ends[row - 1] };
                String::from_utf8_lossy(&self.bytes[start..self.ends[row]]).into()
            }
        }
    }
}

/// The result of `query`: a column of its kept rows for each it outputs,
/// built as each batch comes, as other engines build theirs.
fn run(tables: &Tables, query: &str, allocators: &QueryAllocators) -> Result<Vec<Output>, String> {
    let mut plan =
        compile(&Heap, &REGISTRY, tables, query.as_bytes()).map_err(|e| format!("{e:?}"))?;
    optimize(&Heap, &mut plan).map_err(|e| format!("{e:?}"))?;
    let physical = lower(&Heap, &plan).map_err(|e| format!("{e:?}"))?;
    let mut execution = physical.pipeline().start(allocators).map_err(|e| format!("{e:?}"))?;
    let mut columns: Vec<Output> = physical.columns().iter().map(|_| Output::new()).collect();
    let mut batch = RowBatch::new();
    while execution.next(&mut batch).map_err(|e| format!("{e:?}"))? {
        let kept: Vec<u32> = match batch.selection().kept() {
            Kept::All => (0..batch.row_count()).collect(),
            Kept::None => Vec::new(),
            Kept::Select(rows) => rows.iter().map(|&row| u32::from(row)).collect(),
        };
        for (output, c) in columns.iter_mut().zip(physical.columns()) {
            let column = batch.column(c.position);
            for &row in &kept {
                output.push(column, row);
            }
        }
    }
    Ok(columns)
}
