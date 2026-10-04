//! Inputs and helpers shared by the benchmarks in `benches/`.

use pipit_kernel::allocator::Heap;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType};
use pipit_kernel::pipeline::Pipeline;
use pipit_kernel::row_batch::{BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::DynScannable;
use pipit_kernel::step::{
    DynOperator, DynSource, DynTransform, Operator, Progress, Source, Step, Transform,
};
use pipit_operators::table::Table;
use pipit_pipesql::lexer::{Lexer, TokenKind};
use pipit_pipesql::parser::{parse_expression, parse_query};
use pipit_pipesql::registry::Registry;

pub mod real;

static REGISTRY: Registry = Registry::new(&[pipit_pipesql::stages::RELATIONAL]);

/// Queries shaped like real ones, repeated `count` times.
pub fn typical_queries(count: usize) -> String {
    "FROM slice |> WHERE dur > 1000 AND name != 'binder transaction' \
     |> EXTEND ts + dur AS end_ts |> AGGREGATE count(*) AS n GROUP BY track_id\n"
        .repeat(count)
}

/// Queries with long comments and string literals, repeated `count` times.
pub fn long_text(count: usize) -> String {
    "-- a long comment explaining what the query below does, in some detail\n\
     FROM t |> WHERE description = 'a long string literal, as found in real data'\n"
        .repeat(count)
}

/// A long `WHERE`-style expression mixing every operator.
pub fn expression(terms: usize) -> String {
    let term = "(dur * 2 + ts - 1000) / 3 >= 10.5 AND NOT name = 'binder transaction'";
    vec![term; terms].join(" OR ")
}

pub fn count_nodes(source: &[u8]) -> u32 {
    parse_expression(Heap, source).map_or(0, |ast| ast.node_count())
}

/// Three-stage queries, one per line.
pub fn queries(count: usize) -> String {
    "FROM slice |> WHERE dur > 1000 AND name != 'binder transaction' \
     |> SELECT ts, dur / 1000, name, count(*)\n"
        .repeat(count)
}

/// Parses each line as a query, as an embedder would.
pub fn count_query_nodes(source: &str) -> u32 {
    source
        .lines()
        .map(|query| {
            parse_query(Heap, &REGISTRY, query.as_bytes()).map_or(0, |ast| ast.node_count())
        })
        .sum()
}

pub fn count_tokens(source: &[u8]) -> u32 {
    let Ok(mut lexer) = Lexer::new(source) else { return 0 };
    let mut count = 0;
    while let Ok(token) = lexer.next_token() {
        if token.kind == TokenKind::End {
            break;
        }
        count += 1;
    }
    count
}

/// Batches of `BATCH_ROWS_MAX` rows, all sharing one column.
pub struct Repeat {
    column: ColumnView,
    batches: u32,
}

impl Repeat {
    #[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
    pub fn new(batches: u32) -> Repeat {
        let size_bytes = BATCH_ROWS_MAX as usize * 8;
        let values = Buffer::allocate(Heap, size_bytes).expect("allocates");
        Repeat { column: ColumnView::new(DataType::Int64, values, None), batches }
    }
}

impl Source for Repeat {
    type State = u32;

    fn new_state(&self) -> u32 {
        0
    }

    fn next(&self, batch: &mut RowBatch, made: &mut u32) -> bool {
        if *made == self.batches {
            return false;
        }
        *made += 1;
        batch.reset(BATCH_ROWS_MAX);
        batch.push_column(self.column.clone()).is_ok()
    }
}

/// Does nothing, to measure what a transform costs the pipeline.
pub struct PassTransform;

impl Transform for PassTransform {
    type State = ();

    fn new_state(&self) {}

    fn process(&self, _: &mut RowBatch, (): &mut ()) {}
}

/// Outputs its input, to measure what an operator costs the pipeline.
pub struct PassOperator;

impl Operator for PassOperator {
    type State = ();

    fn new_state(&self) {}

    fn execute(&self, input: &RowBatch, output: &mut RowBatch, (): &mut ()) -> Progress {
        output.reset(input.row_count());
        for i in 0..input.column_count() {
            let _ = output.push_column(input.column(i).clone());
        }
        Progress::NeedInput
    }
}

/// Runs `pipeline` once, returning the rows out.
pub fn run(pipeline: &Pipeline) -> u64 {
    let Ok(mut execution) = pipeline.start(Heap) else { return 0 };
    let mut batch = RowBatch::new();
    let mut rows = 0;
    while execution.next(&mut batch) {
        rows += u64::from(batch.row_count());
    }
    rows
}

/// `batches` full batches through `steps` pass-through transforms, or
/// operators.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn pass_pipeline(batches: u32, steps: usize, operators: bool) -> Pipeline<'static> {
    let mut owned = pipit_kernel::vec::Vec::fixed(Heap, steps).expect("allocates");
    for _ in 0..steps {
        let step = if operators {
            Step::Operator(DynOperator::new(Heap, PassOperator).expect("allocates"))
        } else {
            Step::Transform(DynTransform::new(Heap, PassTransform).expect("allocates"))
        };
        assert!(owned.push(step).is_ok(), "fixed for `steps`");
    }
    Pipeline::new(DynSource::new(Heap, Repeat::new(batches)).expect("allocates"), owned)
}

/// `row_groups` row groups of `rows` rows, with four columns. Kept for the
/// rest of the run, so pipelines can borrow it.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn table(row_groups: usize, rows: u32) -> &'static DynScannable<'static> {
    let values = Buffer::allocate(Heap, rows as usize * 8).expect("allocates");
    let column = ColumnView::new(DataType::Int64, values, None);
    let columns = [column.clone(), column.clone(), column.clone(), column];
    let row_groups = vec![columns.as_slice(); row_groups];
    let schema = [
        ("a", DataType::Int64),
        ("b", DataType::Int64),
        ("c", DataType::Int64),
        ("d", DataType::Int64),
    ];
    let table = Table::new(Heap, &schema, &row_groups).expect("allocates");
    Box::leak(Box::new(DynScannable::new(Heap, table).expect("allocates")))
}

/// Scans two of `table`'s columns.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn scan_pipeline(table: &'static DynScannable<'static>) -> Pipeline<'static> {
    let columns = pipit_kernel::vec::Vec::fixed_from(Heap, [3, 1].into_iter()).expect("allocates");
    let source = table.scan(Heap, columns).expect("allocates");
    Pipeline::new(source, pipit_kernel::vec::Vec::fixed(Heap, 0).expect("allocates"))
}
