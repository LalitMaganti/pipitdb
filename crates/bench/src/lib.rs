//! Inputs and helpers shared by the benchmarks in `benches/`.

use pipit_kernel::allocator::Heap;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType};
use pipit_kernel::lexer::{Lexer, TokenKind};
use pipit_kernel::parser::{parse_expression, parse_query};
use pipit_kernel::pipeline::Pipeline;
use pipit_kernel::registry::Registry;
use pipit_kernel::row_batch::{BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::step::{
    DynOperator, DynSource, DynTransform, Operator, Progress, Source, Step, Transform,
};

static REGISTRY: Registry = Registry::new(&[pipit_std::RELATIONAL]);

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

/// Runs `source` through `steps`, returning the rows out.
pub fn run_pipeline(source: &Repeat, steps: &[Step]) -> u64 {
    let pipeline = Pipeline::new(DynSource::new(source), steps);
    let Ok(mut execution) = pipeline.start(Heap) else { return 0 };
    let mut batch = RowBatch::new();
    let mut rows = 0;
    while execution.next(&mut batch) {
        rows += u64::from(batch.row_count());
    }
    rows
}

/// `count` pass-through transforms, or operators.
pub fn pass_steps(count: usize, operators: bool) -> Vec<Step<'static>> {
    let step = || {
        if operators {
            Step::Operator(DynOperator::new(&PassOperator))
        } else {
            Step::Transform(DynTransform::new(&PassTransform))
        }
    };
    (0..count).map(|_| step()).collect()
}
