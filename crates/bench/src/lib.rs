//! Inputs and helpers shared by the benchmarks in `benches/`.

use pipit_kernel::allocator::Heap;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::error::Error;
use pipit_kernel::filter::{self, Comparison, Value};
use pipit_kernel::pipeline::Pipeline;
use pipit_kernel::predicate::{Leaf, Node, Predicate};
use pipit_kernel::query_allocators::QueryAllocators;
use pipit_kernel::row_batch::{BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::DynScannable;
use pipit_kernel::selection::Selection;
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
    parse_expression(&Heap, source).map_or(0, |ast| ast.node_count())
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
            parse_query(&Heap, &REGISTRY, query.as_bytes()).map_or(0, |ast| ast.node_count())
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
        let values = Buffer::allocate(&Heap, size_bytes).expect("allocates");
        Repeat {
            column: ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, None)
                .expect("allocates"),
            batches,
        }
    }
}

impl Source for Repeat {
    type State = u32;

    fn new_state(&self, _: &mut Context) -> Result<u32, Error> {
        Ok(0)
    }

    fn next(&self, _: &mut Context, made: &mut u32, batch: &mut RowBatch) -> Result<bool, Error> {
        if *made == self.batches {
            return Ok(false);
        }
        *made += 1;
        batch.reset(BATCH_ROWS_MAX);
        Ok(batch.push_column(self.column.clone()).is_ok())
    }
}

/// Does nothing, to measure what a transform costs the pipeline.
pub struct PassTransform;

impl Transform for PassTransform {
    type State = ();

    fn new_state(&self, _: &mut Context) -> Result<(), Error> {
        Ok(())
    }

    fn process(&self, _: &mut Context, (): &mut (), _: &mut RowBatch) -> Result<(), Error> {
        Ok(())
    }
}

/// Outputs its input, to measure what an operator costs the pipeline.
pub struct PassOperator;

impl Operator for PassOperator {
    type State = ();

    fn new_state(&self, _: &mut Context) -> Result<(), Error> {
        Ok(())
    }

    fn execute(
        &self,
        _: &mut Context,
        (): &mut (),
        input: &RowBatch,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        output.reset(input.row_count());
        for i in 0..input.column_count() {
            let _ = output.push_column(input.column(i).clone());
        }
        output.selection_mut().clone_from(input.selection());
        Ok(Progress::NeedInput)
    }
}

/// Runs `pipeline` once, returning the rows out, or as many as it made
/// before failing.
pub fn run(pipeline: &Pipeline) -> u64 {
    let query = QueryAllocators::new(&Heap);
    let Ok(mut execution) = pipeline.start(&query) else { return 0 };
    let mut batch = RowBatch::new();
    let mut rows = 0;
    while execution.next(&mut batch) == Ok(true) {
        rows += u64::from(batch.selection().len());
    }
    rows
}

/// `batches` full batches through `steps` pass-through transforms, or
/// operators.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn pass_pipeline(batches: u32, steps: usize, operators: bool) -> Pipeline<'static> {
    let mut owned = pipit_kernel::slow_vec::SlowVec::fixed(&Heap, steps).expect("allocates");
    for _ in 0..steps {
        let step = if operators {
            Step::Operator(DynOperator::new(&Heap, PassOperator).expect("allocates"))
        } else {
            Step::Transform(DynTransform::new(&Heap, PassTransform).expect("allocates"))
        };
        assert!(owned.push(step).is_ok(), "fixed for `steps`");
    }
    Pipeline::new(DynSource::new(&Heap, Repeat::new(batches)).expect("allocates"), owned)
}

/// `row_groups` full row groups, with four columns. Kept for the rest of the
/// run, so pipelines can borrow it.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn table(row_groups: usize) -> &'static DynScannable<'static> {
    let values = Buffer::allocate(&Heap, BATCH_ROWS_MAX as usize * 8).expect("allocates");
    let column = ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, None)
        .expect("allocates");
    let columns = [column.clone(), column.clone(), column.clone(), column];
    let row_groups = vec![columns.as_slice(); row_groups];
    let schema = [
        ("a", DataType::Int64),
        ("b", DataType::Int64),
        ("c", DataType::Int64),
        ("d", DataType::Int64),
    ];
    let table = Table::new(&Heap, &schema, &row_groups).expect("allocates");
    Box::leak(Box::new(DynScannable::new(&Heap, table).expect("allocates")))
}

/// Scans two of `table`'s columns.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn scan_pipeline(table: &'static DynScannable<'static>) -> Pipeline<'static> {
    let columns =
        pipit_kernel::slow_vec::SlowVec::fixed_from(&Heap, [3, 1].into_iter()).expect("allocates");
    let forms = pipit_kernel::slow_vec::SlowVec::fixed_from(&Heap, [Forms::FLAT; 2].into_iter())
        .expect("allocates");
    let source = table.scan(&Heap, columns, forms).expect("allocates");
    Pipeline::new(source, pipit_kernel::slow_vec::SlowVec::fixed(&Heap, 0).expect("allocates"))
}

/// A batch of values in `0..1000`, spread evenly but out of order, with every
/// 16th row null if `nulls`.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn filter_column(nulls: bool) -> ColumnView {
    let rows = BATCH_ROWS_MAX as usize;
    let mut values = Buffer::allocate(&Heap, rows * 8).expect("allocates");
    for (i, value) in (0_i64..).zip(values.as_mut_slice::<i64>()) {
        *value = (i * 7919) % 1000;
    }
    let validity = nulls.then(|| {
        let mut validity = Buffer::allocate(&Heap, rows / 8).expect("allocates");
        validity.as_mut_slice::<u8>().fill(0xff);
        for row in (0..rows).step_by(16) {
            validity.as_mut_slice::<u8>()[row / 8] &= !(1 << (row % 8));
        }
        validity
    });
    ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, validity).expect("allocates")
}

/// Filters `batches` fresh selections of `column` with `filter`, returning
/// the rows kept.
pub fn run_filter(
    column: &ColumnView,
    batches: u32,
    filter: impl Fn(&ColumnView, &mut Selection),
) -> u64 {
    let mut kept = 0;
    for _ in 0..batches {
        let mut selection = Selection::all(column.row_count());
        filter(column, &mut selection);
        kept += u64::from(selection.len());
    }
    kept
}

/// Keeps values over 500: about half.
pub fn greater(column: &ColumnView, selection: &mut Selection) {
    filter::compare(column, Comparison::Greater, Value::Int64(500), selection, None);
}

/// `x > 500 AND x < 900`, `x < 100 OR x > 900`, or `NOT (x > 500)`, over the
/// first column.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn predicate(shape: &str) -> Predicate {
    let compare = |comparison, value| {
        Node::Leaf(Leaf::Compare { column: 0, comparison, value: Value::Int64(value) })
    };
    let nodes: &[Node] = match shape {
        "and" => {
            &[compare(Comparison::Greater, 500), compare(Comparison::Less, 900), Node::And(0, 1)]
        }
        "or" => {
            &[compare(Comparison::Less, 100), compare(Comparison::Greater, 900), Node::Or(0, 1)]
        }
        _ => &[compare(Comparison::Greater, 500), Node::Not(0)],
    };
    Predicate::new(
        pipit_kernel::slow_vec::SlowVec::fixed_from(&Heap, nodes.iter().copied())
            .expect("allocates"),
    )
}

/// Runs `predicate` over `batches` batches of `column`, returning the rows kept.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn run_predicate(predicate: &Predicate, column: &ColumnView, batches: u32) -> u64 {
    let mut context = Context::new(&Heap);
    context.reserve_selections(predicate.depth() as usize).expect("allocates");
    let mut batch = RowBatch::new();
    let mut kept = 0;
    for _ in 0..batches {
        batch.reset(column.row_count());
        let _ = batch.push_column(column.clone());
        predicate.select(context.selections(), &mut batch);
        kept += u64::from(batch.selection().len());
    }
    kept
}
