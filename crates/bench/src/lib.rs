//! Inputs and helpers shared by the benchmarks in `benches/`.

use pipit_kernel::allocator::Heap;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::error::Error;
use pipit_kernel::expression::{Compute, ExprOp, ExprProgram, ExprStep};
use pipit_kernel::filter::{self, Comparison, Value};
use pipit_kernel::flat_hash_map::FlatHashMap;
use pipit_kernel::lower::{PhysicalPlan, lower};
use pipit_kernel::optimize::optimize;
use pipit_kernel::output_properties::plan_nodes;
use pipit_kernel::pipeline::Pipeline;
use pipit_kernel::predicate::{Leaf, Node, Predicate};
use pipit_kernel::query_allocators::QueryAllocators;
use pipit_kernel::row_batch::{BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::{Catalog, DynScannable};
use pipit_kernel::selection::Selection;
use pipit_kernel::slow_vec::SlowVec;
use pipit_kernel::step::{
    DynOperator, DynSource, DynTransform, Operator, Progress, Source, Step, Transform,
};
use pipit_kernel::string_pool::{Id, StringPool};
use pipit_operators::table::Table;
use pipit_pipesql::compile::compile;
use pipit_pipesql::lexer::{Lexer, TokenKind};
use pipit_pipesql::parser::{parse_expression, parse_query};
use pipit_pipesql::registry::Registry;

pub mod real;

static REGISTRY: Registry =
    Registry::new(&[pipit_pipesql::stages::RELATIONAL, pipit_pipesql::stages::AGGREGATION]);

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

/// Batches of `BATCH_ROWS_MAX` rows, all sharing two columns.
pub struct Pair {
    columns: [ColumnView; 2],
    batches: u32,
}

impl Source for Pair {
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
        let [a, b] = &self.columns;
        Ok(batch.push_column(a.clone()).is_ok() && batch.push_column(b.clone()).is_ok())
    }
}

/// Perfetto's `expression_benchmark.cc`: `op` over columns `a` and `b` of
/// `data_type`, with one row in ten null in each if `nulls`, for `batches`
/// batches.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn compute_pipeline(
    op: ExprOp,
    data_type: DataType,
    nulls: bool,
    batches: u32,
) -> Pipeline<'static> {
    let rows = BATCH_ROWS_MAX as usize;
    let column = |value: fn(usize) -> i64, scale: f64, null: usize| {
        let mut values = Buffer::allocate(&Heap, rows * 8).expect("allocates");
        let mut validity = Buffer::allocate(&Heap, rows / 8).expect("allocates");
        validity.as_mut_slice::<u8>().fill(0);
        for row in 0..rows {
            #[expect(clippy::cast_precision_loss, reason = "small integers")]
            let float = value(row) as f64 * scale;
            values.as_mut_slice::<i64>()[row] = match data_type {
                DataType::Float64 => float.to_bits().cast_signed(),
                _ => value(row),
            };
            validity.as_mut_slice::<u8>()[row / 8] |= u8::from(row % 10 != null) << (row % 8);
        }
        let validity = nulls.then_some(validity);
        ColumnView::new(&mut Context::new(&Heap), data_type, values, validity).expect("allocates")
    };
    #[expect(clippy::cast_possible_wrap, reason = "small integers")]
    let columns = [
        column(|row| (row * 7919 % 1000) as i64, 0.5, 0),
        column(|row| (row % 97 + 1) as i64, 0.25, 5),
    ];
    let step = |op, a, b| ExprStep { op, data_type, a, b, int_value: 0, double_value: 0.0 };
    let steps = [step(ExprOp::Column, 0, 0), step(ExprOp::Column, 1, 0), step(op, 0, 1)];
    let steps = SlowVec::fixed_from(&Heap, steps.into_iter()).expect("allocates");
    let programs = SlowVec::fixed_from(&Heap, [ExprProgram { steps }].into_iter());
    let compute = Compute { programs: programs.expect("allocates") };
    let mut owned = SlowVec::fixed(&Heap, 1).expect("allocates");
    let step = Step::Transform(DynTransform::new(&Heap, compute).expect("allocates"));
    assert!(owned.push(step).is_ok(), "fixed for one");
    let source = DynSource::new(&Heap, Pair { columns, batches }).expect("allocates");
    Pipeline::new(source, owned)
}

/// Case `index` of `PERFETTO_EXPRESSIONS`, over `batches` batches.
pub fn perfetto_expression(index: usize, batches: u32) -> Pipeline<'static> {
    let (_, op, data_type, nulls) = PERFETTO_EXPRESSIONS[index];
    compute_pipeline(op, data_type, nulls, batches)
}

/// Perfetto's `expression_benchmark.cc`'s cases: names, ops, types and
/// whether there are nulls.
pub const PERFETTO_EXPRESSIONS: [(&str, ExprOp, DataType, bool); 7] = [
    ("int_add", ExprOp::Add, DataType::Int64, false),
    ("int_mul", ExprOp::Mul, DataType::Int64, false),
    ("int_div", ExprOp::Div, DataType::Int64, false),
    ("int_add_nullable", ExprOp::Add, DataType::Int64, true),
    ("int_div_nullable", ExprOp::Div, DataType::Int64, true),
    ("double_add", ExprOp::Add, DataType::Float64, false),
    ("double_mul", ExprOp::Mul, DataType::Float64, false),
];

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

    fn finish(&self, _: &mut Context, (): &mut (), _: &mut RowBatch) -> Result<Progress, Error> {
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

/// `query` over the tables in `catalog`, compiled, optimized and lowered,
/// ready to run.
pub fn plan(catalog: &'static dyn Catalog, query: &str) -> Option<PhysicalPlan<'static>> {
    let mut plan = compile(&Heap, &REGISTRY, catalog, query.as_bytes()).ok()?;
    optimize(&Heap, &mut plan).ok()?;
    let planned = plan_nodes(&Heap, &plan).ok()?;
    lower(&Heap, &plan, &planned).ok()
}

/// Plans and runs `query` over `catalog`, as an embedder would each query,
/// returning the rows out.
pub fn plan_and_run(catalog: &'static dyn Catalog, query: &str) -> u64 {
    plan(catalog, query).map_or(0, |plan| run(plan.pipeline()))
}

/// The table `t`.
struct Small(DynScannable<'static>);

impl Catalog for Small {
    fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
        (name == "t").then_some(&self.0)
    }
}

/// A table `t` of `rows` rows, at most a batch's, in one row group: `id`,
/// which counts from 0, and `a`, `b` and `c`. Kept for the rest of the run.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn small_table(rows: u32) -> &'static dyn Catalog {
    assert!(rows <= BATCH_ROWS_MAX, "one row group");
    let column = |value: fn(i64) -> i64| {
        let mut values = Buffer::allocate(&Heap, rows as usize * 8).expect("allocates");
        for (i, slot) in (0_i64..).zip(values.as_mut_slice::<i64>()) {
            *slot = value(i);
        }
        ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, None).expect("allocates")
    };
    let columns = [column(|i| i), column(|i| i * 7), column(|i| i % 10), column(|i| -i)];
    let schema = [
        ("id", DataType::Int64),
        ("a", DataType::Int64),
        ("b", DataType::Int64),
        ("c", DataType::Int64),
    ];
    let table = Table::new(&Heap, &schema, &[columns.as_slice()]).expect("allocates");
    let small = Small(DynScannable::new(&Heap, table).expect("allocates"));
    Box::leak(Box::new(small))
}

/// `query` over `small_table(rows)`, planned once.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn small_plan(rows: u32, query: &str) -> PhysicalPlan<'static> {
    plan(small_table(rows), query).expect("compiles")
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
    let columns = [(3, Forms::FLAT), (1, Forms::FLAT)].into_iter();
    let source = table.scan(&Heap, columns, &[], None).expect("allocates");
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

/// A map whose keys are the first `len` even numbers.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn hash_map(len: i64) -> FlatHashMap<i64, ()> {
    let mut map = FlatHashMap::for_len(&Heap, 0).expect("allocates");
    for key in 0..len {
        map.insert(&Heap, key * 2, ()).expect("allocates");
    }
    map
}

/// How many of the numbers below `count` `map` has.
pub fn find_keys(map: &FlatHashMap<i64, ()>, count: i64) -> u64 {
    (0..count).map(|key| u64::from(map.find(key).is_some())).sum()
}

/// `count` strings of 8 digits, one after another, `distinct` of them
/// different, repeating out of order.
pub fn strings(count: usize, distinct: usize) -> Vec<u8> {
    (0..count).flat_map(|i| format!("{:08}", (i * 7919) % distinct).into_bytes()).collect()
}

/// `strings`' strings interned in a new pool, and their ids.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn intern_strings(strings: &[u8]) -> (StringPool, Vec<Id>) {
    let mut pool = StringPool::new(&Heap).expect("allocates");
    let ids = strings
        .chunks(8)
        .map(|string| pool.intern_string(&Heap, Some(string)).expect("allocates"))
        .collect();
    (pool, ids)
}

/// The total length of the strings `ids` are in `pool`.
pub fn get_strings(pool: &StringPool, ids: &[Id]) -> usize {
    ids.iter().map(|&id| pool.get(id).map_or(0, <[u8]>::len)).sum()
}

/// The phone models `string_column` holds.
const MODELS: [&[u8]; 8] =
    [b"", b"iPad", b"iPhone", b"GT-I9300", b"Lumia 920", b"N8", b"iPod", b"Galaxy Nexus"];

/// Which of `MODELS` row `row` holds: spread evenly but out of order.
fn model(row: usize) -> usize {
    (row * 7919) % 8
}

/// A batch of phone models, as `MobilePhoneModel` in `ClickBench`: eight, of 0
/// to 12 bytes, so 1 in 8 is `iPad`.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn string_column() -> ColumnView {
    let rows = BATCH_ROWS_MAX as usize;
    let mut views = Buffer::allocate(&Heap, rows * 8).expect("allocates");
    let mut starts = [0_u32; 8];
    let mut bytes = Vec::new();
    for (start, model) in starts.iter_mut().zip(MODELS) {
        *start = u32::try_from(bytes.len()).expect("small");
        bytes.extend_from_slice(model);
    }
    for (row, view) in views.as_mut_slice::<u32>().chunks_mut(2).enumerate() {
        let model = model(row);
        let len = u32::try_from(MODELS[model].len()).expect("small");
        view.copy_from_slice(&[starts[model], len]);
    }
    let mut stored = Buffer::allocate(&Heap, bytes.len()).expect("allocates");
    stored.as_mut_slice::<u8>().copy_from_slice(&bytes);
    ColumnView::strings(&mut Context::new(&Heap), views, stored, None).expect("allocates")
}

/// `string_column`'s rows as a dictionary of the eight models, as Parquet
/// reads them.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn string_dictionary() -> ColumnView {
    let mut views = Buffer::allocate(&Heap, MODELS.len() * 8).expect("allocates");
    let mut bytes = Vec::new();
    for (view, model) in views.as_mut_slice::<u32>().chunks_mut(2).zip(MODELS) {
        let start = u32::try_from(bytes.len()).expect("small");
        view.copy_from_slice(&[start, u32::try_from(model.len()).expect("small")]);
        bytes.extend_from_slice(model);
    }
    let mut stored = Buffer::allocate(&Heap, bytes.len()).expect("allocates");
    stored.as_mut_slice::<u8>().copy_from_slice(&bytes);
    let mut context = Context::new(&Heap);
    let entries = ColumnView::strings(&mut context, views, stored, None).expect("allocates");
    let rows = BATCH_ROWS_MAX as usize;
    let mut indices = Buffer::allocate(&Heap, rows * 4).expect("allocates");
    for (row, index) in indices.as_mut_slice::<u32>().iter_mut().enumerate() {
        *index = u32::try_from(model(row)).expect("small");
    }
    ColumnView::dictionary(&mut context, &entries, indices).expect("allocates")
}

/// `x = 'iPad'`, over the first column.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn ipad_predicate() -> Predicate {
    let leaf = Leaf::CompareString { column: 0, comparison: Comparison::Equal, start: 0, len: 4 };
    let nodes = SlowVec::fixed_from(&Heap, [Node::Leaf(leaf)].into_iter()).expect("allocates");
    Predicate::with_literals(&Heap, nodes, b"iPad", &[]).expect("allocates")
}

/// Keeps `iPad`s: 1 in 8.
pub fn string_equal(column: &ColumnView, selection: &mut Selection) {
    filter::compare_string(column, Comparison::Equal, b"iPad", selection, None);
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
    let mut entries = predicate.new_entries(&Heap).expect("allocates");
    let mut batch = RowBatch::new();
    let mut kept = 0;
    for _ in 0..batches {
        batch.reset(column.row_count());
        let _ = batch.push_column(column.clone());
        predicate.select(&Heap, context.selections(), &mut entries, &mut batch).expect("allocates");
        kept += u64::from(batch.selection().len());
    }
    kept
}

/// The table `slices`.
struct PerfettoSlices(DynScannable<'static>);

impl Catalog for PerfettoSlices {
    fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
        (name == "slices").then_some(&self.0)
    }
}

/// The table Perfetto's `filter_benchmark.cc` filters, row for row: 2^20 rows
/// of `id`, `dur`, `cpu`, `depth`, a fifth null, and `name`, a dictionary of
/// 64 strings shared by every row group, as Perfetto's string pool is.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn perfetto_slices() -> &'static dyn Catalog {
    let rows = BATCH_ROWS_MAX as usize;
    let mut context = Context::new(&Heap);
    let names: Vec<String> = (0..64).map(|i| format!("name{i}")).collect();
    let mut views = Buffer::allocate(&Heap, names.len() * 8).expect("allocates");
    let mut bytes = Vec::new();
    for (view, name) in views.as_mut_slice::<u32>().chunks_mut(2).zip(&names) {
        let start = u32::try_from(bytes.len()).expect("small");
        view.copy_from_slice(&[start, u32::try_from(name.len()).expect("small")]);
        bytes.extend_from_slice(name.as_bytes());
    }
    let mut stored = Buffer::allocate(&Heap, bytes.len()).expect("allocates");
    stored.as_mut_slice::<u8>().copy_from_slice(&bytes);
    let entries = ColumnView::strings(&mut context, views, stored, None).expect("allocates");
    let hash = |i: usize| u32::try_from(i).expect("small").wrapping_mul(2_654_435_761);
    let mut groups = Vec::new();
    for group in 0..(1 << 20) / rows {
        let first = group * rows;
        let int = |value: &dyn Fn(usize) -> i64, valid: Option<&dyn Fn(usize) -> bool>| {
            let mut values = Buffer::allocate(&Heap, rows * 8).expect("allocates");
            for (i, slot) in values.as_mut_slice::<i64>().iter_mut().enumerate() {
                *slot = value(first + i);
            }
            let validity = valid.map(|valid| {
                let mut bits = Buffer::allocate(&Heap, rows / 8).expect("allocates");
                bits.as_mut_slice::<u8>().fill(0);
                for i in 0..rows {
                    bits.as_mut_slice::<u8>()[i / 8] |= u8::from(valid(first + i)) << (i % 8);
                }
                bits
            });
            ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, validity)
                .expect("allocates")
        };
        let id = int(&|i| i64::try_from(i).expect("small"), None);
        let dur = int(&|i| i64::from((hash(i) >> 12) % 1000), None);
        let cpu = int(&|i| i64::from((hash(i) >> 8) % 8), None);
        let depth = int(&|i| i64::from((hash(i) >> 4) % 16), Some(&|i| hash(i) % 5 != 0));
        let mut indices = Buffer::allocate(&Heap, rows * 4).expect("allocates");
        for (i, index) in indices.as_mut_slice::<u32>().iter_mut().enumerate() {
            *index = (hash(first + i) >> 16) % 64;
        }
        let name = ColumnView::dictionary(&mut context, &entries, indices).expect("allocates");
        groups.push(vec![id, dur, cpu, depth, name]);
    }
    let groups: Vec<&[ColumnView]> = groups.iter().map(Vec::as_slice).collect();
    let schema = [
        ("id", DataType::Int64),
        ("dur", DataType::Int64),
        ("cpu", DataType::Int64),
        ("depth", DataType::Int64),
        ("name", DataType::String),
    ];
    let table = Table::new(&Heap, &schema, &groups).expect("allocates");
    Box::leak(Box::new(PerfettoSlices(DynScannable::new(&Heap, table).expect("allocates"))))
}

/// The filters Perfetto's `filter_benchmark.cc` times, by its names, over
/// `perfetto_slices`. pipitdb has no `Uint32`, so `cpu` is an `Int64`.
pub const PERFETTO_FILTERS: [(&str, &str); 4] = [
    ("eq_uint32", "FROM slices |> WHERE cpu = 3 |> SELECT id"),
    ("lt_int64", "FROM slices |> WHERE dur < 500 |> SELECT id"),
    ("eq_nullable", "FROM slices |> WHERE depth = 3 |> SELECT id"),
    ("eq_string", "FROM slices |> WHERE name = 'name7' |> SELECT id"),
];

/// `PERFETTO_FILTERS`' `i`th query over `perfetto_slices`, planned once.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn perfetto_filter_plan(i: usize) -> PhysicalPlan<'static> {
    plan(perfetto_slices(), PERFETTO_FILTERS[i].1).expect("compiles")
}
