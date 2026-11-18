//! Inputs and helpers shared by the benchmarks in `benches/`.

use pipit_kernel::allocator::Heap;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::error::Error;
use pipit_kernel::expression::{Compute, ExprOp, ExprProgram, ExprStep, PRESENT};
use pipit_kernel::filter::{self, Comparison, Value};
use pipit_kernel::flat_hash_map::FlatHashMap;
use pipit_kernel::join::JoinOp;
use pipit_kernel::lower::{PhysicalPlan, lower};
use pipit_kernel::optimize::optimize;
use pipit_kernel::output_properties::plan_nodes;
use pipit_kernel::pipeline::{Execution, Pipeline};
use pipit_kernel::plan::{DynOp, LogicalPlan, ScanColumn, ScanOp};
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
    let step = |op, data_type, a, b| ExprStep {
        op,
        data_type,
        a,
        b,
        c: 0,
        int_value: 0,
        double_value: 0.0,
    };
    // A comparison is an Int64, whatever it compares.
    let comparison = matches!(
        op,
        ExprOp::Eq | ExprOp::Ne | ExprOp::Lt | ExprOp::Le | ExprOp::IsNull | ExprOp::IsNotNull
    );
    let result = if comparison { DataType::Int64 } else { data_type };
    let (first, second) =
        (step(ExprOp::Column, data_type, 0, 0), step(ExprOp::Column, data_type, 1, 0));
    let steps = if matches!(op, ExprOp::Case | ExprOp::When) {
        // `iif(first, second, first)`, or for `When`, `coalesce(first,
        // second)`: a `CASE` of one branch.
        let (condition, value, otherwise) =
            if op == ExprOp::Case { (0, 1, 0) } else { (PRESENT, 0, 1) };
        let mut case = step(ExprOp::Case, data_type, 2, 1);
        case.c = otherwise;
        let when = step(ExprOp::When, data_type, condition, value);
        SlowVec::fixed_from(&Heap, [first, second, when, case].into_iter())
    } else {
        SlowVec::fixed_from(&Heap, [first, second, step(op, result, 0, 1)].into_iter())
    };
    let steps = steps.expect("allocates");
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
pub const PERFETTO_EXPRESSIONS: [(&str, ExprOp, DataType, bool); 16] = [
    ("int_add", ExprOp::Add, DataType::Int64, false),
    ("int_mul", ExprOp::Mul, DataType::Int64, false),
    ("int_div", ExprOp::Div, DataType::Int64, false),
    ("int_add_nullable", ExprOp::Add, DataType::Int64, true),
    ("int_div_nullable", ExprOp::Div, DataType::Int64, true),
    ("double_add", ExprOp::Add, DataType::Float64, false),
    ("double_mul", ExprOp::Mul, DataType::Float64, false),
    ("int_lt", ExprOp::Lt, DataType::Int64, false),
    ("int_ne", ExprOp::Ne, DataType::Int64, false),
    ("int_lt_nullable", ExprOp::Lt, DataType::Int64, true),
    ("double_lt", ExprOp::Lt, DataType::Float64, false),
    ("is_null", ExprOp::IsNull, DataType::Int64, true),
    ("is_not_null", ExprOp::IsNotNull, DataType::Int64, true),
    // `iif`, as `CASE`, reads the first column as its condition, then chooses
    // the second or the first.
    ("coalesce_int", ExprOp::When, DataType::Int64, true),
    ("iif_int", ExprOp::Case, DataType::Int64, false),
    ("iif_int_nullable", ExprOp::Case, DataType::Int64, true),
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
    let Ok(execution) = pipeline.start(&query, None) else { return 0 };
    count(execution)
}

/// Runs `plan`, returning the rows out.
pub fn run_plan(plan: &PhysicalPlan) -> u64 {
    let query = QueryAllocators::new(&Heap);
    let Ok(execution) = plan.start(&query) else { return 0 };
    count(execution)
}

/// The rows `execution` puts out.
fn count(mut execution: Execution<'_>) -> u64 {
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
    lower(&Heap, &mut plan, &planned).ok()
}

/// Plans and runs `query` over `catalog`, as an embedder would each query,
/// returning the rows out.
pub fn plan_and_run(catalog: &'static dyn Catalog, query: &str) -> u64 {
    plan(catalog, query).map_or(0, |plan| run_plan(&plan))
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

/// `batches` full batches read through a pipeline run as a source, as a
/// join reads its build side.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn nested_pipeline(batches: u32) -> Pipeline<'static> {
    let inner = pass_pipeline(batches, 0, false).into_source(&Heap).expect("allocates");
    Pipeline::new(inner, pipit_kernel::slow_vec::SlowVec::fixed(&Heap, 0).expect("allocates"))
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

/// `x = 'iPad'`, over the first column, and its statement's literals.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn ipad_predicate() -> (Predicate, StringPool) {
    let mut literals = StringPool::new(&Heap).expect("allocates");
    let string = literals.intern_string(&Heap, Some(b"iPad")).expect("allocates");
    let leaf = Leaf::CompareString { column: 0, comparison: Comparison::Equal, string };
    let nodes = SlowVec::fixed_from(&Heap, [Node::Leaf(leaf)].into_iter()).expect("allocates");
    (Predicate::new(nodes), literals)
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

/// Runs `predicate`, with its statement's `literals`, over `batches` batches
/// of `column`, returning the rows kept.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn run_predicate(
    predicate: &Predicate,
    literals: Option<&StringPool>,
    column: &ColumnView,
    batches: u32,
) -> u64 {
    let mut context = Context::new(&Heap);
    context.reserve_selections(predicate.depth() as usize).expect("allocates");
    let mut entries = predicate.new_entries(&Heap).expect("allocates");
    let sets = predicate.new_sets(&Heap, literals).expect("allocates");
    let mut batch = RowBatch::new();
    let mut kept = 0;
    for _ in 0..batches {
        batch.reset(column.row_count());
        let _ = batch.push_column(column.clone());
        let selections = context.selections();
        predicate
            .select(&Heap, literals, &sets, selections, &mut entries, &mut batch)
            .expect("allocates");
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

/// `slices` slices joined to their threads, of which there are `threads`,
/// as Perfetto's `slice JOIN thread USING (utid)`: each slice's `dur` and its
/// thread's `tid`. A thread's `id` is stored as a sequence, and 9 in 10
/// slices' `utid`s, at random, are a thread's. If not `joined`, the slices'
/// `dur` alone, to tell what the join costs. Kept for the rest of the run.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn join_plan(slices: u32, threads: u32, joined: bool) -> PhysicalPlan<'static> {
    let mut context = Context::new(&Heap);
    let int64s = |values: &mut dyn Iterator<Item = i64>, rows: u32| {
        let mut buffer = Buffer::allocate(&Heap, rows as usize * 8).expect("allocates");
        for (slot, value) in buffer.as_mut_slice::<i64>().iter_mut().zip(values) {
            *slot = value;
        }
        ColumnView::new(&mut Context::new(&Heap), DataType::Int64, buffer, None).expect("allocates")
    };
    let groups = |rows: u32| {
        (0..rows.div_ceil(BATCH_ROWS_MAX)).map(move |g| {
            let first = g * BATCH_ROWS_MAX;
            (first, (rows - first).min(BATCH_ROWS_MAX))
        })
    };
    let mut thread_groups = Vec::new();
    for (first, rows) in groups(threads) {
        let id = ColumnView::sequence(&mut context, i64::from(first), rows).expect("allocates");
        let tid = int64s(&mut (i64::from(first)..).map(|id| id + 1000), rows);
        thread_groups.push([id, tid]);
    }
    // Uniform over ten ninths of the threads' ids, so 9 in 10 are one's.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let span = u64::from(threads) * 10 / 9;
    let mut utid = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        i64::try_from((state >> 33) % span).unwrap_or(0)
    };
    let mut slice_groups = Vec::new();
    for (first, rows) in groups(slices) {
        let utids = int64s(&mut core::iter::repeat_with(&mut utid), rows);
        slice_groups.push([utids, int64s(&mut (i64::from(first)..), rows)]);
    }
    let table = |schema: &[(&str, DataType)],
                 groups: &[[ColumnView; 2]]|
     -> &'static DynScannable<'static> {
        let groups: Vec<&[ColumnView]> = groups.iter().map(|group| &group[..]).collect();
        let table = Table::new(&Heap, schema, &groups).expect("allocates");
        Box::leak(Box::new(DynScannable::new(&Heap, table).expect("allocates")))
    };
    let thread = table(&[("id", DataType::Int64), ("tid", DataType::Int64)], &thread_groups);
    let slice = table(&[("utid", DataType::Int64), ("dur", DataType::Int64)], &slice_groups);

    let mut plan = LogicalPlan::new(&Heap).expect("allocates");
    let scan_of = |plan: &mut LogicalPlan<'static>, table, names: [&str; 2]| {
        let mut reads = SlowVec::fixed(&Heap, 2).expect("allocates");
        let mut bound = [None; 2];
        for (i, name) in (0..).zip(names) {
            let binding = plan.add_column(name, DataType::Int64, false).expect("allocates");
            reads.push(ScanColumn { column: i, binding, forms: Forms::FLAT }).expect("fixed");
            bound[i as usize] = Some(binding);
        }
        let op = DynOp::new(&Heap, ScanOp::new(table, reads)).expect("allocates");
        let node = plan.add_node(op, SlowVec::fixed(&Heap, 0).expect("allocates"));
        (node.expect("allocates"), bound.map(|b| b.expect("bound")))
    };
    let (probe, [utid, dur]) = scan_of(&mut plan, slice, ["utid", "dur"]);
    plan.output.push(dur).expect("allocates");
    if joined {
        let (build, [id, tid]) = scan_of(&mut plan, thread, ["id", "tid"]);
        let columns = |ids: [u32; 2]| SlowVec::fixed_from(&Heap, ids.into_iter()).expect("fixed");
        let op =
            JoinOp::new([utid.id, id.id], [columns([utid.id, dur.id]), columns([id.id, tid.id])]);
        let children = SlowVec::fixed_from(&Heap, [probe, build].into_iter()).expect("allocates");
        plan.add_node(DynOp::new(&Heap, op).expect("allocates"), children).expect("allocates");
        plan.output.push(tid).expect("allocates");
    }
    optimize(&Heap, &mut plan).expect("optimizes");
    let planned = plan_nodes(&Heap, &plan).expect("plans");
    lower(&Heap, &mut plan, &planned).expect("lowers")
}
