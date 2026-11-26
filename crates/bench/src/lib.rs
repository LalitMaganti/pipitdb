//! Inputs and helpers shared by the benchmarks in `benches/`.

#![allow(unsafe_code)]

use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType};
use pipit_kernel::containers::BitSlice;
use pipit_kernel::context::Context;
use pipit_kernel::env::{Env, ThreadEnv};
use pipit_kernel::row_batch::BATCH_ROWS_MAX;
use pipit_kernel::scannable::{Catalog, DynScannable};
use pipit_kernel::session::{Prepared, Session};
use pipit_kernel::table::Table;
use pipit_pipesql::compile::compile;
use pipit_pipesql::lexer::{Lexer, TokenKind};
use pipit_pipesql::parser::{parse_expression, parse_query};
use pipit_pipesql::registry::Registry;

/// This thread's `Env`, as the kernel's `Env::current`, which needs
/// its `std` feature, and so `std` in the size example, which has none.
pub fn env() -> &'static Env {
    std::thread_local! {
        // SAFETY: kept in a thread-local.
        static ENV: ThreadEnv = unsafe { ThreadEnv::new() };
    }
    ENV.with(ThreadEnv::get)
}

pub mod real;

static REGISTRY: Registry = Registry::new(&[
    pipit_pipesql::stages::RELATIONAL,
    pipit_pipesql::stages::AGGREGATION,
    pipit_pipesql::stages::JOINS,
]);

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
    parse_expression(crate::env(), source).map_or(0, |ast| ast.node_count())
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
            parse_query(crate::env(), &REGISTRY, query.as_bytes()).map_or(0, |ast| ast.node_count())
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

/// A query prepared in a session of its own, ready to run, as a
/// benchmark's setup makes it.
pub struct Ready {
    session: Session<'static>,
    prepared: Prepared<'static>,
}

impl Ready {
    /// Runs it, returning the rows out.
    pub fn run(&mut self) -> u64 {
        let Ok(mut out) = self.prepared.run(&mut self.session) else { return 0 };
        let mut rows = 0;
        while let Ok(Some(batch)) = out.next() {
            rows += u64::from(batch.selection().len());
        }
        rows
    }
}

/// `query` over the tables in `catalog`, prepared in a session of its own.
pub fn plan(catalog: &'static dyn Catalog, query: &str) -> Option<Ready> {
    let session = Session::new(env());
    let query = compile(env(), &REGISTRY, catalog, query.as_bytes()).ok()?;
    let prepared = session.prepare(query).ok()?;
    Some(Ready { session, prepared })
}

/// Plans and runs `query` over `catalog` in a session of its own, as an
/// embedder would each query, returning the rows out.
pub fn plan_and_run(catalog: &'static dyn Catalog, query: &str) -> u64 {
    let mut session = Session::new(env());
    let Ok(query) = compile(env(), &REGISTRY, catalog, query.as_bytes()) else { return 0 };
    let Ok(prepared) = session.prepare(query) else { return 0 };
    let Ok(mut out) = prepared.run(&mut session) else { return 0 };
    let mut rows = 0;
    while let Ok(Some(batch)) = out.next() {
        rows += u64::from(batch.selection().len());
    }
    rows
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
        let mut values = Buffer::allocate(crate::env(), rows as usize * 8).expect("allocates");
        for (i, slot) in (0_i64..).zip(values.as_mut_slice::<i64>()) {
            *slot = value(i);
        }
        ColumnView::new(&mut Context::new(env(), None), DataType::Int64, values, None)
            .expect("allocates")
    };
    let columns = [column(|i| i), column(|i| i * 7), column(|i| i % 10), column(|i| -i)];
    let schema = [
        ("id", DataType::Int64),
        ("a", DataType::Int64),
        ("b", DataType::Int64),
        ("c", DataType::Int64),
    ];
    let table = Table::new(crate::env(), &schema, &[columns.as_slice()]).expect("allocates");
    let small = Small(DynScannable::new(crate::env(), table).expect("allocates"));
    Box::leak(Box::new(small))
}

/// `query` over `small_table(rows)`, planned once.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn small_plan(rows: u32, query: &str) -> Ready {
    plan(small_table(rows), query).expect("compiles")
}

/// `row_groups` full row groups, with four columns. Kept for the rest of the
/// run, so pipelines can borrow it.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn table(row_groups: usize) -> &'static DynScannable<'static> {
    let values = Buffer::allocate(crate::env(), BATCH_ROWS_MAX as usize * 8).expect("allocates");
    let column = ColumnView::new(&mut Context::new(env(), None), DataType::Int64, values, None)
        .expect("allocates");
    let columns = [column.clone(), column.clone(), column.clone(), column];
    let row_groups = vec![columns.as_slice(); row_groups];
    let schema = [
        ("a", DataType::Int64),
        ("b", DataType::Int64),
        ("c", DataType::Int64),
        ("d", DataType::Int64),
    ];
    let table = Table::new(crate::env(), &schema, &row_groups).expect("allocates");
    Box::leak(Box::new(DynScannable::new(crate::env(), table).expect("allocates")))
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
    let mut context = Context::new(env(), None);
    let names: Vec<String> = (0..64).map(|i| format!("name{i}")).collect();
    let mut views = Buffer::allocate(crate::env(), names.len() * 8).expect("allocates");
    let mut bytes = Vec::new();
    for (view, name) in views.as_mut_slice::<u32>().chunks_mut(2).zip(&names) {
        let start = u32::try_from(bytes.len()).expect("small");
        view.copy_from_slice(&[start, u32::try_from(name.len()).expect("small")]);
        bytes.extend_from_slice(name.as_bytes());
    }
    let mut stored = Buffer::allocate(crate::env(), bytes.len()).expect("allocates");
    stored.as_mut_slice::<u8>().copy_from_slice(&bytes);
    let entries = ColumnView::strings(&mut context, views, stored, None).expect("allocates");
    let hash = |i: usize| u32::try_from(i).expect("small").wrapping_mul(2_654_435_761);
    let mut groups = Vec::new();
    for group in 0..(1 << 20) / rows {
        let first = group * rows;
        let int = |value: &dyn Fn(usize) -> i64, valid: Option<&dyn Fn(usize) -> bool>| {
            let mut values = Buffer::allocate(crate::env(), rows * 8).expect("allocates");
            for (i, slot) in values.as_mut_slice::<i64>().iter_mut().enumerate() {
                *slot = value(first + i);
            }
            let validity = valid.map(|valid| {
                let mut bits = Buffer::allocate(crate::env(), BitSlice::<u8>::words_for(rows))
                    .expect("allocates");
                bits.as_mut_slice::<u8>().fill(0);
                for i in 0..rows {
                    BitSlice::of_mut(bits.as_mut_slice::<u8>()).set_if(i, valid(first + i));
                }
                bits
            });
            ColumnView::new(&mut Context::new(env(), None), DataType::Int64, values, validity)
                .expect("allocates")
        };
        let id = int(&|i| i64::try_from(i).expect("small"), None);
        let dur = int(&|i| i64::from((hash(i) >> 12) % 1000), None);
        let cpu = int(&|i| i64::from((hash(i) >> 8) % 8), None);
        let depth = int(&|i| i64::from((hash(i) >> 4) % 16), Some(&|i| hash(i) % 5 != 0));
        let mut indices = Buffer::allocate(crate::env(), rows * 4).expect("allocates");
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
    let table = Table::new(crate::env(), &schema, &groups).expect("allocates");
    Box::leak(Box::new(PerfettoSlices(DynScannable::new(crate::env(), table).expect("allocates"))))
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
pub fn perfetto_filter_plan(i: usize) -> Ready {
    plan(perfetto_slices(), PERFETTO_FILTERS[i].1).expect("compiles")
}

/// A flat column of `rows` of `values`.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
fn int64_column(values: &mut dyn Iterator<Item = i64>, rows: u32) -> ColumnView {
    let mut buffer = Buffer::allocate(crate::env(), rows as usize * 8).expect("allocates");
    for (slot, value) in buffer.as_mut_slice::<i64>().iter_mut().zip(values) {
        *slot = value;
    }
    ColumnView::new(&mut Context::new(env(), None), DataType::Int64, buffer, None)
        .expect("allocates")
}

/// Whether `join_plan` joins the slices to their threads, and how the
/// threads' ids are, which decides how `JoinOp` finds a slice's thread.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Joined {
    No,
    /// Stored as a sequence from 0: a thread is found by its id's number.
    Ids,
    /// 0 up, in another order: found in an array, by its id's number.
    Shuffled,
    /// Spread over 2^40: found by hashing.
    Spread,
}

impl Joined {
    /// The ids of the `rows` threads from `first`, of `threads`.
    #[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
    fn ids(self, context: &mut Context, first: u32, rows: u32, threads: u32) -> ColumnView {
        if matches!(self, Joined::No | Joined::Ids) {
            return ColumnView::sequence(context, i64::from(first), rows).expect("allocates");
        }
        int64_column(&mut (i64::from(first)..).map(|i| self.id(i, i64::from(threads))), rows)
    }

    /// Thread `i`'s id, where its ids are `threads`, a power of two.
    fn id(self, i: i64, threads: i64) -> i64 {
        match self {
            Joined::No | Joined::Ids => i,
            // Odd, so a bijection of a power of two's numbers.
            Joined::Shuffled => i.wrapping_mul(40_503) & (threads - 1),
            Joined::Spread => i.wrapping_mul(2_654_435_761) & ((1 << 40) - 1),
        }
    }
}

/// `slices` slices joined to their threads, of which there are `threads`,
/// as Perfetto's `slice JOIN thread USING (utid)`: each slice's `dur` and its
/// thread's `tid`. A thread's `id` is stored as a sequence, and 9 in 10
/// slices' `utid`s, at random, are a thread's. Without a join, the slices'
/// `dur` alone, to tell what the join costs. Kept for the rest of the run.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn join_plan(slices: u32, threads: u32, joined: Joined) -> Ready {
    let mut context = Context::new(env(), None);
    let int64s = int64_column;
    let groups = |rows: u32| {
        (0..rows.div_ceil(BATCH_ROWS_MAX)).map(move |g| {
            let first = g * BATCH_ROWS_MAX;
            (first, (rows - first).min(BATCH_ROWS_MAX))
        })
    };
    let thread_groups: Vec<[ColumnView; 2]> = groups(threads)
        .map(|(first, rows)| {
            [
                joined.ids(&mut context, first, rows, threads),
                int64s(&mut (i64::from(first)..).map(|id| id + 1000), rows),
            ]
        })
        .collect();
    // Uniform over ten ninths of the threads' ids, so 9 in 10 are one's.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let span = u64::from(threads) * 10 / 9;
    let mut utid = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let utid = i64::try_from((state >> 33) % span).unwrap_or(0);
        // A spread id is the thread's, unless it's past the last.
        if joined == Joined::Spread && utid < i64::from(threads) {
            joined.id(utid, 0)
        } else {
            utid
        }
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
        let table = Table::new(crate::env(), schema, &groups).expect("allocates");
        Box::leak(Box::new(DynScannable::new(crate::env(), table).expect("allocates")))
    };
    let thread = table(&[("id", DataType::Int64), ("tid", DataType::Int64)], &thread_groups);
    let slice = table(&[("utid", DataType::Int64), ("dur", DataType::Int64)], &slice_groups);

    let tables: &'static SliceThread = Box::leak(Box::new(SliceThread { slice, thread }));
    let query = if joined == Joined::No {
        "FROM slice |> SELECT dur"
    } else {
        "FROM slice |> JOIN thread ON utid = id |> SELECT dur, tid"
    };
    plan(tables, query).expect("compiles")
}

/// The tables `join_plan` joins.
struct SliceThread {
    slice: &'static DynScannable<'static>,
    thread: &'static DynScannable<'static>,
}

impl Catalog for SliceThread {
    fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
        match name {
            "slice" => Some(self.slice),
            "thread" => Some(self.thread),
            _ => None,
        }
    }
}

/// Two columns of `table(row_groups)`, scanned: `FROM t |> SELECT d, b`.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
pub fn scan_plan(row_groups: usize) -> Ready {
    let catalog: &'static One = Box::leak(Box::new(One(table(row_groups))));
    plan(catalog, "FROM t |> SELECT d, b").expect("compiles")
}

/// One table, `t`.
struct One(&'static DynScannable<'static>);

impl Catalog for One {
    fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
        (name == "t").then_some(self.0)
    }
}

#[cfg(test)]
mod tests {
    use pipit_kernel::bytes::ByteSource;
    use pipit_parquet::Uncompressed;
    use pipit_parquet::table::ParquetTable;

    use super::*;

    /// A Parquet file of 5,000 rows of `id`, `quarter` and `name`.
    static NULLS: &[u8] = include_bytes!("../../parquet/tests/data/nulls.parquet");

    /// Table `t`, a Parquet file, whose columns can be read lazily.
    struct Lazy(DynScannable<'static>);

    impl Catalog for Lazy {
        fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
            (name == "t").then_some(&self.0)
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "leaks its table, as `plan_and_run` takes a `'static` catalog")]
    fn sorts_by_columns_it_then_drops() {
        let sources: &'static [&dyn ByteSource] = Box::leak(Box::new([&NULLS as &dyn ByteSource]));
        let table = ParquetTable::open(env(), &Uncompressed, sources).unwrap();
        let catalog = Box::leak(Box::new(Lazy(DynScannable::new(env(), table).unwrap())));
        // A sort's columns are loaded before it, even ones only above it
        // reads: a top-k can't carry lazy columns through.
        let query = "FROM t |> WHERE id > 10 |> ORDER BY quarter, id |> SELECT name |> LIMIT 3";
        assert_eq!(plan_and_run(catalog, query), 3);
    }
}
