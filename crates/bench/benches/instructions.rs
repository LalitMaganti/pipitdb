//! Instruction-count benchmarks, run under Valgrind by gungraun. CI fails a PR
//! that makes any of them slower than its base by more than the set limit.
//!
//! Each runs whole queries, small and big, as a user would: what users wait
//! for. Parts measured alone, such as one operator over a few batches, belong
//! in `wall_clock.rs`, to investigate with, not here.

use std::hint::black_box;

use gungraun::prelude::*;
use pipit_file::source::FileSource;
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::lower::PhysicalPlan;
use pipit_kernel::scannable::{Catalog, DynScannable};
use pipit_parquet::table::ParquetTable;
use pipit_parquet_full::Codecs;

#[library_benchmark]
#[bench::typical(pipitdb_bench::queries(100))]
fn query(source: String) -> u32 {
    black_box(pipitdb_bench::count_query_nodes(black_box(&source)))
}

library_benchmark_group!(name = parser_group, benchmarks = [query]);

/// Finds one row of a small table by its id, from the query's text: what a
/// query costs before any rows are read.
const POINT: &str = "FROM t |> WHERE id = 42 |> SELECT a, b";

#[library_benchmark]
#[bench::point(pipitdb_bench::small_table(100), POINT)]
fn small(catalog: &'static dyn Catalog, query: &str) -> u64 {
    black_box(pipitdb_bench::plan_and_run(black_box(catalog), black_box(query)))
}

library_benchmark_group!(name = small_group, benchmarks = [small]);

// Perfetto's `filter_benchmark.cc`, on the same 2^20 rows, planned first.
#[library_benchmark]
#[bench::eq_uint32(pipitdb_bench::perfetto_filter_plan(0))]
#[bench::lt_int64(pipitdb_bench::perfetto_filter_plan(1))]
#[bench::eq_nullable(pipitdb_bench::perfetto_filter_plan(2))]
#[bench::eq_string(pipitdb_bench::perfetto_filter_plan(3))]
fn perfetto_filter(plan: PhysicalPlan<'static>) -> u64 {
    black_box(pipitdb_bench::run_plan(black_box(&plan)))
}

library_benchmark_group!(name = perfetto_filter_group, benchmarks = [perfetto_filter]);

// 64K slices joined to 4K threads by id: stored as a sequence, shuffled, or
// spread far apart.
#[library_benchmark]
#[bench::slice_thread(pipitdb_bench::join_plan(1 << 16, 1 << 12, pipitdb_bench::Joined::Ids))]
#[bench::shuffled(pipitdb_bench::join_plan(1 << 16, 1 << 12, pipitdb_bench::Joined::Shuffled))]
#[bench::spread(pipitdb_bench::join_plan(1 << 16, 1 << 12, pipitdb_bench::Joined::Spread))]
fn join(plan: PhysicalPlan<'static>) -> u64 {
    black_box(pipitdb_bench::run_plan(black_box(&plan)))
}

library_benchmark_group!(name = join_group, benchmarks = [join]);

/// The table `hits`, over ClickBench's first file.
struct Hits(DynScannable<'static>);

impl Catalog for Hits {
    fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
        (name == "hits").then_some(&self.0)
    }
}

/// The query called `name` in `clickbench/queries.tsv`, over ClickBench's
/// first file as `hits`, compiled and lowered. The file is `hits_0.parquet`
/// in `$PIPIT_CLICKBENCH`, by default `target/clickbench`, where
/// `clickbench/fetch.sh` downloads and checks it. `None` without it, unless
/// `$PIPIT_CLICKBENCH` says where it should be, as in CI: anything else
/// failing is a broken benchmark, not one to skip.
#[expect(clippy::expect_used, reason = "a benchmark can't run without its input")]
fn hits_query(name: &str) -> Option<PhysicalPlan<'static>> {
    let set = std::env::var("PIPIT_CLICKBENCH").ok();
    let dir = set.clone().unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/clickbench").to_string()
    });
    let path = format!("{dir}/hits_0.parquet");
    if set.is_none() && !std::path::Path::new(&path).exists() {
        return None;
    }
    let file: &'static FileSource =
        Box::leak(Box::new(FileSource::open(&path).expect("clickbench/fetch.sh fetched it")));
    let sources: &'static [&dyn ByteSource] = Box::leak(Box::new([file as &dyn ByteSource]));
    let env = pipitdb_bench::env();
    let table = ParquetTable::open(env, &Codecs, sources).expect("a Parquet file");
    let hits = Hits(DynScannable::new(env, table).expect("allocates"));
    let query = pipitdb_bench::real::clickbench_query(name).expect("a query pipitdb runs");
    Some(pipitdb_bench::plan(Box::leak(Box::new(hits)), query).expect("compiles"))
}

// ClickBench's filters and aggregations over its first file, a million rows. 0 without the
// file, which CI fetches. The file is sorted by `CounterID`, so `fCounter`'s
// 78,186 rows and `fCounterFew`'s 10 are each one run. (gungraun allows no
// doc comment here.)
#[library_benchmark]
#[bench::f1(hits_query("f1"))]
#[bench::f_adv(hits_query("fAdv"))]
#[bench::f40(hits_query("f40"))]
#[bench::f41(hits_query("f41"))]
#[bench::f_phone(hits_query("fPhone"))]
#[bench::f_ipad(hits_query("fIpad"))]
#[bench::f_search(hits_query("fSearch"))]
#[bench::f_google(hits_query("fGoogle"))]
#[bench::f_counter(hits_query("fCounter"))]
#[bench::f_counter_few(hits_query("fCounterFew"))]
#[bench::g_adv(hits_query("gAdv"))]
#[bench::g_region(hits_query("gRegion"))]
#[bench::g_window(hits_query("gWindow"))]
#[bench::g_user(hits_query("gUser"))]
#[bench::q8(hits_query("8"))]
#[bench::q10(hits_query("10"))]
#[bench::q12(hits_query("12"))]
#[bench::q14(hits_query("14"))]
#[bench::q15(hits_query("15"))]
#[bench::q24t(hits_query("24t"))]
#[bench::q25(hits_query("25"))]
fn clickbench(plan: Option<PhysicalPlan<'static>>) -> u64 {
    plan.map_or(0, |plan| black_box(pipitdb_bench::run_plan(black_box(&plan))))
}

library_benchmark_group!(name = clickbench_group, benchmarks = [clickbench]);
// gungraun clears benchmarks' environment: `PIPIT_CLICKBENCH` says where the
// ClickBench file is, and that it must be there.
main!(
    config = LibraryBenchmarkConfig::default().pass_through_env("PIPIT_CLICKBENCH");
    library_benchmark_groups = parser_group,
    small_group,
    perfetto_filter_group,
    join_group,
    clickbench_group
);
