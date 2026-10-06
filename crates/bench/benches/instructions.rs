//! Instruction-count benchmarks, run under Valgrind by gungraun. CI fails a PR
//! that makes any of them slower than its base by more than the set limit.

use std::hint::black_box;

use gungraun::prelude::*;
use pipit_file::source::FileSource;
use pipit_kernel::allocator::Heap;
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::lower::PhysicalPlan;
use pipit_kernel::pipeline::Pipeline;
use pipit_kernel::scannable::{Catalog, DynScannable};
use pipit_parquet::table::ParquetTable;
use pipit_parquet_full::Codecs;
use pipitdb_bench::{count_nodes, count_tokens, expression, long_text, typical_queries};

#[library_benchmark]
#[bench::typical(typical_queries(100))]
#[bench::long_text(long_text(100))]
fn lexer(source: String) -> u32 {
    black_box(count_tokens(black_box(source.as_bytes())))
}

#[library_benchmark]
#[bench::expression(expression(100))]
fn parser(source: String) -> u32 {
    black_box(count_nodes(black_box(source.as_bytes())))
}

library_benchmark_group!(name = lexer_group, benchmarks = [lexer]);
#[library_benchmark]
#[bench::typical(pipitdb_bench::queries(100))]
fn query(source: String) -> u32 {
    black_box(pipitdb_bench::count_query_nodes(black_box(&source)))
}

library_benchmark_group!(name = parser_group, benchmarks = [parser, query]);

#[library_benchmark]
#[bench::source(pipitdb_bench::pass_pipeline(1000, 0, false))]
#[bench::transforms(pipitdb_bench::pass_pipeline(1000, 4, false))]
#[bench::operators(pipitdb_bench::pass_pipeline(1000, 4, true))]
fn pipeline(pipeline: Pipeline<'static>) -> u64 {
    black_box(pipitdb_bench::run(black_box(&pipeline)))
}

#[library_benchmark]
#[bench::table(pipitdb_bench::scan_pipeline(pipitdb_bench::table(1000)))]
fn scan(pipeline: Pipeline<'static>) -> u64 {
    black_box(pipitdb_bench::run(black_box(&pipeline)))
}

#[library_benchmark]
#[bench::greater(pipitdb_bench::filter_column(false), pipitdb_bench::greater)]
#[bench::greater_nulls(pipitdb_bench::filter_column(true), pipitdb_bench::greater)]
fn filter(
    column: pipit_kernel::column::ColumnView,
    filter: fn(&pipit_kernel::column::ColumnView, &mut pipit_kernel::selection::Selection),
) -> u64 {
    black_box(pipitdb_bench::run_filter(black_box(&column), 100, filter))
}

#[library_benchmark]
#[bench::and(pipitdb_bench::predicate("and"))]
#[bench::or(pipitdb_bench::predicate("or"))]
#[bench::not(pipitdb_bench::predicate("not"))]
fn predicate(predicate: pipit_kernel::predicate::Predicate) -> u64 {
    let column = pipitdb_bench::filter_column(false);
    black_box(pipitdb_bench::run_predicate(black_box(&predicate), &column, 100))
}

library_benchmark_group!(name = pipeline_group, benchmarks = [pipeline, scan, filter, predicate]);

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
    let table = ParquetTable::open(&Heap, &Codecs, sources).expect("a Parquet file");
    let hits = Hits(DynScannable::new(&Heap, table).expect("allocates"));
    let query = pipitdb_bench::real::clickbench_query(name).expect("a query pipitdb runs");
    Some(pipitdb_bench::real::plan(Box::leak(Box::new(hits)), query).expect("compiles"))
}

// ClickBench's filters over its first file, a million rows. 0 without the
// file, which CI fetches. (gungraun allows no doc comment here.)
#[library_benchmark]
#[bench::f1(hits_query("f1"))]
#[bench::f_adv(hits_query("fAdv"))]
#[bench::f40(hits_query("f40"))]
#[bench::f41(hits_query("f41"))]
#[bench::f_phone(hits_query("fPhone"))]
#[bench::f_ipad(hits_query("fIpad"))]
#[bench::f_search(hits_query("fSearch"))]
fn clickbench(plan: Option<PhysicalPlan<'static>>) -> u64 {
    plan.map_or(0, |plan| black_box(pipitdb_bench::run(black_box(plan.pipeline()))))
}

library_benchmark_group!(name = clickbench_group, benchmarks = [clickbench]);
// gungraun clears benchmarks' environment: `PIPIT_CLICKBENCH` says where the
// ClickBench file is, and that it must be there.
main!(
    config = LibraryBenchmarkConfig::default().pass_through_env("PIPIT_CLICKBENCH");
    library_benchmark_groups = lexer_group,
    parser_group,
    pipeline_group,
    clickbench_group
);
