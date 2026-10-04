//! Instruction-count benchmarks, run under Valgrind by gungraun. CI fails a PR
//! that makes any of them slower than its base by more than the set limit.

use std::hint::black_box;

use gungraun::prelude::*;
use pipit_kernel::pipeline::Pipeline;
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
#[bench::table(pipitdb_bench::scan_pipeline(pipitdb_bench::table(100, 20_480)))]
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

library_benchmark_group!(name = pipeline_group, benchmarks = [pipeline, scan, filter]);
main!(library_benchmark_groups = lexer_group, parser_group, pipeline_group);
