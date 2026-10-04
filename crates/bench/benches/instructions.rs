//! Instruction-count benchmarks, run under Valgrind by gungraun. CI fails a PR
//! that makes any of them slower than its base by more than the set limit.

use std::hint::black_box;

use gungraun::prelude::*;
use pipit_kernel::step::Step;
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

fn pipeline_input(steps: usize, operators: bool) -> (pipitdb_bench::Repeat, Vec<Step<'static>>) {
    (pipitdb_bench::Repeat::new(1000), pipitdb_bench::pass_steps(steps, operators))
}

#[library_benchmark]
#[bench::source(pipeline_input(0, false))]
#[bench::transforms(pipeline_input(4, false))]
#[bench::operators(pipeline_input(4, true))]
fn pipeline((source, steps): (pipitdb_bench::Repeat, Vec<Step<'static>>)) -> u64 {
    black_box(pipitdb_bench::run_pipeline(black_box(&source), black_box(&steps)))
}

#[library_benchmark]
#[bench::table(pipitdb_bench::table(100, 20_480))]
fn scan(table: pipit_std::table::Table) -> u64 {
    black_box(pipitdb_bench::scan_table(black_box(&table)))
}

library_benchmark_group!(name = pipeline_group, benchmarks = [pipeline, scan]);
main!(library_benchmark_groups = lexer_group, parser_group, pipeline_group);
