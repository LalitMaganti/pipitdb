//! Instruction-count benchmarks, run under Valgrind by gungraun. CI fails a PR
//! that makes any of them slower than its base by more than the set limit.

use std::hint::black_box;

use gungraun::prelude::*;
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
main!(library_benchmark_groups = lexer_group, parser_group);
