//! Wall-clock benchmarks. Run with `cargo bench -p pipitdb-bench --bench wall_clock`.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pipitdb_bench::{count_nodes, count_tokens, expression, long_text, typical_queries};

fn lexer(c: &mut Criterion) {
    let mut group = c.benchmark_group("lexer");
    for (name, source) in [("typical", typical_queries(10_000)), ("long_text", long_text(10_000))] {
        group.throughput(Throughput::Bytes(source.len() as u64));
        group.bench_function(name, |b| b.iter(|| count_tokens(black_box(source.as_bytes()))));
    }
    group.finish();
}

fn parser(c: &mut Criterion) {
    let source = expression(100);
    let mut group = c.benchmark_group("parser");
    group.throughput(Throughput::Bytes(source.len() as u64));
    group.bench_function("expression", |b| b.iter(|| count_nodes(black_box(source.as_bytes()))));
    group.finish();
}

fn query(c: &mut Criterion) {
    let source = pipitdb_bench::queries(1000);
    let mut group = c.benchmark_group("query");
    group.throughput(Throughput::Bytes(source.len() as u64));
    group.bench_function("typical", |b| {
        b.iter(|| pipitdb_bench::count_query_nodes(black_box(&source)));
    });
    group.finish();
}

/// What the pipeline costs per batch, with steps that do nothing.
fn pipeline(c: &mut Criterion) {
    let batches = 10_000;
    let source = pipitdb_bench::Repeat::new(batches);
    let mut group = c.benchmark_group("pipeline");
    group.throughput(Throughput::Elements(u64::from(batches)));
    for (name, steps, operators) in
        [("source", 0, false), ("transforms", 4, false), ("operators", 4, true)]
    {
        let steps = pipitdb_bench::pass_steps(steps, operators);
        group.bench_function(name, |b| {
            b.iter(|| pipitdb_bench::run_pipeline(black_box(&source), black_box(&steps)));
        });
    }
    group.finish();
}

criterion_group!(benches, lexer, parser, query, pipeline);
criterion_main!(benches);
