//! Wall-clock benchmarks. Run with `cargo bench -p pipitdb-bench --bench wall_clock`.

use std::hint::black_box;
use std::time::Duration;

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
    let mut group = c.benchmark_group("pipeline");
    group.throughput(Throughput::Elements(u64::from(batches)));
    for (name, steps, operators) in
        [("source", 0, false), ("transforms", 4, false), ("operators", 4, true)]
    {
        let pipeline = pipitdb_bench::pass_pipeline(batches, steps, operators);
        group.bench_function(name, |b| b.iter(|| pipitdb_bench::run(black_box(&pipeline))));
    }
    group.finish();
}

/// Scanning a table's row groups.
fn scan(c: &mut Criterion) {
    let pipeline = pipitdb_bench::scan_pipeline(pipitdb_bench::table(100, 20_480));
    let mut group = c.benchmark_group("scan");
    group.throughput(Throughput::Elements(100 * 20_480));
    group.bench_function("table", |b| b.iter(|| pipitdb_bench::run(black_box(&pipeline))));
    group.finish();
}

/// Queries over Perfetto's benchmark slices, compiled and lowered once, then
/// run. Skipped without a Perfetto checkout.
fn slices(c: &mut Criterion) {
    let Some(slices) = pipitdb_bench::real::slices() else {
        eprintln!("No Perfetto checkout at $PIPIT_PERFETTO or ~/perfetto: skipping `slices`.");
        return;
    };
    let mut group = c.benchmark_group("slices");
    let queries = [
        ("all", "FROM slice"),
        ("two", "FROM slice |> SELECT ts, dur"),
        ("four", "FROM slice |> SELECT ts, dur, name, depth"),
    ];
    for (name, query) in queries {
        let Some(plan) = pipitdb_bench::real::slice_query(slices, query) else { continue };
        group.bench_function(name, |b| b.iter(|| pipitdb_bench::run(black_box(plan.pipeline()))));
    }
    group.finish();
}

/// Filtering 1,000 batches of 2,048 rows.
fn filter(c: &mut Criterion) {
    let plain = pipitdb_bench::filter_column(false);
    let nullable = pipitdb_bench::filter_column(true);
    let mut group = c.benchmark_group("filter");
    group.throughput(Throughput::Elements(1000 * 2048));
    let cases: [(&str, &_, fn(&_, &mut _)); 3] = [
        ("greater", &plain, pipitdb_bench::greater),
        ("greater_nulls", &nullable, pipitdb_bench::greater),
        ("in_eight", &plain, pipitdb_bench::in_eight),
    ];
    for (name, column, filter) in cases {
        group.bench_function(name, |b| {
            b.iter(|| pipitdb_bench::run_filter(black_box(column), 1000, filter));
        });
    }
    group.finish();
}

// Each iteration takes well under a millisecond, so short runs still give
// thousands of samples.
criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = lexer, parser, query, pipeline, scan, slices, filter
}
criterion_main!(benches);
