//! Wall-clock benchmarks. Run with `cargo bench -p pipitdb-bench --bench wall_clock`.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pipitdb_bench::{Joined, count_nodes, count_tokens, expression, long_text, typical_queries};

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

/// Finds one row of a small table by its id, without and with planning it.
fn small(c: &mut Criterion) {
    let query = "FROM t |> WHERE id = 42 |> SELECT a, b";
    let catalog = pipitdb_bench::small_table(100);
    let mut plan = pipitdb_bench::small_plan(100, query);
    let mut group = c.benchmark_group("small");
    group.bench_function("point", |b| {
        b.iter(|| pipitdb_bench::plan_and_run(black_box(catalog), black_box(query)));
    });
    group.bench_function("point_planned", |b| {
        b.iter(|| black_box(&mut plan).run());
    });
    let mut order = pipitdb_bench::small_plan(2048, "FROM t |> ORDER BY b");
    group.bench_function("order", |b| {
        b.iter(|| black_box(&mut order).run());
    });
    group.finish();
}

/// Scanning a table's row groups.
fn scan(c: &mut Criterion) {
    let mut plan = pipitdb_bench::scan_plan(1000);
    let mut group = c.benchmark_group("scan");
    group.throughput(Throughput::Elements(100 * 20_480));
    group.bench_function("table", |b| b.iter(|| black_box(&mut plan).run()));
    group.finish();
}

/// 1M slices joined to 64K threads by id: ids stored as a sequence, shuffled,
/// or spread far apart, against scanning the slices alone.
fn join(c: &mut Criterion) {
    let mut group = c.benchmark_group("join");
    group.throughput(Throughput::Elements(1 << 20));
    let joins = [
        ("scan", Joined::No),
        ("slice_thread", Joined::Ids),
        ("shuffled", Joined::Shuffled),
        ("spread", Joined::Spread),
    ];
    for (name, joined) in joins {
        let mut plan = pipitdb_bench::join_plan(1 << 20, 1 << 16, joined);
        group.bench_function(name, |b| b.iter(|| black_box(&mut plan).run()));
    }
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
        // 3% of rows, 43%, and 3% through an `OR`.
        ("where", "FROM slice |> WHERE dur > 1000000 |> SELECT ts, dur"),
        ("where_half", "FROM slice |> WHERE depth = 0 |> SELECT ts, dur"),
        (
            "where_or",
            "FROM slice |> WHERE depth = 0 AND (dur > 1000000 OR dur < 1000) |> SELECT ts, dur",
        ),
    ];
    for (name, query) in queries {
        let Some(mut plan) = pipitdb_bench::real::slice_query(slices, query) else { continue };
        group.bench_function(name, |b| b.iter(|| black_box(&mut plan).run()));
    }
    group.finish();
}

/// Perfetto's `filter_benchmark.cc`, on the same 2^20 rows: a scan of `id`
/// and the filtered column, filtered.
fn perfetto_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("perfetto_filter");
    group.throughput(Throughput::Elements(1 << 20));
    for (i, (name, _)) in pipitdb_bench::PERFETTO_FILTERS.iter().enumerate() {
        let mut plan = pipitdb_bench::perfetto_filter_plan(i);
        group.bench_function(*name, |b| b.iter(|| black_box(&mut plan).run()));
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
    targets = lexer, parser, query, small, scan, join, slices, perfetto_filter
}
criterion_main!(benches);
