//! Queries that each make one operator hold more than a memory budget, run
//! first without a budget and then at fractions of the peak that run took,
//! so spilling can be built an operator at a time against a case that fails.
//!
//! Usage: `cargo run --profile speed -p pipitdb-bench --example spill_cases -- [CASE...]`
//!
//! `ROWS` sets each table's rows, 2^20 by default, and `DIVISORS` the budgets,
//! as the peak divided by each, `2,4,8,16` by default. Prints a line per run:
//! the case, its budget, whether it gave the unbudgeted run's rows, its time
//! and its peak. A budget below what a scan alone needs is skipped: no query
//! can run in less. Each case also runs at `SMALLEST` scans' worth, the
//! budget every case should one day finish in.

#![allow(unsafe_code)]

use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::{Duration, Instant};

use pipit_kernel::allocator::Heap;
use pipit_kernel::column::{ColumnView, DataType, Form, Forms};
use pipit_kernel::containers::BitSlice;
use pipit_kernel::context::Context;
use pipit_kernel::env::{Env, Memory};
use pipit_kernel::error::Error;
use pipit_kernel::row_batch::{BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::{Catalog, DynScannable, Key, Scannable, Summary as TableSummary};
use pipit_kernel::selection::Kept;
use pipit_kernel::session::Session;
use pipit_kernel::table::Table;
use pipit_pipesql::compile::compile;
use pipit_pipesql::registry::Registry;

static REGISTRY: Registry = Registry::new(&[
    pipit_pipesql::stages::RELATIONAL,
    pipit_pipesql::stages::AGGREGATION,
    pipit_pipesql::stages::JOINS,
]);

/// A query, what it fills, and whether its rows come in an order it sets.
struct Case {
    name: &'static str,
    query: &'static str,
    ordered: bool,
}

/// A query that holds no rows, whose peak is the least any query needs.
const SCAN: Case = Case { name: "scan", query: "FROM t |> SELECT k, s, p", ordered: false };

/// How many scans' worth of memory every case should finish in.
const SMALLEST: usize = 4;

/// Each operator that holds rows, one at a time, over `t` and `u`. In `t`,
/// `k` is unique, `g` has two rows a value, `v` is unique but a seventh
/// null, `s` is a string a `g`, and `p` is 64 bytes a row. `u` has `uk`,
/// unique, `us`, as `s` of `uk / 2`, and `uw`, an eleventh null.
const CASES: &[Case] = &[
    Case {
        name: "aggregate_int_keys",
        query: "FROM t |> AGGREGATE sum(v) AS total, count(*) AS n GROUP BY g",
        ordered: false,
    },
    Case {
        name: "aggregate_string_keys",
        query: "FROM t |> AGGREGATE sum(v) AS total GROUP BY s",
        ordered: false,
    },
    Case {
        name: "aggregate_string_min",
        query: "FROM t |> AGGREGATE min(s) AS least GROUP BY g",
        ordered: false,
    },
    Case {
        name: "aggregate_distinct",
        query: "FROM t |> AGGREGATE count(DISTINCT k) AS d",
        ordered: false,
    },
    Case {
        name: "join_int_keys",
        query: "FROM t |> JOIN u ON k = uk |> SELECT k, uw",
        ordered: false,
    },
    Case {
        name: "join_string_keys",
        query: "FROM t |> JOIN u ON s = us |> SELECT k, uw",
        ordered: false,
    },
    Case {
        name: "left_join",
        query: "FROM t |> LEFT JOIN u ON g = uk |> SELECT k, uw",
        ordered: false,
    },
    Case { name: "sort_int", query: "FROM t |> SELECT k, v |> ORDER BY k", ordered: true },
    Case { name: "sort_string", query: "FROM t |> SELECT s, k |> ORDER BY s, k", ordered: true },
    Case { name: "sort_wide", query: "FROM t |> SELECT k, p |> ORDER BY k", ordered: true },
    Case { name: "sort_nulls", query: "FROM t |> SELECT v, k |> ORDER BY v DESC", ordered: true },
    Case {
        name: "top_k",
        query: "FROM t |> SELECT s, k |> ORDER BY s DESC, k |> LIMIT 10",
        ordered: true,
    },
];

fn main() -> Result<(), String> {
    let rows: u32 = number("ROWS", 1 << 20)?;
    let divisors: Vec<usize> = std::env::var("DIVISORS")
        .unwrap_or_else(|_| "2,4,8,16".into())
        .split(',')
        .map(|d| d.parse().map_err(|_| format!("DIVISORS: {d}")))
        .collect::<Result<_, _>>()?;
    let only: Vec<String> = std::env::args().skip(1).collect();
    let tables = tables(rows).map_err(|e| format!("{e:?}"))?;
    println!("case\tbudget\tresult\tms\tpeak");
    let (Ok(_), time, least) = run_with(&tables, &SCAN, usize::MAX) else {
        return Err("the scan failed".into());
    };
    report(SCAN.name, "none", "ok", time, least);
    let mut failed = 0;
    for case in CASES.iter().filter(|case| only.is_empty() || only.iter().any(|n| n == case.name)) {
        let (expected, unbudgeted) = match run_with(&tables, case, usize::MAX) {
            (Ok(rows), time, peak) => {
                report(case.name, "none", "ok", time, peak);
                (rows, peak)
            }
            (Err(error), time, peak) => {
                report(case.name, "none", &error, time, peak);
                continue;
            }
        };
        let budgets = divisors.iter().map(|&d| (format!("1/{d}"), unbudgeted / d));
        let smallest = (format!("{SMALLEST}x scan"), SMALLEST * least);
        for (label, budget) in budgets.chain([smallest]) {
            if budget < least {
                continue;
            }
            let (got, time, peak) = run_with(&tables, case, budget);
            let result = match got {
                Ok(got) if got == expected => "ok".into(),
                Ok(_) => "WRONG".into(),
                Err(error) => error,
            };
            failed += usize::from(result != "ok");
            report(case.name, &label, &result, time, peak);
        }
    }
    eprintln!("{failed} runs failed");
    Ok(())
}

/// `name` from the environment, or `default`.
fn number<T: std::str::FromStr>(name: &str, default: T) -> Result<T, String> {
    std::env::var(name).map_or(Ok(default), |v| v.parse().map_err(|_| format!("{name}: {v}")))
}

#[expect(clippy::cast_precision_loss, reason = "a report")]
fn report(case: &str, budget: &str, result: &str, time: Duration, peak: usize) {
    let ms = time.as_secs_f64() * 1e3;
    println!("{case}\t{budget}\t{result}\t{ms:.0}\t{:.1} MB", peak as f64 / 1e6);
}

/// What `case` gives in an `Env` of `budget` bytes, how long it took, and
/// the most the `Env` held.
fn run_with(
    tables: &Tables,
    case: &Case,
    budget: usize,
) -> (Result<Summary, String>, Duration, usize) {
    let Ok(env) = Env::new(Memory::new(Heap, budget)) else {
        return (Err("OutOfMemory".into()), Duration::ZERO, 0);
    };
    let start = Instant::now();
    let got = run(tables, case, &env);
    (got, start.elapsed(), env.peak())
}

/// How many rows a result has and a hash of them, of their order too if
/// `ordered`.
#[derive(PartialEq, Eq, Clone, Copy)]
struct Summary {
    rows: u64,
    hash: u64,
}

fn run(tables: &Tables, case: &Case, env: &Env) -> Result<Summary, String> {
    let mut session = Session::new(env);
    let query = compile(env, &REGISTRY, tables, case.query.as_bytes())
        .map_err(|e| format!("compile: {e:?}"))?;
    let prepared = session.prepare(query).map_err(|e| format!("{e:?}"))?;
    let mut out = prepared.run(&mut session).map_err(|e| format!("{e:?}"))?;
    let mut summary = Summary { rows: 0, hash: 0 };
    while let Some(batch) = out.next().map_err(|e| format!("{e:?}"))? {
        let kept: Vec<u32> = match batch.selection().kept() {
            Kept::Prefix => (0..batch.selection().len()).collect(),
            Kept::None => Vec::new(),
            Kept::Select(rows) => rows.iter().map(|&row| u32::from(row)).collect(),
        };
        for row in kept {
            let mut hasher = DefaultHasher::new();
            for column in prepared.columns() {
                hash_value(batch.column(column.position), row, &mut hasher);
            }
            let hash = hasher.finish();
            summary.hash = if case.ordered {
                summary.hash.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(hash)
            } else {
                summary.hash.wrapping_add(hash)
            };
            summary.rows += 1;
        }
    }
    Ok(summary)
}

/// Hashes row `row` of `column`, reading its value through its form.
fn hash_value(column: &ColumnView, row: u32, hasher: &mut DefaultHasher) {
    if column.is_null(row) {
        0_u8.hash(hasher);
        return;
    }
    let value = match column.form() {
        Form::Sequence(first) => return (first + i64::from(row)).hash(hasher),
        Form::Flat => row as usize,
        Form::Constant => 0,
        Form::Dictionary(indices) => indices[row as usize] as usize,
    };
    let values = column.values();
    match column.data_type() {
        DataType::Int64 => values.int64s()[value].hash(hasher),
        DataType::Float64 => values.float64s()[value].to_bits().hash(hasher),
        DataType::String => values.strings().get(value).hash(hasher),
    }
}

/// The tables, by name.
struct Tables(Vec<(&'static str, DynScannable<'static>)>);

impl Catalog for Tables {
    fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
        self.0.iter().find(|(table, _)| *table == name).map(|(_, table)| table)
    }
}

/// `t` and `u`, of `rows` rows each, as `CASES` says, in the thread's `Env`
/// rather than a budgeted one.
fn tables(rows: u32) -> Result<Tables, Error> {
    let env = pipitdb_bench::env();
    let count = u64::from(rows);
    // Spreads 0..count over itself, so keys arrive out of order: a multiplier
    // with no factor in common with `count` maps each to a different one.
    let mut multiplier = 0x9e37_79b9_u64;
    while gcd(multiplier, count) != 1 {
        multiplier += 1;
    }
    let spread = |i: u64, seed: u64| (i * multiplier + seed) % count;
    let name = |group: i64| format!("s{group:010}").into_bytes();
    let payload = |key: i64| format!("{key:064}").into_bytes();
    let mut t_groups = Vec::new();
    let mut u_groups = Vec::new();
    for start in (0..count).step_by(BATCH_ROWS_MAX as usize) {
        let range = start..(start + u64::from(BATCH_ROWS_MAX)).min(count);
        let keys: Vec<i64> = range.clone().map(|i| spread(i, 0).cast_signed()).collect();
        let groups: Vec<i64> = keys.iter().map(|key| key / 2).collect();
        let values: Vec<Option<i64>> =
            range.clone().map(|i| (i % 7 != 3).then_some(i.cast_signed())).collect();
        let names: Vec<Vec<u8>> = groups.iter().map(|&group| name(group)).collect();
        let payloads: Vec<Vec<u8>> = keys.iter().map(|&key| payload(key)).collect();
        t_groups.push([
            ints(env, keys.iter().copied().map(Some))?,
            ints(env, groups.iter().copied().map(Some))?,
            ints(env, values.into_iter())?,
            strings(env, &names)?,
            strings(env, &payloads)?,
        ]);
        let keys: Vec<i64> = range.map(|i| spread(i, 7).cast_signed()).collect();
        let names: Vec<Vec<u8>> = keys.iter().map(|&key| name(key / 2)).collect();
        let weights = keys.iter().map(|key| (key % 11 != 0).then_some(key * 3));
        let keys = ints(env, keys.iter().copied().map(Some))?;
        u_groups.push([keys, strings(env, &names)?, ints(env, weights)?]);
    }
    let t_schema = [
        ("k", DataType::Int64),
        ("g", DataType::Int64),
        ("v", DataType::Int64),
        ("s", DataType::String),
        ("p", DataType::String),
    ];
    let u_schema = [("uk", DataType::Int64), ("us", DataType::String), ("uw", DataType::Int64)];
    let t_groups: Vec<&[ColumnView]> = t_groups.iter().map(|group| &group[..]).collect();
    let u_groups: Vec<&[ColumnView]> = u_groups.iter().map(|group| &group[..]).collect();
    let t = DynScannable::new(env, Decoded(Table::new(env, &t_schema, &t_groups)?))?;
    let u = DynScannable::new(env, Decoded(Table::new(env, &u_schema, &u_groups)?))?;
    Ok(Tables(vec![("t", t), ("u", u)]))
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn ints(
    env: &Env,
    values: impl ExactSizeIterator<Item = Option<i64>>,
) -> Result<ColumnView, Error> {
    Ok(ColumnView::from_int64s(&mut Context::new(env, None), values)?)
}

fn strings(env: &Env, values: &[Vec<u8>]) -> Result<ColumnView, Error> {
    let context = &mut Context::new(env, None);
    Ok(ColumnView::from_strings(context, values.len(), |i| Some(values[i].as_slice()))?)
}

/// A table whose scans copy each batch into the run's memory, as a scan of
/// a file decodes it there, so what operators hold counts against the
/// budget rather than viewing the table's buffers for free.
struct Decoded(Table);

impl Scannable for Decoded {
    type State<'s> = <Table as Scannable>::State<'s>;
    type Loader = ();

    fn column_count(&self) -> u32 {
        self.0.column_count()
    }

    fn column_name(&self, column: u32) -> &str {
        self.0.column_name(column)
    }

    fn column_type(&self, column: u32) -> DataType {
        self.0.column_type(column)
    }

    fn column_nullable(&self, column: u32) -> bool {
        self.0.column_nullable(column)
    }

    fn column_key(&self, column: u32) -> Key {
        self.0.column_key(column)
    }

    fn forms(&self, column: u32) -> Forms {
        self.0.forms(column)
    }

    fn summary(&self, column: Option<u32>) -> Option<TableSummary> {
        self.0.summary(column)
    }

    fn open<'s>(
        &'s self,
        context: &mut Context,
        columns: &'s [(u32, Forms)],
    ) -> Result<Self::State<'s>, Error> {
        self.0.open(context, columns)
    }

    fn next<'s>(
        &'s self,
        context: &mut Context,
        state: &mut Self::State<'s>,
        batch: &mut RowBatch<'s>,
    ) -> Result<bool, Error> {
        if !self.0.next(context, state, batch)? {
            return Ok(false);
        }
        for c in 0..batch.column_count() {
            let copy = copy(context, batch.column(c))?;
            batch.set_column(c, copy);
        }
        Ok(true)
    }
}

/// `column`, flat, in buffers from `context`.
fn copy(context: &mut Context, column: &ColumnView) -> Result<ColumnView, Error> {
    let rows = column.row_count() as usize;
    let values = column.values();
    if column.data_type() == DataType::String {
        let strings = values.strings();
        #[expect(clippy::cast_possible_truncation, reason = "a row of a batch")]
        let string = |row: usize| (!column.is_null(row as u32)).then(|| strings.get(row));
        return Ok(ColumnView::from_strings(context, rows, string)?);
    }
    let mut validity = None;
    if column.validity().is_some() {
        let mut bits = context.small_buffer(BitSlice::<u8>::words_for(rows))?;
        let bitmap = BitSlice::of_mut(bits.as_mut_slice::<u8>());
        for row in 0..rows {
            #[expect(clippy::cast_possible_truncation, reason = "a row of a batch")]
            let valid = !column.is_null(row as u32);
            if valid { bitmap.set(row) } else { bitmap.clear(row) }
        }
        validity = Some(bits);
    }
    let mut words = context.values_buffer(rows * 8)?;
    words.as_mut_slice::<i64>().copy_from_slice(&values.int64s()[..rows]);
    Ok(ColumnView::new(context, column.data_type(), words, validity)?)
}
