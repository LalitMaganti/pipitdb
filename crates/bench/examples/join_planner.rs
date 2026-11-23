//! Compares join planners. It generates query graphs of many shapes and
//! sizes, runs each planner on them, and reports the cost of its plans
//! against the optimum, or for graphs too big to search, against the best
//! plan any run has found.
//!
//! Usage: `cargo run --release -p pipitdb-bench --example join_planner -- [OPTION]...`
//!
//! - `--planner NAME`: runs this one, as many times as given; all by default.
//! - `--shapes chain,star,...`: of chain, star, snowflake, cycle, clique, random.
//! - `--sizes 5,10,...`: relations per graph; 5, 10, 15, 20, 50, 100, 500 by default.
//! - `--joins inner|mixed|both`: inner joins only, about 20% LEFT, SEMI and
//!   ANTI joins, or both, the default.
//! - `--exhaustive N`: finds the optimum of graphs of up to N relations, 15 by
//!   default.
//! - `--best FILE`: the best known costs, read and updated.
//! - `--graph NAME`: prints a graph, such as `star-10-mixed`, and runs nothing.

use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, Instant};

use pipit_kernel::allocator::Heap;
use pipit_kernel::global::Global;
use pipit_kernel::join_planner::{QueryGraph, RelationSet};
use pipitdb_bench::join_planner::{
    EXHAUSTIVE_MAX, Enumerator, SHAPES, Shape, as_written, cost, exhaustive, generate, name, plan,
};

/// The planners competing, by name. Add one here.
const PLANNERS: &[(&str, Enumerator)] = &[("as_written", as_written), ("plan", plan)];

const SIZES: [usize; 7] = [5, 10, 15, 20, 50, 100, 500];

/// How long to spend timing one planner on one graph.
const TIMING: Duration = Duration::from_millis(200);

fn fail(error: impl Debug) -> String {
    format!("{error:?}")
}

/// What a planner did with one graph.
struct Outcome {
    cost: f64,
    valid: Result<(), String>,
    /// The median of repeated runs.
    time: Duration,
    /// The most bytes it held at once.
    peak: usize,
}

fn run(planner: Enumerator, graph: &QueryGraph) -> Result<Outcome, String> {
    // SAFETY: the tree, all planning keeps, is dropped before it.
    let global = unsafe { Global::new(Heap, usize::MAX) }.map_err(fail)?;
    let tree = planner(&global, graph).map_err(fail)?;
    let costed = cost(graph, &tree)?;
    drop(tree);
    let peak = global.peak();

    let mut times = Vec::new();
    let started = Instant::now();
    while times.len() < 3 || (started.elapsed() < TIMING && times.len() < 101) {
        let start = Instant::now();
        let tree = planner(pipitdb_bench::global(), graph).map_err(fail)?;
        times.push(start.elapsed());
        drop(tree);
        if times.len() == 1 && started.elapsed() > TIMING * 5 {
            break;
        }
    }
    times.sort_unstable();
    let time = times[times.len() / 2];
    Ok(Outcome { cost: costed.cost, valid: costed.valid.map_err(fail), time, peak })
}

fn format_time(time: Duration) -> String {
    let nanos = time.as_nanos();
    match nanos {
        0..1_000 => format!("{nanos}ns"),
        1_000..1_000_000 => format!("{:.1}us", time.as_secs_f64() * 1e6),
        1_000_000..1_000_000_000 => format!("{:.1}ms", time.as_secs_f64() * 1e3),
        _ => format!("{:.2}s", time.as_secs_f64()),
    }
}

fn format_rows(rows: f64) -> String {
    if rows < 100_000.0 { format!("{rows:.0}") } else { format!("{rows:.3e}") }
}

fn format_ratio(ratio: f64) -> String {
    if ratio < 1000.0 { format!("{ratio:.3}") } else { format!("{ratio:.2e}") }
}

fn print_graph(name: &str, graph: &QueryGraph) {
    println!("{name}: {} relations", graph.relations.len());
    for (i, relation) in graph.relations.iter().enumerate() {
        println!("  r{i}: {} of {} rows", relation.rows, relation.unfiltered_rows);
    }
    let relations = |set: &RelationSet| -> Vec<u32> {
        (0..u32::try_from(graph.relations.len()).unwrap_or(u32::MAX))
            .filter(|&r| set.contains(r))
            .collect()
    };
    for edge in graph.edges.iter() {
        let domain = graph.equivalences.get(edge.equivalence as usize).map_or(0, |e| e.domain);
        println!(
            "  {:?} {:?} {:?} on e{} (domain {domain})",
            relations(&edge.left),
            edge.kind,
            relations(&edge.right),
            edge.equivalence
        );
    }
}

/// Options from the command line.
struct Options {
    planners: Vec<(&'static str, Enumerator)>,
    shapes: Vec<Shape>,
    sizes: Vec<usize>,
    mixed: Vec<bool>,
    exhaustive: usize,
    best: Option<String>,
    graph: Option<String>,
}

fn options() -> Result<Options, String> {
    let mut options = Options {
        planners: Vec::new(),
        shapes: SHAPES.to_vec(),
        sizes: SIZES.to_vec(),
        mixed: vec![false, true],
        exhaustive: 15,
        best: None,
        graph: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--planner" => {
                let name = value()?;
                let found = PLANNERS.iter().find(|(n, _)| *n == name);
                options.planners.push(*found.ok_or(format!("no planner {name}"))?);
            }
            "--shapes" => {
                options.shapes = value()?
                    .split(',')
                    .map(|name| {
                        SHAPES.into_iter().find(|s| s.name() == name).ok_or(format!("no {name}"))
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--sizes" => {
                options.sizes =
                    value()?.split(',').map(str::parse).collect::<Result<_, _>>().map_err(fail)?;
            }
            "--joins" => {
                options.mixed = match value()?.as_str() {
                    "inner" => vec![false],
                    "mixed" => vec![true],
                    "both" => vec![false, true],
                    other => return Err(format!("no joins {other}")),
                };
            }
            "--exhaustive" => {
                options.exhaustive = value()?.parse().map_err(fail)?;
                options.exhaustive = options.exhaustive.min(EXHAUSTIVE_MAX);
            }
            "--best" => options.best = Some(value()?),
            "--graph" => options.graph = Some(value()?),
            _ => return Err(format!("unknown option {arg}")),
        }
    }
    if options.planners.is_empty() {
        options.planners = PLANNERS.to_vec();
    }
    if options.sizes.iter().any(|&size| size < 2) {
        return Err("graphs need 2 relations or more".into());
    }
    Ok(options)
}

/// Each planner's ratios to the optimum and to the best known, invalid
/// plans, failures and slowest time.
#[derive(Default)]
struct Summary {
    log_optimum: f64,
    optimum: u32,
    log_best: f64,
    best: u32,
    worst: f64,
    invalid: u32,
    failed: u32,
    slowest: Duration,
}

#[expect(clippy::too_many_lines, reason = "the report, in one place")]
fn main() -> Result<(), String> {
    let options = options()?;
    if let Some(graph) = &options.graph {
        let parts: Vec<&str> = graph.split('-').collect();
        let shape = SHAPES.into_iter().find(|shape| parts.first() == Some(&shape.name()));
        let size = parts.get(1).and_then(|size| size.parse().ok());
        let mixed = parts.get(2).map(|&joins| joins == "mixed");
        let (Some(shape), Some(size @ 2..), Some(mixed)) = (shape, size, mixed) else {
            return Err(format!("no graph {graph}"));
        };
        print_graph(graph, &generate(shape, size, mixed)?);
        return Ok(());
    }
    let mut best: HashMap<String, f64> = HashMap::new();
    if let Some(path) = &options.best
        && let Ok(text) = std::fs::read_to_string(path)
    {
        for line in text.lines() {
            if let Some((name, cost)) = line.split_once('\t')
                && let Ok(cost) = cost.parse()
            {
                best.insert(name.to_string(), cost);
            }
        }
    }

    let mut summaries: Vec<Summary> = options.planners.iter().map(|_| Summary::default()).collect();
    println!(
        "{:<20} {:>5} {:<14} {:>11} {:>11} {:>4} {:>9} {:>9} {:>10}  valid",
        "graph", "edges", "planner", "cost", "reference", "of", "ratio", "time", "peak"
    );
    for &shape in &options.shapes {
        for &size in &options.sizes {
            for &mixed in &options.mixed {
                let name = name(shape, size, mixed);
                let graph = generate(shape, size, mixed)?;
                let mut reference = None;
                if size <= options.exhaustive {
                    let (tree, optimum) = exhaustive(&graph)?;
                    let costed = cost(&graph, &tree)?;
                    if (costed.cost - optimum).abs() > optimum * 1e-9 || costed.valid.is_err() {
                        return Err(format!(
                            "{name}: the exhaustive search's plan costs {optimum}, but costs {costed:?}"
                        ));
                    }
                    reference = Some((optimum, "opt"));
                }
                let outcomes: Vec<Result<Outcome, String>> =
                    options.planners.iter().map(|&(_, p)| run(p, &graph)).collect();
                let found = outcomes
                    .iter()
                    .filter_map(|o| o.as_ref().ok().filter(|o| o.valid.is_ok()).map(|o| o.cost))
                    .chain(best.get(&name).copied())
                    .reduce(f64::min);
                if reference.is_none() {
                    reference = found.map(|cost| (cost, "best"));
                }
                if let Some(cost) = reference.map(|(cost, _)| cost) {
                    let known = best.entry(name.clone()).or_insert(cost);
                    *known = known.min(cost);
                }
                for ((&(planner, _), outcome), summary) in
                    options.planners.iter().zip(&outcomes).zip(&mut summaries)
                {
                    let outcome = match outcome {
                        Ok(outcome) => outcome,
                        Err(error) => {
                            summary.failed += 1;
                            println!(
                                "{name:<20} {:>5} {planner:<14} failed: {error}",
                                graph.edges.len()
                            );
                            continue;
                        }
                    };
                    let (reference_cost, of) = reference.unwrap_or((f64::NAN, "-"));
                    let ratio = outcome.cost / reference_cost;
                    summary.slowest = summary.slowest.max(outcome.time);
                    match &outcome.valid {
                        Ok(()) if of == "opt" => {
                            (summary.log_optimum, summary.optimum) =
                                (summary.log_optimum + ratio.ln(), summary.optimum + 1);
                            summary.worst = summary.worst.max(ratio);
                        }
                        Ok(()) => {
                            (summary.log_best, summary.best) =
                                (summary.log_best + ratio.ln(), summary.best + 1);
                            summary.worst = summary.worst.max(ratio);
                        }
                        Err(_) => summary.invalid += 1,
                    }
                    println!(
                        "{name:<20} {:>5} {planner:<14} {:>11} {:>11} {of:>4} {:>9} {:>9} {:>10}  {}",
                        graph.edges.len(),
                        format_rows(outcome.cost),
                        format_rows(reference_cost),
                        format_ratio(ratio),
                        format_time(outcome.time),
                        outcome.peak,
                        match &outcome.valid {
                            Ok(()) => "yes".to_string(),
                            Err(why) => format!("no: {why}"),
                        },
                    );
                }
            }
        }
    }
    println!();
    println!(
        "{:<14} {:>10} {:>10} {:>10} {:>8} {:>7} {:>9}",
        "planner", "vs opt", "vs best", "worst", "invalid", "failed", "slowest"
    );
    for (&(planner, _), summary) in options.planners.iter().zip(&summaries) {
        let mean = |log: f64, count: u32| (log / f64::from(count.max(1))).exp();
        println!(
            "{planner:<14} {:>10} {:>10} {:>10} {:>8} {:>7} {:>9}",
            format_ratio(mean(summary.log_optimum, summary.optimum)),
            format_ratio(mean(summary.log_best, summary.best)),
            format_ratio(summary.worst),
            summary.invalid,
            summary.failed,
            format_time(summary.slowest)
        );
    }
    if let Some(path) = &options.best {
        let mut lines: Vec<String> =
            best.iter().map(|(name, cost)| format!("{name}\t{cost}")).collect();
        lines.sort();
        std::fs::write(path, lines.join("\n") + "\n").map_err(fail)?;
    }
    Ok(())
}
