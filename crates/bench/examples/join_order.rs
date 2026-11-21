//! Compares join order enumerators. It generates query graphs of many shapes
//! and sizes, runs each enumerator on them, and reports the cost of its plans
//! against the optimum, or for graphs too big to search, against the best
//! plan any run has found.
//!
//! Usage: `cargo run --profile profiling -p pipitdb-bench --example join_order -- [OPTION]...`
//!
//! - `--enumerator NAME`: runs this one, as many times as given; all by default.
//! - `--shapes chain,star,...`: of chain, star, snowflake, cycle, clique, random.
//! - `--sizes 5,10,...`: relations per graph; 5, 10, 15, 20, 50, 100, 500 by default.
//! - `--joins inner|mixed|both`: inner joins only, about 20% LEFT, SEMI and
//!   ANTI joins, or both, the default.
//! - `--exhaustive N`: finds the optimum of graphs of up to N relations, 15 by
//!   default.
//! - `--best FILE`: the best known costs, read and updated.
//! - `--graph NAME`: prints a graph, such as `star-10-mixed`, and runs nothing.
//! - `--list`: names the enumerators.

use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, Instant};

use pipit_kernel::allocator::{Heap, LimitAllocator};
use pipit_kernel::join_order::{
    Enumerator, EquivalenceSet, JoinEdge, JoinKind, JoinTree, QueryGraph, ROWS_MAX, Relation,
    RelationSet, as_written, cost, enumerate,
};

/// The enumerators competing, by name. Add one here.
const ENUMERATORS: &[(&str, Enumerator)] = &[("as_written", as_written), ("enumerate", enumerate)];

const SHAPES: [Shape; 6] =
    [Shape::Chain, Shape::Star, Shape::Snowflake, Shape::Cycle, Shape::Clique, Shape::Random];

const SIZES: [usize; 7] = [5, 10, 15, 20, 50, 100, 500];

/// The most relations the exhaustive search takes, as its time is 3^n.
const EXHAUSTIVE_MAX: usize = 20;

/// How long to spend timing one enumerator on one graph.
const TIMING: Duration = Duration::from_millis(200);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    Chain,
    Star,
    Snowflake,
    Cycle,
    Clique,
    Random,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Shape::Chain => "chain",
            Shape::Star => "star",
            Shape::Snowflake => "snowflake",
            Shape::Cycle => "cycle",
            Shape::Clique => "clique",
            Shape::Random => "random",
        }
    }
}

/// `SplitMix64`: small, and the same everywhere, so graphs are too.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "53 bits")]
        let unit = (self.next() >> 11) as f64 / (1_u64 << 53) as f64;
        unit
    }

    fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }

    /// Uniform in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        #[expect(clippy::cast_possible_truncation, reason = "below `n`")]
        let below = (self.next() % n as u64) as usize;
        below
    }

    /// Log-uniform in `[low, high]`.
    fn log_uniform(&mut self, low: f64, high: f64) -> f64 {
        (low.ln() + self.unit() * (high.ln() - low.ln())).exp()
    }
}

/// What a relation is, which sets its size.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    /// 1e6 to 1e9 rows.
    Fact,
    /// 1e3 to 1e5 rows, or 1e2 to 1e5 outside a snowflake.
    Dimension,
    /// 1e2 to 1e4 rows.
    SubDimension,
}

/// A generated graph, before it's a `QueryGraph`.
struct Spec {
    name: String,
    relations: Vec<Relation>,
    edges: Vec<EdgeSpec>,
    domains: Vec<u64>,
}

struct EdgeSpec {
    left: Vec<u32>,
    right: Vec<u32>,
    kind: JoinKind,
    equivalence: u32,
}

#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "rounding estimates")]
fn round(rows: f64) -> u64 {
    rows.round().max(1.0) as u64
}

/// A graph of `shape` with `count` relations, written left-deep in their
/// order: each relation after the first joins one before it, its parent, and
/// maybe more. If `mixed`, about 20% of relations join by LEFT, SEMI or ANTI.
#[expect(clippy::too_many_lines, reason = "one generator for every shape")]
fn generate(shape: Shape, count: usize, mixed: bool) -> Spec {
    let joins = if mixed { "mixed" } else { "inner" };
    let name = format!("{}-{count}-{joins}", shape.name());
    let seed = name
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3));
    let mut rng = Rng(seed);

    let dimensions = (count - 1).div_ceil(5) * 2;
    let roles: Vec<Role> = (0..count)
        .map(|i| match shape {
            Shape::Star | Shape::Snowflake if i == 0 => Role::Fact,
            Shape::Star => Role::Dimension,
            Shape::Snowflake if i <= dimensions => Role::Dimension,
            Shape::Snowflake => Role::SubDimension,
            _ if i == 0 || rng.chance(0.25) => Role::Fact,
            _ => Role::Dimension,
        })
        .collect();
    let relations: Vec<Relation> = roles
        .iter()
        .map(|role| {
            let unfiltered = match role {
                Role::Fact => rng.log_uniform(1e6, 1e9),
                Role::Dimension if shape == Shape::Snowflake => rng.log_uniform(1e3, 1e5),
                Role::Dimension => rng.log_uniform(1e2, 1e5),
                Role::SubDimension => rng.log_uniform(1e2, 1e4),
            };
            let selectivity = if rng.chance(0.4) { 1.0 } else { rng.log_uniform(1e-3, 1.0) };
            Relation { rows: round(unfiltered * selectivity), unfiltered_rows: round(unfiltered) }
        })
        .collect();

    // The pairs joined, earlier relation first, parents' joins first.
    let mut pairs: Vec<(usize, usize)> = (1..count)
        .map(|k| {
            let parent = match shape {
                Shape::Chain | Shape::Cycle | Shape::Clique => k - 1,
                Shape::Star => 0,
                Shape::Snowflake if k <= dimensions => 0,
                Shape::Snowflake => 1 + rng.below(k - 1),
                Shape::Random => rng.below(k),
            };
            (parent, k)
        })
        .collect();
    match shape {
        Shape::Cycle if count > 2 => pairs.push((0, count - 1)),
        Shape::Clique => {
            for j in 2..count {
                pairs.extend((0..j - 1).map(|i| (i, j)));
            }
        }
        Shape::Random => {
            for _ in 0..count / 2 {
                let (a, b) = (rng.below(count), rng.below(count));
                let pair = (a.min(b), a.max(b));
                if a != b && !pairs.contains(&pair) {
                    pairs.push(pair);
                }
            }
        }
        _ => {}
    }

    // A clique's relations come in groups of 4 joined on one key, so on one
    // equivalence set, whose domain is its fewest rows.
    let group = |i: usize| (shape == Shape::Clique).then_some(i / 4);
    let mut domains = Vec::new();
    let mut groups: HashMap<usize, u32> = HashMap::new();
    let mut equivalences = Vec::with_capacity(pairs.len());
    for &(i, j) in &pairs {
        let (a, b) = (relations[i].unfiltered_rows, relations[j].unfiltered_rows);
        if let (Some(g), true) = (group(i), group(i) == group(j)) {
            let id = *groups.entry(g).or_insert_with(|| {
                let fewest =
                    (g * 4..(g * 4 + 4).min(count)).map(|r| relations[r].unfiltered_rows).min();
                domains.push(fewest.unwrap_or(1));
                u32::try_from(domains.len() - 1).unwrap_or(u32::MAX)
            });
            equivalences.push(id);
            continue;
        }
        // N:1 on the smaller side's unique key, more often from a fact to
        // a dimension, else N:M on fewer distinct values.
        let fact_to_dimension = (roles[i] == Role::Fact) != (roles[j] == Role::Fact);
        let unique = if fact_to_dimension || shape == Shape::Snowflake { 0.95 } else { 0.85 };
        let domain = if rng.chance(unique) {
            a.min(b)
        } else {
            #[expect(clippy::cast_precision_loss, reason = "an estimate")]
            let fewer = a.min(b) as f64 * rng.log_uniform(0.1, 1.0);
            round(fewer)
        };
        domains.push(domain);
        equivalences.push(u32::try_from(domains.len() - 1).unwrap_or(u32::MAX));
    }

    // Which relations join by a non-inner join, and how. A relation others
    // join to later can only be a LEFT join's, whose rows they can see.
    let mut kinds = vec![JoinKind::Inner; count];
    if mixed {
        for (k, kind) in kinds.iter_mut().enumerate().skip(1) {
            if !rng.chance(0.2) {
                continue;
            }
            let later = pairs.iter().any(|&(i, _)| i == k);
            *kind = match rng.below(if later { 1 } else { 3 }) {
                0 => JoinKind::Left,
                1 => JoinKind::Semi,
                _ => JoinKind::Anti,
            };
        }
    }

    // A non-inner join's TES is everything before it as written. So is an
    // inner join's that reads a LEFT join's right side, which it can't pass.
    let edges = pairs
        .iter()
        .zip(equivalences)
        .map(|(&(i, j), equivalence)| {
            #[expect(clippy::cast_possible_truncation, reason = "at most 500")]
            let (i, j, everything) = (i as u32, j as u32, (0..j as u32).collect());
            match kinds[j as usize] {
                JoinKind::Inner if kinds[i as usize] == JoinKind::Left => EdgeSpec {
                    left: everything,
                    right: vec![j],
                    kind: JoinKind::Inner,
                    equivalence,
                },
                JoinKind::Inner => {
                    EdgeSpec { left: vec![i], right: vec![j], kind: JoinKind::Inner, equivalence }
                }
                kind => EdgeSpec { left: everything, right: vec![j], kind, equivalence },
            }
        })
        .collect();
    Spec { name, relations, edges, domains }
}

fn fail(error: impl Debug) -> String {
    format!("{error:?}")
}

fn relation_set(relations: &[u32]) -> Result<RelationSet, String> {
    let mut set = RelationSet::new(&Heap).map_err(fail)?;
    for &relation in relations {
        set.insert(relation).map_err(fail)?;
    }
    Ok(set)
}

fn query_graph(spec: &Spec) -> Result<QueryGraph, String> {
    let mut graph = QueryGraph::new(&Heap).map_err(fail)?;
    for &relation in &spec.relations {
        graph.relations.push(relation).map_err(fail)?;
    }
    for edge in &spec.edges {
        graph
            .edges
            .push(JoinEdge {
                left: relation_set(&edge.left)?,
                right: relation_set(&edge.right)?,
                kind: edge.kind,
                equivalence: edge.equivalence,
            })
            .map_err(fail)?;
    }
    for &domain in &spec.domains {
        graph.equivalences.push(EquivalenceSet { domain }).map_err(fail)?;
    }
    Ok(graph)
}

fn mask(relations: &[u32]) -> u32 {
    relations.iter().fold(0, |mask, &relation| mask | 1 << relation)
}

/// The cheapest valid tree of `spec`, by dynamic programming over every
/// subset of its relations, and its cost. Each subset keeps the rows of its
/// cheapest plan, as the kernel's estimates would give them.
fn exhaustive(spec: &Spec, connected: bool) -> Result<(JoinTree, f64), String> {
    let count = spec.relations.len();
    let edges: Vec<(u32, u32, JoinKind, u32)> = spec
        .edges
        .iter()
        .map(|edge| (mask(&edge.left), mask(&edge.right), edge.kind, edge.equivalence))
        .collect();
    let subsets = 1_usize << count;
    let mut best = vec![f64::INFINITY; subsets];
    let mut rows = vec![0.0_f64; subsets];
    let mut split = vec![0_u32; subsets];
    for (relation, found) in spec.relations.iter().enumerate() {
        best[1 << relation] = 0.0;
        #[expect(clippy::cast_precision_loss, reason = "an estimate")]
        let found = found.rows as f64;
        rows[1 << relation] = found;
    }
    let mut seen = Seen { stamps: vec![u32::MAX; spec.domains.len()], stamp: 0 };
    let mut within = Vec::new();
    #[expect(clippy::cast_possible_truncation, reason = "at most 2^20 subsets")]
    for set in 1..subsets as u32 {
        if set.is_power_of_two() {
            continue;
        }
        within.clear();
        within.extend(edges.iter().filter(|edge| (edge.0 | edge.1) & !set == 0));
        let low = set.isolate_lowest_one();
        let rest = set ^ low;
        let mut sub = rest;
        loop {
            let left = sub | low;
            let right = set ^ left;
            sub = sub.wrapping_sub(1) & rest;
            let (l, r) = (left as usize, right as usize);
            if right != 0 && best[l].is_finite() && best[r].is_finite() {
                let sides = [(left, rows[l]), (right, rows[r])];
                if let Some(joined) = join(&spec.domains, &within, &mut seen, sides, connected) {
                    let total = (best[l] + best[r] + joined.max(1.0)).min(ROWS_MAX);
                    if total < best[set as usize] {
                        (best[set as usize], rows[set as usize], split[set as usize]) =
                            (total, joined, left);
                    }
                }
            }
            if sub == rest {
                break;
            }
        }
    }
    let all = subsets - 1;
    if best[all].is_infinite() {
        return Err("no valid tree".into());
    }
    let mut tree = JoinTree::new(&Heap, count).map_err(fail)?;
    build(&mut tree, &split, all)?;
    Ok((tree, best[all]))
}

/// The equivalence sets a join has divided by: those stamped with its stamp.
struct Seen {
    stamps: Vec<u32>,
    stamp: u32,
}

/// The rows joining two sets of relations, each a mask and its rows, by
/// the edges `within` their union makes, as the kernel's `Estimator`
/// does, or `None` if the join isn't valid.
fn join(
    domains: &[u64],
    within: &[&(u32, u32, JoinKind, u32)],
    seen: &mut Seen,
    [(left, left_rows), (right, right_rows)]: [(u32, f64); 2],
    connected: bool,
) -> Option<f64> {
    seen.stamp += 1;
    let mut rows = left_rows * right_rows;
    let (mut any, mut kept) = (false, None);
    for &&(l, r, kind, equivalence) in within {
        let forward = l & !left == 0 && r & !right == 0;
        if !(forward || l & !right == 0 && r & !left == 0) {
            if (l | r) & !left != 0 && (l | r) & !right != 0 {
                return None;
            }
            continue;
        }
        any = true;
        if kind != JoinKind::Inner {
            kept = Some((kind, if forward { left_rows } else { right_rows }));
        }
        if seen.stamps[equivalence as usize] != seen.stamp {
            seen.stamps[equivalence as usize] = seen.stamp;
            #[expect(clippy::cast_precision_loss, reason = "an estimate")]
            let domain = domains[equivalence as usize].max(1) as f64;
            rows /= domain;
        }
    }
    if !any && connected {
        return None;
    }
    let rows = match kept {
        None => rows,
        Some((JoinKind::Left, kept)) => rows.max(kept),
        Some((JoinKind::Semi, kept)) => rows.min(kept),
        Some((_, kept)) => kept * (1.0 - (rows / kept).min(1.0)).max(0.1),
    };
    Some(rows.min(ROWS_MAX))
}

/// Adds the plan for `set` that `split` chose to `tree`, and returns its node.
fn build(tree: &mut JoinTree, split: &[u32], set: usize) -> Result<u32, String> {
    if set.is_power_of_two() {
        return tree.relation(set.trailing_zeros()).map_err(fail);
    }
    let left = split[set] as usize;
    let left = build(tree, split, left)?;
    let right = build(tree, split, set ^ split[set] as usize)?;
    tree.join(left, right).map_err(fail)
}

/// What an enumerator did with one graph.
struct Outcome {
    cost: f64,
    valid: Result<(), String>,
    /// The median of repeated runs.
    time: Duration,
    /// The most bytes it held at once.
    peak: usize,
}

fn run(enumerator: Enumerator, graph: &QueryGraph) -> Result<Outcome, String> {
    let limit = LimitAllocator::new(&Heap, usize::MAX);
    let tree = enumerator(&limit, graph).map_err(fail)?;
    let costed = cost(&Heap, graph, &tree).map_err(fail)?;
    drop(tree);
    let peak = limit.peak();

    let mut times = Vec::new();
    let started = Instant::now();
    while times.len() < 3 || (started.elapsed() < TIMING && times.len() < 101) {
        let start = Instant::now();
        let tree = enumerator(&Heap, graph).map_err(fail)?;
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

fn print_graph(spec: &Spec) {
    println!("{}: {} relations", spec.name, spec.relations.len());
    for (i, relation) in spec.relations.iter().enumerate() {
        println!("  r{i}: {} of {} rows", relation.rows, relation.unfiltered_rows);
    }
    for edge in &spec.edges {
        let domain = spec.domains[edge.equivalence as usize];
        println!(
            "  {:?} {:?} {:?} on e{} (domain {domain})",
            edge.left, edge.kind, edge.right, edge.equivalence
        );
    }
}

/// Options from the command line.
struct Options {
    enumerators: Vec<(&'static str, Enumerator)>,
    shapes: Vec<Shape>,
    sizes: Vec<usize>,
    mixed: Vec<bool>,
    exhaustive: usize,
    best: Option<String>,
    graph: Option<String>,
}

fn options() -> Result<Options, String> {
    let mut options = Options {
        enumerators: Vec::new(),
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
            "--enumerator" => {
                let name = value()?;
                let found = ENUMERATORS.iter().find(|(n, _)| *n == name);
                options.enumerators.push(*found.ok_or(format!("no enumerator {name}"))?);
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
            "--list" => {
                for (name, _) in ENUMERATORS {
                    println!("{name}");
                }
                std::process::exit(0);
            }
            _ => return Err(format!("unknown option {arg}")),
        }
    }
    if options.enumerators.is_empty() {
        options.enumerators = ENUMERATORS.to_vec();
    }
    if options.sizes.iter().any(|&size| size < 2) {
        return Err("graphs need 2 relations or more".into());
    }
    Ok(options)
}

/// Each enumerator's ratios to the optimum and to the best known, invalid
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
    if let Some(name) = &options.graph {
        let parts: Vec<&str> = name.split('-').collect();
        let shape = SHAPES.into_iter().find(|shape| parts.first() == Some(&shape.name()));
        let size = parts.get(1).and_then(|size| size.parse().ok());
        let mixed = parts.get(2).map(|&joins| joins == "mixed");
        let (Some(shape), Some(size @ 2..), Some(mixed)) = (shape, size, mixed) else {
            return Err(format!("no graph {name}"));
        };
        print_graph(&generate(shape, size, mixed));
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

    let mut summaries: Vec<Summary> =
        options.enumerators.iter().map(|_| Summary::default()).collect();
    println!(
        "{:<20} {:>5} {:<14} {:>11} {:>11} {:>4} {:>9} {:>9} {:>10}  valid",
        "graph", "edges", "enumerator", "cost", "reference", "of", "ratio", "time", "peak"
    );
    for &shape in &options.shapes {
        for &size in &options.sizes {
            for &mixed in &options.mixed {
                let spec = generate(shape, size, mixed);
                let graph = query_graph(&spec)?;
                let connected = graph.is_connected(&Heap).map_err(fail)?;
                let mut reference = None;
                if size <= options.exhaustive {
                    let (tree, optimum) = exhaustive(&spec, connected)?;
                    let costed = cost(&Heap, &graph, &tree).map_err(fail)?;
                    if (costed.cost - optimum).abs() > optimum * 1e-9 || costed.valid.is_err() {
                        return Err(format!(
                            "{}: the exhaustive search's plan costs {optimum}, but the kernel says {costed:?}",
                            spec.name
                        ));
                    }
                    reference = Some((optimum, "opt"));
                }
                let outcomes: Vec<Result<Outcome, String>> =
                    options.enumerators.iter().map(|&(_, e)| run(e, &graph)).collect();
                let found = outcomes
                    .iter()
                    .filter_map(|o| o.as_ref().ok().filter(|o| o.valid.is_ok()).map(|o| o.cost))
                    .chain(best.get(&spec.name).copied())
                    .reduce(f64::min);
                if reference.is_none() {
                    reference = found.map(|cost| (cost, "best"));
                }
                if let Some(cost) = reference.map(|(cost, _)| cost) {
                    let known = best.entry(spec.name.clone()).or_insert(cost);
                    *known = known.min(cost);
                }
                for ((&(name, _), outcome), summary) in
                    options.enumerators.iter().zip(&outcomes).zip(&mut summaries)
                {
                    let outcome = match outcome {
                        Ok(outcome) => outcome,
                        Err(error) => {
                            summary.failed += 1;
                            println!(
                                "{:<20} {:>5} {name:<14} failed: {error}",
                                spec.name,
                                spec.edges.len()
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
                        "{:<20} {:>5} {name:<14} {:>11} {:>11} {of:>4} {:>9} {:>9} {:>10}  {}",
                        spec.name,
                        spec.edges.len(),
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
        "enumerator", "vs opt", "vs best", "worst", "invalid", "failed", "slowest"
    );
    for (&(name, _), summary) in options.enumerators.iter().zip(&summaries) {
        let mean = |log: f64, count: u32| (log / f64::from(count.max(1))).exp();
        println!(
            "{name:<14} {:>10} {:>10} {:>10} {:>8} {:>7} {:>9}",
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
