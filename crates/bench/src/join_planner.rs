//! The reference the join planner is measured against: query graphs of many
//! shapes, the cost and validity of a join tree, the written order, and the
//! optimum by exhaustive search.

use std::collections::HashMap;
use std::fmt::Debug;

use pipit_kernel::env::Env;
use pipit_kernel::error::Error;
use pipit_kernel::join_planner::{
    BUILD_COST, EquivalenceSet, Estimator, JoinEdge, JoinKind, JoinNode, JoinTree, QueryGraph,
    ROWS_MAX, Relation, RelationSet,
};

/// Picks a valid join tree for a graph, with memory from an `Env`.
pub type Enumerator = fn(&Env, &QueryGraph) -> Result<JoinTree, Error>;

/// The most relations the exhaustive search takes, as its time is 3^n.
pub const EXHAUSTIVE_MAX: usize = 20;

/// The shape of a generated graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    /// Each relation joins the one before it.
    Chain,
    /// Each relation joins the first.
    Star,
    /// Dimensions join the first relation, and sub-dimensions join them.
    Snowflake,
    /// A chain whose last relation also joins the first.
    Cycle,
    /// Every relation joins every other.
    Clique,
    /// Each relation joins a random one before it, and half as many more
    /// joins are added at random.
    Random,
}

/// Every shape.
pub const SHAPES: [Shape; 6] =
    [Shape::Chain, Shape::Star, Shape::Snowflake, Shape::Cycle, Shape::Clique, Shape::Random];

impl Shape {
    pub fn name(self) -> &'static str {
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

fn fail(error: impl Debug) -> String {
    format!("{error:?}")
}

#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "rounding estimates")]
fn round(rows: f64) -> u64 {
    rows.round().max(1.0) as u64
}

fn relation_set(relations: impl IntoIterator<Item = u32>) -> Result<RelationSet, String> {
    let mut set = RelationSet::new(crate::env()).map_err(fail)?;
    for relation in relations {
        set.insert(relation).map_err(fail)?;
    }
    Ok(set)
}

/// A graph's name, such as `star-10-mixed`.
pub fn name(shape: Shape, count: usize, mixed: bool) -> String {
    format!("{}-{count}-{}", shape.name(), if mixed { "mixed" } else { "inner" })
}

/// A graph of `shape` with `count` relations, written left-deep in their
/// order: each relation after the first joins one before it, its parent, and
/// maybe more. If `mixed`, about 20% of relations join by LEFT, SEMI or ANTI.
#[expect(clippy::too_many_lines, reason = "one generator for every shape")]
pub fn generate(shape: Shape, count: usize, mixed: bool) -> Result<QueryGraph, String> {
    let seed = name(shape, count, mixed)
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
            let unfiltered_rows = round(unfiltered);
            Relation {
                rows: round(unfiltered * selectivity),
                unfiltered_rows,
                max_rows: unfiltered_rows,
            }
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

    let mut graph = QueryGraph::new(crate::env()).map_err(fail)?;
    for relation in relations {
        graph.relations.push(relation).map_err(fail)?;
    }
    for domain in domains {
        graph.equivalences.push(EquivalenceSet { domain }).map_err(fail)?;
    }
    // A non-inner join's TES is everything before it as written. So is an
    // inner join's that reads a LEFT join's right side, which it can't pass.
    for (&(i, j), equivalence) in pairs.iter().zip(equivalences) {
        #[expect(clippy::cast_possible_truncation, reason = "at most 1000")]
        let (i, j) = (i as u32, j as u32);
        let (left, kind) = match kinds[j as usize] {
            JoinKind::Inner if kinds[i as usize] == JoinKind::Left => (0..j, JoinKind::Inner),
            JoinKind::Inner => (i..i + 1, JoinKind::Inner),
            kind => (0..j, kind),
        };
        let (left, right) = (relation_set(left)?, relation_set([j])?);
        graph.edges.push(JoinEdge { left, right, kind, equivalence }).map_err(fail)?;
    }
    Ok(graph)
}

/// The words of `set`.
fn words(set: &RelationSet) -> Vec<u64> {
    (0..set.words()).map(|i| set.word(i)).collect()
}

/// The rows joining `left`, of `left_rows` rows, to `right`, of
/// `right_rows`, makes by every edge between them, and whether any edge
/// joins them.
pub fn estimate(
    estimator: &mut Estimator,
    graph: &QueryGraph,
    [(left, left_rows), (right, right_rows)]: [(&RelationSet, f64); 2],
) -> (f64, bool) {
    estimator.start(left_rows, right_rows);
    let mut any = false;
    for edge in graph.edges.iter() {
        let forward = edge.left.is_subset(left) && edge.right.is_subset(right);
        if forward || edge.left.is_subset(right) && edge.right.is_subset(left) {
            any = true;
            estimator.apply(edge, forward);
        }
    }
    (estimator.finish(), any)
}

/// Whether the edges join every relation to every other, possibly through
/// others. If not, a plan needs cross products.
pub fn is_connected(graph: &QueryGraph) -> Result<bool, String> {
    if graph.relations.len() < 2 {
        return Ok(true);
    }
    let mut reached = relation_set([0])?;
    let mut grew = true;
    while grew {
        grew = false;
        for edge in graph.edges.iter() {
            let touches = edge.left.overlaps(&reached) || edge.right.overlaps(&reached);
            if touches && !(edge.left.is_subset(&reached) && edge.right.is_subset(&reached)) {
                reached.union_with(&edge.left).map_err(fail)?;
                reached.union_with(&edge.right).map_err(fail)?;
                grew = true;
            }
        }
    }
    Ok(reached.count() as usize == graph.relations.len())
}

/// Whether `set` is one relation of at most one row, by its bound, which may
/// be cross-joined to anything.
fn one_row(graph: &QueryGraph, set: &RelationSet) -> bool {
    let relations = 0..u32::try_from(graph.relations.len()).unwrap_or(u32::MAX);
    set.count() == 1
        && relations.zip(graph.relations.iter()).any(|(i, r)| set.contains(i) && r.max_rows <= 1)
}

/// Why a tree isn't a valid plan for its graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Invalid {
    /// It has no nodes, a node refers to one that doesn't exist, or a node
    /// isn't reached from the root exactly once.
    Shape,
    /// A relation is missing, repeated, or not in the graph.
    Relations,
    /// This edge's relations first meet at a join that can't apply it,
    /// because the join's sides don't split the edge's sides.
    Edge(u32),
    /// A join has no edge, though the graph is connected, and neither side
    /// is one relation of at most one row.
    CrossProduct,
    /// A join the graph's `executable` says can't run.
    Executable,
}

/// A tree's cost, and whether it's valid.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Costed {
    /// The rows of every join, each at least 1, and `BUILD_COST` for each
    /// row of every join's smaller side, added up, at most `ROWS_MAX`.
    pub cost: f64,
    /// Whether the tree is valid, or why not.
    pub valid: Result<(), Invalid>,
}

/// Whether joining `left` and `right` is valid: it applies every edge whose
/// relations it brings together, which needs one side of the edge in each,
/// and the operator can run it.
fn check_join(graph: &QueryGraph, left: &RelationSet, right: &RelationSet) -> Result<(), Invalid> {
    for (index, edge) in (0_u32..).zip(graph.edges.iter()) {
        let meets = edge.left.is_subset_of_union(left, right)
            && edge.right.is_subset_of_union(left, right)
            && !(edge.left.is_subset(left) && edge.right.is_subset(left))
            && !(edge.left.is_subset(right) && edge.right.is_subset(right));
        let applies = (edge.left.is_subset(left) && edge.right.is_subset(right))
            || (edge.left.is_subset(right) && edge.right.is_subset(left));
        if meets && !applies {
            return Err(Invalid::Edge(index));
        }
    }
    match graph.executable {
        Some(executable) if !executable(graph, &words(left), &words(right)) => {
            Err(Invalid::Executable)
        }
        _ => Ok(()),
    }
}

/// The nodes of `tree` in an order that puts each join after its children,
/// found by walking down from the root. `None` if a node is reached more than
/// once, or not at all, or a join refers to a node that doesn't exist.
fn post_order(tree: &JoinTree) -> Option<Vec<usize>> {
    let count = tree.nodes.len();
    let (mut order, mut reached) = (Vec::with_capacity(count), vec![false; count]);
    let mut stack = vec![(count.checked_sub(1)?, false)];
    while let Some((node, done)) = stack.pop() {
        if done {
            order.push(node);
            continue;
        }
        if std::mem::replace(reached.get_mut(node)?, true) {
            return None;
        }
        stack.push((node, true));
        if let JoinNode::Join { left, right } = tree.nodes.get(node)?.kind {
            stack.extend([(right as usize, false), (left as usize, false)]);
        }
    }
    (order.len() == count).then_some(order)
}

/// What a join of sides of `rows` that makes `joined` rows adds to a tree's
/// cost, as the planner counts it.
fn join_cost(rows: [f64; 2], joined: f64) -> f64 {
    joined.max(1.0) + BUILD_COST * rows[0].min(rows[1])
}

/// The cost of `tree` for `graph`, and whether it's valid. A valid tree has
/// every relation once, applies every edge where its relations meet, has no
/// cross product unless the graph isn't connected or a side is one relation
/// of at most one row, and has only joins the operator can run. A tree whose shape isn't valid has no cost.
pub fn cost(graph: &QueryGraph, tree: &JoinTree) -> Result<Costed, String> {
    let invalid = |why| Ok(Costed { cost: ROWS_MAX, valid: Err(why) });
    let Some(order) = post_order(tree) else { return invalid(Invalid::Shape) };
    let connected = is_connected(graph)?;
    let mut estimator = Estimator::new(crate::env(), graph).map_err(fail)?;
    // Each node's relations and rows.
    let mut sets: Vec<Option<RelationSet>> = (0..order.len()).map(|_| None).collect();
    let mut rows = vec![0.0; order.len()];
    let mut seen = RelationSet::new(crate::env()).map_err(fail)?;
    let (mut total, mut valid) = (0.0_f64, Ok(()));
    for node in order {
        let (set, node_rows) = match tree.nodes.get(node).map(|node| node.kind) {
            Some(JoinNode::Relation(relation)) => {
                let Some(found) = graph.relations.get(relation as usize) else {
                    return invalid(Invalid::Relations);
                };
                if seen.contains(relation) {
                    return invalid(Invalid::Relations);
                }
                seen.insert(relation).map_err(fail)?;
                #[expect(clippy::cast_precision_loss, reason = "an estimate")]
                let found = found.rows as f64;
                (relation_set([relation])?, found)
            }
            Some(JoinNode::Join { left, right }) => {
                let (left, right) = (left as usize, right as usize);
                let (Some(mut set), Some(other)) = (sets[left].take(), sets[right].take()) else {
                    return invalid(Invalid::Shape);
                };
                let sides = [(&set, rows[left]), (&other, rows[right])];
                let (joined, any) = estimate(&mut estimator, graph, sides);
                if valid.is_ok() {
                    valid = check_join(graph, &set, &other);
                    let one_row = one_row(graph, &set) || one_row(graph, &other);
                    if valid.is_ok() && !any && connected && !one_row {
                        valid = Err(Invalid::CrossProduct);
                    }
                }
                set.union_with(&other).map_err(fail)?;
                total = (total + join_cost([rows[left], rows[right]], joined)).min(ROWS_MAX);
                (set, joined)
            }
            None => return invalid(Invalid::Shape),
        };
        sets[node] = Some(set);
        rows[node] = node_rows;
    }
    if seen.count() as usize != graph.relations.len() {
        return invalid(Invalid::Relations);
    }
    Ok(Costed { cost: total, valid })
}

/// The relations joined left-deep in the order they're written: the
/// baseline plans are measured against.
pub fn as_written(env: &Env, graph: &QueryGraph) -> Result<JoinTree, Error> {
    let count = graph.relations.len();
    let mut estimator = Estimator::new(env, graph)?;
    let mut tree = JoinTree::new(env, count)?;
    let (mut joined, mut root, mut rows) = (RelationSet::new(env)?, 0, 0.0);
    #[expect(clippy::cast_possible_truncation, reason = "at most `u32::MAX`")]
    for relation in 0..count as u32 {
        #[expect(clippy::cast_precision_loss, reason = "an estimate")]
        let leaf_rows = graph.relations.get(relation as usize).map_or(0.0, |r| r.rows as f64);
        let leaf = tree.relation(relation, leaf_rows)?;
        let mut set = RelationSet::new(env)?;
        set.insert(relation)?;
        (root, rows) = if relation == 0 {
            (leaf, leaf_rows)
        } else {
            let joined_rows =
                estimate(&mut estimator, graph, [(&joined, rows), (&set, leaf_rows)]).0;
            (tree.join(root, leaf, joined_rows)?, joined_rows)
        };
        joined.union_with(&set)?;
    }
    Ok(tree)
}

/// The planner's tree for `graph`, or the written order where it keeps that.
pub fn plan(env: &Env, graph: &QueryGraph) -> Result<JoinTree, Error> {
    pipit_kernel::join_planner::plan(env, graph)?.map_or_else(|| as_written(env, graph), Ok)
}

/// The cheapest valid tree of `graph`, of at most `EXHAUSTIVE_MAX`
/// relations, by dynamic programming over every subset of its relations,
/// and its cost. Each subset keeps the rows of its cheapest tree.
pub fn exhaustive(graph: &QueryGraph) -> Result<(JoinTree, f64), String> {
    let count = graph.relations.len();
    if count > EXHAUSTIVE_MAX {
        return Err(format!("{count} relations is too many to search"));
    }
    let connected = is_connected(graph)?;
    #[expect(clippy::cast_possible_truncation, reason = "at most `EXHAUSTIVE_MAX` relations")]
    let mask = |set: &RelationSet| set.word(0) as u32;
    let edges: Vec<(u32, u32, &JoinEdge)> =
        graph.edges.iter().map(|edge| (mask(&edge.left), mask(&edge.right), edge)).collect();
    let subsets = 1_usize << count;
    let (mut best, mut rows, mut split) =
        (vec![f64::INFINITY; subsets], vec![0.0_f64; subsets], vec![0_u32; subsets]);
    for (relation, found) in graph.relations.iter().enumerate() {
        best[1 << relation] = 0.0;
        #[expect(clippy::cast_precision_loss, reason = "an estimate")]
        let found = found.rows as f64;
        rows[1 << relation] = found;
    }
    let mut estimator = Estimator::new(crate::env(), graph).map_err(fail)?;
    #[expect(clippy::cast_possible_truncation, reason = "at most 2^20 subsets")]
    for set in 1..subsets as u32 {
        let within: Vec<_> = edges.iter().filter(|edge| (edge.0 | edge.1) & !set == 0).collect();
        let low = set.isolate_lowest_one();
        let rest = set ^ low;
        let mut sub = rest;
        while sub != 0 {
            sub = (sub - 1) & rest;
            let (left, right) = (sub | low, rest ^ sub);
            let (l, r) = (left as usize, right as usize);
            if !(best[l] + best[r]).is_finite() {
                continue;
            }
            // Every edge between the sides must apply, and one must unless
            // the graph isn't connected.
            estimator.start(rows[l], rows[r]);
            let (mut any, mut valid) = (false, true);
            for &&(edge_left, edge_right, edge) in &within {
                let forward = edge_left & !left == 0 && edge_right & !right == 0;
                if forward || edge_left & !right == 0 && edge_right & !left == 0 {
                    any = true;
                    estimator.apply(edge, forward);
                } else if (edge_left | edge_right) & !left != 0
                    && (edge_left | edge_right) & !right != 0
                {
                    valid = false;
                }
            }
            if let Some(executable) = graph.executable {
                valid &= executable(graph, &[u64::from(left)], &[u64::from(right)]);
            }
            let one_row = |set: u32| {
                let relation = graph.relations.get(set.trailing_zeros() as usize);
                set.is_power_of_two() && relation.is_some_and(|relation| relation.max_rows <= 1)
            };
            if !valid || !any && connected && !one_row(left) && !one_row(right) {
                continue;
            }
            let joined = estimator.finish();
            let total = (best[l] + best[r] + join_cost([rows[l], rows[r]], joined)).min(ROWS_MAX);
            if total < best[set as usize] {
                (best[set as usize], rows[set as usize], split[set as usize]) =
                    (total, joined, left);
            }
        }
    }
    let all = subsets - 1;
    if best[all].is_infinite() {
        return Err("no valid tree".into());
    }
    let mut tree = JoinTree::new(crate::env(), count).map_err(fail)?;
    build(&mut tree, &split, &rows, all)?;
    Ok((tree, best[all]))
}

/// Adds the tree for `set` that `split` chose to `tree`, and returns its
/// node.
fn build(tree: &mut JoinTree, split: &[u32], rows: &[f64], set: usize) -> Result<u32, String> {
    if set.is_power_of_two() {
        return tree.relation(set.trailing_zeros(), rows[set]).map_err(fail);
    }
    let left = build(tree, split, rows, split[set] as usize)?;
    let right = build(tree, split, rows, set ^ split[set] as usize)?;
    tree.join(left, right, rows[set]).map_err(fail)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fact of 1000 rows, joined to a dimension of 10 on its key, and
    /// left joined to another of 100, which then joins the first dimension.
    fn star() -> QueryGraph {
        let mut graph = QueryGraph::new(crate::env()).unwrap();
        for rows in [1000, 10, 100] {
            graph.relations.push(Relation { rows, unfiltered_rows: rows, max_rows: rows }).unwrap();
        }
        for domain in [10, 100] {
            graph.equivalences.push(EquivalenceSet { domain }).unwrap();
        }
        let edges = [
            (&[0][..], 1, JoinKind::Inner, 0),
            (&[0, 1], 2, JoinKind::Left, 1),
            (&[1], 2, JoinKind::Inner, 1),
        ];
        for (left, right, kind, equivalence) in edges {
            let (left, right) =
                (relation_set(left.iter().copied()).unwrap(), relation_set([right]).unwrap());
            graph.edges.push(JoinEdge { left, right, kind, equivalence }).unwrap();
        }
        graph
    }

    /// A tree joining `first` and `second`, then that to `third`.
    fn left_deep([first, second, third]: [u32; 3]) -> JoinTree {
        let mut tree = JoinTree::new(crate::env(), 3).unwrap();
        let (a, b) = (tree.relation(first, 0.0).unwrap(), tree.relation(second, 0.0).unwrap());
        let joined = tree.join(a, b, 0.0).unwrap();
        let c = tree.relation(third, 0.0).unwrap();
        tree.join(joined, c, 0.0).unwrap();
        tree
    }

    #[test]
    fn costs_and_checks_trees() {
        let graph = star();
        let written = as_written(crate::env(), &graph).unwrap();
        // 1000 * 10 / 10, then a left join keeping all 1000, each building
        // its smaller side's 10, then 100, rows at `BUILD_COST` each.
        let cost_of = 2000.0 + BUILD_COST * 110.0;
        assert_eq!(cost(&graph, &written).unwrap(), Costed { cost: cost_of, valid: Ok(()) });
        // The left join before the inner one breaks its TES.
        assert_eq!(cost(&graph, &left_deep([1, 2, 0])).unwrap().valid, Err(Invalid::Edge(1)));
        assert_eq!(cost(&graph, &left_deep([0, 2, 1])).unwrap().valid, Err(Invalid::CrossProduct));
    }

    #[test]
    fn exhaustive_search_finds_valid_trees() {
        let graph = star();
        let (tree, optimum) = exhaustive(&graph).unwrap();
        assert_eq!(cost(&graph, &tree).unwrap(), Costed { cost: optimum, valid: Ok(()) });
    }

    /// Each relation left joined to all those before it: a long graph with
    /// few connected sets, so it's planned exactly.
    fn left_chain(count: u32) -> QueryGraph {
        let mut graph = QueryGraph::new(crate::env()).unwrap();
        for relation in 0..count {
            let rows = 10 + u64::from(relation * 7919 % 1000);
            graph.relations.push(Relation { rows, unfiltered_rows: rows, max_rows: rows }).unwrap();
            graph.equivalences.push(EquivalenceSet { domain: rows }).unwrap();
            if relation > 0 {
                let (left, right) =
                    (relation_set(0..relation).unwrap(), relation_set([relation]).unwrap());
                let edge = JoinEdge { left, right, kind: JoinKind::Left, equivalence: relation };
                graph.edges.push(edge).unwrap();
            }
        }
        graph
    }

    #[test]
    fn small_graphs_are_planned_optimally() {
        for shape in SHAPES {
            for (count, mixed) in [(2, false), (3, true), (7, false), (10, true)] {
                let graph = generate(shape, count, mixed).unwrap();
                let planned = cost(&graph, &plan(crate::env(), &graph).unwrap()).unwrap();
                let (_, optimum) = exhaustive(&graph).unwrap();
                assert!(planned.valid.is_ok() && planned.cost <= optimum * (1.0 + 1e-9));
            }
        }
    }

    #[test]
    fn larger_graphs_get_valid_plans() {
        for shape in SHAPES {
            for (count, mixed) in [(20, true), (70, false), (70, true)] {
                let graph = generate(shape, count, mixed).unwrap();
                let planned = cost(&graph, &plan(crate::env(), &graph).unwrap()).unwrap();
                let written = cost(&graph, &as_written(crate::env(), &graph).unwrap()).unwrap();
                assert!(planned.valid.is_ok() && planned.cost <= written.cost);
            }
        }
    }

    #[test]
    fn disconnected_graphs_take_cross_products() {
        for count in [5, 70] {
            let mut graph = generate(Shape::Chain, count, true).unwrap();
            graph.relations.push(Relation { rows: 3, unfiltered_rows: 3, max_rows: 3 }).unwrap();
            assert!(cost(&graph, &plan(crate::env(), &graph).unwrap()).unwrap().valid.is_ok());
        }
    }

    #[test]
    fn long_graphs_with_few_connected_sets_are_planned_exactly() {
        let graph = left_chain(50);
        let planned = cost(&graph, &plan(crate::env(), &graph).unwrap()).unwrap();
        let written = cost(&graph, &as_written(crate::env(), &graph).unwrap()).unwrap();
        assert_eq!(planned, written);
    }

    /// A fact of 1000 rows with dimensions of 10 and 20 on its keys, and a
    /// relation of one row: joined to the fact if `connected`, or to nothing.
    fn one_row_star(connected: bool) -> QueryGraph {
        let mut graph = QueryGraph::new(crate::env()).unwrap();
        for (rows, max_rows) in [(1000, 1000), (10, 10), (20, 20), (1, 1)] {
            graph.relations.push(Relation { rows, unfiltered_rows: rows, max_rows }).unwrap();
        }
        let joined: &[u32] = if connected { &[1, 2, 3] } else { &[1, 2] };
        for &dimension in joined {
            let domain = graph.relations.get(dimension as usize).unwrap().rows;
            graph.equivalences.push(EquivalenceSet { domain }).unwrap();
            let (left, right) = (relation_set([0]).unwrap(), relation_set([dimension]).unwrap());
            let edge = JoinEdge { left, right, kind: JoinKind::Inner, equivalence: dimension - 1 };
            graph.edges.push(edge).unwrap();
        }
        graph
    }

    #[test]
    fn one_row_relations_cross_join_anywhere() {
        for connected in [true, false] {
            let graph = one_row_star(connected);
            // The one-row relation joins a dimension, with no edge between.
            let mut tree = JoinTree::new(crate::env(), 4).unwrap();
            let (dimension, one) =
                (tree.relation(1, 10.0).unwrap(), tree.relation(3, 1.0).unwrap());
            let crossed = tree.join(dimension, one, 10.0).unwrap();
            let fact = tree.relation(0, 1000.0).unwrap();
            let joined = tree.join(fact, crossed, 0.0).unwrap();
            let other = tree.relation(2, 20.0).unwrap();
            tree.join(joined, other, 0.0).unwrap();
            assert!(cost(&graph, &tree).unwrap().valid.is_ok());
            let (tree, optimum) = exhaustive(&graph).unwrap();
            assert_eq!(cost(&graph, &tree).unwrap(), Costed { cost: optimum, valid: Ok(()) });
            // A disconnected graph may take any cross product, which exact
            // search doesn't try, so only its validity is checked.
            let planned = cost(&graph, &plan(crate::env(), &graph).unwrap()).unwrap();
            assert!(planned.valid.is_ok());
            assert!(!connected || planned.cost <= optimum * (1.0 + 1e-9));
        }
    }

    #[test]
    fn plans_keep_to_joins_that_can_run() {
        // A chain of 3, whose first two relations can't be joined directly.
        let mut graph = QueryGraph::new(crate::env()).unwrap();
        for rows in [10, 20, 30] {
            graph.relations.push(Relation { rows, unfiltered_rows: rows, max_rows: rows }).unwrap();
            graph.equivalences.push(EquivalenceSet { domain: 10 }).unwrap();
        }
        for (left, right, equivalence) in [(0, 1, 0), (1, 2, 1)] {
            let (left, right) = (relation_set([left]).unwrap(), relation_set([right]).unwrap());
            graph.edges.push(JoinEdge { left, right, kind: JoinKind::Inner, equivalence }).unwrap();
        }
        graph.executable = Some(|_, left, right| (left[0] | right[0]) != 0b11);
        let written = as_written(crate::env(), &graph).unwrap();
        assert_eq!(cost(&graph, &written).unwrap().valid, Err(Invalid::Executable));
        let (tree, _) = exhaustive(&graph).unwrap();
        assert!(cost(&graph, &tree).unwrap().valid.is_ok());
        assert!(cost(&graph, &plan(crate::env(), &graph).unwrap()).unwrap().valid.is_ok());
        // Too big for exact search, so planned greedily, then by windows.
        let mut graph = generate(Shape::Chain, 70, false).unwrap();
        graph.executable = Some(|_, left, right| (left[0] | right[0]) != 0b11);
        assert!(cost(&graph, &plan(crate::env(), &graph).unwrap()).unwrap().valid.is_ok());
    }
}
