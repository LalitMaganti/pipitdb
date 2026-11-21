//! Join ordering after Neumann and Radke's "Adaptive Optimization of Very
//! Large Join Queries" (SIGMOD 2018).
//!
//! A graph is planned exactly by `DPhyp` (Moerkotte and Neumann, SIGMOD
//! 2008) if it has few enough pairs to join. It joins each connected set of
//! relations to each connected set beside it, and keeps the cheapest tree of
//! each set.
//!
//! Any other graph gets a plan from greedy operator ordering (GOO, Fegaras
//! 1998): while more than one tree is left, it joins the two trees whose
//! join makes the fewest rows. Then exact dynamic programming improves it.
//! Bottom up, it takes a window of a few subtrees of the plan and replaces
//! the joins between them by the cheapest way to join them.
//!
//! Everything draws on `WORK_MAX`, so that dense graphs stay bounded. GOO
//! reads a window of the graph's edges for each join, and once the work is
//! spent it takes the first pair of trees that can be joined.
//!
//! Most functions are `#[inline(never)]`: inlined into one another, they were
//! larger.

use core::iter::repeat_n;

use super::{Estimator, JoinTree, LIST_MAX, QueryGraph, RelationSet, as_written};
use crate::allocator::Allocator;
use crate::error::Error;
use crate::slow_vec::SlowVec;

/// No node, tree, edge or unit.
const NONE: u32 = u32::MAX;

/// The most units, relations or subtrees, `DPhyp` joins at once: a set of
/// them is the bits of a `u64`.
const UNITS_MAX: usize = 64;

/// The most connected sets of relations, and pairs of them, `DPhyp` visits to
/// plan a graph exactly.
const VISITS_MAX: usize = 1 << 11;

/// The most relations of a graph that is planned exactly however its
/// relations are joined, as even a clique's 2^n sets and 3^n / 2 pairs are
/// few.
const SMALL_MAX: usize = 10;

/// How many subtrees a window of a bigger graph joins.
const WINDOW: usize = 8;

/// How many pairs of trees each join prices by all the edges between them.
/// They are the pairs whose single edges make the fewest rows. A pair's rows
/// can be far fewer than any one of its edges says, so this has to be more
/// than a few.
const TOP: usize = 16;

/// The most work planning does: edges read, by GOO and by `DPhyp`. It's
/// about 0.8 s at most.
const WORK_MAX: usize = 1 << 27;

/// A node of the plan. The graph's relations come first, then the joins.
#[derive(Clone, Copy)]
struct Node {
    /// The left child, or `NONE` for a relation.
    left: u32,
    /// The right child, or `NONE` for a relation.
    right: u32,
    /// The rows it makes.
    rows: f64,
    /// For a relation, the root of the tree it's in.
    place: u32,
}

/// What the planner reads of the graph's edge of the same index.
#[derive(Clone, Copy)]
struct Edge {
    /// The first relation of its left side, then of its right.
    first: [u32; 2],
    /// Whether its left side, then its right, has more than one relation.
    more: [bool; 2],
    /// For a hyperedge in the list that starts at `Planner::hypers`, the next
    /// one, or `NONE`.
    next: u32,
    /// How many nodes there were when joining its two trees was found to
    /// leave a hyperedge that could never apply. While both trees are older
    /// than that, they still can't be joined.
    dead: u32,
}

/// The cheapest tree found for a set of units, in a hash table.
#[derive(Clone, Copy)]
struct Plan {
    /// The units, or 0 for an empty slot.
    set: u64,
    /// Its cost.
    cost: f64,
    /// Its rows.
    rows: f64,
    /// The units on its left side.
    split: u64,
}

/// An empty slot, or a set of units for which no tree has been found.
const NO_PLAN: Plan = Plan { set: 0, cost: f64::INFINITY, rows: f64::NAN, split: 0 };

/// An edge between units.
#[derive(Clone, Copy)]
struct Link {
    /// The units of its left side, then of its right.
    sides: [u64; 2],
    /// Its index in the graph.
    edge: u32,
}

/// The plan as it is built, and the memory for building it.
struct Planner<'a> {
    /// The graph being planned.
    graph: &'a QueryGraph,
    /// Estimates the rows of the joins it considers.
    estimator: Estimator<'a>,
    /// The work left: see `WORK_MAX`.
    work: usize,
    /// Whether the graph isn't connected, so that a join may be a cross
    /// product.
    crossing: bool,
    /// How many words a set of relations takes.
    words: usize,
    /// The plan's nodes.
    nodes: SlowVec<Node>,
    /// Each node's relations, `words` words a node.
    sets: SlowVec<u64>,
    /// The graph's edges.
    edges: SlowVec<Edge>,
    /// The first of the hyperedges, linked through `Edge::next`, or `NONE`.
    /// A hyperedge with the same sides as the one linked before it is left
    /// out, as it can't change whether a join is valid.
    hypers: u32,
    /// For each relation of the units being planned, its unit.
    label: SlowVec<u8>,
    /// The edges between the units being planned.
    links: SlowVec<Link>,
    /// For each unit, the units an edge of one relation a side joins it to.
    adjacent: [u64; UNITS_MAX],
    /// Whether an edge between the units has a side of more than one.
    hyper: bool,
    /// The units being planned.
    full: u64,
    /// While counting visits, how many more there may be.
    visits: usize,
    /// The cheapest tree of each connected set of units, by `find`.
    plans: SlowVec<Plan>,
}

/// Plans a graph of three relations or more, as above.
pub fn plan(allocator: &dyn Allocator, graph: &QueryGraph) -> Result<JoinTree, Error> {
    let count = graph.relations.len();
    let mut planner = Planner::new(allocator, graph)?;
    if count <= UNITS_MAX {
        // Each relation is a unit, and the root is the last node.
        let (mut units, mut slots) = ([0; UNITS_MAX], [0; UNITS_MAX]);
        #[expect(clippy::cast_possible_truncation, reason = "at most `UNITS_MAX`")]
        for relation in 0..count as u32 {
            *at_mut!(units, relation as usize) = relation;
            *at_mut!(slots, relation as usize) = count as u32 + relation - 1;
            *at_mut!(planner.label, relation as usize) = relation as u8;
        }
        // The root.
        *at_mut!(slots, 0) = *at!(slots, count - 2) + 1;
        planner.links(NONE, count)?;
        // A graph of n relations has at most 2^n connected sets.
        let small = (count <= SMALL_MAX).then_some(1 << count);
        if let Some(found) = small.or_else(|| planner.count(count, VISITS_MAX)) {
            planner.plans = SlowVec::fixed(allocator, table(found, count))?;
            if planner.solve(at!(units, ..count), at!(slots, ..count - 1), found, false)? {
                return planner.tree(allocator);
            }
        }
    }
    planner.plans = SlowVec::fixed(allocator, table(1 << WINDOW, WINDOW))?;
    if !planner.greedy() {
        return as_written(allocator, graph);
    }
    // Joins come after their children, so each window's subtrees have
    // already been improved.
    #[expect(clippy::cast_possible_truncation, reason = "at most `2 * LIST_MAX`")]
    for root in count as u32..2 * count as u32 - 1 {
        planner.refine(root)?;
    }
    planner.tree(allocator)
}

/// The length of a hash table for `sets` sets and `units` units.
fn table(sets: usize, units: usize) -> usize {
    (2 * (sets + units)).next_power_of_two()
}

/// The first relation of `set`, and whether it has more than one.
#[inline(never)]
fn first(set: &RelationSet) -> (u32, bool) {
    let (mut first, mut count) = (NONE, 0);
    for i in 0..set.words() {
        let word = set.word(i);
        if first == NONE && word != 0 {
            #[expect(clippy::cast_possible_truncation, reason = "at most `LIST_MAX`")]
            let relation = i as u32 * 64 + word.trailing_zeros();
            first = relation;
        }
        count += word.count_ones();
    }
    (first, count > 1)
}

/// Whether `side` has any relation of `set`, and whether all its relations
/// are in `set`.
fn meets(side: &RelationSet, set: &[u64]) -> (bool, bool) {
    let (mut some, mut all) = (false, true);
    for (i, set) in set.iter().enumerate() {
        let side = side.word(i);
        some |= side & set != 0;
        all &= side & !set == 0;
    }
    (some, all)
}

impl<'a> Planner<'a> {
    /// A planner, with no room for plans yet.
    fn new(allocator: &dyn Allocator, graph: &'a QueryGraph) -> Result<Planner<'a>, Error> {
        let count = graph.relations.len();
        let words = count.div_ceil(64);
        #[expect(clippy::cast_possible_truncation, reason = "at most `2 * LIST_MAX`")]
        let nodes = (0..2 * count as u32 - 1).map(|node| {
            #[expect(clippy::cast_precision_loss, reason = "an estimate")]
            let rows = graph.relations.get(node as usize).map_or(0.0, |found| found.rows as f64);
            Node { left: NONE, right: NONE, rows, place: node }
        });
        let estimator = Estimator::new(allocator, graph)?;
        let nodes = SlowVec::fixed_from(allocator, nodes)?;
        let sets = SlowVec::zeroed(allocator, (2 * count - 1) * words)?;
        let edges = SlowVec::fixed(allocator, graph.edges.len())?;
        let label = SlowVec::zeroed(allocator, count)?;
        let links = SlowVec::new(allocator, LIST_MAX)?;
        let plans = SlowVec::fixed(allocator, 0)?;
        let (work, crossing, hypers) = (WORK_MAX, false, NONE);
        let mut planner = Planner {
            graph,
            estimator,
            work,
            crossing,
            words,
            nodes,
            sets,
            edges,
            hypers,
            label,
            links,
            adjacent: [0; UNITS_MAX],
            hyper: false,
            full: 0,
            visits: 0,
            plans,
        };
        for relation in 0..count {
            *at_mut!(planner.sets, relation * words + relation / 64) |= 1 << (relation % 64);
        }
        // Each hyperedge is linked in front of those before it, unless it has
        // the same sides as the last one linked.
        #[expect(clippy::cast_possible_truncation, reason = "at most `LIST_MAX`")]
        for (e, edge) in (0..graph.edges.len() as u32).zip(graph.edges.iter()) {
            let ((left, more_left), (right, more_right)) = (first(&edge.left), first(&edge.right));
            let (first, more, next) = ([left, right], [more_left, more_right], planner.hypers);
            let last = graph.edges.get(planner.hypers as usize);
            let same = |a: &RelationSet, b: &RelationSet| a.first == b.first && *a.rest == *b.rest;
            if (more_left || more_right)
                && !last.is_some_and(|last| {
                    same(&last.left, &edge.left) && same(&last.right, &edge.right)
                })
            {
                planner.hypers = e;
            }
            planner.edges.push(Edge { first, more, next, dead: 0 })?;
        }
        Ok(planner)
    }

    /// Sets the place of each relation of node `node` to `value`.
    #[inline(never)]
    fn mark(&mut self, node: u32, value: u32) {
        for i in 0..self.words {
            let mut word = *at!(self.sets, node as usize * self.words + i);
            while word != 0 {
                at_mut!(self.nodes, i * 64 + word.trailing_zeros() as usize).place = value;
                word &= word - 1;
            }
        }
    }

    /// Side `right` of edge `e`: 0 for its left side, 1 for its right.
    #[inline(never)]
    fn side(&self, e: usize, right: usize) -> &RelationSet {
        let edge = at!(self.graph.edges, e);
        if right == 0 { &edge.left } else { &edge.right }
    }

    /// Whether side `right` of edge `e` has any relation of `set`, and
    /// whether all its relations are in `set`. A side of one relation is
    /// answered from that relation alone.
    fn meets(&self, e: usize, right: usize, set: &[u64]) -> (bool, bool) {
        let edge = at!(self.edges, e);
        if *at!(edge.more, right) {
            return meets(self.side(e, right), set);
        }
        let relation = *at!(edge.first, right) as usize;
        let has = at!(set, relation / 64) >> (relation % 64) & 1 != 0;
        (has, has)
    }

    /// The relations of node `node`.
    #[inline(never)]
    fn set(&self, node: u32) -> &[u64] {
        at!(self.sets, node as usize * self.words..(node as usize + 1) * self.words)
    }

    /// The rows node `node` makes.
    #[inline(never)]
    fn rows(&self, node: u32) -> f64 {
        at!(self.nodes, node as usize).rows
    }

    /// Makes node `slot` a join of nodes `left` and `right`, which makes
    /// `rows` rows.
    #[inline(never)]
    fn join(&mut self, slot: u32, left: u32, right: u32, rows: f64) {
        *at_mut!(self.nodes, slot as usize) = Node { left, right, rows, place: NONE };
        for i in 0..self.words {
            let word = at!(self.set(left), i) | at!(self.set(right), i);
            *at_mut!(self.sets, slot as usize * self.words + i) = word;
        }
    }

    /// The trees that hold all of each side of edge `e`. A side whose
    /// relations are in more than one tree has `NONE`.
    #[inline(never)]
    fn trees(&self, e: usize) -> [u32; 2] {
        let edge = at!(self.edges, e);
        let mut trees = [NONE; 2];
        for (right, tree) in trees.iter_mut().enumerate() {
            let owner = at!(self.nodes, *at!(edge.first, right) as usize).place;
            if self.meets(e, right, self.set(owner)).1 {
                *tree = owner;
            }
        }
        trees
    }

    /// Whether joining trees `a` and `b` would still let every hyperedge
    /// apply. A join that brings together relations of both sides of a
    /// hyperedge must apply it, unless one of the trees already has.
    /// Otherwise the hyperedge could never apply above it.
    fn valid(&mut self, a: u32, b: u32) -> bool {
        let (a, b) = (self.set(a), self.set(b));
        let (mut e, mut work, mut fits) = (self.hypers as usize, self.work, true);
        while fits && let Some(edge) = self.edges.get(e) {
            // The right side first: it's often one relation, in neither tree.
            work = work.saturating_sub(1);
            let ((right_a, right_in_a), (right_b, right_in_b)) =
                (self.meets(e, 1, a), self.meets(e, 1, b));
            if right_a || right_b {
                // Reading a side of more than one relation costs a step a word.
                work = work.saturating_sub(self.words);
                let ((left_a, left_in_a), (left_b, left_in_b)) =
                    (self.meets(e, 0, a), self.meets(e, 0, b));
                let applies = left_in_a && right_in_b || left_in_b && right_in_a;
                fits &= applies || !(left_a || left_b) || left_a && right_a || left_b && right_b;
            }
            e = edge.next as usize;
        }
        self.work = work;
        fits
    }

    /// The work that reading an edge takes: 4 steps, and with hyperedges a
    /// step more for each word of a set.
    fn read_cost(&self) -> usize {
        if self.hypers == NONE { 4 } else { 4 + self.words }
    }

    /// Joins trees until one is left. Each join reads a window of the edges,
    /// starting where the last one stopped, and reads on if it finds no pair
    /// of trees in it. Of the pairs it finds, the `TOP` whose single edges
    /// make the fewest rows are priced by all their edges, and the cheapest
    /// that can be joined is. Returns false if it's stuck in a connected
    /// graph.
    #[inline(never)]
    fn greedy(&mut self) -> bool {
        let (graph, count) = (self.graph, self.edges.len());
        let relations = graph.relations.len();
        // A quarter of the work, spread over the joins, for a read of each
        // edge in the window and the rows of the pairs found.
        let wide = (WORK_MAX / (4 * self.read_cost() * relations)).clamp(1, count);
        let mut at = 0;
        #[expect(clippy::cast_possible_truncation, reason = "at most `2 * LIST_MAX`")]
        for made in relations as u32..2 * relations as u32 - 1 {
            // Once the work is spent, the first pair that can be joined is.
            let window = if self.work == 0 { 1 } else { wide };
            let (mut best, mut a, mut b, mut spanning) = (f64::INFINITY, NONE, NONE, false);
            while a == NONE {
                // The edges whose own rows are fewest, fewest first.
                let mut top = [(f64::INFINITY, 0); TOP];
                let mut read = 0;
                while read < count && (read < window || at!(top, 0).0 == f64::INFINITY) {
                    let (e, edge) = (at, at!(graph.edges, at));
                    (at, read) = (if at + 1 == count { 0 } else { at + 1 }, read + 1);
                    let [left, right] = self.trees(e);
                    if left == right && left != NONE {
                        continue;
                    }
                    spanning = true;
                    if left == NONE || right == NONE || left.max(right) < at!(self.edges, e).dead {
                        continue;
                    }
                    self.estimator.start(self.rows(left), self.rows(right));
                    self.estimator.apply(edge, true);
                    let (rows, mut i) = (self.estimator.finish(), TOP - 1);
                    if rows < at!(top, i).0 {
                        *at_mut!(top, i) = (rows, e);
                    }
                    while i > 0 && at!(top, i).0 < at!(top, i - 1).0 {
                        top.swap(i, i - 1);
                        i -= 1;
                    }
                }
                self.work = self.work.saturating_sub(self.read_cost() * read);
                if at!(top, 0).0 == f64::INFINITY {
                    break;
                }
                // Of their pairs, the one whose edges together make the fewest
                // rows, of those that can be joined.
                for &(rows, e) in &top {
                    if rows == f64::INFINITY {
                        break;
                    }
                    let [left, right] = self.trees(e);
                    if self.hypers != NONE && !self.valid(left, right) {
                        at_mut!(self.edges, e).dead = made;
                        continue;
                    }
                    let rows = self.rows_of(left, right, e, window);
                    if rows < best {
                        (best, a, b) = (rows, left, right);
                    }
                }
            }
            if a == NONE {
                if spanning {
                    return false;
                }
                // The graph isn't connected: join the first relation's tree
                // to another tree.
                self.crossing = true;
                a = at!(self.nodes, 0).place;
                let relations = at!(self.nodes, ..relations).iter();
                b = relations.map(|relation| relation.place).find(|&b| b != a).unwrap_or(a);
                best = self.rows_of(a, b, 0, 0);
            }
            self.join(made, a, b, best);
            self.mark(made, made);
        }
        true
    }

    /// The rows that joining trees `a` and `b` makes, by the edges between
    /// them among the `count` edges from edge `start` on.
    #[inline(never)]
    fn rows_of(&mut self, a: u32, b: u32, start: usize, count: usize) -> f64 {
        let graph = self.graph;
        self.work = self.work.saturating_sub(count);
        self.estimator.start(self.rows(a), self.rows(b));
        let mut e = start;
        for _ in 0..count {
            let [left, right] = self.trees(e);
            if left == a && right == b || left == b && right == a {
                self.estimator.apply(at!(graph.edges, e), left == a);
            }
            e = if e + 1 == graph.edges.len() { 0 } else { e + 1 };
        }
        self.estimator.finish()
    }

    /// Replaces the joins under `root` by the cheapest tree of up to `WINDOW`
    /// of its subtrees, its units. The units are `root`'s children at first.
    /// Then the unit that makes the most rows and is a join is replaced by
    /// its children, until there are `WINDOW`.
    #[inline(never)]
    fn refine(&mut self, root: u32) -> Result<(), Error> {
        if self.work == 0 {
            return Ok(());
        }
        let mut units = [NONE; WINDOW];
        // The joins the units are replaced from, which take the new joins.
        let mut slots = [NONE; WINDOW];
        let top = *at!(self.nodes, root as usize);
        (units[0], units[1], slots[0]) = (top.left, top.right, root);
        let mut k = 2;
        while k < WINDOW {
            // The unit that makes the most rows and is a join, if any.
            let (mut i, mut rows) = (k, -1.0);
            for (unit, at) in at!(units, ..k).iter().zip(0..) {
                let node = at!(self.nodes, *unit as usize);
                if node.right != NONE && node.rows > rows {
                    (i, rows) = (at, node.rows);
                }
            }
            if i == k {
                break;
            }
            let unit = *at!(units, i);
            let node = at!(self.nodes, unit as usize);
            (*at_mut!(slots, k - 1), *at_mut!(units, k), *at_mut!(units, i)) =
                (unit, node.right, node.left);
            k += 1;
        }
        if k < 3 {
            return Ok(());
        }
        #[expect(clippy::cast_possible_truncation, reason = "at most `WINDOW`")]
        for (i, &unit) in at!(units, ..k).iter().enumerate() {
            for w in 0..self.words {
                let mut word = *at!(self.sets, unit as usize * self.words + w);
                while word != 0 {
                    *at_mut!(self.label, w * 64 + word.trailing_zeros() as usize) = i as u8;
                    word &= word - 1;
                }
            }
        }
        self.links(root, k)?;
        self.solve(at!(units, ..k), at!(slots, ..k - 1), 1 << k, true)?;
        Ok(())
    }

    /// Finds the edges between `units` units, which `label` gives, of the
    /// edges whose relations are all under `root`, or of all if `root` is
    /// `NONE`.
    #[inline(never)]
    fn links(&mut self, root: u32, units: usize) -> Result<(), Error> {
        self.work = self.work.saturating_sub(self.read_cost() * self.edges.len());
        self.links.clear();
        (self.adjacent, self.hyper) = ([0; UNITS_MAX], false);
        #[expect(clippy::cast_possible_truncation, reason = "at most `LIST_MAX`")]
        'edges: for edge in 0..self.edges.len() as u32 {
            let mut sides = [0_u64; 2];
            for (right, side) in sides.iter_mut().enumerate() {
                let e = edge as usize;
                if root != NONE && !self.meets(e, right, self.set(root)).1 {
                    continue 'edges;
                }
                *side = if *at!(at!(self.edges, e).more, right) {
                    let side = self.side(e, right);
                    (0..side.words()).fold(0, |units, w| {
                        let mut word = side.word(w);
                        let mut units = units;
                        while word != 0 {
                            let relation = w * 64 + word.trailing_zeros() as usize;
                            units |= 1 << at!(self.label, relation);
                            word &= word - 1;
                        }
                        units
                    })
                } else {
                    1 << at!(self.label, *at!(at!(self.edges, e).first, right) as usize)
                };
            }
            let [left, right] = sides;
            if (left | right).is_power_of_two() {
                continue;
            }
            self.links.push(Link { sides, edge })?;
            if left.is_power_of_two() && right.is_power_of_two() {
                *at_mut!(self.adjacent, left.trailing_zeros() as usize) |= right;
                *at_mut!(self.adjacent, right.trailing_zeros() as usize) |= left;
            } else {
                self.hyper = true;
            }
        }
        self.full = u64::MAX >> (UNITS_MAX - units);
        Ok(())
    }

    /// How many connected sets of the first `units` units, and pairs of them,
    /// `DPhyp` visits, if fewer than `limit`. With hyperedges, it counts some
    /// sets that aren't connected and pairs that can't be joined.
    #[inline(never)]
    fn count(&mut self, units: usize, limit: usize) -> Option<usize> {
        self.visits = limit;
        self.enumerate(units, false);
        (self.visits > 0).then(|| limit - self.visits)
    }

    /// Finds the cheapest tree of each connected set of `units`, of which
    /// there are at most `sets`, and if there's one of them all, makes
    /// it, putting its joins in `slots`. Returns whether it did.
    ///
    /// If `every`, it tries every split of every set rather than `DPhyp`'s
    /// pairs. In a window that's as cheap, and `DPhyp`'s other choice between
    /// trees of the same cost made some plans of the generated graphs four
    /// times as costly.
    #[inline(never)]
    fn solve(
        &mut self,
        units: &[u32],
        slots: &[u32],
        sets: usize,
        every: bool,
    ) -> Result<bool, Error> {
        self.plans.clear();
        self.plans.extend_within(repeat_n(NO_PLAN, table(sets, units.len())));
        for (i, &unit) in units.iter().enumerate() {
            let (set, slot) = (1 << i, self.find(1 << i));
            *at_mut!(self.plans, slot) = Plan { set, cost: 0.0, rows: self.rows(unit), split: 0 };
        }
        if every {
            check!(units.len() <= WINDOW);
            // Each split of a set is tried once: the left side holds its
            // lowest unit. Its sides' trees are done, as they're smaller sets.
            for set in 3..=self.full {
                let low = set.isolate_lowest_one();
                let rest = set ^ low;
                let mut sub = rest;
                while sub != 0 {
                    sub = (sub - 1) & rest;
                    if self.has(sub | low) && self.has(rest ^ sub) {
                        self.pair(sub | low, rest ^ sub);
                    }
                }
            }
        } else {
            // It visits no more than were counted, or than `SMALL_MAX` units have.
            self.enumerate(units.len(), true);
        }
        let full = self.find(self.full);
        if at!(self.plans, full).set != self.full {
            return Ok(false);
        }
        self.build(units, &mut slots.iter().rev(), self.full)?;
        Ok(true)
    }

    /// Joins each pair of connected sets of the first `units` units if
    /// `solve`, or counts them.
    fn enumerate(&mut self, units: usize, solve: bool) {
        for unit in (0..units).rev() {
            let set = 1 << unit;
            self.complements(set, solve);
            self.subgraphs(set, set | (set - 1), solve);
        }
    }

    /// The units an edge joins to all of `set`, outside `set` and `excluded`:
    /// for a side of more than one unit, its lowest.
    #[inline(never)]
    fn neighbors(&mut self, set: u64, excluded: u64) -> u64 {
        self.work = self.work.saturating_sub(self.links.len());
        let outside = self.full & !(set | excluded);
        if self.crossing {
            return outside;
        }
        let (mut near, mut rest) = (0, set);
        while rest != 0 {
            near |= at!(self.adjacent, rest.trailing_zeros() as usize);
            rest &= rest - 1;
        }
        near &= outside;
        for link in self.links.iter().filter(|_| self.hyper) {
            let [left, right] = link.sides;
            if left & !set == 0 && right & !outside == 0 {
                near |= right.isolate_lowest_one();
            }
            if right & !set == 0 && left & !outside == 0 {
                near |= left.isolate_lowest_one();
            }
        }
        near
    }

    /// Joins each connected set that grows from `set` outside `excluded` to
    /// the sets beside it, or counts those sets and pairs.
    fn subgraphs(&mut self, set: u64, excluded: u64, solve: bool) {
        let near = self.neighbors(set, excluded);
        for grow in [false, true] {
            let mut more = 0_u64;
            loop {
                more = more.wrapping_sub(near) & near;
                if more == 0 || !solve && self.visits == 0 {
                    break;
                }
                if !solve && !grow {
                    self.visits -= 1;
                }
                if grow {
                    self.subgraphs(set | more, excluded | near, solve);
                } else if !solve || self.has(set | more) {
                    self.complements(set | more, solve);
                }
            }
        }
    }

    /// Joins the connected set `left` to each connected set beside it above
    /// its lowest unit, so that each pair is joined once, or counts them.
    fn complements(&mut self, left: u64, solve: bool) {
        let excluded = left | (left.isolate_lowest_one() - 1);
        let near = self.neighbors(left, excluded);
        let mut rest = near;
        while rest != 0 {
            let unit = 1 << rest.ilog2();
            rest ^= unit;
            self.grow(left, unit, excluded | near & (unit | (unit - 1)), solve);
        }
    }

    /// Joins `left` to `right` and to each connected set that grows from
    /// `right` outside `excluded`, or counts them.
    fn grow(&mut self, left: u64, right: u64, excluded: u64, solve: bool) {
        if !solve {
            if self.visits == 0 {
                return;
            }
            self.visits -= 1;
        } else if self.has(right) {
            self.pair(left, right);
        }
        let near = self.neighbors(right, excluded);
        let mut more = 0_u64;
        loop {
            more = more.wrapping_sub(near) & near;
            if more == 0 {
                break;
            }
            self.grow(left, right | more, excluded | near, solve);
        }
    }

    /// The slot of `set` in `plans`, or the empty slot it would take.
    #[inline(never)]
    fn find(&self, set: u64) -> usize {
        let mask = self.plans.len() - 1;
        let mut i = (set.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & mask;
        for _ in 0..=mask {
            let found = at!(self.plans, i).set;
            if found == 0 || found == set {
                return i;
            }
            i = (i + 1) & mask;
        }
        // The table has room for every set it's given.
        failed!()
    }

    /// Whether `set` has a tree.
    fn has(&self, set: u64) -> bool {
        at!(self.plans, self.find(set)).set == set
    }

    /// Makes the join of units `left` and `right` the cheapest tree of both,
    /// if it can be joined and is cheaper.
    #[inline(never)]
    fn pair(&mut self, left: u64, right: u64) {
        let (l, r) = (*at!(self.plans, self.find(left)), *at!(self.plans, self.find(right)));
        let set = left | right;
        let i = self.find(set);
        let found = *at!(self.plans, i);
        let fresh = found.set != set;
        if l.set != left || r.set != right || !self.joins(set, left, right, [l.rows, r.rows], fresh)
        {
            return;
        }
        let rows = if fresh { self.estimator.finish() } else { found.rows };
        let cost = l.cost + r.cost + rows.max(1.0);
        if fresh || cost < found.cost {
            *at_mut!(self.plans, i) = Plan { set, cost, rows, split: left };
        }
    }

    /// Whether units `left` and `right`, which together are `set`, can be
    /// joined: an edge joins them, and every edge whose relations they bring
    /// together applies. If `estimate`, it estimates the join's rows from
    /// those of each side.
    fn joins(&mut self, set: u64, left: u64, right: u64, rows: [f64; 2], estimate: bool) -> bool {
        self.work = self.work.saturating_sub(self.links.len());
        if estimate {
            self.estimator.start(rows[0], rows[1]);
        }
        let mut any = self.crossing;
        for link in self.links.iter() {
            let [l, r] = link.sides;
            let both = l | r;
            if both & !set != 0 || both & !left == 0 || both & !right == 0 {
                continue;
            }
            let forward = l & !left == 0 && r & !right == 0;
            if !(forward || l & !right == 0 && r & !left == 0) {
                return false;
            }
            if estimate {
                self.estimator.apply(at!(self.graph.edges, link.edge as usize), forward);
            }
            any = true;
        }
        any
    }

    /// Makes the cheapest tree of the units in `set`, putting its joins in
    /// `slots`, and returns its root.
    #[inline(never)]
    fn build(
        &mut self,
        units: &[u32],
        slots: &mut core::iter::Rev<core::slice::Iter<'_, u32>>,
        set: u64,
    ) -> Result<u32, Error> {
        if set.is_power_of_two() {
            return Ok(*at!(units, set.trailing_zeros() as usize));
        }
        let plan = *at!(self.plans, self.find(set));
        let left = self.build(units, slots, plan.split)?;
        let right = self.build(units, slots, set ^ plan.split)?;
        let Some(&slot) = slots.next() else { failed!() };
        self.join(slot, left, right, plan.rows);
        Ok(slot)
    }

    /// The plan as a `JoinTree`, whose nodes are the planner's, in the same
    /// order.
    fn tree(&self, allocator: &dyn Allocator) -> Result<JoinTree, Error> {
        let mut tree = JoinTree::new(allocator, self.graph.relations.len())?;
        // The graph's relations are its first nodes.
        #[expect(clippy::cast_possible_truncation, reason = "at most `2 * LIST_MAX`")]
        for (index, node) in (0..self.nodes.len() as u32).zip(self.nodes.iter()) {
            match node.right {
                NONE => tree.relation(index)?,
                right => tree.join(node.left, right)?,
            };
        }
        Ok(tree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::Heap;
    use crate::join_order::{EquivalenceSet, JoinEdge, JoinKind, Relation, cost, enumerate};

    /// A cycle of `count` relations, with a left join to the last, which
    /// needs all the others first.
    fn cycle(count: u32) -> QueryGraph {
        let mut graph = QueryGraph::new(&Heap).unwrap();
        let set = |relations: &mut dyn Iterator<Item = u32>| {
            let mut set = RelationSet::new(&Heap).unwrap();
            relations.for_each(|relation| set.insert(relation).unwrap());
            set
        };
        for relation in 0..count {
            let rows = 10 + u64::from(relation * 7919 % 1000);
            graph.relations.push(Relation { rows, unfiltered_rows: rows }).unwrap();
            graph.equivalences.push(EquivalenceSet { domain: rows }).unwrap();
            let (left, right) = match relation {
                0 => continue,
                last if last + 1 == count => (set(&mut (0..last)), set(&mut [last].into_iter())),
                _ => (set(&mut [relation - 1].into_iter()), set(&mut [relation].into_iter())),
            };
            let kind = if relation + 1 == count { JoinKind::Left } else { JoinKind::Inner };
            graph.edges.push(JoinEdge { left, right, kind, equivalence: relation }).unwrap();
        }
        let (left, right) = (set(&mut [0].into_iter()), set(&mut [count - 2].into_iter()));
        graph.edges.push(JoinEdge { left, right, kind: JoinKind::Inner, equivalence: 0 }).unwrap();
        graph
    }

    #[test]
    fn plans_are_valid_and_no_worse_than_written() {
        for count in [3, 10, 13, 70] {
            let graph = cycle(count);
            let planned = cost(&Heap, &graph, &enumerate(&Heap, &graph).unwrap()).unwrap();
            let written = cost(&Heap, &graph, &as_written(&Heap, &graph).unwrap()).unwrap();
            assert!(planned.valid.is_ok() && planned.cost <= written.cost);
        }
    }

    /// A chain of 50 relations, each left joined to all those before it.
    /// It's planned exactly, as it has few connected sets, but has too many
    /// relations for each set of them to be read.
    #[test]
    fn long_graphs_with_few_connected_sets_are_planned_exactly() {
        let mut graph = QueryGraph::new(&Heap).unwrap();
        for relation in 0..50_u32 {
            let rows = 10 + u64::from(relation * 7919 % 1000);
            graph.relations.push(Relation { rows, unfiltered_rows: rows }).unwrap();
            graph.equivalences.push(EquivalenceSet { domain: rows }).unwrap();
            let (mut left, mut right) =
                (RelationSet::new(&Heap).unwrap(), RelationSet::new(&Heap).unwrap());
            (0..relation).for_each(|before| left.insert(before).unwrap());
            right.insert(relation).unwrap();
            if relation > 0 {
                let (kind, equivalence) = (JoinKind::Left, relation);
                graph.edges.push(JoinEdge { left, right, kind, equivalence }).unwrap();
            }
        }
        let planned = cost(&Heap, &graph, &enumerate(&Heap, &graph).unwrap()).unwrap();
        assert!(planned.valid.is_ok());
    }

    #[test]
    fn one_or_two_relations_need_no_planning() {
        let mut graph = QueryGraph::new(&Heap).unwrap();
        graph.relations.push(Relation { rows: 10, unfiltered_rows: 10 }).unwrap();
        assert_eq!(enumerate(&Heap, &graph).unwrap().nodes.len(), 1);
        graph.relations.push(Relation { rows: 20, unfiltered_rows: 20 }).unwrap();
        let tree = enumerate(&Heap, &graph).unwrap();
        assert_eq!(cost(&Heap, &graph, &tree).unwrap().valid, Ok(()));
    }

    #[test]
    fn disconnected_graphs_take_cross_products() {
        for count in [5, 70] {
            let mut graph = cycle(count);
            graph.relations.push(Relation { rows: 3, unfiltered_rows: 3 }).unwrap();
            let planned = cost(&Heap, &graph, &enumerate(&Heap, &graph).unwrap()).unwrap();
            assert!(planned.valid.is_ok());
        }
    }
}
