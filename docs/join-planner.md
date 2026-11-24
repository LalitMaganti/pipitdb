# Join planner

The join planner chooses the order a query joins its relations in, and how
each join runs. It plans every graph optimally when exact search is cheap,
and otherwise falls back to greedy ordering improved by exact windows. One
work budget bounds all of it, so planning always ends with a valid plan.

## Stages

1. **Region.** The optimizer finds each maximal tree of joins in the plan.
   Its leaves, scans or subtrees the planner can't look into, are the
   *relations*.
2. **Graph.** Each relation has its estimated rows. Each join condition is
   an *edge* between the relations its condition reads. A non-inner join's
   edge also includes on its left the relations it must wait for, its
   total eligibility set, which makes it a *hyperedge*. Columns joined by
   equality form an *equivalence set*, with a *domain*: how many distinct
   values they can take.
3. **Estimate.** A join's rows follow from its inputs' rows and the domains
   of the equivalence sets it joins on (see Estimation). A tree's cost is
   C_out, the rows of all its joins added up, plus 10 for each row of
   each join's smaller side, the hash table it builds (see Cost).
4. **Valid join.** Two trees may be joined only if an edge connects them,
   unless the graph is disconnected. A join must apply every edge whose
   relations it brings together, so each side of the edge must be on one
   side of the join.
5. **Trivial.** One or two relations, or no better plan found: return none,
   so the optimizer keeps the plan it has.
6. **Count.** Count the connected sets of relations, and the pairs of them,
   that exact search would visit, stopping at the exact budget. Graphs too
   small to exceed it skip the count.
7. **Exact.** If the count fits, plan with DPhyp: join each connected set
   to each connected set beside it, smallest first, keeping the cheapest
   valid tree of each set. The cheapest tree of all relations is the
   optimum.
8. **Greedy.** Otherwise, start with one tree per relation and repeatedly
   join the valid pair whose join makes the fewest rows, until one tree is
   left.
9. **Windows.** If they prove to pay (see Open questions): bottom up, take
   up to 8 subtrees of the greedy plan under each join, and replace the
   joins between them with the cheapest valid tree of those subtrees.
10. **Rebuild.** The optimizer replaces the region with the planned tree,
    choosing each join's build side.

## Integration with the optimizer

The planner runs as one optimizer pass, after filters are placed and
conditions pushed down, so relations' rows reflect their filters, and
before forms are chosen.

- **Before the planner.** Next: a pass that simplifies outer joins. A
  WHERE condition on a left join's right side that is never true for null
  makes it an inner join (Galindo-Legaria and Rosenthal 1997). It will
  also move an ON condition on one side into that side where the join
  kind allows it.
- **Region boundaries.** Any op other than a join ends a region: an
  aggregate, a limit, a sort, a projection that computes a join key. A
  filter on columns of both sides of a join can't move below it, so for
  now it ends the region too. So, for now, does a join that isn't inner,
  or has a condition that isn't an equality of two columns.
- **Relation rows** come from the row estimate of the relation's subtree.
- **Domains** come from what the estimate knows of the join columns. A
  column known to be unique in its relation, such as an id, has a domain
  of that relation's unfiltered rows. Without a distinct count, see
  Estimation.
- **Executable joins.** The join operator can run any join the planner
  makes, so the optimizer passes no constraint on stage 4. The hook stays
  for an operator that can't.
- **Rebuild.** The planned tree takes the place of the region's joins,
  each join taking the equalities whose two relations it brings together,
  and pruning then orders the plan's nodes inputs first again.
- **Build side.** Each join chooses its own after planning: a side whose
  key is an id, which is never hashed; otherwise the side whose hash table
  is smaller, its estimated rows times a word for each column it outputs
  and one for its entry; of sides of one size, one whose key is unique.
  Building on a unique key regardless of size can hash a big side to probe
  a small one, and building costs several times what probing does. DuckDB
  and DataFusion also choose by estimated bytes.

## Estimation

Every rule below follows from one assumption, *containment*: of two
joined columns, the values of the one with fewer distinct values all
appear in the other. Each rule names its source.

| Join | Rows | Source |
|---|---|---|
| Inner | left × right ÷ domain, once per equivalence set | System R (Selinger et al. 1979), by containment |
| Left | max(inner rows, left rows) | A left join keeps every left row. PostgreSQL estimates the same |
| Semi | left × match, where match = min(1, right's distinct values ÷ domain) | Containment. PostgreSQL's semi join estimate has the same form |
| Anti | left × (1 − match) | The rows a semi join drops, so semi + anti = left |

- **Domains without distinct counts.** A unique column's domain is exact.
  A column whose least and greatest values are known, as Parquet's
  statistics give them, can take at most the values between them, and at
  most its relation's rows before filters: DuckDB bounds it the same way.
  Otherwise, until metadata gives distinct counts, the domain is taken as
  the rows of the smaller relation, which assumes the smaller side holds a
  key. This is an assumption to replace with metadata, not a rule.
- **Filters.** A column of known bounds compared with an integer keeps the
  share of its bounds the comparison covers, as if its values were spread
  evenly, and the comparison narrows the bounds, so `a >= x AND a < y`
  keeps the share from `x` to `y`. DataFusion estimates ranges the same
  way. Otherwise, equality on a non-unique column keeps rows ÷ (2 log₂
  rows), Perfetto's heuristic. It has no grounding and is replaced by
  distinct counts once metadata provides them. Both are documented
  divergences.
- **Composite keys.** Equalities between the same two relations, such as
  `l_partkey = ps_partkey AND l_suppkey = ps_suppkey`, are one key, whose
  columns aren't independent. In place of the product of their domains,
  the join divides once by the smaller relation's rows before filters,
  which assumes that side holds the key. DuckDB's `ApplyJoinPairCap` does
  the same, and like it, the cap applies only within 8 times the largest
  of the domains, so a join of two broad tables keeps the product.
- **Cross products.** left × right, with no division.

## Budgets

Budgets trade planning time against plan quality, so no theorem fixes
them. Each is set from a planning-time target and measured.

| Budget | Value | Target |
|---|---|---|
| Exact visits | 2,048 | Exact search stays within ~10 µs of planning per 1,000 visits; to be measured |
| Always exact | up to 10 relations | A clique of 10 has fewer visits than the exact budget allows |
| Work | 2^27 steps | The slowest plan, of 1,000 relations, takes about 1 s; a soft target |
| Window | 8 subtrees | Measured against windows of 9, 10 and 12 |
| Greedy candidates | 16 pairs | Pairs whose rows are fewer together than by any one edge |

## Bounds

- Every loop draws on the work budget, or is bounded by the exact count or
  a fixed window size.
- Once the budget is spent, greedy joins the first valid pair it finds and
  windows stop, so a connected graph always gets a valid plan.

## Coverage

Every function is assumed deterministic, with no side effects, so the
planner may move its calls, reorder them, or run them any number of times.
What it needs of a function is its cost per call and its selectivity.

| Query shape | Status |
|---|---|
| Inner equi-joins | Planned |
| Joins on several columns | Planned: one edge per condition, each equivalence set divided once, and those between the same two relations capped as a composite key |
| Left, semi and anti joins | Planned, conservatively: some legal reorderings aren't tried. The optimizer passes only inner joins so far |
| The same table twice | Planned: each use is its own relation |
| Joins the operator can't run | None so far. Respected, through the executable-join hook |
| Cross products | Planned in disconnected graphs. A relation of at most one row, by its row bound, may be cross-joined anywhere, and greedy takes the cheapest cross product. Next: any two trees may be cross-joined, within the exact budget, and greedy prices cross products with the other pairs |
| Conditions other than equality, including functions of both sides | Next: an edge with a selectivity in place of a domain, and a cost per pair compared. It runs as a cross product filtered batch by batch. On a join that also has an equality, the equality drives it and the rest are checked only on its matches |
| Filters calling expensive functions | Next: placed by rank, (selectivity − 1) ÷ cost per row, so an expensive, unselective filter may sit above a join |
| Table-valued functions | Next: a relation with declared rows, and a hyperedge to the relations its arguments read |
| Right joins | Issue: a left join with its sides swapped |
| Full outer joins | Issue: no join kind yet |
| Outer-join reordering rules | Issue: Moerkotte, Fender and Eich's conflict rules |

- **Cost.** C_out counts the rows each join makes, but not that building
  a hash table costs more than probing one: 8 to 15 times a row, on Intel,
  AMD and Apple CPUs alike, as `crates/bench/build_cost.py` measures.
  PostgreSQL charges about 5 times; DuckDB's thesis on join ordering
  (Ebergen 2022, §6.3) found plans with fewer rows built ran faster despite
  more rows made. So each join also costs 10 for each row of its smaller
  side. TPC-H's and TPC-DS's plans are the same for any factor from 1 to
  64, so the factor needn't match the machine. A join that compares every
  pair also costs left × right comparisons, each at its function's cost
  per call.
- **A function's facts.** Its cost per call and its selectivity, or for a
  table-valued function its rows. An undeclared function is taken as
  expensive and keeping every row, the bound that always holds, so the
  planner never moves it to where it runs more often by mistake.

## Open questions

- Do windows pay once exact search covers the small graphs? Measure greedy
  alone against greedy with windows on the graphs exact search can't plan.
  If they don't clearly improve plans, leave them out.
- Should the budgets scale with the query's estimated cost, so planning a
  cheap query stays cheap?
- What should end a region beyond non-join ops, for example a join the
  operator can't run yet?

## Sources

- Selinger et al. 1979, "Access Path Selection in a Relational Database
  Management System".
- Moerkotte and Neumann 2008, "Dynamic Programming Strikes Back" (DPhyp).
- Moerkotte, Fender and Eich 2013, "On the Correct and Complete Enumeration
  of the Core Search Space" (non-inner joins).
- Fegaras 1998, "A New Heuristic for Optimizing Large Queries" (greedy
  operator ordering).
- Neumann and Radke 2018, "Adaptive Optimization of Very Large Join
  Queries".
- Galindo-Legaria and Rosenthal 1997, "Outerjoin Simplification and
  Reordering for Query Optimization".
- Hellerstein and Stonebraker 1993, "Predicate Migration: Optimizing
  Queries with Expensive Predicates".
- Ebergen 2022, "Join Order Optimization with (Almost) No Statistics"
  (MSc thesis, CWI).
- Leis et al. 2015, "How Good Are Query Optimizers, Really?".
- PostgreSQL, `calc_joinrel_size_estimate` and `initial_cost_hashjoin` in
  `costsize.c`.
