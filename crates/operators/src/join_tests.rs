//! Nested-loop joins planned and run as a frontend's would be, over
//! in-memory tables, checked against each pair of rows tested by hand.

extern crate std;

use pipit_kernel::bitmap::BitSlice;
use std::format;
use std::string::{String, ToString};
use std::vec;
use std::vec::Vec;

use crate::table::Table;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::expression::{Expr, ExprBuilder, ExprOp, Exprs};
use pipit_kernel::global::Global;
use pipit_kernel::join::{JoinKind, JoinOp};
use pipit_kernel::lower::lower;
use pipit_kernel::optimize::optimize;
use pipit_kernel::output_properties::plan_nodes;
use pipit_kernel::plan::{
    DynOp, FilterOp, LogicalPlan, NamedColumn, PlanNodeId, ScanColumn, ScanOp, conjuncts_of,
};
use pipit_kernel::row_batch::{BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::DynScannable;
use pipit_kernel::slow_vec::SlowVec;

fn int64s(values: &[Option<i64>]) -> ColumnView {
    let mut buffer = Buffer::allocate(Global::current(), values.len() * 8).unwrap();
    for (slot, value) in buffer.as_mut_slice::<i64>().iter_mut().zip(values) {
        *slot = value.unwrap_or(0);
    }
    // A bit a row, set where it isn't null.
    let validity = values.iter().any(Option::is_none).then(|| {
        let mut bits =
            Buffer::allocate(Global::current(), BitSlice::<u8>::words_for(values.len())).unwrap();
        let bytes = bits.as_mut_slice::<u8>();
        bytes.fill(0);
        for (row, value) in values.iter().enumerate() {
            BitSlice::of_mut(&mut *bytes).set_if(row, value.is_some());
        }
        bits
    });
    ColumnView::new(&mut Context::new(Global::current(), None), DataType::Int64, buffer, validity)
        .unwrap()
}

/// A table of `k`, as given, and `v`, its row number, in batches.
fn table(keys: &[Option<i64>]) -> DynScannable<'static> {
    let groups: Vec<[ColumnView; 2]> = keys
        .chunks(BATCH_ROWS_MAX as usize)
        .zip((0..).step_by(BATCH_ROWS_MAX as usize))
        .map(|(keys, first)| {
            let rows: Vec<Option<i64>> = (first..).take(keys.len()).map(Some).collect();
            [int64s(keys), int64s(&rows)]
        })
        .collect();
    let groups: Vec<&[ColumnView]> = groups.iter().map(|group| &group[..]).collect();
    let schema = [("k", DataType::Int64), ("v", DataType::Int64)];
    DynScannable::new(Global::current(), Table::new(Global::current(), &schema, &groups).unwrap())
        .unwrap()
}

/// A table of `k`, stored as a sequence from each group's first for its
/// rows, and `v`, its row number; and its keys.
fn sequences(groups: &[(i64, u32)]) -> (DynScannable<'static>, Vec<Option<i64>>) {
    let mut keys = Vec::new();
    let mut row = 0;
    let groups: Vec<[ColumnView; 2]> = groups
        .iter()
        .map(|&(first, rows)| {
            keys.extend((first..).take(rows as usize).map(Some));
            let numbers: Vec<Option<i64>> = (row..).take(rows as usize).map(Some).collect();
            row += i64::from(rows);
            [
                ColumnView::sequence(&mut Context::new(Global::current(), None), first, rows)
                    .unwrap(),
                int64s(&numbers),
            ]
        })
        .collect();
    let groups: Vec<&[ColumnView]> = groups.iter().map(|group| &group[..]).collect();
    let schema = [("k", DataType::Int64), ("v", DataType::Int64)];
    (
        DynScannable::new(
            Global::current(),
            Table::new(Global::current(), &schema, &groups).unwrap(),
        )
        .unwrap(),
        keys,
    )
}

/// A scan of `k` and `v` of `table`, as new columns of `plan`.
fn scan<'c>(
    plan: &mut LogicalPlan<'c>,
    table: &'c DynScannable<'c>,
) -> (PlanNodeId, [NamedColumn; 2]) {
    let mut reads = SlowVec::new(Global::current(), 2).unwrap();
    let mut bound = Vec::new();
    for (i, name) in (0..).zip(["k", "v"]) {
        let binding = plan.add_column(name, DataType::Int64, true).unwrap();
        reads.push(ScanColumn { column: i, binding, forms: Forms::FLAT }).unwrap();
        bound.push(binding);
    }
    let op = DynOp::new(Global::current(), ScanOp::new(table, reads)).unwrap();
    let node = plan.add_node(op, &[]).unwrap();
    (node, [bound[0], bound[1]])
}

/// A join's condition, of the probe and build sides' `k` and `v`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum On {
    /// `k = k`, which the join hashes by.
    Equal,
    /// `k < k`, which it can't.
    Less,
    /// `k = k AND v < v`, which it hashes by `k`, then tests.
    EqualAndLess,
    /// None: a cross join.
    Always,
}

/// The condition `on`, of the probe side's `k` and `v` and the build side's.
fn condition(exprs: &mut Exprs, on: On, [pk, pv, bk, bv]: [u32; 4]) -> Option<Expr> {
    let mut nodes = ExprBuilder::new(Global::current()).unwrap();
    let mut compare = |op, a, b| {
        let (a, b) = (nodes.column(a).unwrap(), nodes.column(b).unwrap());
        nodes.node(op, a, b).unwrap()
    };
    match on {
        On::Equal => _ = compare(ExprOp::Eq, pk, bk),
        On::Less => _ = compare(ExprOp::Lt, pk, bk),
        On::EqualAndLess => {
            let (equal, less) = (compare(ExprOp::Eq, pk, bk), compare(ExprOp::Lt, pv, bv));
            nodes.node(ExprOp::And, equal, less).unwrap();
        }
        On::Always => return None,
    }
    Some(exprs.add(nodes.nodes()).unwrap())
}

/// A filter above a join, on the build side's `v`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Above {
    Nothing,
    IsNull,
    Positive,
}

/// Each output row's probe `v`, and build `v` if the kind outputs it, of
/// `probe <kind> JOIN build ON <on>`, sorted, under filter `above`; and the optimized plan's
/// shape, each node's number of children from the root down.
fn join(
    probe: &DynScannable<'_>,
    build: &DynScannable<'_>,
    kind: JoinKind,
    on: On,
    above: Above,
) -> (Vec<(i64, Option<i64>)>, Vec<usize>) {
    let mut plan = LogicalPlan::new(Global::current()).unwrap();
    let (probe, [probe_k, probe_v]) = scan(&mut plan, probe);
    let (build, [build_k, build_v]) = scan(&mut plan, build);
    let columns = [probe_k.id, probe_v.id, build_k.id, build_v.id];
    let condition = condition(&mut plan.exprs, on, columns);
    let build_columns =
        SlowVec::fixed_from(Global::current(), [build_k.id, build_v.id].into_iter()).unwrap();
    let probe_columns =
        SlowVec::fixed_from(Global::current(), [probe_k.id, probe_v.id].into_iter()).unwrap();
    let conjuncts = match condition {
        Some(condition) => conjuncts_of(Global::current(), &plan.exprs, condition).unwrap(),
        None => SlowVec::fixed(Global::current(), 0).unwrap(),
    };
    let join = JoinOp::new(kind, conjuncts, [probe_columns, build_columns]);
    let children = [probe, build];
    let join = plan.add_node(DynOp::new(Global::current(), join).unwrap(), &children).unwrap();
    if above != Above::Nothing {
        let mut nodes = ExprBuilder::new(Global::current()).unwrap();
        let column = nodes.column(build_v.id).unwrap();
        if above == Above::IsNull {
            nodes.node(ExprOp::IsNull, column, 0).unwrap();
        } else {
            let zero = nodes.int(0).unwrap();
            nodes.node(ExprOp::Gt, column, zero).unwrap();
        }
        let condition = plan.exprs.add(nodes.nodes()).unwrap();
        let filter = FilterOp::new(Global::current(), &plan.exprs, condition).unwrap();
        let filter = DynOp::new(Global::current(), filter).unwrap();
        let children = [join];
        plan.add_node(filter, &children).unwrap();
    }
    plan.output.push(probe_v).unwrap();
    plan.output.push(build_v).unwrap();
    optimize(Global::current(), &mut plan).unwrap();
    let mut shape = Vec::new();
    let mut pending = vec![plan.root];
    while let Some(node) = pending.pop() {
        let children = &plan.nodes[node as usize].children;
        shape.push(children.len());
        pending.extend(children.iter().rev());
    }
    let planned = plan_nodes(Global::current(), &plan).unwrap();
    let physical = lower(Global::current(), &mut plan, &planned).unwrap();
    let query = Global::current();
    let mut execution = physical.start(query).unwrap();
    let mut batch = RowBatch::new();
    let mut rows = Vec::new();
    let positions: Vec<u32> = physical.columns().iter().map(|column| column.position).collect();
    while execution.next(&mut batch).unwrap() {
        let selection = batch.selection();
        for row in (0..selection.len()).map(|i| selection.kept().row(i)) {
            let probe = batch.column(positions[0]).int64s()[row as usize];
            let build = positions
                .get(1)
                .map(|&p| batch.column(p))
                .and_then(|column| (!column.is_null(row)).then(|| column.int64s()[row as usize]));
            rows.push((probe, build));
        }
    }
    rows.sort_unstable();
    (rows, shape)
}

/// What `join` should give, by testing every pair.
fn expected(
    probe: &[Option<i64>],
    build: &[Option<i64>],
    kind: JoinKind,
    on: On,
    above: Above,
) -> Vec<(i64, Option<i64>)> {
    // A probe row's `v` is its number, as is a build row's.
    let holds = |(p, a): (i64, Option<i64>), (b, k): (i64, Option<i64>)| match (on, a, k) {
        (On::Always, _, _) => true,
        (_, None, _) | (_, _, None) => false,
        (On::Equal, Some(a), Some(k)) => a == k,
        (On::Less, Some(a), Some(k)) => a < k,
        (On::EqualAndLess, Some(a), Some(k)) => a == k && p < b,
    };
    let mut rows = Vec::new();
    for (p, &a) in (0..).zip(probe) {
        let matches: Vec<i64> =
            (0..).zip(build).filter(|&(b, &k)| holds((p, a), (b, k))).map(|(b, _)| b).collect();
        rows.extend(matches.iter().map(|&b| (p, Some(b))));
        if matches.is_empty() && kind == JoinKind::Left {
            rows.push((p, None));
        }
    }
    match above {
        Above::Nothing => {}
        Above::IsNull => rows.retain(|&(_, b)| b.is_none()),
        Above::Positive => rows.retain(|&(_, b)| b.is_some_and(|b| b > 0)),
    }
    rows.sort_unstable();
    rows
}

const KINDS: [JoinKind; 2] = [JoinKind::Inner, JoinKind::Left];

#[test]
fn joins_every_kind_on_any_condition() {
    // Probe rows over two batches, so a batch's matches come a build row at
    // a time, and nulls, which match nothing.
    let probe_keys: Vec<Option<i64>> = (0..2100).map(|i| (i % 7 != 3).then_some(i % 5)).collect();
    let build_keys = [Some(1), Some(3), None, Some(3), Some(9)];
    let (probe, build) = (table(&probe_keys), table(&build_keys));
    for kind in KINDS {
        for on in [On::Equal, On::Less, On::EqualAndLess, On::Always] {
            assert_eq!(
                join(&probe, &build, kind, on, Above::Nothing).0,
                expected(&probe_keys, &build_keys, kind, on, Above::Nothing),
                "{kind:?} {on:?}"
            );
        }
    }
}

#[test]
fn hashes_a_build_side_of_many_batches() {
    // Keys repeat within and across the build side's batches, so a probe
    // row's build rows of its keys come over several rounds.
    let probe_keys: Vec<Option<i64>> =
        (0..2100).map(|i| (i % 11 != 0).then_some(i % 900)).collect();
    let build_keys: Vec<Option<i64>> =
        (0..3000).map(|i| (i % 13 != 0).then_some(i % 700)).collect();
    let (probe, build) = (table(&probe_keys), table(&build_keys));
    for kind in KINDS {
        for on in [On::Equal, On::EqualAndLess] {
            assert_eq!(
                join(&probe, &build, kind, on, Above::Nothing).0,
                expected(&probe_keys, &build_keys, kind, on, Above::Nothing),
                "{kind:?} {on:?}"
            );
        }
    }
}

#[test]
fn joins_to_an_empty_build_side() {
    let probe_keys = [Some(1), None];
    let probe = table(&probe_keys);
    for kind in KINDS {
        let op = On::Equal;
        assert_eq!(
            join(&probe, &table(&[]), kind, op, Above::Nothing).0,
            expected(&probe_keys, &[], kind, op, Above::Nothing)
        );
    }
}

#[test]
fn keeps_a_null_test_on_the_build_side_above_a_left_join() {
    // `build.v IS NULL` keeps only the probe rows nothing matched: moved to
    // the build side, it would drop every build row first.
    let probe_keys = [Some(1), Some(2), Some(3)];
    let build_keys = [Some(1), Some(3)];
    let (probe, build) = (table(&probe_keys), table(&build_keys));
    let op = On::Equal;
    let (rows, _) = join(&probe, &build, JoinKind::Left, op, Above::IsNull);
    assert_eq!(rows, [(1, None)]);
    assert_eq!(rows, expected(&probe_keys, &build_keys, JoinKind::Left, op, Above::IsNull));
}

#[test]
fn finds_build_rows_by_their_keys() {
    let probe_keys: Vec<Option<i64>> =
        [Some(5), Some(-1), Some(2999), None, Some(3000), Some(2048), Some(5), Some(2), Some(9)]
            .into_iter()
            .chain((0..3000).map(|i| Some(i * 7 % 3100)))
            .collect();
    let probe = table(&probe_keys);
    // Ids across batches, each following the last, as a table's are, found
    // with no bitmap; with a gap, and in another order, found by a bitmap;
    // and repeated, hashed.
    let groups: [&[(i64, u32)]; 4] =
        [&[(0, 2048), (2048, 952)], &[(0, 4), (5, 4)], &[(10, 4), (0, 4)], &[(0, 4), (2, 4)]];
    for groups in groups {
        let (build, build_keys) = sequences(groups);
        for kind in KINDS {
            for on in [On::Equal, On::EqualAndLess] {
                assert_eq!(
                    join(&probe, &build, kind, on, Above::Nothing).0,
                    expected(&probe_keys, &build_keys, kind, on, Above::Nothing),
                    "{groups:?} {kind:?} {on:?}"
                );
            }
        }
    }
}

#[test]
fn filters_above_a_left_join() {
    // A filter on the build side's `v` can't move below a left join: it
    // would leave probe rows with no build rows, which get nulls.
    let probe_keys = [Some(1), Some(2), Some(3)];
    let build_keys = [Some(1), Some(3), Some(3)];
    let (probe, build) = (table(&probe_keys), table(&build_keys));
    for above in [Above::Positive, Above::IsNull] {
        let (rows, _) = join(&probe, &build, JoinKind::Left, On::Equal, above);
        assert_eq!(rows, expected(&probe_keys, &build_keys, JoinKind::Left, On::Equal, above));
    }
}

#[test]
fn finds_no_build_row_for_a_null_among_ids() {
    // The build side's keys are its rows' numbers but for a null, so a key
    // finds its row by its value, but the null row is no key's.
    let ids: Vec<Option<i64>> = (0..40).map(|i| (i != 5).then_some(i)).collect();
    let probe_keys: Vec<Option<i64>> = (0..50).map(Some).collect();
    let (probe, build) = (table(&probe_keys), table(&ids));
    for kind in KINDS {
        assert_eq!(
            join(&probe, &build, kind, On::Equal, Above::Nothing).0,
            expected(&probe_keys, &ids, kind, On::Equal, Above::Nothing),
            "{kind:?}"
        );
    }
}

#[test]
fn builds_the_smaller_side_of_a_left_join() {
    // The first side is much smaller, so is built, and its rows nothing
    // matched come out at the end. Its keys are its rows' numbers but for a
    // null, which no probe row's number may find.
    let small: Vec<Option<i64>> = (0..40).map(|i| (i != 5).then_some(i)).collect();
    let large: Vec<Option<i64>> = (0..2100).map(|i| (i % 7 != 3).then_some(i % 45)).collect();
    let (left, right) = (table(&small), table(&large));
    for kind in KINDS {
        for on in [On::Equal, On::Less, On::EqualAndLess, On::Always] {
            for above in [Above::Nothing, Above::IsNull, Above::Positive] {
                assert_eq!(
                    join(&left, &right, kind, on, above).0,
                    expected(&small, &large, kind, on, above),
                    "{kind:?} {on:?}"
                );
            }
        }
    }
    // Nothing on the second side, or on the first.
    for (first, second) in [(&small[..], &[][..]), (&[][..], &large[..])] {
        assert_eq!(
            join(&table(first), &table(second), JoinKind::Left, On::Equal, Above::Nothing).0,
            expected(first, second, JoinKind::Left, On::Equal, Above::Nothing)
        );
    }
}

/// Joins `tables` in the written order `order`: each table after the first
/// to those before it, on `a.v = b.k` for each edge `(a, b)` between them.
/// Returns each output row's `v`s, by table, sorted, and the optimized plan's
/// joins as a nested list of tables, each join's in order.
fn joined_in_order(
    tables: &[DynScannable<'_>],
    order: &[usize],
    edges: &[(usize, usize)],
) -> (Vec<Vec<i64>>, String) {
    let mut plan = LogicalPlan::new(Global::current()).unwrap();
    let mut columns = vec![None; tables.len()];
    let mut root = None;
    for &table in order {
        let (node, [k, v]) = scan(&mut plan, &tables[table]);
        columns[table] = Some((k, v));
        let Some(probe) = root else {
            root = Some((node, vec![k.id, v.id]));
            continue;
        };
        let mut nodes = ExprBuilder::new(Global::current()).unwrap();
        for &(a, b) in edges {
            let (Some((_, av)), Some((bk, _))) = (columns[a], columns[b]) else { continue };
            if a != table && b != table {
                continue;
            }
            let before = (!nodes.nodes().is_empty()).then(|| nodes.last());
            let (x, y) = (nodes.column(av.id).unwrap(), nodes.column(bk.id).unwrap());
            let equal = nodes.node(ExprOp::Eq, x, y).unwrap();
            if let Some(before) = before {
                nodes.node(ExprOp::And, before, equal).unwrap();
            }
        }
        let condition = plan.exprs.add(nodes.nodes()).unwrap();
        let (probe, mut probe_columns) = probe;
        let sides = [
            SlowVec::fixed_from(Global::current(), probe_columns.iter().copied()).unwrap(),
            SlowVec::fixed_from(Global::current(), [k.id, v.id].into_iter()).unwrap(),
        ];
        let conjuncts = conjuncts_of(Global::current(), &plan.exprs, condition).unwrap();
        let join = JoinOp::new(JoinKind::Inner, conjuncts, sides);
        let join =
            plan.add_node(DynOp::new(Global::current(), join).unwrap(), &[probe, node]).unwrap();
        probe_columns.extend([k.id, v.id]);
        root = Some((join, probe_columns));
    }
    let vs: Vec<NamedColumn> = columns.iter().map(|c| c.unwrap().1).collect();
    for &v in &vs {
        plan.output.push(v).unwrap();
    }
    optimize(Global::current(), &mut plan).unwrap();
    // Each scan's table, by its rows: every table has a different number.
    let rows: Vec<i64> = tables.iter().map(|t| t.summary(None).unwrap().rows).collect();
    // A scan is the only node without children here.
    let scans = |node: u32| {
        let node = &plan.nodes[node as usize];
        let scanned = node.op.summary(None).filter(|_| node.children.is_empty());
        scanned.map(|summary| rows.iter().position(|&r| r == summary.rows).unwrap())
    };
    let tree = shape(&plan, plan.root, &scans);
    let planned = plan_nodes(Global::current(), &plan).unwrap();
    let physical = lower(Global::current(), &mut plan, &planned).unwrap();
    let mut execution = physical.start(Global::current()).unwrap();
    let mut batch = RowBatch::new();
    let mut rows = Vec::new();
    let positions: Vec<u32> = physical.columns().iter().map(|column| column.position).collect();
    while execution.next(&mut batch).unwrap() {
        let selection = batch.selection();
        for row in (0..selection.len()).map(|i| selection.kept().row(i)) {
            rows.push(positions.iter().map(|&p| batch.column(p).int64s()[row as usize]).collect());
        }
    }
    rows.sort_unstable();
    (rows, tree)
}

/// The joins under `node` of `plan` as a nested list of tables, which
/// `scans` gives each scan's.
fn shape(plan: &LogicalPlan<'_>, node: u32, scans: &dyn Fn(u32) -> Option<usize>) -> String {
    if let Some(table) = scans(node) {
        return table.to_string();
    }
    let children = &plan.nodes[node as usize].children;
    // In either order: each join chooses its build side later.
    let mut parts: Vec<String> = children.iter().map(|&c| shape(plan, c, scans)).collect();
    parts.sort_unstable();
    format!("({})", parts.join(" "))
}

/// What `joined_in_order` should give: every combination of rows each edge
/// holds for, by brute force.
fn joined_expected(keys: &[Vec<Option<i64>>], edges: &[(usize, usize)]) -> Vec<Vec<i64>> {
    // A table's `v` is its row number, and its `k` is given.
    let mut rows: Vec<Vec<i64>> = vec![vec![]];
    for (table, table_keys) in keys.iter().enumerate() {
        let mut next = Vec::new();
        for row in &rows {
            for (v, &k) in (0..).zip(table_keys) {
                let holds = edges.iter().all(|&(a, b)| match (a == table, b == table) {
                    // `a.v = b.k`, once both are in.
                    (false, true) if a < table => k == Some(row[a]),
                    (true, false) if b < table => {
                        Some(v) == keys[b][usize::try_from(row[b]).unwrap()]
                    }
                    _ => true,
                });
                if holds {
                    let mut row = row.clone();
                    row.push(v);
                    next.push(row);
                }
            }
        }
        rows = next;
    }
    rows.sort_unstable();
    rows
}

#[test]
fn plans_the_order_of_several_joins() {
    // Four tables of very different sizes, each `v` its row number, joined
    // along a chain: 0.v = 1.k, 1.v = 2.k, 2.v = 3.k. Any order it's
    // written in gives the same rows, and the same plan.
    // Without distinct counts, a join is estimated to make as many rows as
    // its larger side, so these sizes have one cheapest plan.
    let sizes = [3000, 40, 900, 2000];
    let keys: Vec<Vec<Option<i64>>> = (0..4_u8)
        .map(|t| {
            (0..sizes[usize::from(t)])
                .map(|i| (i % 13 != 4).then_some((i * 7 + i64::from(t)) % 50))
                .collect()
        })
        .collect();
    let tables: Vec<DynScannable<'static>> = keys.iter().map(|keys| table(keys)).collect();
    let edges = [(0, 1), (1, 2), (2, 3)];
    let expected = joined_expected(&keys, &edges);
    let mut trees = Vec::new();
    for order in [[0, 1, 2, 3], [3, 2, 1, 0], [1, 2, 0, 3], [2, 1, 3, 0]] {
        let (rows, tree) = joined_in_order(&tables, &order, &edges);
        assert_eq!(rows, expected, "{order:?}");
        trees.push(tree);
    }
    assert!(trees.iter().all(|tree| *tree == trees[0]), "{trees:?}");
    // The two smallest first, as they make the fewest rows.
    assert_eq!(trees[0], "(((1 2) 3) 0)");
}
