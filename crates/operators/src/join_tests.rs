//! Nested-loop joins planned and run as a frontend's would be, over
//! in-memory tables, checked against each pair of rows tested by hand.

extern crate std;

use std::vec;
use std::vec::Vec;

use crate::table::Table;
use pipit_kernel::allocator::Heap;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::expression::{Expr, ExprBuilder, ExprOp, Exprs};
use pipit_kernel::join::{JoinKind, JoinOp};
use pipit_kernel::lower::lower;
use pipit_kernel::optimize::optimize;
use pipit_kernel::output_properties::plan_nodes;
use pipit_kernel::plan::{
    DynOp, FilterOp, LogicalPlan, NamedColumn, PlanNodeId, ScanColumn, ScanOp,
};
use pipit_kernel::query_allocators::QueryAllocators;
use pipit_kernel::row_batch::{BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::DynScannable;
use pipit_kernel::slow_vec::SlowVec;

fn int64s(values: &[Option<i64>]) -> ColumnView {
    let mut buffer = Buffer::allocate(&Heap, values.len() * 8).unwrap();
    for (slot, value) in buffer.as_mut_slice::<i64>().iter_mut().zip(values) {
        *slot = value.unwrap_or(0);
    }
    // A bit a row, set where it isn't null.
    let validity = values.iter().any(Option::is_none).then(|| {
        let mut bits = Buffer::allocate(&Heap, values.len().div_ceil(8)).unwrap();
        let bytes = bits.as_mut_slice::<u8>();
        bytes.fill(0);
        for (row, value) in values.iter().enumerate() {
            bytes[row / 8] |= u8::from(value.is_some()) << (row % 8);
        }
        bits
    });
    ColumnView::new(&mut Context::new(&Heap), DataType::Int64, buffer, validity).unwrap()
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
    DynScannable::new(&Heap, Table::new(&Heap, &schema, &groups).unwrap()).unwrap()
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
            [ColumnView::sequence(&mut Context::new(&Heap), first, rows).unwrap(), int64s(&numbers)]
        })
        .collect();
    let groups: Vec<&[ColumnView]> = groups.iter().map(|group| &group[..]).collect();
    let schema = [("k", DataType::Int64), ("v", DataType::Int64)];
    (DynScannable::new(&Heap, Table::new(&Heap, &schema, &groups).unwrap()).unwrap(), keys)
}

/// A scan of `k` and `v` of `table`, as new columns of `plan`.
fn scan<'c>(
    plan: &mut LogicalPlan<'c>,
    table: &'c DynScannable<'c>,
) -> (PlanNodeId, [NamedColumn; 2]) {
    let mut reads = SlowVec::new(&Heap, 2).unwrap();
    let mut bound = Vec::new();
    for (i, name) in (0..).zip(["k", "v"]) {
        let binding = plan.add_column(name, DataType::Int64, true).unwrap();
        reads.push(ScanColumn { column: i, binding, forms: Forms::FLAT }).unwrap();
        bound.push(binding);
    }
    let op = DynOp::new(&Heap, ScanOp::new(table, reads)).unwrap();
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
    let mut nodes = ExprBuilder::new(&Heap).unwrap();
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
    let mut plan = LogicalPlan::new(&Heap).unwrap();
    let (probe, [probe_k, probe_v]) = scan(&mut plan, probe);
    let (build, [build_k, build_v]) = scan(&mut plan, build);
    let columns = [probe_k.id, probe_v.id, build_k.id, build_v.id];
    let condition = condition(&mut plan.exprs, on, columns);
    let build_columns = SlowVec::fixed_from(&Heap, [build_k.id, build_v.id].into_iter()).unwrap();
    let probe_columns = SlowVec::fixed_from(&Heap, [probe_k.id, probe_v.id].into_iter()).unwrap();
    let join = JoinOp::new(kind, condition, [probe_columns, build_columns]);
    let children = [probe, build];
    let join = plan.add_node(DynOp::new(&Heap, join).unwrap(), &children).unwrap();
    if above != Above::Nothing {
        let mut nodes = ExprBuilder::new(&Heap).unwrap();
        let column = nodes.column(build_v.id).unwrap();
        if above == Above::IsNull {
            nodes.node(ExprOp::IsNull, column, 0).unwrap();
        } else {
            let zero = nodes.int(0).unwrap();
            nodes.node(ExprOp::Gt, column, zero).unwrap();
        }
        let condition = plan.exprs.add(nodes.nodes()).unwrap();
        let filter = FilterOp::new(&Heap, &plan.exprs, condition).unwrap();
        let filter = DynOp::new(&Heap, filter).unwrap();
        let children = [join];
        plan.add_node(filter, &children).unwrap();
    }
    plan.output.push(probe_v).unwrap();
    plan.output.push(build_v).unwrap();
    optimize(&Heap, &mut plan).unwrap();
    let mut shape = Vec::new();
    let mut pending = vec![plan.root];
    while let Some(node) = pending.pop() {
        let children = &plan.nodes[node as usize].children;
        shape.push(children.len());
        pending.extend(children.iter().rev());
    }
    let planned = plan_nodes(&Heap, &plan).unwrap();
    let physical = lower(&Heap, &mut plan, &planned).unwrap();
    let query = QueryAllocators::new(&Heap);
    let mut execution = physical.start(&query).unwrap();
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
    // with no array; with a gap, and in another order, found in an array;
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
