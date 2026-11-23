//! Joins planned and run as a frontend's would be, over in-memory tables.

extern crate std;

use std::vec;
use std::vec::Vec;

use crate::table::Table;
use pipit_kernel::allocator::Heap;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::error::Error;
use pipit_kernel::expression::{ExprBuilder, ExprOp};
use pipit_kernel::join::JoinOp;
use pipit_kernel::lower::lower;
use pipit_kernel::optimize::optimize;
use pipit_kernel::output_properties::plan_nodes;
use pipit_kernel::plan::{
    DynOp, FilterOp, LogicalPlan, NamedColumn, PlanNodeId, ScanColumn, ScanOp,
};
use pipit_kernel::query_allocators::QueryAllocators;
use pipit_kernel::row_batch::RowBatch;
use pipit_kernel::scannable::DynScannable;
use pipit_kernel::slow_vec::SlowVec;

/// Adds the nodes of `column > value`, and returns its root.
fn greater(nodes: &mut ExprBuilder, column: u32, value: i64) -> u32 {
    let (column, value) = (nodes.column(column).unwrap(), nodes.int(value).unwrap());
    nodes.node(ExprOp::Gt, column, value).unwrap()
}

/// A filter of the expression of `nodes`, added to `plan`'s.
fn filter<'c>(plan: &mut LogicalPlan<'c>, nodes: &ExprBuilder) -> DynOp<'c> {
    let condition = plan.exprs.add(nodes.nodes()).unwrap();
    DynOp::new(&Heap, FilterOp::new(&Heap, &plan.exprs, condition).unwrap()).unwrap()
}

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

fn some(values: impl Iterator<Item = i64>) -> Vec<Option<i64>> {
    values.map(Some).collect()
}

/// `thread`: `id` from 0, in row groups of `groups` rows, as a sequence if
/// `sequence`, and `tid`, 1000 more.
fn threads(groups: &[(i64, u32)], sequence: bool) -> DynScannable<'static> {
    let mut context = Context::new(&Heap);
    let row_groups: Vec<[ColumnView; 2]> = groups
        .iter()
        .map(|&(first, rows)| {
            let ids = first..first + i64::from(rows);
            let id = if sequence {
                ColumnView::sequence(&mut context, first, rows).unwrap()
            } else {
                int64s(&some(ids.clone()))
            };
            [id, int64s(&some(ids.map(|id| id + 1000)))]
        })
        .collect();
    let groups: Vec<&[ColumnView]> = row_groups.iter().map(|group| &group[..]).collect();
    let schema = [("id", DataType::Int64), ("tid", DataType::Int64)];
    DynScannable::new(&Heap, Table::new(&Heap, &schema, &groups).unwrap()).unwrap()
}

/// `slice`: `utid`, as given, and `dur`, its row number.
fn slices(utids: &[Option<i64>]) -> DynScannable<'static> {
    let durs: Vec<Option<i64>> = (0..).take(utids.len()).map(Some).collect();
    let schema = [("utid", DataType::Int64), ("dur", DataType::Int64)];
    let columns = [int64s(utids), int64s(&durs)];
    DynScannable::new(&Heap, Table::new(&Heap, &schema, &[columns.as_slice()]).unwrap()).unwrap()
}

/// A scan of `columns` of `table`, as new columns of `plan`.
fn scan<'c>(
    plan: &mut LogicalPlan<'c>,
    table: &'c DynScannable<'c>,
    columns: &[&str],
) -> (PlanNodeId, Vec<NamedColumn>) {
    let mut reads = SlowVec::new(&Heap, 8).unwrap();
    let mut bound = Vec::new();
    for (i, &name) in (0..).zip(columns) {
        let binding = plan.add_column(name, DataType::Int64, true).unwrap();
        reads.push(ScanColumn { column: i, binding, forms: Forms::FLAT }).unwrap();
        bound.push(binding);
    }
    let op = DynOp::new(&Heap, ScanOp::new(table, reads)).unwrap();
    (plan.add_node(op, &[]).unwrap(), bound)
}

/// `slice JOIN thread ON utid = id`'s `dur` and `tid`, by row, of the
/// slices whose `dur` is above `above`, if given.
fn join(
    slice: &DynScannable<'_>,
    thread: &DynScannable<'_>,
    above: Option<i64>,
) -> Result<Vec<(i64, i64)>, Error> {
    let mut plan = LogicalPlan::new(&Heap).unwrap();
    let (mut probe, probe_columns) = scan(&mut plan, slice, &["utid", "dur"]);
    if let Some(value) = above {
        let mut nodes = ExprBuilder::new(&Heap).unwrap();
        greater(&mut nodes, probe_columns[1].id, value);
        let filter = filter(&mut plan, &nodes);
        let children = [probe];
        probe = plan.add_node(filter, &children).unwrap();
    }
    let (build, build_columns) = scan(&mut plan, thread, &["id", "tid"]);
    let (utid, dur, id, tid) =
        (probe_columns[0], probe_columns[1], build_columns[0], build_columns[1]);
    let joined = SlowVec::fixed_from(&Heap, [tid.id].into_iter()).unwrap();
    let op = JoinOp { probe_key: utid.id, build_key: id.id, build_columns: joined };
    let children = [probe, build];
    plan.add_node(DynOp::new(&Heap, op).unwrap(), &children).unwrap();
    plan.output.push(dur).unwrap();
    plan.output.push(tid).unwrap();
    optimize(&Heap, &mut plan).unwrap();
    let planned = plan_nodes(&Heap, &plan).unwrap();
    let physical = lower(&Heap, &mut plan, &planned).unwrap();
    let query = QueryAllocators::new(&Heap);
    let mut execution = physical.start(&query).unwrap();
    let (mut batch, mut rows) = (RowBatch::new(), Vec::new());
    let [dur, tid] = [0, 1].map(|i| physical.columns()[i].position);
    while execution.next(&mut batch)? {
        let selection = batch.selection();
        for row in (0..selection.len()).map(|i| selection.kept().row(i) as usize) {
            rows.push((batch.column(dur).int64s()[row], batch.column(tid).int64s()[row]));
        }
    }
    Ok(rows)
}

#[test]
fn joins_rows_to_their_ids() {
    // Ids across two row groups, so a batch's sequence follows another's.
    let thread = threads(&[(0, 2048), (2048, 952)], true);
    let utids = [Some(5), Some(-1), Some(2999), None, Some(3000), Some(2048), Some(5)];
    let slice = slices(&utids);
    // Each slice with a thread, in order: rows 1, 3 and 4 have none.
    let expected = [(0, 1005), (2, 3999), (5, 3048), (6, 1005)];
    assert_eq!(join(&slice, &thread, None).unwrap(), expected);
    // Of slices a filter kept.
    assert_eq!(join(&slice, &thread, Some(0)).unwrap(), expected[1..]);
    // No thread, so no slice is joined.
    assert_eq!(join(&slice, &threads(&[], true), None).unwrap(), []);
}

#[test]
fn joins_only_to_sequences_so_far() {
    let slice = slices(&[Some(1)]);
    assert_eq!(join(&slice, &threads(&[(0, 4)], false), None), Err(Error::Unsupported));
    // A gap between batches.
    let gap = threads(&[(0, 4), (5, 4)], true);
    assert_eq!(join(&slice, &gap, None), Err(Error::Unsupported));
}

/// Runs `slice JOIN thread ON utid = id WHERE dur > durs AND tid > tids`,
/// returning each row's `dur` and `tid`, and the optimized plan's shape:
/// each node's number of children, from the root down, first side first.
fn filtered_above(
    slice: &DynScannable<'_>,
    thread: &DynScannable<'_>,
    durs: i64,
    tids: i64,
) -> (Vec<(i64, i64)>, Vec<usize>) {
    let mut plan = LogicalPlan::new(&Heap).unwrap();
    let (probe, probe_columns) = scan(&mut plan, slice, &["utid", "dur"]);
    let (build, build_columns) = scan(&mut plan, thread, &["id", "tid"]);
    let (utid, dur, id, tid) =
        (probe_columns[0], probe_columns[1], build_columns[0], build_columns[1]);
    let joined = SlowVec::fixed_from(&Heap, [tid.id].into_iter()).unwrap();
    let op = JoinOp { probe_key: utid.id, build_key: id.id, build_columns: joined };
    let children = [probe, build];
    let join = plan.add_node(DynOp::new(&Heap, op).unwrap(), &children).unwrap();
    let mut nodes = ExprBuilder::new(&Heap).unwrap();
    let a = greater(&mut nodes, dur.id, durs);
    let b = greater(&mut nodes, tid.id, tids);
    nodes.node(ExprOp::And, a, b).unwrap();
    let filter = filter(&mut plan, &nodes);
    let children = [join];
    plan.add_node(filter, &children).unwrap();
    plan.output.push(dur).unwrap();
    plan.output.push(tid).unwrap();
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
    let (mut batch, mut rows) = (RowBatch::new(), Vec::new());
    let [dur, tid] = [0, 1].map(|i| physical.columns()[i].position);
    while execution.next(&mut batch).unwrap() {
        let selection = batch.selection();
        for row in (0..selection.len()).map(|i| selection.kept().row(i) as usize) {
            rows.push((batch.column(dur).int64s()[row], batch.column(tid).int64s()[row]));
        }
    }
    (rows, shape)
}

#[test]
fn moves_conditions_above_a_join_to_its_sides() {
    let thread = threads(&[(0, 2048), (2048, 952)], true);
    let slice = slices(&[Some(5), Some(-1), Some(2999), None, Some(3000), Some(2048)]);
    let (rows, shape) = filtered_above(&slice, &thread, 0, 1005);
    // `dur > 0` drops slice 0, and `tid > 1005` the slice of thread 5.
    assert_eq!(rows, [(2, 3999), (5, 3048)]);
    // `dur > 0` moves below the join, onto the slice scan. `tid > 1005`
    // stays above the join until the join can handle a build side with gaps
    // in its ids.
    assert_eq!(shape, [1, 2, 1, 0, 0]);
}
