//! Joins planned and run as a frontend's would be, over in-memory tables.

extern crate std;

use std::vec::Vec;

use crate::table::Table;
use pipit_kernel::allocator::Heap;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::error::Error;
use pipit_kernel::filter::{Comparison, Value};
use pipit_kernel::join::{JoinKind, JoinOp};
use pipit_kernel::lower::lower;
use pipit_kernel::optimize::optimize;
use pipit_kernel::output_properties::plan_nodes;
use pipit_kernel::plan::{
    DynOp, FilterOp, LogicalPlan, NamedColumn, PlanNodeId, ScanColumn, ScanOp,
};
use pipit_kernel::predicate::{Leaf, Node, Predicate};
use pipit_kernel::query_allocators::QueryAllocators;
use pipit_kernel::row_batch::RowBatch;
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
    (plan.add_node(op, SlowVec::fixed(&Heap, 0).unwrap()).unwrap(), bound)
}

/// `slice JOIN thread ON utid = id`'s `dur` and `tid`, by row, of the
/// slices whose `dur` is above `above`, if given; written `thread JOIN slice`
/// if `reversed`.
fn join(
    slice: &DynScannable<'_>,
    thread: &DynScannable<'_>,
    above: Option<i64>,
    reversed: bool,
) -> Result<Vec<(i64, i64)>, Error> {
    let mut plan = LogicalPlan::new(&Heap).unwrap();
    let (mut probe, probe_columns) = scan(&mut plan, slice, &["utid", "dur"]);
    if let Some(value) = above {
        let comparison = Comparison::Greater;
        let leaf =
            Leaf::Compare { column: probe_columns[1].id, comparison, value: Value::Int64(value) };
        let nodes = SlowVec::fixed_from(&Heap, [Node::Leaf(leaf)].into_iter()).unwrap();
        let filter = DynOp::new(&Heap, FilterOp { predicate: Predicate::new(nodes) }).unwrap();
        let children = SlowVec::fixed_from(&Heap, [probe].into_iter()).unwrap();
        probe = plan.add_node(filter, children).unwrap();
    }
    let (build, build_columns) = scan(&mut plan, thread, &["id", "tid"]);
    let (utid, dur, id, tid) =
        (probe_columns[0], probe_columns[1], build_columns[0], build_columns[1]);
    let columns = |ids: [u32; 2]| SlowVec::fixed_from(&Heap, ids.into_iter()).unwrap();
    let (slices, threads) = (columns([utid.id, dur.id]), columns([id.id, tid.id]));
    let (op, children) = if reversed {
        (JoinOp::new(JoinKind::Inner, [id.id, utid.id], [threads, slices]), [build, probe])
    } else {
        (JoinOp::new(JoinKind::Inner, [utid.id, id.id], [slices, threads]), [probe, build])
    };
    let children = SlowVec::fixed_from(&Heap, children.into_iter()).unwrap();
    plan.add_node(DynOp::new(&Heap, op).unwrap(), children).unwrap();
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
    assert_eq!(join(&slice, &thread, None, false).unwrap(), expected);
    // Threads first: planning builds them still, as their ids are a sequence.
    assert_eq!(join(&slice, &thread, None, true).unwrap(), expected);
    // Of slices a filter kept.
    assert_eq!(join(&slice, &thread, Some(0), false).unwrap(), expected[1..]);
    // No thread, so no slice is joined.
    assert_eq!(join(&slice, &threads(&[], true), None, false).unwrap(), []);
}

#[test]
fn joins_only_to_sequences_so_far() {
    let slice = slices(&[Some(1)]);
    assert_eq!(join(&slice, &threads(&[(0, 4)], false), None, false), Err(Error::Unsupported));
    // A gap between batches.
    let gap = threads(&[(0, 4), (5, 4)], true);
    assert_eq!(join(&slice, &gap, None, false), Err(Error::Unsupported));
}

/// The `dur`s of the slices whose `utid` is (`Semi`) or isn't (`Anti`) a
/// thread's `id`, in order.
fn filtered(slice: &DynScannable<'_>, thread: &DynScannable<'_>, kind: JoinKind) -> Vec<i64> {
    let mut plan = LogicalPlan::new(&Heap).unwrap();
    let (probe, probe_columns) = scan(&mut plan, slice, &["utid", "dur"]);
    let (build, build_columns) = scan(&mut plan, thread, &["id"]);
    let (utid, dur, id) = (probe_columns[0], probe_columns[1], build_columns[0]);
    let slices = SlowVec::fixed_from(&Heap, [utid.id, dur.id].into_iter()).unwrap();
    let op = JoinOp::new(kind, [utid.id, id.id], [slices, SlowVec::fixed(&Heap, 0).unwrap()]);
    let children = SlowVec::fixed_from(&Heap, [probe, build].into_iter()).unwrap();
    plan.add_node(DynOp::new(&Heap, op).unwrap(), children).unwrap();
    plan.output.push(dur).unwrap();
    optimize(&Heap, &mut plan).unwrap();
    let planned = plan_nodes(&Heap, &plan).unwrap();
    let physical = lower(&Heap, &mut plan, &planned).unwrap();
    let query = QueryAllocators::new(&Heap);
    let mut execution = physical.start(&query).unwrap();
    let (mut batch, mut durs) = (RowBatch::new(), Vec::new());
    let dur = physical.columns()[0].position;
    while execution.next(&mut batch).unwrap() {
        let selection = batch.selection();
        for row in (0..selection.len()).map(|i| selection.kept().row(i) as usize) {
            durs.push(batch.column(dur).int64s()[row]);
        }
    }
    durs
}

#[test]
fn keeps_rows_whose_ids_are_or_arent_in_a_table() {
    let thread = threads(&[(0, 2048), (2048, 952)], true);
    let slice = slices(&[Some(5), Some(-1), Some(2999), None, Some(3000), Some(2048)]);
    assert_eq!(filtered(&slice, &thread, JoinKind::Semi), [0, 2, 5]);
    // A null isn't `NOT IN` a list it isn't in either.
    assert_eq!(filtered(&slice, &thread, JoinKind::Anti), [1, 4]);
    // Of an empty list, everything is `NOT IN`, nulls too, and nothing `IN`.
    let none = threads(&[], true);
    assert_eq!(filtered(&slice, &none, JoinKind::Semi), []);
    assert_eq!(filtered(&slice, &none, JoinKind::Anti), [0, 1, 2, 3, 4, 5]);
}
