//! Cross joins planned and run as a frontend's would be, over in-memory
//! tables.

extern crate std;

use std::vec;
use std::vec::Vec;

use crate::table::Table;
use pipit_kernel::allocator::Heap;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::cross_join::CrossJoinOp;
use pipit_kernel::filter::{Comparison, Value};
use pipit_kernel::lower::lower;
use pipit_kernel::optimize::optimize;
use pipit_kernel::output_properties::plan_nodes;
use pipit_kernel::plan::{
    DynOp, FilterOp, LogicalPlan, NamedColumn, PlanNodeId, ScanColumn, ScanOp,
};
use pipit_kernel::predicate::{Leaf, Node, Predicate};
use pipit_kernel::query_allocators::QueryAllocators;
use pipit_kernel::row_batch::{BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::DynScannable;
use pipit_kernel::slow_vec::SlowVec;

/// A table of `v`, its row number, of `rows` rows in batches.
fn table(rows: u32) -> DynScannable<'static> {
    let groups: Vec<[ColumnView; 1]> = (0..rows)
        .step_by(BATCH_ROWS_MAX as usize)
        .map(|first| {
            let count = (rows - first).min(BATCH_ROWS_MAX);
            let mut values = Buffer::allocate(&Heap, count as usize * 8).unwrap();
            for (slot, v) in values.as_mut_slice::<i64>().iter_mut().zip(first..) {
                *slot = i64::from(v);
            }
            [ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, None).unwrap()]
        })
        .collect();
    let groups: Vec<&[ColumnView]> = groups.iter().map(|group| &group[..]).collect();
    DynScannable::new(&Heap, Table::new(&Heap, &[("v", DataType::Int64)], &groups).unwrap())
        .unwrap()
}

fn scan<'c>(plan: &mut LogicalPlan<'c>, table: &'c DynScannable<'c>) -> (PlanNodeId, NamedColumn) {
    let v = plan.add_column("v", DataType::Int64, false).unwrap();
    let reads = SlowVec::fixed_from(
        &Heap,
        [ScanColumn { column: 0, binding: v, forms: Forms::FLAT }].into_iter(),
    );
    let op = DynOp::new(&Heap, ScanOp::new(table, reads.unwrap())).unwrap();
    (plan.add_node(op, &[]).unwrap(), v)
}

/// Each output row's probe and build `v`, sorted, of `probe CROSS JOIN
/// build`, under a filter on the build side's `v > above` if given; and
/// the optimized plan's shape, each node's number of children from the
/// root down.
fn cross(probe: u32, build: u32, above: Option<i64>) -> (Vec<(i64, i64)>, Vec<usize>) {
    let (probe, build) = (table(probe), table(build));
    let mut plan = LogicalPlan::new(&Heap).unwrap();
    let (probe, probe_v) = scan(&mut plan, &probe);
    let (build, build_v) = scan(&mut plan, &build);
    let build_columns = SlowVec::fixed_from(&Heap, [build_v.id].into_iter()).unwrap();
    let op = DynOp::new(&Heap, CrossJoinOp { build_columns }).unwrap();
    let join = plan.add_node(op, &[probe, build]).unwrap();
    if let Some(value) = above {
        let leaf = Leaf::Compare {
            column: build_v.id,
            comparison: Comparison::Greater,
            value: Value::Int64(value),
        };
        let predicate =
            Predicate::new(SlowVec::fixed_from(&Heap, [Node::Leaf(leaf)].into_iter()).unwrap());
        let filter = DynOp::new(&Heap, FilterOp { predicate }).unwrap();
        plan.add_node(filter, &[join]).unwrap();
    }
    plan.output.push(probe_v).unwrap();
    plan.output.push(build_v).unwrap();
    optimize(&Heap, &mut plan).unwrap();
    let (mut shape, mut pending) = (Vec::new(), vec![plan.root]);
    while let Some(node) = pending.pop() {
        let children = &plan.nodes[node as usize].children;
        shape.push(children.len());
        pending.extend(children.iter().rev());
    }
    let planned = plan_nodes(&Heap, &plan).unwrap();
    let physical = lower(&Heap, &mut plan, &planned).unwrap();
    let query = QueryAllocators::new(&Heap);
    let mut execution = physical.start(&query).unwrap();
    let [p, b] = [0, 1].map(|i| physical.columns()[i].position);
    let (mut batch, mut rows) = (RowBatch::new(), Vec::new());
    while execution.next(&mut batch).unwrap() {
        let selection = batch.selection();
        for row in (0..selection.len()).map(|i| selection.kept().row(i) as usize) {
            rows.push((batch.column(p).int64s()[row], batch.column(b).int64s()[row]));
        }
    }
    rows.sort_unstable();
    (rows, shape)
}

#[test]
fn pairs_every_row_with_every_build_row() {
    // Probe rows over two batches.
    let expected: Vec<(i64, i64)> = (0..2100).flat_map(|p| (0..3).map(move |b| (p, b))).collect();
    assert_eq!(cross(2100, 3, None).0, expected);
    assert_eq!(cross(5, 0, None).0, []);
}

#[test]
fn moves_a_filter_above_to_the_side_it_reads() {
    let (rows, shape) = cross(2, 3, Some(1));
    assert_eq!(rows, [(0, 2), (1, 2)]);
    // The filter moves onto the build side's scan.
    assert_eq!(shape, [2, 0, 1, 0]);
}
