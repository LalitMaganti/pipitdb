//! `FilterOp`: a plan node that keeps the rows a predicate is true for, as
//! `WHERE` does, and `Filter`, the transform it lowers to.

use pipit_kernel::allocator::AllocError;
use pipit_kernel::context::Context;
use pipit_kernel::lower::{LowerError, Lowering};
use pipit_kernel::optimize::{Needed, Pruned};
use pipit_kernel::plan::{Op, PlanNode};
use pipit_kernel::predicate::Predicate;
use pipit_kernel::row_batch::RowBatch;
use pipit_kernel::step::{DynTransform, Step, Transform};

/// Keeps the rows of its one child that `predicate` is true for. The
/// predicate reads columns by `ColumnId`, and each comparison's value has
/// its column's type.
pub struct FilterOp {
    pub predicate: Predicate,
}

impl<'c> Op<'c> for FilterOp {
    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        lowering.lower(*at!(node.children, 0))?;
        let allocator = lowering.allocator();
        let predicate = self.predicate.renumbered(allocator.clone(), |id| lowering.position(id))?;
        lowering.add_step(Step::Transform(DynTransform::new(allocator, Filter { predicate })?))
    }

    /// Makes no columns, and reads its predicate's.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        for column in self.predicate.columns() {
            needed.need(column);
        }
        Pruned::Keep
    }
}

/// Narrows each batch's selection to the rows `predicate` is true for. The
/// predicate reads columns by their position in batches.
pub struct Filter {
    pub predicate: Predicate,
}

impl Transform for Filter {
    type State = ();

    fn new_state(&self, context: &mut Context) -> Result<(), AllocError> {
        context.reserve_selections(self.predicate.depth() as usize)
    }

    fn process(&self, context: &mut Context, (): &mut (), batch: &mut RowBatch) {
        self.predicate.select(context.selections(), batch);
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec::Vec as StdVec;

    use pipit_kernel::allocator::Heap;
    use pipit_kernel::buffer::Buffer;
    use pipit_kernel::column::{ColumnView, DataType};
    use pipit_kernel::filter::{Comparison, Value};
    use pipit_kernel::lower::lower;
    use pipit_kernel::plan::{DynOp, LogicalPlan, ScanColumn, ScanOp};
    use pipit_kernel::predicate::{Leaf, Node};
    use pipit_kernel::scannable::DynScannable;
    use pipit_kernel::selection::Kept;
    use pipit_kernel::vec::Vec;

    use super::*;
    use crate::table::Table;

    /// `cells`, with `None` for null.
    fn column(cells: &[Option<i64>]) -> ColumnView {
        let mut values = Buffer::allocate(Heap, cells.len() * 8).unwrap();
        let mut validity = Buffer::allocate(Heap, 1).unwrap();
        for (row, cell) in cells.iter().enumerate() {
            values.as_mut_slice::<i64>()[row] = cell.unwrap_or(0);
            validity.as_mut_slice::<u8>()[0] |= u8::from(cell.is_some()) << row;
        }
        ColumnView::new(DataType::Int64, values, Some(validity))
    }

    /// `FROM t |> WHERE a > 1 OR b IS NULL |> SELECT b`, pruned, as the
    /// values of `b` it keeps and how many columns its batches have.
    #[test]
    fn keeps_rows_and_the_columns_it_reads() {
        let a = column(&[Some(1), Some(2), None, Some(0)]);
        let b = column(&[Some(10), Some(20), Some(30), None]);
        let schema = [("a", DataType::Int64), ("b", DataType::Int64)];
        let table = Table::new(Heap, &schema, &[&[a, b]]).unwrap();
        let table = DynScannable::new(Heap, table).unwrap();

        let mut plan = LogicalPlan::new(Heap).unwrap();
        let mut columns = Vec::fixed(Heap, 2).unwrap();
        let mut bindings = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64).unwrap();
            assert!(columns.push(ScanColumn { column, binding }).is_ok());
            bindings.push(binding);
        }
        let scan = DynOp::new(Heap, ScanOp { scannable: &table, columns }).unwrap();
        let scan = plan.add_node(scan, Vec::fixed(Heap, 0).unwrap()).unwrap();
        let (a, b) = (bindings[0].id, bindings[1].id);
        let greater =
            Leaf::Compare { column: a, comparison: Comparison::Greater, value: Value::Int64(1) };
        let nodes = [Node::Leaf(greater), Node::Leaf(Leaf::IsNull { column: b }), Node::Or(0, 1)];
        let predicate = Predicate::new(Vec::fixed_from(Heap, nodes.into_iter()).unwrap());
        let filter = DynOp::new(Heap, FilterOp { predicate }).unwrap();
        plan.add_node(filter, Vec::fixed_from(Heap, [scan].into_iter()).unwrap()).unwrap();
        assert!(plan.output.push(bindings[1]).is_ok());
        pipit_kernel::optimize::optimize(Heap, &mut plan).unwrap();

        let physical = lower(Heap, &plan).unwrap();
        let mut execution = physical.pipeline().start(Heap).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch));
        let b = batch.column(physical.columns()[0].position);
        let Kept::Select(rows) = batch.selection().kept() else { panic!("not narrowed") };
        let kept: StdVec<(bool, i64)> = rows
            .iter()
            .map(|&row| (b.is_null(u32::from(row)), b.int64s()[usize::from(row)]))
            .collect();
        // Row 1 has `a > 1`; row 3 has a null `b`.
        assert_eq!(kept, [(false, 20), (true, 0)]);
        // `a` is read by the filter, so it isn't pruned.
        assert_eq!(batch.column_count(), 2);
        assert!(!execution.next(&mut batch));
    }
}
