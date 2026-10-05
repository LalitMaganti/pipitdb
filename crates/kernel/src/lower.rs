//! `lower`: a `LogicalPlan` to a `PhysicalPlan`, a pipeline ready to run.

use crate::allocator::{AllocError, Allocator};
use crate::column::DataType;
use crate::names::{Name, Names};
use crate::pipeline::Pipeline;
use crate::plan::{ColumnId, LogicalPlan, PLAN_NAME_BYTES_MAX, PlanNodeId};
use crate::row_batch::BATCH_COLUMNS_MAX;
use crate::slow_vec::{Full, SlowVec};
use crate::step::{DynSource, Step};

/// Why a plan couldn't be lowered.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LowerError {
    OutOfMemory,
    /// A batch would need more than `BATCH_COLUMNS_MAX` columns.
    TooManyColumns,
}

impl From<AllocError> for LowerError {
    fn from(_: AllocError) -> LowerError {
        LowerError::OutOfMemory
    }
}

impl<T> From<Full<T>> for LowerError {
    fn from(_: Full<T>) -> LowerError {
        LowerError::OutOfMemory
    }
}

/// The most steps a pipeline from a plan can have.
pub const LOWERED_STEPS_MAX: usize = 1 << 6;

/// A column of a plan's result: its name, and its position in the batches
/// the pipeline makes.
#[derive(Clone, Copy)]
pub struct OutputColumn {
    pub name: Name,
    pub position: u32,
}

/// Read-only once built, so it can be run any number of times.
pub struct PhysicalPlan<'c> {
    pipeline: Pipeline<'c>,
    names: Names,
    columns: SlowVec<OutputColumn>,
}

impl<'c> PhysicalPlan<'c> {
    pub fn pipeline(&self) -> &Pipeline<'c> {
        &self.pipeline
    }

    pub fn columns(&self) -> &[OutputColumn] {
        &self.columns
    }

    pub fn name(&self, column: OutputColumn) -> &str {
        self.names.get(column.name)
    }
}

/// A pipeline being built from a plan, which each operation adds to.
pub struct Lowering<'p, 'c> {
    plan: &'p LogicalPlan<'c>,
    allocator: &'p dyn Allocator,
    source: Option<DynSource<'c>>,
    steps: SlowVec<Step<'c>>,
    /// Each column's position in batches, or `u32::MAX` until defined.
    positions: SlowVec<u32>,
    column_count: u32,
}

impl<'p, 'c> Lowering<'p, 'c> {
    /// What to make steps with.
    pub fn allocator(&self) -> &'p dyn Allocator {
        self.allocator
    }

    /// Lowers `node`, such as a child of the node being lowered.
    pub fn lower(&mut self, node: PlanNodeId) -> Result<(), LowerError> {
        let node = at!(self.plan.nodes, node as usize);
        node.op.lower(node, self)
    }

    /// Makes `source` the pipeline's.
    pub fn set_source(&mut self, source: DynSource<'c>) {
        check!(self.source.is_none());
        self.source = Some(source);
    }

    pub fn add_step(&mut self, step: Step<'c>) -> Result<(), LowerError> {
        Ok(self.steps.push(step)?)
    }

    /// Gives `column` the next position in batches, which fail to hold more
    /// than `BATCH_COLUMNS_MAX`.
    pub fn define(&mut self, column: ColumnId) -> Result<(), LowerError> {
        if self.column_count == BATCH_COLUMNS_MAX {
            return Err(LowerError::TooManyColumns);
        }
        *at_mut!(self.positions, column as usize) = self.column_count;
        self.column_count += 1;
        Ok(())
    }

    /// Starts defining columns from the first position again, after a step
    /// that makes new batches, such as an aggregation. Columns defined before
    /// aren't in them.
    pub fn restart_columns(&mut self) {
        self.positions.fill(u32::MAX);
        self.column_count = 0;
    }

    /// What `column` holds.
    pub fn data_type(&self, column: ColumnId) -> DataType {
        at!(self.plan.columns, column as usize).data_type
    }

    /// Where `column` is in batches.
    pub fn position(&self, column: ColumnId) -> u32 {
        let position = *at!(self.positions, column as usize);
        check!(position != u32::MAX);
        position
    }
}

/// Builds the pipeline for `plan`, from its root, with memory from
/// `allocator`.
pub fn lower<'c>(
    allocator: &dyn Allocator,
    plan: &LogicalPlan<'c>,
) -> Result<PhysicalPlan<'c>, LowerError> {
    let unset = core::iter::repeat_n(u32::MAX, plan.columns.len());
    let positions = SlowVec::fixed_from(allocator, unset)?;
    let mut lowering = Lowering {
        plan,
        allocator,
        source: None,
        steps: SlowVec::new(allocator, LOWERED_STEPS_MAX)?,
        positions,
        column_count: 0,
    };
    lowering.lower(plan.root)?;

    let mut names = Names::new(allocator, PLAN_NAME_BYTES_MAX)?;
    let mut columns = SlowVec::fixed(allocator, plan.output.len())?;
    for column in plan.output.iter() {
        let name = names.add(plan.names.get(column.name))?;
        columns.push(OutputColumn { name, position: lowering.position(column.id) })?;
    }
    let Some(source) = lowering.source else { crate::check::check_failed(line!()) };
    Ok(PhysicalPlan { pipeline: Pipeline::new(source, lowering.steps), names, columns })
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec as StdVec;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::{ColumnView, DataType};
    use crate::context::Context;
    use crate::error::Error;
    use crate::filter::{Comparison, Value};
    use crate::plan::{DynOp, FilterOp, ScanColumn, ScanOp};
    use crate::predicate::{Leaf, Node, Predicate};
    use crate::row_batch::RowBatch;
    use crate::scannable::{DynScannable, Scannable};
    use crate::selection::Kept;

    /// One batch of two rows: column `a` holds 1 and 2, `b` 10 and 20.
    struct Ab;

    impl Scannable for Ab {
        type State = bool;

        fn column_count(&self) -> u32 {
            2
        }

        fn column_name(&self, column: u32) -> &str {
            ["a", "b"][column as usize]
        }

        fn column_type(&self, _: u32) -> DataType {
            DataType::Int64
        }

        fn new_state(&self, _: &mut Context) -> Result<bool, Error> {
            Ok(false)
        }

        fn next(
            &self,
            columns: &[u32],
            _: &mut Context,
            done: &mut bool,
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            if *done {
                return Ok(false);
            }
            batch.reset(2);
            for &column in columns {
                let mut values = Buffer::allocate(&Heap, 16).unwrap();
                let first = [1, 10][column as usize];
                values.as_mut_slice::<i64>().copy_from_slice(&[first, first * 2]);
                assert!(batch.push_column(ColumnView::new(DataType::Int64, values, None)).is_ok());
            }
            *done = true;
            Ok(true)
        }
    }

    #[test]
    fn lowers_a_scan_with_its_output_in_order() {
        let table = DynScannable::new(&Heap, Ab).unwrap();
        let mut plan = LogicalPlan::new(&Heap).unwrap();
        let a = plan.add_column("a", DataType::Int64).unwrap();
        let b = plan.add_column("b", DataType::Int64).unwrap();
        let mut columns = SlowVec::fixed(&Heap, 2).unwrap();
        let a_column = ScanColumn { column: 0, binding: a };
        let b_column = ScanColumn { column: 1, binding: b };
        assert!(columns.push(a_column).is_ok() && columns.push(b_column).is_ok());
        let scan = DynOp::new(&Heap, ScanOp { scannable: &table, columns }).unwrap();
        plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
        assert!(plan.output.push(b).is_ok() && plan.output.push(a).is_ok());

        let physical = lower(&Heap, &plan).unwrap();
        let names: StdVec<&str> = physical.columns().iter().map(|&c| physical.name(c)).collect();
        assert_eq!(names, ["b", "a"]);

        let mut execution = physical.pipeline().start(&Heap).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch).unwrap());
        let rows: StdVec<&[i64]> =
            physical.columns().iter().map(|c| batch.column(c.position).int64s()).collect();
        assert_eq!(rows, [[10, 20], [1, 2]]);
        assert!(!execution.next(&mut batch).unwrap());
    }

    /// `Ab` scanned with `output` as the plan's output, and pruned.
    /// How many columns and rows the first batch has, and its first column.
    fn pruned(output: &[usize]) -> (u32, u32, StdVec<i64>) {
        let table = DynScannable::new(&Heap, Ab).unwrap();
        let mut plan = LogicalPlan::new(&Heap).unwrap();
        let mut columns = SlowVec::fixed(&Heap, 2).unwrap();
        let mut bindings = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64).unwrap();
            assert!(columns.push(ScanColumn { column, binding }).is_ok());
            bindings.push(binding);
        }
        let scan = DynOp::new(&Heap, ScanOp { scannable: &table, columns }).unwrap();
        plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
        for &i in output {
            assert!(plan.output.push(bindings[i]).is_ok());
        }
        crate::optimize::optimize(&Heap, &mut plan).unwrap();

        let physical = lower(&Heap, &plan).unwrap();
        let mut execution = physical.pipeline().start(&Heap).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch).unwrap());
        let first = (batch.column_count() > 0).then(|| batch.column(0).int64s().into());
        (batch.column_count(), batch.row_count(), first.unwrap_or_default())
    }

    #[test]
    fn scans_only_needed_columns() {
        assert_eq!(pruned(&[1]), (1, 2, [10, 20].into()));
        assert_eq!(pruned(&[1, 0]), (2, 2, [1, 2].into()));
        // With no columns, the rows are still there, to count.
        assert_eq!(pruned(&[]), (0, 2, [].into()));
    }

    /// One batch of four rows: `a` is 1, 2, null and 0, `b` 10, 20, 30 and
    /// null.
    struct Nullable;

    impl Scannable for Nullable {
        type State = bool;

        fn column_count(&self) -> u32 {
            2
        }

        fn column_name(&self, column: u32) -> &str {
            ["a", "b"][column as usize]
        }

        fn column_type(&self, _: u32) -> DataType {
            DataType::Int64
        }

        fn new_state(&self, _: &mut Context) -> Result<bool, Error> {
            Ok(false)
        }

        fn next(
            &self,
            columns: &[u32],
            _: &mut Context,
            done: &mut bool,
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            if *done {
                return Ok(false);
            }
            batch.reset(4);
            for &column in columns {
                let cells =
                    [[Some(1), Some(2), None, Some(0)], [Some(10), Some(20), Some(30), None]];
                let mut values = Buffer::allocate(&Heap, 32).unwrap();
                let mut validity = Buffer::allocate(&Heap, 1).unwrap();
                for (row, cell) in cells[column as usize].iter().enumerate() {
                    values.as_mut_slice::<i64>()[row] = cell.unwrap_or(0);
                    validity.as_mut_slice::<u8>()[0] |= u8::from(cell.is_some()) << row;
                }
                let column = ColumnView::new(DataType::Int64, values, Some(validity));
                assert!(batch.push_column(column).is_ok());
            }
            *done = true;
            Ok(true)
        }
    }

    /// `WHERE a > 1 OR b IS NULL`, with only `b` in the output, pruned: the
    /// rows kept, as whether `b` is null and its value, and how many columns
    /// batches have.
    #[test]
    fn filters_and_keeps_the_columns_it_reads() {
        let table = DynScannable::new(&Heap, Nullable).unwrap();
        let mut plan = LogicalPlan::new(&Heap).unwrap();
        let mut columns = SlowVec::fixed(&Heap, 2).unwrap();
        let mut bindings = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64).unwrap();
            assert!(columns.push(ScanColumn { column, binding }).is_ok());
            bindings.push(binding);
        }
        let scan = DynOp::new(&Heap, ScanOp { scannable: &table, columns }).unwrap();
        let scan = plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
        let (a, b) = (bindings[0].id, bindings[1].id);
        let greater =
            Leaf::Compare { column: a, comparison: Comparison::Greater, value: Value::Int64(1) };
        let nodes = [Node::Leaf(greater), Node::Leaf(Leaf::IsNull { column: b }), Node::Or(0, 1)];
        let predicate = Predicate::new(SlowVec::fixed_from(&Heap, nodes.into_iter()).unwrap());
        let filter = DynOp::new(&Heap, FilterOp { predicate }).unwrap();
        plan.add_node(filter, SlowVec::fixed_from(&Heap, [scan].into_iter()).unwrap()).unwrap();
        assert!(plan.output.push(bindings[1]).is_ok());
        crate::optimize::optimize(&Heap, &mut plan).unwrap();

        let physical = lower(&Heap, &plan).unwrap();
        let mut execution = physical.pipeline().start(&Heap).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch).unwrap());
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
        assert!(!execution.next(&mut batch).unwrap());
    }

    /// As many columns as it holds, with no rows.
    struct Wide(u32);

    impl Scannable for Wide {
        type State = ();

        fn column_count(&self) -> u32 {
            self.0
        }

        fn column_name(&self, _: u32) -> &'static str {
            "c"
        }

        fn column_type(&self, _: u32) -> DataType {
            DataType::Int64
        }

        fn new_state(&self, _: &mut Context) -> Result<(), Error> {
            Ok(())
        }

        fn next(
            &self,
            _: &[u32],
            _: &mut Context,
            (): &mut (),
            _: &mut RowBatch,
        ) -> Result<bool, Error> {
            Ok(false)
        }
    }

    #[test]
    fn fails_to_lower_batches_too_wide() {
        let table = DynScannable::new(&Heap, Wide(70)).unwrap();
        let plan = |output: usize| {
            let mut plan = LogicalPlan::new(&Heap).unwrap();
            let mut columns = SlowVec::fixed(&Heap, 70).unwrap();
            for column in 0..70 {
                let binding = plan.add_column("c", DataType::Int64).unwrap();
                assert!(columns.push(ScanColumn { column, binding }).is_ok());
                if (column as usize) < output {
                    assert!(plan.output.push(binding).is_ok());
                }
            }
            let scan = DynOp::new(&Heap, ScanOp { scannable: &table, columns }).unwrap();
            plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
            plan
        };
        assert_eq!(lower(&Heap, &plan(70)).err(), Some(LowerError::TooManyColumns));
        // Pruned to the two in the output, it fits.
        let mut narrow = plan(2);
        crate::optimize::optimize(&Heap, &mut narrow).unwrap();
        assert!(lower(&Heap, &narrow).is_ok());
    }
}
