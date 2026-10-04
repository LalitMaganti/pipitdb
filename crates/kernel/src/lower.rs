//! `lower`: a `LogicalPlan` to a `PhysicalPlan`, a pipeline ready to run.

use crate::allocator::{AllocError, Allocator, DynAllocator};
use crate::names::{Name, Names};
use crate::pipeline::Pipeline;
use crate::plan::{ColumnId, LogicalPlan, PLAN_NAME_BYTES_MAX, PlanNodeId};
use crate::step::{DynSource, Step};
use crate::vec::Vec;

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
    columns: Vec<OutputColumn>,
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
    allocator: DynAllocator,
    source: Option<DynSource<'c>>,
    steps: Vec<Step<'c>>,
    /// Each column's position in batches, or `u32::MAX` until defined.
    positions: Vec<u32>,
    column_count: u32,
}

impl<'c> Lowering<'_, 'c> {
    /// What to make steps with.
    pub fn allocator(&self) -> DynAllocator {
        self.allocator.clone()
    }

    /// Lowers `node`, such as a child of the node being lowered.
    pub fn lower(&mut self, node: PlanNodeId) -> Result<(), AllocError> {
        let node = at!(self.plan.nodes, node as usize);
        node.op.lower(node, self)
    }

    /// Makes `source` the pipeline's.
    pub fn set_source(&mut self, source: DynSource<'c>) {
        check!(self.source.is_none());
        self.source = Some(source);
    }

    pub fn add_step(&mut self, step: Step<'c>) -> Result<(), AllocError> {
        Ok(self.steps.push(step)?)
    }

    /// Gives `column` the next position in batches.
    pub fn define(&mut self, column: ColumnId) {
        *at_mut!(self.positions, column as usize) = self.column_count;
        self.column_count += 1;
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
pub fn lower<'c, A: Allocator + Clone + 'static>(
    allocator: A,
    plan: &LogicalPlan<'c>,
) -> Result<PhysicalPlan<'c>, AllocError> {
    let allocator = DynAllocator::new(allocator)?;
    let unset = core::iter::repeat_n(u32::MAX, plan.columns.len());
    let positions = Vec::fixed_from(allocator.clone(), unset)?;
    let mut lowering = Lowering {
        plan,
        allocator: allocator.clone(),
        source: None,
        steps: Vec::new(allocator.clone(), LOWERED_STEPS_MAX)?,
        positions,
        column_count: 0,
    };
    lowering.lower(plan.root)?;

    let mut names = Names::new(allocator.clone(), PLAN_NAME_BYTES_MAX)?;
    let mut columns = Vec::fixed(allocator, plan.output.len())?;
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
    use crate::plan::{DynOp, ScanOp};
    use crate::row_batch::RowBatch;
    use crate::scannable::{DynScannable, Scannable};

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

        fn new_state(&self) -> bool {
            false
        }

        fn next(&self, columns: &[u32], batch: &mut RowBatch, done: &mut bool) -> bool {
            if *done {
                return false;
            }
            batch.reset(2);
            for &column in columns {
                let mut values = Buffer::allocate(Heap, 16).unwrap();
                let first = [1, 10][column as usize];
                values.as_mut_slice::<i64>().copy_from_slice(&[first, first * 2]);
                assert!(batch.push_column(ColumnView::new(DataType::Int64, values, None)).is_ok());
            }
            *done = true;
            true
        }
    }

    #[test]
    fn lowers_a_scan_with_its_output_in_order() {
        let table = DynScannable::new(Heap, Ab).unwrap();
        let mut plan = LogicalPlan::new(Heap).unwrap();
        let a = plan.add_column("a", DataType::Int64).unwrap();
        let b = plan.add_column("b", DataType::Int64).unwrap();
        let mut columns = Vec::fixed(Heap, 2).unwrap();
        assert!(columns.push(a).is_ok() && columns.push(b).is_ok());
        let scan = DynOp::new(Heap, ScanOp { scannable: &table, columns }).unwrap();
        plan.add_node(scan, Vec::fixed(Heap, 0).unwrap()).unwrap();
        assert!(plan.output.push(b).is_ok() && plan.output.push(a).is_ok());

        let physical = lower(Heap, &plan).unwrap();
        let names: StdVec<&str> = physical.columns().iter().map(|&c| physical.name(c)).collect();
        assert_eq!(names, ["b", "a"]);

        let mut execution = physical.pipeline().start(Heap).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch));
        let rows: StdVec<&[i64]> =
            physical.columns().iter().map(|c| batch.column(c.position).int64s()).collect();
        assert_eq!(rows, [[10, 20], [1, 2]]);
        assert!(!execution.next(&mut batch));
    }
}
