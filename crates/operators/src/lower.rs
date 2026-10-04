//! `lower`: a `LogicalPlan` to a `PhysicalPlan`, a pipeline ready to run.

use pipit_kernel::allocator::{AllocError, Allocator};
use pipit_kernel::pipeline::Pipeline;
use pipit_kernel::step::DynSource;
use pipit_kernel::vec::Vec;

use crate::names::{Name, Names};
use crate::plan::{LogicalPlan, Op, PLAN_NAME_BYTES_MAX};
use crate::table::TableScan;

/// A column of a plan's result: its name, and its position in the batches
/// the pipeline makes.
#[derive(Clone, Copy)]
pub struct OutputColumn {
    pub name: Name,
    pub position: u32,
}

/// Read-only once built, so it can be run any number of times.
pub struct PhysicalPlan<'t> {
    pipeline: Pipeline<'t>,
    names: Names,
    columns: Vec<OutputColumn>,
}

impl<'t> PhysicalPlan<'t> {
    pub fn pipeline(&self) -> &Pipeline<'t> {
        &self.pipeline
    }

    pub fn columns(&self) -> &[OutputColumn] {
        &self.columns
    }

    pub fn name(&self, column: OutputColumn) -> &str {
        self.names.get(column.name)
    }
}

/// Builds the pipeline for `plan`, mapping each column id to its position in
/// the batches as steps are added.
pub fn lower<'t, A: Allocator + Clone + 'static>(
    allocator: A,
    plan: &LogicalPlan<'t>,
) -> Result<PhysicalPlan<'t>, AllocError> {
    let mut positions = Vec::fixed(allocator.clone(), plan.columns.len())?;
    for _ in 0..plan.columns.len() {
        positions.push(u32::MAX)?;
    }
    let node = at!(plan.nodes, plan.root as usize);
    let source = match &node.op {
        Op::Scan { table, columns } => {
            // Reads every column, in the table's order.
            for (position, column) in (0..).zip(columns.iter()) {
                *at_mut!(positions, column.id as usize) = position;
            }
            let all = Vec::fixed_from(allocator.clone(), 0..table.column_count())?;
            DynSource::new(allocator.clone(), TableScan::new(table, all))?
        }
    };

    let mut names = Names::new(allocator.clone(), PLAN_NAME_BYTES_MAX)?;
    let mut output = Vec::fixed(allocator.clone(), plan.output.len())?;
    for column in plan.output.iter() {
        let position = *at!(positions, column.id as usize);
        check!(position != u32::MAX);
        let name = names.add(plan.names.get(column.name))?;
        output.push(OutputColumn { name, position })?;
    }
    let pipeline = Pipeline::new(source, Vec::fixed(allocator, 0)?);
    Ok(PhysicalPlan { pipeline, names, columns: output })
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec::Vec as StdVec;

    use pipit_kernel::allocator::Heap;
    use pipit_kernel::buffer::Buffer;
    use pipit_kernel::column::{ColumnView, DataType};
    use pipit_kernel::row_batch::RowBatch;

    use super::*;
    use crate::table::Table;

    fn int64s(values: &[i64]) -> ColumnView {
        let mut buffer = Buffer::allocate(Heap, values.len() * 8).unwrap();
        buffer.as_mut_slice::<i64>().copy_from_slice(values);
        ColumnView::new(DataType::Int64, buffer, None)
    }

    #[test]
    fn lowers_a_scan_with_its_output_in_order() {
        let columns = [int64s(&[1, 2]), int64s(&[10, 20])];
        let schema = [("a", DataType::Int64), ("b", DataType::Int64)];
        let table = Table::new(Heap, &schema, &[&columns]).unwrap();

        let mut plan = LogicalPlan::new(Heap).unwrap();
        let a = plan.add_column("a", DataType::Int64).unwrap();
        let b = plan.add_column("b", DataType::Int64).unwrap();
        let mut bindings = Vec::fixed(Heap, 2).unwrap();
        assert!(bindings.push(a).is_ok());
        assert!(bindings.push(b).is_ok());
        let scan = Op::Scan { table: &table, columns: bindings };
        plan.add_node(scan, Vec::fixed(Heap, 0).unwrap()).unwrap();
        assert!(plan.output.push(b).is_ok());
        assert!(plan.output.push(a).is_ok());

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
