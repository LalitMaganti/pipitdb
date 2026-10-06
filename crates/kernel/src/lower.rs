//! `lower`: a `LogicalPlan` to a `PhysicalPlan`, a pipeline ready to run.

use crate::allocator::{AllocError, Allocator};
use crate::names::{Name, Names};
use crate::pipeline::Pipeline;
use crate::plan::{ColumnId, LogicalPlan, PLAN_NAME_BYTES_MAX, PlanNodeId};
use crate::row_batch::BATCH_COLUMNS_MAX;
use crate::slow_vec::{Full, SlowVec};
use crate::step::{DynSource, Step};

/// Why a plan couldn't be lowered.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LowerError {
    /// An allocation failed.
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
    /// The plan being lowered.
    plan: &'p LogicalPlan<'c>,
    /// What steps are made with.
    allocator: &'p dyn Allocator,
    /// The pipeline's source, once a node sets it.
    source: Option<DynSource<'c>>,
    /// The steps after it, in order.
    steps: SlowVec<Step<'c>>,
    /// Each column's position in batches, or `u32::MAX` until defined.
    positions: SlowVec<u32>,
    /// How many columns batches have.
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
    use alloc::vec;
    use alloc::vec::Vec as StdVec;
    use core::cell::RefCell;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::{ColumnView, DataType, Forms};
    use crate::context::Context;
    use crate::error::Error;
    use crate::filter::{Comparison, Value};
    use crate::optimize::{Needed, choose_forms};
    use crate::plan::{ColumnId, DynOp, FilterOp, Op, PlanNode, ScanColumn, ScanOp};
    use crate::predicate::{Leaf, Node, Predicate};
    use crate::query_allocators::QueryAllocators;
    use crate::row_batch::RowBatch;
    use crate::scannable::{DynScannable, Scannable};
    use crate::selection::{Kept, Selection};

    /// One batch of two rows: column `a` holds 1 and 2, `b` 10 and 20.
    struct Ab;

    impl Scannable for Ab {
        type State = bool;
        type Loader = ();

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
            _: &[Forms],
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
                assert!(
                    batch
                        .push_column(
                            ColumnView::new(
                                &mut Context::new(&Heap),
                                DataType::Int64,
                                values,
                                None
                            )
                            .unwrap()
                        )
                        .is_ok()
                );
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
        let a_column = ScanColumn { column: 0, binding: a, forms: Forms::FLAT };
        let b_column = ScanColumn { column: 1, binding: b, forms: Forms::FLAT };
        assert!(columns.push(a_column).is_ok() && columns.push(b_column).is_ok());
        let scan = DynOp::new(&Heap, ScanOp::new(&table, columns)).unwrap();
        plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
        assert!(plan.output.push(b).is_ok() && plan.output.push(a).is_ok());

        let physical = lower(&Heap, &plan).unwrap();
        let names: StdVec<&str> = physical.columns().iter().map(|&c| physical.name(c)).collect();
        assert_eq!(names, ["b", "a"]);

        let query = QueryAllocators::new(&Heap);
        let mut execution = physical.pipeline().start(&query).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch).unwrap());
        let rows: StdVec<&[i64]> =
            physical.columns().iter().map(|c| batch.column(c.position).int64s()).collect();
        assert_eq!(rows, [[10, 20], [1, 2]]);
        assert!(!execution.next(&mut batch).unwrap());
    }

    /// `Ab` scanned with `output` as the plan's output, and pruned.
    fn pruned(output: &[usize]) -> (usize, StdVec<i64>) {
        let table = DynScannable::new(&Heap, Ab).unwrap();
        let mut plan = LogicalPlan::new(&Heap).unwrap();
        let mut columns = SlowVec::fixed(&Heap, 2).unwrap();
        let mut bindings = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64).unwrap();
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            bindings.push(binding);
        }
        let scan = DynOp::new(&Heap, ScanOp::new(&table, columns)).unwrap();
        plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
        for &i in output {
            assert!(plan.output.push(bindings[i]).is_ok());
        }
        crate::optimize::optimize(&Heap, &mut plan).unwrap();

        let physical = lower(&Heap, &plan).unwrap();
        let query = QueryAllocators::new(&Heap);
        let mut execution = physical.pipeline().start(&query).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch).unwrap());
        (batch.column_count() as usize, batch.column(0).int64s().into())
    }

    #[test]
    fn scans_only_needed_columns() {
        assert_eq!(pruned(&[1]), (1, [10, 20].into()));
        assert_eq!(pruned(&[1, 0]), (2, [1, 2].into()));
        // A batch with no columns has no rows, so one is kept.
        assert_eq!(pruned(&[]), (1, [1, 2].into()));
    }

    /// One batch of four rows: `a` is 1, 2, null and 0, `b` 10, 20, 30 and
    /// null.
    struct Nullable;

    impl Scannable for Nullable {
        type State = bool;
        type Loader = ();

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
            _: &[Forms],
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
                let column = ColumnView::new(
                    &mut Context::new(&Heap),
                    DataType::Int64,
                    values,
                    Some(validity),
                )
                .unwrap();
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
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            bindings.push(binding);
        }
        let scan = DynOp::new(&Heap, ScanOp::new(&table, columns)).unwrap();
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
        let query = QueryAllocators::new(&Heap);
        let mut execution = physical.pipeline().start(&query).unwrap();
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
        type Loader = ();

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
            _: &[Forms],
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
                assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
                if (column as usize) < output {
                    assert!(plan.output.push(binding).is_ok());
                }
            }
            let scan = DynOp::new(&Heap, ScanOp::new(&table, columns)).unwrap();
            plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
            plan
        };
        assert_eq!(lower(&Heap, &plan(70)).err(), Some(LowerError::TooManyColumns));
        // Pruned to the two in the output, it fits.
        let mut narrow = plan(2);
        crate::optimize::optimize(&Heap, &mut narrow).unwrap();
        assert!(lower(&Heap, &narrow).is_ok());
    }

    /// Three batches of four rows of `a`, `b` and `c`: row `r` of batch `n`
    /// holds `100 * k + 10 * n + r` in column `k`. Its columns can be lazy if
    /// `lazy`, with handles saying which column and batch; each load is noted
    /// in `loads`, as the column, the batch and the rows it read.
    struct Counted<'a> {
        lazy: bool,
        loads: &'a RefCell<StdVec<(u8, u8, StdVec<u16>)>>,
    }

    impl Scannable for Counted<'_> {
        type State = u8;
        type Loader = ();

        fn column_count(&self) -> u32 {
            3
        }

        fn column_name(&self, column: u32) -> &str {
            ["a", "b", "c"][column as usize]
        }

        fn column_type(&self, _: u32) -> DataType {
            DataType::Int64
        }

        fn new_state(&self, _: &mut Context) -> Result<u8, Error> {
            Ok(0)
        }

        fn next(
            &self,
            columns: &[u32],
            forms: &[Forms],
            context: &mut Context,
            batches: &mut u8,
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            if *batches == 3 {
                return Ok(false);
            }
            batch.reset(4);
            for (i, &column) in columns.iter().enumerate() {
                let column = if forms[i].contains(Forms::LAZY) {
                    let handle = [u8::try_from(column).unwrap(), *batches];
                    ColumnView::lazy(context, DataType::Int64, &handle, 4).unwrap()
                } else {
                    let mut values = context.values_buffer(4 * 8).unwrap();
                    for (row, value) in (0..).zip(values.as_mut_slice::<i64>()) {
                        *value = 100 * i64::from(column) + 10 * i64::from(*batches) + row;
                    }
                    ColumnView::new(context, DataType::Int64, values, None).unwrap()
                };
                assert!(batch.push_column(column).is_ok());
            }
            *batches += 1;
            Ok(true)
        }

        fn forms(&self, _: u32) -> Forms {
            if self.lazy { Forms::LAZY } else { Forms::FLAT }
        }

        fn new_loader(&self, _: &mut Context) -> Result<(), Error> {
            Ok(())
        }

        fn load(
            &self,
            context: &mut Context,
            (): &mut (),
            lazy: &ColumnView,
            selection: &Selection,
        ) -> Result<ColumnView, Error> {
            let (handle, start) = lazy.handle();
            let (column, batch) = (handle[0], handle[1]);
            let rows: StdVec<u16> = match selection.kept() {
                Kept::All => (0..4).collect(),
                Kept::None => StdVec::new(),
                Kept::Select(rows) => rows.into(),
            };
            let mut values = context.values_buffer(4 * 8).unwrap();
            let out = values.as_mut_slice::<i64>();
            for &row in &rows {
                let at = i64::from(start) + i64::from(row);
                out[usize::from(row)] = 100 * i64::from(column) + 10 * i64::from(batch) + at;
            }
            self.loads.borrow_mut().push((column, batch, rows));
            Ok(ColumnView::new(context, DataType::Int64, values, None).unwrap())
        }
    }

    /// Runs `Counted` through `WHERE`s that each keep rows with a column
    /// less than or greater than a value, with `output` as the plan's output,
    /// optimized: the kept rows of the output, and how many rows there were.
    fn counted(
        table: &DynScannable,
        filters: &[(usize, Comparison, i64)],
        output: &[usize],
    ) -> (StdVec<StdVec<i64>>, u32) {
        let mut plan = LogicalPlan::new(&Heap).unwrap();
        let mut columns = SlowVec::fixed(&Heap, 3).unwrap();
        let mut bindings = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b"), (2, "c")] {
            let binding = plan.add_column(name, DataType::Int64).unwrap();
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            bindings.push(binding);
        }
        let scan = DynOp::new(&Heap, ScanOp::new(table, columns)).unwrap();
        let mut node = plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
        for &(column, comparison, value) in filters {
            let column = bindings[column].id;
            let leaf = Leaf::Compare { column, comparison, value: Value::Int64(value) };
            let nodes = SlowVec::fixed_from(&Heap, [Node::Leaf(leaf)].into_iter()).unwrap();
            let filter = DynOp::new(&Heap, FilterOp { predicate: Predicate::new(nodes) }).unwrap();
            let children = SlowVec::fixed_from(&Heap, [node].into_iter()).unwrap();
            node = plan.add_node(filter, children).unwrap();
        }
        for &i in output {
            assert!(plan.output.push(bindings[i]).is_ok());
        }
        crate::optimize::optimize(&Heap, &mut plan).unwrap();

        let physical = lower(&Heap, &plan).unwrap();
        let query = QueryAllocators::new(&Heap);
        let mut execution = physical.pipeline().start(&query).unwrap();
        let mut batch = RowBatch::new();
        let (mut kept, mut rows) = (StdVec::new(), 0);
        while execution.next(&mut batch).unwrap() {
            rows += batch.row_count();
            let selected: StdVec<u16> = match batch.selection().kept() {
                Kept::All => (0..4).collect(),
                Kept::None => StdVec::new(),
                Kept::Select(rows) => rows.into(),
            };
            for row in selected {
                let columns = physical.columns().iter().map(|c| batch.column(c.position));
                kept.push(columns.map(|column| column.int64s()[usize::from(row)]).collect());
            }
        }
        (kept, rows)
    }

    #[test]
    fn loads_lazy_columns_where_read_for_the_rows_kept() {
        let loads = RefCell::new(StdVec::new());
        let table = DynScannable::new(&Heap, Counted { lazy: true, loads: &loads }).unwrap();
        // `a > 11` keeps rows 2 and 3 of batch 1, and batch 2; `b < 122` then
        // drops rows 2 and 3 of batch 2.
        let filters = [(0, Comparison::Greater, 11), (1, Comparison::Less, 122)];
        let (kept, _) = counted(&table, &filters, &[2, 0]);
        assert_eq!(kept, [[212, 12], [213, 13], [220, 20], [221, 21]]);
        // `a` is read by the scan's parent, so it isn't lazy. `b` is loaded
        // for the second filter, and `c` for the output, only for the batches
        // and rows still kept.
        let b = [(1, 1, vec![2, 3]), (1, 2, vec![0, 1, 2, 3])];
        let c = [(2, 1, vec![2, 3]), (2, 2, vec![0, 1])];
        assert_eq!(*loads.borrow(), [b[0].clone(), c[0].clone(), b[1].clone(), c[1].clone()]);
    }

    #[test]
    fn never_loads_columns_nothing_reads() {
        let loads = RefCell::new(StdVec::new());
        let table = DynScannable::new(&Heap, Counted { lazy: true, loads: &loads }).unwrap();
        // With no output, a column is still scanned, for its rows.
        assert_eq!(counted(&table, &[], &[]), (vec![StdVec::new(); 12], 12));
        assert!(loads.borrow().is_empty());
    }

    #[test]
    fn reads_eagerly_from_tables_that_cant_be_lazy() {
        let loads = RefCell::new(StdVec::new());
        let table = DynScannable::new(&Heap, Counted { lazy: false, loads: &loads }).unwrap();
        let filters = [(0, Comparison::Greater, 11), (1, Comparison::Less, 122)];
        let (kept, _) = counted(&table, &filters, &[2, 0]);
        assert_eq!(kept, [[212, 12], [213, 13], [220, 20], [221, 21]]);
        assert!(loads.borrow().is_empty());
    }

    /// Reads one column, taking it in `accepts`; it's never lowered.
    struct Reads {
        column: ColumnId,
        accepts: Forms,
    }

    impl<'c> Op<'c> for Reads {
        fn lower(&self, _: &PlanNode<'c>, _: &mut Lowering<'_, 'c>) -> Result<(), LowerError> {
            crate::check::check_failed(line!())
        }

        fn reads(&self, reads: &mut Needed) {
            reads.need(self.column);
        }

        fn accepts(&self, column: ColumnId) -> Forms {
            if column == self.column { self.accepts } else { Forms::FLAT }
        }
    }

    /// How many nodes a plan has once its forms are chosen: a scan of
    /// `Counted`, a filter on `a`, and an op reading `b` in `accepts`.
    fn nodes_with(accepts: Forms) -> usize {
        let loads = RefCell::new(StdVec::new());
        let table = DynScannable::new(&Heap, Counted { lazy: true, loads: &loads }).unwrap();
        let mut plan = LogicalPlan::new(&Heap).unwrap();
        let mut columns = SlowVec::fixed(&Heap, 2).unwrap();
        let mut bindings = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64).unwrap();
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            bindings.push(binding);
        }
        let scan = DynOp::new(&Heap, ScanOp::new(&table, columns)).unwrap();
        let scan = plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
        let leaf = Leaf::Compare {
            column: bindings[0].id,
            comparison: Comparison::Greater,
            value: Value::Int64(0),
        };
        let nodes = SlowVec::fixed_from(&Heap, [Node::Leaf(leaf)].into_iter()).unwrap();
        let filter = DynOp::new(&Heap, FilterOp { predicate: Predicate::new(nodes) }).unwrap();
        let filter = plan.add_node(filter, SlowVec::fixed_from(&Heap, [scan].into_iter()).unwrap());
        let reads = DynOp::new(&Heap, Reads { column: bindings[1].id, accepts }).unwrap();
        let children = SlowVec::fixed_from(&Heap, [filter.unwrap()].into_iter()).unwrap();
        plan.add_node(reads, children).unwrap();
        choose_forms(&Heap, &mut plan).unwrap();
        plan.nodes.len()
    }

    #[test]
    fn loads_lazy_columns_only_for_readers_that_dont_take_them_lazy() {
        // `b` passes the filter before it's read, so it's made lazy: loaded
        // below a reader that takes it flat, by one more node, and handed as
        // it is to one that takes it lazy.
        assert_eq!(nodes_with(Forms::FLAT), 4);
        assert_eq!(nodes_with(Forms::LAZY), 3);
    }
}
