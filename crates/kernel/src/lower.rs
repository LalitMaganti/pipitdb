//! `lower`: a `LogicalPlan` to a `PhysicalPlan`, a pipeline ready to run.
//! Planning decides first, node by node, what each needs of its inputs and
//! what it outputs; lowering then only adds the steps that does.

use crate::allocator::AllocError;
use crate::allocator::Allocator;
use crate::column::DataType;
use crate::names::{Name, Names};
use crate::output_properties::{InputStep, OutputProperties, Planned, plan_nodes, step};
use crate::pipeline::Pipeline;
use crate::plan::{ColumnId, LogicalPlan, PLAN_NAME_BYTES_MAX, PlanNode, PlanNodeId};
use crate::slow_vec::SlowVec;
use crate::sort::sort;
use crate::step::{DynSource, Step};
use crate::top_k::top_k;

/// Why a plan couldn't be lowered.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LowerError {
    /// An allocation failed.
    OutOfMemory,
    /// A batch would need more than `BATCH_COLUMNS_MAX` columns.
    TooManyColumns,
    /// An op reads a column of a type it doesn't take.
    WrongType(ColumnId),
}

impl From<AllocError> for LowerError {
    fn from(_: AllocError) -> LowerError {
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

/// A key rows are sorted by.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SortKey {
    pub column: ColumnId,
    pub descending: bool,
}

/// Which rows the steps added so far output.
#[derive(Clone, Copy)]
enum Rows {
    /// A node's output.
    Output(PlanNodeId),
    /// One of a node's inputs, once its steps have run.
    Input(PlanNodeId, u32),
}

/// A pipeline being built from a planned plan, which each operation adds
/// to.
pub struct Lowering<'p, 'c> {
    /// The plan being lowered.
    plan: &'p LogicalPlan<'c>,
    /// What planning decided of each node.
    planned: &'p Planned,
    /// What steps are made with.
    allocator: &'p dyn Allocator,
    /// The pipeline's source, once a node sets it.
    source: Option<DynSource<'c>>,
    /// The steps after it, in order.
    steps: SlowVec<Step<'c>>,
    /// Which rows the steps so far output.
    rows: Rows,
}

/// Its methods are named by what they do. Building ones add to the
/// pipeline, and `query_` ones only read what's known.
impl<'p, 'c> Lowering<'p, 'c> {
    /// What to make steps with.
    pub fn allocator(&self) -> &'p dyn Allocator {
        self.allocator
    }

    fn lower_node(&mut self, id: PlanNodeId) -> Result<(), LowerError> {
        let node = at!(self.plan.nodes, id as usize);
        self.rows = Rows::Output(id);
        node.op.lower(node, self)?;
        self.rows = Rows::Output(id);
        Ok(())
    }

    /// Lowers input `input` of `node`, the node being lowered, then the
    /// steps planning chose to meet its needs.
    pub fn lower_input(&mut self, node: &PlanNode<'c>, input: u32) -> Result<(), LowerError> {
        let id = self.id_of(node);
        self.lower_node(*at!(node.children, input as usize))?;
        self.rows = Rows::Input(id, input);
        let Some(planned) = at!(self.planned, id as usize) else { failed!() };
        for step in at!(planned.inputs, input as usize).steps.iter() {
            self.apply(step)?;
        }
        Ok(())
    }

    fn apply(&mut self, step: &InputStep) -> Result<(), LowerError> {
        let InputStep::Sort { keys, keep, publish } = step;
        let keys = keys.iter().map(|key| crate::sort::SortKey {
            column: self.query_position(key.column),
            descending: key.descending,
        });
        let keys = SlowVec::fixed_from(self.allocator, keys)?;
        let operator = match *keep {
            Some(count) => top_k(self.allocator, keys, count, *publish)?,
            None => sort(self.allocator, keys)?,
        };
        self.add_operator(Step::Operator(operator))
    }

    /// How many of its first rows the open sort just below outputs, if it
    /// outputs only those.
    pub fn query_sort_keep(&self) -> Option<u32> {
        let sort = self.rows().open_sort?;
        let InputStep::Sort { keep, .. } = step(self.planned, sort.at);
        *keep
    }

    /// For `node`, a scan, the column a `TopK` above publishes a cutoff on,
    /// if any.
    pub fn query_cutoff(&self, node: &PlanNode<'c>) -> Option<SortKey> {
        let Some(planned) = at!(self.planned, self.id_of(node) as usize) else { failed!() };
        planned.cutoff
    }

    fn id_of(&self, node: &PlanNode<'c>) -> PlanNodeId {
        let Some(id) = self.plan.nodes.iter().position(|n| core::ptr::eq(n, node)) else {
            failed!()
        };
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
        let id = id as PlanNodeId;
        id
    }

    fn rows(&self) -> &OutputProperties {
        let (id, input) = match self.rows {
            Rows::Output(id) => (id, None),
            Rows::Input(id, input) => (id, Some(input)),
        };
        let Some(planned) = at!(self.planned, id as usize) else { failed!() };
        match input {
            None => &planned.output,
            Some(input) => &at!(planned.inputs, input as usize).rows,
        }
    }

    /// Makes `source` the pipeline's.
    pub fn set_source(&mut self, source: DynSource<'c>) {
        check!(self.source.is_none());
        self.source = Some(source);
    }

    /// Adds `step`.
    pub fn add_operator(&mut self, step: Step<'c>) -> Result<(), LowerError> {
        Ok(self.steps.push(step)?)
    }

    /// What `column` holds.
    pub fn query_type(&self, column: ColumnId) -> DataType {
        at!(self.plan.columns, column as usize).data_type
    }

    /// Where `column` is in batches.
    pub fn query_position(&self, column: ColumnId) -> u32 {
        let Some(position) = self.rows().position_of(column) else { failed!() };
        position
    }
}

/// Plans `plan`, then builds its pipeline from its root, with memory from
/// `allocator`.
pub fn lower<'c>(
    allocator: &dyn Allocator,
    plan: &LogicalPlan<'c>,
) -> Result<PhysicalPlan<'c>, LowerError> {
    let planned = plan_nodes(allocator, plan)?;
    let mut lowering = Lowering {
        plan,
        planned: &planned,
        allocator,
        source: None,
        steps: SlowVec::new(allocator, LOWERED_STEPS_MAX)?,
        rows: Rows::Output(plan.root),
    };
    lowering.lower_node(plan.root)?;

    let mut names = Names::new(allocator, PLAN_NAME_BYTES_MAX)?;
    let mut columns = SlowVec::fixed(allocator, plan.output.len())?;
    for column in plan.output.iter() {
        let name = names.add(plan.names.get(column.name))?;
        columns.push(OutputColumn { name, position: lowering.query_position(column.id) })?;
    }
    let Some(source) = lowering.source else { failed!() };
    Ok(PhysicalPlan { pipeline: Pipeline::new(source, lowering.steps), names, columns })
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec as StdVec;
    use core::cell::RefCell;

    use super::*;
    use crate::allocator::{AllocError, Allocator, Heap};
    use crate::buffer::Buffer;
    use crate::column::Bounds;
    use crate::column::{ColumnView, DataType, Forms};
    use crate::condition::{Condition, Cutoff};
    use crate::context::Context;
    use crate::error::Error;
    use crate::filter::{Comparison, Value};
    use crate::optimize::{Needed, choose_forms};
    use crate::output_properties::{Arrangements, DeriveContext, OutputProperties};
    use crate::plan::{ColumnId, DynOp, FilterOp, NamedColumn, Op, PlanNode, ScanColumn, ScanOp};
    use crate::predicate::{Leaf, Node, Predicate};
    use crate::query_allocators::QueryAllocators;
    use crate::row_batch::RowBatch;
    use crate::scannable::{DynScannable, Scannable, Summary};
    use crate::selection::{Kept, Selection};
    use crate::step::{DynOperator, DynSource};

    /// One batch of two rows: column `a` holds 1 and 2, `b` 10 and 20,
    /// declared sorted in the orders it holds.
    struct Ab(&'static [&'static [(u32, bool)]]);

    /// `Ab` sorted by `a` and then `b`.
    const AB: Ab = Ab(&[&[(0, false), (1, false)]]);

    impl Scannable for Ab {
        type State<'s>
            = (bool, &'s [(u32, Forms)])
        where
            Self: 's;
        type Loader = ();

        fn sorted_by(&self) -> &[&[(u32, bool)]] {
            self.0
        }

        fn column_count(&self) -> u32 {
            2
        }

        fn column_name(&self, column: u32) -> &str {
            ["a", "b"][column as usize]
        }

        fn column_type(&self, _: u32) -> DataType {
            DataType::Int64
        }

        fn open<'s>(
            &'s self,
            _: &mut Context,
            columns: &'s [(u32, Forms)],
        ) -> Result<(bool, &'s [(u32, Forms)]), Error> {
            Ok((false, columns))
        }

        fn next(
            &self,
            _: &mut Context,
            (done, columns): &mut (bool, &[(u32, Forms)]),
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            if *done {
                return Ok(false);
            }
            batch.reset(2);
            for &(column, _) in *columns {
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
        let table = DynScannable::new(&Heap, AB).unwrap();
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

    /// Keys to group by, and keys to sort by within groups, each with whether
    /// it descends.
    type Ask<'t> = (&'t [ColumnId], &'t [(ColumnId, bool)]);

    /// Lowers its child, then, if `breaker`, a breaker, and asks lowering
    /// which of `asks` hold: each is keys to group by and keys to sort by
    /// within groups, by column, a sort key's with whether it descends.
    struct Knows<'t> {
        breaker: bool,
        asks: &'t [Ask<'t>],
        known: &'t RefCell<StdVec<bool>>,
        sorted_by_b: &'t RefCell<usize>,
    }

    impl<'c> Op<'c> for Knows<'_> {
        fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
            let mut out = *c.input(0);
            if self.breaker {
                out.arrangements = Arrangements::NONE;
            }
            for &(keys, sorted_by) in self.asks {
                let sorted_by: StdVec<SortKey> = sorted_by
                    .iter()
                    .map(|&(column, descending)| SortKey { column, descending })
                    .collect();
                self.known.borrow_mut().push(out.is_arranged(keys, &sorted_by));
            }
            *self.sorted_by_b.borrow_mut() = out.sorted_by(&[1]).len();
            Ok(out)
        }

        fn lower(
            &self,
            node: &PlanNode<'c>,
            lowering: &mut Lowering<'_, 'c>,
        ) -> Result<(), LowerError> {
            lowering.lower_input(node, 0)?;
            if self.breaker {
                let operator = DynOperator::new(lowering.allocator(), Nothing)?;
                lowering.add_operator(Step::Operator(operator))?;
            }
            Ok(())
        }
    }

    /// Outputs nothing.
    struct Nothing;

    impl crate::step::Operator for Nothing {
        type State = ();

        fn new_state(&self, _: &mut Context) -> Result<(), Error> {
            Ok(())
        }

        fn execute(
            &self,
            _: &mut Context,
            (): &mut (),
            _: &RowBatch,
            _: &mut RowBatch,
        ) -> Result<crate::step::Progress, Error> {
            Ok(crate::step::Progress::NeedInput)
        }

        fn finish(
            &self,
            _: &mut Context,
            (): &mut (),
            _: &mut RowBatch,
        ) -> Result<crate::step::Progress, Error> {
            Ok(crate::step::Progress::NeedInput)
        }
    }

    /// Which of `asks` hold of `Ab`'s rows, scanning its columns `read`, after
    /// a breaker if `breaker`; and how many keys `b` alone sorts by.
    fn knows(read: &[u32], breaker: bool, asks: &[Ask]) -> (StdVec<bool>, usize) {
        knows_of(AB, read, breaker, asks)
    }

    /// As `knows`, of `table`.
    fn knows_of(table: Ab, read: &[u32], breaker: bool, asks: &[Ask]) -> (StdVec<bool>, usize) {
        let (known, sorted_by_b) = (RefCell::new(StdVec::new()), RefCell::new(0));
        let table = DynScannable::new(&Heap, table).unwrap();
        let mut plan = LogicalPlan::new(&Heap).unwrap();
        let bindings = [
            plan.add_column("a", DataType::Int64).unwrap(),
            plan.add_column("b", DataType::Int64).unwrap(),
        ];
        let mut columns = SlowVec::fixed(&Heap, read.len()).unwrap();
        for &column in read {
            let binding = bindings[column as usize];
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
        }
        let scan = DynOp::new(&Heap, ScanOp::new(&table, columns)).unwrap();
        let scan = plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
        let op = Knows { breaker, asks, known: &known, sorted_by_b: &sorted_by_b };
        let op = DynOp::new(&Heap, op).unwrap();
        plan.add_node(op, SlowVec::fixed_from(&Heap, [scan].into_iter()).unwrap()).unwrap();
        assert!(plan.output.push(bindings[read[0] as usize]).is_ok());
        lower(&Heap, &plan).unwrap();
        drop(plan);
        (known.into_inner(), sorted_by_b.into_inner())
    }

    #[test]
    fn knows_the_order_scans_say_rows_are_in() {
        // `Ab` is sorted by `a` and then `b`, so also grouped by `a`, and by
        // `a` and `b`, and sorted by `b` within groups of `a`; but not grouped
        // by `b`, nor sorted by `b` alone, nor descending.
        let asks: &[Ask] = &[
            (&[], &[]),
            (&[0], &[]),
            (&[0, 1], &[]),
            (&[1, 0], &[]),
            (&[0], &[(1, false)]),
            (&[], &[(0, false), (1, false)]),
            (&[1], &[]),
            (&[], &[(1, false)]),
            (&[], &[(0, true)]),
        ];
        let expected = [true, true, true, true, true, true, false, false, false];
        assert_eq!(knows(&[0, 1], false, asks), (expected.to_vec(), 0));
        // Without `a` read, nothing is known of `b`'s order.
        assert_eq!(knows(&[1], false, &asks[6..8]), ([false, false].to_vec(), 0));
        // A breaker's rows are new.
        assert_eq!(knows(&[0, 1], true, &asks[1..2]), ([false].to_vec(), 0));
        // Declared sorted by `b` alone too, it's known to be in both orders.
        let both = Ab(&[&[(0, false), (1, false)], &[(1, false)]]);
        assert_eq!(knows_of(both, &[0, 1], false, &asks[5..8]), ([true, true, true].to_vec(), 1));
    }

    #[test]
    fn keeps_the_start_of_sorts_too_long_to_hold() {
        let mut rows = OutputProperties::NONE;
        let key = |column| SortKey { column, descending: false };
        let sorted_by: StdVec<SortKey> = (0..10).map(key).collect();
        rows.arrangements.push(&[10], &sorted_by);
        assert!(rows.is_arranged(&[10], &sorted_by[..7]));
        assert!(!rows.is_arranged(&[10], &sorted_by[..8]));
        // Grouping by too many keys says nothing of grouping by fewer.
        let keys: StdVec<ColumnId> = (0..9).collect();
        rows.arrangements.push(&keys, &[]);
        assert!(!rows.is_arranged(&keys[..8], &[]));
    }

    /// The steps of `Ab` scanned, its rows counted by `groups` of its
    /// columns first, if given, then ordered by `keys`, by column, then
    /// limited to `(count, offset)` rows, if given.
    fn ordered(
        groups: Option<&[usize]>,
        keys: &[(usize, bool)],
        limit: Option<(u64, u64)>,
    ) -> alloc::string::String {
        let table = DynScannable::new(&Heap, AB).unwrap();
        let mut plan = LogicalPlan::new(&Heap).unwrap();
        let mut columns = SlowVec::fixed(&Heap, 2).unwrap();
        let mut bindings = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64).unwrap();
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            bindings.push(binding);
        }
        let scan = DynOp::new(&Heap, ScanOp::new(&table, columns)).unwrap();
        let mut input = plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
        if let Some(groups) = groups {
            let binding = plan.add_column("n", DataType::Int64).unwrap();
            let function = crate::aggregate::Function::CountStar;
            let aggregate =
                crate::aggregate::Aggregate { function, input: None, distinct: false, binding };
            let aggregate = crate::aggregate::AggregateOp {
                keys: SlowVec::fixed_from(&Heap, groups.iter().map(|&g| bindings[g].id)).unwrap(),
                aggregates: SlowVec::fixed_from(&Heap, [aggregate].into_iter()).unwrap(),
            };
            let aggregate = DynOp::new(&Heap, aggregate).unwrap();
            let children = SlowVec::fixed_from(&Heap, [input].into_iter()).unwrap();
            input = plan.add_node(aggregate, children).unwrap();
        }
        let sort_keys =
            keys.iter().map(|&(key, descending)| SortKey { column: bindings[key].id, descending });
        let order_by =
            crate::order_by::OrderByOp { keys: SlowVec::fixed_from(&Heap, sort_keys).unwrap() };
        let order_by = DynOp::new(&Heap, order_by).unwrap();
        let children = SlowVec::fixed_from(&Heap, [input].into_iter()).unwrap();
        let ordered = plan.add_node(order_by, children).unwrap();
        if let Some((count, offset)) = limit {
            let limit = DynOp::new(&Heap, crate::limit::LimitOp { count, offset }).unwrap();
            plan.add_node(limit, SlowVec::fixed_from(&Heap, [ordered].into_iter()).unwrap())
                .unwrap();
        }
        assert!(plan.output.push(bindings[keys.first().map_or(0, |key| key.0)]).is_ok());
        lower(&Heap, &plan).unwrap().pipeline().describe()
    }

    #[test]
    fn sorts_only_rows_not_known_to_be_in_order() {
        // `Ab` is sorted by `a` and then `b`.
        assert_eq!(ordered(None, &[(0, false)], None), "Scan");
        assert_eq!(ordered(None, &[(0, false), (1, false)], None), "Scan");
        assert_eq!(ordered(None, &[(1, false)], None), "Scan | Sort");
        assert_eq!(ordered(None, &[(0, true)], None), "Scan | Sort");
        // Groups come out in the order first seen, so in `a`'s order.
        assert_eq!(ordered(Some(&[0]), &[(0, false)], None), "Scan | HashAggregate");
        assert_eq!(ordered(Some(&[1]), &[(1, false)], None), "Scan | HashAggregate | Sort");
    }

    #[test]
    fn keeps_only_the_first_rows_a_limit_reads() {
        assert_eq!(ordered(None, &[(1, false)], Some((10, 0))), "Scan | TopK");
        assert_eq!(ordered(None, &[(1, false)], Some((10, 5))), "Scan | TopK | Limit");
        // Rows already in order are only limited.
        assert_eq!(ordered(None, &[(0, false)], Some((10, 0))), "Scan | Limit");
        let grouped = ordered(Some(&[1]), &[(1, true)], Some((1, 0)));
        assert_eq!(grouped, "Scan | HashAggregate | TopK");
    }

    #[test]
    fn drops_a_sort_a_later_sort_makes_pointless() {
        let table = DynScannable::new(&Heap, AB).unwrap();
        let mut plan = LogicalPlan::new(&Heap).unwrap();
        let mut columns = SlowVec::fixed(&Heap, 2).unwrap();
        let mut ids = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64).unwrap();
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            ids.push(binding);
        }
        let scan = DynOp::new(&Heap, ScanOp::new(&table, columns)).unwrap();
        let mut node = plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
        for (column, descending) in [(1, false), (0, true)] {
            let key = SortKey { column: ids[column].id, descending };
            let keys = SlowVec::fixed_from(&Heap, [key].into_iter()).unwrap();
            let order_by = DynOp::new(&Heap, crate::order_by::OrderByOp { keys }).unwrap();
            let children = SlowVec::fixed_from(&Heap, [node].into_iter()).unwrap();
            node = plan.add_node(order_by, children).unwrap();
        }
        assert!(plan.output.push(ids[0]).is_ok());
        assert_eq!(lower(&Heap, &plan).unwrap().pipeline().describe(), "Scan | Sort");
    }

    /// `Ab` scanned with `output` as the plan's output, and pruned.
    fn pruned(output: &[usize]) -> (usize, StdVec<i64>) {
        let table = DynScannable::new(&Heap, AB).unwrap();
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

    /// What a scan is given to skip by: its conditions' columns, and its
    /// cutoff's column and whether descending.
    type SkippedBy = (StdVec<u32>, Option<(u32, bool)>);

    /// One batch of four rows: `a` is 1, 2, null and 0, `b` 10, 20, 30 and
    /// null.
    struct Nullable;

    impl Scannable for Nullable {
        type State<'s>
            = (bool, &'s [(u32, Forms)])
        where
            Self: 's;
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

        fn open<'s>(
            &'s self,
            _: &mut Context,
            columns: &'s [(u32, Forms)],
        ) -> Result<(bool, &'s [(u32, Forms)]), Error> {
            Ok((false, columns))
        }

        fn next(
            &self,
            _: &mut Context,
            (done, columns): &mut (bool, &[(u32, Forms)]),
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            if *done {
                return Ok(false);
            }
            batch.reset(4);
            for &(column, _) in *columns {
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
        type State<'s>
            = ()
        where
            Self: 's;
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

        fn open(&self, _: &mut Context, _: &[(u32, Forms)]) -> Result<(), Error> {
            Ok(())
        }

        fn next(&self, _: &mut Context, (): &mut (), _: &mut RowBatch) -> Result<bool, Error> {
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
    /// in `loads`, as the column, the batch and the rows it read, and the
    /// columns of each scan's conditions in `conditions`. If `applies`, it
    /// says it applies conditions, but reads one row of 999s, which meets
    /// none.
    struct Counted<'a> {
        lazy: bool,
        loads: &'a RefCell<StdVec<(u8, u8, StdVec<u16>)>>,
        conditions: &'a RefCell<StdVec<SkippedBy>>,
    }

    impl Scannable for Counted<'_> {
        type State<'s>
            = (u8, &'s [(u32, Forms)])
        where
            Self: 's;
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

        fn open<'s>(
            &'s self,
            _: &mut Context,
            columns: &'s [(u32, Forms)],
        ) -> Result<(u8, &'s [(u32, Forms)]), Error> {
            Ok((0, columns))
        }

        fn next(
            &self,
            context: &mut Context,
            (batches, columns): &mut (u8, &[(u32, Forms)]),
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            if *batches == 3 {
                return Ok(false);
            }
            batch.reset(4);
            for &(column, forms) in *columns {
                let column = if forms.contains(Forms::LAZY) {
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

        /// Notes them, and reads with `open`.
        fn scan_within<'s>(
            &'s self,
            _: &dyn Allocator,
            _: &[(u32, Forms)],
            conditions: &[Condition],
            cutoff: Option<Cutoff>,
        ) -> Result<Option<DynSource<'s>>, AllocError> {
            let columns = conditions.iter().map(Condition::column).collect();
            let cutoff = cutoff.map(|cutoff| (cutoff.column, cutoff.descending));
            self.conditions.borrow_mut().push((columns, cutoff));
            Ok(None)
        }

        fn forms(&self, _: u32) -> Forms {
            if self.lazy { Forms::FLAT | Forms::LAZY } else { Forms::FLAT }
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
            _: Forms,
        ) -> Result<ColumnView, Error> {
            let (handle, start) = lazy.handle();
            let (column, batch) = (handle[0], handle[1]);
            let rows: StdVec<u16> = match selection.kept() {
                Kept::Prefix => (0..4).collect(),
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
        let (kept, rows, _) = counted_steps(table, filters, output);
        (kept, rows)
    }

    /// As `counted`, and the pipeline's steps.
    fn counted_steps(
        table: &DynScannable,
        filters: &[(usize, Comparison, i64)],
        output: &[usize],
    ) -> (StdVec<StdVec<i64>>, u32, alloc::string::String) {
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
                Kept::Prefix => (0..4).collect(),
                Kept::None => StdVec::new(),
                Kept::Select(rows) => rows.into(),
            };
            for row in selected {
                let columns = physical.columns().iter().map(|c| batch.column(c.position));
                kept.push(columns.map(|column| column.int64s()[usize::from(row)]).collect());
            }
        }
        (kept, rows, physical.pipeline().describe())
    }

    #[test]
    fn loads_lazy_columns_where_read_for_the_rows_kept() {
        let (loads, conditions) = (RefCell::new(StdVec::new()), RefCell::new(StdVec::new()));
        let table = DynScannable::new(
            &Heap,
            Counted { lazy: true, loads: &loads, conditions: &conditions },
        )
        .unwrap();
        // `a > 11` keeps rows 2 and 3 of batch 1, and batch 2; `b < 122` then
        // drops rows 2 and 3 of batch 2.
        let filters = [(0, Comparison::Greater, 11), (1, Comparison::Less, 122)];
        let (kept, _, steps) = counted_steps(&table, &filters, &[2, 0]);
        assert_eq!(kept, [[212, 12], [213, 13], [220, 20], [221, 21]]);
        assert_eq!(steps, "Scan | Filter | Materialize | Filter | Materialize");
        // `a` is read by the scan's parent, so it isn't lazy. `b` is loaded
        // for the second filter, and `c` for the output, only for the batches
        // and rows still kept.
        let b = [(1, 1, vec![2, 3]), (1, 2, vec![0, 1, 2, 3])];
        let c = [(2, 1, vec![2, 3]), (2, 2, vec![0, 1])];
        assert_eq!(*loads.borrow(), [b[0].clone(), c[0].clone(), b[1].clone(), c[1].clone()]);
    }

    #[test]
    fn never_loads_columns_nothing_reads() {
        let (loads, conditions) = (RefCell::new(StdVec::new()), RefCell::new(StdVec::new()));
        let table = DynScannable::new(
            &Heap,
            Counted { lazy: true, loads: &loads, conditions: &conditions },
        )
        .unwrap();
        // With no output, a column is still scanned, for its rows.
        assert_eq!(counted(&table, &[], &[]), (vec![StdVec::new(); 12], 12));
        assert!(loads.borrow().is_empty());
    }

    #[test]
    fn pushes_conditions_down_to_scans() {
        let (loads, conditions) = (RefCell::new(StdVec::new()), RefCell::new(StdVec::new()));
        let table = Counted { lazy: false, loads: &loads, conditions: &conditions };
        let table = DynScannable::new(&Heap, table).unwrap();
        let filters =
            [(0, Comparison::Greater, 11), (1, Comparison::Less, 122), (0, Comparison::Less, 21)];
        // The filters still test them.
        assert_eq!(counted(&table, &filters, &[0]).0, [[12], [13], [20]]);
        // Each filter's, for the scan to apply: two on `a`, one on `b`.
        assert_eq!(*conditions.borrow(), [(vec![0, 0, 1], None)]);
        // Without filters there's nothing to push down.
        counted(&table, &[], &[0]);
        assert_eq!(conditions.borrow().len(), 1);
    }

    #[test]
    fn reads_eagerly_from_tables_that_cant_be_lazy() {
        let (loads, conditions) = (RefCell::new(StdVec::new()), RefCell::new(StdVec::new()));
        let table = DynScannable::new(
            &Heap,
            Counted { lazy: false, loads: &loads, conditions: &conditions },
        )
        .unwrap();
        let filters = [(0, Comparison::Greater, 11), (1, Comparison::Less, 122)];
        let (kept, _) = counted(&table, &filters, &[2, 0]);
        assert_eq!(kept, [[212, 12], [213, 13], [220, 20], [221, 21]]);
        assert!(loads.borrow().is_empty());
    }

    /// What `Counted`'s scan is given to skip by, under `stages`, bottom up:
    /// a filter `a > 0`, `ORDER BY b DESC` or a `LIMIT` of their count.
    fn skipped_by(stages: &[Stage]) -> StdVec<SkippedBy> {
        let (loads, conditions) = (RefCell::new(StdVec::new()), RefCell::new(StdVec::new()));
        let table = Counted { lazy: false, loads: &loads, conditions: &conditions };
        let table = DynScannable::new(&Heap, table).unwrap();
        let mut plan = LogicalPlan::new(&Heap).unwrap();
        let mut columns = SlowVec::fixed(&Heap, 2).unwrap();
        let mut ids = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64).unwrap();
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            ids.push(binding.id);
        }
        let scan = DynOp::new(&Heap, ScanOp::new(&table, columns)).unwrap();
        let mut node = plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
        for stage in stages {
            let op = match *stage {
                Stage::Filter => {
                    let leaf = Leaf::Compare {
                        column: ids[0],
                        comparison: Comparison::Greater,
                        value: Value::Int64(0),
                    };
                    let nodes = SlowVec::fixed_from(&Heap, [Node::Leaf(leaf)].into_iter()).unwrap();
                    DynOp::new(&Heap, FilterOp { predicate: Predicate::new(nodes) }).unwrap()
                }
                Stage::OrderBy => {
                    let key = SortKey { column: ids[1], descending: true };
                    let keys = SlowVec::fixed_from(&Heap, [key].into_iter()).unwrap();
                    DynOp::new(&Heap, crate::order_by::OrderByOp { keys }).unwrap()
                }
                Stage::Limit(count) => {
                    DynOp::new(&Heap, crate::limit::LimitOp { count, offset: 0 }).unwrap()
                }
            };
            let children = SlowVec::fixed_from(&Heap, [node].into_iter()).unwrap();
            node = plan.add_node(op, children).unwrap();
        }
        let a = NamedColumn { name: plan.columns[0].name, id: ids[0] };
        assert!(plan.output.push(a).is_ok());
        crate::optimize::optimize(&Heap, &mut plan).unwrap();
        lower(&Heap, &plan).unwrap();
        drop(plan);
        conditions.borrow().clone()
    }

    /// A stage of `skipped_by`'s plans.
    enum Stage {
        Filter,
        OrderBy,
        Limit(u64),
    }

    #[test]
    fn tells_scans_the_cutoff_a_top_k_above_publishes() {
        use Stage::{Filter, Limit, OrderBy};
        // Filters only drop rows by their values, so rows past the cutoff
        // can be skipped below them too.
        let skipped = skipped_by(&[Filter, OrderBy, Limit(2)]);
        assert_eq!(skipped, [(vec![0], Some((1, true)))]);
        // A filter over the sort leaves it whole, and a limit below it
        // keeps the first rows, whatever their keys.
        assert!(skipped_by(&[OrderBy, Filter, Limit(2)]).is_empty());
        assert!(skipped_by(&[Limit(5), OrderBy, Limit(2)]).is_empty());
    }

    /// A table of one column, `a`, of 7 rows from -3 to 9, which it says
    /// only through `summary`, if `knows`: it has no rows to read.
    struct Summarized {
        knows: bool,
    }

    impl Scannable for Summarized {
        type State<'s>
            = ()
        where
            Self: 's;
        type Loader = ();

        fn column_count(&self) -> u32 {
            1
        }

        fn column_name(&self, _: u32) -> &'static str {
            "a"
        }

        fn column_type(&self, _: u32) -> DataType {
            DataType::Int64
        }

        fn open(&self, _: &mut Context, _: &[(u32, Forms)]) -> Result<(), Error> {
            Ok(())
        }

        fn next(&self, _: &mut Context, (): &mut (), _: &mut RowBatch) -> Result<bool, Error> {
            Ok(false)
        }

        fn summary(&self, column: Option<u32>) -> Option<Summary> {
            let bounds = column.map(|_| Bounds { min: -3, max: 9 });
            self.knows.then_some(Summary { rows: 7, bounds })
        }
    }

    /// `COUNT(*)`, `MIN(a)` and `MAX(a)` of `Summarized`, optimized: the
    /// pipeline's steps and its one row, or none.
    fn summarized(knows: bool) -> (alloc::string::String, StdVec<i64>) {
        use crate::aggregate::{Aggregate, AggregateOp, Function};
        let table = DynScannable::new(&Heap, Summarized { knows }).unwrap();
        let mut plan = LogicalPlan::new(&Heap).unwrap();
        let a = plan.add_column("a", DataType::Int64).unwrap();
        let columns = [ScanColumn { column: 0, binding: a, forms: Forms::FLAT }];
        let columns = SlowVec::fixed_from(&Heap, columns.into_iter()).unwrap();
        let scan = DynOp::new(&Heap, ScanOp::new(&table, columns)).unwrap();
        let scan = plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
        let mut aggregates = StdVec::new();
        for (function, input) in
            [(Function::CountStar, None), (Function::Min, Some(a.id)), (Function::Max, Some(a.id))]
        {
            let binding = plan.add_column("x", DataType::Int64).unwrap();
            aggregates.push(Aggregate { function, input, distinct: false, binding });
            assert!(plan.output.push(binding).is_ok());
        }
        let aggregate = AggregateOp {
            keys: SlowVec::fixed(&Heap, 0).unwrap(),
            aggregates: SlowVec::fixed_from(&Heap, aggregates.into_iter()).unwrap(),
        };
        let aggregate = DynOp::new(&Heap, aggregate).unwrap();
        plan.add_node(aggregate, SlowVec::fixed_from(&Heap, [scan].into_iter()).unwrap()).unwrap();
        crate::optimize::optimize(&Heap, &mut plan).unwrap();
        let physical = lower(&Heap, &plan).unwrap();
        let query = QueryAllocators::new(&Heap);
        let mut execution = physical.pipeline().start(&query).unwrap();
        let mut batch = RowBatch::new();
        let mut row = StdVec::new();
        if execution.next(&mut batch).unwrap() {
            row = physical.columns().iter().map(|c| batch.column(c.position).int64s()[0]).collect();
        }
        (physical.pipeline().describe(), row)
    }

    #[test]
    fn answers_aggregates_from_what_a_table_knows_without_reading_it() {
        let (steps, row) = summarized(true);
        assert_eq!(steps, "Values");
        assert_eq!(row, [7, -3, 9]);
        // Without a summary, it's scanned, and with no rows, has a row of
        // zero and nulls.
        assert_eq!(summarized(false).0, "Scan | HashAggregate");
    }

    /// Reads one column, taking it in `accepts`; it's never lowered.
    struct Reads {
        column: ColumnId,
        accepts: Forms,
    }

    impl<'c> Op<'c> for Reads {
        fn derive(&self, _: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
            failed!()
        }

        fn lower(&self, _: &PlanNode<'c>, _: &mut Lowering<'_, 'c>) -> Result<(), LowerError> {
            failed!()
        }

        fn reads(&self, reads: &mut Needed) {
            reads.need(self.column);
        }

        fn accepts(&self, column: ColumnId) -> Forms {
            if column == self.column { self.accepts } else { Forms::FLAT }
        }
    }

    /// How many nodes a plan has once its forms are chosen: a scan of
    /// `Counted`, a filter on `a`, and ops reading `b`, each in its forms of
    /// `accepts`, in turn.
    fn nodes_with(accepts: &[Forms]) -> usize {
        let (loads, conditions) = (RefCell::new(StdVec::new()), RefCell::new(StdVec::new()));
        let table = DynScannable::new(
            &Heap,
            Counted { lazy: true, loads: &loads, conditions: &conditions },
        )
        .unwrap();
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
        let mut below =
            plan.add_node(filter, SlowVec::fixed_from(&Heap, [scan].into_iter()).unwrap()).unwrap();
        for &accepts in accepts {
            let reads = DynOp::new(&Heap, Reads { column: bindings[1].id, accepts }).unwrap();
            let children = SlowVec::fixed_from(&Heap, [below].into_iter()).unwrap();
            below = plan.add_node(reads, children).unwrap();
        }
        choose_forms(&Heap, &mut plan).unwrap();
        plan.nodes.len()
    }

    #[test]
    fn loads_lazy_columns_only_for_readers_that_dont_take_them_lazy() {
        // `b` passes the filter before it's read, so it's made lazy: loaded
        // below a reader that takes it flat, by one more node, and handed as
        // it is to one that takes it lazy.
        assert_eq!(nodes_with(&[Forms::FLAT]), 4);
        assert_eq!(nodes_with(&[Forms::LAZY]), 3);
        // Every reader gets the column as the first does, so it's loaded
        // for the first if a later one doesn't take it lazy.
        assert_eq!(nodes_with(&[Forms::LAZY, Forms::FLAT]), 5);
        assert_eq!(nodes_with(&[Forms::LAZY, Forms::LAZY]), 4);
    }
}
