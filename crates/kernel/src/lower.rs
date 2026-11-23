//! `lower`: a `LogicalPlan` to a `PhysicalPlan`, a pipeline ready to run.
//! `plan_nodes` decides first, node by node, what each needs of its inputs
//! and what it outputs; lowering then only adds the steps that does.

use crate::allocator::AllocError;
use crate::buffer::Buffer;
use crate::containers::SmallVec;
use crate::env::Env;
use crate::error::Error;
use crate::expression::Exprs;
use crate::names::{Name, Names};
use crate::optimize::optimize;
use crate::output_properties::{InputStep, OutputProperties, Planned, plan_nodes};
use crate::pipeline::{Execution, Pipeline};
use crate::plan::{ColumnId, LogicalPlan, PLAN_NAME_BYTES_MAX, PlanNode, PlanNodeId, SortKey};
use crate::sort::sort;
use crate::statement::Statement;
use crate::step::{DynSource, Step};
use crate::top_k::top_k;

/// Why a plan couldn't be planned or lowered.
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
    columns: SmallVec<OutputColumn>,
    statement: Statement,
}

impl<'c> PhysicalPlan<'c> {
    /// `plan` optimized, planned and lowered: what an embedder runs.
    #[inline(never)]
    pub fn from_logical(
        env: &Env,
        plan: &mut LogicalPlan<'c>,
    ) -> Result<PhysicalPlan<'c>, LowerError> {
        optimize(env, plan)?;
        let planned = plan_nodes(env, plan)?;
        lower(env, plan, &planned)
    }

    pub fn pipeline(&self) -> &Pipeline<'c> {
        &self.pipeline
    }

    /// Starts a run, as `Pipeline::start` does, of its statement.
    pub fn start<'r>(&'r self, env: &'r Env) -> Result<Execution<'r>, Error> {
        self.pipeline.start(env, Some(&self.statement))
    }

    pub fn columns(&self) -> &[OutputColumn] {
        &self.columns
    }

    pub fn name(&self, column: OutputColumn) -> &str {
        self.names.get(column.name)
    }
}

/// A pipeline being built from a planned plan, which each operation adds
/// to.
pub struct Lowering<'p, 'c> {
    /// The plan being lowered.
    plan: &'p LogicalPlan<'c>,
    /// What planning decided of each node.
    planned: &'p Planned,
    /// What steps are made with.
    env: &'p Env,
    /// The pipeline's source, once a node sets it.
    source: Option<DynSource<'c>>,
    /// The steps after it, in order.
    steps: SmallVec<Step<'c>>,
    /// What is known of the rows the steps so far output.
    rows: &'p OutputProperties,
    /// The values of the plan's expressions' `IN` lists, which each program
    /// and predicate shares.
    expr_values: Option<Buffer>,
}

/// Its methods are named by what they do. Building ones add to the
/// pipeline, and `query_` ones only read what's known.
impl<'p, 'c> Lowering<'p, 'c> {
    /// What to make steps with.
    pub fn env(&self) -> &'p Env {
        self.env
    }

    /// The plan's expressions.
    pub fn exprs(&self) -> &'p Exprs {
        &self.plan.exprs
    }

    /// The values of the plan's expressions' `IN` lists, if any.
    pub fn expr_values(&self) -> Option<Buffer> {
        self.expr_values.clone()
    }

    fn lower_node(&mut self, id: PlanNodeId) -> Result<(), LowerError> {
        let (plan, planned) = (self.plan, self.planned);
        let node = at!(plan.nodes, id as usize);
        let output = &at!(planned, id as usize).output;
        self.rows = output;
        node.op.lower(node, self)?;
        self.rows = output;
        Ok(())
    }

    /// Lowers input `input` of `node`, the node being lowered, then the
    /// steps planning chose to meet its needs.
    pub fn lower_input(&mut self, node: &PlanNode<'c>, input: u32) -> Result<(), LowerError> {
        let id = self.plan.id_of(node);
        self.lower_node(*at!(node.children, input as usize))?;
        let planned = at!(at!(self.planned, id as usize).inputs, input as usize);
        self.rows = &planned.rows;
        if let Some(step) = &planned.step {
            self.apply(step)?;
        }
        Ok(())
    }

    fn apply(&mut self, step: &InputStep) -> Result<(), LowerError> {
        let InputStep::Sort { keys, keep, publish, .. } = step;
        let keys = keys.iter().map(|key| crate::sort::SortKey {
            column: self.query_position(key.column),
            descending: key.descending,
        });
        let keys = SmallVec::fixed_from(self.env, keys)?;
        let operator = match *keep {
            Some(count) => top_k(self.env, keys, count, *publish)?,
            None => sort(self.env, keys)?,
        };
        self.add_operator(Step::Operator(operator))
    }

    /// Lowers input `input` of `node`, the node being lowered, as a pipeline
    /// of its own, as Perfetto's `AddSidePipeline` does: a source for an op
    /// that reads that input whole, such as a join's build side. Returns it,
    /// and what is known of its rows.
    pub fn lower_side(
        &mut self,
        node: &PlanNode<'c>,
        input: u32,
    ) -> Result<(DynSource<'c>, &'p OutputProperties), LowerError> {
        let side = SmallVec::new(self.env, LOWERED_STEPS_MAX)?;
        let (source, steps, rows) =
            (self.source.take(), core::mem::replace(&mut self.steps, side), self.rows);
        let lowered = self.lower_input(node, input);
        let side_steps = core::mem::replace(&mut self.steps, steps);
        let (side_source, side_rows) = (core::mem::replace(&mut self.source, source), self.rows);
        self.rows = rows;
        lowered?;
        let Some(side_source) = side_source else { failed!() };
        let pipeline = Pipeline::new(side_source, side_steps);
        Ok((pipeline.into_source(self.env)?, side_rows))
    }

    /// How many of its first rows the open sort just below outputs, if it
    /// outputs only those.
    pub fn query_sort_keep(&self) -> Option<u32> {
        let sort = self.query_rows().open_sort?;
        let node = at!(self.planned, sort.node as usize);
        let Some(InputStep::Sort { keep, .. }) = &at!(node.inputs, sort.input as usize).step else {
            failed!()
        };
        *keep
    }

    /// For `node`, a scan, the column a `TopK` above publishes a cutoff on,
    /// and the slot it publishes in, if any.
    pub fn query_cutoff(&self, node: &PlanNode<'c>) -> Option<(SortKey, u32)> {
        at!(self.planned, self.plan.id_of(node) as usize).cutoff
    }

    /// What is known of the rows the steps added so far output.
    pub fn query_rows(&self) -> &OutputProperties {
        self.rows
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

    /// Where `column` is in batches.
    pub fn query_position(&self, column: ColumnId) -> u32 {
        let Some(position) = self.query_rows().position_of(column) else { failed!() };
        position
    }
}

/// Builds the pipeline of `plan`, once `planned` by `plan_nodes`, from its
/// root, with memory from `env`. The physical plan takes the plan's
/// statement.
pub fn lower<'c>(
    env: &Env,
    plan: &mut LogicalPlan<'c>,
    planned: &Planned,
) -> Result<PhysicalPlan<'c>, LowerError> {
    let mut lowering = Lowering {
        plan: &*plan,
        planned,
        env,
        source: None,
        steps: SmallVec::new(env, LOWERED_STEPS_MAX)?,
        rows: &at!(planned, plan.root as usize).output,
        expr_values: None,
    };
    let values = plan.exprs.values();
    if !values.is_empty() {
        let mut buffer = Buffer::allocate(env, size_of_val(values))?;
        buffer.as_mut_slice::<i64>().copy_from_slice(values);
        lowering.expr_values = Some(buffer);
    }
    lowering.lower_node(plan.root)?;

    let mut names = Names::new(env, PLAN_NAME_BYTES_MAX)?;
    let mut columns = SmallVec::fixed(env, plan.output.len())?;
    for column in plan.output.iter() {
        let name = names.add(plan.names.get(column.name))?;
        columns.push(OutputColumn { name, position: lowering.query_position(column.id) })?;
    }
    let Some(source) = lowering.source else { failed!() };
    let pipeline = Pipeline::new(source, lowering.steps);
    Ok(PhysicalPlan { pipeline, names, columns, statement: plan.take_statement() })
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec as StdVec;
    use core::cell::RefCell;

    /// The nodes of `column <comparison> value`.
    fn compared(column: ColumnId, comparison: Comparison, value: i64) -> ExprBuilder {
        let env = Env::current();
        let op = ExprOp::from(comparison);
        let mut nodes = ExprBuilder::new(env).unwrap();
        let (column, value) = (nodes.column(column).unwrap(), nodes.int(value).unwrap());
        nodes.node(op, column, value).unwrap();
        nodes
    }

    /// A filter of the expression of `nodes`, added to `plan`'s.
    fn filter<'c>(plan: &mut LogicalPlan<'c>, nodes: &ExprBuilder) -> DynOp<'c> {
        let env = Env::current();
        let condition = plan.exprs.add(nodes.nodes()).unwrap();
        DynOp::new(env, FilterOp::new(env, &plan.exprs, condition).unwrap()).unwrap()
    }

    use super::*;
    use crate::allocator::AllocError;
    use crate::buffer::Buffer;
    use crate::column::Bounds;
    use crate::column::{ColumnView, DataType, Forms};
    use crate::condition::{Condition, Cutoff};
    use crate::context::Context;
    use crate::env::Env;
    use crate::error::Error;
    use crate::expression::{ExprBuilder, ExprOp, Exprs};
    use crate::filter::Comparison;
    use crate::optimize::{Needed, choose_forms};
    use crate::output_properties::plan_nodes;
    use crate::output_properties::{DeriveContext, OutputProperties};
    use crate::plan::{ColumnId, DynOp, FilterOp, NamedColumn, Op, PlanNode, ScanColumn, ScanOp};
    use crate::row_batch::RowBatch;
    use crate::scannable::{DynScannable, Scannable, Summary};
    use crate::selection::{Kept, Selection};
    use crate::step::DynSource;

    /// Plans `plan`, then lowers it.
    fn lower_planned<'c>(plan: &mut LogicalPlan<'c>) -> Result<PhysicalPlan<'c>, LowerError> {
        let env = Env::current();
        let planned = plan_nodes(env, plan)?;
        lower(env, plan, &planned)
    }

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
            let env = Env::current();
            if *done {
                return Ok(false);
            }
            batch.reset(2);
            for &(column, _) in *columns {
                let mut values = Buffer::allocate(env, 16).unwrap();
                let first = [1, 10][column as usize];
                values.as_mut_slice::<i64>().copy_from_slice(&[first, first * 2]);
                assert!(
                    batch
                        .push_column(
                            ColumnView::new(
                                &mut Context::new(env, None),
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
        let env = Env::current();
        let table = DynScannable::new(env, AB).unwrap();
        let mut plan = LogicalPlan::new(env).unwrap();
        let a = plan.add_column("a", DataType::Int64, true).unwrap();
        let b = plan.add_column("b", DataType::Int64, true).unwrap();
        let mut columns = SmallVec::fixed(env, 2).unwrap();
        let a_column = ScanColumn { column: 0, binding: a, forms: Forms::FLAT };
        let b_column = ScanColumn { column: 1, binding: b, forms: Forms::FLAT };
        assert!(columns.push(a_column).is_ok() && columns.push(b_column).is_ok());
        let scan = DynOp::new(env, ScanOp::new(&table, columns)).unwrap();
        plan.add_node(scan, &[]).unwrap();
        assert!(plan.output.push(b).is_ok() && plan.output.push(a).is_ok());

        let physical = lower_planned(&mut plan).unwrap();
        let names: StdVec<&str> = physical.columns().iter().map(|&c| physical.name(c)).collect();
        assert_eq!(names, ["b", "a"]);

        let query = Env::current();
        let mut execution = physical.start(query).unwrap();
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

    /// Records which of `asks` hold of its input's rows: each is keys to
    /// group by and keys to sort by within groups, by column, a sort key's
    /// with whether it descends.
    struct Knows<'t> {
        asks: &'t [Ask<'t>],
        known: &'t RefCell<StdVec<bool>>,
        sorted_by_b: &'t RefCell<usize>,
    }

    impl<'c> Op<'c> for Knows<'_> {
        fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
            let out = *c.input(0);
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
            lowering.lower_input(node, 0)
        }
    }

    /// Which of `asks` hold of `Ab`'s rows, scanning its columns `read`; and
    /// how many keys `b` alone sorts by.
    fn knows(read: &[u32], asks: &[Ask]) -> (StdVec<bool>, usize) {
        knows_of(AB, read, asks)
    }

    /// As `knows`, of `table`.
    fn knows_of(table: Ab, read: &[u32], asks: &[Ask]) -> (StdVec<bool>, usize) {
        let env = Env::current();
        let (known, sorted_by_b) = (RefCell::new(StdVec::new()), RefCell::new(0));
        let table = DynScannable::new(env, table).unwrap();
        let mut plan = LogicalPlan::new(env).unwrap();
        let bindings = [
            plan.add_column("a", DataType::Int64, true).unwrap(),
            plan.add_column("b", DataType::Int64, true).unwrap(),
        ];
        let mut columns = SmallVec::fixed(env, read.len()).unwrap();
        for &column in read {
            let binding = bindings[column as usize];
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
        }
        let scan = DynOp::new(env, ScanOp::new(&table, columns)).unwrap();
        let scan = plan.add_node(scan, &[]).unwrap();
        let op = Knows { asks, known: &known, sorted_by_b: &sorted_by_b };
        let op = DynOp::new(env, op).unwrap();
        plan.add_node(op, &[scan]).unwrap();
        assert!(plan.output.push(bindings[read[0] as usize]).is_ok());
        lower_planned(&mut plan).unwrap();
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
        assert_eq!(knows(&[0, 1], asks), (expected.to_vec(), 0));
        // Without `a` read, nothing is known of `b`'s order.
        assert_eq!(knows(&[1], &asks[6..8]), ([false, false].to_vec(), 0));
        // Declared sorted by `b` alone too, it's known to be in both orders.
        let both = Ab(&[&[(0, false), (1, false)], &[(1, false)]]);
        assert_eq!(knows_of(both, &[0, 1], &asks[5..8]), ([true, true, true].to_vec(), 1));
    }

    /// The steps of `Ab` scanned, its rows counted by `groups` of its
    /// columns first, if given, then ordered by `keys`, by column, then
    /// limited to `(count, offset)` rows, if given.
    fn ordered(
        groups: Option<&[usize]>,
        keys: &[(usize, bool)],
        limit: Option<(u64, u64)>,
    ) -> alloc::string::String {
        let env = Env::current();
        let table = DynScannable::new(env, AB).unwrap();
        let mut plan = LogicalPlan::new(env).unwrap();
        let mut columns = SmallVec::fixed(env, 2).unwrap();
        let mut bindings = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64, true).unwrap();
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            bindings.push(binding);
        }
        let scan = DynOp::new(env, ScanOp::new(&table, columns)).unwrap();
        let mut input = plan.add_node(scan, &[]).unwrap();
        if let Some(groups) = groups {
            let binding = plan.add_column("n", DataType::Int64, true).unwrap();
            let function = crate::aggregate::Function::CountStar;
            let aggregate =
                crate::aggregate::Aggregate { function, input: None, distinct: false, binding };
            let aggregate = crate::aggregate::AggregateOp {
                keys: SmallVec::fixed_from(env, groups.iter().map(|&g| bindings[g].id)).unwrap(),
                aggregates: SmallVec::fixed_from(env, [aggregate].into_iter()).unwrap(),
                answer: None,
            };
            let aggregate = DynOp::new(env, aggregate).unwrap();
            let children = [input];
            input = plan.add_node(aggregate, &children).unwrap();
        }
        let sort_keys =
            keys.iter().map(|&(key, descending)| SortKey { column: bindings[key].id, descending });
        let order_by =
            crate::order_by::OrderByOp { keys: SmallVec::fixed_from(env, sort_keys).unwrap() };
        let order_by = DynOp::new(env, order_by).unwrap();
        let children = [input];
        let ordered = plan.add_node(order_by, &children).unwrap();
        if let Some((count, offset)) = limit {
            let limit = DynOp::new(env, crate::limit::LimitOp { count, offset }).unwrap();
            plan.add_node(limit, &[ordered]).unwrap();
        }
        assert!(plan.output.push(bindings[keys.first().map_or(0, |key| key.0)]).is_ok());
        lower_planned(&mut plan).unwrap().pipeline().describe()
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
        // A TopK counts its rows in `u32`, so keeping more is a sort and a
        // LIMIT, which counts in `u64`.
        assert_eq!(ordered(None, &[(1, false)], Some((1 << 40, 0))), "Scan | Sort | Limit");
        // Rows already in order are only limited.
        assert_eq!(ordered(None, &[(0, false)], Some((10, 0))), "Scan | Limit");
        let grouped = ordered(Some(&[1]), &[(1, true)], Some((1, 0)));
        assert_eq!(grouped, "Scan | HashAggregate | TopK");
    }

    #[test]
    fn drops_a_sort_a_later_sort_makes_pointless() {
        let env = Env::current();
        let table = DynScannable::new(env, AB).unwrap();
        let mut plan = LogicalPlan::new(env).unwrap();
        let mut columns = SmallVec::fixed(env, 2).unwrap();
        let mut ids = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64, true).unwrap();
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            ids.push(binding);
        }
        let scan = DynOp::new(env, ScanOp::new(&table, columns)).unwrap();
        let mut node = plan.add_node(scan, &[]).unwrap();
        for (column, descending) in [(1, false), (0, true)] {
            let key = SortKey { column: ids[column].id, descending };
            let keys = SmallVec::fixed_from(env, [key].into_iter()).unwrap();
            let order_by = DynOp::new(env, crate::order_by::OrderByOp { keys }).unwrap();
            let children = [node];
            node = plan.add_node(order_by, &children).unwrap();
        }
        assert!(plan.output.push(ids[0]).is_ok());
        assert_eq!(lower_planned(&mut plan).unwrap().pipeline().describe(), "Scan | Sort");
    }

    /// `Ab` scanned with `output` as the plan's output, and pruned.
    fn pruned(output: &[usize]) -> (usize, StdVec<i64>) {
        let env = Env::current();
        let table = DynScannable::new(env, AB).unwrap();
        let mut plan = LogicalPlan::new(env).unwrap();
        let mut columns = SmallVec::fixed(env, 2).unwrap();
        let mut bindings = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64, true).unwrap();
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            bindings.push(binding);
        }
        let scan = DynOp::new(env, ScanOp::new(&table, columns)).unwrap();
        plan.add_node(scan, &[]).unwrap();
        for &i in output {
            assert!(plan.output.push(bindings[i]).is_ok());
        }
        crate::optimize::optimize(env, &mut plan).unwrap();

        let physical = lower_planned(&mut plan).unwrap();
        let query = Env::current();
        let mut execution = physical.start(query).unwrap();
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
            let env = Env::current();
            if *done {
                return Ok(false);
            }
            batch.reset(4);
            for &(column, _) in *columns {
                let cells =
                    [[Some(1), Some(2), None, Some(0)], [Some(10), Some(20), Some(30), None]];
                let mut values = Buffer::allocate(env, 32).unwrap();
                let mut validity = Buffer::allocate(env, 1).unwrap();
                for (row, cell) in cells[column as usize].iter().enumerate() {
                    values.as_mut_slice::<i64>()[row] = cell.unwrap_or(0);
                    validity.as_mut_slice::<u8>()[0] |= u8::from(cell.is_some()) << row;
                }
                let column = ColumnView::new(
                    &mut Context::new(env, None),
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
        let env = Env::current();
        let table = DynScannable::new(env, Nullable).unwrap();
        let mut plan = LogicalPlan::new(env).unwrap();
        let mut columns = SmallVec::fixed(env, 2).unwrap();
        let mut bindings = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64, true).unwrap();
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            bindings.push(binding);
        }
        let scan = DynOp::new(env, ScanOp::new(&table, columns)).unwrap();
        let scan = plan.add_node(scan, &[]).unwrap();
        let (a, b) = (bindings[0].id, bindings[1].id);
        let mut nodes = compared(a, Comparison::Greater, 1);
        let greater = nodes.last();
        let b = nodes.column(b).unwrap();
        let null = nodes.node(ExprOp::IsNull, b, 0).unwrap();
        nodes.node(ExprOp::Or, greater, null).unwrap();
        let filter = filter(&mut plan, &nodes);
        plan.add_node(filter, &[scan]).unwrap();
        assert!(plan.output.push(bindings[1]).is_ok());
        crate::optimize::optimize(env, &mut plan).unwrap();

        let physical = lower_planned(&mut plan).unwrap();
        let query = Env::current();
        let mut execution = physical.start(query).unwrap();
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
        let env = Env::current();
        let table = DynScannable::new(env, Wide(70)).unwrap();
        let plan = |output: usize| {
            let mut plan = LogicalPlan::new(env).unwrap();
            let mut columns = SmallVec::fixed(env, 70).unwrap();
            for column in 0..70 {
                let binding = plan.add_column("c", DataType::Int64, true).unwrap();
                assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
                if (column as usize) < output {
                    assert!(plan.output.push(binding).is_ok());
                }
            }
            let scan = DynOp::new(env, ScanOp::new(&table, columns)).unwrap();
            plan.add_node(scan, &[]).unwrap();
            plan
        };
        assert_eq!(lower_planned(&mut plan(70)).err(), Some(LowerError::TooManyColumns));
        // Pruned to the two in the output, it fits.
        let mut narrow = plan(2);
        crate::optimize::optimize(env, &mut narrow).unwrap();
        assert!(lower_planned(&mut narrow).is_ok());
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
            _: &Env,
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
        let env = Env::current();
        let mut plan = LogicalPlan::new(env).unwrap();
        let mut columns = SmallVec::fixed(env, 3).unwrap();
        let mut bindings = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b"), (2, "c")] {
            let binding = plan.add_column(name, DataType::Int64, true).unwrap();
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            bindings.push(binding);
        }
        let scan = DynOp::new(env, ScanOp::new(table, columns)).unwrap();
        let mut node = plan.add_node(scan, &[]).unwrap();
        for &(column, comparison, value) in filters {
            let column = bindings[column].id;
            let filter = filter(&mut plan, &compared(column, comparison, value));
            let children = [node];
            node = plan.add_node(filter, &children).unwrap();
        }
        for &i in output {
            assert!(plan.output.push(bindings[i]).is_ok());
        }
        crate::optimize::optimize(env, &mut plan).unwrap();

        let physical = lower_planned(&mut plan).unwrap();
        let query = Env::current();
        let mut execution = physical.start(query).unwrap();
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
        let env = Env::current();
        let (loads, conditions) = (RefCell::new(StdVec::new()), RefCell::new(StdVec::new()));
        let table =
            DynScannable::new(env, Counted { lazy: true, loads: &loads, conditions: &conditions })
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
        let env = Env::current();
        let (loads, conditions) = (RefCell::new(StdVec::new()), RefCell::new(StdVec::new()));
        let table =
            DynScannable::new(env, Counted { lazy: true, loads: &loads, conditions: &conditions })
                .unwrap();
        // With no output, a column is still scanned, for its rows.
        assert_eq!(counted(&table, &[], &[]), (vec![StdVec::new(); 12], 12));
        assert!(loads.borrow().is_empty());
    }

    #[test]
    fn pushes_conditions_down_to_scans() {
        let env = Env::current();
        let (loads, conditions) = (RefCell::new(StdVec::new()), RefCell::new(StdVec::new()));
        let table = Counted { lazy: false, loads: &loads, conditions: &conditions };
        let table = DynScannable::new(env, table).unwrap();
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
        let env = Env::current();
        let (loads, conditions) = (RefCell::new(StdVec::new()), RefCell::new(StdVec::new()));
        let table =
            DynScannable::new(env, Counted { lazy: false, loads: &loads, conditions: &conditions })
                .unwrap();
        let filters = [(0, Comparison::Greater, 11), (1, Comparison::Less, 122)];
        let (kept, _) = counted(&table, &filters, &[2, 0]);
        assert_eq!(kept, [[212, 12], [213, 13], [220, 20], [221, 21]]);
        assert!(loads.borrow().is_empty());
    }

    /// What `Counted`'s scan is given to skip by, under `stages`, bottom up:
    /// a filter `a > 0`, `ORDER BY b DESC` or a `LIMIT` of their count.
    fn skipped_by(stages: &[Stage]) -> StdVec<SkippedBy> {
        let env = Env::current();
        let (loads, conditions) = (RefCell::new(StdVec::new()), RefCell::new(StdVec::new()));
        let table = Counted { lazy: false, loads: &loads, conditions: &conditions };
        let table = DynScannable::new(env, table).unwrap();
        let mut plan = LogicalPlan::new(env).unwrap();
        let mut columns = SmallVec::fixed(env, 2).unwrap();
        let mut ids = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64, true).unwrap();
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            ids.push(binding.id);
        }
        let scan = DynOp::new(env, ScanOp::new(&table, columns)).unwrap();
        let mut node = plan.add_node(scan, &[]).unwrap();
        for stage in stages {
            let op = match *stage {
                Stage::Filter => filter(&mut plan, &compared(ids[0], Comparison::Greater, 0)),
                Stage::OrderBy => {
                    let key = SortKey { column: ids[1], descending: true };
                    let keys = SmallVec::fixed_from(env, [key].into_iter()).unwrap();
                    DynOp::new(env, crate::order_by::OrderByOp { keys }).unwrap()
                }
                Stage::Limit(count) => {
                    DynOp::new(env, crate::limit::LimitOp { count, offset: 0 }).unwrap()
                }
            };
            let children = [node];
            node = plan.add_node(op, &children).unwrap();
        }
        let a = NamedColumn { name: plan.columns[0].name, id: ids[0] };
        assert!(plan.output.push(a).is_ok());
        crate::optimize::optimize(env, &mut plan).unwrap();
        lower_planned(&mut plan).unwrap();
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
        // A filter written above the sort moves below it, so this plan is
        // the same as the first. A limit below the sort keeps the first rows
        // whatever their keys, so nothing can be skipped.
        assert_eq!(skipped_by(&[OrderBy, Filter, Limit(2)]), skipped);
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
        let env = Env::current();
        let table = DynScannable::new(env, Summarized { knows }).unwrap();
        let mut plan = LogicalPlan::new(env).unwrap();
        let a = plan.add_column("a", DataType::Int64, true).unwrap();
        let columns = [ScanColumn { column: 0, binding: a, forms: Forms::FLAT }];
        let columns = SmallVec::fixed_from(env, columns.into_iter()).unwrap();
        let scan = DynOp::new(env, ScanOp::new(&table, columns)).unwrap();
        let scan = plan.add_node(scan, &[]).unwrap();
        let mut aggregates = StdVec::new();
        for (function, input) in
            [(Function::CountStar, None), (Function::Min, Some(a.id)), (Function::Max, Some(a.id))]
        {
            let binding = plan.add_column("x", DataType::Int64, true).unwrap();
            aggregates.push(Aggregate { function, input, distinct: false, binding });
            assert!(plan.output.push(binding).is_ok());
        }
        let aggregate = AggregateOp {
            keys: SmallVec::fixed(env, 0).unwrap(),
            aggregates: SmallVec::fixed_from(env, aggregates.into_iter()).unwrap(),
            answer: None,
        };
        let aggregate = DynOp::new(env, aggregate).unwrap();
        plan.add_node(aggregate, &[scan]).unwrap();
        crate::optimize::optimize(env, &mut plan).unwrap();
        let physical = lower_planned(&mut plan).unwrap();
        let query = Env::current();
        let mut execution = physical.start(query).unwrap();
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
        fn reads(&self, reads: &mut Needed) {
            reads.need(self.column);
        }

        fn accepts(&self, _: &Exprs, column: ColumnId) -> Forms {
            if column == self.column { self.accepts } else { Forms::FLAT }
        }

        fn derive(&self, _: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
            failed!()
        }

        fn lower(&self, _: &PlanNode<'c>, _: &mut Lowering<'_, 'c>) -> Result<(), LowerError> {
            failed!()
        }
    }

    /// How many nodes a plan has once its forms are chosen: a scan of
    /// `Counted`, a filter on `a`, and ops reading `b`, each in its forms of
    /// `accepts`, in turn.
    fn nodes_with(accepts: &[Forms]) -> usize {
        let env = Env::current();
        let (loads, conditions) = (RefCell::new(StdVec::new()), RefCell::new(StdVec::new()));
        let table =
            DynScannable::new(env, Counted { lazy: true, loads: &loads, conditions: &conditions })
                .unwrap();
        let mut plan = LogicalPlan::new(env).unwrap();
        let mut columns = SmallVec::fixed(env, 2).unwrap();
        let mut bindings = StdVec::new();
        for (column, name) in [(0, "a"), (1, "b")] {
            let binding = plan.add_column(name, DataType::Int64, true).unwrap();
            assert!(columns.push(ScanColumn { column, binding, forms: Forms::FLAT }).is_ok());
            bindings.push(binding);
        }
        let scan = DynOp::new(env, ScanOp::new(&table, columns)).unwrap();
        let scan = plan.add_node(scan, &[]).unwrap();
        let filter = filter(&mut plan, &compared(bindings[0].id, Comparison::Greater, 0));
        let mut below = plan.add_node(filter, &[scan]).unwrap();
        for &accepts in accepts {
            let reads = DynOp::new(env, Reads { column: bindings[1].id, accepts }).unwrap();
            let children = [below];
            below = plan.add_node(reads, &children).unwrap();
        }
        choose_forms(env, &mut plan).unwrap();
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
