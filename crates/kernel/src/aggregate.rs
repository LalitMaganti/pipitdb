//! `AggregateOp`: aggregates, such as `COUNT(*)` and `SUM(x)`, over each
//! group of its child's rows with the same keys, or over all of them if it
//! has none. `hash_aggregate` makes the breaker that runs it.

use crate::column::DataType;
use crate::hash_aggregate::hash_aggregate;
use crate::lower::{LowerError, Lowering, SortKey};
use crate::optimize::{Needed, Pruned};
use crate::plan::{ColumnId, NamedColumn, Op, PlanNode};
use crate::slow_vec::SlowVec;
use crate::step::Step;

/// An aggregate function, as Perfetto's `AggregateCall::Function`, and
/// `Avg`. Each reads `Int64` or `Float64` values.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Function {
    /// How many rows there are.
    CountStar,
    /// How many values aren't null.
    Count,
    /// The total of the values that aren't null, null if there are none.
    Sum,
    /// The least value that isn't null, null if there are none.
    Min,
    /// The greatest value that isn't null, null if there are none.
    Max,
    /// The mean of the values that aren't null, a `Float64`, null if there
    /// are none.
    Avg,
}

impl Function {
    /// What it makes of a column of `data_type`, `None` for `COUNT(*)`, or
    /// `None` if it can't.
    pub fn output_type(self, data_type: Option<DataType>) -> Option<DataType> {
        match (self, data_type) {
            (Function::CountStar, None) | (Function::Count, Some(_)) => Some(DataType::Int64),
            (_, None | Some(DataType::String)) | (Function::CountStar, _) => None,
            (Function::Avg, Some(_)) => Some(DataType::Float64),
            (_, Some(data_type)) => Some(data_type),
        }
    }
}

/// An aggregate in a plan: its function, the column it reads, `None` only for
/// `COUNT(*)`, and the column it makes.
#[derive(Clone, Copy)]
pub struct Aggregate {
    pub function: Function,
    pub input: Option<ColumnId>,
    pub binding: NamedColumn,
}

/// Outputs a row for each group: its keys, then its aggregates, in the order
/// groups were first seen. With no keys, all rows are one group, and there's
/// a row even if there are none.
pub struct AggregateOp {
    /// The `Int64` columns rows are grouped by.
    pub keys: SlowVec<ColumnId>,
    /// What each group's rows make.
    pub aggregates: SlowVec<Aggregate>,
}

impl<'c> Op<'c> for AggregateOp {
    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        lowering.lower(*at!(node.children, 0))?;
        let allocator = lowering.allocator();
        check!(self.keys.iter().all(|&key| lowering.query_type(key) == DataType::Int64));
        check!(self.aggregates.iter().all(|aggregate| {
            let data_type = aggregate.input.map(|id| lowering.query_type(id));
            aggregate.function.output_type(data_type).is_some()
        }));
        let keys = self.keys.iter().map(|&key| lowering.query_position(key));
        let key_columns = SlowVec::fixed_from(allocator, keys)?;
        let aggregates = self.aggregates.iter().map(|aggregate| AggregateCall {
            function: aggregate.function,
            column: aggregate.input.map_or(0, |id| lowering.query_position(id)),
            data_type: aggregate.input.map_or(DataType::Int64, |id| lowering.query_type(id)),
        });
        let aggregates = SlowVec::fixed_from(allocator, aggregates)?;
        // Groups are output in the order first seen, so if the input is sorted
        // by the keys, the output is too.
        let mut sorted = [SortKey { column: 0, descending: false }; 8];
        let known = lowering.query_sorted_by(&self.keys);
        let sorted = at_mut!(sorted, ..known.len().min(8));
        sorted.copy_from_slice(at!(known, ..sorted.len()));
        let operator = hash_aggregate(allocator, HashAggregateSpec { key_columns, aggregates })?;
        lowering.add_breaker(Step::Operator(operator))?;
        // The output has a row for each group, with a new layout.
        lowering.declare_new_batches();
        for &key in self.keys.iter() {
            lowering.declare_column(key)?;
        }
        for aggregate in self.aggregates.iter() {
            lowering.declare_column(aggregate.binding.id)?;
        }
        lowering.declare_sorted(sorted);
        Ok(())
    }

    /// Drops the aggregates nothing reads, keeping one if there are no keys:
    /// a batch with no columns has no rows.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        let any = self.aggregates.iter().any(|aggregate| needed.is_needed(aggregate.binding.id));
        let keep_first = !any && self.keys.is_empty();
        let mut first = true;
        self.aggregates.retain(|aggregate| {
            let keep = needed.is_needed(aggregate.binding.id) || (keep_first && first);
            first = false;
            keep
        });
        self.reads(needed);
        Pruned::Keep
    }

    fn reads(&self, reads: &mut Needed) {
        for &key in self.keys.iter() {
            reads.need(key);
        }
        for input in self.aggregates.iter().filter_map(|aggregate| aggregate.input) {
            reads.need(input);
        }
    }
}

/// What a breaker that aggregates is given, once lowered, as Perfetto's
/// `HashAggregateSpec`.
pub struct HashAggregateSpec {
    /// Where the keys are in batches. With none, all rows are one group,
    /// output even if there are no rows.
    pub key_columns: SlowVec<u32>,
    /// Each aggregate.
    pub aggregates: SlowVec<AggregateCall>,
}

/// One aggregate a breaker computes, as Perfetto's `AggregateCall`.
#[derive(Clone, Copy)]
pub struct AggregateCall {
    /// What it computes.
    pub function: Function,
    /// Where the column it reads is in batches: unused by `COUNT(*)`.
    pub column: u32,
    /// What that column holds: unused by `COUNT(*)`.
    pub data_type: DataType,
}

impl AggregateCall {
    /// What it makes.
    pub fn output_type(&self) -> DataType {
        let input = (self.function != Function::CountStar).then_some(self.data_type);
        self.function.output_type(input).unwrap_or(DataType::Int64)
    }
}
