//! `AggregateOp`: aggregates, such as `COUNT(*)` and `SUM(x)`, over each
//! group of its child's rows with the same keys, or over all of them if it
//! has none. `hash_aggregate` makes the breaker that runs it.

use crate::allocator::{AllocError, Allocator};
use crate::column::{DataType, Forms};
use crate::group_table::KEYS_MAX;
use crate::hash_aggregate::hash_aggregate;
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{DeriveContext, InputNeeds, OutputProperties};
use crate::plan::{ColumnId, DynOp, NamedColumn, Op, PlanNode};
use crate::slow_vec::SlowVec;
use crate::step::DynSource;
use crate::step::Step;
use crate::values::Values;

/// An aggregate function, as Perfetto's `AggregateCall::Function`. Each reads
/// `Int64` values, and makes an `Int64` but for `Avg`.
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
    /// are none. Perfetto has no `AVG`: #170.
    Avg,
}

impl Function {
    /// The type of what it makes.
    pub fn output_type(self) -> DataType {
        if self == Function::Avg { DataType::Float64 } else { DataType::Int64 }
    }
}

/// An aggregate in a plan: its function, the column it reads, `None` only for
/// `COUNT(*)`, whether it reads only each group's distinct values, and the
/// column it makes.
#[derive(Clone, Copy)]
pub struct Aggregate {
    pub function: Function,
    pub input: Option<ColumnId>,
    pub distinct: bool,
    pub binding: NamedColumn,
}

/// Outputs a row for each group: its keys, then its aggregates, in the order
/// groups were first seen. With no keys, all rows are one group, and there's
/// a row even if there are none.
pub struct AggregateOp {
    /// The `Int64` and string columns rows are grouped by.
    pub keys: SlowVec<ColumnId>,
    /// What each group's rows make, from `Int64` columns.
    pub aggregates: SlowVec<Aggregate>,
    /// Each aggregate's value, once known without reading any rows: then it
    /// has no child, and outputs one row of them.
    pub answer: Option<SlowVec<i64>>,
}

impl<'c> Op<'c> for AggregateOp {
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
        if self.answer.is_some() {
            return;
        }
        for &key in self.keys.iter() {
            reads.need(key);
        }
        for input in self.aggregates.iter().filter_map(|aggregate| aggregate.input) {
            reads.need(input);
        }
    }

    /// Its keys in dictionaries too: a string key's entries are each
    /// interned once, rather than each row's. Only strings come in them.
    fn accepts(&self, column: ColumnId) -> Forms {
        if self.keys.contains(&column) { Forms::FLAT | Forms::DICTIONARY } else { Forms::FLAT }
    }

    /// False for every column: what comes out is one row per group, and any
    /// column it doesn't output is gone.
    fn passes_through(&self, _: ColumnId) -> bool {
        false
    }

    /// With no keys, and only `COUNT(*)`, `MIN` and `MAX`: from what `child`
    /// knows of its rows without reading them, if it knows enough.
    fn answer(&mut self, allocator: &dyn Allocator, child: &DynOp<'c>) -> Result<bool, AllocError> {
        if !self.keys.is_empty() {
            return Ok(false);
        }
        let mut values = SlowVec::fixed(allocator, self.aggregates.len())?;
        for aggregate in self.aggregates.iter() {
            // A summary's bounds are the least and greatest values. With no
            // rows, there are none, and MIN and MAX are null: scanned.
            let Some(summary) = child.summary(aggregate.input) else { return Ok(false) };
            let value = match aggregate.function {
                Function::CountStar => Some(summary.rows),
                Function::Min => summary.bounds.map(|b| b.min),
                Function::Max => summary.bounds.map(|b| b.max),
                _ => None,
            };
            let Some(value) = value else { return Ok(false) };
            values.push(value)?;
        }
        self.answer = Some(values);
        Ok(true)
    }

    /// What its aggregates read must be `Int64`.
    fn needs(&self, _: u32) -> InputNeeds<'_> {
        let mut needs = InputNeeds::NONE;
        for column in self.aggregates.iter().filter_map(|aggregate| aggregate.input) {
            needs.types_then.push(column, DataType::Int64);
        }
        needs
    }

    /// A row per group: its keys, then its aggregates. Groups are output in
    /// the order first seen, so if the input is sorted by the keys, the
    /// output is too. Its keys must be `Int64` or strings, which are all the
    /// hash aggregate groups by.
    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        for &key in self.keys.iter() {
            if !matches!(c.input(0).type_of(key), DataType::Int64 | DataType::String) {
                return Err(LowerError::WrongType(key));
            }
        }
        check!(self.keys.len() <= KEYS_MAX);
        let mut out = OutputProperties::NONE;
        for &key in self.keys.iter() {
            out.push_column(c.column_of(key))?;
        }
        for aggregate in self.aggregates.iter() {
            out.push_column(c.column_of(aggregate.binding.id))?;
        }
        if self.answer.is_none() {
            let sorted = c.input(0).sorted_by(&self.keys);
            if !sorted.is_empty() {
                out.arrangements.push(&[], sorted);
            }
        }
        Ok(out)
    }

    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        let allocator = lowering.allocator();
        if let Some(answer) = &self.answer {
            let values = SlowVec::fixed_from(allocator, answer.iter().copied())?;
            lowering.set_source(DynSource::new(allocator, Values { values })?);
            return Ok(());
        }
        check!(node.children.len() == 1);
        lowering.lower_input(node, 0)?;
        let mut string_keys = 0;
        for (k, &key) in self.keys.iter().enumerate() {
            string_keys |= u64::from(lowering.query_rows().type_of(key) == DataType::String) << k;
        }
        let keys = self.keys.iter().map(|&key| lowering.query_position(key));
        let key_columns = SlowVec::fixed_from(allocator, keys)?;
        let aggregates = self.aggregates.iter().map(|aggregate| AggregateCall {
            function: aggregate.function,
            distinct: aggregate.distinct,
            column: aggregate.input.map_or(0, |id| lowering.query_position(id)),
        });
        let aggregates = SlowVec::fixed_from(allocator, aggregates)?;
        let spec = HashAggregateSpec { key_columns, string_keys, aggregates };
        let operator = hash_aggregate(allocator, spec)?;
        lowering.add_operator(Step::Operator(operator))
    }
}

/// What a breaker that aggregates is given, once lowered, as Perfetto's
/// `HashAggregateSpec`.
pub struct HashAggregateSpec {
    /// Where the keys are in batches. With none, all rows are one group,
    /// output even if there are no rows.
    pub key_columns: SlowVec<u32>,
    /// Bit `k` set if key `k` is a string, rather than an `Int64`. Perfetto
    /// learns this from the first batch, but a group-by here serves a batch
    /// with its keys' types even if none came.
    pub string_keys: u64,
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
    /// Whether it reads only the values of each group it hasn't before, as
    /// `COUNT(DISTINCT x)` does. Perfetto's has no such flag: #288.
    pub distinct: bool,
}
