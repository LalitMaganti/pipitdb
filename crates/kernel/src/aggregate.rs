//! `AggregateOp`: aggregates, such as `COUNT(*)` and `SUM(x)`, over all of
//! its child's rows, as one row; and `Aggregation`, the operator that runs it.

use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType};
use crate::context::Context;
use crate::error::Error;
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::plan::{ColumnId, NamedColumn, Op, PlanNode};
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::selection::{Kept, Selection};
use crate::slow_vec::SlowVec;
use crate::step::{DynOperator, Operator, Progress, Step};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Function {
    /// How many rows there are, or, of a column, how many aren't null: an
    /// `Int64`.
    Count,
    /// Of the values that aren't null, as are the rest: of the column's
    /// type, but `Float64` for `Avg`, and null if there are none.
    Sum,
    Min,
    Max,
    Avg,
}

impl Function {
    /// What it makes of a column of `data_type`, or `None` if it can't.
    pub fn output_type(self, data_type: Option<DataType>) -> Option<DataType> {
        match (self, data_type) {
            (Function::Count, _) => Some(DataType::Int64),
            (_, None | Some(DataType::String)) => None,
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

pub struct AggregateOp {
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
        let mut lowered = SlowVec::fixed(allocator, self.aggregates.len())?;
        for aggregate in self.aggregates.iter() {
            let input = aggregate.input.map(|id| (lowering.position(id), lowering.data_type(id)));
            lowered.push(Lowered { function: aggregate.function, input })?;
        }
        // Its batches are new, with only its own columns.
        lowering.restart_columns();
        for aggregate in self.aggregates.iter() {
            lowering.define(aggregate.binding.id)?;
        }
        let operator = DynOperator::new(allocator, Aggregation { aggregates: lowered })?;
        lowering.add_step(Step::Operator(operator))
    }

    /// Drops the aggregates nothing reads, keeping at least one, and reads the
    /// columns of those left.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        let any = self.aggregates.iter().any(|aggregate| needed.is_needed(aggregate.binding.id));
        let mut first = true;
        self.aggregates.retain(|aggregate| {
            let keep = needed.is_needed(aggregate.binding.id) || (!any && first);
            first = false;
            keep
        });
        for aggregate in self.aggregates.iter() {
            if let Some(input) = aggregate.input {
                needed.need(input);
            }
        }
        Pruned::Keep
    }
}

/// An aggregate once lowered: its function, and where its column is in
/// batches, and of what type.
#[derive(Clone, Copy)]
struct Lowered {
    function: Function,
    input: Option<(u32, DataType)>,
}

/// Outputs one row, of each aggregate in turn, once its input ends.
struct Aggregation {
    aggregates: SlowVec<Lowered>,
}

/// An aggregate's running total: how many values it's seen, and their sum,
/// least or greatest so far, as an integer or a float by the column's type.
#[derive(Clone, Copy, Default)]
struct Total {
    count: u64,
    int: i128,
    float: f64,
}

/// The totals so far, and room to gather a batch's values into.
struct Totals {
    totals: SlowVec<Total>,
    gathered: Buffer,
}

impl Operator for Aggregation {
    type State = Totals;

    fn new_state(&self, context: &mut Context) -> Result<Totals, Error> {
        let totals = core::iter::repeat_n(Total::default(), self.aggregates.len());
        let totals = SlowVec::fixed_from(context.allocator(), totals)?;
        let gathered = Buffer::allocate(context.allocator(), BATCH_ROWS_MAX as usize * 8)?;
        Ok(Totals { totals, gathered })
    }

    fn execute(
        &self,
        _: &mut Context,
        totals: &mut Totals,
        input: &RowBatch,
        _: &mut RowBatch,
    ) -> Result<Progress, Error> {
        let selection = input.selection();
        let gathered = totals.gathered.as_mut_slice::<i64>();
        for (aggregate, total) in self.aggregates.iter().zip(totals.totals.iter_mut()) {
            match aggregate.input {
                None => total.count += u64::from(selection.len()),
                Some((position, data_type)) => {
                    let column = input.column(position);
                    add(aggregate.function, data_type, column, selection, gathered, total);
                }
            }
        }
        Ok(Progress::NeedInput)
    }

    fn finish(
        &self,
        context: &mut Context,
        totals: &mut Totals,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        output.reset(1);
        for (aggregate, total) in self.aggregates.iter().zip(totals.totals.iter()) {
            let column = result(context, *aggregate, *total)?;
            let Ok(()) = output.push_column(column) else { crate::check::check_failed(line!()) };
        }
        Ok(Progress::NeedInput)
    }
}

/// Adds the kept rows of `column`, of `data_type`, that aren't null to
/// `total`, gathering them into `gathered` first unless they're all of them.
fn add(
    function: Function,
    data_type: DataType,
    column: &ColumnView,
    selection: &Selection,
    gathered: &mut [i64],
    total: &mut Total,
) {
    if function == Function::Count {
        total.count += count_valid(column, selection);
        return;
    }
    let all = column.words();
    let words = match (selection.kept(), column.validity()) {
        (Kept::All, None) => all,
        (kept, validity) => {
            let mut n = 0;
            let mut gather = |row: u32| {
                // SAFETY: kept rows are below the batch's row count, which is
                // the column's.
                if validity.is_none_or(|validity| unsafe { validity.is_valid_unchecked(row) }) {
                    *at_mut!(gathered, n) = *at!(all, row as usize);
                    n += 1;
                }
            };
            match kept {
                Kept::All => (0..selection.rows()).for_each(&mut gather),
                Kept::Select(rows) => rows.iter().for_each(|&row| gather(u32::from(row))),
                Kept::None => {}
            }
            at!(gathered, ..n)
        }
    };
    let Some(&first) = words.first() else { return };
    match data_type {
        DataType::Int64 => {
            let first = if total.count == 0 { i128::from(first) } else { total.int };
            total.int = match function {
                Function::Min => words.iter().fold(first, |min, &w| min.min(i128::from(w))),
                Function::Max => words.iter().fold(first, |max, &w| max.max(i128::from(w))),
                _ => total.int + words.iter().map(|&w| i128::from(w)).sum::<i128>(),
            };
        }
        DataType::Float64 => {
            let float = |w: &i64| f64::from_bits(w.cast_unsigned());
            let first = if total.count == 0 { float(&first) } else { total.float };
            total.float = match function {
                Function::Min => words.iter().map(float).fold(first, f64::min),
                Function::Max => words.iter().map(float).fold(first, f64::max),
                _ => total.float + words.iter().map(float).sum::<f64>(),
            };
        }
        DataType::String => crate::check::check_failed(line!()),
    }
    total.count += words.len() as u64;
}

/// How many kept rows of `column` aren't null.
fn count_valid(column: &ColumnView, selection: &Selection) -> u64 {
    let Some(validity) = column.validity() else { return u64::from(selection.len()) };
    // SAFETY: kept rows are below the batch's row count, which is the
    // column's.
    let valid = |row: u32| unsafe { validity.is_valid_unchecked(row) };
    let count = match selection.kept() {
        Kept::All => (0..selection.rows()).filter(|&row| valid(row)).count(),
        Kept::Select(rows) => rows.iter().filter(|&&row| valid(u32::from(row))).count(),
        Kept::None => 0,
    };
    count as u64
}

/// The one-row column of `aggregate`'s result.
#[expect(clippy::cast_precision_loss, reason = "averages are approximate")]
fn result(context: &mut Context, aggregate: Lowered, total: Total) -> Result<ColumnView, Error> {
    let data_type = aggregate.input.map(|(_, data_type)| data_type);
    let Some(output_type) = aggregate.function.output_type(data_type) else {
        crate::check::check_failed(line!());
    };
    let word = match (aggregate.function, data_type) {
        (Function::Count, _) => Some(i64::try_from(total.count).map_err(|_| Error::Unsupported)?),
        _ if total.count == 0 => None,
        (Function::Avg, Some(DataType::Int64)) => {
            Some((total.int as f64 / total.count as f64).to_bits().cast_signed())
        }
        (Function::Avg, _) => Some((total.float / total.count as f64).to_bits().cast_signed()),
        // A sum that doesn't fit in an `Int64` would need a wider type.
        (_, Some(DataType::Int64)) => {
            Some(i64::try_from(total.int).map_err(|_| Error::Unsupported)?)
        }
        _ => Some(total.float.to_bits().cast_signed()),
    };
    let mut values = context.column_buffer(8)?;
    *at_mut!(values.as_mut_slice::<i64>(), 0) = word.unwrap_or(0);
    let mut validity = None;
    if word.is_none() {
        let mut bits = context.column_buffer(1)?;
        *at_mut!(bits.as_mut_slice::<u8>(), 0) = 0;
        validity = Some(bits);
    }
    Ok(ColumnView::new(output_type, values, validity))
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec as StdVec;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::lower::lower;
    use crate::plan::{DynOp, LogicalPlan, ScanColumn, ScanOp};
    use crate::scannable::{DynScannable, Scannable};

    /// Two batches of `i` and `f`: 1, 2, null and 4, as integers and halves,
    /// then -3 and 5, of which only the first is kept. With `empty`, none.
    struct Numbers {
        empty: bool,
    }

    fn column(data_type: DataType, words: &[i64], nulls: &[usize]) -> ColumnView {
        let mut values = Buffer::allocate(&Heap, words.len() * 8).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(words);
        let validity = (!nulls.is_empty()).then(|| {
            let mut bits = Buffer::allocate(&Heap, 1).unwrap();
            bits.as_mut_slice::<u8>()[0] = nulls.iter().fold(0xff, |bits, &row| bits & !(1 << row));
            bits
        });
        ColumnView::new(data_type, values, validity)
    }

    impl Scannable for Numbers {
        type State = u32;

        fn column_count(&self) -> u32 {
            2
        }

        fn column_name(&self, column: u32) -> &str {
            ["i", "f"][column as usize]
        }

        fn column_type(&self, column: u32) -> DataType {
            [DataType::Int64, DataType::Float64][column as usize]
        }

        fn new_state(&self, _: &mut Context) -> Result<u32, Error> {
            Ok(0)
        }

        fn next(
            &self,
            columns: &[u32],
            _: &mut Context,
            batches: &mut u32,
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            let ints: &[i64] = match (*batches, self.empty) {
                (0, false) => &[1, 2, 0, 4],
                (1, false) => &[-3, 5],
                _ => return Ok(false),
            };
            batch.reset(u32::try_from(ints.len()).unwrap());
            let nulls: &[usize] = if *batches == 0 { &[2] } else { &[] };
            for &c in columns {
                let column = if c == 0 {
                    column(DataType::Int64, ints, nulls)
                } else {
                    let half = |&i: &i64| bits(f64::from(i32::try_from(i).unwrap()) / 2.0);
                    column(DataType::Float64, &ints.iter().map(half).collect::<StdVec<_>>(), nulls)
                };
                assert!(batch.push_column(column).is_ok());
            }
            if *batches == 1 {
                batch.selection_mut().retain(|row| row == 0);
            }
            *batches += 1;
            Ok(true)
        }
    }

    /// The one row `aggregates`, each a function and an input column, if
    /// any, make of `Numbers`: words, or `None` for nulls.
    fn aggregate(empty: bool, aggregates: &[(Function, Option<u32>)]) -> StdVec<Option<i64>> {
        let table = DynScannable::new(&Heap, Numbers { empty }).unwrap();
        let mut plan = LogicalPlan::new(&Heap).unwrap();
        let mut columns = SlowVec::fixed(&Heap, 2).unwrap();
        let mut inputs = StdVec::new();
        for (column, name, data_type) in [(0, "i", DataType::Int64), (1, "f", DataType::Float64)] {
            let binding = plan.add_column(name, data_type).unwrap();
            assert!(columns.push(ScanColumn { column, binding }).is_ok());
            inputs.push((binding.id, data_type));
        }
        let scan = DynOp::new(&Heap, ScanOp { scannable: &table, columns }).unwrap();
        let scan = plan.add_node(scan, SlowVec::fixed(&Heap, 0).unwrap()).unwrap();
        let mut list = SlowVec::fixed(&Heap, aggregates.len()).unwrap();
        for &(function, input) in aggregates {
            let input = input.map(|c| inputs[c as usize]);
            let data_type = function.output_type(input.map(|(_, data_type)| data_type)).unwrap();
            let binding = plan.add_column("out", data_type).unwrap();
            let input = input.map(|(id, _)| id);
            assert!(list.push(Aggregate { function, input, binding }).is_ok());
            assert!(plan.output.push(binding).is_ok());
        }
        let op = DynOp::new(&Heap, AggregateOp { aggregates: list }).unwrap();
        let children = SlowVec::fixed_from(&Heap, [scan].into_iter()).unwrap();
        plan.add_node(op, children).unwrap();
        crate::optimize::optimize(&Heap, &mut plan).unwrap();

        let physical = lower(&Heap, &plan).unwrap();
        let mut execution = physical.pipeline().start(&Heap).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch).unwrap());
        assert_eq!(batch.row_count(), 1);
        let row = (0..batch.column_count()).map(|c| {
            let column = batch.column(c);
            (!column.is_null(0)).then(|| match column.data_type() {
                DataType::Int64 => column.int64s()[0],
                _ => column.float64s()[0].to_bits().cast_signed(),
            })
        });
        let row = row.collect();
        assert!(!execution.next(&mut batch).unwrap());
        row
    }

    fn bits(value: f64) -> i64 {
        value.to_bits().cast_signed()
    }

    #[test]
    fn aggregates_kept_rows_that_arent_null() {
        use Function::{Avg, Count, Max, Min, Sum};
        let ints = [(Count, None), (Count, Some(0)), (Sum, Some(0)), (Min, Some(0))];
        assert_eq!(aggregate(false, &ints), [Some(5), Some(4), Some(4), Some(-3)]);
        let ints = [(Max, Some(0)), (Avg, Some(0))];
        assert_eq!(aggregate(false, &ints), [Some(4), Some(bits(1.0))]);
        let floats = [(Sum, Some(1)), (Min, Some(1)), (Max, Some(1)), (Avg, Some(1))];
        assert_eq!(aggregate(false, &floats), [2.0, -1.5, 2.0, 0.5].map(|f| Some(bits(f))));
    }

    #[test]
    fn aggregates_of_no_rows_are_null_but_counts() {
        use Function::{Count, Max, Sum};
        let aggregates = [(Count, None), (Count, Some(0)), (Sum, Some(0)), (Max, Some(1))];
        assert_eq!(aggregate(true, &aggregates), [Some(0), Some(0), None, None]);
    }
}
