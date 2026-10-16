//! `LimitOp`: the first `count` rows of its child after skipping `offset`,
//! as `LIMIT count OFFSET offset` does; and `Limit`, the operator that runs
//! it, which stops its pipeline once it has them.

use crate::context::Context;
use crate::error::Error;
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::plan::{Op, PlanNode};
use crate::row_batch::RowBatch;
use crate::step::{DynOperator, Operator, Progress, Step};

pub struct LimitOp {
    pub count: u64,
    pub offset: u64,
}

impl<'c> Op<'c> for LimitOp {
    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        lowering.lower(*at!(node.children, 0))?;
        let limit = Limit { count: self.count, offset: self.offset };
        lowering
            .add_order_independent(Step::Operator(DynOperator::new(lowering.allocator(), limit)?))
    }

    /// Makes no columns, and reads none.
    fn prune(&mut self, _: &mut Needed) -> Pruned {
        Pruned::Keep
    }

    fn reads(&self, _: &mut Needed) {}
}

/// Keeps rows by narrowing each batch's selection, and reads no values.
struct Limit {
    count: u64,
    offset: u64,
}

/// How many rows are still to skip, and then to keep.
struct Left {
    skip: u64,
    keep: u64,
}

impl Operator for Limit {
    type State = Left;

    fn new_state(&self, _: &mut Context) -> Result<Left, Error> {
        Ok(Left { skip: self.offset, keep: self.count })
    }

    fn execute(
        &self,
        _: &mut Context,
        left: &mut Left,
        input: &RowBatch,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        if left.keep == 0 {
            return Ok(Progress::Done);
        }
        output.reset(input.row_count());
        for i in 0..input.column_count() {
            let Ok(()) = output.push_column(input.column(i).clone()) else {
                failed!();
            };
        }
        output.selection_mut().clone_from(input.selection());
        output.selection_mut().retain(|_| {
            if left.skip > 0 {
                left.skip -= 1;
                return false;
            }
            let kept = left.keep > 0;
            left.keep -= u64::from(kept);
            kept
        });
        Ok(if left.keep == 0 { Progress::Done } else { Progress::NeedInput })
    }

    fn finish(&self, _: &mut Context, _: &mut Left, _: &mut RowBatch) -> Result<Progress, Error> {
        Ok(Progress::Done)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::cell::Cell;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::{ColumnView, DataType};
    use crate::pipeline::Pipeline;
    use crate::query_allocators::QueryAllocators;
    use crate::selection::Kept;
    use crate::slow_vec::SlowVec;
    use crate::step::{DynSource, Source};

    /// `batches` batches of two rows, counting up from 0, counting in
    /// `pulled` how many were asked for.
    struct Counting<'a> {
        batches: i64,
        pulled: &'a Cell<i64>,
    }

    impl Source for Counting<'_> {
        type State = i64;

        fn new_state(&self, _: &mut Context) -> Result<i64, Error> {
            Ok(0)
        }

        fn next(&self, _: &mut Context, i: &mut i64, batch: &mut RowBatch) -> Result<bool, Error> {
            if *i == self.batches {
                return Ok(false);
            }
            self.pulled.set(self.pulled.get() + 1);
            batch.reset(2);
            let mut values = Buffer::allocate(&Heap, 16).unwrap();
            values.as_mut_slice::<i64>().copy_from_slice(&[2 * *i, 2 * *i + 1]);
            let column =
                ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, None).unwrap();
            assert!(batch.push_column(column).is_ok());
            *i += 1;
            Ok(true)
        }
    }

    /// The rows kept, and how many batches the source made, of 5.
    fn limited(count: u64, offset: u64) -> (Vec<i64>, i64) {
        let pulled = Cell::new(0);
        let source = DynSource::new(&Heap, Counting { batches: 5, pulled: &pulled }).unwrap();
        let mut steps = SlowVec::new(&Heap, 1).unwrap();
        let limit = DynOperator::new(&Heap, Limit { count, offset }).unwrap();
        assert!(steps.push(Step::Operator(limit)).is_ok());
        let pipeline = Pipeline::new(source, steps);
        let query = QueryAllocators::new(&Heap);
        let mut execution = pipeline.start(&query).unwrap();
        let mut batch = RowBatch::new();
        let mut rows = Vec::new();
        while execution.next(&mut batch).unwrap() {
            let values = batch.column(0).int64s();
            match batch.selection().kept() {
                Kept::All => rows.extend_from_slice(values),
                Kept::Select(kept) => rows.extend(kept.iter().map(|&row| values[usize::from(row)])),
                Kept::None => {}
            }
        }
        drop(execution);
        (rows, pulled.get())
    }

    #[test]
    fn keeps_rows_after_an_offset_and_stops_reading() {
        // Rows 1 to 3 are in the first two batches, so no more are read.
        assert_eq!(limited(3, 1), ([1, 2, 3].to_vec(), 2));
        assert_eq!(limited(100, 8), ([8, 9].to_vec(), 5));
        assert_eq!(limited(0, 0), (Vec::new(), 1));
    }
}
