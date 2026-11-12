//! `LimitOp`: the first `count` rows of its child after skipping `offset`,
//! as `LIMIT count OFFSET offset` does; and `Limit`, the operator that runs
//! it, which stops its pipeline once it has them.

use crate::context::Context;
use crate::error::Error;
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{DeriveContext, OutputProperties};
use crate::plan::{Op, PlanNode};
use crate::row_batch::RowBatch;
use crate::step::{DynOperator, Operator, Progress, Step};

pub struct LimitOp {
    pub count: u64,
    pub offset: u64,
}

impl<'c> Op<'c> for LimitOp {
    /// Its input's, less some rows. The sort just before, if any, only needs
    /// to output the rows read here.
    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        let mut out = *c.input(0);
        if let Some(sort) = out.open_sort {
            let read = self.count.saturating_add(self.offset);
            c.limit_sort(sort, u32::try_from(read).unwrap_or(u32::MAX));
        }
        out.drop_rows();
        Ok(out)
    }

    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        lowering.lower_input(node, 0)?;
        // A sort which outputs only the rows read here leaves nothing to drop.
        let kept = lowering.query_sort_keep().map(u64::from);
        if self.offset == 0 && kept == Some(self.count) {
            return Ok(());
        }
        let limit = Limit { count: self.count, offset: self.offset };
        lowering.add_operator(Step::Operator(DynOperator::new(lowering.allocator(), limit)?))
    }

    /// Makes no columns, and reads none.
    fn prune(&mut self, _: &mut Needed) -> Pruned {
        Pruned::Keep
    }

    fn reads(&self, _: &mut Needed) {}
}

/// Drops the first `offset` rows, then keeps up to `count` rows, in order, by
/// narrowing each batch's selection: nothing is copied, and no values read.
struct Limit {
    count: u64,
    offset: u64,
}

/// How many rows of the offset are dropped, and how many are kept.
struct Counted {
    skipped: u64,
    kept: u64,
}

impl Operator for Limit {
    type State = Counted;

    fn new_state(&self, _: &mut Context) -> Result<Counted, Error> {
        Ok(Counted { skipped: 0, kept: 0 })
    }

    #[expect(clippy::cast_possible_truncation, reason = "at most a batch's rows")]
    fn execute(
        &self,
        _: &mut Context,
        counted: &mut Counted,
        input: &RowBatch,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        let size = u64::from(input.selection().len());
        let skip = (self.offset - counted.skipped).min(size);
        counted.skipped += skip;
        let keep = (size - skip).min(self.count - counted.kept);
        counted.kept += keep;
        output.copy_from(input);
        if keep != size {
            output.selection_mut().keep_range(skip as u32, keep as u32);
        }
        Ok(if counted.kept == self.count { Progress::Done } else { Progress::NeedInput })
    }

    fn finish(
        &self,
        _: &mut Context,
        _: &mut Counted,
        _: &mut RowBatch,
    ) -> Result<Progress, Error> {
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
                Kept::Prefix => rows.extend_from_slice(&values[..batch.selection().len() as usize]),
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
        // Kept rows are a prefix of the batch, or a range of it after an offset.
        assert_eq!(limited(1, 0), ([0].to_vec(), 1));
        assert_eq!(limited(3, 3), ([3, 4, 5].to_vec(), 3));
    }
}
