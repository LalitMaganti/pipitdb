//! `DistinctOp`: its child's rows, but only the first of each set of rows
//! equal in every one of its columns, as Perfetto's `Distinct`; and
//! `Distinct`, the transform that runs it.

use crate::column::{ColumnView, DataType, Forms};
use crate::context::Context;
use crate::error::Error;
use crate::group_table::GroupTable;
pub use crate::group_table::KEYS_MAX;
use crate::lower::{LowerError, Lowering};
use crate::optimize::Needed;
use crate::plan::{ColumnId, Op, PlanNode};
use crate::row_batch::RowBatch;
use crate::selection::Kept;
use crate::slow_vec::SlowVec;
use crate::step::{DynTransform, Step, Transform};

/// The rows of its child that are the first with their values in `columns`,
/// in the order they come in. Rows are compared as in `GroupTable`: nulls
/// are equal to each other.
pub struct DistinctOp {
    /// The `Int64` columns rows are compared by: all of a row's, for pipe
    /// syntax's `DISTINCT`.
    pub columns: SlowVec<ColumnId>,
}

impl<'c> Op<'c> for DistinctOp {
    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        check!(!self.columns.is_empty() && self.columns.len() <= KEYS_MAX);
        lowering.lower(*at!(node.children, 0))?;
        check!(self.columns.iter().all(|&id| lowering.query_type(id) == DataType::Int64));
        let allocator = lowering.allocator();
        let positions = self.columns.iter().map(|&id| lowering.query_position(id));
        let columns = SlowVec::fixed_from(allocator, positions)?;
        let distinct = DynTransform::new(allocator, Distinct { columns })?;
        lowering.add_order_independent(Step::Transform(distinct))
    }

    /// Every one of its columns, which decide which rows it keeps, whether or
    /// not anything after reads them.
    fn reads(&self, reads: &mut Needed) {
        for &column in self.columns.iter() {
            reads.need(column);
        }
    }

    fn accepts(&self, _: ColumnId) -> Forms {
        Forms::FLAT
    }
}

/// Drops each row whose values in `columns` an earlier row had, by narrowing
/// the selection: nothing is copied. Only the keys of the rows kept are
/// stored, so memory grows with the distinct rows, not all of them.
struct Distinct {
    /// Where the columns compared are in batches.
    columns: SlowVec<u32>,
}

impl Transform for Distinct {
    /// Every key seen so far, numbered in the order first seen.
    type State = GroupTable;

    fn new_state(&self, context: &mut Context) -> Result<GroupTable, Error> {
        Ok(GroupTable::new(context.allocator(), self.columns.len(), 0)?)
    }

    fn process(
        &self,
        context: &mut Context,
        keys: &mut GroupTable,
        batch: &mut RowBatch,
    ) -> Result<(), Error> {
        let count = batch.selection().len() as usize;
        if count == 0 {
            return Ok(());
        }
        let mut columns: [&ColumnView; KEYS_MAX] = [batch.column(0); KEYS_MAX];
        for (column, &position) in columns.iter_mut().zip(self.columns.iter()) {
            *column = batch.column(position);
        }
        let columns = at!(columns, ..self.columns.len());
        let select = match batch.selection().kept() {
            Kept::Select(select) => select,
            Kept::Prefix | Kept::None => &[],
        };
        let mut numbers = context.indices_buffer(count * 4)?;
        let numbers = at_mut!(numbers.as_mut_slice::<u32>(), ..count);
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        let mut next = keys.len() as u32;
        keys.find_or_create(context, columns, select, numbers)?;
        if keys.len() == next as usize + count {
            return Ok(());
        }
        // Keys are numbered in the order first seen, so a row is the first
        // with its key exactly when it got the next number.
        let mut numbers = numbers.iter();
        batch.selection_mut().retain(|_| {
            let first = numbers.next() == Some(&next);
            next += u32::from(first);
            first
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;

    /// An `Int64` column of `values`, null where `None`.
    fn column(values: &[Option<i64>]) -> ColumnView {
        let mut words = Buffer::allocate(&Heap, values.len() * 8).unwrap();
        let mut bits = Buffer::allocate(&Heap, values.len().div_ceil(8)).unwrap();
        for (i, value) in values.iter().enumerate() {
            words.as_mut_slice::<i64>()[i] = value.unwrap_or(0);
            bits.as_mut_slice::<u8>()[i / 8] |= u8::from(value.is_some()) << (i % 8);
        }
        let validity = values.iter().any(Option::is_none).then_some(bits);
        ColumnView::new(&mut Context::new(&Heap), DataType::Int64, words, validity).unwrap()
    }

    /// A batch of two columns, and the rows dropped from it.
    type Batch<'a> = (&'a [Option<i64>], &'a [Option<i64>], &'a [u16]);

    /// The rows `distinct` keeps of each batch in turn, as positions in their
    /// batch.
    fn kept(batches: &[Batch]) -> Vec<Vec<u16>> {
        let distinct =
            Distinct { columns: SlowVec::fixed_from(&Heap, [0, 1].into_iter()).unwrap() };
        let mut context = Context::new(&Heap);
        let mut keys = distinct.new_state(&mut context).unwrap();
        let mut out = Vec::new();
        for &(a, b, dropped) in batches {
            let mut batch = RowBatch::new();
            batch.reset(u32::try_from(a.len()).unwrap());
            assert!(batch.push_column(column(a)).is_ok() && batch.push_column(column(b)).is_ok());
            batch.selection_mut().retain(|row| !dropped.contains(&row));
            distinct.process(&mut context, &mut keys, &mut batch).unwrap();
            out.push(match batch.selection().kept() {
                Kept::Prefix => (0..u16::try_from(a.len()).unwrap()).collect(),
                Kept::None => Vec::new(),
                Kept::Select(rows) => rows.to_vec(),
            });
        }
        out
    }

    #[test]
    fn keeps_the_first_row_of_each_key_across_batches() {
        let a: &[Option<i64>] = &[Some(1), Some(1), None, None, Some(1), Some(2)];
        let b: &[Option<i64>] = &[Some(5), Some(5), None, None, Some(6), Some(5)];
        // Row 0 is dropped before, so row 1 is the first `(1, 5)`, and nulls
        // are equal to each other.
        let first = (a, b, &[0_u16][..]);
        // Seen in the first batch, but for `(3, 3)`, which comes twice.
        let second: Batch =
            (&[Some(1), Some(3), Some(3), None], &[Some(6), Some(3), Some(3), None], &[]);
        assert_eq!(kept(&[first, second]), [[1, 2, 4, 5].to_vec(), [1].to_vec()]);
        // A batch of only new keys keeps every row.
        let new: Batch = (&[Some(1)], &[Some(2)], &[]);
        assert_eq!(kept(&[new]), [[0].to_vec()]);
    }
}
