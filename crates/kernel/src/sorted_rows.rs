//! `SortedRows`: rows held so a breaker can output them in another order, as
//! Perfetto's `core/exec/sorted_rows` holds them. Key values are laid out as
//! rows arrive, the output order is a list of row numbers sorted by them,
//! and rows are copied out once, in that order.

use crate::allocator::{AllocError, Allocator};
use crate::column::DataType;
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::row_layout::{self, RowLayout};
use crate::row_layout_sort::sort_row_layout;
use crate::row_store::RowStore;
use crate::selection::Kept;
use crate::slow_vec::SlowVec;

/// A column to sort by, and its direction.
#[derive(Clone, Copy)]
pub struct SortKey {
    pub column: u32,
    pub descending: bool,
}

pub(crate) struct SortedRows {
    /// How each row's keys are laid out in `keys`, once the first batch says
    /// their types.
    layout: Option<RowLayout>,
    /// Each key's type, which every batch's must match.
    types: SlowVec<DataType>,
    /// Each row's keys, one row after another.
    keys: SlowVec<u8>,
    /// The rows themselves.
    rows: RowStore,
    /// The row numbers in output order.
    order: SlowVec<u32>,
}

impl SortedRows {
    pub(crate) fn new(allocator: &dyn Allocator) -> Result<SortedRows, AllocError> {
        Ok(SortedRows {
            layout: None,
            types: SlowVec::new(allocator, 1 << 6)?,
            keys: SlowVec::new(allocator, 1 << 31)?,
            rows: RowStore::new(allocator)?,
            order: SlowVec::new(allocator, 1 << 31)?,
        })
    }

    pub(crate) fn len(&self) -> u32 {
        self.rows.len()
    }

    /// Holds the rows `batch` keeps and their values for `keys`, which must
    /// be the same for every batch. Fails if a key isn't a column of numbers,
    /// or not of the type it was in the first batch.
    pub(crate) fn append(
        &mut self,
        context: &mut Context,
        batch: &RowBatch,
        keys: &[SortKey],
    ) -> Result<(), Error> {
        let count = batch.selection().len();
        if count == 0 {
            return Ok(());
        }
        self.check_types(batch, keys)?;
        let layout = match &self.layout {
            Some(layout) => layout,
            None => self.layout.insert(lay_out(context.allocator(), batch, keys)?),
        };
        let stride = layout.stride() as usize;
        let at = self.keys.len();
        self.keys.extend_zeroed(count as usize * stride)?;
        let rows = at_mut!(self.keys, at..);
        let kept = batch.selection().kept();
        let row = |i: u32| match kept {
            Kept::Select(rows) => u32::from(*at!(rows, i as usize)),
            Kept::Prefix | Kept::None => i,
        };
        for (k, key) in keys.iter().enumerate() {
            let column = batch.column(key.column);
            let slot = layout.slot(k);
            let valid = |i: u32| !column.values().is_null(row(i));
            match column.data_type() {
                DataType::Int64 => {
                    let values = column.int64s();
                    let get = |i| valid(i).then(|| *at!(values, row(i) as usize));
                    RowLayout::write(slot, count, get, rows);
                }
                DataType::Float64 => {
                    let values = column.float64s();
                    let get = |i| valid(i).then(|| *at!(values, row(i) as usize));
                    RowLayout::write(slot, count, get, rows);
                }
                DataType::String => failed!(),
            }
        }
        self.rows.append(context, batch)
    }

    /// Fails unless each key's column has the type it had in the first batch,
    /// as Perfetto checks.
    fn check_types(&mut self, batch: &RowBatch, keys: &[SortKey]) -> Result<(), Error> {
        let data_type = |key: &SortKey| batch.column(key.column).data_type();
        if self.types.is_empty() {
            for key in keys {
                self.types.push(data_type(key))?;
            }
        }
        if keys.iter().zip(self.types.iter()).any(|(key, &t)| data_type(key) != t) {
            return Err(Error::Unsupported);
        }
        Ok(())
    }

    /// Resets the order to the order rows came in, then sorts it stably by
    /// the keys. Nulls sort as in SQLite: first ascending, last descending.
    pub(crate) fn sort(&mut self, allocator: &dyn Allocator) -> Result<(), Error> {
        while self.order.pop().is_some() {}
        for row in 0..self.rows.len() {
            self.order.push(row)?;
        }
        let Some(layout) = &self.layout else { return Ok(()) };
        if layout.stride() > 0 {
            sort_row_layout(allocator, &self.keys, layout.stride(), &mut self.order)?;
        }
        Ok(())
    }

    /// Copies up to a batch's worth of rows into `out`, from position `at` in
    /// the order, returning how many.
    pub(crate) fn gather(
        &mut self,
        context: &mut Context,
        out: &mut RowBatch,
        at: u32,
    ) -> Result<u32, Error> {
        let end = self.order.len().min(at as usize + BATCH_ROWS_MAX as usize);
        let rows = at!(self.order, (at as usize).min(end)..end);
        self.rows.gather(context, out, rows)
    }
}

/// The layout of `keys` of `batch`'s columns. Every key is laid out as
/// nullable, as whether a column has nulls can change from batch to batch.
fn lay_out(
    allocator: &dyn Allocator,
    batch: &RowBatch,
    keys: &[SortKey],
) -> Result<RowLayout, Error> {
    let mut columns = SlowVec::fixed(allocator, keys.len())?;
    for key in keys {
        let data_type = match batch.column(key.column).data_type() {
            DataType::Int64 => row_layout::Type::Int64,
            DataType::Float64 => row_layout::Type::Float64,
            DataType::String => return Err(Error::Unsupported),
        };
        columns.push(row_layout::Column {
            data_type,
            nullable: true,
            descending: key.descending,
        })?;
    }
    Ok(RowLayout::new(allocator, &columns)?)
}
