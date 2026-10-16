//! `SortedRows`: rows held so a breaker can output them in another order,
//! as Perfetto's are: their key bytes, kept as rows arrive, and an order of
//! row numbers, sorted by the keys.

use crate::allocator::{AllocError, Allocator};
use crate::context::Context;
use crate::error::Error;
use crate::radix_sort::sort_by_key;
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::row_layout::{KeyColumn, RowLayout};
use crate::row_store::RowStore;
use crate::selection::Kept;
use crate::slow_vec::SlowVec;

/// Zeros, to make room for keys with.
const ZEROS: [u8; 256] = [0; 256];

pub(crate) struct SortedRows {
    /// How each row's keys are laid out in `keys`.
    layout: RowLayout,
    /// Each row's keys, one row after another.
    keys: SlowVec<u8>,
    /// The rows themselves.
    rows: RowStore,
    /// The row numbers in output order, once sorted.
    order: SlowVec<u32>,
}

impl SortedRows {
    pub(crate) fn new(
        allocator: &dyn Allocator,
        keys: &[KeyColumn],
    ) -> Result<SortedRows, AllocError> {
        Ok(SortedRows {
            layout: RowLayout::new(allocator, keys)?,
            keys: SlowVec::new(allocator, 1 << 31)?,
            rows: RowStore::new(allocator)?,
            order: SlowVec::new(allocator, 1 << 31)?,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    /// Holds the rows `batch` keeps, and their keys, the columns at
    /// `positions`, one for each key column the rows were made with.
    pub(crate) fn append(
        &mut self,
        context: &mut Context,
        batch: &RowBatch,
        positions: &[u32],
    ) -> Result<(), Error> {
        let select = match batch.selection().kept() {
            Kept::None => return Ok(()),
            Kept::All => &[][..],
            Kept::Select(select) => select,
        };
        let rows = batch.selection().len() as usize;
        let start = self.keys.len();
        let stride = self.layout.stride();
        // Room for the keys, written below, a chunk at a time.
        let mut left = rows * stride;
        while left > 0 {
            let chunk = left.min(ZEROS.len());
            self.keys.extend_from_slice(at!(ZEROS, ..chunk))?;
            left -= chunk;
        }
        let out = at_mut!(self.keys, start..start + rows * stride);
        for (k, &position) in positions.iter().enumerate() {
            self.layout.write(k, batch.column(position), select, out);
        }
        self.rows.append(context, batch)
    }

    /// Orders the rows by their keys, stably.
    pub(crate) fn sort(&mut self, allocator: &dyn Allocator) -> Result<(), Error> {
        self.order.retain(|_| false);
        for row in 0..self.rows.len() {
            #[expect(clippy::cast_possible_truncation, reason = "a store holds under 2^32 rows")]
            self.order.push(row as u32)?;
        }
        Ok(sort_by_key(allocator, &mut self.order, &self.keys, self.layout.stride())?)
    }

    /// Fills `out` with the rows from `at` in the order, at most a batch's,
    /// returning how many.
    pub(crate) fn view(
        &self,
        context: &mut Context,
        out: &mut RowBatch,
        at: usize,
    ) -> Result<usize, Error> {
        let end = self.order.len().min(at + BATCH_ROWS_MAX as usize);
        let rows = at!(self.order, at.min(end)..end);
        self.rows.view(context, out, rows)?;
        Ok(rows.len())
    }
}
