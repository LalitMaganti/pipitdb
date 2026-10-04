//! `Scannable`: data a pipeline can read, such as a table, and `DynScannable`,
//! the form a `Catalog` hands out.

use core::alloc::Layout;
use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::boxed::{Box, ErasedBox};
use crate::buffer::BUFFER_ALIGNMENT_BYTES;
use crate::column::DataType;
use crate::context::Context;
use crate::erase::{drop_state, state_of, value_of, write_state};
use crate::row_batch::{BATCH_COLUMNS_MAX, RowBatch};
use crate::step::{DynSource, NewState};
use crate::vec::Vec;

/// Named, typed columns, read into batches. Where a read is lives in
/// `State`, which each run creates, as for a step.
pub trait Scannable {
    /// Where a read is, such as a row group and a row in it.
    type State;

    /// How many columns there are, numbered from 0.
    fn column_count(&self) -> u32;

    /// What frontends call `column`, such as in `SELECT`.
    fn column_name(&self, column: u32) -> &str;

    /// What `column` holds. Every batch's `column` has this type.
    fn column_type(&self, column: u32) -> DataType;

    /// The state of a read from the first row.
    fn new_state(&self, context: &mut Context) -> Result<Self::State, AllocError>;

    /// Fills `batch`, which is empty when called, with the next rows of
    /// `columns`, in that order, or returns false when no rows are left.
    fn next(
        &self,
        columns: &[u32],
        context: &mut Context,
        state: &mut Self::State,
        batch: &mut RowBatch,
    ) -> Result<bool, AllocError>;
}

/// The tables a frontend can read, provided by the embedder.
pub trait Catalog {
    /// What's registered as `name`, if anything.
    fn find(&self, name: &str) -> Option<&DynScannable<'_>>;
}

type ScannableNext = unsafe fn(
    NonNull<()>,
    &[u32],
    &mut Context,
    NonNull<u8>,
    &mut RowBatch,
) -> Result<bool, AllocError>;

/// A `Scannable` of any type that lives for `'a`, owned in memory from an
/// allocator, and functions that know its type.
pub struct DynScannable<'a> {
    scannable: ErasedBox,
    column_count: unsafe fn(NonNull<()>) -> u32,
    column_name: unsafe fn(NonNull<()>, u32) -> *const str,
    column_type: unsafe fn(NonNull<()>, u32) -> DataType,
    state_layout: Layout,
    new_state: NewState,
    drop_state: unsafe fn(NonNull<u8>),
    next: ScannableNext,
    lifetime: PhantomData<&'a ()>,
}

impl<'a> DynScannable<'a> {
    pub fn new<T: Scannable + 'a>(
        allocator: &dyn Allocator,
        scannable: T,
    ) -> Result<DynScannable<'a>, AllocError> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        Ok(DynScannable {
            scannable: Box::new(allocator, scannable)?.erase(),
            // SAFETY: only called with this scannable and its state, as is each below.
            column_count: |scannable| unsafe { value_of::<T>(scannable).column_count() },
            // SAFETY: as above.
            column_name: |scannable, column| unsafe {
                value_of::<T>(scannable).column_name(column)
            },
            // SAFETY: as above.
            column_type: |scannable, column| unsafe {
                value_of::<T>(scannable).column_type(column)
            },
            state_layout: Layout::new::<T::State>(),
            // SAFETY: as above.
            new_state: |scannable, context, state| unsafe {
                let made = value_of::<T>(scannable).new_state(context)?;
                write_state(state, made);
                Ok(())
            },
            drop_state: drop_state::<T::State>,
            // SAFETY: as above.
            next: |scannable, columns, context, state, batch| unsafe {
                let state = state_of::<T::State>(state);
                value_of::<T>(scannable).next(columns, context, state, batch)
            },
            lifetime: PhantomData,
        })
    }

    pub fn column_count(&self) -> u32 {
        // SAFETY: the function matches the scannable's type.
        unsafe { (self.column_count)(self.scannable.as_ptr()) }
    }

    pub fn column_name(&self, column: u32) -> &str {
        // SAFETY: as above, and the name lives as long as the scannable.
        unsafe { &*(self.column_name)(self.scannable.as_ptr(), column) }
    }

    pub fn column_type(&self, column: u32) -> DataType {
        // SAFETY: as above.
        unsafe { (self.column_type)(self.scannable.as_ptr(), column) }
    }

    /// A source of `columns` of this, in that order.
    pub fn scan(
        &self,
        allocator: &dyn Allocator,
        columns: Vec<u32>,
    ) -> Result<DynSource<'_>, AllocError> {
        check!(columns.len() <= BATCH_COLUMNS_MAX as usize);
        check!(columns.iter().all(|&column| column < self.column_count()));
        let scan = Scan { scannable: NonNull::from(self).cast(), columns };
        let step = Box::new(allocator, scan)?.erase();
        // SAFETY: the functions take a `Scan` and the scannable's state, and
        // the `Scan` borrows `self` for as long as the source lives.
        Ok(unsafe {
            DynSource::from_parts(
                step,
                self.state_layout,
                scan_new_state,
                self.drop_state,
                scan_next,
            )
        })
    }
}

/// A source's step that reads a `DynScannable`.
struct Scan {
    scannable: NonNull<DynScannable<'static>>,
    columns: Vec<u32>,
}

unsafe fn scan_new_state(
    step: NonNull<()>,
    context: &mut Context,
    state: NonNull<u8>,
) -> Result<(), AllocError> {
    // SAFETY: `step` is a `Scan`, whose scannable outlives it.
    let scannable = unsafe { step.cast::<Scan>().as_ref().scannable.as_ref() };
    // SAFETY: the function matches the scannable's type.
    unsafe { (scannable.new_state)(scannable.scannable.as_ptr(), context, state) }
}

unsafe fn scan_next(
    step: NonNull<()>,
    context: &mut Context,
    state: NonNull<u8>,
    batch: &mut RowBatch,
) -> Result<bool, AllocError> {
    // SAFETY: as in `scan_new_state`.
    let scan = unsafe { step.cast::<Scan>().as_ref() };
    // SAFETY: as in `scan_new_state`.
    let scannable = unsafe { scan.scannable.as_ref() };
    // SAFETY: the function matches the scannable's type, and `state` holds
    // its state.
    unsafe { (scannable.next)(scannable.scannable.as_ptr(), &scan.columns, context, state, batch) }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec as StdVec;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::ColumnView;
    use crate::pipeline::Pipeline;

    /// One batch, with column `i` holding `i` and `i + 10`.
    struct Columns;

    impl Scannable for Columns {
        type State = bool;

        fn column_count(&self) -> u32 {
            3
        }

        fn column_name(&self, column: u32) -> &str {
            ["a", "b", "c"][column as usize]
        }

        fn column_type(&self, _: u32) -> DataType {
            DataType::Int64
        }

        fn new_state(&self, _: &mut Context) -> Result<bool, AllocError> {
            Ok(false)
        }

        fn next(
            &self,
            columns: &[u32],
            _: &mut Context,
            done: &mut bool,
            batch: &mut RowBatch,
        ) -> Result<bool, AllocError> {
            if *done {
                return Ok(false);
            }
            batch.reset(2);
            for &column in columns {
                let mut values = Buffer::allocate(&Heap, 16).unwrap();
                let value = i64::from(column);
                values.as_mut_slice::<i64>().copy_from_slice(&[value, value + 10]);
                assert!(batch.push_column(ColumnView::new(DataType::Int64, values, None)).is_ok());
            }
            *done = true;
            Ok(true)
        }
    }

    #[test]
    fn describes_its_columns() {
        let scannable = DynScannable::new(&Heap, Columns).unwrap();
        assert_eq!(scannable.column_count(), 3);
        assert_eq!(scannable.column_name(1), "b");
        assert!(scannable.column_type(2) == DataType::Int64);
    }

    #[test]
    fn scans_chosen_columns_through_a_pipeline() {
        let scannable = DynScannable::new(&Heap, Columns).unwrap();
        let columns = Vec::fixed_from(&Heap, [2, 0].into_iter()).unwrap();
        let pipeline =
            Pipeline::new(scannable.scan(&Heap, columns).unwrap(), Vec::new(&Heap, 1).unwrap());
        let mut execution = pipeline.start(&Heap).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch).unwrap());
        let rows: StdVec<&[i64]> = (0..2).map(|i| batch.column(i).int64s()).collect();
        assert_eq!(rows, [[2, 12], [0, 10]]);
        assert!(!execution.next(&mut batch).unwrap());
    }
}
