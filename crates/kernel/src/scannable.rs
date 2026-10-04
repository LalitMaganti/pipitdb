//! `Scannable`: data a pipeline can read, such as a table, and `DynScannable`,
//! the form a `Catalog` hands out.

use core::alloc::Layout;
use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::boxed::{Box, ErasedBox};
use crate::buffer::BUFFER_ALIGNMENT_BYTES;
use crate::column::DataType;
use crate::row_batch::{BATCH_COLUMNS_MAX, RowBatch};
use crate::step::DynSource;
use crate::vec::Vec;

/// Named, typed columns, read into batches. Where a read is lives in
/// `State`, which each run creates, as for a step.
pub trait Scannable {
    type State;

    fn column_count(&self) -> u32;

    fn column_name(&self, column: u32) -> &str;

    fn column_type(&self, column: u32) -> DataType;

    fn new_state(&self) -> Self::State;

    /// Fills `batch`, which is empty when called, with the next rows of
    /// `columns`, in that order, or returns false when no rows are left.
    fn next(&self, columns: &[u32], batch: &mut RowBatch, state: &mut Self::State) -> bool;
}

/// The tables a frontend can read, provided by the embedder.
pub trait Catalog {
    /// What's registered as `name`, if anything.
    fn find(&self, name: &str) -> Option<&DynScannable<'_>>;
}

/// A `Scannable` of any type that lives for `'a`, owned in memory from an
/// allocator, and functions that know its type.
pub struct DynScannable<'a> {
    scannable: ErasedBox,
    column_count: unsafe fn(NonNull<()>) -> u32,
    column_name: unsafe fn(NonNull<()>, u32) -> *const str,
    column_type: unsafe fn(NonNull<()>, u32) -> DataType,
    state_layout: Layout,
    new_state: unsafe fn(NonNull<()>, NonNull<u8>),
    drop_state: unsafe fn(NonNull<u8>),
    next: unsafe fn(NonNull<()>, &[u32], &mut RowBatch, NonNull<u8>) -> bool,
    lifetime: PhantomData<&'a ()>,
}

impl<'a> DynScannable<'a> {
    pub fn new<A: Allocator + Clone + 'static, T: Scannable + 'a>(
        allocator: A,
        scannable: T,
    ) -> Result<DynScannable<'a>, AllocError> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        Ok(DynScannable {
            scannable: Box::new(allocator, scannable)?.erase(),
            column_count: column_count::<T>,
            column_name: column_name::<T>,
            column_type: column_type::<T>,
            state_layout: Layout::new::<T::State>(),
            new_state: new_state::<T>,
            drop_state: drop_state::<T::State>,
            next: next::<T>,
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
    pub fn scan<A: Allocator + Clone + 'static>(
        &self,
        allocator: A,
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

unsafe fn scan_new_state(step: NonNull<()>, state: NonNull<u8>) {
    // SAFETY: `step` is a `Scan`, whose scannable outlives it.
    let scannable = unsafe { step.cast::<Scan>().as_ref().scannable.as_ref() };
    // SAFETY: the function matches the scannable's type.
    unsafe { (scannable.new_state)(scannable.scannable.as_ptr(), state) }
}

unsafe fn scan_next(step: NonNull<()>, batch: &mut RowBatch, state: NonNull<u8>) -> bool {
    // SAFETY: as in `scan_new_state`.
    let scan = unsafe { step.cast::<Scan>().as_ref() };
    // SAFETY: as in `scan_new_state`.
    let scannable = unsafe { scan.scannable.as_ref() };
    // SAFETY: the function matches the scannable's type, and `state` holds
    // its state.
    unsafe { (scannable.next)(scannable.scannable.as_ptr(), &scan.columns, batch, state) }
}

// These undo the erasure. Each is only stored next to a pointer to a `T`, and
// only called with a state of `T`'s type.

unsafe fn column_count<T: Scannable>(scannable: NonNull<()>) -> u32 {
    // SAFETY: see above.
    unsafe { scannable.cast::<T>().as_ref().column_count() }
}

unsafe fn column_name<T: Scannable>(scannable: NonNull<()>, column: u32) -> *const str {
    // SAFETY: see above.
    unsafe { scannable.cast::<T>().as_ref().column_name(column) }
}

unsafe fn column_type<T: Scannable>(scannable: NonNull<()>, column: u32) -> DataType {
    // SAFETY: see above.
    unsafe { scannable.cast::<T>().as_ref().column_type(column) }
}

unsafe fn new_state<T: Scannable>(scannable: NonNull<()>, state: NonNull<u8>) {
    // SAFETY: see above.
    unsafe { state.cast().write(scannable.cast::<T>().as_ref().new_state()) }
}

unsafe fn drop_state<S>(state: NonNull<u8>) {
    // SAFETY: see above.
    unsafe { state.cast::<S>().drop_in_place() }
}

unsafe fn next<T: Scannable>(
    scannable: NonNull<()>,
    columns: &[u32],
    batch: &mut RowBatch,
    state: NonNull<u8>,
) -> bool {
    // SAFETY: see above.
    unsafe { scannable.cast::<T>().as_ref().next(columns, batch, state.cast().as_mut()) }
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

        fn new_state(&self) -> bool {
            false
        }

        fn next(&self, columns: &[u32], batch: &mut RowBatch, done: &mut bool) -> bool {
            if *done {
                return false;
            }
            batch.reset(2);
            for &column in columns {
                let mut values = Buffer::allocate(Heap, 16).unwrap();
                let value = i64::from(column);
                values.as_mut_slice::<i64>().copy_from_slice(&[value, value + 10]);
                assert!(batch.push_column(ColumnView::new(DataType::Int64, values, None)).is_ok());
            }
            *done = true;
            true
        }
    }

    #[test]
    fn describes_its_columns() {
        let scannable = DynScannable::new(Heap, Columns).unwrap();
        assert_eq!(scannable.column_count(), 3);
        assert_eq!(scannable.column_name(1), "b");
        assert!(scannable.column_type(2) == DataType::Int64);
    }

    #[test]
    fn scans_chosen_columns_through_a_pipeline() {
        let scannable = DynScannable::new(Heap, Columns).unwrap();
        let columns = Vec::fixed_from(Heap, [2, 0].into_iter()).unwrap();
        let pipeline =
            Pipeline::new(scannable.scan(Heap, columns).unwrap(), Vec::new(Heap, 1).unwrap());
        let mut execution = pipeline.start(Heap).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch));
        let rows: StdVec<&[i64]> = (0..2).map(|i| batch.column(i).int64s()).collect();
        assert_eq!(rows, [[2, 12], [0, 10]]);
        assert!(!execution.next(&mut batch));
    }
}
