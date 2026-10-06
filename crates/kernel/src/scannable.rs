//! `Scannable`: data a pipeline can read, such as a table, and `DynScannable`,
//! the form a `Catalog` hands out.
//!
//! A scannable can write some columns lazily, with a handle saying where
//! their values are, and load them later, for the rows still kept, when
//! something reads them. The optimizer says which, and where they're loaded.

use core::alloc::Layout;
use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::boxed::{Box, ErasedBox};
use crate::buffer::BUFFER_ALIGNMENT_BYTES;
use crate::column::{ColumnView, DataType, Forms};
use crate::context::Context;
use crate::erase::{drop_state, state_of, value_of, write_state};
use crate::error::Error;
use crate::row_batch::{BATCH_COLUMNS_MAX, RowBatch};
use crate::selection::Selection;
use crate::slow_vec::SlowVec;
use crate::step::{DynSource, DynTransform, ErasedBatch, NewState};

/// Named, typed columns, read into batches. Where a read is lives in
/// `State`, which each run opens, as for a step.
pub trait Scannable {
    /// Where a read is, such as a row group and a row in it. It borrows the
    /// scannable and the columns it was opened with, for `'s`.
    type State<'s>
    where
        Self: 's;

    /// What loading lazy columns keeps between batches, such as a reader for
    /// each column and what it has read. A scannable none of whose columns
    /// can be lazy has `()`.
    type Loader;

    /// How many columns there are, numbered from 0.
    fn column_count(&self) -> u32;

    /// What frontends call `column`, such as in `SELECT`.
    fn column_name(&self, column: u32) -> &str;

    /// What `column` holds. Every batch's `column` has this type.
    fn column_type(&self, column: u32) -> DataType;

    /// A read of `columns`, in that order, from the first row, each with
    /// the forms it may be written in besides flat, some of those `forms`
    /// says it can write it in. What depends only on which columns are read
    /// is worked out here, once, not for each batch.
    fn open<'s>(
        &'s self,
        context: &mut Context,
        columns: &'s [(u32, Forms)],
    ) -> Result<Self::State<'s>, Error>;

    /// Resets `batch`, which holds the last batch, and fills it with the
    /// next rows of the columns `state` was opened with, or returns false
    /// when no rows are left. Its columns may view what the scannable
    /// holds.
    fn next<'s>(
        &'s self,
        context: &mut Context,
        state: &mut Self::State<'s>,
        batch: &mut RowBatch<'s>,
    ) -> Result<bool, Error>;

    /// The forms `column` can be written in, besides flat. By default, none.
    fn forms(&self, column: u32) -> Forms {
        let _ = column;
        Forms::FLAT
    }

    /// A loader, made for each run that loads columns. Only called if
    /// `forms` says a column can be lazy.
    fn new_loader(&self, context: &mut Context) -> Result<Self::Loader, Error> {
        let _ = context;
        crate::check::check_failed(line!())
    }

    /// The values of `lazy`, which this wrote, for the rows `selection`
    /// keeps: a flat, constant or dictionary column of as many rows. The rows
    /// it doesn't keep hold any values. Only called if `forms` says a column
    /// can be lazy.
    fn load(
        &self,
        context: &mut Context,
        loader: &mut Self::Loader,
        lazy: &ColumnView,
        selection: &Selection,
    ) -> Result<ColumnView, Error> {
        let _ = (context, loader, lazy, selection);
        crate::check::check_failed(line!())
    }
}

/// The tables a frontend can read, provided by the embedder.
pub trait Catalog {
    /// What's registered as `name`, if anything.
    fn find(&self, name: &str) -> Option<&DynScannable<'_>>;
}

type ScannableOpen =
    unsafe fn(NonNull<()>, &mut Context, &[(u32, Forms)], NonNull<u8>) -> Result<(), Error>;

type ScannableNext =
    unsafe fn(NonNull<()>, &mut Context, NonNull<u8>, ErasedBatch) -> Result<bool, Error>;

type ScannableLoad = unsafe fn(
    NonNull<()>,
    &mut Context,
    NonNull<u8>,
    &ColumnView,
    &Selection,
) -> Result<ColumnView, Error>;

/// A `Scannable` of any type that lives for `'a`, owned in memory from an
/// allocator, and functions that know its type.
pub struct DynScannable<'a> {
    scannable: ErasedBox,
    column_count: unsafe fn(NonNull<()>) -> u32,
    column_name: unsafe fn(NonNull<()>, u32) -> *const str,
    column_type: unsafe fn(NonNull<()>, u32) -> DataType,
    state_layout: Layout,
    open: ScannableOpen,
    drop_state: unsafe fn(NonNull<u8>),
    next: ScannableNext,
    forms: unsafe fn(NonNull<()>, u32) -> Forms,
    loader_layout: Layout,
    new_loader: NewState,
    drop_loader: unsafe fn(NonNull<u8>),
    load: ScannableLoad,
    lifetime: PhantomData<&'a ()>,
}

impl<'a> DynScannable<'a> {
    pub fn new<T: Scannable + 'a>(
        allocator: &dyn Allocator,
        scannable: T,
    ) -> Result<DynScannable<'a>, AllocError> {
        const { assert!(align_of::<T::State<'a>>() <= BUFFER_ALIGNMENT_BYTES) };
        const { assert!(align_of::<T::Loader>() <= BUFFER_ALIGNMENT_BYTES) };
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
            state_layout: Layout::new::<T::State<'a>>(),
            // SAFETY: as above. The state borrows the scan's columns, which
            // outlive it, as the scan's step drops its state first.
            open: |scannable, context, columns, state| unsafe {
                let made = value_of::<T>(scannable).open(context, columns)?;
                write_state(state, made);
                Ok(())
            },
            drop_state: drop_state::<T::State<'a>>,
            // SAFETY: as above.
            // The batch's columns may view the scannable, which lives for
            // `'a`.
            next: |scannable, context, state, batch| unsafe {
                let state = state_of::<T::State<'a>>(state);
                let batch = batch.cast::<RowBatch<'a>>().as_mut();
                value_of::<T>(scannable).next(context, state, batch)
            },
            // SAFETY: as above.
            forms: |scannable, column| unsafe { value_of::<T>(scannable).forms(column) },
            loader_layout: Layout::new::<T::Loader>(),
            // SAFETY: as above.
            new_loader: |scannable, context, loader| unsafe {
                let made = value_of::<T>(scannable).new_loader(context)?;
                write_state(loader, made);
                Ok(())
            },
            drop_loader: drop_state::<T::Loader>,
            // SAFETY: as above.
            load: |scannable, context, loader, lazy, selection| unsafe {
                let loader = state_of::<T::Loader>(loader);
                value_of::<T>(scannable).load(context, loader, lazy, selection)
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

    /// The forms `column` can be written in, besides flat.
    pub fn forms(&self, column: u32) -> Forms {
        // SAFETY: as above.
        unsafe { (self.forms)(self.scannable.as_ptr(), column) }
    }

    /// A source of `columns` of this, in that order, writing each in the
    /// forms of `forms` it says, besides flat.
    pub fn scan(
        &self,
        allocator: &dyn Allocator,
        columns: impl ExactSizeIterator<Item = (u32, Forms)>,
    ) -> Result<DynSource<'_>, AllocError> {
        let columns = SlowVec::fixed_from(allocator, columns)?;
        check!(columns.len() <= BATCH_COLUMNS_MAX as usize);
        let column_count = self.column_count();
        let allowed = |&(c, f): &(u32, Forms)| c < column_count && self.forms(c).contains(f);
        check!(columns.iter().all(allowed));
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

    /// A transform that loads the lazy columns at `positions` in batches,
    /// which this wrote, for the rows each batch keeps.
    pub fn materialize(
        &self,
        allocator: &dyn Allocator,
        positions: SlowVec<u32>,
    ) -> Result<DynTransform<'_>, AllocError> {
        let materialize = Materialize { scannable: NonNull::from(self).cast(), positions };
        let step = Box::new(allocator, materialize)?.erase();
        // SAFETY: the functions take a `Materialize` and the scannable's
        // loader, and the `Materialize` borrows `self` for as long as the
        // transform lives.
        Ok(unsafe {
            DynTransform::from_parts(
                step,
                self.loader_layout,
                materialize_new_state,
                self.drop_loader,
                materialize_process,
            )
        })
    }
}

/// A source's step that reads a `DynScannable`.
struct Scan {
    /// What's read.
    scannable: NonNull<DynScannable<'static>>,
    /// Which of its columns, in order, each with the forms it may be
    /// written in, besides flat.
    columns: SlowVec<(u32, Forms)>,
}

/// A transform that loads lazy columns a `DynScannable` wrote.
struct Materialize {
    scannable: NonNull<DynScannable<'static>>,
    positions: SlowVec<u32>,
}

unsafe fn scan_new_state(
    step: NonNull<()>,
    context: &mut Context,
    state: NonNull<u8>,
) -> Result<(), Error> {
    // SAFETY: `step` is a `Scan`, whose scannable outlives it.
    let scan = unsafe { step.cast::<Scan>().as_ref() };
    // SAFETY: as above.
    let scannable = unsafe { scan.scannable.as_ref() };
    // SAFETY: the function matches the scannable's type.
    unsafe { (scannable.open)(scannable.scannable.as_ptr(), context, &scan.columns, state) }
}

unsafe fn scan_next(
    step: NonNull<()>,
    context: &mut Context,
    state: NonNull<u8>,
    batch: ErasedBatch,
) -> Result<bool, Error> {
    // SAFETY: as in `scan_new_state`.
    let scan = unsafe { step.cast::<Scan>().as_ref() };
    // SAFETY: as in `scan_new_state`.
    let scannable = unsafe { scan.scannable.as_ref() };
    // SAFETY: the function matches the scannable's type, and `state` holds
    // its state.
    unsafe { (scannable.next)(scannable.scannable.as_ptr(), context, state, batch) }
}

unsafe fn materialize_new_state(
    step: NonNull<()>,
    context: &mut Context,
    loader: NonNull<u8>,
) -> Result<(), Error> {
    // SAFETY: `step` is a `Materialize`, whose scannable outlives it.
    let scannable = unsafe { step.cast::<Materialize>().as_ref().scannable.as_ref() };
    // SAFETY: the function matches the scannable's type.
    unsafe { (scannable.new_loader)(scannable.scannable.as_ptr(), context, loader) }
}

unsafe fn materialize_process(
    step: NonNull<()>,
    context: &mut Context,
    loader: NonNull<u8>,
    batch: &mut RowBatch,
) -> Result<(), Error> {
    // SAFETY: as in `materialize_new_state`.
    let materialize = unsafe { step.cast::<Materialize>().as_ref() };
    // SAFETY: as in `materialize_new_state`.
    let scannable = unsafe { materialize.scannable.as_ref() };
    for &position in materialize.positions.iter() {
        let lazy = batch.column(position);
        check!(lazy.is_lazy());
        // SAFETY: the function matches the scannable's type, and `loader`
        // holds its loader.
        let column = unsafe {
            (scannable.load)(scannable.scannable.as_ptr(), context, loader, lazy, batch.selection())
        }?;
        check!(!column.is_lazy() && column.row_count() == batch.row_count());
        batch.set_column(position, column);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec as StdVec;
    use core::cell::RefCell;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::ColumnView;
    use crate::context::Context;
    use crate::pipeline::Pipeline;
    use crate::query_allocators::QueryAllocators;
    use crate::selection::Kept;
    use crate::step::{Step, Transform};

    /// One batch, with column `i` holding `i` and `i + 10`.
    struct Columns;

    impl Scannable for Columns {
        type State<'s>
            = (bool, &'s [(u32, Forms)])
        where
            Self: 's;
        type Loader = ();

        fn column_count(&self) -> u32 {
            3
        }

        fn column_name(&self, column: u32) -> &str {
            ["a", "b", "c"][column as usize]
        }

        fn column_type(&self, _: u32) -> DataType {
            DataType::Int64
        }

        fn open<'s>(
            &'s self,
            _: &mut Context,
            columns: &'s [(u32, Forms)],
        ) -> Result<(bool, &'s [(u32, Forms)]), Error> {
            Ok((false, columns))
        }

        fn next(
            &self,
            _: &mut Context,
            (done, columns): &mut (bool, &[(u32, Forms)]),
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            if *done {
                return Ok(false);
            }
            batch.reset(2);
            for &(column, _) in *columns {
                let mut values = Buffer::allocate(&Heap, 16).unwrap();
                let value = i64::from(column);
                values.as_mut_slice::<i64>().copy_from_slice(&[value, value + 10]);
                assert!(
                    batch
                        .push_column(
                            ColumnView::new(
                                &mut Context::new(&Heap),
                                DataType::Int64,
                                values,
                                None
                            )
                            .unwrap()
                        )
                        .is_ok()
                );
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
        let columns = [(2, Forms::FLAT), (0, Forms::FLAT)].into_iter();
        let pipeline =
            Pipeline::new(scannable.scan(&Heap, columns).unwrap(), SlowVec::new(&Heap, 1).unwrap());
        let query = QueryAllocators::new(&Heap);
        let mut execution = pipeline.start(&query).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch).unwrap());
        let rows: StdVec<&[i64]> = (0..2).map(|i| batch.column(i).int64s()).collect();
        assert_eq!(rows, [[2, 12], [0, 10]]);
        assert!(!execution.next(&mut batch).unwrap());
    }

    /// Two batches of four rows, of one column, `10 * batch + row`, which can
    /// be lazy, with the batch as its handle. Notes each load: its batch
    /// and the rows it keeps.
    struct Counted<'a> {
        loads: &'a RefCell<StdVec<(u8, StdVec<u16>)>>,
    }

    impl Scannable for Counted<'_> {
        type State<'s>
            = (u8, &'s [(u32, Forms)])
        where
            Self: 's;
        type Loader = ();

        fn column_count(&self) -> u32 {
            1
        }

        fn column_name(&self, _: u32) -> &'static str {
            "a"
        }

        fn column_type(&self, _: u32) -> DataType {
            DataType::Int64
        }

        fn open<'s>(
            &'s self,
            _: &mut Context,
            columns: &'s [(u32, Forms)],
        ) -> Result<(u8, &'s [(u32, Forms)]), Error> {
            Ok((0, columns))
        }

        fn next(
            &self,
            context: &mut Context,
            (batches, columns): &mut (u8, &[(u32, Forms)]),
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            if *batches == 2 {
                return Ok(false);
            }
            assert!(columns[0].1.contains(Forms::LAZY));
            batch.reset(4);
            let column = ColumnView::lazy(context, DataType::Int64, &[*batches], 4).unwrap();
            assert!(batch.push_column(column).is_ok());
            *batches += 1;
            Ok(true)
        }

        fn forms(&self, _: u32) -> Forms {
            Forms::LAZY
        }

        fn new_loader(&self, _: &mut Context) -> Result<(), Error> {
            Ok(())
        }

        fn load(
            &self,
            context: &mut Context,
            (): &mut (),
            lazy: &ColumnView,
            selection: &Selection,
        ) -> Result<ColumnView, Error> {
            let (handle, _) = lazy.handle();
            let rows: StdVec<u16> = match selection.kept() {
                Kept::All => (0..4).collect(),
                Kept::None => StdVec::new(),
                Kept::Select(rows) => rows.into(),
            };
            let mut values = context.values_buffer(4 * 8).unwrap();
            for &row in &rows {
                values.as_mut_slice::<i64>()[usize::from(row)] =
                    10 * i64::from(handle[0]) + i64::from(row);
            }
            self.loads.borrow_mut().push((handle[0], rows));
            Ok(ColumnView::new(context, DataType::Int64, values, None)?)
        }
    }

    /// Keeps a batch's odd rows.
    struct KeepOdd;

    impl Transform for KeepOdd {
        type State = ();

        fn new_state(&self, _: &mut Context) -> Result<(), Error> {
            Ok(())
        }

        fn process(&self, _: &mut Context, (): &mut (), batch: &mut RowBatch) -> Result<(), Error> {
            batch.selection_mut().retain(|row| row % 2 == 1);
            Ok(())
        }
    }

    #[test]
    fn writes_lazy_columns_and_loads_them_for_the_rows_kept() {
        let loads = RefCell::new(StdVec::new());
        let scannable = DynScannable::new(&Heap, Counted { loads: &loads }).unwrap();
        let source = scannable.scan(&Heap, [(0, Forms::LAZY)].into_iter()).unwrap();
        let keep = Step::Transform(DynTransform::new(&Heap, KeepOdd).unwrap());
        let positions = SlowVec::fixed_from(&Heap, [0].into_iter()).unwrap();
        let load = Step::Transform(scannable.materialize(&Heap, positions).unwrap());
        let pipeline =
            Pipeline::new(source, SlowVec::fixed_from(&Heap, [keep, load].into_iter()).unwrap());
        let query = QueryAllocators::new(&Heap);
        let mut execution = pipeline.start(&query).unwrap();
        let mut batch = RowBatch::new();
        let mut kept = StdVec::new();
        while execution.next(&mut batch).unwrap() {
            let column = batch.column(0);
            assert!(!column.is_lazy());
            let Kept::Select(rows) = batch.selection().kept() else { panic!() };
            kept.extend(rows.iter().map(|&row| column.int64s()[usize::from(row)]));
        }
        assert_eq!(kept, [1, 3, 11, 13]);
        // Each batch is loaded once, for the rows it keeps.
        assert_eq!(*loads.borrow(), [(0, vec![1, 3]), (1, vec![1, 3])]);
    }
}
