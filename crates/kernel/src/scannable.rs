//! `Scannable`: data a pipeline can read, such as a table, and `DynScannable`,
//! the form a `Catalog` hands out.
//!
//! A scannable can write some columns lazily, with a handle saying where
//! their values are, and load them later, for the rows still kept, when
//! something reads them. The optimizer says which, and where they're loaded.

#![allow(unsafe_code)]

use core::alloc::Layout;
use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::allocator::AllocError;
use crate::buffer::BUFFER_ALIGNMENT_BYTES;
use crate::column::{Bounds, ColumnView, DataType, Forms};
use crate::condition::{CONDITIONS_MAX, Condition, Cutoff};
use crate::containers::{Box, ErasedBox, SmallVec};
use crate::context::Context;
use crate::env::Env;
use crate::erase::{drop_state, state_of, value_of, write_state};
use crate::error::Error;
use crate::row_batch::{BATCH_COLUMNS_MAX, RowBatch};
use crate::selection::Selection;
use crate::step::{DynSource, DynTransform, Name, SourceNext, StepFunctions, TransformProcess};

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

    /// Whether `column` can hold a null. A scannable that can't tell says it
    /// can.
    fn column_nullable(&self, _column: u32) -> bool {
        true
    }

    /// What kind of key `column` is known to be, if any. By default, none.
    fn column_key(&self, _column: u32) -> Key {
        Key::None
    }

    /// A read of `columns`, in that order, from the first row, each with
    /// the forms it may be written in, some of those `forms` says it can
    /// write it in. What depends only on which columns are read
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

    /// A source of `columns`, read as `open` would, that may leave out rows
    /// failing `conditions`, which the plan pushed down from filters, or
    /// sorting after the `cutoff` a `TopK` above publishes: the filters and
    /// the `TopK` still test every row. It's of whatever type suits, such as
    /// one that skips what statistics rule out, or `None` to read with
    /// `open`. Its lazy columns are loaded by this, as `open`'s are. By
    /// default, `None`.
    fn scan_within<'s>(
        &'s self,
        env: &Env,
        columns: &[(u32, Forms)],
        conditions: &[Condition],
        cutoff: Option<Cutoff>,
    ) -> Result<Option<DynSource<'s>>, AllocError> {
        let _ = (env, columns, conditions, cutoff);
        Ok(None)
    }

    /// How many rows it has, and with `column`, the least and greatest of
    /// that column's values, if they're known exactly without reading them,
    /// as from a file's statistics. By default, `None`.
    fn summary(&self, column: Option<u32>) -> Option<Summary> {
        let _ = column;
        None
    }

    /// The forms `column` can be written in. By default, flat only.
    fn forms(&self, column: u32) -> Forms {
        let _ = column;
        Forms::FLAT
    }

    /// The orders its rows are known to be in, as a scan of it reads them:
    /// each the columns they're sorted by, in order, each with whether it
    /// descends. By default, none.
    fn sorted_by(&self) -> &[&[(u32, bool)]] {
        &[]
    }

    /// A loader, made for each run that loads columns. Only called if
    /// `forms` says a column can be lazy.
    fn new_loader(&self, context: &mut Context) -> Result<Self::Loader, Error> {
        let _ = context;
        failed!()
    }

    /// The values of `lazy`, which this wrote, for the rows `selection`
    /// keeps: a column of as many rows, in one of `forms`. The rows
    /// it doesn't keep hold any values. Only called if `forms` says a column
    /// can be lazy.
    fn load(
        &self,
        context: &mut Context,
        loader: &mut Self::Loader,
        lazy: &ColumnView,
        selection: &Selection,
        forms: Forms,
    ) -> Result<ColumnView, Error> {
        let _ = (context, loader, lazy, selection, forms);
        failed!()
    }
}

/// What's known of a scannable's rows without reading them: how many there
/// are, the least and greatest of a column's values, `None` if it has none
/// that aren't null, and whether its runs of rows, such as row groups,
/// follow on in it: each run's values are above every value before it, and
/// the bounds of any two batches of a run overlap.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Summary {
    pub rows: i64,
    pub bounds: Option<Bounds>,
    pub follows_on: bool,
}

/// What kind of key a column is known to be, from least to most known, as
/// Perfetto's dataframe columns record. Each kind implies the ones before
/// it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Key {
    /// Not known to be a key.
    None,
    /// No two rows have the same value: Perfetto's `kNoDuplicates`.
    Unique,
    /// Each row's value is one more than the previous row's, as a table's
    /// id is: Perfetto's `Id` storage. Finding a value's row is then simple
    /// arithmetic.
    Id,
}

/// The tables a frontend can read, provided by the embedder.
pub trait Catalog {
    /// What's registered as `name`, if anything.
    fn find(&self, name: &str) -> Option<&DynScannable<'_>>;
}

type ScannableScanWithin<'a> = unsafe fn(
    NonNull<()>,
    &Env,
    &[(u32, Forms)],
    &[Condition],
    Option<Cutoff>,
) -> Result<Option<DynSource<'a>>, AllocError>;

/// The orders a scannable's rows are in, lent for as long as it lives.
type ScannableSortedBy = unsafe fn(NonNull<()>) -> *const [&'static [(u32, bool)]];

/// A `Scannable` of any type that lives for `'a`, owned in memory from an
/// `Env`, and its type's functions.
pub struct DynScannable<'a> {
    scannable: ErasedBox,
    functions: &'a ScannableFunctions<'a>,
}

/// A scannable type's functions, and those of the steps that scan it and
/// load its lazy columns: one table for each type.
struct ScannableFunctions<'a> {
    column_count: unsafe fn(NonNull<()>) -> u32,
    column_name: unsafe fn(NonNull<()>, u32) -> *const str,
    column_type: unsafe fn(NonNull<()>, u32) -> DataType,
    column_nullable: unsafe fn(NonNull<()>, u32) -> bool,
    column_key: unsafe fn(NonNull<()>, u32) -> Key,
    scan_within: ScannableScanWithin<'a>,
    forms: unsafe fn(NonNull<()>, u32) -> Forms,
    summary: unsafe fn(NonNull<()>, Option<u32>) -> Option<Summary>,
    sorted_by: ScannableSortedBy,
    /// A `Scan`'s, whose state is the scannable's.
    scan: StepFunctions<SourceNext>,
    /// A `Materialize`'s, whose state is the scannable's loader.
    materialize: StepFunctions<TransformProcess>,
}

/// Holds `T`'s table.
struct Functions<'a, T>(PhantomData<(&'a (), T)>);

impl<'a, T: Scannable + 'a> Functions<'a, T> {
    const TABLE: ScannableFunctions<'a> = ScannableFunctions {
        // SAFETY: only called with this scannable, as is each below.
        column_count: |scannable| unsafe { value_of::<T>(scannable).column_count() },
        // SAFETY: as above.
        column_name: |scannable, column| unsafe { value_of::<T>(scannable).column_name(column) },
        // SAFETY: as above.
        column_type: |scannable, column| unsafe { value_of::<T>(scannable).column_type(column) },
        // SAFETY: as above.
        column_nullable: |scannable, column| unsafe {
            value_of::<T>(scannable).column_nullable(column)
        },
        // SAFETY: as above.
        column_key: |scannable, column| unsafe { value_of::<T>(scannable).column_key(column) },
        // SAFETY: as above. The source borrows the scannable, for as
        // long as `scan` says.
        scan_within: |scannable, env, columns, conditions, cutoff| unsafe {
            value_of::<T>(scannable).scan_within(env, columns, conditions, cutoff)
        },
        // SAFETY: as above.
        forms: |scannable, column| unsafe { value_of::<T>(scannable).forms(column) },
        // SAFETY: as above.
        summary: |scannable, column| unsafe { value_of::<T>(scannable).summary(column) },
        sorted_by: |scannable| {
            // SAFETY: as above. The orders live as long as the scannable,
            // which is all `DynScannable::sorted_by` lends them for.
            let orders: &[&[(u32, bool)]] = unsafe { value_of::<T>(scannable).sorted_by() };
            #[expect(clippy::unnecessary_cast, reason = "it erases the orders' lifetime")]
            let orders = core::ptr::from_ref(orders) as *const [&'static [(u32, bool)]];
            orders
        },
        scan: StepFunctions {
            name: Name::new("Scan"),
            state_layout: Layout::new::<T::State<'a>>(),
            // SAFETY: only called with a `Scan` of this scannable and its
            // state, as is each below. The state borrows the scan's
            // columns, which outlive it, as the scan's step drops its state
            // first.
            new_state: |scan, context, state| unsafe {
                let scan = value_of::<Scan>(scan);
                let made = value_of::<T>(scan.scannable).open(context, &scan.columns)?;
                write_state(state, made);
                Ok(())
            },
            drop_state: drop_state::<T::State<'a>>,
            // SAFETY: as above. The batch's columns may view the scannable,
            // which lives for `'a`.
            run: |scan, context, state, batch| unsafe {
                let scannable = value_of::<T>(value_of::<Scan>(scan).scannable);
                let state = state_of::<T::State<'a>>(state);
                scannable.next(context, state, batch.cast::<RowBatch<'a>>().as_mut())
            },
        },
        materialize: StepFunctions {
            name: Name::new("Materialize"),
            state_layout: Layout::new::<T::Loader>(),
            // SAFETY: only called with a `Materialize` of this scannable and
            // its loader, as is each below.
            new_state: |materialize, context, loader| unsafe {
                let scannable = value_of::<T>(value_of::<Materialize>(materialize).scannable);
                write_state(loader, scannable.new_loader(context)?);
                Ok(())
            },
            drop_state: drop_state::<T::Loader>,
            // SAFETY: as above.
            run: |materialize, context, loader, batch| unsafe {
                let materialize = value_of::<Materialize>(materialize);
                let scannable = value_of::<T>(materialize.scannable);
                let loader = state_of::<T::Loader>(loader);
                materialize.process(context, batch, &mut |context, lazy, selection, forms| {
                    scannable.load(context, loader, lazy, selection, forms)
                })
            },
        },
    };
}

impl<'a> DynScannable<'a> {
    pub fn new<T: Scannable + 'a>(env: &Env, scannable: T) -> Result<DynScannable<'a>, AllocError> {
        const { assert!(align_of::<T::State<'a>>() <= BUFFER_ALIGNMENT_BYTES) };
        const { assert!(align_of::<T::Loader>() <= BUFFER_ALIGNMENT_BYTES) };
        let scannable = Box::new(env, scannable)?.erase();
        Ok(DynScannable { scannable, functions: &Functions::<'a, T>::TABLE })
    }

    pub fn column_count(&self) -> u32 {
        // SAFETY: the function matches the scannable's type.
        unsafe { (self.functions.column_count)(self.scannable.as_ptr()) }
    }

    pub fn column_name(&self, column: u32) -> &str {
        // SAFETY: as above, and the name lives as long as the scannable.
        unsafe { &*(self.functions.column_name)(self.scannable.as_ptr(), column) }
    }

    pub fn column_type(&self, column: u32) -> DataType {
        // SAFETY: as above.
        unsafe { (self.functions.column_type)(self.scannable.as_ptr(), column) }
    }

    pub fn column_nullable(&self, column: u32) -> bool {
        // SAFETY: called with its own scannable.
        unsafe { (self.functions.column_nullable)(self.scannable.as_ptr(), column) }
    }

    pub fn column_key(&self, column: u32) -> Key {
        // SAFETY: called with its own scannable.
        unsafe { (self.functions.column_key)(self.scannable.as_ptr(), column) }
    }

    /// The forms `column` can be written in.
    pub fn summary(&self, column: Option<u32>) -> Option<Summary> {
        // SAFETY: as above.
        unsafe { (self.functions.summary)(self.scannable.as_ptr(), column) }
    }

    pub fn forms(&self, column: u32) -> Forms {
        // SAFETY: as above.
        unsafe { (self.functions.forms)(self.scannable.as_ptr(), column) }
    }

    /// The orders its rows are known to be in.
    pub fn sorted_by(&self) -> &[&[(u32, bool)]] {
        // SAFETY: as above, and the keys live as long as the scannable.
        unsafe { &*(self.functions.sorted_by)(self.scannable.as_ptr()) }
    }

    /// A source of `columns` of this, in that order, each written in one of
    /// the forms with it, maybe skipping rows failing `conditions`, on those
    /// columns, or sorting after `cutoff`'s.
    pub fn scan(
        &self,
        env: &Env,
        columns: impl ExactSizeIterator<Item = (u32, Forms)>,
        conditions: &[Condition],
        cutoff: Option<Cutoff>,
    ) -> Result<DynSource<'_>, AllocError> {
        let columns = SmallVec::fixed_from(env, columns)?;
        check!(columns.len() <= BATCH_COLUMNS_MAX as usize);
        let column_count = self.column_count();
        // Each set holds flat, which every scannable can write, so checked
        // here, once, not for every column of every batch.
        let allowed = |&(c, f): &(u32, Forms)| {
            c < column_count && f.contains(Forms::FLAT) && self.forms(c).contains(f)
        };
        check!(columns.iter().all(allowed));
        check!(conditions.len() <= CONDITIONS_MAX);
        let read = |condition: &Condition| columns.iter().any(|&(c, _)| c == condition.column());
        check!(conditions.iter().all(read));
        check!(cutoff.is_none_or(|cutoff| columns.iter().any(|&(c, _)| c == cutoff.column)));
        if !conditions.is_empty() || cutoff.is_some() {
            // SAFETY: the function matches the scannable's type, and the
            // source borrows `self`.
            let within = unsafe {
                let scannable = self.scannable.as_ptr();
                (self.functions.scan_within)(scannable, env, &columns, conditions, cutoff)?
            };
            if let Some(source) = within {
                return Ok(source);
            }
        }
        let scan = Scan { scannable: self.scannable.as_ptr(), columns };
        let step = Box::new(env, scan)?.erase();
        // SAFETY: the functions take a `Scan` of this scannable, which the
        // `Scan` borrows for as long as the source lives.
        let source = unsafe { DynSource::from_parts(step, &self.functions.scan) };
        Ok(source)
    }

    /// A transform that loads the lazy columns at `positions` in batches,
    /// which this wrote, for the rows each batch keeps, each in one of its
    /// `forms`.
    pub fn materialize(
        &self,
        env: &Env,
        positions: SmallVec<u32>,
        forms: SmallVec<Forms>,
    ) -> Result<DynTransform<'_>, AllocError> {
        check!(forms.len() == positions.len() && forms.iter().all(|f| f.contains(Forms::FLAT)));
        let materialize = Materialize { scannable: self.scannable.as_ptr(), positions, forms };
        let step = Box::new(env, materialize)?.erase();
        let functions = &self.functions.materialize;
        // SAFETY: the functions take a `Materialize` of this scannable, which
        // the `Materialize` borrows for as long as the transform lives.
        Ok(unsafe { DynTransform::from_parts(step, functions) })
    }
}

/// A source's step that reads a `DynScannable`.
struct Scan {
    /// The `DynScannable`'s value, which is read.
    scannable: NonNull<()>,
    /// Which of its columns, in order, each with the forms it may be
    /// written in.
    columns: SmallVec<(u32, Forms)>,
}

/// A transform that loads lazy columns a `DynScannable` wrote.
struct Materialize {
    /// The `DynScannable`'s value, which wrote the lazy columns.
    scannable: NonNull<()>,
    /// Where they are in batches.
    positions: SmallVec<u32>,
    /// The forms each may be loaded in.
    forms: SmallVec<Forms>,
}

/// Loads a lazy column: the scannable's `load`, with its loader.
type Load<'l> =
    dyn FnMut(&mut Context, &ColumnView, &Selection, Forms) -> Result<ColumnView, Error> + 'l;

impl Materialize {
    /// Loads `batch`'s lazy columns at `positions` with `load`.
    fn process(
        &self,
        context: &mut Context,
        batch: &mut RowBatch,
        load: &mut Load<'_>,
    ) -> Result<(), Error> {
        for (&position, &forms) in self.positions.iter().zip(self.forms.iter()) {
            let lazy = batch.column(position);
            // A scannable may read a column it could have left lazy.
            if !lazy.is_lazy() {
                continue;
            }
            let column = load(context, lazy, batch.selection(), forms)?;
            check!(!column.is_lazy() && column.row_count() == batch.row_count());
            batch.set_column(position, column);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec as StdVec;
    use core::cell::RefCell;

    use super::*;

    use crate::buffer::Buffer;
    use crate::column::ColumnView;
    use crate::context::Context;
    use crate::env::Env;
    use crate::pipeline::Pipeline;
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
            let env = Env::current();
            if *done {
                return Ok(false);
            }
            batch.reset(2);
            for &(column, _) in *columns {
                let mut values = Buffer::allocate(env, 16).unwrap();
                let value = i64::from(column);
                values.as_mut_slice::<i64>().copy_from_slice(&[value, value + 10]);
                assert!(
                    batch
                        .push_column(
                            ColumnView::new(
                                &mut Context::new(env, None),
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
        let env = Env::current();
        let scannable = DynScannable::new(env, Columns).unwrap();
        assert_eq!(scannable.column_count(), 3);
        assert_eq!(scannable.column_name(1), "b");
        assert!(scannable.column_type(2) == DataType::Int64);
    }

    #[test]
    fn scans_chosen_columns_through_a_pipeline() {
        let env = Env::current();
        let scannable = DynScannable::new(env, Columns).unwrap();
        let columns = [(2, Forms::FLAT), (0, Forms::FLAT)].into_iter();
        let pipeline = Pipeline::new(
            scannable.scan(env, columns, &[], None).unwrap(),
            SmallVec::new(env, 1).unwrap(),
        );
        let query = Env::current();
        let mut execution = pipeline.start(query, None).unwrap();
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
            Forms::FLAT | Forms::LAZY
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
            _: Forms,
        ) -> Result<ColumnView, Error> {
            let (handle, _) = lazy.handle();
            let rows: StdVec<u16> = match selection.kept() {
                Kept::Prefix => (0..4).collect(),
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
        const NAME: &'static str = "KeepOdd";
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
        let env = Env::current();
        let loads = RefCell::new(StdVec::new());
        let scannable = DynScannable::new(env, Counted { loads: &loads }).unwrap();
        let source =
            scannable.scan(env, [(0, Forms::FLAT | Forms::LAZY)].into_iter(), &[], None).unwrap();
        let keep = Step::Transform(DynTransform::new(env, KeepOdd).unwrap());
        let positions = SmallVec::fixed_from(env, [0].into_iter()).unwrap();
        let flat = SmallVec::fixed_from(env, [Forms::FLAT].into_iter()).unwrap();
        let load = Step::Transform(scannable.materialize(env, positions, flat).unwrap());
        let pipeline =
            Pipeline::new(source, SmallVec::fixed_from(env, [keep, load].into_iter()).unwrap());
        let query = Env::current();
        let mut execution = pipeline.start(query, None).unwrap();
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
