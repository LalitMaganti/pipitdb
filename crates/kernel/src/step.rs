//! The steps of a pipeline: `Source`, `Transform` and `Operator`, and
//! `DynSource`, `DynTransform` and `DynOperator`, the forms a pipeline stores.
//!
//! A step is a plan node: read-only while it runs. What changes lives in its
//! `State`, which each run creates.

use core::alloc::Layout;
use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::buffer::BUFFER_ALIGNMENT_BYTES;
use crate::row_batch::RowBatch;

/// Produces the batches a pipeline runs over.
pub trait Source {
    type State;

    fn new_state(&self) -> Self::State;

    /// Fills `batch`, which is empty when called, or returns false when no
    /// batches are left.
    fn next(&self, batch: &mut RowBatch, state: &mut Self::State) -> bool;
}

/// Changes each batch in place, such as by keeping some of its columns.
pub trait Transform {
    type State;

    fn new_state(&self) -> Self::State;

    fn process(&self, batch: &mut RowBatch, state: &mut Self::State);
}

/// Turns input batches into output batches, for steps that hold rows back or
/// output more than one batch for an input.
pub trait Operator {
    type State;

    fn new_state(&self) -> Self::State;

    /// `output` is empty when called.
    fn execute(&self, input: &RowBatch, output: &mut RowBatch, state: &mut Self::State)
    -> Progress;

    /// Called after the last input, for operators that hold rows back.
    fn finish(&self, output: &mut RowBatch, state: &mut Self::State) -> Progress {
        let _ = (output, state);
        Progress::NeedInput
    }
}

/// What an operator did with its input.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Progress {
    /// The output holds everything left for this input, maybe nothing.
    NeedInput,
    /// The output holds some of it. Call again with the same input.
    MoreOutput,
}

/// A step after a pipeline's source.
pub enum Step<'a> {
    Transform(DynTransform<'a>),
    Operator(DynOperator<'a>),
}

/// A step of any type, borrowed for `'a`: a pointer to it, and functions that
/// know its type. `F` is the function a pipeline calls for each batch.
pub struct Erased<'a, F> {
    step: NonNull<()>,
    pub(crate) state_layout: Layout,
    new_state: unsafe fn(NonNull<()>, NonNull<u8>),
    pub(crate) drop_state: unsafe fn(NonNull<u8>),
    run: F,
    lifetime: PhantomData<&'a ()>,
}

pub type DynSource<'a> = Erased<'a, unsafe fn(NonNull<()>, &mut RowBatch, NonNull<u8>) -> bool>;
pub type DynTransform<'a> = Erased<'a, unsafe fn(NonNull<()>, &mut RowBatch, NonNull<u8>)>;
pub type DynOperator<'a> = Erased<'a, OperatorFunctions>;

pub struct OperatorFunctions {
    execute: unsafe fn(NonNull<()>, &RowBatch, &mut RowBatch, NonNull<u8>) -> Progress,
    finish: unsafe fn(NonNull<()>, &mut RowBatch, NonNull<u8>) -> Progress,
}

impl<'a> DynSource<'a> {
    pub fn new<T: Source>(source: &'a T) -> DynSource<'a> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        Erased {
            step: NonNull::from(source).cast(),
            state_layout: Layout::new::<T::State>(),
            new_state: new_source_state::<T>,
            drop_state: drop_state::<T::State>,
            run: next::<T>,
            lifetime: PhantomData,
        }
    }

    /// # Safety
    ///
    /// `state` must hold a state made by `new_state`.
    pub(crate) unsafe fn next(&self, batch: &mut RowBatch, state: NonNull<u8>) -> bool {
        // SAFETY: `run` matches `step`'s type, and the caller upholds the
        // rest.
        unsafe { (self.run)(self.step, batch, state) }
    }
}

impl<'a> DynTransform<'a> {
    pub fn new<T: Transform>(transform: &'a T) -> DynTransform<'a> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        Erased {
            step: NonNull::from(transform).cast(),
            state_layout: Layout::new::<T::State>(),
            new_state: new_transform_state::<T>,
            drop_state: drop_state::<T::State>,
            run: process::<T>,
            lifetime: PhantomData,
        }
    }

    /// # Safety
    ///
    /// As for `DynSource::next`.
    pub(crate) unsafe fn process(&self, batch: &mut RowBatch, state: NonNull<u8>) {
        // SAFETY: as in `DynSource::next`.
        unsafe { (self.run)(self.step, batch, state) }
    }
}

impl<'a> DynOperator<'a> {
    pub fn new<T: Operator>(operator: &'a T) -> DynOperator<'a> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        Erased {
            step: NonNull::from(operator).cast(),
            state_layout: Layout::new::<T::State>(),
            new_state: new_operator_state::<T>,
            drop_state: drop_state::<T::State>,
            run: OperatorFunctions { execute: execute::<T>, finish: finish::<T> },
            lifetime: PhantomData,
        }
    }

    /// # Safety
    ///
    /// As for `DynSource::next`.
    pub(crate) unsafe fn execute(
        &self,
        input: &RowBatch,
        output: &mut RowBatch,
        state: NonNull<u8>,
    ) -> Progress {
        // SAFETY: as in `DynSource::next`.
        unsafe { (self.run.execute)(self.step, input, output, state) }
    }

    /// # Safety
    ///
    /// As for `DynSource::next`.
    pub(crate) unsafe fn finish(&self, output: &mut RowBatch, state: NonNull<u8>) -> Progress {
        // SAFETY: as in `DynSource::next`.
        unsafe { (self.run.finish)(self.step, output, state) }
    }
}

impl<F> Erased<'_, F> {
    /// # Safety
    ///
    /// `state` must be valid for writes of this step's state.
    pub(crate) unsafe fn new_state(&self, state: NonNull<u8>) {
        // SAFETY: `new_state` matches `step`'s type, and the caller upholds
        // the rest.
        unsafe { (self.new_state)(self.step, state) }
    }
}

// These undo the erasure. Each is only stored next to a pointer to a `T`, and
// only called with a state of `T`'s type.

unsafe fn new_source_state<T: Source>(source: NonNull<()>, state: NonNull<u8>) {
    // SAFETY: see above.
    unsafe { state.cast().write(source.cast::<T>().as_ref().new_state()) }
}

unsafe fn new_transform_state<T: Transform>(transform: NonNull<()>, state: NonNull<u8>) {
    // SAFETY: see above.
    unsafe { state.cast().write(transform.cast::<T>().as_ref().new_state()) }
}

unsafe fn new_operator_state<T: Operator>(operator: NonNull<()>, state: NonNull<u8>) {
    // SAFETY: see above.
    unsafe { state.cast().write(operator.cast::<T>().as_ref().new_state()) }
}

unsafe fn drop_state<S>(state: NonNull<u8>) {
    // SAFETY: see above.
    unsafe { state.cast::<S>().drop_in_place() }
}

unsafe fn next<T: Source>(source: NonNull<()>, batch: &mut RowBatch, state: NonNull<u8>) -> bool {
    // SAFETY: see above.
    unsafe { source.cast::<T>().as_ref().next(batch, state.cast().as_mut()) }
}

unsafe fn process<T: Transform>(transform: NonNull<()>, batch: &mut RowBatch, state: NonNull<u8>) {
    // SAFETY: see above.
    unsafe { transform.cast::<T>().as_ref().process(batch, state.cast().as_mut()) }
}

unsafe fn execute<T: Operator>(
    operator: NonNull<()>,
    input: &RowBatch,
    output: &mut RowBatch,
    state: NonNull<u8>,
) -> Progress {
    // SAFETY: see above.
    unsafe { operator.cast::<T>().as_ref().execute(input, output, state.cast().as_mut()) }
}

unsafe fn finish<T: Operator>(
    operator: NonNull<()>,
    output: &mut RowBatch,
    state: NonNull<u8>,
) -> Progress {
    // SAFETY: see above.
    unsafe { operator.cast::<T>().as_ref().finish(output, state.cast().as_mut()) }
}
