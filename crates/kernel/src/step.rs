//! The steps of a pipeline: `Source`, `Transform` and `Operator`, and
//! `DynSource`, `DynTransform` and `DynOperator`, the forms a pipeline stores.
//!
//! A step is a plan node: read-only while it runs. What changes lives in its
//! `State`, which each run creates, with memory from the run's allocator.

use core::alloc::Layout;
use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator, DynAllocator};
use crate::boxed::{Box, ErasedBox};
use crate::buffer::BUFFER_ALIGNMENT_BYTES;
use crate::erase::{drop_state, state_of, value_of, write_state};
use crate::row_batch::RowBatch;

/// Produces the batches a pipeline runs over.
pub trait Source {
    type State;

    fn new_state(&self, allocator: &DynAllocator) -> Result<Self::State, AllocError>;

    /// Fills `batch`, which is empty when called, or returns false when no
    /// batches are left.
    fn next(&self, batch: &mut RowBatch, state: &mut Self::State) -> bool;
}

/// Changes each batch in place, such as by keeping some of its columns.
pub trait Transform {
    type State;

    fn new_state(&self, allocator: &DynAllocator) -> Result<Self::State, AllocError>;

    fn process(&self, batch: &mut RowBatch, state: &mut Self::State);
}

/// Turns input batches into output batches, for steps that hold rows back or
/// output more than one batch for an input.
pub trait Operator {
    type State;

    fn new_state(&self, allocator: &DynAllocator) -> Result<Self::State, AllocError>;

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

impl Step<'_> {
    pub(crate) fn state_layout(&self) -> Layout {
        match self {
            Step::Transform(transform) => transform.state_layout,
            Step::Operator(operator) => operator.state_layout,
        }
    }

    /// # Safety
    ///
    /// As for `Erased::new_state`.
    pub(crate) unsafe fn new_state(
        &self,
        state: NonNull<u8>,
        allocator: &DynAllocator,
    ) -> Result<(), AllocError> {
        // SAFETY: upheld by the caller.
        unsafe {
            match self {
                Step::Transform(transform) => transform.new_state(state, allocator),
                Step::Operator(operator) => operator.new_state(state, allocator),
            }
        }
    }

    /// # Safety
    ///
    /// `state` must hold this step's state, which is then dropped.
    pub(crate) unsafe fn drop_state(&self, state: NonNull<u8>) {
        let drop_state = match self {
            Step::Transform(transform) => transform.drop_state,
            Step::Operator(operator) => operator.drop_state,
        };
        // SAFETY: upheld by the caller.
        unsafe { drop_state(state) }
    }
}

/// A step of any type that lives for `'a`, owned in memory from an allocator:
/// a pointer to it, and functions that know its type. `F` is the function a
/// pipeline calls for each batch.
pub struct Erased<'a, F> {
    step: ErasedBox,
    pub(crate) state_layout: Layout,
    new_state: NewState,
    pub(crate) drop_state: unsafe fn(NonNull<u8>),
    run: F,
    lifetime: PhantomData<&'a ()>,
}

/// Makes a step's state at the given place, or fails without making it.
pub(crate) type NewState =
    unsafe fn(NonNull<()>, NonNull<u8>, &DynAllocator) -> Result<(), AllocError>;

pub type DynSource<'a> = Erased<'a, unsafe fn(NonNull<()>, &mut RowBatch, NonNull<u8>) -> bool>;
pub type DynTransform<'a> = Erased<'a, unsafe fn(NonNull<()>, &mut RowBatch, NonNull<u8>)>;
pub type DynOperator<'a> = Erased<'a, OperatorFunctions>;

pub struct OperatorFunctions {
    execute: unsafe fn(NonNull<()>, &RowBatch, &mut RowBatch, NonNull<u8>) -> Progress,
    finish: unsafe fn(NonNull<()>, &mut RowBatch, NonNull<u8>) -> Progress,
}

impl<'a> DynSource<'a> {
    /// A source from parts that already agree on its step's and state's types.
    ///
    /// # Safety
    ///
    /// The functions must take `step` and a state of `state_layout`, and the
    /// step must live for `'a`.
    pub(crate) unsafe fn from_parts(
        step: ErasedBox,
        state_layout: Layout,
        new_state: NewState,
        drop_state: unsafe fn(NonNull<u8>),
        next: unsafe fn(NonNull<()>, &mut RowBatch, NonNull<u8>) -> bool,
    ) -> DynSource<'a> {
        Erased { step, state_layout, new_state, drop_state, run: next, lifetime: PhantomData }
    }

    pub fn new<A: Allocator + Clone + 'static, T: Source + 'a>(
        allocator: A,
        source: T,
    ) -> Result<DynSource<'a>, AllocError> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        Ok(Erased {
            step: Box::new(allocator, source)?.erase(),
            state_layout: Layout::new::<T::State>(),
            // SAFETY: only called with this source and its state, as is each below.
            new_state: |source, state, allocator| unsafe {
                let made = value_of::<T>(source).new_state(allocator)?;
                write_state(state, made);
                Ok(())
            },
            drop_state: drop_state::<T::State>,
            // SAFETY: as above.
            run: |source, batch, state| unsafe {
                value_of::<T>(source).next(batch, state_of::<T::State>(state))
            },
            lifetime: PhantomData,
        })
    }

    /// # Safety
    ///
    /// `state` must hold a state made by `new_state`.
    pub(crate) unsafe fn next(&self, batch: &mut RowBatch, state: NonNull<u8>) -> bool {
        // SAFETY: `run` matches `step`'s type, and the caller upholds the
        // rest.
        unsafe { (self.run)(self.step.as_ptr(), batch, state) }
    }
}

impl<'a> DynTransform<'a> {
    pub fn new<A: Allocator + Clone + 'static, T: Transform + 'a>(
        allocator: A,
        transform: T,
    ) -> Result<DynTransform<'a>, AllocError> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        Ok(Erased {
            step: Box::new(allocator, transform)?.erase(),
            state_layout: Layout::new::<T::State>(),
            // SAFETY: only called with this transform and its state, as is each below.
            new_state: |transform, state, allocator| unsafe {
                let made = value_of::<T>(transform).new_state(allocator)?;
                write_state(state, made);
                Ok(())
            },
            drop_state: drop_state::<T::State>,
            // SAFETY: as above.
            run: |transform, batch, state| unsafe {
                value_of::<T>(transform).process(batch, state_of::<T::State>(state));
            },
            lifetime: PhantomData,
        })
    }

    /// # Safety
    ///
    /// As for `DynSource::next`.
    pub(crate) unsafe fn process(&self, batch: &mut RowBatch, state: NonNull<u8>) {
        // SAFETY: as in `DynSource::next`.
        unsafe { (self.run)(self.step.as_ptr(), batch, state) }
    }
}

impl<'a> DynOperator<'a> {
    pub fn new<A: Allocator + Clone + 'static, T: Operator + 'a>(
        allocator: A,
        operator: T,
    ) -> Result<DynOperator<'a>, AllocError> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        Ok(Erased {
            step: Box::new(allocator, operator)?.erase(),
            state_layout: Layout::new::<T::State>(),
            // SAFETY: only called with this operator and its state, as is each below.
            new_state: |operator, state, allocator| unsafe {
                let made = value_of::<T>(operator).new_state(allocator)?;
                write_state(state, made);
                Ok(())
            },
            drop_state: drop_state::<T::State>,
            run: OperatorFunctions {
                // SAFETY: as above.
                execute: |operator, input, output, state| unsafe {
                    value_of::<T>(operator).execute(input, output, state_of::<T::State>(state))
                },
                // SAFETY: as above.
                finish: |operator, output, state| unsafe {
                    value_of::<T>(operator).finish(output, state_of::<T::State>(state))
                },
            },
            lifetime: PhantomData,
        })
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
        unsafe { (self.run.execute)(self.step.as_ptr(), input, output, state) }
    }

    /// # Safety
    ///
    /// As for `DynSource::next`.
    pub(crate) unsafe fn finish(&self, output: &mut RowBatch, state: NonNull<u8>) -> Progress {
        // SAFETY: as in `DynSource::next`.
        unsafe { (self.run.finish)(self.step.as_ptr(), output, state) }
    }
}

impl<F> Erased<'_, F> {
    /// Makes this step's state at `state`, unless it fails.
    ///
    /// # Safety
    ///
    /// `state` must be valid for writes of this step's state.
    pub(crate) unsafe fn new_state(
        &self,
        state: NonNull<u8>,
        allocator: &DynAllocator,
    ) -> Result<(), AllocError> {
        // SAFETY: `new_state` matches `step`'s type, and the caller upholds
        // the rest.
        unsafe { (self.new_state)(self.step.as_ptr(), state, allocator) }
    }
}
