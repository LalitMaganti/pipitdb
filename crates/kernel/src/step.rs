//! The steps of a pipeline: `Source`, `Transform` and `Operator`, and
//! `DynSource`, `DynTransform` and `DynOperator`, the forms a pipeline stores.
//!
//! A step is a plan node: read-only while it runs. What changes lives in its
//! `State`, which each run creates. What a run's steps share is its
//! `Context`, passed to each.
//!
//! A step fails, with an `Error`, if it can't allocate what it needs or
//! read its data, which ends the run. Steps never drop rows instead.

use core::alloc::Layout;
use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::boxed::{Box, ErasedBox};
use crate::buffer::BUFFER_ALIGNMENT_BYTES;
use crate::context::Context;
use crate::erase::{drop_state, state_of, value_of, write_state};
use crate::error::Error;
use crate::row_batch::RowBatch;

/// Produces the batches a pipeline runs over.
pub trait Source {
    type State;

    fn new_state(&self, context: &mut Context) -> Result<Self::State, Error>;

    /// Resets `batch`, which holds the last batch, and fills it, or returns
    /// false when no batches are left. Its columns may view what the source
    /// holds.
    fn next<'s>(
        &'s self,
        context: &mut Context,
        state: &mut Self::State,
        batch: &mut RowBatch<'s>,
    ) -> Result<bool, Error>;
}

/// Changes each batch in place, such as by keeping some of its columns.
pub trait Transform {
    type State;

    fn new_state(&self, context: &mut Context) -> Result<Self::State, Error>;

    fn process(
        &self,
        context: &mut Context,
        state: &mut Self::State,
        batch: &mut RowBatch,
    ) -> Result<(), Error>;
}

/// Turns input batches into output batches, for steps that hold rows back or
/// output more than one batch for an input.
pub trait Operator {
    type State;

    fn new_state(&self, context: &mut Context) -> Result<Self::State, Error>;

    /// `output` is empty when called.
    fn execute(
        &self,
        context: &mut Context,
        state: &mut Self::State,
        input: &RowBatch,
        output: &mut RowBatch,
    ) -> Result<Progress, Error>;

    /// Called after the last input, to output the rows held back, if any.
    /// Required, so an operator that holds rows can't drop them by default.
    fn finish(
        &self,
        context: &mut Context,
        state: &mut Self::State,
        output: &mut RowBatch,
    ) -> Result<Progress, Error>;
}

/// Holds back every row of its input, then outputs rows made from them, such
/// as a sort or an aggregation: an operator, which consumes each input, then
/// finalizes and serves once the input ends.
pub trait Breaker {
    type State;

    fn new_state(&self, context: &mut Context) -> Result<Self::State, Error>;

    fn consume(
        &self,
        context: &mut Context,
        state: &mut Self::State,
        input: &RowBatch,
    ) -> Result<(), Error>;

    /// Called once, after every input is consumed.
    fn finalize(&self, context: &mut Context, state: &mut Self::State) -> Result<(), Error>;

    /// Fills `output`, which is empty when called, with the next rows,
    /// returning whether any are left after them.
    fn serve(
        &self,
        context: &mut Context,
        state: &mut Self::State,
        output: &mut RowBatch,
    ) -> Result<bool, Error>;
}

impl<B: Breaker> Operator for B {
    /// The breaker's state, and whether it's finalized.
    type State = (B::State, bool);

    fn new_state(&self, context: &mut Context) -> Result<Self::State, Error> {
        Ok((Breaker::new_state(self, context)?, false))
    }

    fn execute(
        &self,
        context: &mut Context,
        (state, _): &mut Self::State,
        input: &RowBatch,
        _: &mut RowBatch,
    ) -> Result<Progress, Error> {
        self.consume(context, state, input)?;
        Ok(Progress::NeedInput)
    }

    fn finish(
        &self,
        context: &mut Context,
        (state, finalized): &mut Self::State,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        if !*finalized {
            self.finalize(context, state)?;
            *finalized = true;
        }
        let more = self.serve(context, state, output)?;
        Ok(if more { Progress::MoreOutput } else { Progress::NeedInput })
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
        context: &mut Context,
        state: NonNull<u8>,
    ) -> Result<(), Error> {
        // SAFETY: upheld by the caller.
        unsafe {
            match self {
                Step::Transform(transform) => transform.new_state(context, state),
                Step::Operator(operator) => operator.new_state(context, state),
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
    #[cfg(test)]
    pub(crate) name: &'static str,
}

/// Makes a step's state at the given place, or fails without making it.
pub(crate) type NewState = unsafe fn(NonNull<()>, &mut Context, NonNull<u8>) -> Result<(), Error>;

pub type DynSource<'a> = Erased<'a, SourceNext>;
pub type DynTransform<'a> = Erased<'a, TransformProcess>;
pub type DynOperator<'a> = Erased<'a, OperatorFunctions>;

pub(crate) type TransformProcess =
    unsafe fn(NonNull<()>, &mut Context, NonNull<u8>, &mut RowBatch) -> Result<(), Error>;

/// A source's batch: erased, as its columns may view what the source holds,
/// for as long as the source lives.
pub(crate) type ErasedBatch = NonNull<RowBatch<'static>>;

// TODO: `Result<bool, Error>` and `Result<Progress, Error>` take two bytes,
// where `Result<bool, AllocError>` took one, which costs the pipeline's loop
// a few instructions a batch. Converting to one-byte enums such as
// `enum Next { Batch, End, Failed(Error) }` here, at the function-pointer
// boundary, would win them back while steps keep `?`.
pub(crate) type SourceNext =
    unsafe fn(NonNull<()>, &mut Context, NonNull<u8>, ErasedBatch) -> Result<bool, Error>;

type Executed = Result<Progress, Error>;

pub struct OperatorFunctions {
    execute:
        unsafe fn(NonNull<()>, &mut Context, NonNull<u8>, &RowBatch, &mut RowBatch) -> Executed,
    finish: unsafe fn(NonNull<()>, &mut Context, NonNull<u8>, &mut RowBatch) -> Executed,
}

impl<'a> DynSource<'a> {
    /// A source from parts that already agree on its step's and state's types,
    /// named `name` for tests to print plans.
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
        next: SourceNext,
        name: &'static str,
    ) -> DynSource<'a> {
        #[cfg(not(test))]
        let _ = name;
        Erased {
            step,
            state_layout,
            new_state,
            drop_state,
            run: next,
            lifetime: PhantomData,
            #[cfg(test)]
            name,
        }
    }

    pub fn new<T: Source + 'a>(
        allocator: &dyn Allocator,
        source: T,
    ) -> Result<DynSource<'a>, AllocError> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        Ok(Erased {
            step: Box::new(allocator, source)?.erase(),
            state_layout: Layout::new::<T::State>(),
            // SAFETY: only called with this source and its state, as is each below.
            new_state: |source, context, state| unsafe {
                let made = value_of::<T>(source).new_state(context)?;
                write_state(state, made);
                Ok(())
            },
            drop_state: drop_state::<T::State>,
            // SAFETY: as above.
            // The batch's columns may view the source, which lives for `'a`.
            run: |source, context, state, batch| unsafe {
                let batch = batch.cast::<RowBatch<'a>>().as_mut();
                value_of::<T>(source).next(context, state_of::<T::State>(state), batch)
            },
            lifetime: PhantomData,
            #[cfg(test)]
            name: type_name::<T>(),
        })
    }

    /// # Safety
    ///
    /// `state` must hold a state made by `new_state`.
    pub(crate) unsafe fn next(
        &self,
        context: &mut Context,
        state: NonNull<u8>,
        batch: &mut RowBatch<'a>,
    ) -> Result<bool, Error> {
        // SAFETY: `run` matches `step`'s type, and the caller upholds the
        // rest.
        unsafe { (self.run)(self.step.as_ptr(), context, state, NonNull::from(batch).cast()) }
    }
}

impl<'a> DynTransform<'a> {
    /// A transform from parts that already agree on its step's and state's
    /// types, named `name` for tests to print plans.
    ///
    /// # Safety
    ///
    /// As for `DynSource::from_parts`.
    pub(crate) unsafe fn from_parts(
        step: ErasedBox,
        state_layout: Layout,
        new_state: NewState,
        drop_state: unsafe fn(NonNull<u8>),
        process: TransformProcess,
        name: &'static str,
    ) -> DynTransform<'a> {
        #[cfg(not(test))]
        let _ = name;
        Erased {
            step,
            state_layout,
            new_state,
            drop_state,
            run: process,
            lifetime: PhantomData,
            #[cfg(test)]
            name,
        }
    }

    pub fn new<T: Transform + 'a>(
        allocator: &dyn Allocator,
        transform: T,
    ) -> Result<DynTransform<'a>, AllocError> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        Ok(Erased {
            step: Box::new(allocator, transform)?.erase(),
            state_layout: Layout::new::<T::State>(),
            // SAFETY: only called with this transform and its state, as is each below.
            new_state: |transform, context, state| unsafe {
                let made = value_of::<T>(transform).new_state(context)?;
                write_state(state, made);
                Ok(())
            },
            drop_state: drop_state::<T::State>,
            // SAFETY: as above.
            run: |transform, context, state, batch| unsafe {
                value_of::<T>(transform).process(context, state_of::<T::State>(state), batch)
            },
            lifetime: PhantomData,
            #[cfg(test)]
            name: type_name::<T>(),
        })
    }

    /// # Safety
    ///
    /// As for `DynSource::next`.
    pub(crate) unsafe fn process(
        &self,
        context: &mut Context,
        state: NonNull<u8>,
        batch: &mut RowBatch,
    ) -> Result<(), Error> {
        // SAFETY: as in `DynSource::next`.
        unsafe { (self.run)(self.step.as_ptr(), context, state, batch) }
    }
}

impl<'a> DynOperator<'a> {
    pub fn new<T: Operator + 'a>(
        allocator: &dyn Allocator,
        operator: T,
    ) -> Result<DynOperator<'a>, AllocError> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        Ok(Erased {
            step: Box::new(allocator, operator)?.erase(),
            state_layout: Layout::new::<T::State>(),
            // SAFETY: only called with this operator and its state, as is each below.
            new_state: |operator, context, state| unsafe {
                let made = value_of::<T>(operator).new_state(context)?;
                write_state(state, made);
                Ok(())
            },
            drop_state: drop_state::<T::State>,
            run: OperatorFunctions {
                // SAFETY: as above.
                execute: |operator, context, state, input, output| unsafe {
                    let state = state_of::<T::State>(state);
                    value_of::<T>(operator).execute(context, state, input, output)
                },
                // SAFETY: as above.
                finish: |operator, context, state, output| unsafe {
                    value_of::<T>(operator).finish(context, state_of::<T::State>(state), output)
                },
            },
            lifetime: PhantomData,
            #[cfg(test)]
            name: type_name::<T>(),
        })
    }

    /// # Safety
    ///
    /// As for `DynSource::next`.
    pub(crate) unsafe fn execute(
        &self,
        context: &mut Context,
        state: NonNull<u8>,
        input: &RowBatch,
        output: &mut RowBatch,
    ) -> Executed {
        // SAFETY: as in `DynSource::next`.
        unsafe { (self.run.execute)(self.step.as_ptr(), context, state, input, output) }
    }

    /// # Safety
    ///
    /// As for `DynSource::next`.
    pub(crate) unsafe fn finish(
        &self,
        context: &mut Context,
        state: NonNull<u8>,
        output: &mut RowBatch,
    ) -> Executed {
        // SAFETY: as in `DynSource::next`.
        unsafe { (self.run.finish)(self.step.as_ptr(), context, state, output) }
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
        context: &mut Context,
        state: NonNull<u8>,
    ) -> Result<(), Error> {
        // SAFETY: `new_state` matches `step`'s type, and the caller upholds
        // the rest.
        unsafe { (self.new_state)(self.step.as_ptr(), context, state) }
    }
}

/// `T`'s name, without its path or parameters, for tests to print plans.
#[cfg(test)]
fn type_name<T>() -> &'static str {
    let name = core::any::type_name::<T>();
    let name = name.split('<').next().unwrap_or(name);
    name.rsplit("::").next().unwrap_or(name)
}
