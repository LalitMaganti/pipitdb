//! The steps of a pipeline: `Source`, `Transform` and `Operator`, and
//! `DynSource`, `DynTransform` and `DynOperator`, the forms a pipeline stores.
//!
//! A step is a plan node: read-only while it runs. What changes lives in its
//! `State`, which each run creates. What a run's steps share is its
//! `Context`, passed to each.
//!
//! A step fails, with an `Error`, if it can't allocate what it needs or
//! read its data, which ends the run. Steps never drop rows instead.

#![allow(unsafe_code)]

use core::alloc::Layout;
use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::allocator::AllocError;
use crate::buffer::BUFFER_ALIGNMENT_BYTES;
use crate::containers::{Array, Box, ErasedBox};
use crate::context::Context;
use crate::env::Env;
use crate::erase::{drop_state, state_of, value_of, write_state};
use crate::error::Error;
use crate::row_batch::RowBatch;

/// Produces the batches a pipeline runs over.
pub trait Source {
    /// What a query's explanation calls it, such as `Filter`.
    const NAME: &'static str;

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
    /// What a query's explanation calls it, such as `Filter`.
    const NAME: &'static str;

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
    /// What a query's explanation calls it, such as `Filter`.
    const NAME: &'static str;

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
/// as a sort or an aggregation: the sink of the pipeline it consumes, which
/// it finalizes once that ends, and served by the source of the next.
pub trait Breaker {
    /// What a query's explanation calls it, such as `Filter`.
    const NAME: &'static str;

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

/// A breaker is the sink of the pipeline it consumes: one run's state, as
/// merging runs' comes with the executor that runs more than one. A
/// `Served` source serves it, as the next pipeline's.
impl<B: Breaker> Sink for B {
    const NAME: &'static str = B::NAME;
    type State = Option<B::State>;
    type Local = Option<B::State>;

    fn new_state(&self, _: &mut Context) -> Result<Self::State, Error> {
        Ok(None)
    }

    fn new_local(&self, context: &mut Context) -> Result<Self::Local, Error> {
        Ok(Some(Breaker::new_state(self, context)?))
    }

    fn consume(
        &self,
        context: &mut Context,
        local: &mut Self::Local,
        input: &RowBatch,
    ) -> Result<(), Error> {
        let Some(local) = local else { failed!() };
        Breaker::consume(self, context, local, input)
    }

    fn combine(
        &self,
        _: &mut Context,
        state: &mut Self::State,
        local: &mut Self::Local,
    ) -> Result<(), Error> {
        check!(state.is_none());
        *state = local.take();
        Ok(())
    }

    fn finalize(&self, context: &mut Context, state: &mut Self::State) -> Result<(), Error> {
        let Some(state) = state else { failed!() };
        Breaker::finalize(self, context, state)
    }
}

/// Consumes every batch of the pipeline it ends, building what a step of
/// another pipeline reads once it's finalized, such as a join's hash table
/// of its build side. Each run of the pipeline consumes into a local state
/// of its own, combined into the shared one when that run ends.
pub trait Sink {
    /// What a query's explanation calls it, such as `Filter`.
    const NAME: &'static str;

    /// What it builds, which what reads it gets once it's finalized.
    type State;
    /// What a run of its pipeline builds, until it's combined.
    type Local;

    fn new_state(&self, context: &mut Context) -> Result<Self::State, Error>;

    fn new_local(&self, context: &mut Context) -> Result<Self::Local, Error>;

    /// A sink that can spill prepares the batch first: it allocates all it
    /// needs, changing nothing that applying the batch wouldn't. Then it
    /// applies it, changing its state, allocating nothing. Failing for want
    /// of memory then leaves it as it was.
    fn consume(
        &self,
        context: &mut Context,
        local: &mut Self::Local,
        input: &RowBatch,
    ) -> Result<(), Error>;

    /// Moves what `local` holds into `state`, once its run ends. `local` is
    /// dropped after.
    fn combine(
        &self,
        context: &mut Context,
        state: &mut Self::State,
        local: &mut Self::Local,
    ) -> Result<(), Error>;

    /// Called once, after every run is combined.
    fn finalize(&self, context: &mut Context, state: &mut Self::State) -> Result<(), Error>;
}

/// What an operator did with its input.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Progress {
    /// The output holds everything left for this input, maybe nothing.
    NeedInput,
    /// The output holds some of it. Call again with the same input.
    MoreOutput,
    /// The output holds the last rows it will make, maybe none: it needs no
    /// more input, nor to finish, so nothing before it runs again.
    Done,
}

/// A step after a pipeline's source.
pub enum Step<'a> {
    Transform(DynTransform<'a>),
    Operator(DynOperator<'a>),
}

impl Step<'_> {
    pub(crate) fn state_layout(&self) -> Layout {
        match self {
            Step::Transform(transform) => transform.functions.state_layout,
            Step::Operator(operator) => operator.functions.state_layout,
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
            Step::Transform(transform) => transform.functions.drop_state,
            Step::Operator(operator) => operator.functions.drop_state,
        };
        // SAFETY: upheld by the caller.
        unsafe { drop_state(state) }
    }
}

/// A step of any type that lives for `'a`, owned in memory from an `Env`,
/// and its type's functions.
pub struct Erased<'a, F: 'static> {
    step: ErasedBox,
    pub(crate) functions: &'a StepFunctions<F>,
}

/// A step type's functions, and the layout of its state: one table for each
/// type. `F` is what a pipeline calls for each batch.
pub(crate) struct StepFunctions<F> {
    #[cfg_attr(
        not(any(test, feature = "explain")),
        expect(dead_code, reason = "only `explain` reads it")
    )]
    pub(crate) name: Name,
    pub(crate) state_layout: Layout,
    pub(crate) new_state: NewState,
    pub(crate) drop_state: unsafe fn(NonNull<u8>),
    pub(crate) run: F,
}

/// A step type's name, which only builds that explain queries keep.
#[derive(Clone, Copy)]
pub(crate) struct Name(#[cfg(any(test, feature = "explain"))] pub(crate) &'static str);

impl Name {
    pub(crate) const fn new(name: &'static str) -> Name {
        #[cfg(not(any(test, feature = "explain")))]
        let _ = name;
        Name(
            #[cfg(any(test, feature = "explain"))]
            name,
        )
    }
}

/// Makes a step's state at the given place, or fails without making it.
pub(crate) type NewState = unsafe fn(NonNull<()>, &mut Context, NonNull<u8>) -> Result<(), Error>;

pub type DynSource<'a> = Erased<'a, SourceNext>;
pub type DynTransform<'a> = Erased<'a, TransformProcess>;
pub type DynOperator<'a> = Erased<'a, OperatorFunctions>;
pub type DynSink<'a> = Erased<'a, SinkFunctions>;

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

/// A sink's functions besides those of its shared state, which its
/// `StepFunctions` has: those of its local state, and what a run calls.
pub struct SinkFunctions {
    pub(crate) local_layout: Layout,
    new_local: NewState,
    drop_local: unsafe fn(NonNull<u8>),
    consume: unsafe fn(NonNull<()>, &mut Context, NonNull<u8>, &RowBatch) -> Sunk,
    combine: unsafe fn(NonNull<()>, &mut Context, NonNull<u8>, NonNull<u8>) -> Sunk,
    finalize: unsafe fn(NonNull<()>, &mut Context, NonNull<u8>) -> Sunk,
}

type Sunk = Result<(), Error>;

impl<'a, F> Erased<'a, F> {
    /// A step from a table of functions for its type.
    ///
    /// # Safety
    ///
    /// The functions must take `step`, and the step must live for `'a`.
    pub(crate) unsafe fn from_parts(
        step: ErasedBox,
        functions: &'a StepFunctions<F>,
    ) -> Erased<'a, F> {
        Erased { step, functions }
    }
}

/// Holds `T`'s table.
struct Functions<'a, T>(PhantomData<(&'a (), T)>);

impl<'a, T: Source + 'a> Functions<'a, T> {
    const SOURCE: StepFunctions<SourceNext> = StepFunctions {
        name: Name::new(T::NAME),
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
    };
}

impl<'a, T: Transform + 'a> Functions<'a, T> {
    const TRANSFORM: StepFunctions<TransformProcess> = StepFunctions {
        name: Name::new(T::NAME),
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
    };
}

impl<'a, T: Operator + 'a> Functions<'a, T> {
    const OPERATOR: StepFunctions<OperatorFunctions> = StepFunctions {
        name: Name::new(T::NAME),
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
    };
}

impl<'a, T: Sink + 'a> Functions<'a, T> {
    const SINK: StepFunctions<SinkFunctions> = StepFunctions {
        name: Name::new(T::NAME),
        state_layout: Layout::new::<T::State>(),
        // SAFETY: only called with this sink and its states, as is each below.
        new_state: |sink, context, state| unsafe {
            let made = value_of::<T>(sink).new_state(context)?;
            write_state(state, made);
            Ok(())
        },
        drop_state: drop_state::<T::State>,
        run: SinkFunctions {
            local_layout: Layout::new::<T::Local>(),
            // SAFETY: as above.
            new_local: |sink, context, local| unsafe {
                let made = value_of::<T>(sink).new_local(context)?;
                write_state(local, made);
                Ok(())
            },
            drop_local: drop_state::<T::Local>,
            // SAFETY: as above.
            consume: |sink, context, local, input| unsafe {
                value_of::<T>(sink).consume(context, state_of::<T::Local>(local), input)
            },
            // SAFETY: as above.
            combine: |sink, context, state, local| unsafe {
                let (state, local) = (state_of::<T::State>(state), state_of::<T::Local>(local));
                value_of::<T>(sink).combine(context, state, local)
            },
            // SAFETY: as above.
            finalize: |sink, context, state| unsafe {
                value_of::<T>(sink).finalize(context, state_of::<T::State>(state))
            },
        },
    };
}

impl<'a> DynSource<'a> {
    pub fn new<T: Source + 'a>(env: &Env, source: T) -> Result<DynSource<'a>, AllocError> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        let step = Box::new(env, source)?.erase();
        let functions = &Functions::<'a, T>::SOURCE;
        Ok(Erased { step, functions })
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
        let batch = NonNull::from(batch).cast();
        // SAFETY: `run` matches `step`'s type, and the caller upholds the
        // rest.
        unsafe { (self.functions.run)(self.step.as_ptr(), context, state, batch) }
    }
}

impl<'a> DynTransform<'a> {
    pub fn new<T: Transform + 'a>(env: &Env, transform: T) -> Result<DynTransform<'a>, AllocError> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        let step = Box::new(env, transform)?.erase();
        let functions = &Functions::<'a, T>::TRANSFORM;
        Ok(Erased { step, functions })
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
        unsafe { (self.functions.run)(self.step.as_ptr(), context, state, batch) }
    }
}

impl<'a> DynOperator<'a> {
    pub fn new<T: Operator + 'a>(env: &Env, operator: T) -> Result<DynOperator<'a>, AllocError> {
        const { assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES) };
        let step = Box::new(env, operator)?.erase();
        let functions = &Functions::<'a, T>::OPERATOR;
        Ok(Erased { step, functions })
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
        unsafe { (self.functions.run.execute)(self.step.as_ptr(), context, state, input, output) }
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
        unsafe { (self.functions.run.finish)(self.step.as_ptr(), context, state, output) }
    }
}

impl<'a> DynSink<'a> {
    pub fn new<T: Sink + 'a>(env: &Env, sink: T) -> Result<DynSink<'a>, AllocError> {
        const {
            assert!(align_of::<T::State>() <= BUFFER_ALIGNMENT_BYTES);
            assert!(align_of::<T::Local>() <= BUFFER_ALIGNMENT_BYTES);
        };
        let step = Box::new(env, sink)?.erase();
        let functions = &Functions::<'a, T>::SINK;
        Ok(Erased { step, functions })
    }

    /// A new local state, for a run of its pipeline.
    pub(crate) fn new_local(&self, context: &mut Context) -> Result<OwnedState, Error> {
        let run = &self.functions.run;
        let made = (run.local_layout, run.new_local, run.drop_local);
        // SAFETY: `new_local` makes a local state of this sink, of its
        // layout, which `drop_local` drops.
        unsafe { OwnedState::new(context, self.step.as_ptr(), made) }
    }

    /// Its new shared state.
    pub(crate) fn new_shared(&self, context: &mut Context) -> Result<OwnedState, Error> {
        let functions = self.functions;
        let made = (functions.state_layout, functions.new_state, functions.drop_state);
        // SAFETY: as in `new_local`, for its shared state.
        unsafe { OwnedState::new(context, self.step.as_ptr(), made) }
    }

    /// # Safety
    ///
    /// `local` must be one of this sink's local states, from `new_local`.
    pub(crate) unsafe fn consume(
        &self,
        context: &mut Context,
        local: &OwnedState,
        input: &RowBatch,
    ) -> Result<(), Error> {
        // SAFETY: as in `DynSource::next`.
        unsafe { (self.functions.run.consume)(self.step.as_ptr(), context, local.state, input) }
    }

    /// # Safety
    ///
    /// `state` and `local` must be this sink's, from `new_shared` and
    /// `new_local`.
    pub(crate) unsafe fn combine(
        &self,
        context: &mut Context,
        state: &OwnedState,
        local: &OwnedState,
    ) -> Result<(), Error> {
        let (step, combine) = (self.step.as_ptr(), self.functions.run.combine);
        // SAFETY: as in `DynSource::next`.
        unsafe { combine(step, context, state.state, local.state) }
    }

    /// # Safety
    ///
    /// `state` must be this sink's, from `new_shared`.
    pub(crate) unsafe fn finalize(
        &self,
        context: &mut Context,
        state: &OwnedState,
    ) -> Result<(), Error> {
        // SAFETY: as in `DynSource::next`.
        unsafe { (self.functions.run.finalize)(self.step.as_ptr(), context, state.state) }
    }
}

/// A step's state outside a run's memory, such as a sink's, in memory of its
/// own from an `Env`: dropped with it.
pub(crate) struct OwnedState {
    /// Where the state is, in `_memory`.
    state: NonNull<u8>,
    /// Drops it.
    drop: unsafe fn(NonNull<u8>),
    /// Holds it.
    _memory: Array<u8>,
}

impl OwnedState {
    /// A state of `layout`, which `new` makes for `step`, and `drop` drops.
    ///
    /// # Safety
    ///
    /// `new` must make a value of `layout`, or fail without making one, and
    /// `drop` must drop that value.
    #[inline(never)]
    unsafe fn new(
        context: &mut Context,
        step: NonNull<()>,
        (layout, new, drop): (Layout, NewState, unsafe fn(NonNull<u8>)),
    ) -> Result<OwnedState, Error> {
        check!(layout.align() <= BUFFER_ALIGNMENT_BYTES);
        // SAFETY: `new` writes the state, and nothing reads the rest.
        let memory = unsafe { Array::<u8>::uninit(context.env(), layout.size().max(1))? };
        let state = memory.as_non_null();
        // SAFETY: upheld by the caller.
        unsafe { new(step, context, state)? };
        Ok(OwnedState { state, drop, _memory: memory })
    }

    /// Where the state is.
    pub(crate) fn as_ptr(&self) -> NonNull<u8> {
        self.state
    }
}

impl Drop for OwnedState {
    fn drop(&mut self) {
        // SAFETY: `new` made the state, which is dropped once.
        unsafe { (self.drop)(self.state) }
    }
}

impl<F> Erased<'_, F> {
    /// The step, of its type.
    pub(crate) fn step(&self) -> NonNull<()> {
        self.step.as_ptr()
    }

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
        unsafe { (self.functions.new_state)(self.step.as_ptr(), context, state) }
    }
}
