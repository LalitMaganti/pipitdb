//! `Pipeline`: a source and the steps its batches go through. `Execution`: one
//! run of it.
//!
//! A pipeline is also a source, of the batches it outputs, as Perfetto's
//! is: `into_source` makes one, whose runs are on their caller's `Context`.
//!
//! A pipeline can also end in a sink, which consumes what it outputs and
//! builds what a step of another pipeline reads whole, such as a join's hash
//! table of its build side. A plan's sinks are built when first read, by
//! `built`, once in a run, so a join whose probe side is empty builds none.
//!
//! The steps are cut into segments at each operator. The first segment is the
//! source and the transforms after it; each other is an operator and the
//! transforms after it. A segment's transforms run in place on the batches its
//! head makes, and each segment's batches are the next operator's input.

#![allow(unsafe_code)]

use core::alloc::Layout;
use core::cell::Cell;
use core::marker::PhantomData;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::allocator::AllocError;
use crate::buffer::BUFFER_ALIGNMENT_BYTES;
use crate::containers::{Array, Box, SmallVec};
use crate::context::Context;
use crate::env::Env;
use crate::erase::{drop_state, state_of, value_of, write_state};
use crate::error::Error;
use crate::plan::PLAN_NODES_MAX;
use crate::row_batch::RowBatch;
use crate::statement::Statement;
use crate::step::{
    Breaker, DynSink, DynSource, ErasedBatch, Progress, Sink, Source, SourceNext, Step,
    StepFunctions,
};

/// Read-only, so it can be run any number of times. Owns its source and steps.
pub struct Pipeline<'a> {
    source: DynSource<'a>,
    steps: SmallVec<Step<'a>>,
    segment_count: usize,
}

impl<'a> Pipeline<'a> {
    pub fn new(source: DynSource<'a>, steps: SmallVec<Step<'a>>) -> Pipeline<'a> {
        let operators = steps.iter().filter(|step| matches!(step, Step::Operator(_))).count();
        Pipeline { source, steps, segment_count: operators + 1 }
    }

    /// Creates the state of a run, in one allocation from `env`'s base
    /// allocator, which steps' states also allocate from; batches allocate
    /// from its slab allocators. Its steps read its `statement`, if it has
    /// one.
    pub fn start<'r>(
        &'r self,
        env: &'r Env,
        statement: Option<&'r Statement>,
    ) -> Result<Execution<'r>, Error> {
        let mut context = Context::new(env, statement);
        let run = Run::new(self, &mut context)?;
        Ok(Execution { run, context })
    }

    /// As `start`, of a plan whose pipelines that end in sinks are `sinks`,
    /// which its steps read by their ids.
    pub fn start_with_sinks<'r>(
        &'r self,
        env: &'r Env,
        statement: Option<&'r Statement>,
        sinks: &'r Sinks<'r>,
    ) -> Result<Execution<'r>, Error> {
        let mut context = Context::with_sinks(env, statement, sinks);
        let run = Run::new(self, &mut context)?;
        Ok(Execution { run, context })
    }

    /// This pipeline as a source of the batches it outputs. Each run of it
    /// is on the `Context` of the run it's in, so shares its memory, string
    /// pool and statement.
    pub fn into_source(self, env: &Env) -> Result<DynSource<'a>, AllocError> {
        let pipeline = Box::new(env, self)?.erase();
        // SAFETY: each function takes a pipeline and a run of it.
        Ok(unsafe { DynSource::from_parts(pipeline, &PIPELINE_SOURCE, "Pipeline") })
    }

    /// A `Segment` per segment, a `Slot` per step, then the source's state,
    /// then each step's. Calls `state` with each state's step, or `None` for
    /// the source, and its offset. Returns the slots' offset too.
    fn layout(
        &self,
        mut state: impl FnMut(Option<usize>, usize),
    ) -> Result<(Layout, usize), AllocError> {
        let segments = Layout::array::<Segment>(self.segment_count).map_err(|_| AllocError)?;
        let slots = Layout::array::<Slot>(self.steps.len()).map_err(|_| AllocError)?;
        let (layout, slots) = segments.extend(slots).map_err(|_| AllocError)?;
        let (mut layout, offset) =
            layout.extend(self.source.functions.state_layout).map_err(|_| AllocError)?;
        state(None, offset);
        for (i, step) in self.steps.iter().enumerate() {
            let offset;
            (layout, offset) = layout.extend(step.state_layout()).map_err(|_| AllocError)?;
            state(Some(i), offset);
        }
        check!(layout.align() <= BUFFER_ALIGNMENT_BYTES);
        Ok((layout, slots))
    }
}

/// A pipeline that ends in a sink: what a step reads whole, such as a
/// join's build side.
pub(crate) struct SinkPipeline<'a> {
    pub(crate) pipeline: Pipeline<'a>,
    pub(crate) sink: DynSink<'a>,
}

/// A plan's pipelines that end in sinks, which the steps of its other
/// pipelines read by id, with `with_built`.
pub struct Sinks<'a> {
    pipelines: SmallVec<SinkPipeline<'a>>,
    /// A number no other list has, which its ids carry, so an id is only
    /// read in a run of its own list's plan.
    list: u32,
}

/// The next list's number. 0 is no list's: a run without sinks has it.
static LISTS: AtomicU32 = AtomicU32::new(1);

impl<'a> Sinks<'a> {
    pub fn new(env: &Env) -> Result<Sinks<'a>, AllocError> {
        let pipelines = SmallVec::new(env, PLAN_NODES_MAX)?;
        Ok(Sinks { pipelines, list: LISTS.fetch_add(1, Ordering::Relaxed) })
    }

    /// Ends `pipeline` in `sink`, and returns its id.
    pub fn add<T: Sink + 'a>(
        &mut self,
        env: &Env,
        pipeline: Pipeline<'a>,
        sink: T,
    ) -> Result<SinkId<T>, AllocError> {
        let index = self.add_dyn(SinkPipeline { pipeline, sink: DynSink::new(env, sink)? })?;
        Ok(SinkId { index, list: self.list, sink: PhantomData })
    }

    /// As `add`, of any type, returning where it is.
    fn add_dyn(&mut self, pipeline: SinkPipeline<'a>) -> Result<u32, AllocError> {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
        let index = self.pipelines.len() as u32;
        self.pipelines.push(pipeline)?;
        Ok(index)
    }

    pub(crate) fn get(&self, index: u32) -> &SinkPipeline<'a> {
        at!(self.pipelines, index as usize)
    }

    pub(crate) fn list(&self) -> u32 {
        self.list
    }
}

/// The id of a sink of type `T` in a list of `Sinks`, which makes it: so
/// the state `with_built` gives for it is a `T`'s.
pub struct SinkId<T> {
    /// Where it is in its list.
    index: u32,
    /// Its list's number.
    list: u32,
    /// Its type.
    sink: PhantomData<fn() -> T>,
}

impl<T> Clone for SinkId<T> {
    fn clone(&self) -> SinkId<T> {
        *self
    }
}

impl<T> Copy for SinkId<T> {}

/// The source that serves what `breaker` builds: the first of the pipeline
/// after it.
pub fn served<'a, B: Breaker + 'a>(
    env: &Env,
    breaker: SinkId<B>,
) -> Result<DynSource<'a>, AllocError> {
    let (index, list) = (breaker.index, breaker.list);
    DynSource::new(env, Served { index, list, serve: serve::<B> })
}

/// Serves the next rows of what breaker `B`, at `breaker`, built, at
/// `state`, as `Breaker::serve`.
///
/// # Safety
///
/// `breaker` must be a `B`, and `state` its state, finalized.
unsafe fn serve<B: Breaker>(
    breaker: NonNull<()>,
    context: &mut Context,
    state: NonNull<u8>,
    output: &mut RowBatch,
) -> Result<bool, Error> {
    // SAFETY: upheld by the caller.
    let (breaker, state) = unsafe { (value_of::<B>(breaker), state_of::<Option<B::State>>(state)) };
    let Some(state) = state else { failed!() };
    breaker.serve(context, state, output)
}

/// Calls `f` with sink `id`, and its state, finalized, as `built` gives it.
/// Only the step `id` was made for reads it, so it changes it alone.
pub(crate) fn with_built<T: Sink, R>(
    context: &mut Context,
    id: SinkId<T>,
    f: impl FnOnce(&mut Context, &T, &mut T::State) -> R,
) -> Result<R, Error> {
    let (sink, state) = built_in(context, id.index, id.list)?;
    // SAFETY: sink `id` is a `T`, as only `Lowering::add_sink` makes ids,
    // so its state is a `T::State`. The plan holds the sink while the run
    // lasts. The state is in memory of its own, apart from the context's,
    // and the one step that reads it isn't running `f` already.
    let (sink, state) = unsafe { (sink.cast::<T>().as_ref(), state.cast::<T::State>().as_mut()) };
    Ok(f(context, sink, state))
}

/// The source of a pipeline after a breaker, which serves what the breaker
/// built of the pipeline before. One type for every breaker's, which only
/// its `serve` knows.
struct Served {
    /// Where the breaker, the sink of the pipeline before, is in its list.
    index: u32,
    /// Its list's number.
    list: u32,
    /// The breaker's `serve`.
    serve: Serve,
}

/// A breaker's `serve`, given it and its state.
type Serve =
    unsafe fn(NonNull<()>, &mut Context, NonNull<u8>, &mut RowBatch) -> Result<bool, Error>;

impl Source for Served {
    /// Whether the breaker has served all it holds.
    type State = bool;

    fn new_state(&self, _: &mut Context) -> Result<bool, Error> {
        Ok(false)
    }

    fn next(
        &self,
        context: &mut Context,
        done: &mut bool,
        batch: &mut RowBatch,
    ) -> Result<bool, Error> {
        if *done {
            return Ok(false);
        }
        batch.reset(0);
        let (breaker, state) = built_in(context, self.index, self.list)?;
        // SAFETY: `served` made this source for its breaker, of its type,
        // which `built_in` checked is the run's.
        let more = unsafe { (self.serve)(breaker, context, state, batch)? };
        *done = !more;
        Ok(true)
    }
}

/// Sink `index` of list `list`, which must be the run's, and its state,
/// finalized, as `built` gives it.
fn built_in(
    context: &mut Context,
    index: u32,
    list: u32,
) -> Result<(NonNull<()>, NonNull<u8>), Error> {
    check!(list == context.sinks_list());
    let state = built(context, index)?;
    Ok((context.sink_pipeline(index).sink.step(), state))
}

/// Sink `id`'s state, finalized: built by running its pipeline to its end
/// the first time the run asks for it. The run keeps it until it ends, so
/// it outlives the states of the steps that read it.
pub(crate) fn built(context: &mut Context, id: u32) -> Result<NonNull<u8>, Error> {
    if let Some(state) = context.built(id) {
        return Ok(state);
    }
    let SinkPipeline { pipeline, sink } = context.sink_pipeline(id);
    // One run, consuming into one local state, which it combines.
    let local = sink.new_local(context)?;
    {
        let mut run = Run::new(pipeline, context)?;
        let mut batch = RowBatch::new();
        while run.next(context, &mut batch)? {
            // SAFETY: `local` is the sink's.
            unsafe { sink.consume(context, &local, &batch)? };
        }
    }
    let state = sink.new_shared(context)?;
    // SAFETY: both states are the sink's.
    unsafe {
        sink.combine(context, &state, &local)?;
        drop(local);
        sink.finalize(context, &state)?;
    }
    let built = state.as_ptr();
    context.keep_built(id, state)?;
    Ok(built)
}

/// `after`, run after `breaker` of `input`'s rows, as a plan runs them:
/// `input` ends in it, and `after` starts with the source that serves it.
/// Returns the sinks the second reads, and it.
#[cfg(test)]
pub(crate) fn after_breaker<'a, B: Breaker + 'a>(
    input: Pipeline<'a>,
    breaker: B,
    after: impl IntoIterator<Item = Step<'a>, IntoIter: ExactSizeIterator>,
) -> (Sinks<'a>, Pipeline<'a>) {
    let env = Env::current();
    let mut sinks = Sinks::new(env).unwrap();
    let breaker = sinks.add(env, input, breaker).unwrap();
    let steps = SmallVec::fixed_from(env, after.into_iter()).unwrap();
    (sinks, Pipeline::new(served(env, breaker).unwrap(), steps))
}

#[cfg(test)]
impl Sinks<'_> {
    /// Each pipeline, then the sink it ends in, as `Pipeline::describe`
    /// does, and `then` after: such as `Scan | Sort / Served`.
    pub(crate) fn describe(&self, then: &Pipeline) -> alloc::string::String {
        let mut out = alloc::string::String::new();
        for SinkPipeline { pipeline, sink } in self.pipelines.iter() {
            out = out + &pipeline.describe() + " | " + sink.name + " / ";
        }
        out + &then.describe()
    }
}

#[cfg(test)]
impl Pipeline<'_> {
    /// Its source's and steps' types, in order, such as `Scan | Filter`.
    pub(crate) fn describe(&self) -> alloc::string::String {
        let steps = self.steps.iter().map(|step| match step {
            Step::Transform(transform) => transform.name,
            Step::Operator(operator) => operator.name,
        });
        let names: alloc::vec::Vec<&str> =
            core::iter::once(self.source.name).chain(steps).collect();
        names.join(" | ")
    }
}

/// Where a run is with a segment. Kept together, apart from the slots, so
/// moving between segments touches little memory.
struct Segment {
    /// The operator heading it; unused for the source's.
    head: usize,
    /// Where its transforms end: the next operator, or the number of steps.
    end: usize,
    /// For an operator's segment: where the operator is with its input.
    status: Status,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Status {
    /// Needs another input batch.
    Waiting,
    /// Its input holds a batch to execute.
    Ready,
    /// There's no more input; the operator is finishing.
    Ended,
    /// The operator has nothing left to output.
    Done,
}

/// The status an operator leaves its segment in, by what it said it did and
/// the status it ran in: a lookup, as every operator call makes one.
static NEXT: [[Status; 4]; 3] = next();

const fn next() -> [[Status; 4]; 3] {
    use Status::{Done, Ended, Ready, Waiting};
    let mut next = [[Waiting, Ready, Ended, Done]; 3];
    next[Progress::NeedInput as usize] = [Waiting, Waiting, Done, Done];
    next[Progress::Done as usize] = [Done; 4];
    next
}

/// What a run keeps for a step.
struct Slot<'p> {
    /// Where the step's state is in the run's memory.
    state: usize,
    /// For operators: their input.
    input: RowBatch<'p>,
}

/// What a segment did when run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Made {
    /// A batch with rows.
    Batch,
    /// A batch with no rows, which is dropped.
    Nothing,
    /// No more batches.
    End,
}

pub struct Execution<'p> {
    run: Run<'p>,
    context: Context<'p>,
}

/// A run's memory, and where it is in it.
struct Run<'p> {
    /// What's run.
    pipeline: &'p Pipeline<'p>,
    /// The segments, steps' slots and states, laid out as `layout` says.
    memory: NonNull<u8>,
    /// Where the slots start in `memory`.
    slots: usize,
    /// Where the source's state starts in `memory`.
    source_state: usize,
    /// How many states are made: the source's, then each step's, in order.
    made: usize,
    /// Why a step failed, if one has. Its segment then reports that it
    /// ended, so the loop pays for failures only where segments end.
    failed: Cell<Option<Error>>,
    /// Frees `memory`.
    _memory: Array<u8>,
}

/// A pipeline's functions as a source, whose state is a run of it.
const PIPELINE_SOURCE: StepFunctions<SourceNext> = StepFunctions {
    state_layout: Layout::new::<Run<'static>>(),
    new_state: new_run,
    drop_state: drop_state::<Run<'static>>,
    run: next_of_run,
};

/// Makes a run of `pipeline`, a `Pipeline`, at `state`, as a source's state.
///
/// # Safety
///
/// As for `DynSource::from_parts`'s functions.
unsafe fn new_run(
    pipeline: NonNull<()>,
    context: &mut Context,
    state: NonNull<u8>,
) -> Result<(), Error> {
    // SAFETY: a source outlives its states, so `'static` stands in for how
    // long the pipeline lives, which its run doesn't outlive.
    let pipeline = unsafe { value_of::<Pipeline<'static>>(pipeline) };
    let run = Run::new(pipeline, context)?;
    // SAFETY: `state` has room for a run, as the source's layout says.
    unsafe { write_state(state, run) };
    Ok(())
}

/// Fills `batch` from the run at `state`, as a source's next batch.
///
/// # Safety
///
/// As for `DynSource::from_parts`'s functions.
unsafe fn next_of_run(
    _: NonNull<()>,
    context: &mut Context,
    state: NonNull<u8>,
    batch: ErasedBatch,
) -> Result<bool, Error> {
    // SAFETY: `state` holds a run made by `new_run`, and `batch` is the
    // caller's, erased as the run's is.
    unsafe { state_of::<Run>(state).next(context, batch.cast().as_mut()) }
}

impl<'p> Execution<'p> {
    /// As `Run::next`.
    pub fn next(&mut self, output: &mut RowBatch<'p>) -> Result<bool, Error> {
        self.run.next(&mut self.context, output)
    }
}

impl<'p> Run<'p> {
    /// A run of `pipeline` on `context`, whose `Env` its memory is from,
    /// with every state made, or, if one fails, those made dropped.
    fn new(pipeline: &'p Pipeline<'p>, context: &mut Context) -> Result<Run<'p>, Error> {
        let Ok((layout, slots)) = pipeline.layout(|_, _| {}) else {
            failed!();
        };
        // SAFETY: what follows writes every byte it reads.
        let owned = unsafe { Array::<u8>::uninit(context.env(), layout.size())? };
        let memory = owned.as_non_null();
        let failed = Cell::new(None);
        let mut run =
            Run { pipeline, memory, slots, source_state: 0, made: 0, failed, _memory: owned };
        let mut failed = None;
        // SAFETY: the memory was allocated with this layout, so it has room
        // for each state at its offset, and for the slots.
        let layout = pipeline.layout(|i, state| unsafe {
            if failed.is_some() {
                return;
            }
            let made = match i {
                None => pipeline.source.new_state(context, memory.add(state)),
                Some(i) => at!(pipeline.steps, i).new_state(context, memory.add(state)),
            };
            if let Err(error) = made {
                failed = Some(error);
                return;
            }
            match i {
                None => run.source_state = state,
                Some(i) => {
                    // Built in place: a batch is too big to build and then copy.
                    let slot = memory.add(slots).cast::<Slot>().add(i).as_ptr();
                    (&raw mut (*slot).state).write(state);
                    RowBatch::init(&raw mut (*slot).input);
                }
            }
            run.made += 1;
        });
        check!(layout.is_ok());
        if let Some(error) = failed {
            // Dropping it drops the states made.
            return Err(error);
        }

        let steps = &pipeline.steps;
        let mut k = 0;
        let mut segment = Segment { head: 0, end: 0, status: Status::Ready };
        for (i, step) in steps.iter().enumerate() {
            if let Step::Operator(_) = step {
                segment.end = i;
                // SAFETY: there is a segment per operator, and one more.
                unsafe { run.segment(k).write(segment) };
                k += 1;
                segment = Segment { head: i, end: 0, status: Status::Waiting };
            }
        }
        segment.end = steps.len();
        // SAFETY: as above.
        unsafe { run.segment(k).write(segment) };
        Ok(run)
    }

    /// Fills `output` with the next batch, or returns false when there are
    /// none left. Fails if a step does, such as if it can't allocate what it
    /// needs; the run can't go on after that, so later calls fail too.
    ///
    /// Starting from the last segment, it moves to the one before when a
    /// segment's operator needs input, and to the one after when a segment
    /// makes a batch or ends.
    fn next(&mut self, context: &mut Context, output: &mut RowBatch<'p>) -> Result<bool, Error> {
        if let Some(error) = self.failed.get() {
            return Err(error);
        }
        let last = self.pipeline.segment_count - 1;
        let mut k = self.ready(last);
        loop {
            match self.make(context, k, output) {
                Made::Nothing => k = self.ready(k),
                Made::Batch if k == last => return Ok(true),
                Made::End if let Some(error) = self.failed.get() => return Err(error),
                Made::End if k == last => return Ok(false),
                made => {
                    k += 1;
                    let status = if made == Made::Batch { Status::Ready } else { Status::Ended };
                    // SAFETY: `k` is at most `last`.
                    unsafe { (*self.segment(k)).status = status };
                }
            }
        }
    }

    /// The nearest segment from `k` back whose head can run: the first whose
    /// operator isn't waiting for input, or the source's.
    #[inline]
    fn ready(&self, k: usize) -> usize {
        if k > 0 && self.waiting(k) { self.ready_before(k) } else { k }
    }

    /// As `ready`, for segment `k`, which is waiting.
    #[inline(never)]
    fn ready_before(&self, mut k: usize) -> usize {
        k -= 1;
        while k > 0 && self.waiting(k) {
            k -= 1;
        }
        k
    }

    fn waiting(&self, k: usize) -> bool {
        // SAFETY: the segments were written by `Run::new`.
        unsafe { (*self.segment(k)).status == Status::Waiting }
    }

    /// Runs segment `k` once. Its batch goes to the next operator's input, or
    /// to `output` if it is the last segment.
    fn make(&self, context: &mut Context, k: usize, output: &mut RowBatch<'p>) -> Made {
        // SAFETY: the segments were written by `Run::new`.
        let segment = unsafe { &mut *self.segment(k) };
        let end = segment.end;
        let batch = if end == self.pipeline.steps.len() {
            output
        } else {
            // SAFETY: the next operator's slot, which nothing else borrows.
            unsafe { &mut (*self.slot(end)).input }
        };
        let (made, first) = if k == 0 {
            (self.read_source(context, batch), 0)
        } else {
            (self.execute(context, segment, batch), segment.head + 1)
        };
        if made != Made::Batch {
            return made;
        }
        let mut i = first;
        while i < end {
            let Step::Transform(transform) = at!(self.pipeline.steps, i) else {
                failed!();
            };
            // SAFETY: the transform's state was made by `Run::new`.
            if let Err(error) = unsafe { transform.process(context, self.state(Some(i)), batch) } {
                return self.fail(error);
            }
            // Later steps would do nothing with it, or load values no one
            // reads.
            if batch.selection().is_empty() {
                return Made::Nothing;
            }
            i += 1;
        }
        if batch.selection().is_empty() { Made::Nothing } else { Made::Batch }
    }

    fn read_source(&self, context: &mut Context, batch: &mut RowBatch<'p>) -> Made {
        // Not reset here: the source resets it as it fills it.
        // SAFETY: the source's state was made by `Run::new`.
        match unsafe { self.pipeline.source.next(context, self.state(None), batch) } {
            Ok(true) => Made::Batch,
            Ok(false) => Self::end(batch),
            Err(error) => self.fail(error),
        }
    }

    /// Ends a segment, freeing its last batch's buffers before what follows
    /// finishes.
    #[cold]
    fn end(batch: &mut RowBatch) -> Made {
        batch.reset(0);
        Made::End
    }

    /// Records that a step failed, and ends its segment.
    #[cold]
    fn fail(&self, error: Error) -> Made {
        self.failed.set(Some(error));
        Made::End
    }

    /// Runs the operator heading `segment` once.
    fn execute(
        &self,
        context: &mut Context,
        segment: &mut Segment,
        output: &mut RowBatch<'p>,
    ) -> Made {
        let op = segment.head;
        let Step::Operator(operator) = at!(self.pipeline.steps, op) else {
            failed!();
        };
        // SAFETY: the operator's slot, which nothing else borrows.
        let input = unsafe { &(*self.slot(op)).input };
        let state = self.state(Some(op));
        // SAFETY: the operator's state was made by `Run::new`.
        let progress = unsafe {
            match segment.status {
                Status::Done => return Self::end(output),
                Status::Ready => {
                    output.reset(0);
                    operator.execute(context, state, input, output)
                }
                Status::Ended => {
                    output.reset(0);
                    operator.finish(context, state, output)
                }
                Status::Waiting => failed!(),
            }
        };
        let progress = match progress {
            Ok(progress) => progress,
            Err(error) => return self.fail(error),
        };
        segment.status = *at!(at!(NEXT, progress as usize), segment.status as usize);
        // As in Perfetto, the steps after an operator see only batches with rows.
        if output.selection().is_empty() { Made::Nothing } else { Made::Batch }
    }

    fn segment(&self, k: usize) -> *mut Segment {
        check!(k < self.pipeline.segment_count);
        // SAFETY: the segments start the run's memory.
        unsafe { self.memory.cast::<Segment>().as_ptr().add(k) }
    }

    fn slot(&self, i: usize) -> *mut Slot<'p> {
        check!(i < self.pipeline.steps.len());
        // SAFETY: the slots are at `self.slots`.
        unsafe { self.memory.add(self.slots).cast::<Slot>().as_ptr().add(i) }
    }

    /// The state of step `i`, or of the source.
    fn state(&self, i: Option<usize>) -> NonNull<u8> {
        let offset = match i {
            None => self.source_state,
            // SAFETY: each slot was written by `Run::new`.
            Some(i) => unsafe { (*self.slot(i)).state },
        };
        // SAFETY: offsets come from the run's layout.
        unsafe { self.memory.add(offset) }
    }
}

impl Drop for Run<'_> {
    fn drop(&mut self) {
        let pipeline = self.pipeline;
        if self.made == 0 {
            return;
        }
        // SAFETY: each was made by `Run::new`, and is dropped once.
        unsafe {
            (pipeline.source.functions.drop_state)(self.state(None));
            for (i, step) in pipeline.steps.iter().enumerate().take(self.made - 1) {
                step.drop_state(self.state(Some(i)));
                self.slot(i).drop_in_place();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::buffer::Buffer;
    use alloc::rc::Rc;
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;

    use crate::column::{ColumnView, DataType};
    use crate::context::Context;
    use crate::env::Env;
    use crate::selection::Kept;
    use crate::step::{Breaker, DynOperator, DynTransform, Operator, Sink, Source, Transform};

    fn int64s(values: &[i64]) -> ColumnView {
        let env = Env::current();
        let mut buffer = Buffer::allocate(env, values.len() * 8).unwrap();
        buffer.as_mut_slice::<i64>().copy_from_slice(values);
        ColumnView::new(&mut Context::new(env, None), DataType::Int64, buffer, None).unwrap()
    }

    /// `batches` batches of two rows, with two columns, `[2i, 2i + 1]` and
    /// their negations.
    struct Numbers {
        batches: i64,
    }

    impl Source for Numbers {
        type State = i64;

        fn new_state(&self, _: &mut Context) -> Result<i64, Error> {
            Ok(0)
        }

        fn next(&self, _: &mut Context, i: &mut i64, batch: &mut RowBatch) -> Result<bool, Error> {
            if *i == self.batches {
                return Ok(false);
            }
            batch.reset(2);
            let first = *i * 2;
            assert!(batch.push_column(int64s(&[first, first + 1])).is_ok());
            assert!(batch.push_column(int64s(&[-first, -first - 1])).is_ok());
            *i += 1;
            Ok(true)
        }
    }

    struct Reverse;

    impl Transform for Reverse {
        type State = ();

        fn new_state(&self, _: &mut Context) -> Result<(), Error> {
            Ok(())
        }

        fn process(&self, _: &mut Context, (): &mut (), batch: &mut RowBatch) -> Result<(), Error> {
            let count = batch.column_count();
            let reversed: Vec<ColumnView> =
                (0..count).rev().map(|i| batch.column(i).clone()).collect();
            for (i, column) in (0..).zip(reversed) {
                batch.set_column(i, column);
            }
            Ok(())
        }
    }

    /// Adds the batch's position in the run as a column. The state holds
    /// `live`, to check states are dropped.
    struct Position {
        live: Rc<()>,
    }

    impl Transform for Position {
        type State = (i64, Rc<()>);

        fn new_state(&self, _: &mut Context) -> Result<(i64, Rc<()>), Error> {
            Ok((0, self.live.clone()))
        }

        fn process(
            &self,
            _: &mut Context,
            (position, _): &mut (i64, Rc<()>),
            batch: &mut RowBatch,
        ) -> Result<(), Error> {
            let column = int64s(&alloc::vec![*position; batch.row_count() as usize]);
            assert!(batch.push_column(column).is_ok());
            *position += 1;
            Ok(())
        }
    }

    /// Empties every other batch.
    struct SkipOdd;

    impl Transform for SkipOdd {
        type State = bool;

        fn new_state(&self, _: &mut Context) -> Result<bool, Error> {
            Ok(false)
        }

        fn process(
            &self,
            _: &mut Context,
            odd: &mut bool,
            batch: &mut RowBatch,
        ) -> Result<(), Error> {
            if *odd {
                batch.reset(0);
            }
            *odd = !*odd;
            Ok(())
        }
    }

    /// Keeps the rows whose first column is even.
    struct KeepEven;

    impl Transform for KeepEven {
        type State = ();

        fn new_state(&self, _: &mut Context) -> Result<(), Error> {
            Ok(())
        }

        fn process(&self, _: &mut Context, (): &mut (), batch: &mut RowBatch) -> Result<(), Error> {
            let column = batch.column(0).clone();
            batch.selection_mut().retain(|row| column.int64s()[row as usize] % 2 == 0);
            Ok(())
        }
    }

    /// Keeps no rows.
    struct KeepNone;

    impl Transform for KeepNone {
        type State = ();

        fn new_state(&self, _: &mut Context) -> Result<(), Error> {
            Ok(())
        }

        fn process(&self, _: &mut Context, (): &mut (), batch: &mut RowBatch) -> Result<(), Error> {
            batch.selection_mut().retain(|_| false);
            Ok(())
        }
    }

    /// One batch of one row, its column's values allocated from `live`.
    struct Counted<'a> {
        live: &'a Env,
    }

    impl Source for Counted<'_> {
        type State = bool;

        fn new_state(&self, _: &mut Context) -> Result<bool, Error> {
            Ok(false)
        }

        fn next(
            &self,
            _: &mut Context,
            made: &mut bool,
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            let env = Env::current();
            if *made {
                return Ok(false);
            }
            *made = true;
            batch.reset(1);
            let values = Buffer::allocate(self.live, 8).unwrap();
            let column =
                ColumnView::new(&mut Context::new(env, None), DataType::Int64, values, None)
                    .unwrap();
            assert!(batch.push_column(column).is_ok());
            Ok(true)
        }
    }

    /// Outputs nothing, and checks when it finishes that nothing allocated
    /// from `live` is held: it uses only `empty` bytes.
    struct Released<'a> {
        live: &'a Env,
        empty: usize,
    }

    impl Operator for Released<'_> {
        type State = ();

        fn new_state(&self, _: &mut Context) -> Result<(), Error> {
            Ok(())
        }

        fn execute(
            &self,
            _: &mut Context,
            (): &mut (),
            _: &RowBatch,
            _: &mut RowBatch,
        ) -> Result<Progress, Error> {
            Ok(Progress::NeedInput)
        }

        fn finish(
            &self,
            _: &mut Context,
            (): &mut (),
            _: &mut RowBatch,
        ) -> Result<Progress, Error> {
            assert_eq!(self.live.used(), self.empty);
            Ok(Progress::NeedInput)
        }
    }

    /// Outputs each input a row at a time.
    struct Split;

    impl Operator for Split {
        type State = u32;

        fn new_state(&self, _: &mut Context) -> Result<u32, Error> {
            Ok(0)
        }

        fn execute(
            &self,
            _: &mut Context,
            row: &mut u32,
            input: &RowBatch,
            output: &mut RowBatch,
        ) -> Result<Progress, Error> {
            output.reset(1);
            for column in 0..input.column_count() {
                assert!(output.push_column(input.column(column).slice(*row, 1)).is_ok());
            }
            *row += 1;
            if *row < input.row_count() {
                return Ok(Progress::MoreOutput);
            }
            *row = 0;
            Ok(Progress::NeedInput)
        }

        fn finish(
            &self,
            _: &mut Context,
            _: &mut u32,
            _: &mut RowBatch,
        ) -> Result<Progress, Error> {
            Ok(Progress::NeedInput)
        }
    }

    /// Outputs one row, the sum of its input's first column, once the input
    /// ends.
    struct Sum;

    impl Operator for Sum {
        type State = i64;

        fn new_state(&self, _: &mut Context) -> Result<i64, Error> {
            Ok(0)
        }

        fn execute(
            &self,
            _: &mut Context,
            sum: &mut i64,
            input: &RowBatch,
            _: &mut RowBatch,
        ) -> Result<Progress, Error> {
            *sum += input.column(0).int64s().iter().sum::<i64>();
            Ok(Progress::NeedInput)
        }

        fn finish(
            &self,
            _: &mut Context,
            sum: &mut i64,
            output: &mut RowBatch,
        ) -> Result<Progress, Error> {
            output.reset(1);
            assert!(output.push_column(int64s(&[*sum])).is_ok());
            Ok(Progress::NeedInput)
        }
    }

    /// Allocates its state, a number, from the run's `Env`, or fails to
    /// if `fail`.
    struct Boxed {
        fail: bool,
    }

    impl Transform for Boxed {
        type State = crate::containers::Box<i64>;

        fn new_state(&self, context: &mut Context) -> Result<Self::State, Error> {
            if self.fail {
                return Err(Error::OutOfMemory);
            }
            Ok(crate::containers::Box::new(context.env(), 7)?)
        }

        fn process(
            &self,
            _: &mut Context,
            number: &mut Self::State,
            batch: &mut RowBatch,
        ) -> Result<(), Error> {
            let column = int64s(&alloc::vec![**number; batch.row_count() as usize]);
            assert!(batch.push_column(column).is_ok());
            Ok(())
        }
    }

    /// Outputs its input's first column backwards, two rows at a time.
    struct Backwards;

    impl Breaker for Backwards {
        type State = Vec<i64>;

        fn new_state(&self, _: &mut Context) -> Result<Vec<i64>, Error> {
            Ok(Vec::new())
        }

        fn consume(
            &self,
            _: &mut Context,
            values: &mut Vec<i64>,
            input: &RowBatch,
        ) -> Result<(), Error> {
            let column = input.column(0).int64s();
            match input.selection().kept() {
                Kept::Prefix => {
                    values.extend_from_slice(&column[..input.selection().len() as usize]);
                }
                Kept::None => {}
                Kept::Select(rows) => {
                    values.extend(rows.iter().map(|&row| column[usize::from(row)]));
                }
            }
            Ok(())
        }

        fn finalize(&self, _: &mut Context, values: &mut Vec<i64>) -> Result<(), Error> {
            values.reverse();
            Ok(())
        }

        fn serve(
            &self,
            _: &mut Context,
            values: &mut Vec<i64>,
            output: &mut RowBatch,
        ) -> Result<bool, Error> {
            let n = values.len().min(2);
            output.reset(u32::try_from(n).unwrap());
            assert!(output.push_column(int64s(&values[..n])).is_ok());
            values.drain(..n);
            Ok(!values.is_empty())
        }
    }

    /// Fails on its second batch, as a step reading damaged data would.
    struct FailSecond;

    impl Transform for FailSecond {
        type State = bool;

        fn new_state(&self, _: &mut Context) -> Result<bool, Error> {
            Ok(false)
        }

        fn process(&self, _: &mut Context, seen: &mut bool, _: &mut RowBatch) -> Result<(), Error> {
            if *seen {
                return Err(Error::Corrupt);
            }
            *seen = true;
            Ok(())
        }
    }

    /// The sum of the first column of what it consumes. It counts in
    /// `builds` how many times it's finalized.
    struct Total<'a> {
        builds: &'a Cell<u32>,
    }

    impl Sink for Total<'_> {
        type State = i64;
        type Local = i64;

        fn new_state(&self, _: &mut Context) -> Result<i64, Error> {
            Ok(0)
        }

        fn new_local(&self, _: &mut Context) -> Result<i64, Error> {
            Ok(0)
        }

        fn consume(&self, _: &mut Context, sum: &mut i64, input: &RowBatch) -> Result<(), Error> {
            *sum += input.column(0).int64s().iter().sum::<i64>();
            Ok(())
        }

        fn combine(&self, _: &mut Context, total: &mut i64, sum: &mut i64) -> Result<(), Error> {
            *total += *sum;
            Ok(())
        }

        fn finalize(&self, _: &mut Context, _: &mut i64) -> Result<(), Error> {
            self.builds.set(self.builds.get() + 1);
            Ok(())
        }
    }

    /// Adds a column of sink `total`'s sum to each batch.
    struct WithTotal<'a> {
        total: SinkId<Total<'a>>,
    }

    impl Operator for WithTotal<'_> {
        type State = ();

        fn new_state(&self, _: &mut Context) -> Result<(), Error> {
            Ok(())
        }

        fn execute(
            &self,
            context: &mut Context,
            (): &mut (),
            input: &RowBatch,
            output: &mut RowBatch,
        ) -> Result<Progress, Error> {
            let total = with_built(context, self.total, |_, _, total| *total)?;
            output.copy_from(input);
            let column = int64s(&alloc::vec![total; input.row_count() as usize]);
            assert!(output.push_column(column).is_ok());
            Ok(Progress::NeedInput)
        }

        fn finish(
            &self,
            _: &mut Context,
            (): &mut (),
            _: &mut RowBatch,
        ) -> Result<Progress, Error> {
            Ok(Progress::NeedInput)
        }
    }

    /// Each batch's rows, as lists of values.
    fn batches(execution: &mut Execution) -> Vec<Vec<Vec<i64>>> {
        let mut batch = RowBatch::new();
        let mut batches = Vec::new();
        while execution.next(&mut batch).unwrap() {
            let rows = (0..batch.row_count() as usize).map(|row| {
                (0..batch.column_count()).map(|i| batch.column(i).int64s()[row]).collect()
            });
            batches.push(rows.collect());
        }
        batches
    }

    fn transform<'a>(transform: impl Transform + 'a) -> Step<'a> {
        let env = Env::current();
        Step::Transform(DynTransform::new(env, transform).unwrap())
    }

    fn operator<'a>(operator: impl Operator + 'a) -> Step<'a> {
        let env = Env::current();
        Step::Operator(DynOperator::new(env, operator).unwrap())
    }

    fn pipeline<'a>(
        source: impl Source + 'a,
        steps: impl IntoIterator<Item = Step<'a>>,
    ) -> Pipeline<'a> {
        let env = Env::current();
        with_source(DynSource::new(env, source).unwrap(), steps)
    }

    /// `steps` after `inner`, run as a source.
    fn nested<'a>(inner: Pipeline<'a>, steps: impl IntoIterator<Item = Step<'a>>) -> Pipeline<'a> {
        let env = Env::current();
        with_source(inner.into_source(env).unwrap(), steps)
    }

    fn with_source<'a>(
        source: DynSource<'a>,
        steps: impl IntoIterator<Item = Step<'a>>,
    ) -> Pipeline<'a> {
        let env = Env::current();
        let mut owned = crate::containers::SmallVec::new(env, 8).unwrap();
        for step in steps {
            assert!(owned.push(step).is_ok());
        }
        Pipeline::new(source, owned)
    }

    #[test]
    fn runs_a_source_alone() {
        let pipeline = pipeline(Numbers { batches: 2 }, []);
        let expected = [[[0, 0], [1, -1]], [[2, -2], [3, -3]]];
        let query = Env::current();
        assert_eq!(batches(&mut pipeline.start(query, None).unwrap()), expected);
    }

    #[test]
    fn transforms_each_batch_in_order() {
        let position = Position { live: Rc::new(()) };
        let pipeline = pipeline(Numbers { batches: 2 }, [transform(Reverse), transform(position)]);
        let expected = [[[0, 0, 0], [-1, 1, 0]], [[-2, 2, 1], [-3, 3, 1]]];
        let query = Env::current();
        assert_eq!(batches(&mut pipeline.start(query, None).unwrap()), expected);
    }

    #[test]
    fn operators_output_more_than_once_for_an_input() {
        let position = Position { live: Rc::new(()) };
        let pipeline = pipeline(Numbers { batches: 2 }, [operator(Split), transform(position)]);
        let expected = [[[0, 0, 0]], [[1, -1, 1]], [[2, -2, 2]], [[3, -3, 3]]];
        let query = Env::current();
        assert_eq!(batches(&mut pipeline.start(query, None).unwrap()), expected);
    }

    #[test]
    fn operators_finish_after_their_input_ends() {
        let steps = [operator(Split), transform(SkipOdd), operator(Sum)];
        let pipeline = pipeline(Numbers { batches: 3 }, steps);
        let query = Env::current();
        assert_eq!(batches(&mut pipeline.start(query, None).unwrap()), [[[2 + 4]]]);
    }

    #[test]
    fn operators_finish_without_their_last_input() {
        let live = Env::new(crate::env::Memory::new(Heap, usize::MAX)).unwrap();
        let empty = live.used();
        let released = Released { live: &live, empty };
        let pipeline = pipeline(Counted { live: &live }, [operator(released)]);
        assert!(batches(&mut pipeline.start(Env::current(), None).unwrap()).is_empty());
    }

    #[test]
    fn breakers_output_once_their_input_ends() {
        let input = pipeline(Numbers { batches: 3 }, []);
        let (sinks, pipeline) = after_breaker(input, Backwards, []);
        assert_eq!(pipeline.describe(), "Served");
        let query = Env::current();
        assert_eq!(
            batches(&mut pipeline.start_with_sinks(query, None, &sinks).unwrap()),
            [[[5], [4]], [[3], [2]], [[1], [0]]]
        );
    }

    #[test]
    fn steps_after_a_source_see_only_batches_with_rows() {
        let input = pipeline(Numbers { batches: 3 }, []);
        let (sinks, pipeline) = after_breaker(input, Backwards, [transform(SkipOdd)]);
        let query = Env::current();
        let mut execution = pipeline.start_with_sinks(query, None, &sinks).unwrap();
        assert_eq!(batches(&mut execution), [[[5], [4]], [[1], [0]]]);
    }

    #[test]
    fn each_run_has_its_own_state() {
        let live = Rc::new(());
        let position = Position { live: live.clone() };
        let pipeline = pipeline(Numbers { batches: 2 }, [operator(Split), transform(position)]);

        let query = Env::current();
        let mut first = pipeline.start(query, None).unwrap();
        let query = Env::current();
        let mut second = pipeline.start(query, None).unwrap();
        assert_eq!(batches(&mut first), batches(&mut second));
        // Ours, the pipeline's step, and a state for each run.
        assert_eq!(Rc::strong_count(&live), 4);
        drop((first, second));
        assert_eq!(Rc::strong_count(&live), 2);
        drop(pipeline);
        assert_eq!(Rc::strong_count(&live), 1);
    }

    #[test]
    fn states_allocate_and_can_fail() {
        let boxed = pipeline(Numbers { batches: 1 }, [transform(Boxed { fail: false })]);
        let query = Env::current();
        assert_eq!(batches(&mut boxed.start(query, None).unwrap()), [[[0, 0, 7], [1, -1, 7]]]);

        // The states made before the one that fails are dropped.
        let live = Rc::new(());
        let position = Position { live: live.clone() };
        let steps = [transform(position), transform(Boxed { fail: true })];
        let failing = pipeline(Numbers { batches: 1 }, steps);
        let query = Env::current();
        assert!(failing.start(query, None).is_err());
        assert_eq!(Rc::strong_count(&live), 2);
    }

    #[test]
    fn passes_selections_on_and_skips_batches_with_none() {
        let even = pipeline(Numbers { batches: 3 }, [transform(KeepEven), transform(KeepEven)]);
        let query = Env::current();
        let mut execution = even.start(query, None).unwrap();
        let mut batch = RowBatch::new();
        let mut kept = alloc::vec::Vec::new();
        while execution.next(&mut batch).unwrap() {
            let Kept::Select(rows) = batch.selection().kept() else { panic!("not narrowed") };
            kept.extend(rows.iter().map(|&row| batch.column(0).int64s()[row as usize]));
        }
        // Each batch has 2i and 2i + 1, so one row of each is kept.
        assert_eq!(kept, [0, 2, 4]);

        let none = pipeline(Numbers { batches: 3 }, [transform(KeepNone)]);
        let query = Env::current();
        assert!(!none.start(query, None).unwrap().next(&mut RowBatch::new()).unwrap());
        // Steps after one that keeps no rows don't run: this one would fail.
        let stopped =
            pipeline(Numbers { batches: 3 }, [transform(KeepNone), transform(FailSecond)]);
        assert_eq!(stopped.start(query, None).unwrap().next(&mut RowBatch::new()), Ok(false));
    }

    #[test]
    fn a_failing_step_ends_the_run() {
        let failing = pipeline(Numbers { batches: 3 }, [transform(FailSecond)]);
        let query = Env::current();
        let mut execution = failing.start(query, None).unwrap();
        let mut batch = RowBatch::new();
        assert_eq!(execution.next(&mut batch), Ok(true));
        assert_eq!(execution.next(&mut batch), Err(Error::Corrupt));
        // Rows aren't skipped: the run stays failed.
        assert_eq!(execution.next(&mut batch), Err(Error::Corrupt));
    }

    /// Interns `string` in the run's string pool, checking it's numbered
    /// `number`.
    struct Interns {
        string: &'static [u8],
        number: u32,
    }

    impl Transform for Interns {
        type State = ();

        fn new_state(&self, _: &mut Context) -> Result<(), Error> {
            Ok(())
        }

        fn process(
            &self,
            context: &mut Context,
            (): &mut (),
            _: &mut RowBatch,
        ) -> Result<(), Error> {
            let env = context.env();
            let (_, number) = context.string_pool()?.intern_numbered(env, self.string)?;
            assert_eq!(number, self.number);
            Ok(())
        }
    }

    #[test]
    fn runs_a_pipeline_as_a_source() {
        // As its steps would run after its source, before the others.
        let flat = pipeline(Numbers { batches: 2 }, [transform(Reverse), operator(Split)]);
        let inner = pipeline(Numbers { batches: 2 }, [transform(Reverse)]);
        let split = nested(inner, [operator(Split)]);
        assert_eq!(split.describe(), "Pipeline | Split");
        let query = Env::current();
        let expected = batches(&mut flat.start(query, None).unwrap());
        assert_eq!(batches(&mut split.start(query, None).unwrap()), expected);
    }

    #[test]
    fn a_pipeline_as_a_source_runs_on_its_callers_context() {
        let live = Rc::new(());
        let position = Position { live: live.clone() };
        let steps = [transform(position), transform(Interns { string: b"x", number: 0 })];
        let inner = pipeline(Numbers { batches: 1 }, steps);
        // The outer run's pool already has the inner run's string.
        let outer = nested(inner, [transform(Interns { string: b"y", number: 1 })]);
        let query = Env::current();
        let mut execution = outer.start(query, None).unwrap();
        assert_eq!(batches(&mut execution).len(), 1);
        // Ours, the inner pipeline's step, and its run's state, dropped
        // with the outer run.
        assert_eq!(Rc::strong_count(&live), 3);
        drop(execution);
        assert_eq!(Rc::strong_count(&live), 2);

        // A failure within it ends the run it's in.
        let failing = pipeline(Numbers { batches: 3 }, [transform(FailSecond)]);
        let outer = nested(failing, [operator(Sum)]);
        let mut execution = outer.start(query, None).unwrap();
        assert_eq!(execution.next(&mut RowBatch::new()), Err(Error::Corrupt));
    }

    #[test]
    fn builds_a_sink_once_when_first_read() {
        let (env, builds) = (Env::current(), Cell::new(0));
        let built = pipeline(Numbers { batches: 2 }, []);
        let mut sinks = Sinks::new(env).unwrap();
        let total = sinks.add(env, built, Total { builds: &builds }).unwrap();
        let reads = pipeline(Numbers { batches: 2 }, [operator(WithTotal { total })]);
        let mut execution = reads.start_with_sinks(env, None, &sinks).unwrap();
        // 0 + 1 + 2 + 3, built once for both batches.
        let expected = [[[0, 0, 6], [1, -1, 6]], [[2, -2, 6], [3, -3, 6]]];
        assert_eq!(batches(&mut execution), expected);
        assert_eq!(builds.get(), 1);
        // A step with no input never reads it, so it isn't built.
        let empty = pipeline(Numbers { batches: 0 }, [operator(WithTotal { total })]);
        let mut execution = empty.start_with_sinks(env, None, &sinks).unwrap();
        assert!(batches(&mut execution).is_empty());
        assert_eq!(builds.get(), 1);
    }

    #[test]
    fn operators_dont_finish_after_a_failure_before_them() {
        let failing = pipeline(Numbers { batches: 3 }, [transform(FailSecond), operator(Sum)]);
        let query = Env::current();
        let mut execution = failing.start(query, None).unwrap();
        let mut batch = RowBatch::new();
        // A sum of the first batch alone would be wrong.
        assert_eq!(execution.next(&mut batch), Err(Error::Corrupt));
    }
}
