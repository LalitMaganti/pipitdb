//! `Pipeline`: a source and the steps its batches go through. `Execution`: one
//! run of it.
//!
//! The steps are cut into segments at each operator. The first segment is the
//! source and the transforms after it; each other is an operator and the
//! transforms after it. A segment's transforms run in place on the batches its
//! head makes, and each segment's batches are the next operator's input.

use core::alloc::Layout;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator, DynAllocator};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, Buffer};
use crate::context::Context;
use crate::row_batch::RowBatch;
use crate::step::{DynSource, Progress, Step};
use crate::vec::Vec;

/// Read-only, so it can be run any number of times. Owns its source and steps.
pub struct Pipeline<'a> {
    source: DynSource<'a>,
    steps: Vec<Step<'a>>,
    segment_count: usize,
}

impl<'a> Pipeline<'a> {
    pub fn new(source: DynSource<'a>, steps: Vec<Step<'a>>) -> Pipeline<'a> {
        let operators = steps.iter().filter(|step| matches!(step, Step::Operator(_))).count();
        Pipeline { source, steps, segment_count: operators + 1 }
    }

    /// Creates the state of a run, in one allocation from `allocator`, which
    /// steps' states also allocate from.
    pub fn start<A: Allocator + Clone + 'static>(
        &self,
        allocator: A,
    ) -> Result<Execution<'_>, AllocError> {
        let size_bytes = self.layout(|_, _| {})?.0.size();
        // SAFETY: `Execution::new` writes every byte it reads.
        let memory = unsafe { Buffer::allocate_uninit(allocator, size_bytes)? };
        Execution::new(self, memory)
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
            layout.extend(self.source.state_layout).map_err(|_| AllocError)?;
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

/// What a run keeps for a step.
struct Slot {
    /// Where the step's state is in the run's memory.
    state: usize,
    /// For operators: their input.
    input: RowBatch,
}

/// What a segment did when run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Made {
    Batch,
    /// A batch with no rows, which is dropped.
    Nothing,
    End,
}

pub struct Execution<'p> {
    run: Run<'p>,
    context: Context,
    /// Whether a step has failed, which ends the run.
    failed: bool,
}

/// A run's memory, and where it is in it.
struct Run<'p> {
    pipeline: &'p Pipeline<'p>,
    memory: NonNull<u8>,
    slots: usize,
    source_state: usize,
    /// How many states are made: the source's, then each step's, in order.
    made: usize,
    // Frees `memory`.
    _buffer: Buffer,
}

impl<'p> Execution<'p> {
    /// Makes every state, or, if one fails, drops those made.
    fn new(pipeline: &'p Pipeline<'p>, mut buffer: Buffer) -> Result<Execution<'p>, AllocError> {
        let Some(memory) = NonNull::new(buffer.as_mut_ptr::<u8>()) else {
            crate::check::check_failed(line!());
        };
        let mut context = Context::new(DynAllocator::of(&buffer));
        let Ok((_, slots)) = pipeline.layout(|_, _| {}) else {
            crate::check::check_failed(line!());
        };
        let mut run = Run { pipeline, memory, slots, source_state: 0, made: 0, _buffer: buffer };
        let mut failed = false;
        // SAFETY: the memory was allocated with this layout, so it has room
        // for each state at its offset, and for the slots.
        let layout = pipeline.layout(|i, state| unsafe {
            if failed {
                return;
            }
            let made = match i {
                None => pipeline.source.new_state(&mut context, memory.add(state)),
                Some(i) => at!(pipeline.steps, i).new_state(&mut context, memory.add(state)),
            };
            if made.is_err() {
                failed = true;
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
        if failed {
            // Dropping it drops the states made.
            return Err(AllocError);
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
        Ok(Execution { run, context, failed: false })
    }

    /// Fills `output` with the next batch, or returns false when there are
    /// none left. Fails if a step can't allocate what it needs; the run
    /// can't go on after that, so later calls fail too.
    pub fn next(&mut self, output: &mut RowBatch) -> Result<bool, AllocError> {
        if self.failed {
            return Err(AllocError);
        }
        let next = self.run.next(&mut self.context, output);
        self.failed = next.is_err();
        next
    }
}

impl Run<'_> {
    /// As `Execution::next`.
    ///
    /// Starting from the last segment, it moves to the one before when a
    /// segment's operator needs input, and to the one after when a segment
    /// makes a batch or ends.
    fn next(&mut self, context: &mut Context, output: &mut RowBatch) -> Result<bool, AllocError> {
        let last = self.pipeline.segment_count - 1;
        let mut k = self.ready(last);
        loop {
            match self.make(context, k, output)? {
                Made::Nothing => k = self.ready(k),
                Made::Batch if k == last => return Ok(true),
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
        // SAFETY: the segments were written by `Execution::new`.
        unsafe { (*self.segment(k)).status == Status::Waiting }
    }

    /// Runs segment `k` once. Its batch goes to the next operator's input, or
    /// to `output` if it is the last segment.
    fn make(
        &self,
        context: &mut Context,
        k: usize,
        output: &mut RowBatch,
    ) -> Result<Made, AllocError> {
        // SAFETY: the segments were written by `Execution::new`.
        let segment = unsafe { &mut *self.segment(k) };
        let end = segment.end;
        let batch = if end == self.pipeline.steps.len() {
            output
        } else {
            // SAFETY: the next operator's slot, which nothing else borrows.
            unsafe { &mut (*self.slot(end)).input }
        };
        let (made, first) = if k == 0 {
            (self.read_source(context, batch)?, 0)
        } else {
            (self.execute(context, segment, batch)?, segment.head + 1)
        };
        if made == Made::Batch {
            for i in first..end {
                let Step::Transform(transform) = at!(self.pipeline.steps, i) else {
                    crate::check::check_failed(line!());
                };
                // SAFETY: the transform's state was made by `Execution::new`.
                unsafe { transform.process(context, self.state(Some(i)), batch)? };
            }
        }
        Ok(if made == Made::Batch && batch.selection().is_empty() { Made::Nothing } else { made })
    }

    fn read_source(&self, context: &mut Context, batch: &mut RowBatch) -> Result<Made, AllocError> {
        batch.reset(0);
        // SAFETY: the source's state was made by `Execution::new`.
        let more = unsafe { self.pipeline.source.next(context, self.state(None), batch)? };
        Ok(if more { Made::Batch } else { Made::End })
    }

    /// Runs the operator heading `segment` once.
    fn execute(
        &self,
        context: &mut Context,
        segment: &mut Segment,
        output: &mut RowBatch,
    ) -> Result<Made, AllocError> {
        let op = segment.head;
        let Step::Operator(operator) = at!(self.pipeline.steps, op) else {
            crate::check::check_failed(line!());
        };
        // SAFETY: the operator's slot, which nothing else borrows.
        let input = unsafe { &(*self.slot(op)).input };
        let state = self.state(Some(op));
        // SAFETY: the operator's state was made by `Execution::new`.
        let progress = unsafe {
            match segment.status {
                Status::Done => return Ok(Made::End),
                Status::Ready => {
                    output.reset(0);
                    operator.execute(context, state, input, output)
                }
                Status::Ended => {
                    output.reset(0);
                    operator.finish(context, state, output)
                }
                Status::Waiting => crate::check::check_failed(line!()),
            }
        }?;
        if progress == Progress::NeedInput {
            segment.status =
                if segment.status == Status::Ready { Status::Waiting } else { Status::Done };
        }
        Ok(Made::Batch)
    }

    fn segment(&self, k: usize) -> *mut Segment {
        check!(k < self.pipeline.segment_count);
        // SAFETY: the segments start the run's memory.
        unsafe { self.memory.cast::<Segment>().as_ptr().add(k) }
    }

    fn slot(&self, i: usize) -> *mut Slot {
        check!(i < self.pipeline.steps.len());
        // SAFETY: the slots are at `self.slots`.
        unsafe { self.memory.add(self.slots).cast::<Slot>().as_ptr().add(i) }
    }

    /// The state of step `i`, or of the source.
    fn state(&self, i: Option<usize>) -> NonNull<u8> {
        let offset = match i {
            None => self.source_state,
            // SAFETY: each slot was written by `Execution::new`.
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
        // SAFETY: each was made by `Execution::new`, and is dropped once.
        unsafe {
            (pipeline.source.drop_state)(self.state(None));
            for (i, step) in pipeline.steps.iter().enumerate().take(self.made - 1) {
                step.drop_state(self.state(Some(i)));
                self.slot(i).drop_in_place();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;
    use crate::column::{ColumnView, DataType};
    use crate::selection::Kept;
    use crate::step::{DynOperator, DynTransform, Operator, Source, Transform};

    fn int64s(values: &[i64]) -> ColumnView {
        let mut buffer = Buffer::allocate(Heap, values.len() * 8).unwrap();
        buffer.as_mut_slice::<i64>().copy_from_slice(values);
        ColumnView::new(DataType::Int64, buffer, None)
    }

    /// `batches` batches of two rows, with two columns, `[2i, 2i + 1]` and
    /// their negations.
    struct Numbers {
        batches: i64,
    }

    impl Source for Numbers {
        type State = i64;

        fn new_state(&self, _: &mut Context) -> Result<i64, AllocError> {
            Ok(0)
        }

        fn next(
            &self,
            _: &mut Context,
            i: &mut i64,
            batch: &mut RowBatch,
        ) -> Result<bool, AllocError> {
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

        fn new_state(&self, _: &mut Context) -> Result<(), AllocError> {
            Ok(())
        }

        fn process(
            &self,
            _: &mut Context,
            (): &mut (),
            batch: &mut RowBatch,
        ) -> Result<(), AllocError> {
            batch.columns_mut().reverse();
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

        fn new_state(&self, _: &mut Context) -> Result<(i64, Rc<()>), AllocError> {
            Ok((0, self.live.clone()))
        }

        fn process(
            &self,
            _: &mut Context,
            (position, _): &mut (i64, Rc<()>),
            batch: &mut RowBatch,
        ) -> Result<(), AllocError> {
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

        fn new_state(&self, _: &mut Context) -> Result<bool, AllocError> {
            Ok(false)
        }

        fn process(
            &self,
            _: &mut Context,
            odd: &mut bool,
            batch: &mut RowBatch,
        ) -> Result<(), AllocError> {
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

        fn new_state(&self, _: &mut Context) -> Result<(), AllocError> {
            Ok(())
        }

        fn process(
            &self,
            _: &mut Context,
            (): &mut (),
            batch: &mut RowBatch,
        ) -> Result<(), AllocError> {
            let column = batch.column(0).clone();
            batch.selection_mut().retain(|row| column.int64s()[row as usize] % 2 == 0);
            Ok(())
        }
    }

    /// Keeps no rows.
    struct KeepNone;

    impl Transform for KeepNone {
        type State = ();

        fn new_state(&self, _: &mut Context) -> Result<(), AllocError> {
            Ok(())
        }

        fn process(
            &self,
            _: &mut Context,
            (): &mut (),
            batch: &mut RowBatch,
        ) -> Result<(), AllocError> {
            batch.selection_mut().retain(|_| false);
            Ok(())
        }
    }

    /// Outputs each input a row at a time.
    struct Split;

    impl Operator for Split {
        type State = u32;

        fn new_state(&self, _: &mut Context) -> Result<u32, AllocError> {
            Ok(0)
        }

        fn execute(
            &self,
            _: &mut Context,
            row: &mut u32,
            input: &RowBatch,
            output: &mut RowBatch,
        ) -> Result<Progress, AllocError> {
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
    }

    /// Outputs one row, the sum of its input's first column, once the input
    /// ends.
    struct Sum;

    impl Operator for Sum {
        type State = i64;

        fn new_state(&self, _: &mut Context) -> Result<i64, AllocError> {
            Ok(0)
        }

        fn execute(
            &self,
            _: &mut Context,
            sum: &mut i64,
            input: &RowBatch,
            _: &mut RowBatch,
        ) -> Result<Progress, AllocError> {
            *sum += input.column(0).int64s().iter().sum::<i64>();
            Ok(Progress::NeedInput)
        }

        fn finish(
            &self,
            _: &mut Context,
            sum: &mut i64,
            output: &mut RowBatch,
        ) -> Result<Progress, AllocError> {
            output.reset(1);
            assert!(output.push_column(int64s(&[*sum])).is_ok());
            Ok(Progress::NeedInput)
        }
    }

    /// Allocates its state, a number, from the run's allocator, or fails to
    /// if `fail`.
    struct Boxed {
        fail: bool,
    }

    impl Transform for Boxed {
        type State = crate::boxed::Box<i64>;

        fn new_state(&self, context: &mut Context) -> Result<Self::State, AllocError> {
            if self.fail {
                return Err(AllocError);
            }
            crate::boxed::Box::new(context.allocator().clone(), 7)
        }

        fn process(
            &self,
            _: &mut Context,
            number: &mut Self::State,
            batch: &mut RowBatch,
        ) -> Result<(), AllocError> {
            let column = int64s(&alloc::vec![**number; batch.row_count() as usize]);
            assert!(batch.push_column(column).is_ok());
            Ok(())
        }
    }

    /// Fails on its second batch, as a step that can't allocate would.
    struct FailSecond;

    impl Transform for FailSecond {
        type State = bool;

        fn new_state(&self, _: &mut Context) -> Result<bool, AllocError> {
            Ok(false)
        }

        fn process(
            &self,
            _: &mut Context,
            seen: &mut bool,
            _: &mut RowBatch,
        ) -> Result<(), AllocError> {
            if *seen {
                return Err(AllocError);
            }
            *seen = true;
            Ok(())
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
        Step::Transform(DynTransform::new(Heap, transform).unwrap())
    }

    fn operator<'a>(operator: impl Operator + 'a) -> Step<'a> {
        Step::Operator(DynOperator::new(Heap, operator).unwrap())
    }

    fn pipeline<'a>(
        source: impl Source + 'a,
        steps: impl IntoIterator<Item = Step<'a>>,
    ) -> Pipeline<'a> {
        let mut owned = crate::vec::Vec::new(Heap, 8).unwrap();
        for step in steps {
            assert!(owned.push(step).is_ok());
        }
        Pipeline::new(DynSource::new(Heap, source).unwrap(), owned)
    }

    #[test]
    fn runs_a_source_alone() {
        let pipeline = pipeline(Numbers { batches: 2 }, []);
        let expected = [[[0, 0], [1, -1]], [[2, -2], [3, -3]]];
        assert_eq!(batches(&mut pipeline.start(Heap).unwrap()), expected);
    }

    #[test]
    fn transforms_each_batch_in_order() {
        let position = Position { live: Rc::new(()) };
        let pipeline = pipeline(Numbers { batches: 2 }, [transform(Reverse), transform(position)]);
        let expected = [[[0, 0, 0], [-1, 1, 0]], [[-2, 2, 1], [-3, 3, 1]]];
        assert_eq!(batches(&mut pipeline.start(Heap).unwrap()), expected);
    }

    #[test]
    fn operators_output_more_than_once_for_an_input() {
        let position = Position { live: Rc::new(()) };
        let pipeline = pipeline(Numbers { batches: 2 }, [operator(Split), transform(position)]);
        let expected = [[[0, 0, 0]], [[1, -1, 1]], [[2, -2, 2]], [[3, -3, 3]]];
        assert_eq!(batches(&mut pipeline.start(Heap).unwrap()), expected);
    }

    #[test]
    fn operators_finish_after_their_input_ends() {
        let steps = [operator(Split), transform(SkipOdd), operator(Sum)];
        let pipeline = pipeline(Numbers { batches: 3 }, steps);
        assert_eq!(batches(&mut pipeline.start(Heap).unwrap()), [[[2 + 4]]]);
    }

    #[test]
    fn each_run_has_its_own_state() {
        let live = Rc::new(());
        let position = Position { live: live.clone() };
        let pipeline = pipeline(Numbers { batches: 2 }, [operator(Split), transform(position)]);

        let mut first = pipeline.start(Heap).unwrap();
        let mut second = pipeline.start(Heap).unwrap();
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
        assert_eq!(batches(&mut boxed.start(Heap).unwrap()), [[[0, 0, 7], [1, -1, 7]]]);

        // The states made before the one that fails are dropped.
        let live = Rc::new(());
        let position = Position { live: live.clone() };
        let steps = [transform(position), transform(Boxed { fail: true })];
        let failing = pipeline(Numbers { batches: 1 }, steps);
        assert!(failing.start(Heap).is_err());
        assert_eq!(Rc::strong_count(&live), 2);
    }

    #[test]
    fn passes_selections_on_and_skips_batches_with_none() {
        let even = pipeline(Numbers { batches: 3 }, [transform(KeepEven), transform(KeepEven)]);
        let mut execution = even.start(Heap).unwrap();
        let mut batch = RowBatch::new();
        let mut kept = alloc::vec::Vec::new();
        while execution.next(&mut batch).unwrap() {
            let Kept::Select(rows) = batch.selection().kept() else { panic!("not narrowed") };
            kept.extend(rows.iter().map(|&row| batch.column(0).int64s()[row as usize]));
        }
        // Each batch has 2i and 2i + 1, so one row of each is kept.
        assert_eq!(kept, [0, 2, 4]);

        let none = pipeline(Numbers { batches: 3 }, [transform(KeepNone)]);
        assert!(!none.start(Heap).unwrap().next(&mut batch).unwrap());
    }

    #[test]
    fn a_failing_step_ends_the_run() {
        let failing = pipeline(Numbers { batches: 3 }, [transform(FailSecond)]);
        let mut execution = failing.start(Heap).unwrap();
        let mut batch = RowBatch::new();
        assert_eq!(execution.next(&mut batch), Ok(true));
        assert_eq!(execution.next(&mut batch), Err(AllocError));
        // Rows aren't skipped: the run stays failed.
        assert_eq!(execution.next(&mut batch), Err(AllocError));
    }
}
