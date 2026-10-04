//! `Pipeline`: a source and the steps its batches go through. `Execution`: one
//! run of it.
//!
//! Transforms run in place on the batch from the source or operator before
//! them. An operator's input is the batches out of the steps before it.

use core::alloc::Layout;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, Buffer};
use crate::row_batch::RowBatch;
use crate::step::{DynSource, Progress, Step};

/// Read-only, so it can be run any number of times.
pub struct Pipeline<'a> {
    source: DynSource<'a>,
    steps: &'a [Step<'a>],
}

impl<'a> Pipeline<'a> {
    pub fn new(source: DynSource<'a>, steps: &'a [Step<'a>]) -> Pipeline<'a> {
        Pipeline { source, steps }
    }

    /// Creates the state of a run, in one allocation from `allocator`.
    pub fn start<A: Allocator + Clone + 'static>(
        &self,
        allocator: A,
    ) -> Result<Execution<'_>, AllocError> {
        let size_bytes = self.layout(|_, _| {})?.size();
        // SAFETY: `Execution::new` writes every byte it reads.
        let memory = unsafe { Buffer::allocate_uninit(allocator, size_bytes)? };
        Ok(Execution::new(self, memory))
    }

    /// A `Slot` per step, then the source's state, then each step's. Calls
    /// `state` with each state's step, or `None` for the source, and its
    /// offset.
    fn layout(&self, mut state: impl FnMut(Option<usize>, usize)) -> Result<Layout, AllocError> {
        let slots = Layout::array::<Slot>(self.steps.len()).map_err(|_| AllocError)?;
        let (mut layout, offset) =
            slots.extend(self.source.state_layout).map_err(|_| AllocError)?;
        state(None, offset);
        for (i, step) in self.steps.iter().enumerate() {
            let offset;
            (layout, offset) = layout.extend(step.state_layout()).map_err(|_| AllocError)?;
            state(Some(i), offset);
        }
        check!(layout.align() <= BUFFER_ALIGNMENT_BYTES);
        Ok(layout)
    }
}

/// What a run keeps for a step.
struct Slot {
    /// Where the step's state is in the run's memory.
    state: usize,
    /// For operators: their input, and where they are with it.
    input: RowBatch,
    status: Status,
    /// For operators: where their segment ends, and the operator heading the
    /// segment before, or `None` for the source.
    end: usize,
    before: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    /// Needs another input batch.
    Waiting,
    /// `input` holds a batch to execute.
    Ready,
    /// There's no more input; the operator is finishing.
    Ended,
    /// The operator has nothing left to output.
    Done,
}

/// What a segment did when run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Made {
    Batch,
    /// A batch with no rows, which is dropped.
    Nothing,
    /// Its operator needs a batch from the segment before.
    NeedsInput,
    End,
}

pub struct Execution<'p> {
    pipeline: &'p Pipeline<'p>,
    memory: NonNull<u8>,
    source_state: usize,
    /// Where the source's segment ends.
    source_end: usize,
    /// The operator heading the last segment, or `None` for the source.
    last: Option<usize>,
    // Frees `memory`.
    _buffer: Buffer,
}

impl<'p> Execution<'p> {
    fn new(pipeline: &'p Pipeline<'p>, mut buffer: Buffer) -> Execution<'p> {
        let Some(memory) = NonNull::new(buffer.as_mut_ptr::<u8>()) else {
            crate::check::check_failed(line!());
        };
        let mut source_state = 0;
        // SAFETY: the memory was allocated with this layout, so it has room
        // for each state at its offset, and for the slots at the start.
        let layout = pipeline.layout(|i, state| unsafe {
            let Some(i) = i else {
                source_state = state;
                pipeline.source.new_state(memory.add(state));
                return;
            };
            at!(pipeline.steps, i).new_state(memory.add(state));
            let input = RowBatch::new();
            let slot = Slot { state, input, status: Status::Waiting, end: 0, before: None };
            memory.cast::<Slot>().add(i).write(slot);
        });
        check!(layout.is_ok());
        let mut execution = Execution {
            pipeline,
            memory,
            source_state,
            source_end: 0,
            last: None,
            _buffer: buffer,
        };
        for (i, step) in pipeline.steps.iter().enumerate() {
            if let Step::Operator(_) = step {
                execution.set_end(execution.last, i);
                // SAFETY: the slots were written above.
                unsafe { (*execution.slot(i)).before = execution.last };
                execution.last = Some(i);
            }
        }
        execution.set_end(execution.last, pipeline.steps.len());
        execution
    }

    fn set_end(&mut self, head: Option<usize>, end: usize) {
        match head {
            None => self.source_end = end,
            // SAFETY: the slots were written by `Execution::new`.
            Some(op) => unsafe { (*self.slot(op)).end = end },
        }
    }

    /// Fills `output` with the next batch, or returns false when there are
    /// none left.
    ///
    /// Each segment is a head, the source or an operator, and the transforms
    /// after it. Starting from the last, it moves to the segment before when
    /// a segment's operator needs input, and to the one after when a segment
    /// makes a batch or ends.
    pub fn next(&mut self, output: &mut RowBatch) -> bool {
        let last = self.pipeline.steps.len();
        let mut head = self.last;
        loop {
            let end = match head {
                None => self.source_end,
                // SAFETY: the slots were written by `Execution::new`.
                Some(op) => unsafe { (*self.slot(op)).end },
            };
            match self.make(head, end, output) {
                // SAFETY: as above.
                Made::NeedsInput => head = head.and_then(|op| unsafe { (*self.slot(op)).before }),
                Made::Nothing => {}
                Made::Batch if end == last => return true,
                Made::End if end == last => return false,
                Made::Batch | Made::End => head = Some(end),
            }
        }
    }

    /// Runs the segment from `head` to `end` once. Its batch goes to the
    /// operator at `end`, or to `output` if `end` is past the last step.
    fn make(&self, head: Option<usize>, end: usize, output: &mut RowBatch) -> Made {
        let next = (end < self.pipeline.steps.len()).then(|| self.slot(end));
        // SAFETY: a different slot from `head`'s.
        let batch = next.map_or(output, |next| unsafe { &mut (*next).input });
        let made = match head {
            None => self.read_source(batch),
            Some(op) => self.execute(op, batch),
        };
        if made == Made::Batch {
            for i in head.map_or(0, |op| op + 1)..end {
                let Step::Transform(transform) = at!(self.pipeline.steps, i) else {
                    crate::check::check_failed(line!());
                };
                // SAFETY: the transform's state was made by `Execution::new`.
                unsafe { transform.process(batch, self.state(Some(i))) };
            }
        }
        let status = match made {
            Made::Batch if batch.row_count() == 0 => return Made::Nothing,
            Made::Batch => Status::Ready,
            Made::End => Status::Ended,
            Made::NeedsInput | Made::Nothing => return made,
        };
        if let Some(next) = next {
            // SAFETY: as above.
            unsafe { (*next).status = status };
        }
        made
    }

    fn read_source(&self, batch: &mut RowBatch) -> Made {
        batch.reset(0);
        // SAFETY: the source's state was made by `Execution::new`.
        if unsafe { self.pipeline.source.next(batch, self.state(None)) } {
            Made::Batch
        } else {
            Made::End
        }
    }

    /// Runs operator `op` once.
    fn execute(&self, op: usize, output: &mut RowBatch) -> Made {
        let Step::Operator(operator) = at!(self.pipeline.steps, op) else {
            crate::check::check_failed(line!());
        };
        // SAFETY: `make` doesn't touch `op`'s slot otherwise.
        let slot = unsafe { &mut *self.slot(op) };
        let state = self.state(Some(op));
        // SAFETY: the operator's state was made by `Execution::new`.
        let progress = unsafe {
            match slot.status {
                Status::Waiting => return Made::NeedsInput,
                Status::Done => return Made::End,
                Status::Ready => {
                    output.reset(0);
                    operator.execute(&slot.input, output, state)
                }
                Status::Ended => {
                    output.reset(0);
                    operator.finish(output, state)
                }
            }
        };
        if progress == Progress::NeedInput {
            slot.status = if slot.status == Status::Ready { Status::Waiting } else { Status::Done };
        }
        Made::Batch
    }

    fn slot(&self, i: usize) -> *mut Slot {
        check!(i < self.pipeline.steps.len());
        // SAFETY: the slots start the run's memory.
        unsafe { self.memory.cast::<Slot>().as_ptr().add(i) }
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

impl Drop for Execution<'_> {
    fn drop(&mut self) {
        let pipeline = self.pipeline;
        // SAFETY: each was made by `Execution::new`, and is dropped once.
        unsafe {
            (pipeline.source.drop_state)(self.state(None));
            for (i, step) in pipeline.steps.iter().enumerate() {
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

        fn new_state(&self) -> i64 {
            0
        }

        fn next(&self, batch: &mut RowBatch, i: &mut i64) -> bool {
            if *i == self.batches {
                return false;
            }
            batch.reset(2);
            let first = *i * 2;
            assert!(batch.push_column(int64s(&[first, first + 1])).is_ok());
            assert!(batch.push_column(int64s(&[-first, -first - 1])).is_ok());
            *i += 1;
            true
        }
    }

    struct Reverse;

    impl Transform for Reverse {
        type State = ();

        fn new_state(&self) {}

        fn process(&self, batch: &mut RowBatch, (): &mut ()) {
            batch.columns_mut().reverse();
        }
    }

    /// Adds the batch's position in the run as a column. The state holds
    /// `live`, to check states are dropped.
    struct Position {
        live: Rc<()>,
    }

    impl Transform for Position {
        type State = (i64, Rc<()>);

        fn new_state(&self) -> (i64, Rc<()>) {
            (0, self.live.clone())
        }

        fn process(&self, batch: &mut RowBatch, (position, _): &mut (i64, Rc<()>)) {
            let column = int64s(&alloc::vec![*position; batch.row_count() as usize]);
            assert!(batch.push_column(column).is_ok());
            *position += 1;
        }
    }

    /// Empties every other batch.
    struct SkipOdd;

    impl Transform for SkipOdd {
        type State = bool;

        fn new_state(&self) -> bool {
            false
        }

        fn process(&self, batch: &mut RowBatch, odd: &mut bool) {
            if *odd {
                batch.reset(0);
            }
            *odd = !*odd;
        }
    }

    /// Outputs each input a row at a time.
    struct Split;

    impl Operator for Split {
        type State = u32;

        fn new_state(&self) -> u32 {
            0
        }

        fn execute(&self, input: &RowBatch, output: &mut RowBatch, row: &mut u32) -> Progress {
            output.reset(1);
            for column in 0..input.column_count() {
                assert!(output.push_column(input.column(column).slice(*row, 1)).is_ok());
            }
            *row += 1;
            if *row < input.row_count() {
                return Progress::MoreOutput;
            }
            *row = 0;
            Progress::NeedInput
        }
    }

    /// Outputs one row, the sum of its input's first column, once the input
    /// ends.
    struct Sum;

    impl Operator for Sum {
        type State = i64;

        fn new_state(&self) -> i64 {
            0
        }

        fn execute(&self, input: &RowBatch, _: &mut RowBatch, sum: &mut i64) -> Progress {
            *sum += input.column(0).int64s().iter().sum::<i64>();
            Progress::NeedInput
        }

        fn finish(&self, output: &mut RowBatch, sum: &mut i64) -> Progress {
            output.reset(1);
            assert!(output.push_column(int64s(&[*sum])).is_ok());
            Progress::NeedInput
        }
    }

    /// Each batch's rows, as lists of values.
    fn batches(execution: &mut Execution) -> Vec<Vec<Vec<i64>>> {
        let mut batch = RowBatch::new();
        let mut batches = Vec::new();
        while execution.next(&mut batch) {
            let rows = (0..batch.row_count() as usize).map(|row| {
                (0..batch.column_count()).map(|i| batch.column(i).int64s()[row]).collect()
            });
            batches.push(rows.collect());
        }
        batches
    }

    #[test]
    fn runs_a_source_alone() {
        let source = Numbers { batches: 2 };
        let pipeline = Pipeline::new(DynSource::new(&source), &[]);
        let expected = [[[0, 0], [1, -1]], [[2, -2], [3, -3]]];
        assert_eq!(batches(&mut pipeline.start(Heap).unwrap()), expected);
    }

    #[test]
    fn transforms_each_batch_in_order() {
        let source = Numbers { batches: 2 };
        let position = Position { live: Rc::new(()) };
        let steps = [
            Step::Transform(DynTransform::new(&Reverse)),
            Step::Transform(DynTransform::new(&position)),
        ];
        let pipeline = Pipeline::new(DynSource::new(&source), &steps);
        let expected = [[[0, 0, 0], [-1, 1, 0]], [[-2, 2, 1], [-3, 3, 1]]];
        assert_eq!(batches(&mut pipeline.start(Heap).unwrap()), expected);
    }

    #[test]
    fn operators_output_more_than_once_for_an_input() {
        let source = Numbers { batches: 2 };
        let position = Position { live: Rc::new(()) };
        let steps = [
            Step::Operator(DynOperator::new(&Split)),
            Step::Transform(DynTransform::new(&position)),
        ];
        let pipeline = Pipeline::new(DynSource::new(&source), &steps);
        let expected = [[[0, 0, 0]], [[1, -1, 1]], [[2, -2, 2]], [[3, -3, 3]]];
        assert_eq!(batches(&mut pipeline.start(Heap).unwrap()), expected);
    }

    #[test]
    fn operators_finish_after_their_input_ends() {
        let source = Numbers { batches: 3 };
        let steps = [
            Step::Operator(DynOperator::new(&Split)),
            Step::Transform(DynTransform::new(&SkipOdd)),
            Step::Operator(DynOperator::new(&Sum)),
        ];
        let pipeline = Pipeline::new(DynSource::new(&source), &steps);
        assert_eq!(batches(&mut pipeline.start(Heap).unwrap()), [[[2 + 4]]]);
    }

    #[test]
    fn each_run_has_its_own_state() {
        let source = Numbers { batches: 2 };
        let position = Position { live: Rc::new(()) };
        let steps = [
            Step::Operator(DynOperator::new(&Split)),
            Step::Transform(DynTransform::new(&position)),
        ];
        let pipeline = Pipeline::new(DynSource::new(&source), &steps);

        let mut first = pipeline.start(Heap).unwrap();
        let mut second = pipeline.start(Heap).unwrap();
        assert_eq!(batches(&mut first), batches(&mut second));
        assert_eq!(Rc::strong_count(&position.live), 3);
        drop((first, second));
        assert_eq!(Rc::strong_count(&position.live), 1);
    }
}
