//! `Pipeline`: a source and the steps its batches go through. `Execution`: one
//! run of it.
//!
//! The steps are cut into segments at each operator. The first segment is the
//! source and the transforms after it; each other is an operator and the
//! transforms after it. A segment's transforms run in place on the batches it
//! makes, and each segment's batches are the input of the next segment's
//! operator.

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
    operator_count: usize,
}

impl<'a> Pipeline<'a> {
    pub fn new(source: DynSource<'a>, steps: &'a [Step<'a>]) -> Pipeline<'a> {
        let operator_count = steps.iter().filter(|step| matches!(step, Step::Operator(_))).count();
        Pipeline { source, steps, operator_count }
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

    /// A `Boundary` per operator, the offset of each step's state, then the
    /// source's state, then each step's. Calls `state` with each state's step,
    /// or `None` for the source, and its offset.
    fn layout(&self, mut state: impl FnMut(Option<usize>, usize)) -> Result<Layout, AllocError> {
        let boundaries = Layout::array::<Boundary>(self.operator_count).map_err(|_| AllocError)?;
        let offsets = Layout::array::<usize>(self.steps.len()).map_err(|_| AllocError)?;
        let (layout, _) = boundaries.extend(offsets).map_err(|_| AllocError)?;
        let (mut layout, offset) =
            layout.extend(self.source.state_layout).map_err(|_| AllocError)?;
        state(None, offset);
        for (i, step) in self.steps.iter().enumerate() {
            let step_layout = match step {
                Step::Transform(transform) => transform.state_layout,
                Step::Operator(operator) => operator.state_layout,
            };
            let offset;
            (layout, offset) = layout.extend(step_layout).map_err(|_| AllocError)?;
            state(Some(i), offset);
        }
        check!(layout.align() <= BUFFER_ALIGNMENT_BYTES);
        Ok(layout)
    }
}

/// An operator's input, and where the operator is with it.
struct Boundary {
    input: RowBatch,
    status: Status,
    /// The operator's index in the steps.
    step: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    /// Needs a batch from the segment before.
    Waiting,
    /// `input` holds a batch to execute.
    Ready,
    /// There's no more input; the operator is finishing.
    Ended,
    /// The operator has nothing left to output.
    Done,
}

/// What a segment did when asked for a batch.
enum Made {
    Batch,
    Nothing,
    End,
}

pub struct Execution<'p> {
    pipeline: &'p Pipeline<'p>,
    memory: NonNull<u8>,
    source_state: usize,
    // Frees `memory`.
    _buffer: Buffer,
}

impl<'p> Execution<'p> {
    fn new(pipeline: &'p Pipeline<'p>, mut buffer: Buffer) -> Execution<'p> {
        let Some(memory) = NonNull::new(buffer.as_mut_ptr::<u8>()) else {
            crate::check::check_failed(line!());
        };
        let mut execution = Execution { pipeline, memory, source_state: 0, _buffer: buffer };
        let mut operators = 0;
        // SAFETY: the memory was allocated with this layout, so it has room
        // for each state at its offset, and for the boundaries and offsets at
        // the start.
        let layout = pipeline.layout(|i, state| unsafe {
            let Some(i) = i else {
                execution.source_state = state;
                pipeline.source.new_state(memory.add(state));
                return;
            };
            execution.offsets().add(i).write(state);
            match at!(pipeline.steps, i) {
                Step::Transform(transform) => transform.new_state(memory.add(state)),
                Step::Operator(operator) => {
                    operator.new_state(memory.add(state));
                    let input = RowBatch::new();
                    let boundary = Boundary { input, status: Status::Waiting, step: i };
                    memory.cast::<Boundary>().add(operators).write(boundary);
                    operators += 1;
                }
            }
        });
        check!(layout.is_ok());
        execution
    }

    /// Fills `output` with the next batch, or returns false when there are
    /// none left.
    pub fn next(&mut self, output: &mut RowBatch) -> bool {
        let last = self.pipeline.operator_count;
        let mut segment = last;
        loop {
            // Segment `segment` makes batches into the next operator's input,
            // or into `output` if it is the last.
            let made = if segment == 0 {
                self.make_from_source(self.batch(segment, output))
            } else {
                // SAFETY: a different boundary from the one `batch` returns.
                let boundary = unsafe { &mut *self.boundary(segment - 1) };
                if boundary.status == Status::Waiting {
                    segment -= 1;
                    continue;
                }
                self.make_from_operator(boundary, self.batch(segment, output))
            };
            let next_status = match made {
                Made::Nothing => continue,
                Made::Batch if segment == last => return true,
                Made::End if segment == last => return false,
                Made::Batch => Status::Ready,
                Made::End => Status::Ended,
            };
            // SAFETY: `segment` is below `last`.
            unsafe { (*self.boundary(segment)).status = next_status };
            segment += 1;
        }
    }

    fn make_from_source(&self, batch: &mut RowBatch) -> Made {
        batch.reset(0);
        // SAFETY: the source's state was made by `Execution::new`.
        if !unsafe { self.pipeline.source.next(batch, self.state(None)) } {
            return Made::End;
        }
        let end = self.segment_end(0);
        self.transform(batch, 0, end)
    }

    fn make_from_operator(&self, boundary: &mut Boundary, batch: &mut RowBatch) -> Made {
        let Step::Operator(operator) = at!(self.pipeline.steps, boundary.step) else {
            crate::check::check_failed(line!());
        };
        let state = self.state(Some(boundary.step));
        batch.reset(0);
        // SAFETY: the operator's state was made by `Execution::new`.
        let progress = unsafe {
            match boundary.status {
                Status::Ready => operator.execute(&boundary.input, batch, state),
                Status::Ended => operator.finish(batch, state),
                Status::Waiting | Status::Done => return Made::End,
            }
        };
        if progress == Progress::NeedInput {
            boundary.status = match boundary.status {
                Status::Ready => Status::Waiting,
                _ => Status::Done,
            };
        }
        let end = self.segment_end(boundary.step + 1);
        self.transform(batch, boundary.step + 1, end)
    }

    /// Runs the transforms in `steps[start..end]` on `batch`.
    fn transform(&self, batch: &mut RowBatch, start: usize, end: usize) -> Made {
        for i in start..end {
            let Step::Transform(transform) = at!(self.pipeline.steps, i) else {
                crate::check::check_failed(line!());
            };
            // SAFETY: the transform's state was made by `Execution::new`.
            unsafe { transform.process(batch, self.state(Some(i))) };
        }
        if batch.row_count() > 0 { Made::Batch } else { Made::Nothing }
    }

    /// Where the transforms starting at step `start` end: at the next
    /// operator, or at the end of the steps.
    fn segment_end(&self, start: usize) -> usize {
        let steps = at!(self.pipeline.steps, start..);
        let next = steps.iter().position(|step| matches!(step, Step::Operator(_)));
        start + next.unwrap_or(steps.len())
    }

    /// Where segment `segment` puts its batches.
    fn batch<'b>(&self, segment: usize, output: &'b mut RowBatch) -> &'b mut RowBatch {
        if segment == self.pipeline.operator_count {
            return output;
        }
        // SAFETY: each boundary's input is only borrowed by the segment
        // filling it, or by the operator reading it, never both at once.
        unsafe { &mut (*self.boundary(segment)).input }
    }

    fn boundary(&self, operator: usize) -> *mut Boundary {
        check!(operator < self.pipeline.operator_count);
        // SAFETY: the boundaries start the run's memory.
        unsafe { self.memory.cast::<Boundary>().as_ptr().add(operator) }
    }

    fn offsets(&self) -> NonNull<usize> {
        let boundaries = self.pipeline.operator_count * size_of::<Boundary>();
        // SAFETY: the offsets follow the boundaries.
        unsafe { self.memory.add(boundaries).cast() }
    }

    /// The state of step `i`, or of the source.
    fn state(&self, i: Option<usize>) -> NonNull<u8> {
        let offset = match i {
            None => self.source_state,
            Some(i) => {
                check!(i < self.pipeline.steps.len());
                // SAFETY: there is an offset for each step.
                unsafe { self.offsets().add(i).read() }
            }
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
                let drop_state = match step {
                    Step::Transform(transform) => transform.drop_state,
                    Step::Operator(operator) => operator.drop_state,
                };
                drop_state(self.state(Some(i)));
            }
            for operator in 0..pipeline.operator_count {
                self.boundary(operator).drop_in_place();
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
