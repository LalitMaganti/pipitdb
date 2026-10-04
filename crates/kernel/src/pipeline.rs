//! `Pipeline`: a source and the transforms each of its batches goes through.
//! `Execution`: one run of it.

use core::alloc::Layout;
use core::ptr::NonNull;

use crate::allocator::{AllocError, Allocator};
use crate::buffer::{BUFFER_ALIGNMENT_BYTES, Buffer};
use crate::row_batch::RowBatch;
use crate::step::{DynSource, DynTransform};

/// Read-only, so it can be run any number of times.
pub struct Pipeline<'a> {
    source: DynSource<'a>,
    transforms: &'a [DynTransform<'a>],
}

impl<'a> Pipeline<'a> {
    pub fn new(source: DynSource<'a>, transforms: &'a [DynTransform<'a>]) -> Pipeline<'a> {
        Pipeline { source, transforms }
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

    /// The offset of each transform's state, then the source's state, then
    /// each transform's. Calls `state` with each state's transform, or `None`
    /// for the source, and its offset.
    fn layout(&self, mut state: impl FnMut(Option<usize>, usize)) -> Result<Layout, AllocError> {
        let offsets = Layout::array::<usize>(self.transforms.len()).map_err(|_| AllocError)?;
        let (mut layout, offset) =
            offsets.extend(self.source.state_layout).map_err(|_| AllocError)?;
        state(None, offset);
        for (i, transform) in self.transforms.iter().enumerate() {
            let offset;
            (layout, offset) = layout.extend(transform.state_layout).map_err(|_| AllocError)?;
            state(Some(i), offset);
        }
        check!(layout.align() <= BUFFER_ALIGNMENT_BYTES);
        Ok(layout)
    }
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
        let mut source_state = 0;
        // SAFETY: `buffer` was allocated with this layout, so it has room for
        // each state at its offset, and for the offsets at the start.
        let layout = pipeline.layout(|i, state| unsafe {
            match i {
                None => {
                    source_state = state;
                    pipeline.source.new_state(memory.add(state));
                }
                Some(i) => {
                    at!(pipeline.transforms, i).new_state(memory.add(state));
                    memory.cast::<usize>().add(i).write(state);
                }
            }
        });
        check!(layout.is_ok());
        Execution { pipeline, memory, source_state, _buffer: buffer }
    }

    /// Fills `batch` with the next batch, or returns false when there are
    /// none left.
    pub fn next(&mut self, batch: &mut RowBatch) -> bool {
        batch.reset(0);
        // SAFETY: each state was made by `Execution::new`.
        unsafe {
            if !self.pipeline.source.next(batch, self.state(None)) {
                return false;
            }
            for (i, transform) in self.pipeline.transforms.iter().enumerate() {
                transform.process(batch, self.state(Some(i)));
            }
        }
        true
    }

    /// The state of transform `i`, or of the source.
    fn state(&self, i: Option<usize>) -> NonNull<u8> {
        let offset = match i {
            None => self.source_state,
            Some(i) => {
                check!(i < self.pipeline.transforms.len());
                // SAFETY: the offsets start the run's memory.
                unsafe { self.memory.cast::<usize>().add(i).read() }
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
            for (i, transform) in pipeline.transforms.iter().enumerate() {
                (transform.drop_state)(self.state(Some(i)));
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
    use crate::step::{Source, Transform};

    fn int64s(values: &[i64]) -> ColumnView {
        let mut buffer = Buffer::allocate(Heap, values.len() * 8).unwrap();
        buffer.as_mut_slice::<i64>().copy_from_slice(values);
        ColumnView::new(DataType::Int64, buffer, None)
    }

    /// `batches` one-row batches with two columns, `[i, -i]`.
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
            batch.reset(1);
            assert!(batch.push_column(int64s(&[*i])).is_ok());
            assert!(batch.push_column(int64s(&[-*i])).is_ok());
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

    /// Adds the batch's position in the run as a third column. The state
    /// holds `live`, to check states are dropped.
    struct Position {
        live: Rc<()>,
    }

    impl Transform for Position {
        type State = (i64, Rc<()>);

        fn new_state(&self) -> (i64, Rc<()>) {
            (0, self.live.clone())
        }

        fn process(&self, batch: &mut RowBatch, (position, _): &mut (i64, Rc<()>)) {
            assert!(batch.push_column(int64s(&[*position])).is_ok());
            *position += 1;
        }
    }

    fn rows(execution: &mut Execution) -> Vec<Vec<i64>> {
        let mut batch = RowBatch::new();
        let mut rows = Vec::new();
        while execution.next(&mut batch) {
            rows.push((0..batch.column_count()).map(|i| batch.column(i).int64s()[0]).collect());
        }
        rows
    }

    #[test]
    fn runs_a_source_alone() {
        let source = Numbers { batches: 2 };
        let pipeline = Pipeline::new(DynSource::new(&source), &[]);
        assert_eq!(rows(&mut pipeline.start(Heap).unwrap()), [[0, 0], [1, -1]]);
    }

    #[test]
    fn transforms_each_batch_in_order() {
        let source = Numbers { batches: 2 };
        let position = Position { live: Rc::new(()) };
        let transforms = [DynTransform::new(&Reverse), DynTransform::new(&position)];
        let pipeline = Pipeline::new(DynSource::new(&source), &transforms);
        assert_eq!(rows(&mut pipeline.start(Heap).unwrap()), [[0, 0, 0], [-1, 1, 1]]);
    }

    #[test]
    fn each_run_has_its_own_state() {
        let source = Numbers { batches: 2 };
        let position = Position { live: Rc::new(()) };
        let transforms = [DynTransform::new(&position)];
        let pipeline = Pipeline::new(DynSource::new(&source), &transforms);

        let mut first = pipeline.start(Heap).unwrap();
        let mut second = pipeline.start(Heap).unwrap();
        assert_eq!(rows(&mut first), [[0, 0, 0], [1, -1, 1]]);
        assert_eq!(rows(&mut second), [[0, 0, 0], [1, -1, 1]]);
        assert_eq!(Rc::strong_count(&position.live), 3);
        drop((first, second));
        assert_eq!(Rc::strong_count(&position.live), 1);
    }
}
