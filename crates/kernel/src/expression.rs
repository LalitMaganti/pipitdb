//! `Compute`: a transform which adds a column computed from each row's
//! values, as Perfetto's.

use crate::allocator::AllocError;
use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType, Forms, Validity};
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::selection::Selection;
use crate::slow_vec::SlowVec;
use crate::step::Transform;

/// What a step of an `ExprProgram` computes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExprOp {
    /// Reads batch column `a`.
    Column,
    /// A constant: `int_value` or `double_value` by type.
    Constant,
    /// A null of its type.
    Null,
    /// Int64 to Float64.
    ToDouble,
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`.
    Mul,
    /// `a / b`.
    Div,
    /// `a % b`, of integers.
    Mod,
}

/// One step of an `ExprProgram`. Step `i` computes register `i` from the
/// batch or from earlier registers.
#[derive(Clone, Copy)]
pub struct ExprStep {
    /// What it computes.
    pub op: ExprOp,
    /// The type of the value it computes.
    pub data_type: DataType,
    /// The first register it reads, or for `Column` the batch column.
    pub a: u32,
    /// The second register it reads.
    pub b: u32,
    /// A `Constant`'s value, if it's an Int64.
    pub int_value: i64,
    /// A `Constant`'s value, if it's a Float64.
    pub double_value: f64,
}

/// A typed expression, flattened so that each step reads only steps before
/// it. The last step is the result.
///
/// Every step is typed when the plan is built, so evaluating one dispatches
/// on its op and type once per batch, then runs one loop over every row of
/// the batch. A row the batch no longer keeps costs no more to compute than
/// to skip, and computing it keeps the loop free of branches. Only the rows
/// kept count for failures.
///
/// Any null operand gives null. Division or remainder by zero gives null, as
/// in SQLite. An integer overflow, which SQLite would answer with a double,
/// fails the run.
pub struct ExprProgram {
    pub steps: SlowVec<ExprStep>,
}

/// Adds one column to each batch for each program. Programs read only the
/// columns the batch came with.
pub struct Compute {
    pub programs: SlowVec<ExprProgram>,
}

/// What a run of a `Compute` keeps between batches.
pub struct ComputeState {
    /// By program, then step, a constant step's value at every row a batch
    /// can have, made once per run.
    constants: SlowVec<SlowVec<Option<ColumnView>>>,
    /// By step, what each computed for the program being run. Emptied after
    /// each, so the buffers they shared can be reused.
    registers: SlowVec<Option<ColumnView>>,
}

impl Transform for Compute {
    type State = ComputeState;

    fn new_state(&self, context: &mut Context) -> Result<ComputeState, Error> {
        let allocator = context.allocator();
        let mut constants = SlowVec::fixed(allocator, self.programs.len())?;
        let mut steps_max = 0;
        for program in self.programs.iter() {
            let mut values = SlowVec::fixed(allocator, program.steps.len())?;
            for step in program.steps.iter() {
                values.push(constant(context, step)?)?;
            }
            constants.push(values)?;
            steps_max = steps_max.max(program.steps.len());
        }
        let registers = SlowVec::fixed_from(allocator, (0..steps_max).map(|_| None))?;
        Ok(ComputeState { constants, registers })
    }

    fn process(
        &self,
        context: &mut Context,
        state: &mut ComputeState,
        batch: &mut RowBatch,
    ) -> Result<(), Error> {
        for (program, constants) in self.programs.iter().zip(state.constants.iter()) {
            let result = run(context, program, constants, &mut state.registers, batch);
            state.registers.iter_mut().for_each(|register| *register = None);
            let Ok(()) = batch.push_column(result?) else { failed!() };
        }
        Ok(())
    }
}

/// `step`'s value at every row a batch can have, if it's a constant.
fn constant(context: &mut Context, step: &ExprStep) -> Result<Option<ColumnView>, AllocError> {
    if !matches!(step.op, ExprOp::Constant | ExprOp::Null) {
        return Ok(None);
    }
    let rows = BATCH_ROWS_MAX as usize;
    let mut values = context.values_buffer(rows * 8)?;
    if step.data_type == DataType::Float64 {
        values.as_mut_slice::<f64>().fill(step.double_value);
    } else {
        check!(step.data_type == DataType::Int64);
        values.as_mut_slice::<i64>().fill(step.int_value);
    }
    let mut validity = None;
    if step.op == ExprOp::Null {
        let mut none = context.small_buffer(rows / 8)?;
        none.as_mut_slice::<u8>().fill(0);
        validity = Some(none);
    }
    Ok(Some(ColumnView::new(context, step.data_type, values, validity)?))
}

/// Computes each of `program`'s steps for `batch`, returning the last.
fn run(
    context: &mut Context,
    program: &ExprProgram,
    constants: &[Option<ColumnView>],
    registers: &mut [Option<ColumnView>],
    batch: &RowBatch,
) -> Result<ColumnView, Error> {
    let rows = batch.row_count();
    for (i, step) in program.steps.iter().enumerate() {
        let value = match step.op {
            ExprOp::Column => {
                let mut column = batch.column(step.a).clone();
                check!(column.data_type() == step.data_type);
                column.make_in(context, Forms::FLAT)?;
                column
            }
            ExprOp::Constant | ExprOp::Null => {
                let Some(constant) = at!(constants, i) else { failed!() };
                constant.slice(0, rows)
            }
            ExprOp::ToDouble => {
                let a = register(registers, step.a);
                let mut values = context.values_buffer(rows as usize * 8)?;
                for (o, &x) in values.as_mut_slice::<f64>().iter_mut().zip(a.int64s()) {
                    #[expect(clippy::cast_precision_loss, reason = "as SQLite converts")]
                    let x = x as f64;
                    *o = x;
                }
                let validity = validity(context, a, None, None)?;
                ColumnView::new(context, DataType::Float64, values, validity)?
            }
            ExprOp::Add | ExprOp::Sub | ExprOp::Mul | ExprOp::Div | ExprOp::Mod => {
                arithmetic(context, step, registers, batch.selection())?
            }
        };
        *at_mut!(registers, i) = Some(value);
    }
    let Some(Some(result)) =
        registers.get_mut(program.steps.len().wrapping_sub(1)).map(Option::take)
    else {
        failed!()
    };
    Ok(result)
}

/// The value register `index` holds.
fn register(registers: &[Option<ColumnView>], index: u32) -> &ColumnView {
    let Some(register) = at!(registers, index as usize) else { failed!() };
    register
}

/// Computes `step`, which reads two registers. Fails if an integer result
/// doesn't fit at a row the batch keeps.
fn arithmetic(
    context: &mut Context,
    step: &ExprStep,
    registers: &[Option<ColumnView>],
    selection: &Selection,
) -> Result<ColumnView, Error> {
    let (a, b) = (register(registers, step.a), register(registers, step.b));
    let rows = a.row_count() as usize;
    let doubles = step.data_type == DataType::Float64;
    let division = matches!(step.op, ExprOp::Div | ExprOp::Mod);
    let divisor = division.then(|| (b.values().words(), doubles));
    let validity = validity(context, a, Some(b), divisor)?;
    let mut values = context.values_buffer(rows * 8)?;
    if doubles {
        double_arithmetic(step.op, a.float64s(), b.float64s(), values.as_mut_slice());
        return Ok(ColumnView::new(context, step.data_type, values, validity)?);
    }
    let (x, y) = (a.int64s(), b.int64s());
    let all_fit = int_arithmetic(step.op, x, y, values.as_mut_slice());
    let out = ColumnView::new(context, step.data_type, values, validity)?;
    if !all_fit && kept_row_fails(step.op, x, y, &out, selection) {
        return Err(Error::Overflow);
    }
    Ok(out)
}

/// Computes `op` for every row, returning whether each result fit.
fn int_arithmetic(op: ExprOp, x: &[i64], y: &[i64], o: &mut [i64]) -> bool {
    // A zero divisor gives null, so any value will do for its row.
    let nonzero = |b: i64| b | i64::from(b == 0);
    match op {
        // Overflow is found from the signs, which, unlike the CPU's flag,
        // vectorizes.
        ExprOp::Add => each(x, y, o, |a, b| {
            let c = a.wrapping_add(b);
            (c, (a ^ c) & (b ^ c) >= 0)
        }),
        ExprOp::Sub => each(x, y, o, |a, b| {
            let c = a.wrapping_sub(b);
            (c, (a ^ b) & (a ^ c) >= 0)
        }),
        ExprOp::Mul => each(x, y, o, |a, b| fits(a.overflowing_mul(b))),
        ExprOp::Div => each(x, y, o, |a, b| fits(a.overflowing_div(nonzero(b)))),
        ExprOp::Mod => each(x, y, o, |a, b| (a.wrapping_rem(nonzero(b)), true)),
        ExprOp::Column | ExprOp::Constant | ExprOp::Null | ExprOp::ToDouble => failed!(),
    }
}

fn double_arithmetic(op: ExprOp, x: &[f64], y: &[f64], o: &mut [f64]) {
    match op {
        ExprOp::Add => each(x, y, o, |a, b| (a + b, true)),
        ExprOp::Sub => each(x, y, o, |a, b| (a - b, true)),
        ExprOp::Mul => each(x, y, o, |a, b| (a * b, true)),
        ExprOp::Div => each(x, y, o, |a, b| (a / b, true)),
        ExprOp::Mod | ExprOp::Column | ExprOp::Constant | ExprOp::Null | ExprOp::ToDouble => {
            failed!()
        }
    };
}

/// An overflowing operation's result, and whether it fit.
fn fits((value, overflowed): (i64, bool)) -> (i64, bool) {
    (value, !overflowed)
}

/// The loop for each operation and type, kept to the arithmetic itself so
/// that each copy is small. Returns whether every result fit, checking once
/// at the end rather than at each row.
#[inline(never)]
fn each<T: Copy>(x: &[T], y: &[T], o: &mut [T], f: impl Fn(T, T) -> (T, bool)) -> bool {
    check!(x.len() == o.len() && y.len() == o.len());
    let mut all_fit = true;
    for ((o, &x), &y) in o.iter_mut().zip(x).zip(y) {
        let (value, fit) = f(x, y);
        *o = value;
        all_fit &= fit;
    }
    all_fit
}

/// Whether `op` overflows at some row `selection` keeps where `out` isn't
/// null. The loops compute every row, so this only runs once one of them
/// has overflowed somewhere.
#[inline(never)]
fn kept_row_fails(
    op: ExprOp,
    x: &[i64],
    y: &[i64],
    out: &ColumnView,
    selection: &Selection,
) -> bool {
    let kept = selection.kept();
    (0..selection.len()).any(|i| {
        let row = kept.row(i);
        let one = row as usize..=row as usize;
        !out.is_null(row) && !int_arithmetic(op, at!(x, one.clone()), at!(y, one), &mut [0])
    })
}

/// Which rows of a step reading `a`, and `b` if any, aren't null: those where
/// each isn't, and for a division, where `divisor`, the divisor's words and
/// whether they're doubles' bits, isn't zero. `None` if that's every row.
fn validity(
    context: &mut Context,
    a: &ColumnView,
    b: Option<&ColumnView>,
    divisor: Option<(&[i64], bool)>,
) -> Result<Option<Buffer>, AllocError> {
    let rows = a.row_count() as usize;
    let (a, b) = (a.validity(), b.and_then(ColumnView::validity));
    if a.is_none() && b.is_none() && divisor.is_none() {
        return Ok(None);
    }
    let mut buffer = context.small_buffer(rows.div_ceil(8))?;
    for (i, bytes) in buffer.as_mut_slice::<u8>().chunks_mut(8).enumerate() {
        let (at, count) = (i * 64, (rows - i * 64).min(64));
        let word = word(a, at, count) & word(b, at, count) & nonzero(divisor, at, count);
        let word = word.to_le_bytes();
        bytes.copy_from_slice(at!(word, ..bytes.len()));
    }
    Ok(Some(buffer))
}

/// Which of the `count` rows from `at`, a multiple of 64, aren't null, as
/// the bits of a word.
fn word(validity: Option<Validity>, at: usize, count: usize) -> u64 {
    let Some(validity) = validity else { return u64::MAX };
    let (bits, shift) = validity.bits();
    if count == 64 {
        // The rows past the first eight bytes' are in the next, which covers
        // them.
        let mut low = [0; 8];
        low.copy_from_slice(at!(bits, at / 8..at / 8 + 8));
        let low = u64::from_le_bytes(low);
        let high = if shift == 0 { 0 } else { *at!(bits, at / 8 + 8) };
        #[expect(clippy::cast_possible_truncation, reason = "64 bits")]
        let word = ((u128::from(low) | u128::from(high) << 64) >> shift) as u64;
        return word;
    }
    (0..count).fold(0, |word, k| {
        let bit = shift as usize + at + k;
        word | u64::from(at!(bits, bit / 8) >> (bit % 8) & 1) << k
    })
}

/// Which of the `count` divisors from `at` aren't zero, as the bits of a
/// word: all if there are none. A double is zero if every bit but its sign
/// is.
fn nonzero(divisor: Option<(&[i64], bool)>, at: usize, count: usize) -> u64 {
    let Some((words, doubles)) = divisor else { return u64::MAX };
    let words = at!(words, at..at + count);
    words
        .iter()
        .enumerate()
        .fold(0, |nonzero, (k, &word)| nonzero | u64::from(word << u32::from(doubles) != 0) << k)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec as StdVec;

    use super::*;
    use crate::allocator::Heap;

    /// A batch's `n`, with a null.
    const N: [Option<i64>; 5] = [Some(10), Some(20), None, Some(40), Some(50)];
    /// A batch's `u`, with zeros.
    const U: [Option<i64>; 5] = [Some(1), Some(0), Some(3), Some(0), Some(5)];

    fn column(cells: &[Option<i64>]) -> ColumnView {
        let mut values = Buffer::allocate(&Heap, cells.len() * 8).unwrap();
        let mut validity = Buffer::allocate(&Heap, cells.len().div_ceil(8)).unwrap();
        validity.as_mut_slice::<u8>().fill(0);
        for (row, cell) in cells.iter().enumerate() {
            values.as_mut_slice::<i64>()[row] = cell.unwrap_or(i64::MAX);
            validity.as_mut_slice::<u8>()[row / 8] |= u8::from(cell.is_some()) << (row % 8);
        }
        let validity = cells.contains(&None).then_some(validity);
        ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, validity).unwrap()
    }

    fn batch(columns: &[ColumnView]) -> RowBatch<'static> {
        let mut batch = RowBatch::new();
        batch.reset(columns[0].row_count());
        for column in columns {
            assert!(batch.push_column(column.clone()).is_ok());
        }
        batch
    }

    /// Builds a program a step at a time; each call returns the step's
    /// register.
    struct Builder(StdVec<ExprStep>);

    impl Builder {
        fn push(&mut self, op: ExprOp, data_type: DataType, a: u32, b: u32) -> u32 {
            let step = ExprStep { op, data_type, a, b, int_value: 0, double_value: 0.0 };
            self.0.push(step);
            u32::try_from(self.0.len() - 1).unwrap()
        }

        fn int(&mut self, value: i64) -> u32 {
            let at = self.push(ExprOp::Constant, DataType::Int64, 0, 0);
            self.0[at as usize].int_value = value;
            at
        }

        fn double(&mut self, value: f64) -> u32 {
            let at = self.push(ExprOp::Constant, DataType::Float64, 0, 0);
            self.0[at as usize].double_value = value;
            at
        }

        fn program(self) -> ExprProgram {
            ExprProgram { steps: SlowVec::fixed_from(&Heap, self.0.into_iter()).unwrap() }
        }
    }

    fn compute(programs: impl ExactSizeIterator<Item = ExprProgram>) -> Compute {
        Compute { programs: SlowVec::fixed_from(&Heap, programs).unwrap() }
    }

    /// `program`'s column of `batch`, at its kept rows, as words, or the
    /// run's error.
    fn process(program: Builder, mut batch: RowBatch) -> Result<StdVec<Option<i64>>, Error> {
        let compute = compute([program.program()].into_iter());
        let mut context = Context::new(&Heap);
        let mut state = compute.new_state(&mut context)?;
        compute.process(&mut context, &mut state, &mut batch)?;
        let column = batch.column(batch.column_count() - 1);
        let words = column.values().words();
        let kept = batch.selection().kept();
        let rows = (0..batch.selection().len()).map(|i| kept.row(i));
        Ok(rows.map(|row| (!column.is_null(row)).then(|| words[row as usize])).collect())
    }

    fn run(program: Builder, batch: RowBatch) -> StdVec<Option<i64>> {
        process(program, batch).unwrap()
    }

    fn doubles(cells: StdVec<Option<i64>>) -> StdVec<Option<f64>> {
        cells
            .into_iter()
            .map(|cell| cell.map(|word| f64::from_bits(word.cast_unsigned())))
            .collect()
    }

    const INT: DataType = DataType::Int64;

    #[test]
    fn arithmetic_propagates_nulls() {
        let mut b = Builder(StdVec::new());
        let (n, u, two) =
            (b.push(ExprOp::Column, INT, 0, 0), b.push(ExprOp::Column, INT, 1, 0), b.int(2));
        let doubled = b.push(ExprOp::Mul, INT, n, two);
        b.push(ExprOp::Add, INT, doubled, u);
        let cells = run(b, batch(&[column(&N), column(&U)]));
        assert_eq!(cells, [Some(21), Some(40), None, Some(80), Some(105)]);
    }

    #[test]
    fn division_by_zero_is_null() {
        let mut b = Builder(StdVec::new());
        let (n, u) = (b.push(ExprOp::Column, INT, 0, 0), b.push(ExprOp::Column, INT, 1, 0));
        b.push(ExprOp::Div, INT, n, u);
        let cells = run(b, batch(&[column(&N), column(&U)]));
        assert_eq!(cells, [Some(10), None, None, None, Some(10)]);

        let mut m = Builder(StdVec::new());
        let (n, u) = (m.push(ExprOp::Column, INT, 0, 0), m.push(ExprOp::Column, INT, 1, 0));
        m.push(ExprOp::Mod, INT, n, u);
        let cells = run(m, batch(&[column(&N), column(&U)]));
        assert_eq!(cells, [Some(0), None, None, None, Some(0)]);
    }

    #[test]
    fn overflow_fails() {
        let mut b = Builder(StdVec::new());
        let (n, max) = (b.push(ExprOp::Column, INT, 0, 0), b.int(i64::MAX));
        b.push(ExprOp::Add, INT, n, max);
        assert_eq!(process(b, batch(&[column(&N)])), Err(Error::Overflow));

        let mut d = Builder(StdVec::new());
        let (min, minus_one) = (d.int(i64::MIN), d.int(-1));
        d.push(ExprOp::Div, INT, min, minus_one);
        assert_eq!(process(d, batch(&[column(&N)])), Err(Error::Overflow));
    }

    /// The null row holds a value which would overflow, but has no value to
    /// overflow.
    #[test]
    fn null_rows_do_not_overflow() {
        let mut b = Builder(StdVec::new());
        let (n, one) = (b.push(ExprOp::Column, INT, 0, 0), b.int(1));
        b.push(ExprOp::Add, INT, n, one);
        let cells = run(b, batch(&[column(&N)]));
        assert_eq!(cells, [Some(11), Some(21), None, Some(41), Some(51)]);
    }

    #[test]
    fn null_constant() {
        let mut b = Builder(StdVec::new());
        let (n, null) = (b.push(ExprOp::Column, INT, 0, 0), b.push(ExprOp::Null, INT, 0, 0));
        b.push(ExprOp::Add, INT, n, null);
        assert_eq!(run(b, batch(&[column(&N)])), [None; 5]);
    }

    #[test]
    fn doubles_divide_by_zero_to_null() {
        let float = DataType::Float64;
        let mut b = Builder(StdVec::new());
        let n = b.push(ExprOp::Column, INT, 0, 0);
        let (n, quarter) = (b.push(ExprOp::ToDouble, float, n, 0), b.double(0.25));
        b.push(ExprOp::Mul, float, n, quarter);
        let cells = doubles(run(b, batch(&[column(&N)])));
        assert_eq!(cells, [Some(2.5), Some(5.0), None, Some(10.0), Some(12.5)]);

        let mut z = Builder(StdVec::new());
        let u = z.push(ExprOp::Column, INT, 0, 0);
        let (u, zero) = (z.push(ExprOp::ToDouble, float, u, 0), z.double(-0.0));
        z.push(ExprOp::Div, float, u, zero);
        assert_eq!(doubles(run(z, batch(&[column(&U)]))), [None; 5]);
    }

    /// Every row is computed, but only the rows kept can fail, and are
    /// returned.
    #[test]
    fn dropped_rows_do_not_overflow() {
        let mut b = Builder(StdVec::new());
        let (n, two) = (b.push(ExprOp::Column, INT, 0, 0), b.int(2));
        b.push(ExprOp::Mul, INT, n, two);
        let mut batch = batch(&[column(&[Some(10), Some(i64::MAX), Some(20), Some(40)])]);
        batch.selection_mut().retain(|row| row != 1);
        assert_eq!(run(b, batch), [Some(20), Some(40), Some(80)]);
    }

    /// Columns which start mid-byte of their validity, as a scan's later
    /// batches do, over more than a byte of rows.
    #[test]
    fn windowed_batch() {
        let cells: StdVec<Option<i64>> = (0..30).map(|i| (i % 3 != 0).then_some(i)).collect();
        let window = |column: ColumnView| column.slice(3, 20);
        let mut b = Builder(StdVec::new());
        let (n, m) = (b.push(ExprOp::Column, INT, 0, 0), b.push(ExprOp::Column, INT, 1, 0));
        b.push(ExprOp::Sub, INT, n, m);
        let (n, m) = (window(column(&cells)), window(column(&cells[1..])));
        let expected: StdVec<Option<i64>> =
            (3..23).map(|i| (i % 3 != 0 && (i + 1) % 3 != 0).then_some(-1)).collect();
        assert_eq!(run(b, batch(&[n, m])), expected);
    }

    /// Each program reads the batch as it came, and adds its column.
    #[test]
    fn adds_a_column_for_each_program() {
        let mut first = Builder(StdVec::new());
        let n = first.push(ExprOp::Column, INT, 0, 0);
        first.push(ExprOp::Add, INT, n, n);
        let mut second = Builder(StdVec::new());
        second.push(ExprOp::Column, INT, 0, 0);
        let compute = compute([first.program(), second.program()].into_iter());
        let mut context = Context::new(&Heap);
        let mut state = compute.new_state(&mut context).unwrap();
        let mut batch = batch(&[column(&U)]);
        for _ in 0..2 {
            batch.reset(5);
            assert!(batch.push_column(column(&U)).is_ok());
            compute.process(&mut context, &mut state, &mut batch).unwrap();
        }
        assert_eq!(batch.column_count(), 3);
        assert_eq!(batch.column(1).int64s(), [2, 0, 6, 0, 10]);
        assert_eq!(batch.column(2).int64s(), [1, 0, 3, 0, 5]);
    }
}
