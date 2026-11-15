//! Expressions over a row's columns, as Perfetto's: `Expr`, as a plan holds
//! one; `ComputeOp`, the op that adds columns computed by them; and
//! `Compute`, the transform that runs it.

use crate::allocator::{AllocError, Allocator};
use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType, Forms, Validity};
use crate::context::Context;
use crate::error::Error;
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{Column, DeriveContext, OutputProperties};
use crate::plan::{ColumnId, Op, PlanNode};
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::selection::Selection;
use crate::slow_vec::SlowVec;
use crate::step::{DynTransform, Step, Transform};
use crate::string_pool::Id;

/// What a step of an `ExprProgram` computes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExprOp {
    /// Reads batch column `a`.
    Column,
    /// A constant: `int_value` or `double_value` by type, or for a string,
    /// the string whose id in the statement's literals is `a`.
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
    /// `a % b`. Of Int64s only.
    Mod,
    /// `a = b`, of one type, as an Int64 0 or 1: null if either is null, as
    /// are the other comparisons.
    Eq,
    /// `a != b`.
    Ne,
    /// `a < b`. Strings are ordered by their bytes.
    Lt,
    /// `a <= b`.
    Le,
    /// `a > b`.
    Gt,
    /// `a >= b`.
    Ge,
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
/// columns the batch came with, which mustn't be lazy, and the batch must
/// have room for a column more for each.
pub struct Compute {
    pub programs: SlowVec<ExprProgram>,
}

/// What a run of a `Compute` keeps between batches.
pub struct ComputeState {
    /// By program, then step, a constant step's value at every row a batch
    /// can have, made once per run.
    constants: SlowVec<SlowVec<Option<ColumnView>>>,
    /// By step, what each computed for the program last run.
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
            let result = run(context, program, constants, &mut state.registers, batch)?;
            let Ok(()) = batch.push_column(result) else { failed!() };
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
    let mut validity = None;
    if step.op == ExprOp::Null {
        let mut none = context.small_buffer(rows / 8)?;
        none.as_mut_slice::<u8>().fill(0);
        validity = Some(none);
    }
    let mut values = context.values_buffer(rows * 8)?;
    if step.data_type == DataType::String {
        // Every row's view is of one copy of its bytes: none for a null.
        let text = match step.op {
            ExprOp::Constant => {
                let literal = context.literals().and_then(|literals| literals.get(Id::raw(step.a)));
                let Some(literal) = literal else { failed!() };
                literal
            }
            _ => &[],
        };
        let mut bytes = context.bytes_buffer(text.len())?;
        bytes.as_mut_slice::<u8>().copy_from_slice(text);
        let Ok(len) = u32::try_from(text.len()) else { failed!() };
        values.as_mut_slice::<u32>().as_chunks_mut::<2>().0.fill([0, len]);
        return Ok(Some(ColumnView::strings(context, values, bytes, validity)?));
    }
    if step.data_type == DataType::Float64 {
        values.as_mut_slice::<f64>().fill(step.double_value);
    } else {
        check!(step.data_type == DataType::Int64);
        values.as_mut_slice::<i64>().fill(step.int_value);
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
            ExprOp::Eq | ExprOp::Ne | ExprOp::Lt | ExprOp::Le | ExprOp::Gt | ExprOp::Ge => {
                compare(context, step, registers)?
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
        _ => failed!(),
    }
}

fn double_arithmetic(op: ExprOp, x: &[f64], y: &[f64], o: &mut [f64]) {
    match op {
        ExprOp::Add => each(x, y, o, |a, b| (a + b, true)),
        ExprOp::Sub => each(x, y, o, |a, b| (a - b, true)),
        ExprOp::Mul => each(x, y, o, |a, b| (a * b, true)),
        ExprOp::Div => each(x, y, o, |a, b| (a / b, true)),
        _ => failed!(),
    };
}

/// Computes the comparison `step`, as `Eq`, `Lt` or `Le`, so that there are
/// fewer copies of the loops: `a > b` is `b < a`, `a >= b` is `b <= a`, and
/// `a != b` is the opposite of `a = b`, which holds for doubles too.
#[inline(never)]
fn compare(
    context: &mut Context,
    step: &ExprStep,
    registers: &[Option<ColumnView>],
) -> Result<ColumnView, AllocError> {
    let swap = matches!(step.op, ExprOp::Gt | ExprOp::Ge);
    let (a, b) = if swap { (step.b, step.a) } else { (step.a, step.b) };
    let (a, b) = (register(registers, a), register(registers, b));
    let op = match step.op {
        ExprOp::Ne => ExprOp::Eq,
        ExprOp::Gt => ExprOp::Lt,
        ExprOp::Ge => ExprOp::Le,
        op => op,
    };
    let rows = a.row_count() as usize;
    let validity = validity(context, a, Some(b), None)?;
    let mut values = context.values_buffer(rows * 8)?;
    let o = values.as_mut_slice::<i64>();
    match a.data_type() {
        DataType::Int64 => compare_numbers(op, a.int64s(), b.int64s(), o),
        DataType::Float64 => compare_numbers(op, a.float64s(), b.float64s(), o),
        DataType::String => compare_strings(op, a, b, o),
    }
    if step.op == ExprOp::Ne {
        for o in o.iter_mut() {
            *o ^= 1;
        }
    }
    ColumnView::new(context, DataType::Int64, values, validity)
}

/// `x op y` at every row, as 0 or 1, for `op` `Eq`, `Lt` or `Le`.
fn compare_numbers<T: PartialOrd + Copy>(op: ExprOp, x: &[T], y: &[T], o: &mut [i64]) {
    match op {
        ExprOp::Eq => compare_each(x, y, o, |a, b| a == b),
        ExprOp::Lt => compare_each(x, y, o, |a, b| a < b),
        _ => compare_each(x, y, o, |a, b| a <= b),
    }
}

/// The loop for each comparison and type, kept to the comparison itself.
#[inline(never)]
fn compare_each<T: Copy>(x: &[T], y: &[T], o: &mut [i64], f: impl Fn(T, T) -> bool) {
    check!(x.len() == o.len() && y.len() == o.len());
    for ((o, &x), &y) in o.iter_mut().zip(x).zip(y) {
        *o = i64::from(f(x, y));
    }
}

/// As `compare_numbers`, for strings, by their bytes, at the rows where
/// neither is null: others' views may not be of their bytes.
fn compare_strings(op: ExprOp, a: &ColumnView, b: &ColumnView, o: &mut [i64]) {
    let (left, right) = (a.string_values(), b.string_values());
    for (row, o) in o.iter_mut().enumerate() {
        #[expect(clippy::cast_possible_truncation, reason = "a batch's rows")]
        let at = row as u32;
        if a.is_null(at) || b.is_null(at) {
            *o = 0;
            continue;
        }
        let order = left.get(row).cmp(right.get(row));
        *o = i64::from(match op {
            ExprOp::Eq => order.is_eq(),
            ExprOp::Lt => order.is_lt(),
            _ => order.is_le(),
        });
    }
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

/// The most nodes an `Expr` has, so typing one needs only the stack.
pub const EXPR_NODES_MAX: usize = 1 << 8;

/// A scalar expression over the columns of a row, as a plan holds it: column
/// references, literals, NULL and arithmetic.
///
/// It has the shape of an `ExprProgram`: nodes in post-order, each reading
/// only nodes before it, the last being the result. A `Column` node's `a` is
/// a `ColumnId` rather than a batch position, and only a `Constant` node's
/// `data_type` means anything: the rest are typed when the plan is lowered. A
/// string `Constant`'s `a` is its id in the statement's literals.
///
/// Only expressions whose meaning is the same as SQLite's are accepted.
/// Arithmetic on strings is an error, and an integer overflow fails the query
/// where SQLite would answer with a double.
pub struct Expr {
    pub nodes: SlowVec<ExprStep>,
}

/// Why an expression can't be computed: what it needed instead, or memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExprError {
    /// Numbers, not strings, in arithmetic.
    Numbers,
    /// Integers on both sides of `%`.
    Integers,
    /// Values of one kind, numbers or strings, on both sides of a comparison.
    Kinds,
    /// Values of one type on both sides of a comparison, unless one is an
    /// integer literal compared with a double.
    Types,
    /// An integer literal compared with a double which can hold it exactly.
    Inexact,
    /// An allocation failed.
    OutOfMemory,
}

impl From<AllocError> for ExprError {
    fn from(_: AllocError) -> ExprError {
        ExprError::OutOfMemory
    }
}

/// A step of no value, reading registers `a` and `b`.
const fn step(op: ExprOp, data_type: DataType, a: u32, b: u32) -> ExprStep {
    ExprStep { op, data_type, a, b, int_value: 0, double_value: 0.0 }
}

/// Types an expression's nodes in order and, given the rows it reads,
/// appends the steps computing them to a program. Without them it only
/// checks, as compiling does.
struct Typer<'t> {
    /// Each column's type.
    type_of: &'t dyn Fn(ColumnId) -> DataType,
    /// The rows the program reads, and its steps so far, if building one.
    build: Option<(&'t OutputProperties, &'t mut SlowVec<ExprStep>)>,
    /// By node, its type, and the register holding it. A bare NULL has no
    /// type of its own: it takes the type of whatever it meets.
    nodes: [(Option<DataType>, u32); EXPR_NODES_MAX],
}

impl Typer<'_> {
    /// The type of the expression's result.
    #[inline(never)]
    fn run(&mut self, expr: &Expr) -> Result<Option<DataType>, ExprError> {
        check!(!expr.nodes.is_empty() && expr.nodes.len() <= EXPR_NODES_MAX);
        for (i, node) in expr.nodes.iter().enumerate() {
            *at_mut!(self.nodes, i) = self.type_node(&expr.nodes, node)?;
        }
        Ok(at!(self.nodes, expr.nodes.len() - 1).0)
    }

    /// The register holding node `i` as `data_type`: a null of that type for
    /// a bare NULL, an integer widened for a double.
    fn materialize(&mut self, i: u32, data_type: DataType) -> Result<u32, ExprError> {
        let (node_type, register) = *at!(self.nodes, i as usize);
        match node_type {
            None => self.add(step(ExprOp::Null, data_type, 0, 0)),
            Some(DataType::Int64) if data_type == DataType::Float64 => {
                self.add(step(ExprOp::ToDouble, data_type, register, 0))
            }
            Some(_) => Ok(register),
        }
    }

    /// Appends `step`, if building, and returns its register.
    fn add(&mut self, step: ExprStep) -> Result<u32, ExprError> {
        let Some((_, steps)) = &mut self.build else { return Ok(0) };
        steps.push(step)?;
        #[expect(clippy::cast_possible_truncation, reason = "at most `4 * EXPR_NODES_MAX`")]
        Ok(steps.len() as u32 - 1)
    }

    /// `node`, one of `nodes`, its type, and the register holding it.
    fn type_node(
        &mut self,
        nodes: &[ExprStep],
        node: &ExprStep,
    ) -> Result<(Option<DataType>, u32), ExprError> {
        let (a, b) = (node.a as usize, node.b as usize);
        match node.op {
            ExprOp::Column => {
                let data_type = (self.type_of)(node.a);
                let mut read = step(ExprOp::Column, data_type, node.a, 0);
                if let Some((rows, _)) = &self.build {
                    let Some(position) = rows.position_of(node.a) else { failed!() };
                    read.a = position;
                }
                Ok((Some(data_type), self.add(read)?))
            }
            ExprOp::Constant => Ok((Some(node.data_type), self.add(*node)?)),
            ExprOp::Null => Ok((None, 0)),
            ExprOp::ToDouble => failed!(),
            ExprOp::Add | ExprOp::Sub | ExprOp::Mul | ExprOp::Div | ExprOp::Mod => {
                let (a_type, b_type) = (at!(self.nodes, a).0, at!(self.nodes, b).0);
                let either = |data_type| a_type == Some(data_type) || b_type == Some(data_type);
                if either(DataType::String) {
                    return Err(ExprError::Numbers);
                }
                if node.op == ExprOp::Mod && either(DataType::Float64) {
                    return Err(ExprError::Integers);
                }
                // Any arithmetic with NULL is NULL.
                if a_type.is_none() || b_type.is_none() {
                    return Ok((None, 0));
                }
                let data_type =
                    if either(DataType::Float64) { DataType::Float64 } else { DataType::Int64 };
                let a = self.materialize(node.a, data_type)?;
                let b = self.materialize(node.b, data_type)?;
                Ok((Some(data_type), self.add(step(node.op, data_type, a, b))?))
            }
            ExprOp::Eq | ExprOp::Ne | ExprOp::Lt | ExprOp::Le | ExprOp::Gt | ExprOp::Ge => {
                self.type_comparison(nodes, node)
            }
        }
    }

    /// Types the comparison `node`, one of `nodes`, which is an Int64.
    fn type_comparison(
        &mut self,
        nodes: &[ExprStep],
        node: &ExprStep,
    ) -> Result<(Option<DataType>, u32), ExprError> {
        let (a_type, b_type) =
            (at!(self.nodes, node.a as usize).0, at!(self.nodes, node.b as usize).0);
        let string = |data_type| data_type == Some(DataType::String);
        if a_type.is_some() && b_type.is_some() && string(a_type) != string(b_type) {
            return Err(ExprError::Kinds);
        }
        // A comparison with NULL is NULL.
        let (Some(a_type), Some(b_type)) = (a_type, b_type) else { return Ok((None, 0)) };
        let mut a = at!(self.nodes, node.a as usize).1;
        let mut b = at!(self.nodes, node.b as usize).1;
        if a_type != b_type {
            // An integer literal compared with a double is that double, as
            // a filter compares it.
            let literal = |i: u32| {
                let node = at!(nodes, i as usize);
                (node.op == ExprOp::Constant && node.data_type == DataType::Int64)
                    .then_some(node.int_value)
            };
            let (side, value) = match (a_type, literal(node.a), literal(node.b)) {
                (DataType::Int64, Some(value), _) => (&mut a, value),
                (DataType::Float64, _, Some(value)) => (&mut b, value),
                _ => return Err(ExprError::Types),
            };
            #[expect(clippy::cast_precision_loss, reason = "checked to be exact")]
            let double = value as f64;
            #[expect(clippy::cast_possible_truncation, reason = "a check, not a use")]
            if double as i128 != i128::from(value) {
                return Err(ExprError::Inexact);
            }
            let mut constant = step(ExprOp::Constant, DataType::Float64, 0, 0);
            constant.double_value = double;
            *side = self.add(constant)?;
        }
        Ok((Some(DataType::Int64), self.add(step(node.op, DataType::Int64, a, b))?))
    }
}

/// The type `expr` computes, given each column's type, as compiling checks
/// it.
/// Whether `expr` can compute a null, given whether each column it reads
/// can hold one: it can if one can, or if it divides by what may be zero,
/// anything but a constant that isn't, or has a `NULL`.
pub fn expr_can_be_null(expr: &Expr, nullable_of: &dyn Fn(ColumnId) -> bool) -> bool {
    let nonzero = |node: &ExprStep| {
        node.op == ExprOp::Constant && (node.int_value != 0 || node.double_value != 0.0)
    };
    expr.nodes.iter().any(|node| match node.op {
        ExprOp::Column => nullable_of(node.a),
        ExprOp::Div | ExprOp::Mod => expr.nodes.get(node.b as usize).is_none_or(|b| !nonzero(b)),
        ExprOp::Null => true,
        _ => false,
    })
}

pub fn check_expr_type(
    expr: &Expr,
    type_of: &dyn Fn(ColumnId) -> DataType,
) -> Result<DataType, ExprError> {
    let mut typer = Typer { type_of, build: None, nodes: [(None, 0); EXPR_NODES_MAX] };
    // A result which is only ever NULL is a column of Int64 nulls.
    Ok(typer.run(expr)?.unwrap_or(DataType::Int64))
}

/// Marks the columns `expr` reads.
pub fn mark_expr_columns(expr: &Expr, needed: &mut Needed) {
    for node in expr.nodes.iter().filter(|node| node.op == ExprOp::Column) {
        needed.need(node.a);
    }
}

/// The program computing `expr`, with columns at their positions in `rows`.
pub fn type_expr(
    allocator: &dyn Allocator,
    rows: &OutputProperties,
    expr: &Expr,
) -> Result<ExprProgram, ExprError> {
    // Each node takes at most three steps, and a NULL result one more.
    let mut steps = SlowVec::new(allocator, 4 * EXPR_NODES_MAX)?;
    let type_of = |column| rows.type_of(column);
    let nodes = [(None, 0); EXPR_NODES_MAX];
    let mut typer = Typer { type_of: &type_of, build: Some((rows, &mut steps)), nodes };
    let result = typer.run(expr)?.unwrap_or(DataType::Int64);
    #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
    let register = typer.materialize(expr.nodes.len() as u32 - 1, result)?;
    // Unlike Perfetto's, no column is only ever NULL, so the last node's step
    // is the last.
    check!(register as usize == steps.len() - 1);
    Ok(ExprProgram { steps })
}

/// A column a `ComputeOp` adds.
pub struct ComputeItem {
    /// The column.
    pub output: ColumnId,
    /// What computes its values.
    pub expr: Expr,
}

/// Adds columns computed from the rows of its one child: what `SELECT`,
/// `EXTEND` and `SET` add for an item which is more than a column. Those
/// stages add it, so it has no syntax of its own.
pub struct ComputeOp {
    pub items: SlowVec<ComputeItem>,
}

/// An error typing the expression of `column`, as lowering reports it.
fn lower_error(error: ExprError, column: ColumnId) -> LowerError {
    match error {
        ExprError::OutOfMemory => LowerError::OutOfMemory,
        ExprError::Numbers
        | ExprError::Integers
        | ExprError::Kinds
        | ExprError::Types
        | ExprError::Inexact => LowerError::WrongType(column),
    }
}

impl<'c> Op<'c> for ComputeOp {
    /// Drops the items whose columns aren't needed, and itself with the
    /// last of them.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        self.items.retain(|item| needed.is_needed(item.output));
        if self.items.is_empty() {
            return Pruned::Child(0);
        }
        self.reads(needed);
        Pruned::Keep
    }

    /// Its expressions' columns.
    fn reads(&self, reads: &mut Needed) {
        for item in self.items.iter() {
            mark_expr_columns(&item.expr, reads);
        }
    }

    /// Its input's, with each item's column after them.
    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        let input = c.input(0);
        let mut out = *input;
        for item in self.items.iter() {
            let data_type = check_expr_type(&item.expr, &|column| input.type_of(column))
                .map_err(|error| lower_error(error, item.output))?;
            let nullable = expr_can_be_null(&item.expr, &|column| input.nullable_of(column));
            out.push_column(Column { id: item.output, data_type, nullable })?;
        }
        Ok(out)
    }

    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        lowering.lower_input(node, 0)?;
        let allocator = lowering.allocator();
        let mut programs = SlowVec::fixed(allocator, self.items.len())?;
        for item in self.items.iter() {
            let program = type_expr(allocator, lowering.query_rows(), &item.expr)
                .map_err(|error| lower_error(error, item.output))?;
            programs.push(program)?;
        }
        let compute = DynTransform::new(allocator, Compute { programs })?;
        lowering.add_operator(Step::Transform(compute))
    }
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

    /// Each comparison, as 0 or 1, null where either side is.
    #[test]
    fn comparisons() {
        let cases = [
            (ExprOp::Eq, [0, 1, 0, 0]),
            (ExprOp::Ne, [1, 0, 1, 1]),
            (ExprOp::Lt, [1, 0, 0, 0]),
            (ExprOp::Le, [1, 1, 0, 0]),
            (ExprOp::Gt, [0, 0, 1, 1]),
            (ExprOp::Ge, [0, 1, 1, 1]),
        ];
        for (op, [first, second, fourth, fifth]) in cases {
            let mut b = Builder(StdVec::new());
            let (n, twenty) = (b.push(ExprOp::Column, INT, 0, 0), b.int(20));
            b.push(op, INT, n, twenty);
            let expected = [Some(first), Some(second), None, Some(fourth), Some(fifth)];
            assert_eq!(run(b, batch(&[column(&N)])), expected, "{op:?}");
        }
    }

    /// A string constant is every row's, read from the statement's literals,
    /// and a string null none.
    #[test]
    fn string_constants() {
        let mut literals = crate::string_pool::StringPool::new(&Heap).unwrap();
        let hi = literals.intern_string(&Heap, Some(b"hi")).unwrap();
        let string = DataType::String;
        let query = crate::query_allocators::QueryAllocators::new(&Heap);
        let mut context = Context::for_query(&query, Some(&literals));
        for (op, expected) in [(ExprOp::Constant, Some(&b"hi"[..])), (ExprOp::Null, None)] {
            let mut b = Builder(StdVec::new());
            b.push(op, string, hi.raw_id(), 0);
            let compute = compute([b.program()].into_iter());
            let mut state = compute.new_state(&mut context).unwrap();
            let mut batch = batch(&[column(&N)]);
            compute.process(&mut context, &mut state, &mut batch).unwrap();
            let column = batch.column(1);
            for row in 0..5 {
                let cell = (!column.is_null(row)).then(|| column.string_values().get(row as usize));
                assert_eq!(cell, expected);
            }
        }
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
