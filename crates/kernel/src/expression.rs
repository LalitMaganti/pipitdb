//! Expressions over a row's columns, as Perfetto's: `Expr`, as a plan holds
//! one; `ComputeOp`, the op that adds columns computed by them; and
//! `Compute`, the transform that runs it.

use crate::containers::{BitSlice, SmallVec};
use core::ops::BitOr;

use crate::allocator::AllocError;
use crate::buffer::Buffer;
use crate::column::{ColumnView, DataType, Forms, Validity};
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;
use crate::estimate::Estimate;
use crate::filter::Entries;
use crate::function::ARGUMENTS_MAX;
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{Column, DeriveContext, OutputProperties};
use crate::plan::{ColumnId, Op, PlanNode};
use crate::predicate::{Leaf, Predicate, Set};
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::selection::Selection;
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
    /// Whether the string `a` matches the `LIKE` pattern whose id in the
    /// statement's literals is `b`, as 0 or 1: null if `a` is.
    Like,
    /// Whether `a` is null, as 0 or 1, never null.
    IsNull,
    /// Whether `a` isn't null, as 0 or 1, never null.
    IsNotNull,
    /// Whether `a` is one of the `c` values from value `b` of the
    /// expression's values, as 0 or 1: null if `a` is. They're integers, or
    /// for a string `a`, ids of strings in the statement's literals; in an
    /// `Expr`, its type says which.
    In,
    /// `a AND b`, of integers true where they aren't 0, as 0 or 1: null
    /// only where it's unknown, as SQL's is: `NULL AND 0` is 0.
    And,
    /// `a OR b`: `NULL OR 1` is 1.
    Or,
    /// `NOT a`.
    Not,
    /// A branch of the `Case` after it: its condition `a`, or `PRESENT` to
    /// take the rows where its value isn't null, as `coalesce`'s do, and its
    /// value `b`. Computes nothing itself.
    When,
    /// For each row, the value of the first of the `b` `When`s from step `a`
    /// whose condition isn't null or zero, else `c`. Every step runs for
    /// every row, so what only some rows use mustn't be able to fail.
    Case,
    /// An argument of the `Call` after it: its value `a`, of the type the
    /// function takes. Computes nothing itself.
    Argument,
    /// The statement's function `c`, called with the `b` `Argument`s from
    /// step `a`.
    Call,
}

/// A `When`'s condition for the rows where its value isn't null.
pub const PRESENT: u32 = u32::MAX;

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
    /// The third register it reads, for `Case`, or a `Call`'s function.
    pub c: u32,
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
    pub steps: SmallVec<ExprStep>,
    /// Its `In` lists' values, if it has any.
    pub values: Option<Buffer>,
}

/// Adds one column to each batch for each program. Programs read only the
/// columns the batch came with, which mustn't be lazy, and the batch must
/// have room for a column more for each.
pub struct Compute {
    pub programs: SmallVec<ExprProgram>,
}

/// What a run of a `Compute` keeps between batches.
pub struct ComputeState {
    /// By program, then step, what's made for the step once per run, all
    /// in one list.
    prepared: SmallVec<Prepared>,
    /// By step, what each computed for the program last run.
    registers: SmallVec<Option<ColumnView>>,
}

/// What a run makes for a step once, rather than for each batch.
enum Prepared {
    /// Nothing.
    None,
    /// A constant step's value at every row a batch can have.
    Constant(ColumnView),
    /// An `In` or `Like` step's test, as a predicate of one leaf on its
    /// operand, as a filter tests it, and the sets and entries selecting
    /// with it uses.
    Test { predicate: Predicate, sets: SmallVec<Option<Set>>, entries: SmallVec<Entries> },
}

impl Transform for Compute {
    type State = ComputeState;

    fn new_state(&self, context: &mut Context) -> Result<ComputeState, Error> {
        let env = context.env();
        let (mut steps, mut steps_max) = (0, 0);
        for program in self.programs.iter() {
            steps += program.steps.len();
            steps_max = steps_max.max(program.steps.len());
        }
        let mut prepared = SmallVec::fixed(env, steps)?;
        for program in self.programs.iter() {
            for step in program.steps.iter() {
                prepared.push(prepare(context, program, step)?)?;
            }
        }
        let registers = SmallVec::fixed_from(env, (0..steps_max).map(|_| None))?;
        Ok(ComputeState { prepared, registers })
    }

    fn process(
        &self,
        context: &mut Context,
        state: &mut ComputeState,
        batch: &mut RowBatch,
    ) -> Result<(), Error> {
        let mut start = 0;
        for program in self.programs.iter() {
            let end = start + program.steps.len();
            let prepared = at_mut!(state.prepared, start..end);
            let result = run(context, program, prepared, &mut state.registers, batch)?;
            start = end;
            let Ok(()) = batch.push_column(result) else { failed!() };
        }
        Ok(())
    }
}

/// What a run makes for `step`, one of `program`'s, once.
fn prepare(
    context: &mut Context,
    program: &ExprProgram,
    step: &ExprStep,
) -> Result<Prepared, AllocError> {
    let (column, start, len) = (0, step.b, step.c);
    let strings = || at!(program.steps, step.a as usize).data_type == DataType::String;
    let leaf = match step.op {
        ExprOp::Like => Leaf::Like { column, pattern: Id::raw(step.b) },
        ExprOp::In if strings() => Leaf::InStrings { column, start, len },
        ExprOp::In => Leaf::In { column, start, len },
        _ => return Ok(constant(context, step)?.map_or(Prepared::None, Prepared::Constant)),
    };
    let env = context.env();
    let predicate = Predicate::of_leaf(env, leaf, program.values.clone())?;
    let sets = predicate.new_sets(env, context.literals())?;
    let entries = predicate.new_entries(env)?;
    Ok(Prepared::Test { predicate, sets, entries })
}

/// `step`'s value at every row a batch can have, if it's a constant.
pub(crate) fn constant(
    context: &mut Context,
    step: &ExprStep,
) -> Result<Option<ColumnView>, AllocError> {
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
    prepared: &mut [Prepared],
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
                let Prepared::Constant(constant) = at!(prepared, i) else { failed!() };
                constant.slice(0, rows)
            }
            ExprOp::In | ExprOp::Like => test(context, at_mut!(prepared, i), step, registers)?,
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
            ExprOp::IsNull | ExprOp::IsNotNull => null_test(context, step, registers)?,
            ExprOp::And | ExprOp::Or | ExprOp::Not => logic(context, step, registers)?,
            ExprOp::When | ExprOp::Argument => {
                *at_mut!(registers, i) = None;
                continue;
            }
            ExprOp::Case => {
                let whens = at!(program.steps, step.a as usize..(step.a + step.b) as usize);
                let branch = |k: usize| {
                    let when = at!(whens, k);
                    let condition = (when.a != PRESENT).then(|| register(registers, when.a));
                    (condition, register(registers, when.b))
                };
                pick(context, whens.len(), &branch, register(registers, step.c))?
            }
            ExprOp::Call => call(context, program, step, registers, rows)?,
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

/// Which rows of `columns`, of one batch, are null in none of them, or
/// `None` if none can be: a function's result's, null where an argument is.
pub fn validity_of(
    context: &mut Context,
    columns: &[&ColumnView],
) -> Result<Option<Buffer>, AllocError> {
    let Some(first) = columns.first() else { failed!() };
    if columns.iter().all(|column| column.validity().is_none()) {
        return Ok(None);
    }
    let rows = first.row_count() as usize;
    let mut buffer = context.small_buffer(BitSlice::<u8>::words_for(rows))?;
    for (i, bytes) in buffer.as_mut_slice::<u8>().chunks_mut(8).enumerate() {
        let (at, count) = (i * 64, (rows - i * 64).min(64));
        let word = columns.iter().fold(!0, |w, column| w & word(column.validity(), at, count));
        let word = word.to_le_bytes();
        bytes.copy_from_slice(at!(word, ..bytes.len()));
    }
    Ok(Some(buffer))
}

/// Calls `step`'s function with its arguments' registers.
#[inline(never)]
fn call(
    context: &mut Context,
    program: &ExprProgram,
    step: &ExprStep,
    registers: &[Option<ColumnView>],
    rows: u32,
) -> Result<ColumnView, Error> {
    let count = step.b as usize;
    check!(0 < count && count <= ARGUMENTS_MAX);
    let first = step.a as usize;
    let arguments: [&ColumnView; ARGUMENTS_MAX] = core::array::from_fn(|k| {
        register(registers, at!(program.steps, first + k.min(count - 1)).a)
    });
    let result = context.function(step.c).run(context, at!(arguments, ..count))?;
    check!(result.data_type() == step.data_type && result.row_count() == rows);
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
        ExprOp::Add => {
            each(x, y, o, |a, b| {
                let c = a.wrapping_add(b);
                (c, (a ^ c) & (b ^ c))
            }) >= 0
        }
        ExprOp::Sub => {
            each(x, y, o, |a, b| {
                let c = a.wrapping_sub(b);
                (c, (a ^ b) & (a ^ c))
            }) >= 0
        }
        ExprOp::Mul => {
            each(x, y, o, |a, b| {
                let (c, overflowed) = a.overflowing_mul(b);
                (c, -i64::from(overflowed))
            }) >= 0
        }
        ExprOp::Div => !each(x, y, o, |a, b| a.overflowing_div(nonzero(b))),
        ExprOp::Mod => !each(x, y, o, |a, b| (a.wrapping_rem(nonzero(b)), false)),
        _ => failed!(),
    }
}

fn double_arithmetic(op: ExprOp, x: &[f64], y: &[f64], o: &mut [f64]) {
    match op {
        ExprOp::Add => each(x, y, o, |a, b| (a + b, false)),
        ExprOp::Sub => each(x, y, o, |a, b| (a - b, false)),
        ExprOp::Mul => each(x, y, o, |a, b| (a * b, false)),
        ExprOp::Div => each(x, y, o, |a, b| (a / b, false)),
        _ => failed!(),
    };
}

/// Computes the comparison `step`, as `Eq`, `Ne`, `Lt` or `Le`, so that there
/// are fewer copies of the loops: `a > b` is `b < a`, and `a >= b` is `b <= a`.
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
    ColumnView::new(context, DataType::Int64, values, validity)
}

/// `x op y` at every row, as 0 or 1, for `op` `Eq`, `Ne`, `Lt` or `Le`.
fn compare_numbers<T: PartialOrd + Copy>(op: ExprOp, x: &[T], y: &[T], o: &mut [i64]) {
    match op {
        ExprOp::Eq => compare_each(x, y, o, |a, b| a == b),
        ExprOp::Ne => compare_each(x, y, o, |a, b| a != b),
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
            ExprOp::Ne => order.is_ne(),
            ExprOp::Lt => order.is_lt(),
            _ => order.is_le(),
        });
    }
}

/// Whether each row of `step`'s register is null, or isn't, as 0 or 1.
#[inline(never)]
fn null_test(
    context: &mut Context,
    step: &ExprStep,
    registers: &[Option<ColumnView>],
) -> Result<ColumnView, AllocError> {
    let a = register(registers, step.a);
    let rows = a.row_count() as usize;
    let mut values = context.values_buffer(rows * 8)?;
    present(a.validity(), step.op == ExprOp::IsNull, values.as_mut_slice());
    ColumnView::new(context, DataType::Int64, values, None)
}

/// Whether `step`'s test, `IN` or `LIKE`, is true of its operand at every
/// row, as 0 or 1: null where the operand is. Tested as a filter tests it,
/// by its predicate.
#[inline(never)]
fn test(
    context: &mut Context,
    test: &mut Prepared,
    step: &ExprStep,
    registers: &[Option<ColumnView>],
) -> Result<ColumnView, AllocError> {
    let Prepared::Test { predicate, sets, entries } = test else { failed!() };
    let a = register(registers, step.a);
    let rows = a.row_count();
    let mut operand = RowBatch::new();
    operand.reset(rows);
    operand.push_borrowed([a]);
    predicate.select(context.env(), context.literals(), sets, &mut [], entries, &mut operand)?;
    let mut values = context.values_buffer(rows as usize * 8)?;
    let out = values.as_mut_slice::<i64>();
    out.fill(0);
    let listed = operand.selection();
    for i in 0..listed.len() {
        *at_mut!(out, listed.kept().row(i) as usize) = 1;
    }
    let validity = validity(context, a, None, None)?;
    ColumnView::new(context, DataType::Int64, values, validity)
}

/// `step`'s `AND`, `OR` or `NOT` at every row, as 0 or 1, or null where
/// it's unknown.
#[inline(never)]
fn logic(
    context: &mut Context,
    step: &ExprStep,
    registers: &[Option<ColumnView>],
) -> Result<ColumnView, AllocError> {
    let a = register(registers, step.a);
    let b = (step.op != ExprOp::Not).then(|| register(registers, step.b));
    let rows = a.row_count();
    let truth = |column: &ColumnView, row: u32| {
        (!column.is_null(row)).then(|| *at!(column.int64s(), row as usize) != 0)
    };
    let mut values = context.values_buffer(rows as usize * 8)?;
    let mut validity = context.small_buffer(BitSlice::<u8>::words_for(rows as usize))?;
    validity.as_mut_slice::<u8>().fill(0);
    let known = BitSlice::of_mut(validity.as_mut_slice::<u8>());
    let mut unknown = false;
    for (row, o) in (0..rows).zip(values.as_mut_slice::<i64>()) {
        let (x, y) = (truth(a, row), b.map(|b| truth(b, row)));
        let value = match (step.op, y) {
            (ExprOp::And, Some(y)) if x == Some(false) || y == Some(false) => Some(false),
            (ExprOp::Or, Some(y)) if x == Some(true) || y == Some(true) => Some(true),
            (ExprOp::And | ExprOp::Or, Some(y)) => x.and(y),
            _ => x.map(|x| !x),
        };
        *o = i64::from(value == Some(true));
        unknown |= value.is_none();
        known.set_if(row as usize, value.is_some());
    }
    ColumnView::new(context, DataType::Int64, values, unknown.then_some(validity))
}

/// The branch `k` of a choice: its condition, or `None` to take the rows
/// where its value isn't null, and its value.
type Branch<'r> = dyn Fn(usize) -> (Option<&'r ColumnView>, &'r ColumnView) + 'r;

/// Each row's value from the first of `branches` branches that takes it,
/// else from `otherwise`: a `Case`'s. Going through them
/// from the last, each overwrites the rows it takes, through a mask of 0s
/// and 1s so the loops vectorize, and the first wins; strings keep which
/// branch each row took, then copy the chosen ones' bytes once. Validity is
/// worked out a word at a time.
#[inline(never)]
fn pick<'r>(
    context: &mut Context,
    branches: usize,
    branch: &Branch<'r>,
    otherwise: &'r ColumnView,
) -> Result<ColumnView, AllocError> {
    let rows = otherwise.row_count() as usize;
    let strings = otherwise.data_type() == DataType::String;
    let nullable =
        otherwise.validity().is_some() || (0..branches).any(|k| branch(k).1.validity().is_some());
    // Each row's value, or for strings the branch it took, the `ELSE` last.
    let mut out = context.values_buffer(rows * 8)?;
    if strings {
        #[expect(clippy::cast_possible_wrap, reason = "a step's branches")]
        out.as_mut_slice::<i64>().fill(branches as i64);
    } else {
        out.as_mut_slice::<i64>().copy_from_slice(at!(otherwise.values().words(), ..rows));
    }
    // Each row's presence, a bit each.
    let mut present_out = [0_u64; BATCH_ROWS_MAX as usize / 64];
    let present_out = at_mut!(present_out, ..BitSlice::<u64>::words_for(rows));
    for (i, bits) in present_out.iter_mut().enumerate() {
        *bits = word(otherwise.validity(), i * 64, (rows - i * 64).min(64));
    }
    for k in (0..branches).rev() {
        let (condition, value) = branch(k);
        let tested = condition.unwrap_or(value).validity();
        #[expect(clippy::cast_possible_wrap, reason = "a step's branches")]
        let x = if strings { &[k as i64][..] } else { value.values().words() };
        // 64 rows at a time: the bits of the rows it takes, then its values
        // and presence where they're set.
        for ((i, o), bits) in
            out.as_mut_slice::<i64>().chunks_mut(64).enumerate().zip(&mut *present_out)
        {
            let (at, count) = (i * 64, o.len());
            let mut take = word(tested, at, count);
            if let Some(condition) = condition {
                let values = at!(condition.int64s(), at..at + count);
                take &= values.iter().enumerate().fold(0, |b, (j, &c)| b | u64::from(c != 0) << j);
            }
            blend_bits(x, at, take, o);
            *bits = take & word(value.validity(), at, count) | !take & *bits;
        }
    }
    let validity = if nullable { Some(pack_words(context, rows, present_out)?) } else { None };
    if !strings {
        return ColumnView::new(context, otherwise.data_type(), out, validity);
    }
    let column = |k: usize| if k < branches { branch(k).1 } else { otherwise };
    choose_strings(context, &column, out.as_slice(), validity)
}

/// Writes 1 for each of `o`'s rows `validity` has present, else 0, or the
/// opposite if `null`.
#[inline(never)]
fn present(validity: Option<Validity>, null: bool, o: &mut [i64]) {
    let flip = u64::from(null);
    for (i, o) in o.chunks_mut(64).enumerate() {
        let bits = word(validity, i * 64, o.len());
        for (k, o) in o.iter_mut().enumerate() {
            *o = (bits >> k & 1 ^ flip).cast_signed();
        }
    }
}

/// Sets each of `o`'s rows, from row `at` of `x`, to `x`'s where its bit of
/// `take` is set, keeping it elsewhere. `x` of one value is that value at
/// every row.
#[inline(never)]
fn blend_bits(x: &[i64], at: usize, take: u64, o: &mut [i64]) {
    let mask = |j: usize| -((take >> j & 1).cast_signed());
    if let [x] = *x {
        for (j, o) in o.iter_mut().enumerate() {
            *o ^= (x ^ *o) & mask(j);
        }
        return;
    }
    let x = at!(x, at..at + o.len());
    for (j, (o, &x)) in o.iter_mut().zip(x).enumerate() {
        *o ^= (x ^ *o) & mask(j);
    }
}

/// A validity of `rows` rows from `words`, a bit each.
fn pack_words(context: &mut Context, rows: usize, words: &[u64]) -> Result<Buffer, AllocError> {
    let mut buffer = context.small_buffer(BitSlice::<u8>::words_for(rows))?;
    for (bytes, word) in buffer.as_mut_slice::<u8>().chunks_mut(8).zip(words) {
        let word = word.to_le_bytes();
        bytes.copy_from_slice(at!(word, ..bytes.len()));
    }
    Ok(buffer)
}

/// The strings of a choice, row `r` from `column(which[r])`: the chosen
/// ones' bytes, copied into one buffer. A null's view may not be of its
/// bytes, so it's left empty.
fn choose_strings<'r>(
    context: &mut Context,
    column: &dyn Fn(usize) -> &'r ColumnView,
    which: &[i64],
    validity: Option<Buffer>,
) -> Result<ColumnView, AllocError> {
    let rows = which.len();
    let chosen = |row: usize| {
        let Ok(k) = usize::try_from(*at!(which, row)) else { failed!() };
        let column = column(k);
        #[expect(clippy::cast_possible_truncation, reason = "a batch's rows")]
        let null = column.is_null(row as u32);
        if null { &[][..] } else { column.string_values().get(row) }
    };
    let len: usize = (0..rows).map(|row| chosen(row).len()).sum();
    let mut bytes = context.bytes_buffer(len)?;
    let mut views = context.values_buffer(rows * 8)?;
    let mut end = 0;
    let out = bytes.as_mut_slice::<u8>();
    for (row, view) in views.as_mut_slice::<u32>().as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let string = chosen(row);
        at_mut!(out, end..end + string.len()).copy_from_slice(string);
        #[expect(clippy::cast_possible_truncation, reason = "a batch's strings fit `u32`")]
        let (start, len) = (end as u32, string.len() as u32);
        *view = [start, len];
        end += string.len();
    }
    ColumnView::strings(context, views, bytes, validity)
}

/// The loop for each operation and type, kept to the arithmetic itself so
/// that each copy is small. `f` gives each row's value and whether it
/// overflowed, as a bool or a word's sign: returns them or'd together, so
/// that's checked once at the end rather than at each row.
#[inline(never)]
fn each<T: Copy, F: Copy + Default + BitOr<Output = F>>(
    x: &[T],
    y: &[T],
    o: &mut [T],
    f: impl Fn(T, T) -> (T, F),
) -> F {
    check!(x.len() == o.len() && y.len() == o.len());
    let mut overflows = F::default();
    for ((o, &x), &y) in o.iter_mut().zip(x).zip(y) {
        let (value, overflow) = f(x, y);
        *o = value;
        overflows = overflows | overflow;
    }
    overflows
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
    let mut buffer = context.small_buffer(BitSlice::<u8>::words_for(rows))?;
    let aligned = |validity: Option<Validity>| validity.is_none_or(|v| v.bits().1 == 0);
    if divisor.is_none() && aligned(a) && aligned(b) {
        // Nearly always: only a scan's later batches start mid-byte. The
        // bits past the last row are never read.
        let len = BitSlice::<u8>::words_for(rows);
        let out = buffer.as_mut_slice::<u8>();
        match (a.map(|a| at!(a.bits().0, ..len)), b.map(|b| at!(b.bits().0, ..len))) {
            (Some(x), Some(y)) => {
                for ((o, &x), &y) in out.iter_mut().zip(x).zip(y) {
                    *o = x & y;
                }
            }
            (Some(only), None) | (None, Some(only)) => out.copy_from_slice(only),
            (None, None) => failed!(),
        }
        return Ok(Some(buffer));
    }
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
#[inline(never)]
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
        word | u64::from(BitSlice::of(bits).get(bit)) << k
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
/// references, literals, NULL and arithmetic. Its nodes are in its plan's
/// `Exprs`, from `start`, its root `root` further on, so a part of it, such
/// as a conjunct, is the same `start` with another root.
///
/// It has the shape of an `ExprProgram`: nodes in post-order, each reading
/// only nodes before it, the last being the result. A `Column` node's `a` is
/// a `ColumnId` rather than a batch position, and only a `Constant`'s
/// `data_type` means anything, and an `Argument`'s or a `Call`'s, which are
/// the types its function was bound with: the rest are typed when the plan
/// is lowered. A
/// string `Constant`'s `a` is its id in the statement's literals.
///
/// Only expressions whose meaning is the same as SQLite's are accepted.
/// Arithmetic on strings is an error, and an integer overflow fails the query
/// where SQLite would answer with a double.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Expr {
    pub start: u32,
    pub root: u32,
}

/// The most nodes of all of a plan's expressions.
pub const PLAN_EXPR_NODES_MAX: usize = 1 << 14;

/// The most values of all of a plan's `IN` lists.
pub const PLAN_EXPR_VALUES_MAX: usize = 1 << 20;

/// An expression's nodes, added one at a time, each reading only those
/// before it: the last is its root. Each method that adds a node returns
/// where it is, and is out of line, as building runs once a query.
pub struct ExprBuilder {
    nodes: SmallVec<ExprStep>,
}

impl ExprBuilder {
    pub fn new(env: &Env) -> Result<ExprBuilder, AllocError> {
        Ok(ExprBuilder { nodes: SmallVec::new(env, EXPR_NODES_MAX)? })
    }

    /// Adds `node`. Fails if there are `EXPR_NODES_MAX` already.
    pub fn push(&mut self, node: ExprStep) -> Result<u32, AllocError> {
        self.nodes.push(node)?;
        Ok(self.last())
    }

    /// Adds a node of `op` reading `a` and `b`, as `ExprStep::new`.
    #[inline(never)]
    pub fn node(&mut self, op: ExprOp, a: u32, b: u32) -> Result<u32, AllocError> {
        self.push(ExprStep::new(op, a, b, 0))
    }

    /// Adds column `column`, an `Int64`.
    #[inline(never)]
    pub fn column(&mut self, column: ColumnId) -> Result<u32, AllocError> {
        self.push(ExprStep::new(ExprOp::Column, column, 0, 0))
    }

    /// Adds the integer `value`.
    #[inline(never)]
    pub fn int(&mut self, value: i64) -> Result<u32, AllocError> {
        self.push(ExprStep::new(ExprOp::Constant, 0, 0, value))
    }

    /// Where the last node added is.
    pub fn last(&self) -> u32 {
        #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
        let last = self.nodes.len() as u32 - 1;
        last
    }

    /// Whether another node would be one too many.
    pub fn is_full(&self) -> bool {
        self.nodes.len() == EXPR_NODES_MAX
    }

    /// Takes the last node off.
    pub fn pop(&mut self) -> Option<ExprStep> {
        self.nodes.pop()
    }

    pub fn nodes(&self) -> &[ExprStep] {
        &self.nodes
    }

    pub fn nodes_mut(&mut self) -> &mut [ExprStep] {
        &mut self.nodes
    }
}

/// The nodes of a plan's expressions, each one's together, and the values
/// of their `IN` lists, which an `In` node's `b` counts from the first of.
pub struct Exprs {
    nodes: SmallVec<ExprStep>,
    values: SmallVec<i64>,
}

impl Exprs {
    pub fn new(env: &Env) -> Result<Exprs, AllocError> {
        let values = SmallVec::new(env, PLAN_EXPR_VALUES_MAX)?;
        Ok(Exprs { nodes: SmallVec::new(env, PLAN_EXPR_NODES_MAX)?, values })
    }

    /// Adds the expression of `nodes`, which read only each other, the last
    /// its root.
    pub fn add(&mut self, nodes: &[ExprStep]) -> Result<Expr, AllocError> {
        check!(!nodes.is_empty() && nodes.len() <= EXPR_NODES_MAX);
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_EXPR_NODES_MAX`")]
        let start = self.nodes.len() as u32;
        self.nodes.extend_from_slice(nodes)?;
        #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
        let root = nodes.len() as u32 - 1;
        Ok(Expr { start, root })
    }

    /// Adds a value of an `IN` list, and returns where it is.
    pub fn add_value(&mut self, value: i64) -> Result<u32, AllocError> {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_EXPR_VALUES_MAX`")]
        let at = self.values.len() as u32;
        self.values.push(value)?;
        Ok(at)
    }

    /// `expr`'s nodes, from its start to its root, the last.
    pub fn nodes(&self, expr: Expr) -> &[ExprStep] {
        at!(self.nodes, expr.start as usize..=(expr.start + expr.root) as usize)
    }

    /// The values of the `IN` lists.
    pub fn values(&self) -> &[i64] {
        &self.values
    }
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
    /// A double in a list of integers, as `IN`'s.
    DoubleIn,
    /// A string, not a number, on the left of `LIKE`.
    Strings,
    /// Integer arithmetic in an argument of `coalesce`, or a branch of
    /// `CASE`, which only some rows use, which could overflow on the others.
    SomeRows,
    /// A number, not a string, as a condition of `CASE`, or an integer as
    /// an operand of `AND`, `OR` or `NOT`.
    Condition,
    /// Arguments of `coalesce`, or values of `CASE`, of one type.
    Choices,
    /// An argument of the type its function takes.
    Arguments,
    /// An allocation failed.
    OutOfMemory,
}

impl From<AllocError> for ExprError {
    fn from(_: AllocError) -> ExprError {
        ExprError::OutOfMemory
    }
}

/// How an op whose operands are typed alike is typed.
#[derive(Clone, Copy)]
struct Rule {
    /// The error if an operand is a string, then if one is a double.
    errors: [Option<ExprError>; 2],
    /// Whether a NULL operand makes it NULL.
    null_is_null: bool,
    /// Whether it's an integer, rather than of its operands' type.
    integer: bool,
}

const OP_COUNT: usize = ExprOp::Call as usize + 1;

/// By op, how it's typed, if by a `Rule`.
static RULES: [Option<Rule>; OP_COUNT] = rules();

const fn rules() -> [Option<Rule>; OP_COUNT] {
    let arithmetic =
        Rule { errors: [Some(ExprError::Numbers), None], null_is_null: true, integer: false };
    let condition = Some(ExprError::Condition);
    let logic = Rule { errors: [condition, condition], null_is_null: false, integer: true };
    let test = Rule { errors: [None, None], null_is_null: false, integer: true };
    let mut rules = [None; OP_COUNT];
    rules[ExprOp::Add as usize] = Some(arithmetic);
    rules[ExprOp::Sub as usize] = Some(arithmetic);
    rules[ExprOp::Mul as usize] = Some(arithmetic);
    rules[ExprOp::Div as usize] = Some(arithmetic);
    rules[ExprOp::Mod as usize] =
        Some(Rule { errors: [Some(ExprError::Numbers), Some(ExprError::Integers)], ..arithmetic });
    rules[ExprOp::And as usize] = Some(logic);
    rules[ExprOp::Or as usize] = Some(logic);
    rules[ExprOp::Not as usize] = Some(logic);
    rules[ExprOp::IsNull as usize] = Some(test);
    rules[ExprOp::IsNotNull as usize] = Some(test);
    rules
}

/// A step of no value, reading registers `a` and `b`.
const fn step(op: ExprOp, data_type: DataType, a: u32, b: u32) -> ExprStep {
    ExprStep { op, data_type, a, b, c: 0, int_value: 0, double_value: 0.0 }
}

/// Types an expression's nodes in order and, given the rows it reads,
/// appends the steps computing them to a program. Without them it only
/// checks, as compiling does.
struct Typer<'t> {
    /// Each column's type.
    type_of: &'t dyn Fn(ColumnId) -> DataType,
    /// The rows the program reads, and its steps so far, if building one.
    build: Option<(&'t OutputProperties, &'t mut SmallVec<ExprStep>)>,
    /// By node, its type, the register holding it, and whether computing it
    /// can fail. A bare NULL has no type of its own: it takes the type of
    /// whatever it meets.
    nodes: [(Option<DataType>, u32, bool); EXPR_NODES_MAX],
}

impl Typer<'_> {
    /// The type of the expression's result.
    #[inline(never)]
    fn run(&mut self, nodes: &[ExprStep]) -> Result<Option<DataType>, ExprError> {
        check!(!nodes.is_empty() && nodes.len() <= EXPR_NODES_MAX);
        // Only what the root reads: the nodes before it can be others'.
        let reached = reached(nodes);
        for (i, node) in nodes.iter().enumerate() {
            if *at!(reached, i) {
                let (data_type, register) = self.type_node(nodes, node)?;
                let fails = self.can_fail(nodes, node, data_type);
                *at_mut!(self.nodes, i) = (data_type, register, fails);
            }
        }
        Ok(at!(self.nodes, nodes.len() - 1).0)
    }

    /// The register holding node `i` as `data_type`: a null of that type for
    /// a bare NULL, an integer widened for a double.
    fn materialize(&mut self, i: u32, data_type: DataType) -> Result<u32, ExprError> {
        let (node_type, register, _) = *at!(self.nodes, i as usize);
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
            // Its `Case` types a `When`.
            ExprOp::Null | ExprOp::When => Ok((None, 0)),
            ExprOp::Argument | ExprOp::Call => self.type_call(node),
            ExprOp::ToDouble => failed!(),
            ExprOp::Eq | ExprOp::Ne | ExprOp::Lt | ExprOp::Le | ExprOp::Gt | ExprOp::Ge => {
                self.type_comparison(nodes, node)
            }
            ExprOp::Like => {
                let (a_type, a, _) = *at!(self.nodes, node.a as usize);
                match a_type {
                    // `NULL LIKE p` is NULL.
                    None => Ok((None, 0)),
                    Some(DataType::String) => {
                        let like = step(ExprOp::Like, DataType::Int64, a, node.b);
                        Ok((Some(DataType::Int64), self.add(like)?))
                    }
                    Some(_) => Err(ExprError::Strings),
                }
            }
            ExprOp::Case => self.type_case(nodes, node),
            ExprOp::In => self.type_in(node),
            ExprOp::Add
            | ExprOp::Sub
            | ExprOp::Mul
            | ExprOp::Div
            | ExprOp::Mod
            | ExprOp::And
            | ExprOp::Or
            | ExprOp::Not
            | ExprOp::IsNull
            | ExprOp::IsNotNull => self.type_by_rule(node),
        }
    }

    /// Types `node` by its op's `Rule`: its operands, which are of one
    /// type, a double where any is, or an integer if all are NULL.
    fn type_by_rule(&mut self, node: &ExprStep) -> Result<(Option<DataType>, u32), ExprError> {
        let Some(rule) = *at!(RULES, node.op as usize) else { failed!() };
        let [(a, _), (b, b_len)] = node.operands();
        let both = [a, b];
        let operands = at!(both, ..=b_len as usize);
        let mut types = 0_u8;
        let mut null = false;
        for &operand in operands {
            match at!(self.nodes, operand as usize).0 {
                Some(data_type) => types |= 1 << data_type as u8,
                None => null = true,
            }
        }
        for (data_type, error) in [DataType::String, DataType::Float64].into_iter().zip(rule.errors)
        {
            if types >> data_type as u8 & 1 != 0
                && let Some(error) = error
            {
                return Err(error);
            }
        }
        if null && rule.null_is_null {
            return Ok((None, 0));
        }
        let data_type = match types {
            _ if types >> DataType::Float64 as u8 & 1 != 0 => DataType::Float64,
            _ if types >> DataType::String as u8 & 1 != 0 => DataType::String,
            _ => DataType::Int64,
        };
        let mut registers = [0; 2];
        for (register, &operand) in registers.iter_mut().zip(operands) {
            *register = self.materialize(operand, data_type)?;
        }
        let result = if rule.integer { DataType::Int64 } else { data_type };
        let [a, b] = registers;
        Ok((Some(result), self.add(step(node.op, result, a, b))?))
    }

    /// Types `node`, an `In`, whose operand must be of its list's kind:
    /// integers, or strings.
    fn type_in(&mut self, node: &ExprStep) -> Result<(Option<DataType>, u32), ExprError> {
        let (a_type, a, _) = *at!(self.nodes, node.a as usize);
        match a_type {
            // `NULL IN (...)` is NULL.
            None => Ok((None, 0)),
            Some(data_type) if data_type == node.data_type => {
                let mut test = step(ExprOp::In, DataType::Int64, a, node.b);
                test.c = node.c;
                Ok((Some(DataType::Int64), self.add(test)?))
            }
            Some(DataType::Float64) if node.data_type == DataType::Int64 => {
                Err(ExprError::DoubleIn)
            }
            Some(_) => Err(ExprError::Kinds),
        }
    }

    /// Whether computing `node`, one of `nodes`, of type `data_type`, can
    /// fail: only integer arithmetic can, or what reads it.
    fn can_fail(&self, nodes: &[ExprStep], node: &ExprStep, data_type: Option<DataType>) -> bool {
        let fails = |i: u32| at!(self.nodes, i as usize).2;
        match node.op {
            ExprOp::Add | ExprOp::Sub | ExprOp::Mul | ExprOp::Div | ExprOp::Mod
                if data_type == Some(DataType::Int64) =>
            {
                true
            }
            // Only its first condition is computed for every row: a `PRESENT`
            // one tests its value.
            ExprOp::Case => {
                let first = at!(nodes, node.a as usize);
                fails(if first.a == PRESENT { first.b } else { first.a })
            }
            // Its `Case` reads what it does.
            ExprOp::When => false,
            _ => node.operands().into_iter().any(|(start, len)| (start..start + len).any(fails)),
        }
    }

    /// Types `node`, a `Case` of the `When`s from node `a`, one of `nodes`:
    /// its values and `ELSE` of one type, `NULL` going with any, and its
    /// conditions numbers. Every step runs for every row, so only its first
    /// condition, which every row is tested by, can fail: a `PRESENT` one
    /// tests its value, as `coalesce`'s first argument.
    fn type_case(
        &mut self,
        nodes: &[ExprStep],
        node: &ExprStep,
    ) -> Result<(Option<DataType>, u32), ExprError> {
        let whens = at!(nodes, node.a as usize..(node.a + node.b) as usize);
        check!(!whens.is_empty() && whens.iter().all(|when| when.op == ExprOp::When));
        let at = |i: u32| *at!(self.nodes, i as usize);
        for (k, when) in whens.iter().enumerate() {
            let tested = if when.a == PRESENT { when.b } else { when.a };
            if (k > 0 && at(tested).2) || (tested != when.b && at(when.b).2) {
                return Err(ExprError::SomeRows);
            }
            if when.a != PRESENT && at(when.a).0 == Some(DataType::String) {
                return Err(ExprError::Condition);
            }
        }
        if at(node.c).2 {
            return Err(ExprError::SomeRows);
        }
        let types = whens.iter().map(|when| at(when.b).0).chain([at(node.c).0]);
        let Some(data_type) = one_type(types)? else { return Ok((None, 0)) };
        // Each branch's condition and value, then its `When`s together, as
        // the `Case` reads them.
        let mut branches = [(0, 0); EXPR_NODES_MAX];
        for (k, when) in whens.iter().enumerate() {
            let condition = if when.a == PRESENT { PRESENT } else { self.condition(when.a)? };
            *at_mut!(branches, k) = (condition, self.materialize(when.b, data_type)?);
        }
        let otherwise = self.materialize(node.c, data_type)?;
        let mut first = None;
        for &(condition, value) in at!(branches, ..whens.len()) {
            let when = self.add(step(ExprOp::When, data_type, condition, value))?;
            first.get_or_insert(when);
        }
        let mut case = step(ExprOp::Case, data_type, first.unwrap_or(0), node.b);
        case.c = otherwise;
        Ok((Some(data_type), self.add(case)?))
    }

    /// Types `node`, an `Argument`, which must be of the type its function
    /// takes, or an integer for a double, or `NULL`; or a `Call` of the
    /// `Argument`s from node `a`, which it adds the steps of together.
    fn type_call(&mut self, node: &ExprStep) -> Result<(Option<DataType>, u32), ExprError> {
        let (a, b) = (node.a as usize, node.b as usize);
        if node.op == ExprOp::Argument {
            let takes = |data_type| data_type == node.data_type;
            match at!(self.nodes, a).0 {
                Some(DataType::Int64) if takes(DataType::Float64) => {}
                Some(data_type) if !takes(data_type) => return Err(ExprError::Arguments),
                _ => {}
            }
            return Ok((Some(node.data_type), self.materialize(node.a, node.data_type)?));
        }
        let mut first = None;
        for k in a..a + b {
            let (Some(data_type), register, _) = *at!(self.nodes, k) else { failed!() };
            first.get_or_insert(self.add(step(ExprOp::Argument, data_type, register, 0))?);
        }
        let mut call = step(ExprOp::Call, node.data_type, first.unwrap_or(0), node.b);
        call.c = node.c;
        Ok((Some(node.data_type), self.add(call)?))
    }

    /// The register holding node `i` as an Int64 condition: a double is true
    /// unless it's zero.
    fn condition(&mut self, i: u32) -> Result<u32, ExprError> {
        let (data_type, register, _) = *at!(self.nodes, i as usize);
        if data_type != Some(DataType::Float64) {
            return self.materialize(i, DataType::Int64);
        }
        let zero = self.add(step(ExprOp::Constant, DataType::Float64, 0, 0))?;
        self.add(step(ExprOp::Ne, DataType::Int64, register, zero))
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

/// The one type of `types`, `None` for NULL going with any: `None` if all are,
/// and an error if two differ.
fn one_type(types: impl Iterator<Item = Option<DataType>>) -> Result<Option<DataType>, ExprError> {
    let mut one = None;
    for data_type in types.flatten() {
        if one.is_some_and(|one| one != data_type) {
            return Err(ExprError::Choices);
        }
        one = Some(data_type);
    }
    Ok(one)
}

/// The type `expr` computes, given each column's type, as compiling checks
/// it.
/// Whether `expr` can compute a null, given whether each column it reads
/// can hold one: it can if one can, or if it divides by what may be zero,
/// anything but a constant that isn't, or has a `NULL`.
pub fn expr_can_be_null(nodes: &[ExprStep], nullable_of: &dyn Fn(ColumnId) -> bool) -> bool {
    // A null test never is, whatever it tests.
    if nodes.last().is_some_and(|node| matches!(node.op, ExprOp::IsNull | ExprOp::IsNotNull)) {
        return false;
    }
    let nonzero = |node: &ExprStep| {
        node.op == ExprOp::Constant && (node.int_value != 0 || node.double_value != 0.0)
    };
    nodes.iter().any(|node| match node.op {
        ExprOp::Column => nullable_of(node.a),
        ExprOp::Div | ExprOp::Mod => nodes.get(node.b as usize).is_none_or(|b| !nonzero(b)),
        ExprOp::Null => true,
        _ => false,
    })
}

pub fn check_expr_type(
    nodes: &[ExprStep],
    type_of: &dyn Fn(ColumnId) -> DataType,
) -> Result<DataType, ExprError> {
    let mut typer = Typer { type_of, build: None, nodes: [(None, 0, false); EXPR_NODES_MAX] };
    // A result which is only ever NULL is a column of Int64 nulls.
    Ok(typer.run(nodes)?.unwrap_or(DataType::Int64))
}

impl ExprStep {
    /// A node of `op`, an `Int64`, reading nodes `a` and `b`, or of value
    /// `int_value`.
    pub const fn new(op: ExprOp, a: u32, b: u32, int_value: i64) -> ExprStep {
        ExprStep { op, data_type: DataType::Int64, a, b, c: 0, int_value, double_value: 0.0 }
    }

    /// The nodes this reads, as `a`, `b` and `c` say by its op: the first
    /// of a run, and how many, or one.
    pub(crate) fn operands(&self) -> [(u32, u32); 2] {
        let one = |node| (node, 1);
        let none = (0, 0);
        match self.op {
            ExprOp::Column | ExprOp::Constant | ExprOp::Null => [none, none],
            ExprOp::ToDouble
            | ExprOp::Like
            | ExprOp::IsNull
            | ExprOp::IsNotNull
            | ExprOp::Not
            | ExprOp::In
            | ExprOp::Argument => [one(self.a), none],
            ExprOp::When if self.a == PRESENT => [one(self.b), none],
            ExprOp::Add
            | ExprOp::Sub
            | ExprOp::Mul
            | ExprOp::Div
            | ExprOp::Mod
            | ExprOp::Eq
            | ExprOp::Ne
            | ExprOp::Lt
            | ExprOp::Le
            | ExprOp::Gt
            | ExprOp::Ge
            | ExprOp::And
            | ExprOp::Or
            | ExprOp::When => [one(self.a), one(self.b)],
            ExprOp::Case => [(self.a, self.b), one(self.c)],
            ExprOp::Call => [(self.a, self.b), none],
        }
    }
}

/// The nodes the last of `nodes` reads, and it: marked from the last back,
/// as a node's operands come before it.
fn reached(nodes: &[ExprStep]) -> [bool; EXPR_NODES_MAX] {
    let mut reached = [false; EXPR_NODES_MAX];
    *at_mut!(reached, nodes.len() - 1) = true;
    for (i, node) in nodes.iter().enumerate().rev() {
        if *at!(reached, i) {
            for (start, len) in node.operands() {
                at_mut!(reached, start as usize..(start + len) as usize).fill(true);
            }
        }
    }
    reached
}

/// Node `node`'s top-level conjuncts, the parts of its `AND`s, of `nodes`:
/// each passed to `each`.
pub fn conjuncts(nodes: &[ExprStep], node: u32, each: &mut dyn FnMut(u32)) {
    let step = at!(nodes, node as usize);
    if step.op == ExprOp::And {
        conjuncts(nodes, step.a, each);
        conjuncts(nodes, step.b, each);
    } else {
        each(node);
    }
}

/// Passes each column node `node` of `nodes` reads to `each`, maybe more
/// than once.
pub fn columns_of(nodes: &[ExprStep], node: u32, each: &mut dyn FnMut(ColumnId)) {
    let step = at!(nodes, node as usize);
    if step.op == ExprOp::Column {
        return each(step.a);
    }
    for (first, count) in step.operands() {
        for operand in first..first + count {
            columns_of(nodes, operand, each);
        }
    }
}

/// Marks the columns `expr` reads.
pub fn mark_expr_columns(expr: Expr, needed: &mut Needed<'_>) {
    let nodes = needed.exprs().nodes(expr);
    columns_of(nodes, expr.root, &mut |column| needed.need(column));
}

/// The program computing the expression of `nodes`, the last its root, with
/// columns at their positions in `rows`, and its `IN` lists of `values`. As a
/// `condition`, its result is an integer: a double is true unless it's 0.
pub fn type_expr(
    env: &Env,
    rows: &OutputProperties,
    nodes: &[ExprStep],
    values: Option<Buffer>,
    condition: bool,
) -> Result<ExprProgram, ExprError> {
    // Each node takes at most three steps, and a NULL result one more.
    let mut steps = SmallVec::new(env, 4 * EXPR_NODES_MAX)?;
    let type_of = |column| rows.type_of(column);
    let table = [(None, 0, false); EXPR_NODES_MAX];
    let mut typer = Typer { type_of: &type_of, build: Some((rows, &mut steps)), nodes: table };
    let result = typer.run(nodes)?.unwrap_or(DataType::Int64);
    #[expect(clippy::cast_possible_truncation, reason = "at most `EXPR_NODES_MAX`")]
    let root = nodes.len() as u32 - 1;
    let register =
        if condition { typer.condition(root)? } else { typer.materialize(root, result)? };
    // Unlike Perfetto's, no column is only ever NULL, so the last node's step
    // is the last.
    check!(register as usize == steps.len() - 1);
    Ok(ExprProgram { steps, values })
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
    pub items: SmallVec<ComputeItem>,
}

/// An error typing the expression of `column`, as lowering reports it.
fn lower_error(error: ExprError, column: ColumnId) -> LowerError {
    match error {
        ExprError::OutOfMemory => LowerError::OutOfMemory,
        ExprError::Numbers
        | ExprError::Integers
        | ExprError::Kinds
        | ExprError::Types
        | ExprError::Inexact
        | ExprError::DoubleIn
        | ExprError::Strings
        | ExprError::SomeRows
        | ExprError::Condition
        | ExprError::Choices
        | ExprError::Arguments => LowerError::WrongType(column),
    }
}

impl<'c> Op<'c> for ComputeOp {
    /// A condition on a column it doesn't compute can be applied to its
    /// input: it neither drops nor adds rows.
    fn moves_condition(&self, column: ColumnId) -> Option<u32> {
        (!self.items.iter().any(|item| item.output == column)).then_some(0)
    }

    /// The same as its input's: it keeps every row, in order.
    fn estimate(&self, _: &Exprs, inputs: &[Estimate]) -> Option<Estimate> {
        Some(inputs.first().map_or(Estimate::UNKNOWN, |input| input.after_filters()))
    }

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
            mark_expr_columns(item.expr, reads);
        }
    }

    /// Its input's, with each item's column after them.
    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        let input = c.input(0);
        let mut out = *input;
        let exprs = c.exprs();
        for item in self.items.iter() {
            let nodes = exprs.nodes(item.expr);
            let data_type = check_expr_type(nodes, &|column| input.type_of(column))
                .map_err(|error| lower_error(error, item.output))?;
            let nullable = expr_can_be_null(nodes, &|column| input.nullable_of(column));
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
        let env = lowering.env();
        let mut programs = SmallVec::fixed(env, self.items.len())?;
        for item in self.items.iter() {
            let (nodes, values) = (lowering.exprs().nodes(item.expr), lowering.expr_values());
            let program = type_expr(env, lowering.query_rows(), nodes, values, false)
                .map_err(|error| lower_error(error, item.output))?;
            programs.push(program)?;
        }
        let compute = DynTransform::new(env, Compute { programs })?;
        lowering.add_operator(Step::Transform(compute))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec as StdVec;

    use super::*;

    use crate::env::Env;

    /// A batch's `n`, with a null.
    const N: [Option<i64>; 5] = [Some(10), Some(20), None, Some(40), Some(50)];
    /// A batch's `u`, with zeros.
    const U: [Option<i64>; 5] = [Some(1), Some(0), Some(3), Some(0), Some(5)];

    fn column(cells: &[Option<i64>]) -> ColumnView {
        let context = &mut Context::new(Env::current(), None);
        ColumnView::from_int64s(context, cells.iter().copied()).unwrap()
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
            let step = ExprStep { op, data_type, a, b, c: 0, int_value: 0, double_value: 0.0 };
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
            ExprProgram {
                steps: SmallVec::fixed_from(Env::current(), self.0.into_iter()).unwrap(),
                values: None,
            }
        }
    }

    fn compute(programs: impl ExactSizeIterator<Item = ExprProgram>) -> Compute {
        Compute { programs: SmallVec::fixed_from(Env::current(), programs).unwrap() }
    }

    /// `program`'s column of `batch`, at its kept rows, as words, or the
    /// run's error.
    fn process(program: Builder, mut batch: RowBatch) -> Result<StdVec<Option<i64>>, Error> {
        let compute = compute([program.program()].into_iter());
        let mut context = Context::new(Env::current(), None);
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

    /// Both sides' nulls, each starting on a byte, as most batches' do.
    #[test]
    fn nulls_on_both_sides() {
        let mut b = Builder(StdVec::new());
        let (n, m) = (b.push(ExprOp::Column, INT, 0, 0), b.push(ExprOp::Column, INT, 1, 0));
        b.push(ExprOp::Add, INT, n, m);
        let m = column(&[None, Some(1), Some(2), Some(3), None]);
        assert_eq!(run(b, batch(&[column(&N), m])), [None, Some(21), None, Some(43), None]);
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

    /// `coalesce`, a `PRESENT` branch, fills nulls from its second argument,
    /// and `CASE` takes each row from its first branch whose condition is
    /// true, else its `ELSE`.
    #[test]
    fn choices() {
        let mut c = Builder(StdVec::new());
        let (n, u) = (c.push(ExprOp::Column, INT, 0, 0), c.push(ExprOp::Column, INT, 1, 0));
        let when = c.push(ExprOp::When, INT, PRESENT, n);
        let at = c.push(ExprOp::Case, INT, when, 1);
        c.0[at as usize].c = u;
        let cells = run(c, batch(&[column(&N), column(&U)]));
        assert_eq!(cells, [Some(10), Some(20), Some(3), Some(40), Some(50)]);

        // `CASE WHEN u THEN n ELSE u END`, then with a second branch,
        // `WHEN n THEN 7`, for the rows the first doesn't take.
        for (second, expected) in [(false, [10, 0, -1, 0, 50]), (true, [10, 7, -1, 7, 50])] {
            let mut i = Builder(StdVec::new());
            let (n, u) = (i.push(ExprOp::Column, INT, 0, 0), i.push(ExprOp::Column, INT, 1, 0));
            let seven = i.int(7);
            let first = i.push(ExprOp::When, INT, u, n);
            if second {
                i.push(ExprOp::When, INT, n, seven);
            }
            let at = i.push(ExprOp::Case, INT, first, 1 + u32::from(second));
            i.0[at as usize].c = u;
            let cells = run(i, batch(&[column(&N), column(&U)]));
            let expected = expected.map(|cell| (cell >= 0).then_some(cell));
            assert_eq!(cells, expected, "{second}");
        }
    }

    /// Whether each row is null, or isn't, as 0 or 1, never null.
    #[test]
    fn null_tests() {
        for (op, expected) in
            [(ExprOp::IsNull, [0, 0, 1, 0, 0]), (ExprOp::IsNotNull, [1, 1, 0, 1, 1])]
        {
            let mut b = Builder(StdVec::new());
            let n = b.push(ExprOp::Column, INT, 0, 0);
            b.push(op, INT, n, 0);
            assert_eq!(run(b, batch(&[column(&N)])), expected.map(Some), "{op:?}");
        }
    }

    /// A string constant is every row's, read from the statement's literals,
    /// and a string null none.
    #[test]
    fn string_constants() {
        let mut statement = crate::statement::Statement::default();
        let hi = statement.add_literal(Env::current(), b"hi").unwrap();
        let string = DataType::String;
        let mut context = Context::new(Env::current(), Some(&statement));
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
        let mut context = Context::new(Env::current(), None);
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
