//! `Flushed`: a breaker run in its input's pipeline, which outputs and drops
//! what it holds each time its input says no row so far gets another, so it
//! holds only the rows since.

use crate::allocator::AllocError;
use crate::column::Forms;
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;
use crate::row_batch::RowBatch;
use crate::step::{Breaker, DynOperator, Operator, Progress};

/// Where a key's values say the rows before can be flushed.
#[derive(PartialEq, Eq)]
pub enum FlushAt {
    /// Where a batch's first value of the key differs from the last before
    /// it, as the rows are grouped by it.
    Grouped,
    /// Where a batch's bounds are above every value before it, of a scan
    /// whose runs follow on, as `Summary` says.
    Runs,
}

/// The operator that runs `breaker`, flushing it at `at` of the key at `key`,
/// an `Int64` without nulls.
pub fn flushed<'a, B: Breaker + 'a>(
    env: &Env,
    breaker: B,
    key: u32,
    at: FlushAt,
) -> Result<DynOperator<'a>, AllocError> {
    DynOperator::new(env, Flushed { breaker, key, at })
}

struct Flushed<B> {
    /// The breaker.
    breaker: B,
    /// Where the key is in batches.
    key: u32,
    /// Where to flush.
    at: FlushAt,
}

struct FlushedState<S> {
    /// The breaker's state, of the rows since the last flush, if any.
    breaker: Option<S>,
    /// Whether `breaker` is finalized, and being served.
    serving: bool,
    /// The last value of the key consumed, if `Grouped`, or at least every
    /// one, if `Runs`.
    last: i64,
}

impl<B: Breaker> Operator for Flushed<B> {
    const NAME: &'static str = "Flushed";
    type State = FlushedState<B::State>;

    fn new_state(&self, _: &mut Context) -> Result<Self::State, Error> {
        Ok(FlushedState { breaker: None, serving: false, last: i64::MIN })
    }

    fn execute(
        &self,
        context: &mut Context,
        state: &mut Self::State,
        input: &RowBatch,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        if state.serving || self.reaches_past(context, state, input)? {
            if self.flush(context, state, output)? {
                return Ok(Progress::MoreOutput);
            }
            state.breaker = None;
            state.serving = false;
        }
        let breaker = match &mut state.breaker {
            Some(breaker) => breaker,
            None => state.breaker.insert(self.breaker.new_state(context)?),
        };
        self.breaker.consume(context, breaker, input)?;
        Ok(Progress::NeedInput)
    }

    fn finish(
        &self,
        context: &mut Context,
        state: &mut Self::State,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        if self.flush(context, state, output)? {
            Ok(Progress::MoreOutput)
        } else {
            Ok(Progress::Done)
        }
    }
}

impl<B: Breaker> Flushed<B> {
    /// Whether no row of `input` or after reaches a group held, so they
    /// can be flushed first.
    fn reaches_past(
        &self,
        context: &mut Context,
        state: &mut FlushedState<B::State>,
        input: &RowBatch,
    ) -> Result<bool, Error> {
        let kept = input.selection().kept();
        let len = input.selection().len();
        let key = input.column(self.key);
        if self.at == FlushAt::Runs {
            let (min, max) = key.bounds().map_or((i64::MIN, i64::MAX), |b| (b.min, b.max));
            let flush = min > state.last;
            state.last = state.last.max(max);
            return Ok(flush);
        }
        let mut key = key.clone();
        key.make_in(context, Forms::FLAT)?;
        let values = key.int64s();
        let first = *at!(values, kept.row(0) as usize);
        let flush = first != state.last;
        state.last = *at!(values, kept.row(len - 1) as usize);
        Ok(flush)
    }

    /// Serves what the breaker holds, if anything, into `output`, finalizing
    /// it first if it isn't yet, returning whether more is left.
    fn flush(
        &self,
        context: &mut Context,
        state: &mut FlushedState<B::State>,
        output: &mut RowBatch,
    ) -> Result<bool, Error> {
        let Some(breaker) = &mut state.breaker else { return Ok(false) };
        if !state.serving {
            self.breaker.finalize(context, breaker)?;
            state.serving = true;
        }
        self.breaker.serve(context, breaker, output)
    }
}
