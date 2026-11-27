//! `sort`: the breaker that stably sorts rows by their keys, as Perfetto's
//! `core/exec/sort` does. Nulls sort as in SQLite: first ascending, last
//! descending.

use crate::containers::SmallVec;
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::RowBatch;
pub use crate::sorted_rows::SortKey;
use crate::sorted_rows::SortedRows;
use crate::step::Breaker;

#[cfg(feature = "spill")]
mod spill;

/// The breaker that sorts rows by `keys`, the first first, which must be
/// columns of numbers.
pub fn sort(keys: SmallVec<SortKey>) -> Sort {
    Sort { keys }
}

pub struct Sort {
    keys: SmallVec<SortKey>,
}

pub struct SortState {
    /// The rows held, with what sorting them takes.
    rows: SortedRows,
    /// How many of the rows held batches have been applied to: a batch
    /// that failed to be held may have left some of its rows after them.
    #[cfg(feature = "spill")]
    applied: u32,
    /// The runs of rows sorted and written out, to merge once all are in.
    #[cfg(feature = "spill")]
    runs: spill::Runs,
}

impl Breaker for Sort {
    const NAME: &'static str = "Sort";
    type State = SortState;

    fn new_state(&self, context: &mut Context) -> Result<SortState, Error> {
        Ok(SortState {
            rows: SortedRows::new(context.env())?,
            #[cfg(feature = "spill")]
            applied: 0,
            #[cfg(feature = "spill")]
            runs: spill::Runs::new(context.env()),
        })
    }

    fn consume(
        &self,
        context: &mut Context,
        state: &mut SortState,
        input: &RowBatch,
    ) -> Result<(), Error> {
        let count = input.selection().len();
        if count == 0 {
            return Ok(());
        }
        // Room to sort the rows held with the batch's is made as they're
        // held, so the sort can sort them without allocating.
        let env = context.env();
        state.rows.check_keys(env, input, &self.keys)?;
        state.rows.reserve(env, count)?;
        #[cfg(feature = "spill")]
        state.runs.reserve(env)?;
        state.rows.append(context, input, &self.keys)?;
        #[cfg(feature = "spill")]
        context.env().applying(|| state.applied = state.rows.len());
        Ok(())
    }

    fn finalize(&self, context: &mut Context, state: &mut SortState) -> Result<(), Error> {
        #[cfg(feature = "spill")]
        if state.runs.any() {
            spill::write_run(context.env(), state)?;
            state.rows = SortedRows::new(context.env())?;
            return state.runs.finish(context, &self.keys);
        }
        state.rows.sort(context)?;
        state.rows.scatter(context)
    }

    fn serve(
        &self,
        context: &mut Context,
        state: &mut SortState,
        output: &mut RowBatch,
    ) -> Result<bool, Error> {
        #[cfg(feature = "spill")]
        if state.runs.any() {
            return state.runs.serve(context, &self.keys, output);
        }
        state.rows.serve(context, output)
    }

    #[cfg(feature = "spill")]
    fn spill(&self, env: &crate::env::Env, state: &mut SortState) -> Result<bool, Error> {
        spill::write_run(env, state)
    }
}

#[cfg(test)]
mod tests {
    use crate::containers::BitSlice;
    use crate::pipeline::tests::after_breaker;
    use alloc::vec::Vec;

    use super::*;

    use crate::buffer::Buffer;
    use crate::column::{ColumnView, DataType};
    use crate::env::Env;
    use crate::pipeline::Pipeline;
    use crate::selection::Kept;
    use crate::step::{DynSource, Source};
    use crate::top_k::top_k;

    /// Rows of a key, maybe null, a value and a name.
    type Row = (Option<i64>, i64, &'static str);

    fn batch_of(rows: &[Row], batch: &mut RowBatch) {
        let n = rows.len();
        batch.reset(u32::try_from(n).unwrap());
        let env = Env::current();
        let mut context = Context::new(env, None);
        let mut keys = Buffer::allocate(env, n * 8).unwrap();
        let mut bits = Buffer::allocate(env, BitSlice::<u8>::words_for(n)).unwrap();
        let mut values = Buffer::allocate(env, n * 8).unwrap();
        for (i, &(key, value, _)) in rows.iter().enumerate() {
            keys.as_mut_slice::<i64>()[i] = key.unwrap_or(0);
            BitSlice::of_mut(bits.as_mut_slice::<u8>()).set_if(i, key.is_some());
            values.as_mut_slice::<i64>()[i] = value;
        }
        let keys = ColumnView::new(&mut context, DataType::Int64, keys, Some(bits)).unwrap();
        let values = ColumnView::new(&mut context, DataType::Int64, values, None).unwrap();
        let name = |i: usize| Some(rows[i].2.as_bytes());
        let names = ColumnView::from_strings(&mut context, n, name).unwrap();
        for column in [keys, values, names] {
            assert!(batch.push_column(column).is_ok());
        }
    }

    /// Batches of rows, keeping all but every `drop`th, if any.
    struct Rows {
        batches: Vec<Vec<Row>>,
        drop: Option<u16>,
    }

    impl Source for Rows {
        const NAME: &'static str = "Rows";
        type State = usize;

        fn new_state(&self, _: &mut Context) -> Result<usize, Error> {
            Ok(0)
        }

        fn next(
            &self,
            _: &mut Context,
            i: &mut usize,
            batch: &mut RowBatch,
        ) -> Result<bool, Error> {
            let Some(rows) = self.batches.get(*i) else { return Ok(false) };
            *i += 1;
            batch_of(rows, batch);
            if let Some(drop) = self.drop {
                batch.selection_mut().retain(|row| row % drop != 0);
            }
            Ok(true)
        }
    }

    /// Keys of `(column, descending)`.
    fn by(keys: &[(u32, bool)]) -> SmallVec<SortKey> {
        let env = Env::current();
        let keys = keys.iter().map(|&(column, descending)| SortKey { column, descending });
        SmallVec::fixed_from(env, keys).unwrap()
    }

    /// By the key ascending, then the value descending.
    const KEY_THEN_VALUE: &[(u32, bool)] = &[(0, false), (1, true)];

    /// A row out of a sort: its key, value and name.
    type Out = (Option<i64>, i64, Vec<u8>);

    /// Each row out of `breaker`, in order.
    fn run<B: Breaker + 'static>(rows: Rows, breaker: B) -> Vec<Out> {
        run_in(Env::current(), rows, breaker).unwrap()
    }

    /// As `run`, in `run_env`.
    fn run_in<B: Breaker + 'static>(
        run_env: &Env,
        rows: Rows,
        breaker: B,
    ) -> Result<Vec<Out>, Error> {
        let env = Env::current();
        let input =
            Pipeline::new(DynSource::new(env, rows).unwrap(), SmallVec::new(env, 1).unwrap());
        let (sinks, pipeline) = after_breaker(input, breaker, []);
        let mut execution = pipeline.start_with_sinks(run_env, None, &sinks, None)?;
        let mut batch = RowBatch::new();
        let mut out = Vec::new();
        while execution.next(&mut batch)? {
            assert!(matches!(batch.selection().kept(), Kept::Prefix));
            let (keys, values, names) = (batch.column(0), batch.column(1), batch.column(2));
            for row in 0..batch.row_count() {
                let key = (!keys.is_null(row)).then(|| keys.int64s()[row as usize]);
                let name = names.string_values().get(row as usize).to_vec();
                out.push((key, values.int64s()[row as usize], name));
            }
        }
        Ok(out)
    }

    #[test]
    fn sorts_by_keys_with_nulls_first() {
        let rows = Rows {
            batches: alloc::vec![
                alloc::vec![(Some(2), 1, "a"), (None, 5, "n"), (Some(1), 1, "b")],
                alloc::vec![(Some(2), 9, "c"), (None, 3, "m"), (Some(1), 1, "e")],
            ],
            drop: None,
        };
        let sorted = run(rows, sort(by(KEY_THEN_VALUE)));
        let names: Vec<Vec<u8>> = sorted.into_iter().map(|row| row.2).collect();
        // Nulls first, `5` before `3`; `b` and `e` tie and stay in order.
        assert_eq!(
            names,
            [
                b"n".to_vec(),
                b"m".to_vec(),
                b"b".to_vec(),
                b"e".to_vec(),
                b"c".to_vec(),
                b"a".to_vec()
            ]
        );
    }

    #[test]
    fn fails_if_a_key_changes_type() {
        let env = Env::current();
        let mut context = Context::new(env, None);
        let keys = [SortKey { column: 0, descending: false }];
        let mut held = SortedRows::new(env).unwrap();
        let mut batch = RowBatch::new();
        batch_of(&[(Some(1), 1, "a")], &mut batch);
        assert!(held.append(&mut context, &batch, &keys).is_ok());
        let mut floats = RowBatch::new();
        floats.reset(1);
        let mut values = Buffer::allocate(env, 8).unwrap();
        values.as_mut_slice::<f64>()[0] = 1.0;
        let column = ColumnView::new(&mut context, DataType::Float64, values, None).unwrap();
        assert!(floats.push_column(column).is_ok());
        assert!(matches!(held.append(&mut context, &floats, &keys), Err(Error::Unsupported)));
    }

    #[test]
    #[cfg_attr(miri, ignore = "slow under Miri")]
    fn top_k_keeps_the_first_rows_a_sort_would() {
        // Names that tie, share prefixes, and include the empty one.
        const NAMES: [&str; 7] = ["", "a", "ab", "abc", "b", "ba", "\u{e9}"];
        // Keys at random, ascending and descending, with ties and nulls, over
        // sparse batches; sorted by both keys, and by the first alone, so
        // ties must keep their order.
        let shapes: [fn(i64) -> i64; 3] = [|i| (i * 7919) % 1000, |i| i / 3, |i| -(i / 3)];
        for shape in shapes {
            let rows: Vec<Row> = (0..9000)
                .map(|i| {
                    (
                        (i % 97 != 5).then(|| shape(i)),
                        i,
                        NAMES[usize::try_from(i * 31 % 7).unwrap()],
                    )
                })
                .collect();
            let batches: Vec<Vec<Row>> = rows.chunks(2048).map(<[Row]>::to_vec).collect();
            let rows = || Rows { batches: batches.clone(), drop: Some(5) };
            let sorted = run(rows(), sort(by(KEY_THEN_VALUE)));
            assert!(sorted.windows(2).all(|w| (w[0].0, -w[0].1) <= (w[1].0, -w[1].1)));
            let sorted = run(rows(), sort(by(&[(2, false)])));
            // By name, then in the order rows came, as their values are.
            assert!(sorted.windows(2).all(|w| (&w[0].2, w[0].1) < (&w[1].2, w[1].1)));
            for keys in [KEY_THEN_VALUE, &[(0, false)], &[(2, false)], &[(2, true), (0, false)]] {
                let all = run(rows(), sort(by(keys)));
                for count in [0, 1, 100, 3000, u32::MAX] {
                    let top = run(rows(), top_k(by(keys), count, None));
                    assert_eq!(top, all[..all.len().min(count as usize)], "{count}");
                }
            }
        }
    }

    #[cfg(feature = "spill")]
    #[test]
    #[cfg_attr(miri, ignore = "slow under Miri")]
    fn spills_its_rows_in_sorted_runs_and_merges_them_within_a_budget() {
        use crate::allocator::Heap;
        use crate::env::Memory;
        use crate::spill::tests::Strict;

        const NAMES: [&str; 5] = ["", "a", "ab", "b", "\u{e9}"];
        // Keys out of order, some null, with ties; full batches, which are
        // held as they are, and smaller ones, which are packed.
        let row = |n: i64| {
            let name = NAMES[usize::try_from(n * 31 % 5).unwrap()];
            ((n % 11 != 3).then_some(n * 7919 % 500), n, name)
        };
        let mut batches = Vec::new();
        let mut n = 0;
        for b in 0..40 {
            let len = if b % 3 == 0 { 700 } else { 2048 };
            batches.push((n..n + len).map(row).collect());
            n += len;
        }
        let rows = || Rows { batches: batches.clone(), drop: None };
        let keys = || by(&[(2, false), (0, true)]);
        let unlimited = Env::new(Memory::new(Heap, usize::MAX)).unwrap();
        let expected = run_in(&unlimited, rows(), sort(keys())).unwrap();
        let peak = unlimited.peak();
        drop(unlimited);
        assert_eq!(expected.len(), usize::try_from(n).unwrap());
        for divisor in [2, 4, 8] {
            let store: &'static Strict = alloc::boxed::Box::leak(alloc::boxed::Box::default());
            let env = Env::new(Memory::new(Heap, peak / divisor)).unwrap();
            env.spill_to(store);
            let got = run_in(&env, rows(), sort(keys()));
            let got = got.unwrap_or_else(|e| panic!("{e:?} at 1/{divisor} of {peak}"));
            assert!(got == expected, "differs at a budget of 1/{divisor}");
            assert!(env.peak() <= peak / divisor);
            let (made, left) = store.logs();
            assert!(made > 0 && left == 0, "{made} logs made, {left} left at 1/{divisor}");
        }
    }
}
