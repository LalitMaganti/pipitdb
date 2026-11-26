//! `sort`: the breaker that stably sorts rows by their keys, as Perfetto's
//! `core/exec/sort` does. Nulls sort as in SQLite: first ascending, last
//! descending.

use crate::containers::SmallVec;
use crate::context::Context;
use crate::error::Error;
#[cfg(feature = "spill")]
use crate::radix_sort::Scratch;
use crate::row_batch::RowBatch;
#[cfg(feature = "spill")]
use crate::sort_runs::{Runs, batch_bytes};
pub use crate::sorted_rows::SortKey;
use crate::sorted_rows::SortedRows;
use crate::step::Breaker;

/// The breaker that sorts rows by `keys`, the first first, which must be
/// columns of numbers.
pub fn sort(keys: SmallVec<SortKey>) -> Sort {
    Sort { keys }
}

pub struct Sort {
    keys: SmallVec<SortKey>,
}

pub struct SortState {
    /// The rows held, sorted once all are in.
    rows: SortedRows,
    /// About how many bytes of rows `rows` holds.
    #[cfg(feature = "spill")]
    held: usize,
    /// What the run's memory held before any rows were, so what the rows
    /// take is known until the first run is written.
    #[cfg(feature = "spill")]
    base: usize,
    /// The sorted runs written once it could hold no more, if any.
    #[cfg(feature = "spill")]
    runs: Option<Runs>,
}

impl Breaker for Sort {
    const NAME: &'static str = "Sort";
    type State = SortState;

    fn new_state(&self, context: &mut Context) -> Result<SortState, Error> {
        Ok(SortState {
            rows: SortedRows::new(context.env())?,
            #[cfg(feature = "spill")]
            held: 0,
            #[cfg(feature = "spill")]
            base: context.env().used(),
            #[cfg(feature = "spill")]
            runs: None,
        })
    }

    fn consume(
        &self,
        context: &mut Context,
        state: &mut SortState,
        input: &RowBatch,
    ) -> Result<(), Error> {
        #[cfg(feature = "spill")]
        spill_if_full(context, state, input)?;
        state.rows.append(context, input, &self.keys)
    }

    fn finalize(&self, context: &mut Context, state: &mut SortState) -> Result<(), Error> {
        #[cfg(feature = "spill")]
        if let Some(runs) = &mut state.runs {
            if state.rows.len() > 0 {
                runs.write(context, &mut state.rows)?;
                state.rows = SortedRows::new(context.env())?;
            }
            return runs.finish(context, &self.keys);
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
        if let Some(runs) = &mut state.runs {
            return runs.serve(context, &self.keys, output);
        }
        state.rows.serve(context, output)
    }
}

/// Writes the rows held as a sorted run if `input` wouldn't fit with
/// them: the first time, if they wouldn't leave room to be sorted; and then
/// once they hold as many bytes of rows as the first. Memory freed stays
/// with the run's slabs, so only the first can be measured.
#[cfg(feature = "spill")]
fn spill_if_full(
    context: &mut Context,
    state: &mut SortState,
    input: &RowBatch,
) -> Result<(), Error> {
    let env = context.env();
    let Some(store) = env.spill() else { return Ok(()) };
    let bytes = batch_bytes(input);
    let full = if let Some(runs) = &state.runs {
        state.held + bytes > runs.capacity()
    } else {
        // Sorting takes the radix sort's scratch, past a few thousand rows a
        // fixed table of half a megabyte, then a word a row and as many again
        // to merge, and the rows' places may double: about 24 bytes a row.
        // Ranking strings takes what's hard to bound, so with string keys,
        // room for the rows to double and then be sorted, as much again.
        let rows = state.rows.len() as usize + input.selection().len() as usize;
        let taken = env.used().saturating_sub(state.base);
        let held = if state.rows.has_string_keys() { 3 * taken } else { 24 * rows };
        let sorting = Scratch::bytes(rows) + held;
        state.rows.len() > 0 && !env.leaves_room(sorting + 2 * bytes)
    };
    if full {
        if let Some(runs) = &mut state.runs {
            runs.write(context, &mut state.rows)?;
        } else {
            let available = env.room_above(state.base);
            let runs = Runs::new(context, store, &mut state.rows, state.held, available)?;
            state.runs = Some(runs);
        }
        state.rows = SortedRows::new(env)?;
        state.held = 0;
    }
    state.held += bytes;
    Ok(())
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
        let mut views = Buffer::allocate(env, n * 8).unwrap();
        let mut bytes = Vec::new();
        for (i, &(key, value, name)) in rows.iter().enumerate() {
            keys.as_mut_slice::<i64>()[i] = key.unwrap_or(0);
            BitSlice::of_mut(bits.as_mut_slice::<u8>()).set_if(i, key.is_some());
            values.as_mut_slice::<i64>()[i] = value;
            let start = u32::try_from(bytes.len()).unwrap();
            views.as_mut_slice::<u32>()[2 * i..2 * i + 2]
                .copy_from_slice(&[start, u32::try_from(name.len()).unwrap()]);
            bytes.extend_from_slice(name.as_bytes());
        }
        let mut stored = Buffer::allocate(env, bytes.len()).unwrap();
        stored.as_mut_slice::<u8>().copy_from_slice(&bytes);
        let keys = ColumnView::new(&mut context, DataType::Int64, keys, Some(bits)).unwrap();
        let values = ColumnView::new(&mut context, DataType::Int64, values, None).unwrap();
        let names = ColumnView::strings(&mut context, views, stored, None).unwrap();
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

    /// Each row out of `breaker`, in order.
    fn run<B: Breaker + 'static>(rows: Rows, breaker: B) -> Vec<(Option<i64>, i64, Vec<u8>)> {
        let env = Env::current();
        let input =
            Pipeline::new(DynSource::new(env, rows).unwrap(), SmallVec::new(env, 1).unwrap());
        let (sinks, pipeline) = after_breaker(input, breaker, []);
        let mut execution = pipeline.start_with_sinks(env, None, &sinks, None).unwrap();
        let mut batch = RowBatch::new();
        let mut out = Vec::new();
        while execution.next(&mut batch).unwrap() {
            assert!(matches!(batch.selection().kept(), Kept::Prefix));
            let (keys, values, names) = (batch.column(0), batch.column(1), batch.column(2));
            for row in 0..batch.row_count() {
                let key = (!keys.is_null(row)).then(|| keys.int64s()[row as usize]);
                let name = names.string_values().get(row as usize).to_vec();
                out.push((key, values.int64s()[row as usize], name));
            }
        }
        out
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
    fn writes_sorted_runs_past_its_budget() {
        use crate::allocator::Heap;
        use crate::env::Memory;
        use crate::spill::tests::Strict;

        // Keys out of order, a fifth of them null, and many ties.
        const NAMES: [&str; 3] = ["x", "yy", "zzz"];
        let name = |n: i64| NAMES[usize::try_from(n % 3).unwrap()];
        let row = |n: i64| ((n % 5 != 0).then_some(n * 7919 % 1000), n, name(n));
        let rows = Rows {
            batches: (0..40_i64).map(|b| (b * 2000..(b + 1) * 2000).map(row).collect()).collect(),
            drop: None,
        };
        let run = |limit: usize, store: Option<&'static Strict>| {
            let env = Env::new(Memory::new(Heap, limit)).unwrap();
            if let Some(store) = store {
                env.spill_to(store);
            }
            let sort = sort(by(KEY_THEN_VALUE));
            let context = &mut Context::new(&env, None);
            let (mut out, mut i) = (Vec::new(), 0);
            let mut state = Breaker::new_state(&sort, context)?;
            let mut batch = RowBatch::new();
            while rows.next(context, &mut i, &mut batch)? {
                Breaker::consume(&sort, context, &mut state, &batch)?;
            }
            Breaker::finalize(&sort, context, &mut state)?;
            loop {
                let more = Breaker::serve(&sort, context, &mut state, &mut batch)?;
                let (keys, values, names) = (batch.column(0), batch.column(1), batch.column(2));
                for row in 0..batch.row_count() {
                    let key = (!keys.is_null(row)).then(|| keys.int64s()[row as usize]);
                    let name = names.string_values().get(row as usize).to_vec();
                    out.push((key, values.int64s()[row as usize], name));
                }
                if !more {
                    break;
                }
            }
            batch.reset(0);
            drop(state);
            Ok::<_, Error>((out, env.peak()))
        };
        let (expected, peak) = run(usize::MAX, None).unwrap();
        let mut sorted: Vec<_> = (0..80_000).map(row).collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        assert!(expected.iter().map(|r| (r.0, r.1)).eq(sorted.iter().map(|r| (r.0, r.1))));
        let limit = peak / 4;
        assert_eq!(run(limit, None).err(), Some(Error::OutOfMemory));
        let store: &'static Strict = alloc::boxed::Box::leak(alloc::boxed::Box::default());
        let (out, spilled) = run(limit, Some(store)).unwrap();
        assert!(out == expected && spilled <= limit);
        // It wrote more runs than it merges at once, and deleted them.
        let logs = store.logs.borrow();
        assert!(logs.len() > crate::sort_runs::FAN_IN_MAX);
        assert!(logs.iter().all(Option::is_none));
        drop(logs);
        // From the least budget it runs in up, it runs in every one, and
        // whether it runs or fails, it leaves no log behind.
        let mut ran = false;
        for limit in (peak / 32..=peak).step_by(peak / 32) {
            let store: &'static Strict = alloc::boxed::Box::leak(alloc::boxed::Box::default());
            match run(limit, Some(store)) {
                Ok((out, _)) => {
                    assert_eq!(out, expected);
                    ran = true;
                }
                Err(error) => assert!(!ran && error == Error::OutOfMemory, "{limit}: {error:?}"),
            }
            assert!(store.logs.borrow().iter().all(Option::is_none), "{limit} left a log");
        }
        assert!(ran);
    }
}
