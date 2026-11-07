//! `sort`: the breaker that stably sorts rows by their keys, as Perfetto's
//! `core/exec/sort` does. Nulls sort as in SQLite: first ascending, last
//! descending.

use crate::allocator::{AllocError, Allocator};
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::RowBatch;
use crate::slow_vec::SlowVec;
pub use crate::sorted_rows::SortKey;
use crate::sorted_rows::SortedRows;
use crate::step::{Breaker, DynOperator};

/// The breaker that sorts rows by `keys`, the first first, which must be
/// columns of numbers.
pub fn sort<'a>(
    allocator: &dyn Allocator,
    keys: SlowVec<SortKey>,
) -> Result<DynOperator<'a>, AllocError> {
    DynOperator::new(allocator, Sort { keys })
}

struct Sort {
    keys: SlowVec<SortKey>,
}

/// The rows so far, and how many have been output.
struct Sorting {
    rows: SortedRows,
    emitted: u32,
}

impl Breaker for Sort {
    type State = Sorting;

    fn new_state(&self, context: &mut Context) -> Result<Sorting, Error> {
        Ok(Sorting { rows: SortedRows::new(context.allocator())?, emitted: 0 })
    }

    fn consume(
        &self,
        context: &mut Context,
        sorting: &mut Sorting,
        input: &RowBatch,
    ) -> Result<(), Error> {
        sorting.rows.append(context, input, &self.keys)
    }

    fn finalize(&self, context: &mut Context, sorting: &mut Sorting) -> Result<(), Error> {
        sorting.rows.sort(context.allocator())
    }

    fn serve(
        &self,
        context: &mut Context,
        sorting: &mut Sorting,
        output: &mut RowBatch,
    ) -> Result<bool, Error> {
        sorting.emitted += sorting.rows.gather(context, output, sorting.emitted)?;
        Ok(sorting.emitted < sorting.rows.order_len())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::{ColumnView, DataType};
    use crate::pipeline::Pipeline;
    use crate::query_allocators::QueryAllocators;
    use crate::selection::Kept;
    use crate::step::{DynSource, Source, Step};
    use crate::top_k::top_k;

    /// Rows of a key, maybe null, a value and a name.
    type Row = (Option<i64>, i64, &'static str);

    fn batch_of(rows: &[Row], batch: &mut RowBatch) {
        let n = rows.len();
        batch.reset(u32::try_from(n).unwrap());
        let mut context = Context::new(&Heap);
        let mut keys = Buffer::allocate(&Heap, n * 8).unwrap();
        let mut bits = Buffer::allocate(&Heap, n.div_ceil(8)).unwrap();
        let mut values = Buffer::allocate(&Heap, n * 8).unwrap();
        let mut views = Buffer::allocate(&Heap, n * 8).unwrap();
        let mut bytes = Vec::new();
        for (i, &(key, value, name)) in rows.iter().enumerate() {
            keys.as_mut_slice::<i64>()[i] = key.unwrap_or(0);
            bits.as_mut_slice::<u8>()[i / 8] |= u8::from(key.is_some()) << (i % 8);
            values.as_mut_slice::<i64>()[i] = value;
            let start = u32::try_from(bytes.len()).unwrap();
            views.as_mut_slice::<u32>()[2 * i..2 * i + 2]
                .copy_from_slice(&[start, u32::try_from(name.len()).unwrap()]);
            bytes.extend_from_slice(name.as_bytes());
        }
        let mut stored = Buffer::allocate(&Heap, bytes.len()).unwrap();
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

    /// By the key ascending, then the value descending.
    fn keys() -> SlowVec<SortKey> {
        let keys =
            [SortKey { column: 0, descending: false }, SortKey { column: 1, descending: true }];
        SlowVec::fixed_from(&Heap, keys.into_iter()).unwrap()
    }

    /// Each row out, in order.
    fn sorted(rows: Rows) -> Vec<(Option<i64>, i64, Vec<u8>)> {
        run(rows, sort(&Heap, keys()).unwrap())
    }

    /// Each row out of `step`, in order.
    fn run(rows: Rows, step: DynOperator<'static>) -> Vec<(Option<i64>, i64, Vec<u8>)> {
        let mut steps = SlowVec::new(&Heap, 1).unwrap();
        assert!(steps.push(Step::Operator(step)).is_ok());
        let pipeline = Pipeline::new(DynSource::new(&Heap, rows).unwrap(), steps);
        let query = QueryAllocators::new(&Heap);
        let mut execution = pipeline.start(&query).unwrap();
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
                alloc::vec![(Some(2), 1, "a"), (None, 5, "f"), (Some(1), 1, "b")],
                alloc::vec![(Some(2), 9, "c"), (None, 3, "d"), (Some(1), 1, "e")],
            ],
            drop: Some(2),
        };
        // Rows 0 and 2 of each batch are dropped, leaving `f` and `d`, whose
        // keys are null, so their values decide, descending.
        let names: Vec<Vec<u8>> = sorted(rows).into_iter().map(|row| row.2).collect();
        assert_eq!(names, [b"f".to_vec(), b"d".to_vec()]);
        let rows = Rows {
            batches: alloc::vec![
                alloc::vec![(Some(2), 1, "a"), (None, 5, "n"), (Some(1), 1, "b")],
                alloc::vec![(Some(2), 9, "c"), (None, 3, "m"), (Some(1), 1, "e")],
            ],
            drop: None,
        };
        let names: Vec<Vec<u8>> = sorted(rows).into_iter().map(|row| row.2).collect();
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
        let mut context = Context::new(&Heap);
        let keys = [SortKey { column: 0, descending: false }];
        let mut held = crate::sorted_rows::SortedRows::new(&Heap).unwrap();
        let mut batch = RowBatch::new();
        batch_of(&[(Some(1), 1, "a")], &mut batch);
        assert!(held.append(&mut context, &batch, &keys).is_ok());
        let mut floats = RowBatch::new();
        floats.reset(1);
        let mut values = Buffer::allocate(&Heap, 8).unwrap();
        values.as_mut_slice::<f64>()[0] = 1.0;
        let column = ColumnView::new(&mut context, DataType::Float64, values, None).unwrap();
        assert!(floats.push_column(column).is_ok());
        assert!(matches!(held.append(&mut context, &floats, &keys), Err(Error::Unsupported)));
    }

    /// 2,100 rows over two batches, of 100 keys.
    fn many() -> Vec<Vec<Row>> {
        let rows: Vec<Row> = (0..2100).map(|i| (Some((i * 7919) % 100), i, "x")).collect();
        rows.chunks(2048).map(<[Row]>::to_vec).collect()
    }

    #[test]
    fn sorts_many_rows() {
        let out = sorted(Rows { batches: many(), drop: None });
        assert_eq!(out.len(), 2100);
        assert!(out.windows(2).all(|w| (w[0].0, -w[0].1) <= (w[1].0, -w[1].1)));
    }

    #[test]
    #[cfg_attr(miri, ignore = "slow under Miri")]
    fn top_k_keeps_the_first_rows_a_sort_would() {
        // Keys at random, ascending and descending, with ties and nulls, over
        // sparse batches; sorted by both keys, and by the first alone, so
        // ties must keep their order.
        let shapes: [fn(i64) -> i64; 3] = [|i| (i * 7919) % 1000, |i| i / 3, |i| -(i / 3)];
        let by_key = || {
            SlowVec::fixed_from(&Heap, [SortKey { column: 0, descending: false }].into_iter())
                .unwrap()
        };
        for shape in shapes {
            let rows: Vec<Row> =
                (0..9000).map(|i| ((i % 97 != 5).then(|| shape(i)), i, "x")).collect();
            let batches: Vec<Vec<Row>> = rows.chunks(2048).map(<[Row]>::to_vec).collect();
            let rows = || Rows { batches: batches.clone(), drop: Some(5) };
            for keys in [keys(), by_key()] {
                let all = run(
                    rows(),
                    sort(&Heap, SlowVec::fixed_from(&Heap, keys.iter().copied()).unwrap()).unwrap(),
                );
                for count in [0, 1, 100, 3000, 20_000, 1 << 31, u32::MAX] {
                    let top = run(
                        rows(),
                        top_k(
                            &Heap,
                            SlowVec::fixed_from(&Heap, keys.iter().copied()).unwrap(),
                            count,
                        )
                        .unwrap(),
                    );
                    assert_eq!(top, all[..all.len().min(count as usize)], "{count}");
                }
            }
        }
    }
}
