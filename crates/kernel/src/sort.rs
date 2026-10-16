//! `sort`: the breaker that sorts rows by their keys, stably, as Perfetto's
//! `Sort` does. Nulls sort as in SQLite: first ascending, last descending.

use crate::allocator::{AllocError, Allocator};
use crate::column::DataType;
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::RowBatch;
use crate::row_layout::KeyColumn;
use crate::slow_vec::SlowVec;
use crate::sorted_rows::SortedRows;
use crate::step::{Breaker, DynOperator};

/// A key a breaker that sorts is given, once lowered: where its column is
/// in batches, of what type, and whether it descends.
#[derive(Clone, Copy)]
pub struct SortKey {
    pub position: u32,
    pub data_type: DataType,
    pub descending: bool,
}

/// The breaker that sorts rows by `keys`, the first first, which must be
/// integers or floats.
pub fn sort<'a>(
    allocator: &dyn Allocator,
    keys: SlowVec<SortKey>,
) -> Result<DynOperator<'a>, AllocError> {
    check!(keys.iter().all(|key| !key.data_type.has_bytes()));
    let positions = SlowVec::fixed_from(allocator, keys.iter().map(|key| key.position))?;
    DynOperator::new(allocator, Sort { keys, positions })
}

struct Sort {
    keys: SlowVec<SortKey>,
    /// Where each key's column is in batches.
    positions: SlowVec<u32>,
}

/// The rows so far, and how many have been output.
struct Sorting {
    rows: SortedRows,
    served: usize,
}

impl Breaker for Sort {
    type State = Sorting;

    fn new_state(&self, context: &mut Context) -> Result<Sorting, Error> {
        // Whether a key has nulls isn't known before the rows come, so every
        // key leaves room to say.
        let keys = self.keys.iter().map(|key| KeyColumn {
            data_type: key.data_type,
            nullable: true,
            descending: key.descending,
        });
        let keys = SlowVec::fixed_from(context.allocator(), keys)?;
        Ok(Sorting { rows: SortedRows::new(context.allocator(), &keys)?, served: 0 })
    }

    fn consume(
        &self,
        context: &mut Context,
        sorting: &mut Sorting,
        input: &RowBatch,
    ) -> Result<(), Error> {
        sorting.rows.append(context, input, &self.positions)
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
        sorting.served += sorting.rows.view(context, output, sorting.served)?;
        Ok(sorting.served < sorting.rows.len())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::allocator::Heap;
    use crate::buffer::Buffer;
    use crate::column::ColumnView;
    use crate::pipeline::Pipeline;
    use crate::query_allocators::QueryAllocators;
    use crate::selection::Kept;
    use crate::step::{DynSource, Source, Step};

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
        let keys = [
            SortKey { position: 0, data_type: DataType::Int64, descending: false },
            SortKey { position: 1, data_type: DataType::Int64, descending: true },
        ];
        SlowVec::fixed_from(&Heap, keys.into_iter()).unwrap()
    }

    /// Each row out, in order.
    fn sorted(rows: Rows) -> Vec<(Option<i64>, i64, Vec<u8>)> {
        let mut steps = SlowVec::new(&Heap, 1).unwrap();
        assert!(steps.push(Step::Operator(sort(&Heap, keys()).unwrap())).is_ok());
        let pipeline = Pipeline::new(DynSource::new(&Heap, rows).unwrap(), steps);
        let query = QueryAllocators::new(&Heap);
        let mut execution = pipeline.start(&query).unwrap();
        let mut batch = RowBatch::new();
        let mut out = Vec::new();
        while execution.next(&mut batch).unwrap() {
            assert!(matches!(batch.selection().kept(), Kept::All));
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
}
