//! `top_k`: the breaker that outputs the first rows in the order `sort`
//! would put them, as `ORDER BY` then `LIMIT` does, as Perfetto's
//! `core/exec/top_k` does.

use core::cmp::Ordering;

use crate::allocator::{AllocError, Allocator};
use crate::column::DataType;
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::row_layout::{Encode, RowLayout, Word, decode, encode_nullable};
use crate::slow_vec::SlowVec;
use crate::sorted_rows::{KeyColumn, SortKey, SortedRows, lay_out_keys_from};
use crate::step::{Breaker, DynOperator};

/// The breaker that outputs the first `count` rows sorted by `keys`. If it
/// has a slot to `publish` in, whenever it prunes it publishes the first key
/// of its cutoff, an `Int64`, there in the run's `Context`, for its scan to
/// skip by.
///
/// Once it holds `count` rows plus `count` or a batch's worth more,
/// whichever is larger, it sorts them and keeps only the first `count`, by
/// appending them in order to rows held afresh. The last of those becomes
/// the cutoff: from then on, rows which don't come before it are dropped as
/// they arrive, so on most inputs pruning soon becomes rare.
pub fn top_k<'a>(
    allocator: &dyn Allocator,
    keys: SlowVec<SortKey>,
    count: u32,
    publish: Option<u32>,
) -> Result<DynOperator<'a>, AllocError> {
    DynOperator::new(allocator, TopK { keys, count, publish })
}

struct TopK {
    keys: SlowVec<SortKey>,
    count: u32,
    publish: Option<u32>,
}

/// The rows held, the cutoff once there is one, the keys of a batch's rows
/// laid out, and how many rows have been output.
struct Kept {
    rows: SortedRows,
    cutoff: Option<Cutoff>,
    keys: SlowVec<u8>,
    emitted: u32,
}

/// The last row kept: its keys laid out, and by key, where the strings of its
/// string keys are in `bytes`, if not null.
struct Cutoff {
    keys: SlowVec<u8>,
    strings: SlowVec<Option<(usize, usize)>>,
    bytes: SlowVec<u8>,
}

impl Breaker for TopK {
    type State = Kept;

    fn new_state(&self, context: &mut Context) -> Result<Kept, Error> {
        if let Some(slot) = self.publish {
            context.reserve_cutoff(slot)?;
        }
        let allocator = context.allocator();
        let rows = SortedRows::new(allocator)?;
        let keys = SlowVec::new(allocator, 1 << 31)?;
        Ok(Kept { rows, cutoff: None, keys, emitted: 0 })
    }

    fn consume(
        &self,
        context: &mut Context,
        kept: &mut Kept,
        input: &RowBatch,
    ) -> Result<(), Error> {
        if input.selection().is_empty() {
            return Ok(());
        }
        // Keys are checked even when no row is output, as in Perfetto.
        kept.rows.check_keys(context.allocator(), input, &self.keys)?;
        if self.count == 0 {
            return Ok(());
        }
        if kept.cutoff.is_some() {
            let passed = self.before_cutoff(kept, input)?;
            kept.rows.append_laid_out(context, &passed, &self.keys, &kept.keys)?;
        } else {
            kept.rows.append(context, input, &self.keys)?;
        }
        let most = u64::from(self.count) + u64::from(self.count.max(BATCH_ROWS_MAX));
        if u64::from(kept.rows.len()) >= most {
            self.prune(context, kept)?;
        }
        Ok(())
    }

    fn finalize(&self, context: &mut Context, kept: &mut Kept) -> Result<(), Error> {
        kept.rows.sort(context)?;
        kept.rows.truncate_order(self.count);
        Ok(())
    }

    fn serve(
        &self,
        context: &mut Context,
        kept: &mut Kept,
        output: &mut RowBatch,
    ) -> Result<bool, Error> {
        kept.emitted += kept.rows.gather(context, output, kept.emitted)?;
        Ok(kept.emitted < kept.rows.order_len())
    }
}

impl TopK {
    /// The rows of `input` which come before the cutoff, their keys laid out
    /// at the start of `kept.keys`. A row equal to the cutoff comes after it,
    /// having arrived later.
    fn before_cutoff<'b>(
        &self,
        kept: &mut Kept,
        input: &RowBatch<'b>,
    ) -> Result<RowBatch<'b>, Error> {
        let (Some(cutoff), keys, held) = (&kept.cutoff, &mut kept.keys, &kept.rows) else {
            failed!()
        };
        let (layout, columns) = (held.layout(), held.key_columns());
        check!(!columns.is_empty());
        let stride = layout.stride() as usize;
        // Strings' slots are left as they were: `sort` lays them out.
        let len = input.selection().len() as usize * stride;
        keys.extend_zeroed(len.saturating_sub(keys.len()))?;
        // Most rows come after the cutoff on the first key alone: they're
        // dropped before any key is laid out.
        let (slot, column) = (layout.slot(0), input.column(at!(self.keys, 0).column));
        let (values, string) = (column.values(), column.data_type() == DataType::String);
        let float = column.data_type() == DataType::Float64;
        let words = if string { &[][..] } else { values.words() };
        let their_string =
            at!(cutoff.strings, 0).map(|(start, len)| at!(cutoff.bytes, start..start + len));
        let at = usize::from(slot.offset);
        let mut bits = [0; 8];
        if !string {
            let cutoff_bits = at!(cutoff.keys, at + 1..=at + Word::SIZE);
            at_mut!(bits, 8 - Word::SIZE..).copy_from_slice(cutoff_bits);
        }
        let theirs = (*at!(cutoff.keys, at), u64::from_be_bytes(bits));
        let only = columns.len() == 1;
        let (mut i, mut passing) = (0, 0);
        let mut passed = RowBatch::new();
        passed.copy_from(input);
        passed.selection_mut().retain(|row| {
            let from = i;
            i += 1;
            let row = u32::from(row);
            let present = !values.is_null(row);
            let ordering = if string {
                let mine = present.then(|| values.strings().get(row as usize));
                // Nulls first, as `None` orders.
                let ordering = mine.cmp(&their_string);
                if slot.descending { ordering.reverse() } else { ordering }
            } else {
                let mine = present.then(|| Word { bits: *at!(words, row as usize), float });
                encode_nullable(slot, mine).cmp(&theirs)
            };
            if ordering.is_gt() || only && ordering.is_eq() {
                return false;
            }
            let laid_out = at_mut!(keys, passing * stride..(passing + 1) * stride);
            let before = lay_out_passing(held, &self.keys, cutoff, input, from, ordering, laid_out);
            passing += usize::from(before);
            before
        });
        Ok(passed)
    }

    /// Keeps only the first `count` rows held, in order, and makes the last
    /// of them the cutoff.
    fn prune(&self, context: &mut Context, kept: &mut Kept) -> Result<(), Error> {
        let rows = &mut kept.rows;
        rows.sort(context)?;
        rows.truncate_order(self.count);
        let mut first = SortedRows::new(context.allocator())?;
        let mut batch = RowBatch::new();
        let mut at = 0;
        while at < self.count {
            at += rows.gather(context, &mut batch, at)?;
            first.append(context, &batch, &self.keys)?;
        }
        first.flush(context)?;
        let last = self.count - 1;
        let allocator = context.allocator();
        let keys = SlowVec::fixed_from(allocator, first.keys_of(last).iter().copied())?;
        if let Some(slot) = self.publish {
            context.set_cutoff(slot, decode(first.layout().slot(0), &keys, false));
        }
        let mut strings = SlowVec::fixed(allocator, self.keys.len())?;
        let mut bytes = SlowVec::new(allocator, 1 << 31)?;
        for column in first.key_columns() {
            let string = column.string().and_then(|column| first.string(column, last));
            strings.push(string.map(|string| (bytes.len(), string.len())))?;
            bytes.extend_from_slice(string.unwrap_or_default())?;
        }
        (kept.rows, kept.cutoff) = (first, Some(Cutoff { keys, strings, bytes }));
        Ok(())
    }
}

/// Lays out at `laid_out` the keys of the `from`th row `input` keeps, which
/// compares with the cutoff on its first key as `first`, and returns whether
/// it comes before the cutoff. Few rows do, so it's kept out of the loop.
#[cold]
#[inline(never)]
fn lay_out_passing(
    held: &SortedRows,
    keys: &[SortKey],
    cutoff: &Cutoff,
    input: &RowBatch,
    from: u32,
    first: Ordering,
    laid_out: &mut [u8],
) -> bool {
    let (layout, columns) = (held.layout(), held.key_columns());
    lay_out_keys_from(layout, input, keys, from, laid_out);
    let row = input.selection().kept().row(from);
    first.is_lt() || compare(layout, columns, cutoff, input, row, laid_out).is_lt()
}

/// How the row `row` of `input`, its keys laid out at `laid_out` by
/// `layout`, compares with the cutoff. String keys, whose ranks aren't known
/// until rows are sorted, are compared by their bytes, as ranks order them.
fn compare(
    layout: &RowLayout,
    columns: &[KeyColumn],
    cutoff: &Cutoff,
    input: &RowBatch,
    row: u32,
    laid_out: &[u8],
) -> Ordering {
    for (k, key) in columns.iter().enumerate() {
        let slot = layout.slot(k);
        let ordering = if let Some(column) = key.string() {
            let values = input.column(column).values();
            let mine = (!values.is_null(row)).then(|| values.strings().get(row as usize));
            let theirs =
                at!(cutoff.strings, k).map(|(start, len)| at!(cutoff.bytes, start..start + len));
            // Nulls first, as `None` orders.
            let ordering = mine.cmp(&theirs);
            if slot.descending { ordering.reverse() } else { ordering }
        } else {
            let start = usize::from(slot.offset);
            let end = columns
                .get(k + 1)
                .map_or(layout.stride() as usize, |_| usize::from(layout.slot(k + 1).offset));
            at!(laid_out, start..end).cmp(at!(cutoff.keys, start..end))
        };
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}
