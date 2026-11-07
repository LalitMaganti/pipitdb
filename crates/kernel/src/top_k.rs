//! `top_k`: the breaker that outputs the first rows in the order `sort`
//! would put them, as `ORDER BY` then `LIMIT` does, as Perfetto's
//! `core/exec/top_k` does.

use core::cmp::Ordering;

use crate::allocator::{AllocError, Allocator};
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::{BATCH_ROWS_MAX, RowBatch};
use crate::row_layout::RowLayout;
use crate::slow_vec::SlowVec;
use crate::sorted_rows::{SortKey, SortedRows, lay_out_keys};
use crate::step::{Breaker, DynOperator};

/// The breaker that outputs the first `count` rows sorted by `keys`.
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
) -> Result<DynOperator<'a>, AllocError> {
    DynOperator::new(allocator, TopK { keys, count })
}

struct TopK {
    keys: SlowVec<SortKey>,
    count: u32,
}

/// The rows held, the cutoff once there is one, and how many rows have
/// been output.
struct Kept {
    rows: SortedRows,
    cutoff: Option<Cutoff>,
    /// The keys of a batch's rows, laid out.
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
        if self.count == 0 {
            kept.rows.check_keys(context.allocator(), input, &self.keys)?;
            return Ok(());
        }
        match &kept.cutoff {
            None => kept.rows.append(context, input, &self.keys)?,
            Some(cutoff) => {
                kept.rows.check_keys(context.allocator(), input, &self.keys)?;
                let (layout, strings) = (kept.rows.layout(), kept.rows.string_columns());
                let passed = self.before_cutoff(layout, strings, cutoff, &mut kept.keys, input)?;
                kept.rows.append_laid_out(context, &passed, &self.keys, &kept.keys)?;
            }
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
    /// by `layout` at the start of `keys`. A row equal to the cutoff comes
    /// after it, having arrived later.
    fn before_cutoff<'b>(
        &self,
        layout: &RowLayout,
        strings: &[Option<u32>],
        cutoff: &Cutoff,
        keys: &mut SlowVec<u8>,
        input: &RowBatch<'b>,
    ) -> Result<RowBatch<'b>, Error> {
        let stride = layout.stride() as usize;
        let count = input.selection().len() as usize;
        while keys.pop().is_some() {}
        keys.extend_zeroed(count * stride)?;
        lay_out_keys(layout, input, &self.keys, keys);
        let mut passing = 0;
        let mut i = 0;
        let mut passed = RowBatch::new();
        passed.reset(input.row_count());
        for c in 0..input.column_count() {
            let Ok(()) = passed.push_column(input.column(c).clone()) else { failed!() };
        }
        passed.selection_mut().clone_from(input.selection());
        passed.selection_mut().retain(|row| {
            let laid_out = at!(keys, i * stride..(i + 1) * stride);
            i += 1;
            let ordering = compare(layout, strings, input, u32::from(row), laid_out, cutoff);
            if ordering != Ordering::Less {
                return false;
            }
            if passing != i - 1 {
                keys.copy_within((i - 1) * stride..i * stride, passing * stride);
            }
            passing += 1;
            true
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
        let laid_out = first.keys_of(last);
        let mut keys = SlowVec::fixed(allocator, laid_out.len())?;
        keys.extend_from_slice(laid_out)?;
        let mut strings = SlowVec::fixed(allocator, self.keys.len())?;
        let mut bytes = SlowVec::new(allocator, 1 << 31)?;
        for column in first.string_columns() {
            let string = column.and_then(|column| first.string(column, last));
            strings.push(string.map(|string| (bytes.len(), string.len())))?;
            bytes.extend_from_slice(string.unwrap_or_default())?;
        }
        (kept.rows, kept.cutoff) = (first, Some(Cutoff { keys, strings, bytes }));
        Ok(())
    }
}

/// How the row `row` of `input`, its keys laid out at `laid_out`, compares
/// with the cutoff. String keys, whose ranks aren't known until rows are
/// sorted, are compared by their bytes, as ranks order them.
fn compare(
    layout: &RowLayout,
    strings: &[Option<u32>],
    input: &RowBatch,
    row: u32,
    laid_out: &[u8],
    cutoff: &Cutoff,
) -> Ordering {
    if strings.iter().all(Option::is_none) {
        return laid_out.cmp(&cutoff.keys);
    }
    for (k, string) in strings.iter().enumerate() {
        let slot = layout.slot(k);
        let ordering = if let Some(column) = *string {
            let values = input.column(column).values();
            let mine = (!values.is_null(row)).then(|| values.strings().get(row as usize));
            let theirs =
                at!(cutoff.strings, k).map(|(start, len)| at!(cutoff.bytes, start..start + len));
            // Nulls first, as `None` orders.
            let ordering = mine.cmp(&theirs);
            if slot.descending { ordering.reverse() } else { ordering }
        } else {
            let start = usize::from(slot.offset);
            let end = strings
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
