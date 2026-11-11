//! `RowBlocks`: rows of a fixed number of bytes, appended in blocks the
//! size of a batch's column of values, so a block let go of serves the next
//! column buffer taken, and rows that are done with can be let go of a
//! block at a time.

use crate::buffer::Buffer;
use crate::context::Context;
use crate::error::Error;
use crate::row_batch::BATCH_ROWS_MAX;
use crate::slow_vec::SlowVec;

/// How many bytes a block holds.
const BLOCK_BYTES: usize = BATCH_ROWS_MAX as usize * 8;

pub(crate) struct RowBlocks {
    /// How many bytes a row has.
    stride: usize,
    /// How many rows a block holds.
    block_rows: u32,
    /// `block_rows`' reciprocal, to divide by it with a multiply.
    reciprocal: u64,
    /// Each block, its rows below `len` written; `None` once let go of.
    blocks: SlowVec<Option<Buffer>>,
    /// The first block not let go of.
    kept: usize,
    /// How many rows there are.
    len: u32,
}

impl RowBlocks {
    pub(crate) fn new(context: &Context, stride: usize) -> Result<RowBlocks, Error> {
        check!(stride > 0 && stride <= BLOCK_BYTES);
        #[expect(clippy::cast_possible_truncation, reason = "at most `BLOCK_BYTES`")]
        let block_rows = (BLOCK_BYTES / stride) as u32;
        let reciprocal = u64::MAX / u64::from(block_rows) + 1;
        let blocks = SlowVec::new(context.allocator(), 1 << 24)?;
        Ok(RowBlocks { stride, block_rows, reciprocal, blocks, kept: 0, len: 0 })
    }

    pub(crate) fn len(&self) -> u32 {
        self.len
    }

    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// The block row `row` is in, and which of its rows it is.
    #[inline]
    fn locate(&self, row: u32) -> (usize, usize) {
        #[expect(clippy::cast_possible_truncation, reason = "the high word")]
        let block = ((u128::from(self.reciprocal) * u128::from(row)) >> 64) as u32;
        (block as usize, ((row - block * self.block_rows) as usize) * self.stride)
    }

    /// The bytes of row `row`, which mustn't be let go of.
    #[inline]
    pub(crate) fn get(&self, row: u32) -> &[u8] {
        check!(row < self.len);
        let (block, at) = self.locate(row);
        let Some(block) = at!(self.blocks, block) else { failed!() };
        // SAFETY: every block holds `BLOCK_BYTES`, at least `block_rows`
        // rows, and the row is one of them, written as it's below `len`.
        unsafe { core::slice::from_raw_parts(block.as_non_null().as_ptr().add(at), self.stride) }
    }

    /// Appends `count` rows, which `write` writes into the bytes given, whole
    /// rows one after another, each call's first its row's number.
    pub(crate) fn extend(
        &mut self,
        context: &mut Context,
        count: u32,
        mut write: impl FnMut(u32, &mut [u8]),
    ) -> Result<(), Error> {
        let end = self.len.checked_add(count).ok_or(Error::OutOfMemory)?;
        while self.len < end {
            let (block, at) = self.locate(self.len);
            if block == self.blocks.len() {
                self.blocks.push(Some(context.values_buffer(BLOCK_BYTES)?))?;
            }
            let take = (self.block_rows as usize - at / self.stride).min((end - self.len) as usize);
            let Some(buffer) = at_mut!(self.blocks, block) else { failed!() };
            write(self.len, at_mut!(buffer.as_mut_slice::<u8>(), at..at + take * self.stride));
            #[expect(clippy::cast_possible_truncation, reason = "at most a block's")]
            let take = take as u32;
            self.len += take;
        }
        Ok(())
    }

    /// Calls `write` with the rows of each block, and its first row's
    /// number.
    pub(crate) fn for_each_block_mut(&mut self, mut write: impl FnMut(u32, &mut [u8])) {
        let mut first = 0;
        for block in self.blocks.iter_mut() {
            let rows = self.block_rows.min(self.len - first);
            let Some(block) = block else { failed!() };
            write(first, at_mut!(block.as_mut_slice::<u8>(), ..rows as usize * self.stride));
            first += rows;
        }
    }

    /// Lets go of the blocks whose rows are all below `row`.
    pub(crate) fn let_go_below(&mut self, row: u32) {
        let (block, _) = self.locate(row.min(self.len));
        for block in at_mut!(self.blocks, self.kept.min(block)..block) {
            *block = None;
        }
        self.kept = self.kept.max(block);
    }

    /// Lets go of every block.
    pub(crate) fn free(&mut self) {
        for block in self.blocks.iter_mut() {
            *block = None;
        }
        self.kept = self.blocks.len();
    }
}
