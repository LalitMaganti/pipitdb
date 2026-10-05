//! Uses the kernel so CI can measure its size. Built as a library, as
//! `crates/bench/check_size.sh` does.

#![no_std]

use core::alloc::{GlobalAlloc, Layout};

use pipit_kernel::allocator::{Budget, Heap};
use pipit_kernel::boxed::Box;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::column::{ColumnView, DataType};
use pipit_kernel::context::Context;
use pipit_parquet::chunk::ChunkReader;
use pipit_parquet::footer::ParquetFile;
use pipit_pipesql::lexer::{Lexer, TokenKind};
use pipit_pipesql::parser::{parse_expression, parse_query};
use pipit_pipesql::registry::Registry;

static REGISTRY: Registry = Registry::new(&[pipit_pipesql::stages::RELATIONAL]);
use pipit_kernel::error::Error;
use pipit_kernel::row_batch::{BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::slow_vec::SlowVec;
use pipit_kernel::spill::{Block, LogId, SpillStore, read_column, write_column};

#[unsafe(no_mangle)]
pub extern "C" fn column_sum(count: u32, value: i64) -> i64 {
    let Ok(mut values) = Buffer::allocate(&Heap, count as usize * size_of::<i64>()) else {
        return 0;
    };
    values.as_mut_slice::<i64>().fill(value);
    let mut context = Context::new(&Heap);
    let Ok(column) = ColumnView::new(&mut context, DataType::Int64, values, None) else {
        return 0;
    };
    let column = column.slice(1, count - 1);
    let mut batch = RowBatch::new();
    batch.reset(column.row_count());
    if batch.push_column(column).is_err() || batch.column(0).is_null(0) {
        return 0;
    }
    batch.column(0).int64s().iter().sum()
}

/// `value` doubled, through a `Box`.
#[unsafe(no_mangle)]
pub extern "C" fn box_double(value: u64) -> u64 {
    let Ok(mut boxed) = Box::new(&Heap, value) else { return 0 };
    *boxed *= 2;
    *boxed
}

/// The most bytes a `Box` of `value` takes from a budget of `limit` bytes,
/// or 0 if it doesn't fit.
#[unsafe(no_mangle)]
pub extern "C" fn budget_peak(value: u64, limit: usize) -> usize {
    let budget = Budget::new(&Heap, limit);
    let Ok(boxed) = Box::new(&budget, value) else { return 0 };
    drop(boxed);
    budget.peak()
}

/// One log, in a fixed array.
struct Fixed {
    bytes: core::cell::UnsafeCell<[u8; 256]>,
    len: core::cell::Cell<usize>,
}

impl Fixed {
    /// Bytes `start..start + len` of the log, if it has room.
    fn range(&self, start: usize, len: usize) -> Result<*mut u8, Error> {
        if start.checked_add(len).is_none_or(|end| end > 256) {
            return Err(Error::Io);
        }
        // SAFETY: `start` is within the array.
        Ok(unsafe { self.bytes.get().cast::<u8>().add(start) })
    }
}

impl SpillStore for Fixed {
    fn create(&self) -> Result<LogId, Error> {
        Ok(LogId(0))
    }

    fn append(&self, _: LogId, bytes: &[u8]) -> Result<Block, Error> {
        let offset = self.len.get();
        let to = self.range(offset, bytes.len())?;
        // SAFETY: `to` has room for `bytes`, which don't overlap the array.
        unsafe { to.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len()) };
        self.len.set(offset + bytes.len());
        Ok(Block { offset: offset as u64, len: bytes.len() as u64 })
    }

    fn seal(&self, _: LogId) -> Result<(), Error> {
        Ok(())
    }

    fn read(&self, _: LogId, block: Block, into: &mut [u8]) -> Result<(), Error> {
        let offset = usize::try_from(block.offset).map_err(|_| Error::Io)?;
        let from = self.range(offset, into.len())?;
        // SAFETY: as in `append`.
        unsafe { into.as_mut_ptr().copy_from_nonoverlapping(from, into.len()) };
        Ok(())
    }

    fn delete(&self, _: LogId) {}
}

/// `value`, spilled as a one-row column and read back, or 0 if that fails.
#[unsafe(no_mangle)]
pub extern "C" fn spill_round_trip(value: i64) -> i64 {
    let store =
        Fixed { bytes: core::cell::UnsafeCell::new([0; 256]), len: core::cell::Cell::new(0) };
    let Ok(mut values) = Buffer::allocate(&Heap, 8) else { return 0 };
    values.as_mut_slice::<i64>()[0] = value;
    let mut context = Context::new(&Heap);
    let Ok(column) = ColumnView::new(&mut context, DataType::Int64, values, None) else {
        return 0;
    };
    let Ok(log) = store.create() else { return 0 };
    let Ok(spilled) = write_column(&store, log, &column) else { return 0 };
    if store.seal(log).is_err() {
        return 0;
    }
    let Ok(read) = read_column(&mut context, &store, log, &spilled) else { return 0 };
    read.int64s()[0]
}

/// The length of the second of two strings, of `first` and `second` bytes.
#[unsafe(no_mangle)]
pub extern "C" fn second_string_len(first: u32, second: u32) -> usize {
    let Ok(mut offsets) = Buffer::allocate(&Heap, 3 * 4) else { return 0 };
    let ends = offsets.as_mut_slice::<u32>();
    (ends[1], ends[2]) = (first, first + second);
    let Ok(bytes) = Buffer::allocate(&Heap, (first + second) as usize) else { return 0 };
    let mut context = Context::new(&Heap);
    let Ok(column) = ColumnView::strings(&mut context, offsets, bytes, None) else { return 0 };
    column.string_values().get(1).len()
}

/// The byte at `at` of `len` bytes from `data`, read through a
/// `ByteSource`, or 0 if there's none.
///
/// # Safety
///
/// `data` must be valid for `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn read_byte(data: *const u8, len: usize, at: u64) -> u8 {
    // SAFETY: guaranteed by the caller.
    let source: &[u8] = unsafe { core::slice::from_raw_parts(data, len) };
    let mut byte = [0];
    match (&source as &dyn ByteSource).read(at, &mut byte) {
        Ok(()) => byte[0],
        Err(_) => 0,
    }
}

/// How many rows the first column of the Parquet file in `len` bytes from
/// `data` has, read a page at a time, or 0 if it can't be read.
///
/// # Safety
///
/// `data` must be valid for `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn parquet_rows(data: *const u8, len: usize) -> usize {
    // SAFETY: guaranteed by the caller.
    let bytes: &[u8] = unsafe { core::slice::from_raw_parts(data, len) };
    let Ok(file) = ParquetFile::open(&Heap, &bytes) else { return 0 };
    let Some(&column) = file.columns().first() else { return 0 };
    let mut context = Context::new(&Heap);
    let mut rows = 0;
    for group in 0..file.row_groups() {
        let Ok(mut reader) = ChunkReader::new(&bytes, column, file.chunk(group, 0)) else {
            return 0;
        };
        while let Ok(left) = reader.page_left(&mut context) {
            let batch = left.min(BATCH_ROWS_MAX as usize);
            if batch == 0 || reader.read(&mut context, batch).is_err() {
                break;
            }
            rows += batch;
        }
    }
    rows
}

/// Pushes `0..count` to a `SlowVec`, and returns the last.
#[unsafe(no_mangle)]
pub extern "C" fn vec_sum(count: u32) -> u64 {
    let Ok(mut values) = SlowVec::new(&Heap, (count as usize).next_power_of_two()) else {
        return 0;
    };
    for i in 0..count {
        if values.push(u64::from(i)).is_err() {
            return 0;
        }
    }
    values.last().copied().unwrap_or(0)
}

/// Returns the number of tokens, or `u32::MAX` on an error.
///
/// # Safety
///
/// `source` must be valid for `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn token_count(source: *const u8, len: usize) -> u32 {
    // SAFETY: guaranteed by the caller.
    let source = unsafe { core::slice::from_raw_parts(source, len) };
    let Ok(mut lexer) = Lexer::new(source) else { return u32::MAX };
    let mut count = 0;
    loop {
        match lexer.next_token() {
            Ok(token) if token.kind == TokenKind::End => return count,
            Ok(_) => count += 1,
            Err(_) => return u32::MAX,
        }
    }
}

/// Returns the number of nodes, or the error code.
///
/// # Safety
///
/// `source` must be valid for `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn expression_node_count(source: *const u8, len: usize) -> u32 {
    // SAFETY: guaranteed by the caller.
    let source = unsafe { core::slice::from_raw_parts(source, len) };
    match parse_expression(&Heap, source) {
        Ok(ast) => ast.node_count(),
        Err(error) => u32::from(error.code as u16),
    }
}

/// Returns the number of nodes, or the error code.
///
/// # Safety
///
/// `source` must be valid for `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn query_node_count(source: *const u8, len: usize) -> u32 {
    // SAFETY: guaranteed by the caller.
    let source = unsafe { core::slice::from_raw_parts(source, len) };
    match parse_query(&Heap, &REGISTRY, source) {
        Ok(ast) => ast.node_count(),
        Err(error) => u32::from(error.code as u16),
    }
}

#[cfg_attr(target_arch = "wasm32", link(wasm_import_module = "env"))]
#[cfg_attr(not(target_arch = "wasm32"), link(name = "c"))]
unsafe extern "C" {
    fn aligned_alloc(align: usize, size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

struct LibcAllocator;

// SAFETY: `aligned_alloc` returns memory valid for the layout.
unsafe impl GlobalAlloc for LibcAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // `aligned_alloc` needs the size to be a multiple of the alignment.
        let layout = layout.pad_to_align();
        // SAFETY: the alignment is a power of two and divides the size.
        unsafe { aligned_alloc(layout.align(), layout.size()) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _: Layout) {
        // SAFETY: `ptr` came from `aligned_alloc`.
        unsafe { free(ptr) }
    }
}

#[global_allocator]
static ALLOCATOR: LibcAllocator = LibcAllocator;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
