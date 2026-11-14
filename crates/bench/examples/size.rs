//! Uses the kernel so CI can measure its size. Built as a library, as
//! `crates/bench/check_size.sh` does.

#![no_std]

use core::alloc::{GlobalAlloc, Layout};

use pipit_kernel::allocator::{Heap, LimitAllocator};
use pipit_kernel::boxed::Box;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::bytes::ByteSource;
use pipit_kernel::column::{ColumnView, DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::flat_hash_map::FlatHashMap;
use pipit_kernel::lower::lower;
use pipit_kernel::optimize::optimize;
use pipit_kernel::output_properties::plan_nodes;
use pipit_kernel::query_allocators::QueryAllocators;
use pipit_parquet::Uncompressed;
use pipit_parquet::table::ParquetTable;
use pipit_pipesql::compile::compile;
use pipit_pipesql::lexer::{Lexer, TokenKind};
use pipit_pipesql::parser::{parse_expression, parse_query};
use pipit_pipesql::registry::Registry;

static REGISTRY: Registry =
    Registry::new(&[pipit_pipesql::stages::RELATIONAL, pipit_pipesql::stages::AGGREGATION]);
use pipit_kernel::error::Error;
use pipit_kernel::expression::{Compute, ExprOp, ExprProgram, ExprStep};
use pipit_kernel::row_batch::{BATCH_ROWS_MAX, RowBatch};
use pipit_kernel::scannable::{Catalog, DynScannable, Scannable};
use pipit_kernel::slow_vec::SlowVec;
use pipit_kernel::spill::{Block, LogId, SpillStore, read_column, write_column};
use pipit_kernel::step::Transform;

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

/// How many of the numbers below `count` a map of the even ones has, less
/// those of them below `erased`, which it erases.
#[unsafe(no_mangle)]
pub extern "C" fn hash_map_found(count: i64, erased: i64) -> usize {
    let Ok(mut map) = FlatHashMap::for_len(&Heap, 0) else { return 0 };
    for key in (0..count).step_by(2) {
        if map.insert(&Heap, key, ()).is_err() {
            return 0;
        }
    }
    (0..erased).for_each(|key| _ = map.erase(key));
    let found = (0..count).filter(|&key| map.find(key).is_some()).count();
    map.clear();
    found + map.len()
}

/// `value` doubled, through a `Box`.
#[unsafe(no_mangle)]
pub extern "C" fn box_double(value: u64) -> u64 {
    let Ok(mut boxed) = Box::new(&Heap, value) else { return 0 };
    *boxed *= 2;
    *boxed
}

/// The most bytes a `Box` of `value` takes from an allocator limited to
/// `limit` bytes, or 0 if it doesn't fit.
#[unsafe(no_mangle)]
pub extern "C" fn limit_peak(value: u64, limit: usize) -> usize {
    let limited = LimitAllocator::new(&Heap, limit);
    let Ok(boxed) = Box::new(&limited, value) else { return 0 };
    drop(boxed);
    limited.peak()
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

/// `value`, in a column filled as a run's are, or 0 if that fails.
#[unsafe(no_mangle)]
pub extern "C" fn run_column(value: i64) -> i64 {
    let query = QueryAllocators::new(&Heap);
    let mut context = Context::for_query(&query);
    let Ok(mut values) = context.values_buffer(8) else { return 0 };
    values.as_mut_slice::<i64>()[0] = value;
    let Ok(column) = ColumnView::new(&mut context, DataType::Int64, values, None) else {
        return 0;
    };
    column.int64s()[0]
}

/// The sum of `row op value` over rows `0..count`, computed as doubles if
/// `doubles`, or 0 if that fails. `op` picks `+ - * / %`; `%` is of
/// integers only.
#[unsafe(no_mangle)]
pub extern "C" fn compute_sum(count: u32, op: u8, value: i64, doubles: bool) -> i64 {
    use ExprOp::{Add, Column, Constant, Div, Mod, Mul, Sub, ToDouble};
    let Some(&op) = [Add, Sub, Mul, Div, Mod].get(usize::from(op)) else { return 0 };
    if doubles && op == Mod {
        return 0;
    }
    let Ok(mut values) = Buffer::allocate(&Heap, count as usize * 8) else { return 0 };
    values.as_mut_slice::<i64>().iter_mut().zip(0..).for_each(|(cell, row)| *cell = row);
    let mut context = Context::new(&Heap);
    let Ok(column) = ColumnView::new(&mut context, DataType::Int64, values, None) else {
        return 0;
    };
    let (data_type, row) = if doubles { (DataType::Float64, 1) } else { (DataType::Int64, 0) };
    #[expect(clippy::cast_precision_loss, reason = "any value will do")]
    let double_value = value as f64;
    let step =
        |op, data_type, a| ExprStep { op, data_type, a, b: 2, int_value: value, double_value };
    let steps = [
        step(Column, DataType::Int64, 0),
        step(ToDouble, DataType::Float64, 0),
        step(Constant, data_type, 0),
        step(op, data_type, row),
    ];
    let Ok(steps) = SlowVec::fixed_from(&Heap, steps.into_iter()) else { return 0 };
    let Ok(programs) = SlowVec::fixed_from(&Heap, [ExprProgram { steps }].into_iter()) else {
        return 0;
    };
    let compute = Compute { programs };
    let Ok(mut state) = compute.new_state(&mut context) else { return 0 };
    let mut batch = RowBatch::new();
    batch.reset(count.min(BATCH_ROWS_MAX));
    if batch.push_column(column.slice(0, batch.row_count())).is_err()
        || compute.process(&mut context, &mut state, &mut batch).is_err()
    {
        return 0;
    }
    #[expect(clippy::cast_possible_truncation, reason = "any value will do")]
    let sum = match data_type {
        DataType::Float64 => batch.column(1).float64s().iter().sum::<f64>() as i64,
        _ => batch.column(1).int64s().iter().sum(),
    };
    sum
}

/// How many rows a lazy column of `rows` rows has, its values unread, or 0
/// if making it fails.
#[unsafe(no_mangle)]
pub extern "C" fn lazy_rows(rows: u32) -> u32 {
    let mut context = Context::new(&Heap);
    let Ok(column) = ColumnView::lazy(&mut context, DataType::Int64, &[0; 8], rows) else {
        return 0;
    };
    if column.is_lazy() { column.row_count() } else { 0 }
}

/// The length of the second of two strings, of `first` and `second` bytes.
#[unsafe(no_mangle)]
pub extern "C" fn second_string_len(first: u32, second: u32) -> usize {
    let Ok(mut views) = Buffer::allocate(&Heap, 2 * 8) else { return 0 };
    views.as_mut_slice::<u32>().copy_from_slice(&[0, first, first, second]);
    let Ok(bytes) = Buffer::allocate(&Heap, (first + second) as usize) else { return 0 };
    let mut context = Context::new(&Heap);
    let Ok(column) = ColumnView::strings(&mut context, views, bytes, None) else { return 0 };
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
/// `data` has, scanned as a table, or 0 if it can't be read.
///
/// # Safety
///
/// `data` must be valid for `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn parquet_rows(data: *const u8, len: usize) -> usize {
    // SAFETY: guaranteed by the caller.
    let bytes: &[u8] = unsafe { core::slice::from_raw_parts(data, len) };
    let Ok(table) = ParquetTable::open(&Heap, &Uncompressed, &[&bytes]) else { return 0 };
    let mut context = Context::new(&Heap);
    let Ok(mut state) = table.open(&mut context, &[(0, Forms::FLAT)]) else { return 0 };
    let mut batch = RowBatch::new();
    let mut rows = 0;
    while let Ok(true) = table.next(&mut context, &mut state, &mut batch) {
        rows += batch.row_count() as usize;
    }
    rows
}

/// The Parquet table a query reads as `hits`.
struct Hits<'a>(DynScannable<'a>);

impl Catalog for Hits<'_> {
    fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
        (name == "hits").then_some(&self.0)
    }
}

/// How many rows `query`, of `query_len` bytes, outputs over the Parquet file
/// in `len` bytes from `data` as `hits`, or `u32::MAX` if it fails.
///
/// # Safety
///
/// `data` must be valid for `len` bytes, and `query` for `query_len`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn query_rows(
    data: *const u8,
    len: usize,
    query: *const u8,
    query_len: usize,
) -> u32 {
    // SAFETY: guaranteed by the caller.
    let (bytes, query): (&[u8], &[u8]) = unsafe {
        (core::slice::from_raw_parts(data, len), core::slice::from_raw_parts(query, query_len))
    };
    let Ok(table) = ParquetTable::open(&Heap, &Uncompressed, &[&bytes]) else { return u32::MAX };
    let Ok(table) = DynScannable::new(&Heap, table) else { return u32::MAX };
    let hits = Hits(table);
    let Ok(mut plan) = compile(&Heap, &REGISTRY, &hits, query) else { return u32::MAX };
    if optimize(&Heap, &mut plan).is_err() {
        return u32::MAX;
    }
    let Ok(planned) = plan_nodes(&Heap, &plan) else { return u32::MAX };
    let Ok(physical) = lower(&Heap, &plan, &planned) else { return u32::MAX };
    let allocators = QueryAllocators::new(&Heap);
    let Ok(mut execution) = physical.pipeline().start(&allocators) else { return u32::MAX };
    let mut batch = RowBatch::new();
    let mut rows = 0;
    loop {
        match execution.next(&mut batch) {
            Ok(true) => rows += batch.selection().len(),
            Ok(false) => return rows,
            Err(_) => return u32::MAX,
        }
    }
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
