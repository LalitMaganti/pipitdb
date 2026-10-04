//! Uses the kernel so CI can measure its size. Built as a library, as
//! `crates/bench/check_size.sh` does.

#![no_std]

use core::alloc::{GlobalAlloc, Layout};

use pipit_kernel::allocator::Heap;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType};
use pipit_pipesql::lexer::{Lexer, TokenKind};
use pipit_pipesql::parser::{parse_expression, parse_query};
use pipit_pipesql::registry::Registry;

static REGISTRY: Registry = Registry::new(&[pipit_pipesql::stages::RELATIONAL]);
use pipit_kernel::row_batch::RowBatch;
use pipit_kernel::vec::Vec;

#[unsafe(no_mangle)]
pub extern "C" fn column_sum(count: u32, value: i64) -> i64 {
    let Ok(mut values) = Buffer::allocate(Heap, count as usize * size_of::<i64>()) else {
        return 0;
    };
    values.as_mut_slice::<i64>().fill(value);
    let column = ColumnView::new(DataType::Int64, values, None).slice(1, count - 1);
    let mut batch = RowBatch::new();
    batch.reset(column.row_count());
    if batch.push_column(column).is_err() || batch.column(0).is_null(0) {
        return 0;
    }
    batch.column(0).int64s().iter().sum()
}

/// Pushes `0..count` to a `Vec`, and returns the last.
#[unsafe(no_mangle)]
pub extern "C" fn vec_sum(count: u32) -> u64 {
    let Ok(mut values) = Vec::new(Heap) else { return 0 };
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
    match parse_expression(Heap, source) {
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
    match parse_query(Heap, &REGISTRY, source) {
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
