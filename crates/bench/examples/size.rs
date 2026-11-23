//! What an embedder ships: SQL over Parquet files, compiled, planned and run
//! within a memory budget, and every value of its result read. CI measures
//! its size, built as a library as `crates/bench/check_size.sh` does, so a
//! feature counts once SQL can reach it.

#![no_std]

use core::alloc::{GlobalAlloc, Layout};

use pipit_kernel::allocator::Heap;
use pipit_kernel::column::{DataType, Forms};
use pipit_kernel::context::Context;
use pipit_kernel::global::Global;
use pipit_kernel::lower::lower;
use pipit_kernel::optimize::optimize;
use pipit_kernel::output_properties::plan_nodes;
use pipit_kernel::row_batch::RowBatch;
use pipit_kernel::scannable::{Catalog, DynScannable};
use pipit_kernel::slow_vec::SlowVec;
use pipit_parquet::Uncompressed;
use pipit_parquet::table::ParquetTable;
use pipit_pipesql::choices::CHOICES;
use pipit_pipesql::compile::compile;
use pipit_pipesql::registry::Registry;
use pipit_pipesql::stages::{AGGREGATION, JOINS, RELATIONAL};

/// Every stage and function PipeSQL comes with, but those behind features.
static REGISTRY: Registry =
    Registry::new(&[RELATIONAL, AGGREGATION, JOINS]).with_functions(&[CHOICES]);

/// A table a query can read: its name, and its Parquet file.
#[repr(C)]
pub struct TableFile {
    pub name: *const u8,
    pub name_len: usize,
    pub data: *const u8,
    pub len: usize,
}

/// The tables, by name.
struct Tables<'a> {
    names: SlowVec<&'a str>,
    tables: SlowVec<DynScannable<'a>>,
}

impl Catalog for Tables<'_> {
    fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
        let at = self.names.iter().position(|&table| table == name)?;
        self.tables.get(at)
    }
}

/// Runs `query`, of `query_len` bytes, over the `count` tables of `tables`,
/// in `budget` bytes, and returns a hash of every value of its result, or
/// `u64::MAX` if it fails.
///
/// # Safety
///
/// `tables` must be valid for `count` tables, each of whose names and files
/// valid for their lengths, and `query` for `query_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn run_query(
    tables: *const TableFile,
    count: usize,
    query: *const u8,
    query_len: usize,
    budget: usize,
) -> u64 {
    // SAFETY: guaranteed by the caller.
    let (files, query) = unsafe {
        (core::slice::from_raw_parts(tables, count), core::slice::from_raw_parts(query, query_len))
    };
    // SAFETY: everything the query allocates is dropped within `run`.
    let Ok(global) = (unsafe { Global::new(Heap, budget) }) else { return u64::MAX };
    run(&global, files, query).unwrap_or(u64::MAX)
}

/// As `run_query`, with every allocation from `allocator`.
fn run(allocator: &Global, files: &[TableFile], query: &[u8]) -> Option<u64> {
    // The files, which their tables borrow for the query.
    let mut sources: SlowVec<&[u8]> = SlowVec::fixed(allocator, files.len()).ok()?;
    let mut tables = Tables {
        names: SlowVec::fixed(allocator, files.len()).ok()?,
        tables: SlowVec::fixed(allocator, files.len()).ok()?,
    };
    for file in files {
        // SAFETY: guaranteed by `run_query`'s caller.
        let (name, bytes) = unsafe {
            (
                core::slice::from_raw_parts(file.name, file.name_len),
                core::slice::from_raw_parts(file.data, file.len),
            )
        };
        tables.names.push(core::str::from_utf8(name).ok()?).ok()?;
        sources.push(bytes).ok()?;
    }
    for source in sources.iter() {
        let table = ParquetTable::open(allocator, &Uncompressed, &[source]).ok()?;
        tables.tables.push(DynScannable::new(allocator, table).ok()?).ok()?;
    }
    let mut plan = compile(allocator, &REGISTRY, &tables, query).ok()?;
    optimize(allocator, &mut plan).ok()?;
    let planned = plan_nodes(allocator, &plan).ok()?;
    let physical = lower(allocator, &mut plan, &planned).ok()?;
    let mut execution = physical.start(allocator).ok()?;
    let mut context = Context::new(allocator, None);
    let mut batch = RowBatch::new();
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut mix = |word: u64| hash = (hash ^ word).wrapping_mul(0x0100_0000_01b3);
    while execution.next(&mut batch).ok()? {
        for column in physical.columns() {
            let mut values = batch.column(column.position).clone();
            values.make_in(&mut context, Forms::FLAT).ok()?;
            let selection = batch.selection();
            for row in (0..selection.len()).map(|i| selection.kept().row(i)) {
                if values.is_null(row) {
                    mix(u64::MAX);
                    continue;
                }
                match values.data_type() {
                    DataType::Int64 => mix(values.int64s()[row as usize].cast_unsigned()),
                    DataType::Float64 => mix(values.float64s()[row as usize].to_bits()),
                    DataType::String => {
                        for &byte in values.string_values().get(row as usize) {
                            mix(u64::from(byte));
                        }
                    }
                }
            }
        }
    }
    Some(hash)
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
