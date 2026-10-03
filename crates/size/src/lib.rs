//! Uses the kernel so CI can measure its size.

#![no_std]

use core::alloc::{GlobalAlloc, Layout};

use pipit_kernel::allocator::Heap;
use pipit_kernel::buffer::Buffer;

#[unsafe(no_mangle)]
pub extern "C" fn buffer_sum(count: usize, value: i64) -> i64 {
    let Some(size_bytes) = count.checked_mul(size_of::<i64>()) else { return 0 };
    let Ok(mut buffer) = Buffer::allocate(Heap, size_bytes) else { return 0 };
    buffer.as_mut_slice::<i64>().fill(value);
    buffer.clone().as_slice::<i64>().iter().sum()
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
