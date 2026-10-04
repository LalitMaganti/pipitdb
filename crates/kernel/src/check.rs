//! `check!`: an assertion that costs a few bytes in release builds, and
//! `at!`, indexing that does the same.
//!
//! Debug builds use `assert!`. Release builds drop the message and location
//! and call `check_failed(line!())`. To find a failed check, symbolize the
//! caller's address in the backtrace against a build of the same commit with
//! debug info.

#[macro_export]
macro_rules! check {
    ($condition:expr $(,)?) => {
        if cfg!(debug_assertions) {
            assert!($condition);
        } else if !$condition {
            $crate::check::check_failed(line!());
        }
    };
}

/// `slice[index]`, for an index or a range, without a panic location in
/// release builds: the same trade as `check!`.
#[macro_export]
macro_rules! at {
    ($slice:expr, $index:expr) => {
        if cfg!(debug_assertions) {
            &$slice[$index]
        } else {
            match $slice.get($index) {
                Some(item) => item,
                None => $crate::check::check_failed(line!()),
            }
        }
    };
}

/// `&mut slice[index]`, as `at!`.
#[macro_export]
macro_rules! at_mut {
    ($slice:expr, $index:expr) => {
        if cfg!(debug_assertions) {
            &mut $slice[$index]
        } else {
            match $slice.get_mut($index) {
                Some(item) => item,
                None => $crate::check::check_failed(line!()),
            }
        }
    };
}

/// Each check passes its own line, so the compiler can't merge two failing
/// checks into one call and each keeps its own address.
#[cold]
#[inline(never)]
pub fn check_failed(line: u32) -> ! {
    core::hint::black_box(line);
    panic!("check failed");
}
