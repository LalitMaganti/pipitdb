//! `check!`: an assertion that costs a few bytes in release builds, and
//! `at!`, indexing that does the same.
//!
//! Debug builds use `assert!`. Release builds drop the message and call
//! `check_failed`, with the check's line only with the `check-lines`
//! feature. Without it, a function's failing checks can share one call,
//! which saves a few kilobytes, but a crash names only the function: a build
//! with the feature, or a debug build, of the same input says which check.

/// Whether failed checks in release builds say their line.
pub const LINES: bool = cfg!(feature = "check-lines");

/// Fails a check: `check_failed_at` its line if `LINES`, or else
/// `check_failed`.
#[macro_export]
macro_rules! failed {
    () => {
        if $crate::check::LINES {
            $crate::check::check_failed_at(line!())
        } else {
            $crate::check::check_failed()
        }
    };
}

#[macro_export]
macro_rules! check {
    ($condition:expr $(,)?) => {
        if cfg!(debug_assertions) {
            assert!($condition);
        } else if !$condition {
            $crate::failed!();
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
                None => $crate::failed!(),
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
                None => $crate::failed!(),
            }
        }
    };
}

/// Fails a check. Takes nothing, so a function's failing checks can share
/// one call.
#[cold]
#[inline(never)]
pub fn check_failed() -> ! {
    panic!("check failed");
}

/// Fails the check at `line`.
#[cold]
#[inline(never)]
pub fn check_failed_at(line: u32) -> ! {
    core::hint::black_box(line);
    panic!("check failed");
}
