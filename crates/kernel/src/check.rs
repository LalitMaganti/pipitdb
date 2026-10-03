//! `check!`: an assertion that costs a few bytes in release builds.
//!
//! Debug builds use `assert!`. Release builds drop the message and location
//! and call `check_failed(line!())`. To find a failed check, symbolize the
//! caller's address in the backtrace against a build of the same commit with
//! debug info.

macro_rules! check {
    ($condition:expr $(,)?) => {
        if cfg!(debug_assertions) {
            assert!($condition);
        } else if !$condition {
            $crate::check::check_failed(line!());
        }
    };
}

/// Each check passes its own line, so the compiler can't merge two failing
/// checks into one call and each keeps its own address.
#[cold]
#[inline(never)]
pub(crate) fn check_failed(line: u32) -> ! {
    core::hint::black_box(line);
    panic!("check failed");
}
