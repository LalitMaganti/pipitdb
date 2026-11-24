//! Settings fixed at build time, from environment variables.

/// How deep a query's plan can be: preparing it recurses, taking stack
/// for each level. Set at build time with `PIPIT_PLAN_DEPTH_MAX`.
pub const PLAN_DEPTH_MAX: usize = build_setting(option_env!("PIPIT_PLAN_DEPTH_MAX"), 1000);

/// `value`, a decimal number set at build time, or `default` if it isn't set.
/// Fails the build if it isn't a number.
pub const fn build_setting(value: Option<&str>, default: usize) -> usize {
    let Some(value) = value else { return default };
    let digits = value.as_bytes();
    assert!(!digits.is_empty(), "build settings must be decimal numbers");
    let mut number: usize = 0;
    let mut i = 0;
    while i < digits.len() {
        assert!(digits[i].is_ascii_digit(), "build settings must be decimal numbers");
        number = number * 10 + (digits[i] - b'0') as usize;
        i += 1;
    }
    number
}
