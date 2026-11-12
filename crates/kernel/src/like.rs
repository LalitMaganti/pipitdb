//! Matching strings with SQL's `LIKE` patterns, where `%` is any run of bytes
//! and `_` is any one byte, as Perfetto's `GlobMatcher` matches globs: the
//! pattern is split at each `%` into segments, the first must start the
//! string, the last must end it, and the others are found in order between.
//! There's no escape character, as in DuckDB by default.

use core::cell::Cell;

/// Whether `text` matches `pattern`.
#[inline(never)]
pub fn like(text: &[u8], pattern: &[u8]) -> bool {
    let mut segments = pattern.split(|&byte| byte == b'%');
    let Some(first) = segments.next() else { failed!() };
    if !starts_with(text, first) {
        return false;
    }
    let rest = at!(text, first.len()..);
    let Some(last) = segments.next_back() else {
        return rest.is_empty();
    };
    let Some(mut middle) = rest.len().checked_sub(last.len()).map(|end| at!(rest, ..end)) else {
        return false;
    };
    if !starts_with(at!(rest, middle.len()..), last) {
        return false;
    }
    for segment in segments {
        let Some(at) = find(middle, segment) else { return false };
        middle = at!(middle, at + segment.len()..);
    }
    true
}

/// The middle of a pattern `%middle%`, if the middle has no `%` or `_`: the
/// strings it matches are those containing the middle.
pub fn contained(pattern: &[u8]) -> Option<&[u8]> {
    let middle = pattern.strip_prefix(b"%")?.strip_suffix(b"%")?;
    let plain = !middle.is_empty() && !middle.iter().any(|&byte| byte == b'%' || byte == b'_');
    plain.then_some(middle)
}

/// Which strings of one buffer contain a needle. Each search runs on past
/// the string it's for, to the needle's next place, which answers every
/// string between: strings in order, as a page's are, are searched in one
/// pass, not one each, as Pivot's `NeedleSearcher` does.
pub struct Contains<'a> {
    bytes: &'a [u8],
    needle: &'a [u8],
    /// Where the last search started.
    from: Cell<usize>,
    /// Where the last search ended: the needle's first place after `from`
    /// is `next`, if it's before here.
    until: Cell<usize>,
    /// The needle's first place after `from`, or `usize::MAX` if it isn't
    /// before `until`.
    next: Cell<usize>,
}

impl<'a> Contains<'a> {
    /// How far a search goes past the string it's for, which bounds what a
    /// batch's last search reads past its last string.
    const AHEAD: usize = 4096;

    pub fn new(bytes: &'a [u8], needle: &'a [u8]) -> Self {
        check!(!needle.is_empty());
        let (from, until, next) = (Cell::new(usize::MAX), Cell::new(0), Cell::new(usize::MAX));
        Contains { bytes, needle, from, until, next }
    }

    /// Whether `text`, a string in the bytes, contains the needle.
    #[inline(never)]
    pub fn within(&self, text: &[u8]) -> bool {
        let start = text.as_ptr().addr().wrapping_sub(self.bytes.as_ptr().addr());
        let end = start.wrapping_add(text.len());
        check!(start <= end && end <= self.bytes.len());
        let next = self.next.get();
        let known = self.from.get() <= start
            && if next == usize::MAX { end <= self.until.get() } else { next >= start };
        if !known {
            let until = end.max(start.saturating_add(Self::AHEAD).min(self.bytes.len()));
            let found = find(at!(self.bytes, start..until), self.needle);
            self.from.set(start);
            self.until.set(until);
            self.next.set(found.map_or(usize::MAX, |at| start + at));
        }
        self.next.get().saturating_add(self.needle.len()) <= end
    }
}

/// Whether `text` starts with `segment`, whose `_`s match any byte.
fn starts_with(text: &[u8], segment: &[u8]) -> bool {
    text.len() >= segment.len()
        && text.iter().zip(segment).all(|(&byte, &want)| want == b'_' || byte == want)
}

/// Where `segment` first starts in `text`, its `_`s matching any byte: at
/// each place its first byte is, as `string_view::find` looks, unless that's
/// a `_`.
fn find(text: &[u8], segment: &[u8]) -> Option<usize> {
    let starts = text.len().checked_sub(segment.len())?;
    let Some(&first) = segment.first().filter(|&&first| first != b'_') else {
        return (0..=starts).find(|&at| starts_with(at!(text, at..), segment));
    };
    let mut at = 0;
    while at <= starts {
        at += find_byte(at!(text, at..=starts), first)?;
        if starts_with(at!(text, at..), segment) {
            return Some(at);
        }
        at += 1;
    }
    None
}

/// Where `byte` first is in `text`. Blocks of 32 bytes without it are skipped
/// whole.
fn find_byte(text: &[u8], byte: u8) -> Option<usize> {
    let mut at = 0;
    while let Some(block) = text.get(at..at + 32).and_then(<[u8]>::first_chunk::<32>) {
        if block::holds(block, byte) {
            break;
        }
        at += 32;
    }
    at!(text, at..).iter().position(|&b| b == byte).map(|found| at + found)
}

/// Whether a block of 32 bytes holds a byte, compared with every byte at once
/// with each target's vector instructions. Left to itself, the compiler
/// vectorizes the comparison or not depending on what it's inlined into.
mod block {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline]
    pub(super) fn holds(block: &[u8; 32], byte: u8) -> bool {
        use core::arch::x86_64::{
            __m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_testz_si256,
        };
        // SAFETY: the build has AVX2, and the load reads `block`'s 32 bytes.
        unsafe {
            #[expect(clippy::cast_ptr_alignment, reason = "an unaligned load")]
            let bytes = _mm256_loadu_si256(block.as_ptr().cast::<__m256i>());
            let equal = _mm256_cmpeq_epi8(bytes, _mm256_set1_epi8(byte.cast_signed()));
            _mm256_testz_si256(equal, equal) == 0
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[inline]
    pub(super) fn holds(block: &[u8; 32], byte: u8) -> bool {
        use core::arch::aarch64::{vceqq_u8, vdupq_n_u8, vld1q_u8, vmaxvq_u8, vorrq_u8};
        // SAFETY: NEON is part of aarch64, and the loads read `block`'s 32
        // bytes.
        unsafe {
            let byte = vdupq_n_u8(byte);
            let half = |at: usize| vceqq_u8(vld1q_u8(block.as_ptr().add(at)), byte);
            vmaxvq_u8(vorrq_u8(half(0), half(16))) != 0
        }
    }

    /// Anywhere else: a fold the compiler can vectorize, as it can't a loop
    /// that stops at a match. On wasm, explicit vectors made the build
    /// bigger, with no benchmark there to show a gain.
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon"),
    )))]
    #[inline]
    #[expect(clippy::needless_bitwise_bool, reason = "`||` would stop it vectorizing")]
    pub(super) fn holds(block: &[u8; 32], byte: u8) -> bool {
        block.iter().fold(false, |found, &b| found | (b == byte))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_as_sql_does() {
        let cases: &[(&str, &str, bool)] = &[
            ("google", "google", true),
            ("google", "goog", false),
            ("www.google.com", "%google%", true),
            ("www.yandex.ru", "%google%", false),
            ("google.com", "google%", true),
            ("x.google", "%google", true),
            ("x.google.y", "%google", false),
            ("abc", "a_c", true),
            ("abbc", "a_c", false),
            ("a.b.c", "%.%.%", true),
            ("a.b", "%.%.%", false),
            ("ab", "a%%b", true),
            ("aXbYc", "a%b%c", true),
            ("acb", "a%b%c", false),
            ("", "%", true),
            ("", "", true),
            ("a", "", false),
            ("ab", "%ab%b", false),
            // The first byte, again and again before the match, and in a long
            // string past the first block.
            ("gggoogle", "%google%", true),
            ("googoogle", "%google", true),
            ("________________________________________google", "%google%", true),
            ("________________________________________googl", "%google%", false),
        ];
        for &(text, pattern, expected) in cases {
            assert_eq!(
                like(text.as_bytes(), pattern.as_bytes()),
                expected,
                "{text} LIKE {pattern}"
            );
        }
    }

    #[test]
    fn contains_as_like_does() {
        assert_eq!(contained(b"%google%"), Some(&b"google"[..]));
        assert_eq!(contained(b"%goo_le%"), None);
        assert_eq!(contained(b"%a%b%"), None);
        assert_eq!(contained(b"%%"), None);
        assert_eq!(contained(b"google%"), None);

        // Strings back to back, one with the needle across its end, then
        // one past a search's reach, then the same again out of order.
        let mut bytes = alloc::vec::Vec::from(&b"xgooglexgoogxgooglexx"[..]);
        bytes.extend([b'_'; 2 * Contains::AHEAD]);
        bytes.extend(b"google");
        let ranges =
            [0..8, 8..12, 12..14, 14..21, 21..bytes.len() - 6, bytes.len() - 7..bytes.len()];
        let mut order: alloc::vec::Vec<usize> = (0..ranges.len()).collect();
        order.extend([5, 0, 3, 1, 4, 2]);
        let contains = Contains::new(&bytes, b"google");
        for i in order {
            let range = ranges[i].clone();
            let expected = like(&bytes[range.clone()], b"%google%");
            assert_eq!(contains.within(&bytes[range.clone()]), expected, "{range:?}");
        }
    }
}
