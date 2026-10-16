//! Matching strings with SQL's `LIKE` patterns, where `%` is any run of bytes
//! and `_` is any one byte, as Perfetto's `GlobMatcher` matches globs: the
//! pattern is split at each `%` into segments, the first must start the
//! string, the last must end it, and the others are found in order between.
//! There's no escape character, as in DuckDB by default.

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

/// Where `byte` first is in `text`. Blocks without it are skipped whole, in a
/// loop the compiler vectorizes, as it can't one that stops at a match.
fn find_byte(text: &[u8], byte: u8) -> Option<usize> {
    const BLOCK: usize = 32;
    let mut at = 0;
    while let Some(block) = text.get(at..at + BLOCK) {
        #[expect(clippy::needless_bitwise_bool, reason = "`||` would stop it vectorizing")]
        let found = block.iter().fold(false, |found, &b| found | (b == byte));
        if found {
            break;
        }
        at += BLOCK;
    }
    at!(text, at..).iter().position(|&b| b == byte).map(|found| at + found)
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
}
