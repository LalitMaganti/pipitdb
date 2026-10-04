//! What every frontend needs to turn its errors into messages, in the style
//! of rustc. Frontends write the messages; this lays them out.
//!
//! Errors carry no text, so that small builds stay small. They can be printed
//! in a compact form, `pipit:E0007:2+0:5` (code, span start and length,
//! detail), and explained later against the query.

#![no_std]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use core::fmt::Write;

/// An error as numbers, which is all its compact form holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Compact {
    pub code: u16,
    pub start: u32,
    pub len: u32,
    pub detail: u16,
}

impl Compact {
    /// The error `text`, as `format` writes it, describes; `None` if it isn't
    /// one.
    pub fn parse(text: &str) -> Option<Compact> {
        let rest = text.strip_prefix("pipit:E")?;
        let mut parts = rest.split(':');
        let code = parts.next()?.parse().ok()?;
        let (start, len) = parts.next()?.split_once('+')?;
        let detail = parts.next()?.parse().ok()?;
        if parts.next().is_some() {
            return None;
        }
        Some(Compact { code, start: start.parse().ok()?, len: len.parse().ok()?, detail })
    }

    pub fn format(&self) -> String {
        let Compact { code, start, len, detail } = self;
        format!("pipit:E{code:04}:{start}+{len}:{detail}")
    }
}

/// What a message says: a title, and a label under the source.
pub struct Message {
    pub title: String,
    pub label: String,
}

/// `message`, about error `code` at bytes `start..start + len` of `source`,
/// which is called `name`:
///
/// ```text
/// error[E0007]: expected `)`, found the end of the query
///  --> query:1:3
///   |
/// 1 | (a
///   |   ^ expected `)`
/// ```
pub fn render(
    source: &str,
    name: &str,
    code: u16,
    start: u32,
    len: u32,
    message: &Message,
) -> String {
    let (line_number, line, column) = locate(source, start as usize);
    let width = underline_width(line, column, len as usize);
    let gutter = " ".repeat(line_number.to_string().len());

    let mut out = format!("error[E{code:04}]: {}\n", message.title);
    let _ = writeln!(out, "{gutter}--> {name}:{line_number}:{}", column + 1);
    let _ = writeln!(out, "{gutter} |");
    let _ = writeln!(out, "{line_number} | {line}");
    let carets = "^".repeat(width);
    let _ = writeln!(out, "{gutter} | {}{carets} {}", " ".repeat(column), message.label);
    out
}

/// What bytes `start..start + len` of `source` hold, for "found ...".
pub fn found(source: &str, start: u32, len: u32) -> String {
    let start = start as usize;
    match source.get(start..start + len as usize) {
        Some("") | None => "the end of the query".into(),
        Some(text) => format!("`{text}`"),
    }
}

/// The 1-based line number of byte `offset`, that line's text, and the
/// 0-based column of `offset` in characters.
fn locate(source: &str, offset: usize) -> (usize, &str, usize) {
    let offset = offset.min(source.len());
    let line_start = source[..offset].rfind('\n').map_or(0, |i| i + 1);
    let line_end = source[offset..].find('\n').map_or(source.len(), |i| offset + i);
    let line_number = source[..line_start].matches('\n').count() + 1;
    let column = source[line_start..offset].chars().count();
    (line_number, &source[line_start..line_end], column)
}

/// How many carets to draw: the span's characters on its first line, and at
/// least one.
fn underline_width(line: &str, column: usize, len: usize) -> usize {
    let rest: String = line.chars().skip(column).collect();
    let covered = rest.char_indices().take_while(|&(i, _)| i < len).count();
    covered.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_a_message() {
        let message = Message { title: "bad thing".into(), label: "here".into() };
        assert_eq!(
            render("a +\n  (b * c", "query", 7, 6, 2, &message),
            "error[E0007]: bad thing\n --> query:2:3\n  |\n2 |   (b * c\n  |   ^^ here\n"
        );
    }

    #[test]
    fn compact_round_trips() {
        let compact = Compact { code: 6, start: 2, len: 1, detail: 0 };
        assert_eq!(compact.format(), "pipit:E0006:2+1:0");
        assert_eq!(Compact::parse("pipit:E0006:2+1:0"), Some(compact));
        assert_eq!(Compact::parse("pipit:E0006:2+1"), None);
    }
}
