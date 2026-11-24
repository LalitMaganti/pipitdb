//! Regular expressions in expressions: `regexp_replace`. The engine is a
//! feature: `lite`, the small `regex-lite`, or `full`, the fast `regex`.

#[cfg(not(any(feature = "lite", feature = "full")))]
compile_error!("pipitdb-regex needs an engine: the `lite` or `full` feature");

use pipit_kernel::allocator::AllocError;
use pipit_kernel::column::{ColumnView, DataType};
use pipit_kernel::containers::SmallVec;
use pipit_kernel::context::Context;
use pipit_kernel::error::Error;
use pipit_kernel::function::validity_of;
use pipit_kernel::function::{DynFunction, Function};
use pipit_pipesql::ast::Node;
use pipit_pipesql::compile::Compiler;
use pipit_pipesql::error::{Error as CompileError, ErrorCode};
use pipit_pipesql::expression::{Bind, call_bound};
use pipit_pipesql::registry::FunctionRule;

pub const REGEX_FUNCTIONS: &[FunctionRule] = &[FunctionRule {
    name: "regexp_replace",
    compile: |compiler, node, depth| call_bound(compiler, (node, depth), STRINGS, bind as Bind),
}];

/// Three strings, and a string result.
const STRINGS: (&[DataType], DataType) = (&[DataType::String; 3], DataType::String);

/// Every match of `pattern` in a string, replaced with `replacement`: in
/// which `\0` is the match, `\1` to `\9` its groups, and `\\` a backslash, as
/// in `ClickHouse`'s and `DuckDB`'s.
struct Replace {
    pattern: engine::Pattern,
    replacement: SmallVec<u8>,
    /// Whether the replacement has a backslash: if not, it's as it is, and
    /// the matches' groups, which cost more to find, needn't be.
    groups: bool,
}

/// Binds the call `node`, whose pattern and replacement must be string
/// literals, compiling the pattern once.
fn bind(compiler: &mut Compiler<'_, '_>, node: Node) -> Result<DynFunction, CompileError> {
    let argument = |compiler: &Compiler<'_, '_>, i| compiler.node(node.first_child() + i);
    let invalid = |compiler: &Compiler<'_, '_>, i| {
        CompileError::new(ErrorCode::InvalidArgument, compiler.span(argument(compiler, i)))
    };
    let pattern = compiler.string_literal(argument(compiler, 2))?;
    let Some(pattern) = pattern.as_deref().and_then(engine::compile) else {
        return Err(invalid(compiler, 2));
    };
    let env = compiler.env();
    let Some(replacement) = compiler.string_literal(argument(compiler, 3))? else {
        return Err(invalid(compiler, 3));
    };
    let groups = replacement.contains(&b'\\');
    Ok(DynFunction::new(env, Replace { pattern, replacement, groups })?)
}

impl Function for Replace {
    fn run(&self, context: &mut Context, arguments: &[&ColumnView]) -> Result<ColumnView, Error> {
        let Some(&text) = arguments.first() else { pipit_kernel::failed!() };
        let rows = text.row_count() as usize;
        let validity = validity_of(context, arguments)?;
        let strings = text.string_values();
        let mut views = context.values_buffer(rows * 8)?;
        let mut out = SmallVec::new(context.env(), 1 << 31)?;
        for (row, view) in views.as_mut_slice::<u32>().as_chunks_mut::<2>().0.iter_mut().enumerate()
        {
            let start = out.len();
            // A null's view may not be of its bytes.
            #[expect(clippy::cast_possible_truncation, reason = "a batch's rows")]
            if !text.is_null(row as u32) {
                self.replace(strings.get(row), &mut out)?;
            }
            #[expect(clippy::cast_possible_truncation, reason = "at most `1 << 31` bytes")]
            let view_of = [start as u32, (out.len() - start) as u32];
            *view = view_of;
        }
        let mut bytes = context.bytes_buffer(out.len())?;
        bytes.as_mut_slice::<u8>().copy_from_slice(&out);
        Ok(ColumnView::strings(context, views, bytes, validity)?)
    }
}

impl Replace {
    /// Appends `text`, with each match replaced, to `out`.
    fn replace(&self, text: &[u8], out: &mut SmallVec<u8>) -> Result<(), AllocError> {
        let mut last = 0;
        each_match(&self.pattern, text, self.groups, |start, end, group| {
            out.extend_from_slice(at(text, last, start))?;
            self.expand(group, text, out)?;
            last = end;
            Ok(())
        })?;
        out.extend_from_slice(at(text, last, text.len()))
    }

    /// Appends the replacement for a match of `text` whose groups are
    /// `group`'s, by number, to `out`.
    fn expand(
        &self,
        group: &dyn Fn(usize) -> Option<(usize, usize)>,
        text: &[u8],
        out: &mut SmallVec<u8>,
    ) -> Result<(), AllocError> {
        if !self.groups {
            return out.extend_from_slice(&self.replacement);
        }
        let mut bytes = self.replacement.iter();
        while let Some(&byte) = bytes.next() {
            let escaped = (byte == b'\\').then(|| bytes.clone().next()).flatten();
            match escaped {
                Some(&digit) if digit.is_ascii_digit() => {
                    bytes.next();
                    if let Some((start, end)) = group(usize::from(digit - b'0')) {
                        out.extend_from_slice(at(text, start, end))?;
                    }
                }
                Some(&b'\\') => {
                    bytes.next();
                    out.push(b'\\')?;
                }
                _ => out.push(byte)?,
            }
        }
        Ok(())
    }
}

/// `text[start..end]`, which the engine found.
fn at(text: &[u8], start: usize, end: usize) -> &[u8] {
    text.get(start..end).unwrap_or_default()
}

/// Calls `each` with each match of `pattern` in `text`'s start and end, in
/// order, and its groups' if `groups`.
fn each_match(
    pattern: &engine::Pattern,
    text: &[u8],
    groups: bool,
    mut each: impl FnMut(
        usize,
        usize,
        &dyn Fn(usize) -> Option<(usize, usize)>,
    ) -> Result<(), AllocError>,
) -> Result<(), AllocError> {
    let Some(text) = engine::text(text) else { return Ok(()) };
    if !groups {
        for found in pattern.find_iter(text) {
            each(found.start(), found.end(), &|_| None)?;
        }
        return Ok(());
    }
    for captures in pattern.captures_iter(text) {
        let Some(found) = captures.get(0) else { continue };
        each(found.start(), found.end(), &|i| captures.get(i).map(|g| (g.start(), g.end())))?;
    }
    Ok(())
}

#[cfg(feature = "full")]
mod engine {
    pub type Pattern = regex::bytes::Regex;

    pub fn compile(pattern: &[u8]) -> Option<Pattern> {
        Pattern::new(core::str::from_utf8(pattern).ok()?).ok()
    }

    /// `text`, as the engine matches it.
    #[expect(clippy::unnecessary_wraps, reason = "as `lite`'s, which can fail")]
    pub fn text(text: &[u8]) -> Option<&[u8]> {
        Some(text)
    }
}

/// Matches strings that are UTF-8: one that isn't has no matches.
#[cfg(all(feature = "lite", not(feature = "full")))]
mod engine {
    pub type Pattern = regex_lite::Regex;

    pub fn compile(pattern: &[u8]) -> Option<Pattern> {
        Pattern::new(core::str::from_utf8(pattern).ok()?).ok()
    }

    /// `text`, as the engine matches it.
    pub fn text(text: &[u8]) -> Option<&str> {
        core::str::from_utf8(text).ok()
    }
}

#[cfg(test)]
mod tests {
    use pipit_kernel::buffer::Buffer;
    use pipit_kernel::env::Env;
    use pipit_kernel::scannable::{Catalog, DynScannable};
    use pipit_kernel::selection::Kept;
    use pipit_kernel::session::Session;
    use pipit_kernel::table::Table;
    use pipit_pipesql::compile::compile;
    use pipit_pipesql::registry::Registry;
    use pipit_pipesql::stages::RELATIONAL;

    use super::*;

    static REGISTRY: Registry = Registry::new(&[RELATIONAL]).with_functions(&[REGEX_FUNCTIONS]);

    const ROWS: [&str; 3] = ["https://www.example.com/a/b", "plain", "a1b22"];

    /// Table `t`, whose `s` is `ROWS`.
    struct Tables(DynScannable<'static>);

    impl Catalog for Tables {
        fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
            (name == "t").then_some(&self.0)
        }
    }

    fn catalog() -> Tables {
        let text = ROWS.concat();
        let mut views = Buffer::allocate(Env::current(), ROWS.len() * 8).unwrap();
        let mut start = 0;
        for (view, row) in views.as_mut_slice::<u32>().chunks_mut(2).zip(ROWS) {
            view.copy_from_slice(&[start, u32::try_from(row.len()).unwrap()]);
            start += u32::try_from(row.len()).unwrap();
        }
        let mut bytes = Buffer::allocate(Env::current(), text.len()).unwrap();
        bytes.as_mut_slice::<u8>().copy_from_slice(text.as_bytes());
        let s = ColumnView::strings(&mut Context::new(Env::current(), None), views, bytes, None)
            .unwrap();
        let table = Table::new(Env::current(), &[("s", DataType::String)], &[&[s]]).unwrap();
        Tables(DynScannable::new(Env::current(), table).unwrap())
    }

    /// The strings of `query`'s one column, a null as `None`.
    fn strings(query: &str) -> Vec<Option<String>> {
        let catalog = catalog();
        let mut session = Session::new(Env::current());
        let query = compile(Env::current(), &REGISTRY, &catalog, query.as_bytes()).unwrap();
        let prepared = session.prepare(query).unwrap();
        let mut rows = prepared.run(&mut session).unwrap();
        let mut out = Vec::new();
        while let Some(batch) = rows.next().unwrap() {
            assert!(matches!(batch.selection().kept(), Kept::Prefix));
            let column = batch.column(prepared.columns()[0].position);
            for row in 0..batch.selection().len() {
                let string = column.string_values().get(row as usize);
                let string = String::from_utf8(string.to_vec()).unwrap();
                out.push((!column.is_null(row)).then_some(string));
            }
        }
        out
    }

    fn error(query: &str) -> (ErrorCode, u32) {
        let catalog = catalog();
        let error = compile(Env::current(), &REGISTRY, &catalog, query.as_bytes()).err().unwrap();
        (error.code, error.span.start)
    }

    #[test]
    fn replaces_matches() {
        let some = |rows: [&str; 3]| rows.map(|row| Some(row.to_string())).to_vec();
        // ClickBench's Q28: a URL's host, and other strings as they are.
        let host =
            r"FROM t |> SELECT regexp_replace(s, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS x";
        assert_eq!(strings(host), some(["example.com", "plain", "a1b22"]));
        let every = r"FROM t |> SELECT regexp_replace(s, '[0-9]+', '<\0\\>') AS x";
        assert_eq!(strings(every), some(["https://www.example.com/a/b", "plain", r"a<1\>b<22\>"]));
        // With no backslash, the replacement is as it is.
        let plain = r"FROM t |> SELECT regexp_replace(s, '[0-9]', '#') AS x";
        assert_eq!(strings(plain), some(["https://www.example.com/a/b", "plain", "a#b##"]));
        let null = "FROM t |> SELECT regexp_replace(NULL, 'a', 'b') AS x";
        assert_eq!(strings(null), [None, None, None]);
    }

    #[test]
    fn needs_a_valid_literal_pattern() {
        let column = "FROM t |> SELECT regexp_replace(s, s, 'x') AS x";
        assert_eq!(error(column), (ErrorCode::InvalidArgument, 35));
        let invalid = "FROM t |> SELECT regexp_replace(s, '(', 'x') AS x";
        assert_eq!(error(invalid), (ErrorCode::InvalidArgument, 35));
    }
}
