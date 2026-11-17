//! String functions: `length`, as SQLite counts it, and `octet_length`.

use pipit_kernel::column::{ColumnView, DataType};
use pipit_kernel::context::Context;
use pipit_kernel::error::Error;
use pipit_kernel::expression::validity_of;
use pipit_kernel::function::{DynFunction, Function};

use crate::expression::{Bind, call_bound};
use crate::registry::FunctionRule;

pub const STRING_FUNCTIONS: &[FunctionRule] = &[
    FunctionRule {
        name: "length",
        compile: |compiler, node, depth, nodes| {
            let bind: Bind = |c, _| Ok(DynFunction::new(c.allocator(), Length { bytes: false })?);
            call_bound(compiler, (node, depth), nodes, STRING_TO_INT, bind)
        },
    },
    FunctionRule {
        name: "octet_length",
        compile: |compiler, node, depth, nodes| {
            let bind: Bind = |c, _| Ok(DynFunction::new(c.allocator(), Length { bytes: true })?);
            call_bound(compiler, (node, depth), nodes, STRING_TO_INT, bind)
        },
    },
];

/// A string's arguments and an integer result.
const STRING_TO_INT: (&[DataType], DataType) = (&[DataType::String], DataType::Int64);

/// A string's length: in bytes, or in characters before any NUL, as
/// SQLite's `length` counts them.
struct Length {
    bytes: bool,
}

impl Function for Length {
    fn run(&self, context: &mut Context, arguments: &[&ColumnView]) -> Result<ColumnView, Error> {
        let &[text] = arguments else { pipit_kernel::failed!() };
        let rows = text.row_count() as usize;
        let validity = validity_of(context, arguments)?;
        let mut values = context.values_buffer(rows * 8)?;
        let strings = text.string_values();
        for (row, o) in values.as_mut_slice::<i64>().iter_mut().enumerate() {
            #[expect(clippy::cast_possible_wrap, reason = "a string's length fits")]
            let len = strings.len_of(row) as i64;
            // A null's view may not be of its bytes.
            #[expect(clippy::cast_possible_truncation, reason = "a batch's rows")]
            let null = text.is_null(row as u32);
            *o = if self.bytes {
                len
            } else if null {
                0
            } else {
                characters(strings.get(row))
            };
        }
        Ok(ColumnView::new(context, DataType::Int64, values, validity)?)
    }
}

/// The characters in `text` before any NUL: the bytes that don't continue
/// one in UTF-8.
fn characters(text: &[u8]) -> i64 {
    let end = text.iter().position(|&b| b == 0).unwrap_or(text.len());
    #[expect(clippy::cast_possible_wrap, reason = "bytes 0x80 to 0xbf continue")]
    at!(text, ..end).iter().map(|&b| i64::from(b as i8 >= -0x40)).sum()
}
