//! Real data: the slices Perfetto exported for its own benchmarks, in
//! `test/data` of a Perfetto checkout at `$PIPIT_PERFETTO` (`~/perfetto` by
//! default). Not in CI, which has no checkout.

use std::collections::HashMap;
use std::path::PathBuf;

use pipit_kernel::allocator::Heap;
use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType};
use pipit_kernel::lower::{PhysicalPlan, lower};
use pipit_kernel::scannable::{Catalog, DynScannable};
use pipit_operators::table::Table;
use pipit_pipesql::compile::compile;

use crate::REGISTRY;

/// Rows per row group, as a Parquet writer might use.
const ROW_GROUP_ROWS: u32 = 1 << 14;

fn data_dir() -> PathBuf {
    let perfetto = std::env::var_os("PIPIT_PERFETTO").map_or_else(
        || PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join("perfetto"),
        PathBuf::from,
    );
    perfetto.join("test/data")
}

/// The `slice` table, or `None` if there's no Perfetto checkout.
pub fn slices() -> Option<&'static DynScannable<'static>> {
    let text = std::fs::read_to_string(data_dir().join("slice_table_for_benchmarks.csv")).ok()?;
    let table = csv_table(&text)?;
    Some(Box::leak(Box::new(DynScannable::new(Heap, table).ok()?)))
}

/// `query` over `slices`, compiled and lowered, ready to run.
pub fn slice_query(
    slices: &'static DynScannable<'static>,
    query: &str,
) -> Option<PhysicalPlan<'static>> {
    let catalog: &'static Slices = Box::leak(Box::new(Slices(slices)));
    let plan = compile(Heap, &REGISTRY, catalog, query.as_bytes()).ok()?;
    lower(Heap, &plan).ok()
}

struct Slices(&'static DynScannable<'static>);

impl Catalog for Slices {
    fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
        (name == "slice").then_some(self.0)
    }
}

/// A table of CSV `text`, with a header. Every column is `Int64`: one whose
/// values aren't all numbers holds an id for each distinct string, and
/// `[NULL]` or nothing is null.
fn csv_table(text: &str) -> Option<Table> {
    let mut lines = text.lines();
    let names = fields(lines.next()?);
    let mut columns: Vec<Vec<Option<String>>> = vec![Vec::new(); names.len()];
    for line in lines {
        for (column, value) in columns.iter_mut().zip(fields(line)) {
            column.push((!value.is_empty() && value != "[NULL]").then_some(value));
        }
    }
    let rows = u32::try_from(columns.first()?.len()).ok()?;
    let views: Vec<ColumnView> =
        columns.iter().map(|values| int64s(values)).collect::<Option<_>>()?;
    let schema: Vec<(&str, DataType)> =
        names.iter().map(|name| (name.as_str(), DataType::Int64)).collect();
    let groups: Vec<Vec<ColumnView>> = (0..rows)
        .step_by(ROW_GROUP_ROWS as usize)
        .map(|start| {
            let len = ROW_GROUP_ROWS.min(rows - start);
            views.iter().map(|view| view.slice(start, len)).collect()
        })
        .collect();
    let groups: Vec<&[ColumnView]> = groups.iter().map(Vec::as_slice).collect();
    Table::new(Heap, &schema, &groups).ok()
}

/// One column: numbers as they are, or ids for strings, with a validity
/// bitmap if any is null.
fn int64s(values: &[Option<String>]) -> Option<ColumnView> {
    let numbers = values.iter().flatten().all(|value| value.parse::<i64>().is_ok());
    let mut ids = HashMap::new();
    let mut data = Buffer::allocate(Heap, values.len() * 8).ok()?;
    let mut validity = Buffer::allocate(Heap, values.len().div_ceil(8)).ok()?;
    let mut any_null = false;
    for (i, value) in values.iter().enumerate() {
        let Some(value) = value else {
            any_null = true;
            continue;
        };
        let next = i64::try_from(ids.len()).ok()?;
        let number =
            if numbers { value.parse().ok()? } else { *ids.entry(value.as_str()).or_insert(next) };
        data.as_mut_slice::<i64>()[i] = number;
        validity.as_mut_slice::<u8>()[i / 8] |= 1 << (i % 8);
    }
    Some(ColumnView::new(DataType::Int64, data, any_null.then_some(validity)))
}

/// The fields of a CSV line: quoted ones can hold commas, and `""` for a
/// quote.
fn fields(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                chars.next();
                field.push('"');
            }
            '"' => quoted = !quoted,
            ',' if !quoted => fields.push(std::mem::take(&mut field)),
            c => field.push(c),
        }
    }
    fields.push(field);
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_csv_fields() {
        assert_eq!(fields(r#"1,"a, b","say ""hi""",,x"#), ["1", "a, b", r#"say "hi""#, "", "x"]);
    }
}
