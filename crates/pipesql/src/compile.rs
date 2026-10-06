//! `compile`: PipeSQL text to a `LogicalPlan`, finding tables in a `Catalog`.
//! Each stage's rule compiles it, so a stage an extension adds compiles
//! the same way.

use pipit_kernel::allocator::Allocator;
use pipit_kernel::plan::{LogicalPlan, NamedColumn, PLAN_COLUMNS_MAX};
use pipit_kernel::scannable::Catalog;
use pipit_kernel::slow_vec::SlowVec;

use crate::ast::{Ast, Node};
use crate::error::{Error, ErrorCode, Span};
use crate::parser::parse_query;
use crate::registry::Registry;

/// A plan being compiled from a query, which each stage's rule adds to.
pub struct Compiler<'q, 'c> {
    source: &'q [u8],
    ast: &'q Ast,
    catalog: &'c dyn Catalog,
    allocator: &'q dyn Allocator,
    pub plan: LogicalPlan<'c>,
    /// The columns the stages so far make, by name.
    pub scope: SlowVec<NamedColumn>,
}

impl<'q, 'c> Compiler<'q, 'c> {
    pub fn node(&self, index: u32) -> Node {
        self.ast.node(index)
    }

    /// The text of a name, such as a table's or a column's. Names are ASCII:
    /// the lexer allows no other bytes in them.
    pub fn text(&self, span: Span) -> &'q str {
        let start = span.start as usize;
        let bytes = at!(self.source, start..start + span.len as usize);
        check!(bytes.is_ascii());
        // SAFETY: ASCII is UTF-8.
        unsafe { core::str::from_utf8_unchecked(bytes) }
    }

    /// The bytes of `span`, such as a string literal's, which can hold any.
    pub fn bytes(&self, span: Span) -> &'q [u8] {
        let start = span.start as usize;
        at!(self.source, start..start + span.len as usize)
    }

    pub fn catalog(&self) -> &'c dyn Catalog {
        self.catalog
    }

    /// What to make plan nodes with.
    pub fn allocator(&self) -> &'q dyn Allocator {
        self.allocator
    }

    /// The column in scope that `name`, a `Name` node, names.
    pub fn find_column(&self, name: Node) -> Result<NamedColumn, Error> {
        let text = self.text(name.span());
        let found = self.scope.iter().find(|column| self.plan.names.get(column.name) == text);
        found.copied().ok_or_else(|| Error::new(ErrorCode::UnknownColumn, name.span()))
    }

    /// Where `node` starts, for errors about it: its leftmost leaf.
    pub fn span(&self, mut node: Node) -> Span {
        while !node.is_leaf() {
            node = self.node(node.first_child());
        }
        node.span()
    }
}

/// Parses `source` with the stages in `registry`, and compiles it, finding
/// tables in `catalog`. The plan's result is the last stage's columns.
pub fn compile<'c>(
    allocator: &dyn Allocator,
    registry: &Registry,
    catalog: &'c dyn Catalog,
    source: &[u8],
) -> Result<LogicalPlan<'c>, Error> {
    let ast = parse_query(allocator, registry, source)?;
    let mut compiler = Compiler {
        source,
        ast: &ast,
        catalog,
        plan: LogicalPlan::new(allocator)?,
        scope: SlowVec::new(allocator, PLAN_COLUMNS_MAX)?,
        allocator,
    };
    let query = ast.node(ast.root());
    for i in 0..query.child_count() {
        let stage = ast.node(query.first_child() + i);
        (registry.rule(stage.rule()).compile)(&mut compiler, stage)?;
    }
    let Compiler { mut plan, scope, .. } = compiler;
    plan.output = scope;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::format;
    use std::vec::Vec as StdVec;

    use pipit_kernel::allocator::Heap;
    use pipit_kernel::buffer::Buffer;
    use pipit_kernel::column::{ColumnView, DataType};
    use pipit_kernel::context::Context;
    use pipit_kernel::lower::lower;
    use pipit_kernel::query_allocators::QueryAllocators;
    use pipit_kernel::row_batch::RowBatch;
    use pipit_kernel::scannable::DynScannable;
    use pipit_kernel::selection::Kept;
    use pipit_operators::table::Table;

    use super::*;
    use crate::error::ErrorCode;
    use crate::stages::RELATIONAL;

    static REGISTRY: Registry = Registry::new(&[RELATIONAL]);

    /// One table, `t`: `a` is 1 and 2, `b` is 10 and 20, `f` is 0.5 and 2.5,
    /// `s` is `it's` and the empty string.
    struct OneTable(DynScannable<'static>);

    impl Catalog for OneTable {
        fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
            (name == "t").then_some(&self.0)
        }
    }

    fn catalog() -> OneTable {
        let column = |data_type, values: [i64; 2]| {
            let mut buffer = Buffer::allocate(&Heap, 16).unwrap();
            buffer.as_mut_slice::<i64>().copy_from_slice(&values);
            ColumnView::new(&mut Context::new(&Heap), data_type, buffer, None).unwrap()
        };
        let floats = [0.5_f64.to_bits().cast_signed(), 2.5_f64.to_bits().cast_signed()];
        let mut views = Buffer::allocate(&Heap, 16).unwrap();
        views.as_mut_slice::<u32>().copy_from_slice(&[0, 4, 4, 0]);
        let mut bytes = Buffer::allocate(&Heap, 4).unwrap();
        bytes.as_mut_slice::<u8>().copy_from_slice(b"it's");
        let strings = ColumnView::strings(&mut Context::new(&Heap), views, bytes, None).unwrap();
        let columns = [
            column(DataType::Int64, [1, 2]),
            column(DataType::Int64, [10, 20]),
            column(DataType::Float64, floats),
            strings,
        ];
        let schema = [
            ("a", DataType::Int64),
            ("b", DataType::Int64),
            ("f", DataType::Float64),
            ("s", DataType::String),
        ];
        let table = Table::new(&Heap, &schema, &[&columns]).unwrap();
        OneTable(DynScannable::new(&Heap, table).unwrap())
    }

    /// Runs `query`, returning each result column's name and the values of
    /// the rows kept, as floats.
    fn run(query: &str) -> StdVec<(StdVec<u8>, StdVec<f64>)> {
        let catalog = catalog();
        let plan = compile(&Heap, &REGISTRY, &catalog, query.as_bytes()).unwrap();
        let physical = lower(&Heap, &plan).unwrap();
        let query = QueryAllocators::new(&Heap);
        let mut execution = physical.pipeline().start(&query).unwrap();
        let mut batch = RowBatch::new();
        let mut columns: StdVec<_> =
            physical.columns().iter().map(|&c| (physical.name(c).into(), StdVec::new())).collect();
        while execution.next(&mut batch).unwrap() {
            let rows: StdVec<usize> = match batch.selection().kept() {
                Kept::All => (0..batch.row_count() as usize).collect(),
                Kept::None => StdVec::new(),
                Kept::Select(rows) => rows.iter().map(|&row| usize::from(row)).collect(),
            };
            for (&c, (_, values)) in physical.columns().iter().zip(&mut columns) {
                let column = batch.column(c.position);
                values.extend(rows.iter().map(|&row| match column.data_type() {
                    #[expect(clippy::cast_precision_loss, reason = "small test values")]
                    DataType::Int64 => column.int64s()[row] as f64,
                    DataType::Float64 => column.float64s()[row],
                    // A string's length, as the tests only need to tell
                    // them apart.
                    #[expect(clippy::cast_precision_loss, reason = "small test values")]
                    DataType::String => column.string_values().get(row).len() as f64,
                }));
            }
        }
        columns
    }

    /// `query`'s values of its one result column.
    fn kept(query: &str) -> StdVec<f64> {
        run(query).remove(0).1
    }

    fn error(query: &str) -> (ErrorCode, u32) {
        let catalog = catalog();
        let error = compile(&Heap, &REGISTRY, &catalog, query.as_bytes()).err().unwrap();
        (error.code, error.span.start)
    }

    #[test]
    fn runs_from_and_select() {
        let a = (b"a".to_vec(), [1.0, 2.0].to_vec());
        let b = (b"b".to_vec(), [10.0, 20.0].to_vec());
        let f = (b"f".to_vec(), [0.5, 2.5].to_vec());
        let s = (b"s".to_vec(), [4.0, 0.0].to_vec());
        assert_eq!(run("FROM t"), [a.clone(), b.clone(), f, s]);
        assert_eq!(run("FROM t |> SELECT b, a"), [b.clone(), a.clone()]);
        assert_eq!(run("FROM t |> SELECT b |> SELECT b"), [b]);
    }

    #[test]
    fn filters_with_where() {
        assert_eq!(kept("FROM t |> WHERE a > 1 |> SELECT b"), [20.0]);
        assert_eq!(kept("FROM t |> WHERE 15 < b AND NOT a = 1 |> SELECT b"), [20.0]);
        assert_eq!(kept("FROM t |> WHERE (a = 1 OR b >= 20) |> SELECT a"), [1.0, 2.0]);
        assert_eq!(kept("FROM t |> WHERE a > -9223372036854775808 |> SELECT a"), [1.0, 2.0]);
        assert_eq!(kept("FROM t |> WHERE a > 5 |> SELECT a"), []);
        // An integer compared with a float column, which holds it exactly.
        assert_eq!(kept("FROM t |> WHERE f > 1 AND f < 3 |> SELECT f"), [2.5]);
    }

    #[test]
    fn filters_strings_with_where() {
        // `''` in a string is a quote.
        assert_eq!(kept("FROM t |> WHERE s = 'it''s' |> SELECT a"), [1.0]);
        assert_eq!(kept("FROM t |> WHERE s <> '' |> SELECT a"), [1.0]);
        // Either way round, and byte by byte: `i` is after `I`.
        assert_eq!(kept("FROM t |> WHERE 'I' < s |> SELECT a"), [1.0]);
        assert_eq!(kept("FROM t |> WHERE s >= '' AND NOT s > 'it' |> SELECT a"), [2.0]);
    }

    #[test]
    fn reports_what_it_cant_compile() {
        assert_eq!(error("FROM u"), (ErrorCode::UnknownTable, 5));
        assert_eq!(error("FROM t |> WHERE a = 'x'"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> WHERE s > 1"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> SELECT c"), (ErrorCode::UnknownColumn, 17));
        assert_eq!(error("FROM t |> SELECT b |> SELECT a"), (ErrorCode::UnknownColumn, 29));
        assert_eq!(error("FROM t |> SELECT a + 1"), (ErrorCode::Unsupported, 17));
        assert_eq!(error("FROM t |> WHERE c > 1"), (ErrorCode::UnknownColumn, 16));
        assert_eq!(error("FROM t |> WHERE a > b"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> WHERE a + 1 > 2"), (ErrorCode::Unsupported, 16));
        assert_eq!(error("FROM t |> WHERE a"), (ErrorCode::Unsupported, 16));
        assert_eq!(error("FROM t |> WHERE f > 1.5"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> WHERE f > 9007199254740993"), (ErrorCode::Unsupported, 20));
        assert_eq!(
            error("FROM t |> WHERE a > 9223372036854775808"),
            (ErrorCode::NumberTooLarge, 20)
        );
        let wide = (0..40).map(|i| format!("a = {i}")).collect::<StdVec<_>>().join(" OR ");
        let wide = format!("FROM t |> WHERE {wide}");
        assert_eq!(error(&wide).0, ErrorCode::ConditionTooLarge);
    }

    #[test]
    fn reads_only_the_columns_selected() {
        let catalog = catalog();
        let mut plan = compile(&Heap, &REGISTRY, &catalog, b"FROM t |> SELECT b").unwrap();
        pipit_kernel::optimize::optimize(&Heap, &mut plan).unwrap();
        let physical = lower(&Heap, &plan).unwrap();
        let query = QueryAllocators::new(&Heap);
        let mut execution = physical.pipeline().start(&query).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch).unwrap());
        assert_eq!(batch.column_count(), 1);
        assert_eq!(batch.column(physical.columns()[0].position).int64s(), [10, 20]);
    }
}
