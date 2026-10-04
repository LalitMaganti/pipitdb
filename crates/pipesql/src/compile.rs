//! `compile`: PipeSQL text to a `LogicalPlan`, finding tables in a `Catalog`.
//! Each stage's rule compiles it, so a stage an extension adds compiles
//! the same way.

use pipit_kernel::allocator::{Allocator, DynAllocator};
use pipit_kernel::plan::{LogicalPlan, NamedColumn, PLAN_COLUMNS_MAX};
use pipit_kernel::scannable::Catalog;
use pipit_kernel::vec::Vec;

use crate::ast::{Ast, Node};
use crate::error::{Error, Span};
use crate::parser::parse_query;
use crate::registry::Registry;

/// A plan being compiled from a query, which each stage's rule adds to.
pub struct Compiler<'q, 'c> {
    source: &'q [u8],
    ast: &'q Ast,
    catalog: &'c dyn Catalog,
    allocator: DynAllocator,
    pub plan: LogicalPlan<'c>,
    /// The columns the stages so far make, by name.
    pub scope: Vec<NamedColumn>,
}

impl<'q, 'c> Compiler<'q, 'c> {
    pub fn node(&self, index: u32) -> Node {
        self.ast.node(index)
    }

    /// The text of a name, such as a table's or a column's. Names are ASCII
    /// for now: any other is empty, so matches nothing.
    pub fn text(&self, span: Span) -> &'q str {
        let start = span.start as usize;
        let bytes = at!(self.source, start..start + span.len as usize);
        if !bytes.is_ascii() {
            return "";
        }
        // SAFETY: ASCII is UTF-8.
        unsafe { core::str::from_utf8_unchecked(bytes) }
    }

    pub fn catalog(&self) -> &'c dyn Catalog {
        self.catalog
    }

    /// What to make plan nodes with.
    pub fn allocator(&self) -> DynAllocator {
        self.allocator.clone()
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
pub fn compile<'c, A: Allocator + Clone + 'static>(
    allocator: A,
    registry: &Registry,
    catalog: &'c dyn Catalog,
    source: &[u8],
) -> Result<LogicalPlan<'c>, Error> {
    let ast = parse_query(allocator.clone(), registry, source)?;
    let allocator = DynAllocator::new(allocator)?;
    let mut compiler = Compiler {
        source,
        ast: &ast,
        catalog,
        plan: LogicalPlan::new(allocator.clone())?,
        scope: Vec::new(allocator.clone(), PLAN_COLUMNS_MAX)?,
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

    use std::vec::Vec as StdVec;

    use pipit_kernel::allocator::Heap;
    use pipit_kernel::buffer::Buffer;
    use pipit_kernel::column::{ColumnView, DataType};
    use pipit_kernel::lower::lower;
    use pipit_kernel::row_batch::RowBatch;
    use pipit_kernel::scannable::DynScannable;
    use pipit_operators::table::Table;

    use super::*;
    use crate::error::ErrorCode;
    use crate::stages::RELATIONAL;

    static REGISTRY: Registry = Registry::new(&[RELATIONAL]);

    /// One table, `t`: `a` is 1 and 2, `b` is 10 and 20.
    struct OneTable(DynScannable<'static>);

    impl Catalog for OneTable {
        fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
            (name == "t").then_some(&self.0)
        }
    }

    fn catalog() -> OneTable {
        let column = |values: [i64; 2]| {
            let mut buffer = Buffer::allocate(Heap, 16).unwrap();
            buffer.as_mut_slice::<i64>().copy_from_slice(&values);
            ColumnView::new(DataType::Int64, buffer, None)
        };
        let columns = [column([1, 2]), column([10, 20])];
        let schema = [("a", DataType::Int64), ("b", DataType::Int64)];
        let table = Table::new(Heap, &schema, &[&columns]).unwrap();
        OneTable(DynScannable::new(Heap, table).unwrap())
    }

    /// Runs `query`, returning each result column's name and values.
    fn run(query: &str) -> StdVec<(StdVec<u8>, StdVec<i64>)> {
        let catalog = catalog();
        let plan = compile(Heap, &REGISTRY, &catalog, query.as_bytes()).unwrap();
        let physical = lower(Heap, &plan).unwrap();
        let mut execution = physical.pipeline().start(Heap).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch));
        let columns = physical.columns().iter();
        columns
            .map(|&c| (physical.name(c).into(), batch.column(c.position).int64s().into()))
            .collect()
    }

    fn error(query: &str) -> (ErrorCode, u32) {
        let catalog = catalog();
        let error = compile(Heap, &REGISTRY, &catalog, query.as_bytes()).err().unwrap();
        (error.code, error.span.start)
    }

    #[test]
    fn runs_from_and_select() {
        let a = (b"a".to_vec(), [1, 2].to_vec());
        let b = (b"b".to_vec(), [10, 20].to_vec());
        assert_eq!(run("FROM t"), [a.clone(), b.clone()]);
        assert_eq!(run("FROM t |> SELECT b, a"), [b.clone(), a.clone()]);
        assert_eq!(run("FROM t |> SELECT b |> SELECT b"), [b]);
    }

    #[test]
    fn reports_what_it_cant_compile() {
        assert_eq!(error("FROM u"), (ErrorCode::UnknownTable, 5));
        assert_eq!(error("FROM t |> SELECT c"), (ErrorCode::UnknownColumn, 17));
        assert_eq!(error("FROM t |> SELECT b |> SELECT a"), (ErrorCode::UnknownColumn, 29));
        assert_eq!(error("FROM t |> SELECT a + 1"), (ErrorCode::Unsupported, 17));
        assert_eq!(error("FROM t |> WHERE a > 1"), (ErrorCode::Unsupported, 16));
    }
}
