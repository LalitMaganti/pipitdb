//! `compile`: PipeSQL text to a `LogicalPlan`, finding tables in a `Catalog`.
//! Each stage's rule compiles it, so a stage an extension adds compiles
//! the same way.

use pipit_kernel::allocator::Allocator;
use pipit_kernel::column::Forms;
use pipit_kernel::plan::{
    DynOp, LogicalPlan, NamedColumn, PLAN_COLUMNS_MAX, PlanNodeId, ScanColumn, ScanOp,
};
use pipit_kernel::scannable::Catalog;
use pipit_kernel::slow_vec::SlowVec;

use crate::ast::{Ast, Node, Tag};
use crate::error::{Error, ErrorCode, Span};
use crate::keywords::same_word;
use crate::parser::parse_query;
use crate::registry::Registry;

/// What a query reads rows from, compiled: its plan node, and its columns, by
/// name.
pub struct Relation {
    pub node: PlanNodeId,
    pub columns: SlowVec<NamedColumn>,
}

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

    /// Whether two of `columns` have one name, ignoring case: a relation's
    /// columns must each have their own, as in Perfetto.
    #[inline(never)]
    fn repeats_a_name(&self, columns: &[NamedColumn]) -> bool {
        let name = |i: usize| self.plan.names.get(at!(columns, i).name);
        (1..columns.len())
            .any(|i| (0..i).any(|j| same_word(name(j).as_bytes(), name(i).as_bytes())))
    }

    /// The column in scope that `name`, a `Name` node, names, ignoring ASCII
    /// case, as SQL does. Two columns may share a name, as after `SELECT a,
    /// a`, but then the name can't be used, as in Perfetto.
    pub fn find_column(&self, name: Node) -> Result<NamedColumn, Error> {
        let text = self.text(name.span());
        let named = |column: &&NamedColumn| {
            same_word(self.plan.names.get(column.name).as_bytes(), text.as_bytes())
        };
        let mut found = self.scope.iter().filter(named);
        let Some(&column) = found.next() else {
            return Err(Error::new(ErrorCode::UnknownColumn, name.span()));
        };
        if found.next().is_some() {
            return Err(Error::new(ErrorCode::AmbiguousColumn, name.span()));
        }
        Ok(column)
    }

    /// The relation `name`, a `Name` node, names: a table in the catalog,
    /// scanned. Every source of rows is compiled through this, so a query
    /// read by another can be one more kind of relation.
    #[inline]
    pub fn compile_relation(&mut self, name: Node) -> Result<Relation, Error> {
        let span = name.span();
        let Some(table) = self.catalog.find(self.text(span)) else {
            return Err(Error::new(ErrorCode::UnknownTable, span));
        };
        let count = table.column_count();
        let mut reads = SlowVec::fixed(self.allocator, count as usize)?;
        let mut columns = SlowVec::new(self.allocator, PLAN_COLUMNS_MAX)?;
        for i in 0..count {
            let binding = self.plan.add_column(table.column_name(i), table.column_type(i))?;
            reads.push(ScanColumn { column: i, binding, forms: Forms::FLAT })?;
            columns.push(binding)?;
        }
        if self.repeats_a_name(&columns) {
            return Err(Error::new(ErrorCode::DuplicateColumn, span));
        }
        let op = DynOp::new(self.allocator, ScanOp::new(table, reads))?;
        let node = self.plan.add_node(op, SlowVec::fixed(self.allocator, 0)?)?;
        Ok(Relation { node, columns })
    }

    /// The value of `node`, an `Integer`.
    pub fn integer(&self, node: Node) -> Result<u64, Error> {
        check!(node.tag() == Tag::Integer);
        // The lexer's integers are digits, so the only error is size.
        let too_large = Error::new(ErrorCode::NumberTooLarge, node.span());
        self.text(node.span()).parse::<u64>().map_err(|_| too_large)
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
    use crate::stages::{AGGREGATION, RELATIONAL};

    static REGISTRY: Registry = Registry::new(&[RELATIONAL, AGGREGATION]);

    /// Table `t`: `a` is 1 and 2, `b` is 10 and 20, `f` is 0.5 and 2.5, `s`
    /// is `it's` and the empty string. And `dup`, of `a` and `A`.
    struct Tables(DynScannable<'static>, DynScannable<'static>, DynScannable<'static>);

    impl Catalog for Tables {
        fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
            match name {
                "t" => Some(&self.0),
                "dup" => Some(&self.1),
                "n" => Some(&self.2),
                _ => None,
            }
        }
    }

    fn catalog() -> Tables {
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
        let same = [column(DataType::Int64, [1, 2]), column(DataType::Int64, [3, 4])];
        let same_names = [("a", DataType::Int64), ("A", DataType::Int64)];
        let same = Table::new(&Heap, &same_names, &[&same]).unwrap();
        // `x` is 1 and null.
        let mut values = Buffer::allocate(&Heap, 16).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(&[1, 0]);
        let mut validity = Buffer::allocate(&Heap, 8).unwrap();
        validity.as_mut_slice::<u8>().copy_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);
        let x = ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, Some(validity));
        let nulls = Table::new(&Heap, &[("x", DataType::Int64)], &[&[x.unwrap()]]).unwrap();
        let scannable = |table| DynScannable::new(&Heap, table).unwrap();
        Tables(scannable(table), scannable(same), scannable(nulls))
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
        assert_eq!(run("FROM t |> SELECT b |> SELECT b"), core::slice::from_ref(&b));
        // Names match ignoring case, and keep the case they were made with.
        assert_eq!(run("FROM t |> SELECT B"), [b]);
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
        assert_eq!(kept("FROM t |> WHERE s LIKE '%t''%' |> SELECT a"), [1.0]);
        assert_eq!(kept("FROM n |> WHERE x IS NULL |> SELECT x").len(), 1);
        assert_eq!(kept("FROM n |> WHERE x is not null |> SELECT x"), [1.0]);
        assert_eq!(kept("FROM n |> WHERE x IS NULL OR x = 1 |> SELECT x").len(), 2);
        assert_eq!(kept("FROM t |> WHERE NOT s like 'i_%' |> SELECT a"), [2.0]);
        assert_eq!(kept("FROM t |> WHERE s <> '' |> SELECT a"), [1.0]);
        // Either way round, and byte by byte: `i` is after `I`.
        assert_eq!(kept("FROM t |> WHERE 'I' < s |> SELECT a"), [1.0]);
        assert_eq!(kept("FROM t |> WHERE s >= '' AND NOT s > 'it' |> SELECT a"), [2.0]);
    }

    #[test]
    fn names_and_drops_columns() {
        let names = |query| -> StdVec<StdVec<u8>> { run(query).into_iter().map(|c| c.0).collect() };
        assert_eq!(kept("FROM t |> SELECT b AS c |> SELECT c"), [10.0, 20.0]);
        assert_eq!(names("FROM t |> EXTEND b AS c, a |> DROP f, s, A"), [b"b", b"c"]);
        // Renames happen at once, and values are set from the scope before.
        assert_eq!(kept("FROM t |> RENAME a AS b, b AS a |> SELECT a"), [10.0, 20.0]);
        assert_eq!(kept("FROM t |> SET a = b, b = a |> SELECT b"), [1.0, 2.0]);
    }

    #[test]
    fn limits_rows() {
        assert_eq!(kept("FROM t |> LIMIT 1 |> SELECT b"), [10.0]);
        assert_eq!(kept("FROM t |> LIMIT 5 OFFSET 1 |> SELECT b"), [20.0]);
        assert_eq!(kept("FROM t |> WHERE a > 1 |> LIMIT 0 |> SELECT b"), []);
    }

    #[test]
    fn aggregates_with_and_without_keys() {
        let count = (b"count(*)".to_vec(), [2.0].to_vec());
        let sum = (b"sum(b)".to_vec(), [30.0].to_vec());
        let avg = (b"avg(f)".to_vec(), [1.5].to_vec());
        assert_eq!(run("FROM t |> AGGREGATE count(*), SUM(b), avg(f)"), [count, sum, avg]);
        // Each `a` is a group, of one row; the keys come first.
        let mut groups = run("FROM t |> WHERE b > 10 |> AGGREGATE max(b) GROUP BY a");
        assert_eq!(groups.remove(0), (b"a".to_vec(), [2.0].to_vec()));
        assert_eq!(groups, [(b"max(b)".to_vec(), [20.0].to_vec())]);
        let aliased = run("FROM t |> AGGREGATE count(*) AS n, sum(b) AS total GROUP BY a AS k");
        let names: StdVec<&[u8]> = aliased.iter().map(|(name, _)| name.as_slice()).collect();
        assert_eq!(names, [b"k".as_slice(), b"n", b"total"]);
        let mut keys = kept("FROM t |> AGGREGATE count(*) GROUP BY a |> SELECT a");
        keys.sort_by(f64::total_cmp);
        assert_eq!(keys, [1.0, 2.0]);
    }

    #[test]
    fn orders_rows() {
        assert_eq!(kept("FROM t |> ORDER BY a DESC |> SELECT b"), [20.0, 10.0]);
        assert_eq!(kept("FROM t |> ORDER BY f |> SELECT s"), [4.0, 0.0]);
        let top = "FROM t |> AGGREGATE count(*) AS n GROUP BY b |> ORDER BY b DESC |> LIMIT 1";
        assert_eq!(run(top)[0].1, [20.0]);
    }

    #[test]
    fn reports_what_it_cant_compile() {
        assert_eq!(error("FROM u"), (ErrorCode::UnknownTable, 5));
        assert_eq!(error("FROM t |> WHERE a = 'x'"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> WHERE s > 1"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> WHERE a LIKE 'x'"), (ErrorCode::Unsupported, 16));
        assert_eq!(error("FROM t |> WHERE a + 1 IS NULL"), (ErrorCode::Unsupported, 16));
        assert_eq!(error("FROM t |> SELECT c"), (ErrorCode::UnknownColumn, 17));
        assert_eq!(run("FROM t |> SELECT a, b, a").len(), 3);
        assert_eq!(error("FROM t |> SELECT a, b, a |> SELECT A"), (ErrorCode::AmbiguousColumn, 35));
        assert_eq!(error("FROM dup"), (ErrorCode::DuplicateColumn, 5));
        assert_eq!(error("FROM t |> DROP a, A"), (ErrorCode::ListedTwice, 18));
        assert_eq!(error("FROM t |> DROP c"), (ErrorCode::UnknownColumn, 15));
        assert_eq!(error("FROM t |> DROP a, b, f, s"), (ErrorCode::NoColumnsLeft, 15));
        assert_eq!(error("FROM t |> RENAME a AS b, A AS c"), (ErrorCode::ListedTwice, 25));
        assert_eq!(error("FROM t |> SET a = a + 1"), (ErrorCode::Unsupported, 18));
        assert_eq!(error("FROM t |> EXTEND a |> SELECT a"), (ErrorCode::AmbiguousColumn, 29));
        assert_eq!(error("FROM t |> LIMIT a"), (ErrorCode::Unsupported, 16));
        assert_eq!(error("FROM t |> LIMIT 1 OFFSET 1, 2"), (ErrorCode::Unsupported, 28));
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
        assert_eq!(error("FROM t |> AGGREGATE median(a)"), (ErrorCode::UnknownFunction, 20));
        assert_eq!(error("FROM t |> SELECT b AS c |> SELECT b"), (ErrorCode::UnknownColumn, 34));
        assert_eq!(error("FROM t |> ORDER BY s"), (ErrorCode::Unsupported, 19));
        assert_eq!(error("FROM t |> ORDER BY a + 1"), (ErrorCode::Unsupported, 19));
        assert_eq!(error("FROM t |> AGGREGATE sum(s)"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> AGGREGATE sum(a + 1)"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> AGGREGATE count(*) GROUP BY f"), (ErrorCode::Unsupported, 38));
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
