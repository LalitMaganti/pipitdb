//! `compile`: PipeSQL text to a `LogicalPlan`, finding tables in a `Catalog`.
//! Each stage's rule compiles it, so a stage an extension adds compiles
//! the same way.

use pipit_kernel::allocator::{AllocError, Allocator};
use pipit_kernel::column::Forms;
use pipit_kernel::names::Names;
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
    registry: &'q Registry,
    allocator: &'q dyn Allocator,
    pub plan: LogicalPlan<'c>,
    /// The columns the stages so far make, by name.
    scope: SlowVec<NamedColumn>,
    /// The scope's positions, by the hash of their names.
    index: NameIndex,
}

/// Positions of columns, in slots by a hash of their names ignoring ASCII
/// case, so a name is found without comparing it with every other.
struct NameIndex {
    /// Each slot's position plus one, or 0 if it's empty: a power of two of
    /// them, more than twice as many as positions, so probes are short.
    slots: SlowVec<u32>,
    /// Whether two of the columns have one name.
    repeats: bool,
}

impl NameIndex {
    /// An index of `columns`, whose names are in `names`.
    fn new(
        allocator: &dyn Allocator,
        names: &Names,
        columns: &[NamedColumn],
    ) -> Result<NameIndex, AllocError> {
        let slots = SlowVec::zeroed(allocator, (2 * columns.len() + 1).next_power_of_two())?;
        let mut index = NameIndex { slots, repeats: false };
        for (position, column) in (1..).zip(columns) {
            let name = names.get(column.name);
            let same = |other: u32| {
                same_word(names.get(at!(columns, other as usize).name).as_bytes(), name.as_bytes())
            };
            let (count, _, empty) = index.find(name, same);
            index.repeats |= count > 0;
            *at_mut!(index.slots, empty) = position;
        }
        Ok(index)
    }

    /// Of the positions `name` probes, how many `named` says it names, and
    /// the first; and the empty slot the probe ends at, where it would go.
    fn find(&self, name: &str, named: impl Fn(u32) -> bool) -> (u32, u32, usize) {
        let mask = self.slots.len() - 1;
        #[expect(clippy::cast_possible_truncation, reason = "the low bits pick the slot")]
        let (mut slot, mut count, mut first) = (hash(name) as usize & mask, 0, 0);
        loop {
            let position = *at!(self.slots, slot);
            if position == 0 {
                return (count, first, slot);
            }
            if named(position - 1) {
                first = if count == 0 { position - 1 } else { first };
                count += 1;
            }
            slot = (slot + 1) & mask;
        }
    }
}

/// FNV-1a of `name`, ignoring ASCII case.
fn hash(name: &str) -> u64 {
    let fold = |hash: u64, &byte: &u8| {
        (hash ^ u64::from(byte.to_ascii_lowercase())).wrapping_mul(0x100_0000_01b3)
    };
    name.as_bytes().iter().fold(0xcbf2_9ce4_8422_2325, fold)
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

    /// The bytes of `node`, if it's a string literal, which it adds to the
    /// statement's literals: as a function's binding reads a pattern.
    pub fn string_literal(&mut self, node: Node) -> Result<Option<&[u8]>, Error> {
        if node.tag() != Tag::String {
            return Ok(None);
        }
        let id = crate::condition::string(self, node)?;
        Ok(Some(self.plan.literal(id)))
    }

    /// The stages and functions queries can use.
    pub fn registry(&self) -> &'q Registry {
        self.registry
    }

    /// What to make plan nodes with.
    pub fn allocator(&self) -> &'q dyn Allocator {
        self.allocator
    }

    /// The columns the stages so far make, by name.
    pub fn scope(&self) -> &[NamedColumn] {
        &self.scope
    }

    /// Makes `columns` the scope.
    pub fn set_scope(&mut self, columns: SlowVec<NamedColumn>) -> Result<(), Error> {
        self.index = NameIndex::new(self.allocator, &self.plan.names, &columns)?;
        self.scope = columns;
        Ok(())
    }

    /// The column in scope that `name`, a `Name` node, names, ignoring ASCII
    /// case, as SQL does. Two columns may share a name, as after `SELECT a,
    /// a`, but then the name can't be used, as in Perfetto.
    pub fn find_column(&self, name: Node) -> Result<NamedColumn, Error> {
        Ok(self.find_in_scope(name)?.1)
    }

    /// As `find_column`, with the column's position in the scope.
    pub fn find_in_scope(&self, name: Node) -> Result<(usize, NamedColumn), Error> {
        let text = self.text(name.span());
        let named = |position: u32| {
            let column = at!(self.scope, position as usize);
            same_word(self.plan.names.get(column.name).as_bytes(), text.as_bytes())
        };
        let (count, position, _) = self.index.find(text, named);
        if count == 0 {
            return Err(Error::new(ErrorCode::UnknownColumn, name.span()));
        }
        if count > 1 {
            return Err(Error::new(ErrorCode::AmbiguousColumn, name.span()));
        }
        Ok((position as usize, *at!(self.scope, position as usize)))
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
            let (name, data_type) = (table.column_name(i), table.column_type(i));
            let binding = self.plan.add_column(name, data_type, table.column_nullable(i))?;
            reads.push(ScanColumn { column: i, binding, forms: Forms::FLAT })?;
            columns.push(binding)?;
        }
        // A relation's columns must each have their own name, as in Perfetto.
        if NameIndex::new(self.allocator, &self.plan.names, &columns)?.repeats {
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
        registry,
        plan: LogicalPlan::new(allocator)?,
        scope: SlowVec::new(allocator, PLAN_COLUMNS_MAX)?,
        index: NameIndex::new(allocator, &Names::fixed(allocator, 0)?, &[])?,
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
    use pipit_kernel::output_properties::plan_nodes;
    use pipit_kernel::query_allocators::QueryAllocators;
    use pipit_kernel::row_batch::RowBatch;
    use pipit_kernel::scannable::DynScannable;
    use pipit_kernel::selection::Kept;
    use pipit_operators::table::Table;

    use super::*;
    use pipit_kernel::expression::validity_of;
    use pipit_kernel::function::{DynFunction, Function};

    use crate::error::ErrorCode;
    use crate::registry::FunctionRule;
    use crate::stages::{AGGREGATION, RELATIONAL};

    #[cfg(not(feature = "string-functions"))]
    static REGISTRY: Registry = Registry::new(&[RELATIONAL, AGGREGATION]).with_functions(&[HALF]);
    #[cfg(feature = "string-functions")]
    static REGISTRY: Registry = Registry::new(&[RELATIONAL, AGGREGATION])
        .with_functions(&[HALF, crate::string_functions::STRING_FUNCTIONS]);

    /// `half(x)`, as a crate would add a function.
    const HALF: &[FunctionRule] = &[FunctionRule {
        name: "half",
        arguments: &[DataType::Float64],
        result: DataType::Float64,
        bind: |compiler, _| Ok(DynFunction::new(compiler.allocator(), Half)?),
    }];

    struct Half;

    impl Function for Half {
        fn run(
            &self,
            context: &mut Context,
            arguments: &[&ColumnView],
        ) -> Result<ColumnView, pipit_kernel::error::Error> {
            let validity = validity_of(context, arguments)?;
            let x = arguments[0].float64s();
            let mut values = context.values_buffer(x.len() * 8)?;
            for (o, &x) in values.as_mut_slice::<f64>().iter_mut().zip(x) {
                *o = x / 2.0;
            }
            Ok(ColumnView::new(context, DataType::Float64, values, validity)?)
        }
    }

    /// Table `t`: `a` is 1 and 2, `b` is 10 and 20, `f` is 0.5 and 2.5, `s`
    /// is `it's` and the empty string. And `dup`, of `a` and `A`.
    struct Tables(
        DynScannable<'static>,
        DynScannable<'static>,
        DynScannable<'static>,
        DynScannable<'static>,
    );

    impl Catalog for Tables {
        fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
            match name {
                "t" => Some(&self.0),
                "dup" => Some(&self.1),
                "n" => Some(&self.2),
                "ids" => Some(&self.3),
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
        // `id` is 10, 11 and 12, stored as a sequence, and `v` 1, 2 and 3.
        let id = ColumnView::sequence(&mut Context::new(&Heap), 10, 3).unwrap();
        let mut values = Buffer::allocate(&Heap, 24).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(&[1, 2, 3]);
        let v = ColumnView::new(&mut Context::new(&Heap), DataType::Int64, values, None).unwrap();
        let schema = [("id", DataType::Int64), ("v", DataType::Int64)];
        let ids = Table::new(&Heap, &schema, &[&[id, v]]).unwrap();
        Tables(scannable(table), scannable(same), scannable(nulls), scannable(ids))
    }

    /// Runs `query`, returning each result column's name and the values of
    /// the rows kept, as floats.
    fn run(query: &str) -> StdVec<(StdVec<u8>, StdVec<f64>)> {
        let catalog = catalog();
        let mut plan = compile(&Heap, &REGISTRY, &catalog, query.as_bytes()).unwrap();
        let planned = plan_nodes(&Heap, &plan).unwrap();
        let physical = lower(&Heap, &mut plan, &planned).unwrap();
        let query = QueryAllocators::new(&Heap);
        let mut execution = physical.start(&query).unwrap();
        let mut batch = RowBatch::new();
        let mut columns: StdVec<_> =
            physical.columns().iter().map(|&c| (physical.name(c).into(), StdVec::new())).collect();
        while execution.next(&mut batch).unwrap() {
            let rows: StdVec<usize> = match batch.selection().kept() {
                Kept::Prefix => (0..batch.selection().len() as usize).collect(),
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
        assert_eq!(kept("FROM t |> WHERE a IN (-1, 2) |> SELECT b"), [20.0]);
        // Long enough to be kept in a set.
        let long = (2..100).map(|i| format!("{i}")).collect::<StdVec<_>>().join(", ");
        assert_eq!(kept(&format!("FROM t |> WHERE a IN ({long}) |> SELECT b")), [20.0]);
        // As in SQL, a null is in no list, and not out of one either.
        assert_eq!(kept("FROM n |> WHERE NOT x IN (2, 3) |> SELECT x"), [1.0]);
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
        assert_eq!(kept("FROM t |> WHERE s IN ('it''s', 'x') |> SELECT a"), [1.0]);
        let long = (0..100).map(|i| format!("'{i}'")).collect::<StdVec<_>>().join(", ");
        assert_eq!(kept(&format!("FROM t |> WHERE NOT s IN ({long}, '') |> SELECT a")), [1.0]);
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
    fn knows_which_columns_can_be_null() {
        let catalog = catalog();
        let query = b"FROM t |> EXTEND a / 60 % 60 AS x, a / b AS y |> AGGREGATE count(*) AS c, sum(b) AS total GROUP BY x, y";
        let plan = compile(&Heap, &REGISTRY, &catalog, query).unwrap();
        let nullable = |name: &str| {
            let column = plan.columns.iter().find(|column| plan.names.get(column.name) == name);
            column.unwrap().nullable
        };
        // The table has no nulls; dividing by a column may make one, and a
        // sum of none is one, but dividing by a constant that isn't zero can't.
        assert!(!nullable("a") && !nullable("x") && !nullable("c"));
        assert!(nullable("y") && nullable("total"));
    }

    #[test]
    fn computes_columns() {
        assert_eq!(kept("FROM t |> SELECT a + b AS c"), [11.0, 22.0]);
        assert_eq!(kept("FROM t |> EXTEND b - a * 2 AS c |> SELECT c"), [8.0, 16.0]);
        // Every item sees the scope before the stage.
        assert_eq!(kept("FROM t |> SET a = a * 10, b = a |> SELECT b"), [1.0, 2.0]);
        assert_eq!(kept("FROM t |> SET a = a * 10 |> SELECT a"), [10.0, 20.0]);
        // Integers widen to meet floats.
        assert_eq!(kept("FROM t |> SELECT f / a AS x"), [0.5, 1.25]);
        assert_eq!(kept("FROM t |> SELECT -a AS x, 7 AS y"), [-1.0, -2.0]);
        // `%` binds as `*` does, and takes the sign of its left side, as in SQLite.
        assert_eq!(kept("FROM t |> SELECT a + b % 3 AS x"), [2.0, 4.0]);
        assert_eq!(kept("FROM t |> SELECT -b % 3 AS x"), [-1.0, -2.0]);
        let smallest = kept("FROM t |> SELECT -9223372036854775808 AS x");
        assert_eq!(smallest, [-9_223_372_036_854_775_808.0; 2]);
        // Dividing by zero, or by NULL, is NULL.
        assert_eq!(kept("FROM t |> EXTEND b / (a - 1) AS x |> WHERE x IS NULL |> SELECT a"), [1.0]);
        assert_eq!(kept("FROM t |> EXTEND b % (a - 1) AS x |> WHERE x IS NULL |> SELECT a"), [1.0]);
        assert_eq!(kept("FROM t |> EXTEND a + NULL AS x |> WHERE x IS NULL |> SELECT a").len(), 2);
        // Comparisons, as 0 or 1, with a literal on either side, an integer
        // literal compared with a double as one, and NULL where either is.
        assert_eq!(kept("FROM t |> SELECT a > 1 AS x"), [0.0, 1.0]);
        assert_eq!(kept("FROM t |> SELECT 1 >= a AS x"), [1.0, 0.0]);
        assert_eq!(kept("FROM t |> SELECT b != a * 10 AS x"), [0.0, 0.0]);
        assert_eq!(kept("FROM t |> SELECT f > 2 AS x"), [0.0, 1.0]);
        assert_eq!(kept("FROM t |> SELECT 1 < f AS x"), [0.0, 1.0]);
        assert_eq!(kept("FROM t |> SELECT s < 'j' AS x"), [1.0, 1.0]);
        assert_eq!(kept("FROM t |> SELECT s = '' AS x"), [0.0, 1.0]);
        assert_eq!(kept("FROM n |> EXTEND x = 1 AS y |> WHERE y IS NULL |> SELECT x").len(), 1);
        assert_eq!(kept("FROM t |> EXTEND a > 1 AS big |> WHERE big = 1 |> SELECT a"), [2.0]);
        // Null tests, never null, and `LIKE`, null where its string is.
        assert_eq!(kept("FROM n |> SELECT x IS NULL AS y"), [0.0, 1.0]);
        assert_eq!(kept("FROM n |> SELECT x IS NOT NULL AS y"), [1.0, 0.0]);
        assert_eq!(kept("FROM t |> SELECT s LIKE 'it%' AS x"), [1.0, 0.0]);
        assert_eq!(kept("FROM t |> SELECT s LIKE '%' AS x"), [1.0, 1.0]);
        assert_eq!(kept("FROM t |> SELECT s LIKE '%t''s%' AS x"), [1.0, 0.0]);
        assert_eq!(kept("FROM t |> EXTEND s LIKE '%s' AS m |> WHERE m = 1 |> SELECT a"), [1.0]);
        let null = "FROM t |> EXTEND NULL LIKE 'x' AS y |> WHERE y IS NULL |> SELECT a";
        assert_eq!(kept(null).len(), 2);
        // `iif` and `coalesce`, of integers and strings, with conditions which
        // are doubles or null.
        assert_eq!(kept("FROM t |> SELECT iif(a > 1, b, a) AS x"), [1.0, 20.0]);
        assert_eq!(kept("FROM t |> SELECT iif(a = 1, s, '') AS x"), [4.0, 0.0]);
        assert_eq!(kept("FROM t |> SELECT iif(a = 2, s, 'xyz') AS x"), [3.0, 0.0]);
        assert_eq!(kept("FROM t |> SELECT iif(f, 1, 0) AS x"), [1.0, 1.0]);
        assert_eq!(kept("FROM n |> SELECT iif(x = 1, 1, 0) AS y"), [1.0, 0.0]);
        assert_eq!(kept("FROM n |> SELECT coalesce(x, 7) AS y"), [1.0, 7.0]);
        assert_eq!(kept("FROM n |> SELECT coalesce(x, NULL, 3) AS y"), [1.0, 3.0]);
        assert_eq!(kept("FROM n |> SELECT ifnull(x, 0) AS y"), [1.0, 0.0]);
        assert_eq!(kept("FROM t |> SELECT coalesce(a * 2, 0) AS x"), [2.0, 4.0]);
        let groups = "FROM t |> EXTEND iif(a > 1, 'big', 'small') AS k |> AGGREGATE count(*) AS c GROUP BY k |> SELECT c";
        assert_eq!(kept(groups), [1.0, 1.0]);
        // `CASE`, as `iif`s.
        assert_eq!(kept("FROM t |> SELECT CASE WHEN a > 1 THEN b ELSE a END AS x"), [1.0, 20.0]);
        let no_else =
            "FROM t |> EXTEND CASE WHEN a > 1 THEN b END AS x |> WHERE x IS NULL |> SELECT a";
        assert_eq!(kept(no_else), [1.0]);
        let whens =
            "FROM t |> SELECT CASE WHEN a = 1 THEN 'one' WHEN a = 2 THEN 'three' ELSE '' END AS x";
        assert_eq!(kept(whens), [3.0, 5.0]);
        let subject = "FROM t |> SELECT CASE a WHEN 1 THEN 10 WHEN 2 THEN 20 END AS x";
        assert_eq!(kept(subject), [10.0, 20.0]);
        let nested = "FROM t |> SELECT CASE WHEN a = 1 THEN CASE WHEN b = 10 THEN s ELSE 'x' END ELSE 'yy' END AS x";
        assert_eq!(kept(nested), [4.0, 2.0]);
        assert_eq!(kept("FROM t |> SELECT case when a > 1 then 1 else 0 end AS x"), [0.0, 1.0]);
        // String literals, as lengths here, which later stages read.
        assert_eq!(kept("FROM t |> SELECT 'it''s' AS x"), [4.0, 4.0]);
        assert_eq!(kept("FROM t |> SELECT '' AS x"), [0.0, 0.0]);
        let grouped = "FROM t |> EXTEND 'ab' AS k |> WHERE k = 'ab' |> AGGREGATE count(*) AS c GROUP BY k |> SELECT c";
        assert_eq!(kept(grouped), [2.0]);
        // An integer overflow fails the query.
        let catalog = catalog();
        let query = b"FROM t |> SELECT -(-9223372036854775808) AS x";
        let mut plan = compile(&Heap, &REGISTRY, &catalog, query).unwrap();
        let planned = plan_nodes(&Heap, &plan).unwrap();
        let physical = lower(&Heap, &mut plan, &planned).unwrap();
        let query = QueryAllocators::new(&Heap);
        let mut execution = physical.start(&query).unwrap();
        let overflow = execution.next(&mut RowBatch::new()).err();
        assert_eq!(overflow, Some(pipit_kernel::error::Error::Overflow));
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
        let least = (b"min(a)".to_vec(), [1.0].to_vec());
        assert_eq!(run("FROM t |> AGGREGATE count(*), SUM(b), min(a)"), [count, sum, least]);
        assert_eq!(run("FROM t |> AGGREGATE avg(b)"), [(b"avg(b)".to_vec(), [15.0].to_vec())]);
        let distinct = (b"count(DISTINCT a)".to_vec(), [2.0].to_vec());
        assert_eq!(run("FROM t |> AGGREGATE count(DISTINCT a)"), [distinct]);
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
        // A string key, shown as its length.
        let strings = run("FROM t |> AGGREGATE count(*) GROUP BY s");
        let s = (b"s".to_vec(), [4.0, 0.0].to_vec());
        assert_eq!(strings, [s, (b"count(*)".to_vec(), [1.0, 1.0].to_vec())]);
        // The empty string is the least.
        let extremes = run("FROM t |> AGGREGATE min(s), max(DISTINCT s)");
        let least = (b"min(s)".to_vec(), [0.0].to_vec());
        assert_eq!(extremes, [least, (b"max(DISTINCT s)".to_vec(), [4.0].to_vec())]);
    }

    #[test]
    fn calls_functions_from_the_registry() {
        assert_eq!(kept("FROM t |> SELECT half(f) AS x"), [0.25, 1.25]);
        assert_eq!(kept("FROM t |> SELECT half(a) AS x"), [0.5, 1.0]);
        let null = "FROM t |> EXTEND half(NULL) AS x |> WHERE x IS NULL |> SELECT a";
        assert_eq!(kept(null), [1.0, 2.0]);
    }

    #[cfg(feature = "string-functions")]
    #[test]
    fn measures_strings() {
        assert_eq!(kept("FROM t |> SELECT length(s) AS n"), [4.0, 0.0]);
        assert_eq!(kept("FROM t |> SELECT LENGTH('h\u{e9}llo') AS n"), [5.0, 5.0]);
        assert_eq!(kept("FROM t |> SELECT octet_length('h\u{e9}llo') AS n"), [6.0, 6.0]);
    }

    #[test]
    fn reads_id_columns_stored_as_sequences() {
        // A filter takes the sequence; everything else, flat values.
        assert_eq!(kept("FROM ids |> WHERE id > 10 |> SELECT v"), [2.0, 3.0]);
        assert_eq!(kept("FROM ids |> WHERE id = 11 OR id IS NULL |> SELECT id"), [11.0]);
        assert_eq!(kept("FROM ids |> SELECT id + 1 AS x"), [11.0, 12.0, 13.0]);
        assert_eq!(kept("FROM ids |> AGGREGATE sum(id)"), [33.0]);
    }

    #[test]
    fn orders_rows() {
        assert_eq!(kept("FROM t |> ORDER BY a DESC |> SELECT b"), [20.0, 10.0]);
        assert_eq!(kept("FROM t |> ORDER BY f |> SELECT s"), [4.0, 0.0]);
        assert_eq!(kept("FROM t |> ORDER BY s |> SELECT a"), [2.0, 1.0]);
        let top = "FROM t |> AGGREGATE count(*) AS n GROUP BY b |> ORDER BY b DESC |> LIMIT 1";
        assert_eq!(run(top)[0].1, [20.0]);
    }

    #[test]
    fn reports_what_expressions_cant_compute() {
        assert_eq!(
            error("FROM t |> SELECT CASE WHEN a THEN 1 AS x"),
            (ErrorCode::ExpectedWord, 36)
        );
        assert_eq!(error("FROM t |> SELECT CASE END AS x"), (ErrorCode::ExpectedExpression, 22));
        assert_eq!(error("FROM t |> SELECT CASE a THEN 1 END AS x"), (ErrorCode::ExpectedWord, 24));
        assert_eq!(error("FROM t |> SELECT s = 1 AS x"), (ErrorCode::ExpectedOneKind, 17));
        assert_eq!(error("FROM t |> SELECT a LIKE 'x' AS y"), (ErrorCode::ExpectedString, 17));
        assert_eq!(error("FROM t |> SELECT iif(a, 1) AS y"), (ErrorCode::ArgumentCount, 17));
        assert_eq!(error("FROM t |> SELECT coalesce(a) AS y"), (ErrorCode::ArgumentCount, 17));
        assert_eq!(error("FROM t |> SELECT iif(s, 1, 0) AS y"), (ErrorCode::NumberCondition, 17));
        let overflows = "FROM t |> SELECT iif(a > 1, b * 2, 0) AS y";
        assert_eq!(error(overflows), (ErrorCode::ArithmeticOnSomeRows, 17));
        assert_eq!(error("FROM t |> SELECT upper(s) AS y"), (ErrorCode::UnknownFunction, 17));
        assert_eq!(error("FROM t |> SELECT half(s) AS y"), (ErrorCode::ArgumentType, 17));
        assert_eq!(error("FROM t |> SELECT half(a, b) AS y"), (ErrorCode::ArgumentCount, 17));
        assert_eq!(error("FROM t |> SELECT s LIKE s AS y"), (ErrorCode::Unsupported, 24));
        assert_eq!(error("FROM t |> SELECT f > a AS x"), (ErrorCode::ExpectedOneType, 17));
        let inexact = "FROM t |> SELECT f = 9007199254740993 AS x";
        assert_eq!(error(inexact), (ErrorCode::Unsupported, 17));
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
        assert_eq!(error("FROM t |> SET a = s + 1"), (ErrorCode::ExpectedNumber, 18));
        assert_eq!(error("FROM t |> EXTEND a |> SELECT a"), (ErrorCode::AmbiguousColumn, 29));
        assert_eq!(error("FROM t |> LIMIT a"), (ErrorCode::Unsupported, 16));
        assert_eq!(error("FROM t |> LIMIT 1 OFFSET 1, 2"), (ErrorCode::Unsupported, 28));
        assert_eq!(error("FROM t |> SELECT b |> SELECT a"), (ErrorCode::UnknownColumn, 29));
        assert_eq!(error("FROM t |> SELECT a + 1"), (ErrorCode::ExpectedName, 17));
        assert_eq!(error("FROM t |> EXTEND NULL - s AS x"), (ErrorCode::ExpectedNumber, 17));
        assert_eq!(error("FROM t |> EXTEND f % 2 AS x"), (ErrorCode::ExpectedInteger, 17));
        assert_eq!(error("FROM t |> SELECT count(a) AS x"), (ErrorCode::Unsupported, 17));
        assert_eq!(
            error("FROM t |> SELECT coalesce(a, s) AS y"),
            (ErrorCode::ChoicesOfOneType, 17)
        );
        assert_eq!(error("FROM t |> SELECT 'x' + 1 AS x"), (ErrorCode::ExpectedNumber, 17));
        assert_eq!(error("FROM t |> SELECT -'x' AS x"), (ErrorCode::ExpectedNumber, 18));
        assert_eq!(error("FROM t |> SELECT 1.5 AS x"), (ErrorCode::Unsupported, 17));
        assert_eq!(
            error("FROM t |> SELECT 9223372036854775808 AS x"),
            (ErrorCode::NumberTooLarge, 17)
        );
        let long = format!("FROM t |> SELECT a{} AS x", " + a".repeat(200));
        assert_eq!(error(&long), (ErrorCode::ExpressionTooLarge, 17));
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
        assert_eq!(error("FROM t |> WHERE f IN (1)"), (ErrorCode::Unsupported, 22));
        assert_eq!(error("FROM t |> WHERE s IN ('a', 1)"), (ErrorCode::Unsupported, 27));
        assert_eq!(error("FROM t |> WHERE a IN ('a')"), (ErrorCode::Unsupported, 22));
        assert_eq!(error("FROM t |> AGGREGATE median(a)"), (ErrorCode::UnknownFunction, 20));
        assert_eq!(error("FROM t |> ORDER BY a + 1"), (ErrorCode::Unsupported, 19));
        assert_eq!(error("FROM t |> AGGREGATE sum(s)"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> AGGREGATE sum(f)"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> AGGREGATE min(f)"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> AGGREGATE sum(a + 1)"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> AGGREGATE count(*) GROUP BY f"), (ErrorCode::Unsupported, 38));
        assert_eq!(error("FROM t |> AGGREGATE count(DISTINCT *)"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> WHERE DISTINCT a"), (ErrorCode::ExpectedExpression, 16));
    }

    /// How many columns `query`'s batches have once optimized, and its
    /// result column's values.
    fn optimized(query: &str) -> (u32, StdVec<i64>) {
        let catalog = catalog();
        let mut plan = compile(&Heap, &REGISTRY, &catalog, query.as_bytes()).unwrap();
        pipit_kernel::optimize::optimize(&Heap, &mut plan).unwrap();
        let planned = plan_nodes(&Heap, &plan).unwrap();
        let physical = lower(&Heap, &mut plan, &planned).unwrap();
        let query = QueryAllocators::new(&Heap);
        let mut execution = physical.start(&query).unwrap();
        let mut batch = RowBatch::new();
        assert!(execution.next(&mut batch).unwrap());
        let values = batch.column(physical.columns()[0].position).int64s().to_vec();
        (batch.column_count(), values)
    }

    #[test]
    fn reads_only_the_columns_selected() {
        assert_eq!(optimized("FROM t |> SELECT b"), (1, [10, 20].to_vec()));
        // Only `b` is read, and only `d` computed.
        let computed = "FROM t |> EXTEND a + 1 AS c, b * 2 AS d |> SELECT d";
        assert_eq!(optimized(computed), (2, [20, 40].to_vec()));
    }
}
