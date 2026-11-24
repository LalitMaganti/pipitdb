//! `compile`: PipeSQL text to a `LogicalPlan`, finding tables in a `Catalog`.
//! Each stage's rule compiles it, so a stage an extension adds compiles
//! the same way.

#![allow(unsafe_code)]

use pipit_kernel::column::Forms;
use pipit_kernel::containers::{SmallVec, SpillVec};
use pipit_kernel::names::Name;
use pipit_kernel::plan::{
    DynOp, LogicalPlan, NamedColumn, PLAN_COLUMNS_MAX, PLAN_NODES_MAX, PlanNodeId, ScanColumn,
    ScanOp,
};
use pipit_kernel::scannable::Catalog;
use pipit_kernel::session::Query;

use crate::ast::{Ast, Node, Tag};
use crate::error::{Error, ErrorCode, Span};
use crate::keywords::same_word;
use crate::parser::parse_query;
use crate::registry::Registry;
use pipit_kernel::env::Env;

/// What a query reads rows from, compiled: its plan node, and its columns, by
/// name.
pub struct Relation {
    pub node: PlanNodeId,
    pub columns: SmallVec<NamedColumn>,
}

/// A table alias, as `GoogleSQL`'s: a name for the columns a table had when
/// it was named, so `t.x` is that `x` whatever stages since have done to
/// the scope. Its columns are `alias_columns[start..end]`.
#[derive(Clone, Copy)]
struct TableAlias {
    name: Name,
    start: u32,
    end: u32,
}

/// The most aliases in scope at once, one for each table joined: as many
/// as a plan has nodes.
const ALIASES_MAX: usize = PLAN_NODES_MAX;

/// How many aliases are held in place, as most queries have no more.
const ALIASES_INLINE: usize = 8;

/// The most columns of all aliases at once.
const ALIAS_COLUMNS_MAX: usize = ALIASES_MAX * PLAN_COLUMNS_MAX;

/// A plan being compiled from a query, which each stage's rule adds to.
pub struct Compiler<'q, 'c> {
    /// The query's text, which the tree's spans are into.
    source: &'q [u8],
    /// The query, parsed.
    ast: &'q Ast,
    /// The tables `FROM` can name.
    catalog: &'c dyn Catalog,
    /// The stages' rules and the functions calls can name.
    registry: &'q Registry,
    /// What the plan's memory comes from.
    env: &'q Env,
    /// The plan so far.
    pub plan: LogicalPlan<'c>,
    /// The columns the stages so far make, by name.
    pub scope: SmallVec<NamedColumn>,
    /// The aliases `t.x` can name columns by.
    aliases: SpillVec<'q, TableAlias, ALIASES_INLINE>,
    /// The columns of `aliases`, each alias's after the one before's.
    alias_columns: SmallVec<NamedColumn>,
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
        let id = crate::expression::string(self, node)?;
        Ok(Some(self.plan.literal(id)))
    }

    /// The stages and functions queries can use.
    pub fn registry(&self) -> &'q Registry {
        self.registry
    }

    /// What to make plan nodes with.
    pub fn env(&self) -> &'q Env {
        self.env
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
    #[inline(never)]
    pub fn find_column(&self, name: Node) -> Result<NamedColumn, Error> {
        let text = self.text(name.span()).as_bytes();
        let Some(dot) = text.iter().position(|&byte| byte == b'.') else {
            return self.find_named(&self.scope, text, name.span());
        };
        let (wanted, column) = (at!(text, ..dot).trim_ascii(), at!(text, dot + 1..).trim_ascii());
        let mut found = self
            .aliases
            .iter()
            .filter(|a| same_word(self.plan.names.get(a.name).as_bytes(), wanted));
        let Some(alias) = found.next() else {
            return Err(Error::new(ErrorCode::UnknownColumn, name.span()));
        };
        if found.next().is_some() {
            return Err(Error::new(ErrorCode::AmbiguousColumn, name.span()));
        }
        let columns = at!(self.alias_columns, alias.start as usize..alias.end as usize);
        self.find_named(columns, column, name.span())
    }

    /// The one column of `columns` that `name` names, ignoring case.
    pub fn find_in(&self, columns: &[NamedColumn], name: Node) -> Result<NamedColumn, Error> {
        self.find_named(columns, self.text(name.span()).as_bytes(), name.span())
    }

    /// The one column of `columns` named `text`, ignoring case, for `span`.
    fn find_named(
        &self,
        columns: &[NamedColumn],
        text: &[u8],
        span: Span,
    ) -> Result<NamedColumn, Error> {
        let named =
            |column: &&NamedColumn| same_word(self.plan.names.get(column.name).as_bytes(), text);
        let mut found = columns.iter().filter(named);
        let Some(&column) = found.next() else {
            return Err(Error::new(ErrorCode::UnknownColumn, span));
        };
        if found.next().is_some() {
            return Err(Error::new(ErrorCode::AmbiguousColumn, span));
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
        let mut reads = SmallVec::fixed(self.env, count as usize)?;
        let mut columns = SmallVec::new(self.env, PLAN_COLUMNS_MAX)?;
        for i in 0..count {
            let (name, data_type) = (table.column_name(i), table.column_type(i));
            let binding = self.plan.add_column(name, data_type, table.column_nullable(i))?;
            reads.push(ScanColumn { column: i, binding, forms: Forms::FLAT })?;
            columns.push(binding)?;
        }
        if self.repeats_a_name(&columns) {
            return Err(Error::new(ErrorCode::DuplicateColumn, span));
        }
        let op = DynOp::new(self.env, ScanOp::new(table, reads))?;
        let node = self.plan.add_node(op, &[])?;
        Ok(Relation { node, columns })
    }

    /// Adds the alias `name`, a name, for `columns`.
    pub fn add_alias(&mut self, name: Node, columns: &[NamedColumn]) -> Result<(), Error> {
        let name = self.plan.names.add(self.text(name.span()))?;
        #[expect(clippy::cast_possible_truncation, reason = "at most `ALIAS_COLUMNS_MAX`")]
        let start = self.alias_columns.len() as u32;
        self.alias_columns.extend_from_slice(columns)?;
        #[expect(clippy::cast_possible_truncation, reason = "at most `ALIAS_COLUMNS_MAX`")]
        let end = self.alias_columns.len() as u32;
        Ok(self.aliases.push(TableAlias { name, start, end })?)
    }

    /// Drops every alias, as a stage that makes a new scope does.
    pub fn clear_aliases(&mut self) {
        self.aliases.truncate(0);
        self.alias_columns.retain(|_| false);
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

/// Parses `source` with the stages in `registry`, and compiles it into a
/// query with memory from `env`, finding tables in `catalog`. The query's
/// result is the last stage's columns.
pub fn compile<'e>(
    env: &'e Env,
    registry: &Registry,
    catalog: &'e dyn Catalog,
    source: &[u8],
) -> Result<Query<'e>, Error> {
    Query::new(env, |env| {
        let ast = parse_query(env, registry, source)?;
        let mut compiler = Compiler {
            source,
            ast: &ast,
            catalog,
            registry,
            plan: LogicalPlan::new(env)?,
            scope: SmallVec::new(env, PLAN_COLUMNS_MAX)?,
            aliases: SpillVec::new(env, ALIASES_MAX),
            alias_columns: SmallVec::new(env, ALIAS_COLUMNS_MAX)?,
            env,
        };
        let query = ast.node(ast.root());
        for i in 0..query.child_count() {
            let stage = ast.node(query.first_child() + i);
            (registry.rule(stage.rule()).compile)(&mut compiler, stage)?;
        }
        let Compiler { mut plan, scope, .. } = compiler;
        plan.output = scope;
        Ok(plan)
    })
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::format;
    use std::vec::Vec as StdVec;

    use pipit_kernel::buffer::Buffer;
    use pipit_kernel::column::{ColumnView, DataType};
    use pipit_kernel::context::Context;
    use pipit_kernel::env::Env;
    use pipit_kernel::scannable::DynScannable;
    use pipit_kernel::selection::Kept;
    use pipit_kernel::session::{Prepared, Session};
    use pipit_operators::table::Table;

    use super::*;
    use pipit_kernel::expression::validity_of;
    use pipit_kernel::function::{DynFunction, Function};

    use crate::choices::CHOICES;
    use crate::error::ErrorCode;
    use crate::expression::{Bind, call_bound};
    use crate::registry::FunctionRule;
    use crate::stages::{AGGREGATION, JOINS, RELATIONAL};

    #[cfg(not(feature = "string-functions"))]
    static REGISTRY: Registry =
        Registry::new(&[RELATIONAL, AGGREGATION, JOINS]).with_functions(&[CHOICES, HALF, TOTAL]);
    #[cfg(feature = "string-functions")]
    static REGISTRY: Registry = Registry::new(&[RELATIONAL, AGGREGATION, JOINS]).with_functions(&[
        CHOICES,
        HALF,
        TOTAL,
        crate::string_functions::STRING_FUNCTIONS,
    ]);

    /// `half(x)`, as a crate would add a function.
    const HALF: &[FunctionRule] = &[FunctionRule {
        name: "half",
        compile: |compiler, node, depth, nodes| {
            let bind: Bind = |c, _| Ok(DynFunction::new(c.env(), Half)?);
            let signature = (&[DataType::Float64][..], DataType::Float64);
            call_bound(compiler, (node, depth), nodes, signature, bind)
        },
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

    /// `total(x1, ..., x20)`: more arguments than a call holds in place.
    const TOTAL: &[FunctionRule] = &[FunctionRule {
        name: "total",
        compile: |compiler, node, depth, nodes| {
            let bind: Bind = |c, _| Ok(DynFunction::new(c.env(), Total)?);
            let signature = (&[DataType::Float64; 20][..], DataType::Float64);
            call_bound(compiler, (node, depth), nodes, signature, bind)
        },
    }];

    struct Total;

    impl Function for Total {
        fn run(
            &self,
            context: &mut Context,
            arguments: &[&ColumnView],
        ) -> Result<ColumnView, pipit_kernel::error::Error> {
            let validity = validity_of(context, arguments)?;
            let rows = arguments[0].float64s().len();
            let mut values = context.values_buffer(rows * 8)?;
            let out = values.as_mut_slice::<f64>();
            out.fill(0.0);
            for argument in arguments {
                for (o, &x) in out.iter_mut().zip(argument.float64s()) {
                    *o += x;
                }
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
            let mut buffer = Buffer::allocate(Env::current(), 16).unwrap();
            buffer.as_mut_slice::<i64>().copy_from_slice(&values);
            ColumnView::new(&mut Context::new(Env::current(), None), data_type, buffer, None)
                .unwrap()
        };
        let floats = [0.5_f64.to_bits().cast_signed(), 2.5_f64.to_bits().cast_signed()];
        let mut views = Buffer::allocate(Env::current(), 16).unwrap();
        views.as_mut_slice::<u32>().copy_from_slice(&[0, 4, 4, 0]);
        let mut bytes = Buffer::allocate(Env::current(), 4).unwrap();
        bytes.as_mut_slice::<u8>().copy_from_slice(b"it's");
        let strings =
            ColumnView::strings(&mut Context::new(Env::current(), None), views, bytes, None)
                .unwrap();
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
        let table = Table::new(Env::current(), &schema, &[&columns]).unwrap();
        let same = [column(DataType::Int64, [1, 2]), column(DataType::Int64, [3, 4])];
        let same_names = [("a", DataType::Int64), ("A", DataType::Int64)];
        let same = Table::new(Env::current(), &same_names, &[&same]).unwrap();
        // `x` is 1 and null.
        let mut values = Buffer::allocate(Env::current(), 16).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(&[1, 0]);
        let mut validity = Buffer::allocate(Env::current(), 8).unwrap();
        validity.as_mut_slice::<u8>().copy_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);
        let x = ColumnView::new(
            &mut Context::new(Env::current(), None),
            DataType::Int64,
            values,
            Some(validity),
        );
        let nulls =
            Table::new(Env::current(), &[("x", DataType::Int64)], &[&[x.unwrap()]]).unwrap();
        let scannable = |table| DynScannable::new(Env::current(), table).unwrap();
        // `id` is 10, 11 and 12, stored as a sequence, and `v` 1, 2 and 3.
        let id = ColumnView::sequence(&mut Context::new(Env::current(), None), 10, 3).unwrap();
        let mut values = Buffer::allocate(Env::current(), 24).unwrap();
        values.as_mut_slice::<i64>().copy_from_slice(&[1, 2, 3]);
        let v =
            ColumnView::new(&mut Context::new(Env::current(), None), DataType::Int64, values, None)
                .unwrap();
        let schema = [("id", DataType::Int64), ("v", DataType::Int64)];
        let ids = Table::new(Env::current(), &schema, &[&[id, v]]).unwrap();
        Tables(scannable(table), scannable(same), scannable(nulls), scannable(ids))
    }

    /// `query` over `catalog`, compiled and prepared in `session`.
    fn prepare<'e>(session: &Session<'e>, catalog: &'e Tables, query: &[u8]) -> Prepared<'e> {
        let query = compile(Env::current(), &REGISTRY, catalog, query).unwrap();
        session.prepare(query).unwrap()
    }

    /// Runs `query`, returning each result column's name and the values of
    /// the rows kept, as floats.
    fn run(query: &str) -> StdVec<(StdVec<u8>, StdVec<f64>)> {
        let catalog = catalog();
        let mut session = Session::new(Env::current());
        let prepared = prepare(&session, &catalog, query.as_bytes());
        let mut rows = prepared.run(&mut session).unwrap();
        let mut columns: StdVec<_> =
            prepared.columns().iter().map(|&c| (prepared.name(c).into(), StdVec::new())).collect();
        while let Some(batch) = rows.next().unwrap() {
            let rows: StdVec<usize> = match batch.selection().kept() {
                Kept::Prefix => (0..batch.selection().len() as usize).collect(),
                Kept::None => StdVec::new(),
                Kept::Select(rows) => rows.iter().map(|&row| usize::from(row)).collect(),
            };
            for (&c, (_, values)) in prepared.columns().iter().zip(&mut columns) {
                let column = batch.column(c.position);
                values.extend(rows.iter().map(|&row| match column.data_type() {
                    // A null, which no test value is.
                    #[expect(clippy::cast_possible_truncation, reason = "a batch's row")]
                    _ if column.is_null(row as u32) => f64::NEG_INFINITY,
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
        let error = compile(Env::current(), &REGISTRY, &catalog, query.as_bytes()).err().unwrap();
        (error.code, error.span.start)
    }

    #[test]
    fn runs_only_in_the_session_it_was_prepared_in() {
        let catalog = catalog();
        let (mut first, mut second) = (Session::new(Env::current()), Session::new(Env::current()));
        let prepared = prepare(&first, &catalog, b"FROM t");
        assert_eq!(prepared.run(&mut second).err(), Some(pipit_kernel::error::Error::WrongSession));
        assert!(prepared.run(&mut first).unwrap().next().unwrap().is_some());
    }

    #[test]
    fn explains_how_a_query_runs() {
        let (catalog, session) = (catalog(), Session::new(Env::current()));
        let prepared = prepare(&session, &catalog, b"FROM t |> ORDER BY a |> LIMIT 1");
        let mut explained = std::string::String::new();
        prepared.explain(&mut explained).unwrap();
        assert_eq!(explained, "Scan | TopK / Served");
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
    fn joins_tables() {
        let joined = |query| {
            let mut columns = run(query).into_iter().map(|(_, values)| values);
            (columns.next().unwrap(), columns.next().unwrap())
        };
        assert_eq!(
            joined("FROM t |> JOIN ids ON b = id |> SELECT a, v"),
            ([1.0].into(), [1.0].into())
        );
        // Pairs come out a row of the joined table at a time.
        let less = ([1.0, 1.0, 2.0].into(), [2.0, 3.0, 3.0].into());
        assert_eq!(joined("FROM t |> INNER JOIN ids ON a < v |> SELECT a, v"), less);
        assert_eq!(kept("FROM t |> JOIN ids ON b = id AND a = v |> SELECT a"), [1.0]);
        assert_eq!(kept("FROM t |> JOIN ids ON b = id AND a = 2 |> SELECT a"), []);
        // Conditions that aren't comparisons, which have no keys. The
        // optimizer picks the sides, so the order isn't fixed.
        let mut all = kept("FROM t |> JOIN ids ON a |> SELECT a");
        all.sort_by(f64::total_cmp);
        assert_eq!(all, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        assert_eq!(kept("FROM t |> JOIN ids ON s LIKE 'it%' |> SELECT v"), [1.0, 2.0, 3.0]);
        let null = f64::NEG_INFINITY;
        let left = ([10.0, 11.0, 12.0].into(), [1.0, null, null].into());
        assert_eq!(joined("FROM ids |> LEFT JOIN t ON id = b |> SELECT id, a"), left);
        assert_eq!(joined("FROM ids |> left outer join t ON id = b |> SELECT id, a"), left);
        assert_eq!(
            kept("FROM ids |> LEFT JOIN t ON id = b |> WHERE a IS NULL |> SELECT id"),
            [11.0, 12.0]
        );
        // A condition never true for nulls makes the join inner.
        assert_eq!(kept("FROM ids |> LEFT JOIN t ON id = b |> WHERE a > 0 |> SELECT id"), [10.0]);
        // The optimizer picks a cross join's sides, so its rows' order isn't
        // fixed.
        let (a, id) = joined("FROM t |> CROSS JOIN ids |> SELECT a, id");
        let mut product: StdVec<(f64, f64)> = a.into_iter().zip(id).collect();
        product.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.total_cmp(&y.1)));
        let pairs = [(1.0, 10.0), (1.0, 11.0), (1.0, 12.0), (2.0, 10.0), (2.0, 11.0), (2.0, 12.0)];
        assert_eq!(product, pairs);
        // Of only the rows so far's columns, so no column of `ids` is held.
        let mut a = kept("FROM t |> CROSS JOIN ids |> SELECT a");
        a.sort_by(f64::total_cmp);
        assert_eq!(a, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn joins_using_columns() {
        let names = |query| run(query).into_iter().map(|(name, _)| name).collect::<StdVec<_>>();
        // The column joined on once, first, then the rest of each side's.
        let joined = names("FROM t |> RENAME b AS id |> JOIN ids USING (id)");
        assert_eq!(joined, [&b"id"[..], b"a", b"f", b"s", b"v"]);
        // A second copy of a column stays.
        let twice = names("FROM t |> RENAME b AS id |> EXTEND a |> JOIN ids USING (id)");
        assert_eq!(twice, [&b"id"[..], b"a", b"f", b"s", b"a", b"v"]);
        assert_eq!(kept("FROM t |> RENAME b AS id |> JOIN ids USING (id) |> SELECT v"), [1.0]);
        // A left join's comes from the rows so far, so is never null.
        let null = f64::NEG_INFINITY;
        let left = run("FROM ids |> RENAME v AS a |> LEFT JOIN t USING (a) |> SELECT a, id, b");
        let values: StdVec<_> = left.into_iter().map(|(_, values)| values).collect();
        assert_eq!(values, [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [10.0, 20.0, null]]);
        let two = "FROM t |> RENAME b AS id |> EXTEND a AS v |> JOIN ids USING (id, v) |> SELECT a";
        assert_eq!(kept(two), [1.0]);
        assert_eq!(
            kept("FROM t |> RENAME b AS id, a AS v |> JOIN ids USING (v, id) |> SELECT v"),
            [1.0]
        );
    }

    #[test]
    fn names_columns_by_table_aliases() {
        // A table's name is its alias, and a column keeps its own name.
        assert_eq!(run("FROM t |> SELECT t.a"), [(b"a".to_vec(), [1.0, 2.0].to_vec())]);
        assert_eq!(kept("FROM t AS x |> WHERE x.a > 1 |> SELECT x.a"), [2.0]);
        // `AS` names what a stage made.
        assert_eq!(kept("FROM t |> SELECT a + 1 AS a |> AS s |> SELECT s.a"), [2.0, 3.0]);
        // An alias names the columns its table had, whatever their names now.
        assert_eq!(kept("FROM t |> RENAME a AS z |> SELECT t.a"), [1.0, 2.0]);
        // `SELECT` makes a new table, which no alias names.
        assert_eq!(error("FROM t |> SELECT a |> SELECT t.a"), (ErrorCode::UnknownColumn, 29));
        assert_eq!(error("FROM t |> SELECT u.a"), (ErrorCode::UnknownColumn, 17));
        assert_eq!(error("FROM t AS x |> SELECT t.a"), (ErrorCode::UnknownColumn, 22));
    }

    #[test]
    fn names_joined_columns_by_table_aliases() {
        let values =
            |query| run(query).into_iter().map(|(_, values)| values).collect::<StdVec<_>>();
        // A table joined to itself, told apart by aliases.
        let selfjoin = values("FROM t AS x |> JOIN t AS y ON x.a < y.a |> SELECT x.a, y.b");
        assert_eq!(selfjoin, [[1.0], [20.0]]);
        let named = run("FROM t |> JOIN ids ON t.b = ids.id |> SELECT t.a, ids.v");
        assert_eq!(named, [(b"a".to_vec(), [1.0].to_vec()), (b"v".to_vec(), [1.0].to_vec())]);
        let renamed =
            values("FROM t |> SELECT a + 1 AS a |> AS s |> JOIN t ON s.a = t.a |> SELECT s.a, t.b");
        assert_eq!(renamed, [[2.0], [20.0]]);
        let joined =
            "FROM t AS x |> JOIN t AS y ON x.a < y.a |> WHERE x.b + y.b = 30 |> SELECT x.a";
        assert_eq!(kept(joined), [1.0]);
        assert_eq!(error("FROM t |> JOIN t ON t.a = 1"), (ErrorCode::AmbiguousColumn, 20));
        // More keys than the 4 a join hashed by.
        let keys = "x.a = y.a AND x.b = y.b AND x.s = y.s AND y.a = x.a AND y.b = x.b";
        let joined = format!("FROM t AS x |> JOIN t AS y ON {keys} |> SELECT y.b");
        assert_eq!(values(&joined), [[10.0, 20.0]]);
        // More tables than the 64 aliases there were room for.
        let many: StdVec<_> =
            (1..70).map(|i| format!(" |> JOIN ids AS a{i} ON a0.id = a{i}.id")).collect();
        let many = format!("FROM ids AS a0{} |> SELECT a0.v, a69.v", many.concat());
        assert_eq!(values(&many).len(), 2);
    }

    #[test]
    fn rejects_joins_it_cant_run() {
        assert_eq!(error("FROM t |> JOIN ids"), (ErrorCode::ExpectedWord, 15));
        assert_eq!(error("FROM t |> JOIN ids USING (id)"), (ErrorCode::UnknownColumn, 26));
        assert_eq!(
            error("FROM t |> RENAME b AS id |> JOIN ids USING (id, id)"),
            (ErrorCode::ListedTwice, 48)
        );
        assert_eq!(
            error("FROM t |> RENAME b AS id |> JOIN ids USING id"),
            (ErrorCode::ExpectedToken, 43)
        );
        let both = error("FROM t |> RENAME b AS id |> JOIN ids ON a = v USING (id)");
        assert_eq!(both, (ErrorCode::UnexpectedToken, 53));
        // `s` is a string, and `id` an integer.
        let types = error("FROM t |> RENAME s AS id |> JOIN ids USING (id)");
        assert_eq!(types, (ErrorCode::ExpectedOneKind, 44));
        assert_eq!(error("FROM t |> CROSS JOIN ids ON a = id"), (ErrorCode::UnexpectedToken, 25));
        assert_eq!(error("FROM t |> LEFT JOIN ids, n ON a = id"), (ErrorCode::UnexpectedToken, 25));
        assert_eq!(error("FROM t |> LEFT ids"), (ErrorCode::ExpectedWord, 15));
        // Both tables have `a`.
        assert_eq!(error("FROM t |> JOIN t ON a = 1"), (ErrorCode::AmbiguousColumn, 20));
        assert_eq!(error("FROM t |> JOIN ids ON s"), (ErrorCode::NumberCondition, 22));
        assert_eq!(error("FROM t |> JOIN ids ON f"), (ErrorCode::Unsupported, 22));
    }

    #[test]
    fn computes_logic() {
        // `AND`, `OR` and `NOT`, null only where unknown: `x` is 1, then null.
        let null = f64::NEG_INFINITY;
        assert_eq!(kept("FROM n |> SELECT x = 1 AND x = 1 AS y"), [1.0, null]);
        assert_eq!(kept("FROM n |> SELECT x = 2 AND x = 1 AS y"), [0.0, null]);
        assert_eq!(kept("FROM n |> SELECT x = 1 OR x IS NULL AS y"), [1.0, 1.0]);
        assert_eq!(kept("FROM n |> SELECT x = 2 OR x = 1 AS y"), [1.0, null]);
        assert_eq!(kept("FROM n |> SELECT x IS NULL AND x = 1 AS y"), [0.0, null]);
        assert_eq!(kept("FROM n |> SELECT NOT x = 1 AS y"), [0.0, null]);
        assert_eq!(error("FROM t |> SELECT s AND a = 1 AS y"), (ErrorCode::NumberCondition, 17));
    }

    #[test]
    fn computes_in_lists() {
        let null = f64::NEG_INFINITY;
        assert_eq!(kept("FROM t |> SELECT a IN (2, 3) AS y"), [0.0, 1.0]);
        assert_eq!(kept("FROM t |> SELECT a + 1 IN (2, -5) AS y"), [1.0, 0.0]);
        assert_eq!(kept("FROM t |> SELECT s IN ('it''s', 'x') AS y"), [1.0, 0.0]);
        assert_eq!(kept("FROM n |> SELECT x IN (1) AS y"), [1.0, null]);
        // More than a filter scans, so looked up in a set.
        assert_eq!(
            kept(
                "FROM t |> SELECT b IN (30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 20) AS y"
            ),
            [0.0, 1.0]
        );
        assert_eq!(kept("FROM t |> WHERE a * 10 IN (20) |> SELECT a"), [2.0]);
        assert_eq!(error("FROM t |> SELECT a IN (1, 'x') AS y"), (ErrorCode::ExpectedOneKind, 26));
        assert_eq!(error("FROM t |> SELECT s IN (1) AS y"), (ErrorCode::ExpectedOneKind, 17));
    }

    #[test]
    fn filters_by_computed_conditions() {
        // What a predicate can't test is computed.
        assert_eq!(kept("FROM t |> WHERE a < b |> SELECT a"), [1.0, 2.0]);
        assert_eq!(kept("FROM t |> WHERE a * 10 = b AND a > 1 |> SELECT a"), [2.0]);
        assert_eq!(kept("FROM n |> WHERE x = x |> SELECT x"), [1.0]);
        assert_eq!(error("FROM t |> WHERE s"), (ErrorCode::NumberCondition, 16));
        assert_eq!(error("FROM t |> WHERE f"), (ErrorCode::Unsupported, 16));
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
        let query = compile(Env::current(), &REGISTRY, &catalog, query).unwrap();
        let plan = query.plan();
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
        let mut session = Session::new(Env::current());
        let prepared = prepare(&session, &catalog, query);
        let overflow = prepared.run(&mut session).unwrap().next().err();
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
        // Doubles, of 0.5 and 2.5, without keys and in a group each.
        let doubles = "FROM t |> AGGREGATE sum(f), avg(f), min(f), max(f)";
        let values: StdVec<StdVec<f64>> = run(doubles).into_iter().map(|(_, v)| v).collect();
        assert_eq!(values, [[3.0], [1.5], [0.5], [2.5]]);
        let mut grouped = kept("FROM t |> AGGREGATE sum(f) AS x GROUP BY a |> SELECT x");
        grouped.sort_by(f64::total_cmp);
        assert_eq!(grouped, [0.5, 2.5]);
        // Of expressions, and by one, as if computed by an `EXTEND` first.
        let expressions = run("FROM t |> AGGREGATE sum(a + 1) AS x, max(f * 2) AS y");
        assert_eq!(expressions, [(b"x".to_vec(), [5.0].to_vec()), (b"y".to_vec(), [5.0].to_vec())]);
        let mut by = run("FROM t |> AGGREGATE sum(b) AS total GROUP BY a * 10 AS k");
        assert_eq!(by.remove(0).0, b"k");
        let mut totals = by.remove(0).1;
        totals.sort_by(f64::total_cmp);
        assert_eq!(totals, [10.0, 20.0]);
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
        let total = format!("FROM t |> SELECT total({}) AS x", ["f"; 20].join(", "));
        assert_eq!(kept(&total), [10.0, 50.0]);
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
    fn runs_long_chains_of_stages() {
        // A filter can't move below a limit, so each stays a node.
        let limits = |n: usize| format!("FROM t{}", " |> LIMIT 9 |> WHERE a > 1".repeat(n));
        // Two nodes a stage pair, under a projection and over a scan.
        let deepest = (pipit_kernel::settings::PLAN_DEPTH_MAX - 2) / 2;
        assert_eq!(kept(&format!("{} |> SELECT a", limits(deepest))), [2.0]);
        let deep = limits(pipit_kernel::settings::PLAN_DEPTH_MAX);
        let catalog = catalog();
        let query = compile(Env::current(), &REGISTRY, &catalog, deep.as_bytes()).unwrap();
        let prepared = Session::new(Env::current()).prepare(query);
        assert_eq!(prepared.err(), Some(pipit_kernel::error::Error::TooDeep));
    }

    #[test]
    fn reports_what_it_cant_compile() {
        assert_eq!(error("FROM u"), (ErrorCode::UnknownTable, 5));
        // Conditions a predicate can't test are typed as expressions.
        assert_eq!(error("FROM t |> WHERE a = 'x'"), (ErrorCode::ExpectedOneKind, 16));
        assert_eq!(error("FROM t |> WHERE s > 1"), (ErrorCode::ExpectedOneKind, 16));
        assert_eq!(error("FROM t |> WHERE a LIKE 'x'"), (ErrorCode::ExpectedString, 16));
        assert_eq!(kept("FROM t |> WHERE a + 1 IS NULL |> SELECT a"), []);
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
        // A chain of operators is as deep as it is long.
        let chain = |n| format!("FROM t |> SELECT a{} AS x", " + a".repeat(n));
        let deepest = chain(crate::parser::NESTING_MAX - 1);
        assert!(compile(Env::current(), &REGISTRY, &catalog(), deepest.as_bytes()).is_ok());
        assert_eq!(error(&chain(crate::parser::NESTING_MAX)), (ErrorCode::NestingTooDeep, 17));
        assert_eq!(error("FROM t |> WHERE c > 1"), (ErrorCode::UnknownColumn, 16));
        // Computed, as a predicate can't test them.
        assert_eq!(kept("FROM t |> WHERE a > b |> SELECT a"), []);
        assert_eq!(kept("FROM t |> WHERE a + 1 > 2 |> SELECT a"), [2.0]);
        assert_eq!(kept("FROM t |> WHERE a |> SELECT a"), [1.0, 2.0]);
        assert_eq!(error("FROM t |> WHERE f > 1.5"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> WHERE f > 9007199254740993"), (ErrorCode::Unsupported, 16));
        assert_eq!(
            error("FROM t |> WHERE a > 9223372036854775808"),
            (ErrorCode::NumberTooLarge, 20)
        );
        // As wide as nesting and the tree allow: an `OR` of comparisons nests
        // as deep as it has terms, of four nodes each.
        let room = crate::ast::BLOCK_NODES * crate::ast::BLOCK_SLOTS / 8;
        let terms = (crate::parser::NESTING_MAX - 1).min(room);
        let wide = (0..terms).map(|i| format!("a = {i}")).collect::<StdVec<_>>().join(" OR ");
        assert_eq!(kept(&format!("FROM t |> WHERE {wide} |> SELECT a")), [1.0, 2.0]);
        assert_eq!(error("FROM t |> WHERE f IN (1)"), (ErrorCode::Unsupported, 16));
        // As `a = 'a'` is.
        assert_eq!(error("FROM t |> WHERE s IN ('a', 1)"), (ErrorCode::ExpectedOneKind, 27));
        assert_eq!(error("FROM t |> WHERE a IN ('a')"), (ErrorCode::ExpectedOneKind, 16));
        assert_eq!(error("FROM t |> AGGREGATE median(a)"), (ErrorCode::UnknownFunction, 20));
        assert_eq!(error("FROM t |> ORDER BY a + 1"), (ErrorCode::Unsupported, 19));
        assert_eq!(error("FROM t |> AGGREGATE sum(s)"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> AGGREGATE count(*) GROUP BY f"), (ErrorCode::Unsupported, 38));
        assert_eq!(error("FROM t |> AGGREGATE count(DISTINCT *)"), (ErrorCode::Unsupported, 20));
        assert_eq!(error("FROM t |> AGGREGATE sum(a + 1)"), (ErrorCode::ExpectedName, 24));
        assert_eq!(
            error("FROM t |> AGGREGATE count(*) GROUP BY a + 1"),
            (ErrorCode::ExpectedName, 38)
        );
        assert_eq!(error("FROM t |> WHERE DISTINCT a"), (ErrorCode::ExpectedExpression, 16));
    }

    /// How many columns `query`'s batches have once optimized, and its
    /// result column's values.
    fn optimized(query: &str) -> (u32, StdVec<i64>) {
        let catalog = catalog();
        let mut session = Session::new(Env::current());
        let prepared = prepare(&session, &catalog, query.as_bytes());
        let mut rows = prepared.run(&mut session).unwrap();
        let batch = rows.next().unwrap().unwrap();
        let values = batch.column(prepared.columns()[0].position).int64s().to_vec();
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
