//! Joins compiled from PipeSQL and run over in-memory tables, checked
//! against each pair of rows tested by hand.

extern crate std;

use core::fmt::Write;
use std::format;
use std::string::String;
use std::vec::Vec;

use pipit_kernel::buffer::Buffer;
use pipit_kernel::column::{ColumnView, DataType};
use pipit_kernel::containers::BitSlice;
use pipit_kernel::context::Context;
use pipit_kernel::env::Env;
use pipit_kernel::row_batch::BATCH_ROWS_MAX;
use pipit_kernel::scannable::{Catalog, DynScannable};
use pipit_kernel::session::Session;
use pipit_operators::table::Table;

use crate::compile::compile;
use crate::registry::Registry;
use crate::stages::{JOINS, RELATIONAL};

static REGISTRY: Registry = Registry::new(&[RELATIONAL, JOINS]);

fn int64s(values: &[Option<i64>]) -> ColumnView {
    let mut buffer = Buffer::allocate(Env::current(), values.len() * 8).unwrap();
    for (slot, value) in buffer.as_mut_slice::<i64>().iter_mut().zip(values) {
        *slot = value.unwrap_or(0);
    }
    // A bit a row, set where it isn't null.
    let validity = values.iter().any(Option::is_none).then(|| {
        let mut bits =
            Buffer::allocate(Env::current(), BitSlice::<u8>::words_for(values.len())).unwrap();
        let bytes = bits.as_mut_slice::<u8>();
        bytes.fill(0);
        for (row, value) in values.iter().enumerate() {
            BitSlice::of_mut(&mut *bytes).set_if(row, value.is_some());
        }
        bits
    });
    ColumnView::new(&mut Context::new(Env::current(), None), DataType::Int64, buffer, validity)
        .unwrap()
}

/// A table of its key column, as given, then its row number column, named
/// `key` and `row`, in batches.
fn table(names: [&str; 2], keys: &[Option<i64>]) -> DynScannable<'static> {
    let groups: Vec<[ColumnView; 2]> = keys
        .chunks(BATCH_ROWS_MAX as usize)
        .zip((0..).step_by(BATCH_ROWS_MAX as usize))
        .map(|(keys, first)| {
            let rows: Vec<Option<i64>> = (first..).take(keys.len()).map(Some).collect();
            [int64s(keys), int64s(&rows)]
        })
        .collect();
    tabled(names, &groups)
}

/// A table of its key column, stored as a sequence from each group's first
/// for its rows, then its row number column; and its keys.
fn sequences(names: [&str; 2], groups: &[(i64, u32)]) -> (DynScannable<'static>, Vec<Option<i64>>) {
    let mut keys = Vec::new();
    let mut row = 0;
    let groups: Vec<[ColumnView; 2]> = groups
        .iter()
        .map(|&(first, rows)| {
            keys.extend((first..).take(rows as usize).map(Some));
            let numbers: Vec<Option<i64>> = (row..).take(rows as usize).map(Some).collect();
            row += i64::from(rows);
            let context = &mut Context::new(Env::current(), None);
            [ColumnView::sequence(context, first, rows).unwrap(), int64s(&numbers)]
        })
        .collect();
    (tabled(names, &groups), keys)
}

fn tabled(names: [&str; 2], groups: &[[ColumnView; 2]]) -> DynScannable<'static> {
    let groups: Vec<&[ColumnView]> = groups.iter().map(|group| &group[..]).collect();
    let schema = [(names[0], DataType::Int64), (names[1], DataType::Int64)];
    DynScannable::new(Env::current(), Table::new(Env::current(), &schema, &groups).unwrap())
        .unwrap()
}

/// Tables by name.
struct Tables<'t>(Vec<(String, &'t DynScannable<'static>)>);

impl Catalog for Tables<'_> {
    fn find(&self, name: &str) -> Option<&DynScannable<'_>> {
        self.0.iter().find(|(table, _)| table == name).map(|&(_, table)| table)
    }
}

/// Each row of `query` over `tables`, as its columns' values, a null as
/// `None`, sorted.
fn run(tables: &Tables<'_>, query: &str) -> Vec<Vec<Option<i64>>> {
    let mut session = Session::new(Env::current());
    let query = compile(Env::current(), &REGISTRY, tables, query.as_bytes()).unwrap();
    let prepared = session.prepare(query).unwrap();
    let positions: Vec<u32> = prepared.columns().iter().map(|column| column.position).collect();
    let mut rows = Vec::new();
    let mut out = prepared.run(&mut session).unwrap();
    while let Some(batch) = out.next().unwrap() {
        let selection = batch.selection();
        for row in (0..selection.len()).map(|i| selection.kept().row(i)) {
            let value = |&position: &u32| {
                let column = batch.column(position);
                (!column.is_null(row)).then(|| column.int64s()[row as usize])
            };
            rows.push(positions.iter().map(value).collect());
        }
    }
    rows.sort_unstable();
    rows
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Inner,
    Left,
}

/// A join's condition, of the probe side's `pk` and `pv` and the build
/// side's `bk` and `bv`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum On {
    /// `pk = bk`, which the join hashes by.
    Equal,
    /// `pk < bk`, which it can't.
    Less,
    /// `pk = bk AND pv < bv`, which it hashes by `pk`, then tests.
    EqualAndLess,
    /// None: a cross join.
    Always,
}

/// A filter above a join, on the build side's `bv`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Above {
    Nothing,
    IsNull,
    Positive,
}

const KINDS: [Kind; 2] = [Kind::Inner, Kind::Left];

/// Each output row's `pv` and `bv` of `p <kind> JOIN b ON <on>`, under
/// filter `above`, sorted.
fn join(
    probe: &[Option<i64>],
    build: &DynScannable<'static>,
    kind: Kind,
    on: On,
    above: Above,
) -> Vec<(i64, Option<i64>)> {
    let p = table(["pk", "pv"], probe);
    let tables = Tables(Vec::from([(String::from("p"), &p), (String::from("b"), build)]));
    let condition = match on {
        On::Equal => "ON pk = bk",
        On::Less => "ON pk < bk",
        On::EqualAndLess => "ON pk = bk AND pv < bv",
        On::Always => "ON 1 = 1",
    };
    let joined = match (kind, on) {
        (Kind::Inner, On::Always) => String::from("CROSS JOIN b"),
        (Kind::Inner, _) => format!("JOIN b {condition}"),
        (Kind::Left, _) => format!("LEFT JOIN b {condition}"),
    };
    let filter = match above {
        Above::Nothing => "",
        Above::IsNull => " |> WHERE bv IS NULL",
        Above::Positive => " |> WHERE bv > 0",
    };
    let query = format!("FROM p |> {joined}{filter} |> SELECT pv, bv");
    run(&tables, &query).into_iter().map(|row| (row[0].unwrap(), row[1])).collect()
}

/// What `join` should give, by testing every pair.
fn expected(
    probe: &[Option<i64>],
    build: &[Option<i64>],
    kind: Kind,
    on: On,
    above: Above,
) -> Vec<(i64, Option<i64>)> {
    // A probe row's `pv` is its number, as is a build row's `bv`.
    let holds = |(p, a): (i64, Option<i64>), (b, k): (i64, Option<i64>)| match (on, a, k) {
        (On::Always, _, _) => true,
        (_, None, _) | (_, _, None) => false,
        (On::Equal, Some(a), Some(k)) => a == k,
        (On::Less, Some(a), Some(k)) => a < k,
        (On::EqualAndLess, Some(a), Some(k)) => a == k && p < b,
    };
    let mut rows = Vec::new();
    for (p, &a) in (0..).zip(probe) {
        let matches: Vec<i64> =
            (0..).zip(build).filter(|&(b, &k)| holds((p, a), (b, k))).map(|(b, _)| b).collect();
        rows.extend(matches.iter().map(|&b| (p, Some(b))));
        if matches.is_empty() && kind == Kind::Left {
            rows.push((p, None));
        }
    }
    match above {
        Above::Nothing => {}
        Above::IsNull => rows.retain(|&(_, b)| b.is_none()),
        Above::Positive => rows.retain(|&(_, b)| b.is_some_and(|b| b > 0)),
    }
    rows.sort_unstable();
    rows
}

#[test]
fn joins_every_kind_on_any_condition() {
    // Probe rows over two batches, so a batch's matches come a build row at
    // a time, and nulls, which match nothing.
    let probe: Vec<Option<i64>> = (0..2100).map(|i| (i % 7 != 3).then_some(i % 5)).collect();
    let build_keys = [Some(1), Some(3), None, Some(3), Some(9)];
    let build = table(["bk", "bv"], &build_keys);
    for kind in KINDS {
        for on in [On::Equal, On::Less, On::EqualAndLess, On::Always] {
            assert_eq!(
                join(&probe, &build, kind, on, Above::Nothing),
                expected(&probe, &build_keys, kind, on, Above::Nothing),
                "{kind:?} {on:?}"
            );
        }
    }
}

#[test]
fn hashes_a_build_side_of_many_batches() {
    // Keys repeat within and across the build side's batches, so a probe
    // row's build rows of its keys come over several rounds.
    let probe: Vec<Option<i64>> = (0..2100).map(|i| (i % 11 != 0).then_some(i % 900)).collect();
    let build_keys: Vec<Option<i64>> =
        (0..3000).map(|i| (i % 13 != 0).then_some(i % 700)).collect();
    let build = table(["bk", "bv"], &build_keys);
    for kind in KINDS {
        for on in [On::Equal, On::EqualAndLess] {
            assert_eq!(
                join(&probe, &build, kind, on, Above::Nothing),
                expected(&probe, &build_keys, kind, on, Above::Nothing),
                "{kind:?} {on:?}"
            );
        }
    }
}

#[test]
fn joins_to_an_empty_build_side() {
    let probe = [Some(1), None];
    let build = table(["bk", "bv"], &[]);
    for kind in KINDS {
        assert_eq!(
            join(&probe, &build, kind, On::Equal, Above::Nothing),
            expected(&probe, &[], kind, On::Equal, Above::Nothing)
        );
    }
}

#[test]
fn keeps_a_null_test_on_the_build_side_above_a_left_join() {
    // `bv IS NULL` keeps only the probe rows nothing matched: moved to the
    // build side, it would drop every build row first.
    let probe = [Some(1), Some(2), Some(3)];
    let build_keys = [Some(1), Some(3)];
    let build = table(["bk", "bv"], &build_keys);
    let rows = join(&probe, &build, Kind::Left, On::Equal, Above::IsNull);
    assert_eq!(rows, [(1, None)]);
    assert_eq!(rows, expected(&probe, &build_keys, Kind::Left, On::Equal, Above::IsNull));
}

#[test]
fn finds_build_rows_by_their_keys() {
    let probe: Vec<Option<i64>> =
        [Some(5), Some(-1), Some(2999), None, Some(3000), Some(2048), Some(5), Some(2), Some(9)]
            .into_iter()
            .chain((0..3000).map(|i| Some(i * 7 % 3100)))
            .collect();
    // Ids across batches, each following the last, as a table's are, found
    // with no bitmap; in another order, without gaps, found by their number;
    // with a gap, and in another order, found by a bitmap; and repeated,
    // hashed.
    let groups: [&[(i64, u32)]; 5] = [
        &[(0, 2048), (2048, 952)],
        &[(4, 4), (0, 4)],
        &[(0, 4), (5, 4)],
        &[(10, 4), (0, 4)],
        &[(0, 4), (2, 4)],
    ];
    for groups in groups {
        let (build, build_keys) = sequences(["bk", "bv"], groups);
        for kind in KINDS {
            for on in [On::Equal, On::EqualAndLess] {
                assert_eq!(
                    join(&probe, &build, kind, on, Above::Nothing),
                    expected(&probe, &build_keys, kind, on, Above::Nothing),
                    "{groups:?} {kind:?} {on:?}"
                );
            }
        }
    }
}

#[test]
fn filters_above_a_left_join() {
    // A filter on the build side's `bv` can't move below a left join: it
    // would leave probe rows with no build rows, which get nulls.
    let probe = [Some(1), Some(2), Some(3)];
    let build_keys = [Some(1), Some(3), Some(3)];
    let build = table(["bk", "bv"], &build_keys);
    for above in [Above::Positive, Above::IsNull] {
        assert_eq!(
            join(&probe, &build, Kind::Left, On::Equal, above),
            expected(&probe, &build_keys, Kind::Left, On::Equal, above)
        );
    }
}

#[test]
fn finds_no_build_row_for_a_null_among_ids() {
    // The build side's keys are its rows' numbers but for a null, so a key
    // finds its row by its value, but the null row is no key's.
    let ids: Vec<Option<i64>> = (0..40).map(|i| (i != 5).then_some(i)).collect();
    let probe: Vec<Option<i64>> = (0..50).map(Some).collect();
    let build = table(["bk", "bv"], &ids);
    for kind in KINDS {
        assert_eq!(
            join(&probe, &build, kind, On::Equal, Above::Nothing),
            expected(&probe, &ids, kind, On::Equal, Above::Nothing),
            "{kind:?}"
        );
    }
}

#[test]
fn builds_the_smaller_side_of_a_left_join() {
    // The first side is much smaller, so is built, and its rows nothing
    // matched come out at the end. Its keys are its rows' numbers but for a
    // null, which no probe row's number may find.
    let small: Vec<Option<i64>> = (0..40).map(|i| (i != 5).then_some(i)).collect();
    let large: Vec<Option<i64>> = (0..2100).map(|i| (i % 7 != 3).then_some(i % 45)).collect();
    let right = table(["bk", "bv"], &large);
    for kind in KINDS {
        for on in [On::Equal, On::Less, On::EqualAndLess, On::Always] {
            for above in [Above::Nothing, Above::IsNull, Above::Positive] {
                assert_eq!(
                    join(&small, &right, kind, on, above),
                    expected(&small, &large, kind, on, above),
                    "{kind:?} {on:?}"
                );
            }
        }
    }
    // Nothing on the second side, or on the first.
    for (first, second) in [(&small[..], &[][..]), (&[][..], &large[..])] {
        assert_eq!(
            join(first, &table(["bk", "bv"], second), Kind::Left, On::Equal, Above::Nothing),
            expected(first, second, Kind::Left, On::Equal, Above::Nothing)
        );
    }
}

#[test]
fn joins_several_tables_in_any_order() {
    // Four tables of very different sizes, each `vN` its row number, joined
    // along a chain: v0 = k1, v1 = k2, v2 = k3. Any order it's written in
    // gives the same rows.
    let sizes = [3000, 40, 900, 2000];
    let keys: Vec<Vec<Option<i64>>> = (0..4_u8)
        .map(|t| {
            (0..sizes[usize::from(t)])
                .map(|i| (i % 13 != 4).then_some((i * 7 + i64::from(t)) % 50))
                .collect()
        })
        .collect();
    let owned: Vec<DynScannable<'static>> = (0..4)
        .map(|t| table([format!("k{t}").as_str(), format!("v{t}").as_str()], &keys[t]))
        .collect();
    let tables = Tables((0..).zip(&owned).map(|(t, table)| (format!("t{t}"), table)).collect());
    let edges = [(0, 1), (1, 2), (2, 3)];
    let expected = joined_expected(&keys, &edges);
    for order in [[0, 1, 2, 3], [3, 2, 1, 0], [1, 2, 0, 3], [2, 1, 3, 0]] {
        // Each table after the first joins those before it on each edge
        // between them, one condition each.
        let mut query = format!("FROM t{}", order[0]);
        for (i, &table) in order.iter().enumerate().skip(1) {
            let before = &order[..i];
            let conditions: Vec<String> = edges
                .iter()
                .filter(|&&(a, b)| {
                    (a == table && before.contains(&b)) || (b == table && before.contains(&a))
                })
                .map(|&(a, b)| format!("v{a} = k{b}"))
                .collect();
            if conditions.is_empty() {
                _ = write!(query, " |> CROSS JOIN t{table}");
            } else {
                _ = write!(query, " |> JOIN t{table} ON {}", conditions.join(" AND "));
            }
        }
        query += " |> SELECT v0, v1, v2, v3";
        let rows: Vec<Vec<i64>> = run(&tables, &query)
            .into_iter()
            .map(|row| row.into_iter().map(Option::unwrap).collect())
            .collect();
        assert_eq!(rows, expected, "{order:?}");
    }
}

/// What `joins_several_tables_in_any_order` should give: every combination
/// of rows each edge holds for, by brute force.
fn joined_expected(keys: &[Vec<Option<i64>>], edges: &[(usize, usize)]) -> Vec<Vec<i64>> {
    // A table's `v` is its row number, and its `k` is given.
    let mut rows: Vec<Vec<i64>> = Vec::from([Vec::new()]);
    for (table, table_keys) in keys.iter().enumerate() {
        let mut next = Vec::new();
        for row in &rows {
            for (v, &k) in (0..).zip(table_keys) {
                let holds = edges.iter().all(|&(a, b)| match (a == table, b == table) {
                    // `a.v = b.k`, once both are in.
                    (false, true) if a < table => k == Some(row[a]),
                    (true, false) if b < table => {
                        Some(v) == keys[b][usize::try_from(row[b]).unwrap()]
                    }
                    _ => true,
                });
                if holds {
                    let mut row = row.clone();
                    row.push(v);
                    next.push(row);
                }
            }
        }
        rows = next;
    }
    rows.sort_unstable();
    rows
}
