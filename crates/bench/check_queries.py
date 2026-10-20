#!/usr/bin/env python3
"""Runs a suite's queries with pipitdb and with DuckDB, over the same Parquet
files, checks the rows match, and says how many of the suite's queries
pipitdb runs. Floats match if they're within a relative 1e-9, and rows in any
order.

Suites are ours over ClickBench's data, in clickbench/queries.tsv;
ClickBench's own 43 queries, in clickbench/official.tsv; TPC-H and TPC-DS,
in tpc/, whose files tpc/generate.py makes; and queries over nullable
columns, in nulls/, over TPC-DS's tables and those nulls/generate.py and
nulls/perfetto.py make. A query pipitdb can't run yet has
no PipeSQL there.

Usage: crates/bench/check_queries.py SUITE FILE...
SUITE is clickbench, clickbench-official, tpch, tpcds or nulls. Each file is a
table named after it, less a trailing `_N`, as the `suite` example names
them.
"""

import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
QUERIES = {
    "clickbench": HERE / "clickbench/queries.tsv",
    "clickbench-official": HERE / "clickbench/official.tsv",
    "tpch": HERE / "tpc/tpch.tsv",
    "tpcds": HERE / "tpc/tpcds.tsv",
    "nulls": HERE / "nulls/queries.tsv",
}


def rows_by_query(lines):
    rows = {}
    for line in lines:
        number, *values = line.split("\t")
        rows.setdefault(number, []).append(values)
    return rows


def same(a, b):
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    return abs(x - y) <= 1e-9 * max(abs(x), abs(y))


def order(row):
    """A key sorting rows by their values, as numbers where they are."""
    def value(text):
        try:
            return (0, float(text), "")
        except ValueError:
            return (1, 0.0, text)
    return [value(text) for text in row]


def table(path):
    """The table a file is part of, as the `suite` example names it."""
    return re.sub(r"_\d+$", "", pathlib.Path(path).stem)


def main():
    suite, files = sys.argv[1], sys.argv[2:]
    queries = QUERIES[suite]
    run = ["cargo", "run", "-q", "--profile", "speed", "-p", "pipitdb-bench", "--example", "suite", "--"]
    ours = subprocess.run(run + [str(queries)] + files, capture_output=True, text=True, check=True)
    # Each query that ran reports its time; one that failed, its error.
    errors = {}
    for line in ours.stderr.splitlines():
        number, _, report = line.partition(": ")
        if number.startswith("Q") and " peak " not in report:
            errors[number[1:]] = report
    ours = rows_by_query(ours.stdout.splitlines())

    tables = {}
    for file in files:
        tables.setdefault(table(file), []).append(f"'{file}'")
    # ClickBench's strings are stored as binary. Its own queries read its dates
    # as dates, as its DuckDB setup does; ours and pipitdb read the day number.
    select = "* REPLACE (make_date(EventDate) AS EventDate)" if suite == "clickbench-official" else "*"
    views = "".join(
        f"CREATE VIEW {name} AS SELECT {select} FROM read_parquet([{', '.join(paths)}], binary_as_string=true);\n"
        for name, paths in tables.items()
    )
    if suite == "clickbench-official":
        views += "CREATE MACRO toDateTime(t) AS epoch_ms(t * 1000);\n"
    total, written, matched = 0, 0, 0
    for line in queries.read_text().splitlines():
        number, pipesql, sql = line.split("\t")
        total += 1
        if not pipesql:
            continue
        written += 1
        if number in errors:
            print(f"Q{number}: fails in pipitdb: {errors[number]}")
            continue
        duck = subprocess.run(
            ["duckdb", "-noheader", "-list", "-separator", "\t", "-nullvalue", "NULL"],
            input=f"{views}{sql};", capture_output=True, text=True, check=True,
        )
        # Groups come in any order, so rows are compared sorted.
        expected = sorted((line.split("\t") for line in duck.stdout.splitlines()), key=order)
        got = sorted(ours.get(number, []), key=order)
        ok = len(got) == len(expected) and all(
            len(g) == len(e) and all(map(same, g, e)) for g, e in zip(got, expected)
        )
        print(f"Q{number}: {'ok' if ok else 'DIFFERS'}")
        if ok:
            matched += 1
        else:
            print(f"  pipitdb: {got}\n  duckdb:  {expected}")
    print(f"\n{suite}: {matched} of {total} queries match DuckDB; "
          f"{written} have PipeSQL, {written - matched} of those fail or differ")
    sys.exit(0 if matched == written else 1)


if __name__ == "__main__":
    main()
