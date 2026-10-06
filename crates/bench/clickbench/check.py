#!/usr/bin/env python3
"""Runs the ClickBench queries in queries.tsv with pipitdb and with DuckDB, over
the same Parquet files, and checks the rows match. Floats match if they're
within a relative 1e-9, and rows in any order.

Usage: crates/bench/clickbench/check.py FILE...
"""

import pathlib
import subprocess
import sys

QUERIES = pathlib.Path(__file__).with_name("queries.tsv")


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


def main():
    files = sys.argv[1:]
    run = ["cargo", "run", "-q", "--profile", "speed", "-p", "pipitdb-bench", "--example", "clickbench", "--"]
    ours = subprocess.run(run + files, capture_output=True, text=True, check=True)
    sys.stderr.write(ours.stderr)
    ours = rows_by_query(ours.stdout.splitlines())

    paths = ", ".join(f"'{file}'" for file in files)
    failed = False
    for line in QUERIES.read_text().splitlines():
        number, pipesql, sql = line.split("\t")
        # pipitdb can't run it yet.
        if not pipesql:
            continue
        # ClickBench's strings are stored as binary.
        view = f"CREATE VIEW hits AS SELECT * FROM read_parquet([{paths}], binary_as_string=true)"
        script = f"{view};\n{sql};"
        duck = subprocess.run(
            ["duckdb", "-noheader", "-list", "-separator", "\t", "-nullvalue", "NULL"],
            input=script, capture_output=True, text=True, check=True,
        )
        # Groups come in any order, so rows are compared sorted.
        expected = sorted((line.split("\t") for line in duck.stdout.splitlines()), key=order)
        got = sorted(ours.get(number, []), key=order)
        ok = len(got) == len(expected) and all(
            len(g) == len(e) and all(map(same, g, e)) for g, e in zip(got, expected)
        )
        print(f"Q{number}: {'ok' if ok else 'DIFFERS'}")
        if not ok:
            print(f"  pipitdb: {got}\n  duckdb:  {expected}")
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
