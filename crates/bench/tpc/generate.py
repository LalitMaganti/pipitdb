#!/usr/bin/env python3
"""Generates TPC-H and TPC-DS data as Parquet, one file per table, with
DuckDB's dbgen and dsdgen, and writes their queries to tpch.tsv and
tpcds.tsv beside this script.

pipitdb can't read decimals yet, so they're written as doubles. Every engine
reads the same files, so answers and times still compare.

Needs `pip install duckdb`.

Usage: crates/bench/tpc/generate.py DIR [--scale 1] [--suites tpch,tpcds]
       crates/bench/tpc/generate.py --queries
DIR gets a directory per suite, such as DIR/tpch/lineitem.parquet.
"""

import argparse
import pathlib
import re

import duckdb

HERE = pathlib.Path(__file__).parent
GENERATE = {"tpch": "CALL dbgen(sf={scale})", "tpcds": "CALL dsdgen(sf={scale})"}


def write_queries(db):
    """Each suite's queries, a line each: its number, its PipeSQL, empty until
    pipitdb can run it, and its SQL on one line."""
    for suite in GENERATE:
        lines = []
        for number, sql in db.execute(f"SELECT query_nr, query FROM {suite}_queries()").fetchall():
            sql = re.sub(r"--[^\n]*", "", sql)
            sql = " ".join(sql.split()).rstrip(";")
            lines.append(f"{number}\t\t{sql}\n")
        (HERE / f"{suite}.tsv").write_text("".join(lines))


def write_data(db, directory, scale, suites):
    for suite in suites:
        db.execute(GENERATE[suite].format(scale=scale))
        out = pathlib.Path(directory) / suite
        out.mkdir(parents=True, exist_ok=True)
        for (table,) in db.execute("SHOW TABLES").fetchall():
            columns = db.execute(f"DESCRIBE {table}").fetchall()
            select = ", ".join(
                f"CAST({name} AS DOUBLE) AS {name}" if kind.startswith("DECIMAL") else name
                for name, kind, *_ in columns
            )
            db.execute(f"COPY (SELECT {select} FROM {table}) TO '{out / table}.parquet' (FORMAT parquet)")
            db.execute(f"DROP TABLE {table}")
        print(f"{suite}: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?")
    parser.add_argument("--scale", default="1")
    parser.add_argument("--suites", default="tpch,tpcds")
    parser.add_argument("--queries", action="store_true")
    args = parser.parse_args()
    db = duckdb.connect()
    db.execute("INSTALL tpch; LOAD tpch; INSTALL tpcds; LOAD tpcds")
    if args.queries:
        write_queries(db)
    else:
        write_data(db, args.directory, args.scale, args.suites.split(","))


if __name__ == "__main__":
    main()
