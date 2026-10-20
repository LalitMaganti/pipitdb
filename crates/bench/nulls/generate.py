#!/usr/bin/env python3
"""Generates the tables the nulls suite reads beside TPC-DS's, as Parquet:
`collide`, whose integers use the whole range, the least included, with
nulls among them, so no value is free to stand for null in every row group.

Needs `pip install duckdb`.

Usage: crates/bench/nulls/generate.py DIR [--rows 2000000]
"""

import argparse
import pathlib

import duckdb

# Spread over the whole range by DuckDB's hash, with the least integer every
# 997th row and a null every 50th.
COLLIDE = """
SELECT
    CASE WHEN i % 50 = 1 THEN NULL
         WHEN i % 997 = 0 THEN -9223372036854775808
         ELSE (hash(i)::HUGEINT - 9223372036854775808)::BIGINT END AS h,
    CASE WHEN i % 37 = 0 THEN NULL
         WHEN i % 1009 = 0 THEN -9223372036854775808
         ELSE i % 1000 END AS k,
    CASE WHEN i % 13 = 0 THEN NULL ELSE (i * 7) % 10000 - 5000 END AS v
FROM range({rows}) t(i)
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--rows", default="2000000")
    args = parser.parse_args()
    out = pathlib.Path(args.directory)
    out.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect()
    db.execute(f"COPY ({COLLIDE.format(rows=int(args.rows))}) TO '{out / 'collide'}.parquet' (FORMAT parquet)")
    print(f"nulls: {out}")


if __name__ == "__main__":
    main()
