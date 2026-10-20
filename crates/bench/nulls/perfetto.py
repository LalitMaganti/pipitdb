#!/usr/bin/env python3
"""Exports the integer columns of a Perfetto trace's `slice`, `thread_state`
and `args` tables as Parquet, for the nulls suite: real tables, whose nulls
come row by row, as they do in traces, and not uniformly, as TPC-DS's do.

Needs `pip install duckdb`, and a trace_processor_shell, such as Perfetto's
`out/linux_clang_release/trace_processor_shell`.

Usage: crates/bench/nulls/perfetto.py TRACE_PROCESSOR TRACE DIR
"""

import pathlib
import subprocess
import sys
import tempfile

import duckdb

TABLES = {
    "slice": "id, ts, dur, track_id, depth, parent_id, arg_set_id, thread_ts, thread_dur",
    "thread_state": "id, ts, dur, utid, cpu, ucpu, waker_utid, waker_id",
    "args": "id, arg_set_id, int_value",
}


def main():
    shell, trace, directory = sys.argv[1:4]
    out = pathlib.Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect()
    with tempfile.TemporaryDirectory() as tmp:
        for table, columns in TABLES.items():
            query = pathlib.Path(tmp) / "query.sql"
            query.write_text(f"SELECT {columns} FROM {table} ORDER BY id")
            csv = pathlib.Path(tmp) / f"{table}.csv"
            with csv.open("w") as f:
                subprocess.run([shell, "-q", str(query), trace], stdout=f, stderr=subprocess.DEVNULL, check=True)
            types = ", ".join(f"'{c.strip()}': 'BIGINT'" for c in columns.split(","))
            db.execute(
                f"COPY (SELECT * FROM read_csv('{csv}', header=true, nullstr='[NULL]', columns={{{types}}})) "
                f"TO '{out / table}.parquet' (FORMAT parquet)"
            )
    print(f"perfetto: {out}")


if __name__ == "__main__":
    main()
