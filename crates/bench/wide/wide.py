#!/usr/bin/env python3
"""Times queries over tables 100, 1,000 and 10,000 columns wide, with pipitdb
and with DuckDB, to show what width costs each. Every table has the same
rows, so a query's time across widths shows what each column it doesn't read
costs, as well as each one it does.

`generate` writes DIR/WIDTH/wide.parquet with DuckDB: 100,000 rows in row
groups of 10,000, so a footer describes 10 chunks of each column. Every
tenth column is a string; the rest are integers.

`run` times each query, a fresh process for each run, best of RUNS, in ms:
pipitdb's `suite` example, which needs building first with
`cargo build --profile speed -p pipitdb-bench --example suite`, and DuckDB's
CLI on one thread and on its default. A query an engine can't run is
"failed".

Usage: crates/bench/wide/wide.py generate DIR [--widths 100,1000,10000]
       crates/bench/wide/wide.py run DIR [--widths ...] [--runs 3]
DUCKDB names DuckDB's CLI, `duckdb` by default.
"""

import argparse
import os
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
SUITE = ROOT / "target/speed/examples/suite"
DUCKDB = os.environ.get("DUCKDB", "duckdb")
ROWS = 100_000
GROUP_ROWS = 10_000


def column(k):
    """Column `k`'s name and how it's made from the row number `i`."""
    if k % 10 == 9:
        return f"c{k}", f"'s' || ((i + {k}) % 100)"
    return f"c{k}", f"(i * 7 + {k}) % 1000"


def queries(width):
    """Each query's name, PipeSQL and SQL over a table `width` columns wide."""
    ints = [f"c{k}" for k in range(width) if k % 10 != 9]
    sums = ", ".join(f"sum({c})" for c in ints)
    return [
        # One column, which is all that should be read.
        ("one", "FROM wide |> AGGREGATE sum(c0)", "SELECT sum(c0) FROM wide"),
        # Every column, for ten rows.
        ("star_limit", "FROM wide |> LIMIT 10", "SELECT * FROM wide LIMIT 10"),
        # Every column, for the ten least rows by one: only those ten
        # need more than that one.
        ("star_top", "FROM wide |> ORDER BY c1 |> LIMIT 10",
         "SELECT * FROM wide ORDER BY c1 LIMIT 10"),
        # Every integer column, each read in full.
        ("sum_all", f"FROM wide |> AGGREGATE {sums}", f"SELECT {sums} FROM wide"),
    ]


def generate(directory, widths):
    for width in widths:
        out = pathlib.Path(directory) / str(width)
        out.mkdir(parents=True, exist_ok=True)
        select = ", ".join(f"{make} AS {name}" for name, make in map(column, range(width)))
        sql = (f"COPY (SELECT {select} FROM range({ROWS}) t(i)) TO '{out / 'wide.parquet'}'"
               f" (FORMAT parquet, ROW_GROUP_SIZE {GROUP_ROWS});")
        with tempfile.NamedTemporaryFile("w", suffix=".sql") as script:
            script.write(sql)
            script.flush()
            subprocess.run([DUCKDB, "-no-agent", "-f", script.name], check=True)
        print(f"{width}: {(out / 'wide.parquet').stat().st_size / 1e6:.1f} MB")


def pipitdb_ms(file, name, pipesql):
    """pipitdb's time for one query, or None if it fails."""
    with tempfile.NamedTemporaryFile("w", suffix=".tsv") as tsv:
        tsv.write(f"{name}\t{pipesql}\t\n")
        tsv.flush()
        run = subprocess.run([SUITE, tsv.name, str(file)], capture_output=True, text=True,
                             env=dict(os.environ, QUIET="1"))
    match = re.search(r"^Q\S+: ([\d.]+)(ns|µs|ms|s) ", run.stderr, re.M)
    if not match:
        return None
    value, unit = match.groups()
    return float(value) * {"ns": 1e-6, "µs": 1e-3, "ms": 1.0, "s": 1e3}[unit]


def duckdb_ms(file, sql, threads):
    """DuckDB's time for one query, with `threads` or its default, or None if
    it fails."""
    setup = f"SET threads={threads};\n" if threads else ""
    script = (f"{setup}CREATE VIEW wide AS SELECT * FROM read_parquet('{file}');\n"
              f".mode trash\n.timer on\n{sql};\n")
    run = subprocess.run([DUCKDB, "-no-agent"], input=script, capture_output=True, text=True)
    match = re.search(r"Run Time \(s\): real ([\d.]+)", run.stdout + run.stderr)
    return float(match.group(1)) * 1e3 if match and run.returncode == 0 else None


def best(times):
    times = [t for t in times if t is not None]
    return f"{min(times):.1f}" if times else "failed"


def run(directory, widths, runs):
    print("width\tquery\tpipitdb\tduckdb-1\tduckdb-N")
    for width in widths:
        file = (pathlib.Path(directory) / str(width) / "wide.parquet").resolve()
        for name, pipesql, sql in queries(width):
            ours = best(pipitdb_ms(file, name, pipesql) for _ in range(runs))
            one = best(duckdb_ms(file, sql, 1) for _ in range(runs))
            many = best(duckdb_ms(file, sql, None) for _ in range(runs))
            print(f"{width}\t{name}\t{ours}\t{one}\t{many}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "run"])
    parser.add_argument("directory")
    parser.add_argument("--widths", default="100,1000,10000")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    widths = [int(w) for w in args.widths.split(",")]
    if args.command == "generate":
        generate(args.directory, widths)
    else:
        run(args.directory, widths, args.runs)


if __name__ == "__main__":
    main()
