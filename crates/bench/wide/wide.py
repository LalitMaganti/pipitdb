#!/usr/bin/env python3
"""Times queries over tables 100, 1,000 and 10,000 columns wide, with pipitdb
and other engines, to show what width costs each. Every table has the same
rows, so a query's time across widths shows what each column it doesn't read
costs, as well as each one it does.

`generate` writes DIR/WIDTH/wide.parquet with DuckDB: 100,000 rows in row
groups of 10,000, so a footer describes 10 chunks of each column. Every
tenth column is a string; the rest are integers.

`run` times each query, best of RUNS, in ms:
each run is a fresh process that runs every query once. Each engine first
runs `SELECT 1`, which starts it without reading the file.
pipitdb is its `suite` example, which needs building first with
`cargo build --profile speed -p pipitdb-bench --example suite`. Other engines
run in-process, on one thread and on their default, and return their rows
as Arrow. A query an engine can't run is "failed".

Needs `pip install duckdb chdb datafusion polars`.

Usage: crates/bench/wide/wide.py generate DIR [--widths 100,1000,10000]
       crates/bench/wide/wide.py run DIR [--widths ...] [--engines ...] [--runs 3]
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
SUITE = ROOT / "target/speed/examples/suite"
ENGINES = ["pipitdb", "duckdb", "clickhouse", "datafusion", "polars"]
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
    import duckdb
    for width in widths:
        out = pathlib.Path(directory) / str(width)
        out.mkdir(parents=True, exist_ok=True)
        select = ", ".join(f"{make} AS {name}" for name, make in map(column, range(width)))
        duckdb.connect().execute(
            f"COPY (SELECT {select} FROM range({ROWS}) t(i)) TO '{out / 'wide.parquet'}'"
            f" (FORMAT parquet, ROW_GROUP_SIZE {GROUP_ROWS})")
        print(f"{width}: {(out / 'wide.parquet').stat().st_size / 1e6:.1f} MB")


def connect(engine, threads, file):
    """A function running SQL over `file` as `wide` with `engine`, on
    `threads` or its default, returning Arrow."""
    if engine == "duckdb":
        import duckdb
        con = duckdb.connect()
        if threads:
            con.execute(f"SET threads={threads}")
        con.execute(f"CREATE VIEW wide AS SELECT * FROM read_parquet('{file}')")
        fetch = "to_arrow_table" if hasattr(duckdb.DuckDBPyConnection, "to_arrow_table") else "fetch_arrow_table"
        return lambda sql: getattr(con.execute(sql), fetch)()
    if engine == "clickhouse":
        from chdb import session
        s = session.Session()
        settings = ""
        if threads:
            settings = (f" SETTINGS max_threads={threads}, max_parsing_threads={threads},"
                        f" max_download_threads={threads}")
        table = f"file('{file}', Parquet)"
        return lambda sql: s.query(sql.replace("FROM wide", f"FROM {table}") + settings, "ArrowTable")
    if engine == "datafusion":
        import datafusion
        config = datafusion.SessionConfig()
        if threads:
            config = config.with_target_partitions(threads)
        ctx = datafusion.SessionContext(config)
        ctx.register_parquet("wide", str(file))
        return lambda sql: ctx.sql(sql).collect()
    if engine == "polars":
        # Read when polars is imported, so set before.
        if threads:
            os.environ["POLARS_MAX_THREADS"] = str(threads)
        import polars
        ctx = polars.SQLContext(wide=polars.scan_parquet(file))
        return lambda sql: ctx.execute(sql).collect()
    raise ValueError(engine)


def once(engine, threads, file, width):
    """Runs every query once with `engine`, in this process, printing each
    one's ms or "failed"."""
    run = connect(engine, threads, file)
    # Starts the engine, as pipitdb's process start isn't timed either, but
    # reads nothing of the file.
    run("SELECT 1")
    for name, _, sql in queries(width):
        try:
            start = time.perf_counter()
            run(sql)
            print(f"{name}\t{(time.perf_counter() - start) * 1000:.1f}", flush=True)
        except Exception as error:
            print(f"{name}\tfailed", flush=True)
            print(f"{engine} {name}: {str(error)[:200]}", file=sys.stderr)


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


def times(engine, threads, file, width, runs):
    """Each query's best time with `engine` on `threads`, of `runs` fresh
    processes."""
    best = {}
    for _ in range(runs):
        if engine == "pipitdb":
            found = {name: pipitdb_ms(file, name, pipesql) for name, pipesql, _ in queries(width)}
        else:
            out = subprocess.run([sys.executable, __file__, "once", str(file), "--widths", str(width),
                                  "--engines", f"{engine}:{threads}"],
                                 capture_output=True, text=True).stdout
            found = {name: None if ms == "failed" else float(ms)
                     for name, ms in (line.split("\t") for line in out.splitlines())}
        for name, ms in found.items():
            if ms is not None:
                best[name] = min(ms, best.get(name, float("inf")))
    return best


def run(directory, widths, engines, runs):
    # pipitdb runs on one thread, so it has one column.
    columns = [(engine, threads) for engine in engines
               for threads in ([1] if engine == "pipitdb" else [1, 0])]
    print("width\tquery\t" + "\t".join(f"{e}-{'1' if t else 'N'}" for e, t in columns))
    for width in widths:
        file = (pathlib.Path(directory) / str(width) / "wide.parquet").resolve()
        results = [times(engine, threads, file, width, runs) for engine, threads in columns]
        for name, _, _ in queries(width):
            cells = [f"{r[name]:.1f}" if name in r else "failed" for r in results]
            print(f"{width}\t{name}\t" + "\t".join(cells), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "run", "once"])
    parser.add_argument("directory")
    parser.add_argument("--widths", default="100,1000,10000")
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    widths = [int(w) for w in args.widths.split(",")]
    if args.command == "generate":
        generate(args.directory, widths)
    elif args.command == "once":
        engine, threads = args.engines.split(":")
        once(engine, int(threads), args.directory, widths[0])
    else:
        run(args.directory, widths, args.engines.split(","), args.runs)


if __name__ == "__main__":
    main()
