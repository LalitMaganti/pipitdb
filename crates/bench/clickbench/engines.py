#!/usr/bin/env python3
"""Times other engines on every query in queries.tsv, over ClickBench's
Parquet files, one thread and as many as the engine likes, and writes each
to results/ENGINE-THREADS.tsv, with the machine, version and date, so we
know where pipitdb stands without running them again.

Each engine runs in-process, best of RUNS after a warm-up, in ms. "1" is
one thread, ClickHouse's parsing and download threads limited too; "N" is
the engine's default.

Needs `pip install duckdb chdb datafusion polars`.

Usage: crates/bench/clickbench/engines.py DIR [--engines duckdb,...]
       [--threads 1,N] [--runs 2]
DIR holds hits_*.parquet.
"""

import argparse
import datetime
import os
import pathlib
import platform
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
ENGINES = ["duckdb", "clickhouse", "datafusion", "polars"]


def queries():
    """Each query's name and SQL."""
    for line in (HERE / "queries.tsv").read_text().splitlines():
        name, _, sql = line.split("\t")
        yield name, sql


def machine():
    """The CPU and its cores, as one line."""
    if platform.system() == "Darwin":
        sysctl = lambda key: subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        return f"{sysctl('machdep.cpu.brand_string')}, {sysctl('hw.ncpu')} cores, macOS {platform.mac_ver()[0]}"
    return f"{platform.processor() or platform.machine()}, {os.cpu_count()} cores, {platform.platform()}"


def connect(engine, threads, directory):
    """A function running SQL with `engine`, and the engine's version."""
    files = f"{directory}/hits_*.parquet"
    if engine == "duckdb":
        import duckdb
        con = duckdb.connect()
        if threads:
            con.execute(f"SET threads={threads}")
        con.execute(f"CREATE VIEW hits AS SELECT * FROM read_parquet('{files}', binary_as_string=true)")
        return (lambda sql: con.execute(sql).fetch_arrow_table()), duckdb.__version__
    if engine == "clickhouse":
        import chdb
        from chdb import session
        s = session.Session()
        settings = ""
        if threads:
            settings = (f" SETTINGS max_threads={threads}, max_parsing_threads={threads},"
                        f" max_download_threads={threads}")
        table = f"file('{files}', Parquet)"
        return (lambda sql: s.query(sql.replace("FROM hits", f"FROM {table}") + settings,
                                    "ArrowTable")), chdb.__version__
    if engine == "datafusion":
        import datafusion
        config = (datafusion.SessionConfig()
                  .set("datafusion.sql_parser.enable_ident_normalization", "false")
                  .set("datafusion.execution.parquet.binary_as_string", "true"))
        if threads:
            config = config.with_target_partitions(threads)
        ctx = datafusion.SessionContext(config)
        ctx.register_parquet("hits", f"{directory}/")
        return (lambda sql: ctx.sql(sql).collect()), datafusion.__version__
    if engine == "polars":
        # Read when polars is imported, so set before.
        if threads:
            os.environ["POLARS_MAX_THREADS"] = str(threads)
        import polars
        ctx = polars.SQLContext(hits=polars.scan_parquet(files))
        return (lambda sql: ctx.execute(sql).collect()), polars.__version__
    raise ValueError(engine)


def run_one(engine, threads, runs, directory):
    """Times every query with one engine, in this process, and writes its file."""
    run, version = connect(engine, threads, directory)
    count = len(list(pathlib.Path(directory).glob("hits_*.parquet")))
    label = "1" if threads == 1 else "N"
    lines = [
        f"# {engine} {version}, {'one thread' if threads == 1 else 'default threads'},"
        f" best of {runs} after a warm-up, ms",
        f"# {machine()}, {datetime.date.today()}, {count} files",
    ]
    for name, sql in queries():
        try:
            run(sql)
            best = float("inf")
            for _ in range(runs):
                start = time.perf_counter()
                run(sql)
                best = min(best, time.perf_counter() - start)
            lines.append(f"{name}\t{best * 1000:.1f}")
        except Exception as error:
            lines.append(f"{name}\tfailed")
            print(f"{engine}-{label} {name}: {str(error)[:200]}", file=sys.stderr)
        print(f"{engine}-{label} {lines[-1]}", file=sys.stderr, flush=True)
    (HERE / "results" / f"{engine}-{label}.tsv").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--threads", default="1,N")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        engine, threads = args.one.split(":")
        return run_one(engine, int(threads), args.runs, args.directory)
    # Each in its own process, so one engine's threads and memory don't
    # affect the next, and Polars reads its thread count fresh.
    for engine in args.engines.split(","):
        for threads in args.threads.split(","):
            one = f"{engine}:{1 if threads == '1' else 0}"
            subprocess.run([sys.executable, __file__, args.directory, "--runs", str(args.runs),
                            "--one", one], check=True)


if __name__ == "__main__":
    main()
