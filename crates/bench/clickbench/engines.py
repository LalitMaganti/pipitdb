#!/usr/bin/env python3
"""Times other engines on every query in queries.tsv, over ClickBench's
Parquet files, one thread and as many as the engine likes, and writes each
to results/MACHINE/ENGINE-THREADS.tsv, with the machine, version, date and
load, so we know where pipitdb stands without running them again. Times
from one machine only compare with times from the same one.

Each engine runs in-process and is timed as pipitdb is: each of RUNS runs
is a fresh process that runs every query once, with no warm-up, and each
query's best is kept, in ms. "1" is one thread, ClickHouse's parsing and
download threads limited too; "N" is the engine's default.

Pivot is a server, so its times include sending the rows to this process,
which in-process engines never pay. pivot-cold clears Pivot's caches of
compressed and decompressed pages before each timed run, as its own
benchmark's cold runs do, so it reads the files as every other engine does:
only cold runs are compared. The OS page cache stays warm for every engine.

Needs `pip install duckdb chdb datafusion polars psycopg[binary]`, and Pivot
from `curl https://pivotlake.io | sh`, in $PIVOT_HOME or ~/.pivot.

Usage: crates/bench/clickbench/engines.py DIR --machine NAME
       [--engines duckdb,...] [--threads 1,N] [--runs 3] [--quiet LOAD]
DIR holds hits_*.parquet. --quiet waits until the machine's load average
stays below LOAD before starting.
"""

import argparse
import atexit
import datetime
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).parent
ENGINES = ["duckdb", "clickhouse", "datafusion", "polars", "pivot-cold"]


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


def results(name):
    """The directory of a machine's results."""
    path = HERE / "results" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def wait_quiet(limit, timeout_s=4 * 3600):
    """Waits until the one-minute load average is below `limit` three
    checks in a row, 20 s apart, so other work doesn't skew the times.
    Fails after `timeout_s`."""
    start, calm = time.monotonic(), 0
    while calm < 3:
        if time.monotonic() - start > timeout_s:
            sys.exit(f"the load stayed above {limit} for {timeout_s // 3600} hours")
        calm = calm + 1 if os.getloadavg()[0] < limit else 0
        if calm < 3:
            time.sleep(20)


def load():
    return os.getloadavg()[0]


def connect(engine, threads, directory):
    """A function running SQL with `engine`, and the engine's version."""
    files = f"{directory}/hits_*.parquet"
    if engine == "duckdb":
        import duckdb
        con = duckdb.connect()
        if threads:
            con.execute(f"SET threads={threads}")
        con.execute(f"CREATE VIEW hits AS SELECT * FROM read_parquet('{files}', binary_as_string=true)")
        # DuckDB 2 renamed fetch_arrow_table to to_arrow_table.
        fetch = "to_arrow_table" if hasattr(duckdb.DuckDBPyConnection, "to_arrow_table") else "fetch_arrow_table"
        return (lambda sql: getattr(con.execute(sql), fetch)()), duckdb.__version__
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
    if engine == "pivot-cold":
        return connect_pivot(threads, directory)
    raise ValueError(engine)


def connect_pivot(threads, directory):
    """Pivot is a server: one is started for this process, with `threads`
    workers, its table over the Parquet files where they are, as Pivot's own
    ClickBench setup has it, and stopped when the process exits. A query
    that runs out of memory fails alone: the server is capped where systemd
    can, and started again for the next."""
    import psycopg
    binary = pathlib.Path(os.environ.get("PIVOT_HOME", pathlib.Path.home() / ".pivot")) / "cli/latest/pivot"
    work = pathlib.Path(tempfile.mkdtemp(prefix="pivot-"))
    port = 54329
    # Pivot's buffer pool takes most of free memory unless told otherwise.
    config = (f"memory: 16g\nserver:\n  bind: 127.0.0.1:{port}\n"
              f"datastores:\n  default:\n    kind: pivot\n    location: {work}/catalog\n    default: true\n")
    if threads:
        config = f"workers: {threads}\n" + config
    (work / "config.yaml").write_text(config)
    schema = (HERE / "pivot.sql").read_text().replace("{source}", str(pathlib.Path(directory).resolve()))
    schema = "\n".join(line for line in schema.splitlines() if not line.startswith("--"))
    # Pivot prepares each statement on its own.
    statements = [s for s in schema.split(";") if s.strip()]
    cap = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=24G"] if shutil.which("systemd-run") else []
    state = {}

    def start():
        log = open(work / "server.log", "a")
        server = subprocess.Popen(cap + [binary, "server", "--config", work / "config.yaml"], stdout=log, stderr=log)
        atexit.register(server.wait)
        atexit.register(server.terminate)
        for _ in range(300):
            try:
                con = psycopg.connect(f"host=127.0.0.1 port={port} user=pivot dbname=postgres", autocommit=True)
                break
            except psycopg.OperationalError:
                time.sleep(0.1)
        else:
            raise RuntimeError(f"Pivot didn't start: {(work / 'server.log').read_text()[-500:]}")
        for statement in statements:
            con.execute(statement)
        state["server"], state["con"] = server, con

    start()
    version = subprocess.run([binary, "--version"], capture_output=True, text=True).stdout.strip()

    def run(sql):
        if state["server"].poll() is not None:
            start()
        # Rows left in libpq: decoding them in Python would cost more than
        # Pivot does. Pivot doesn't send binary rows yet.
        result = state["con"].pgconn.exec_params(sql.encode(), [])
        if result.status != psycopg.pq.ExecStatus.TUPLES_OK:
            raise RuntimeError(result.error_message.decode())
    return run, version.removeprefix("pivot ")


def run_once(engine, threads, directory):
    """Runs every query once with one engine, in this process, printing the
    engine's version, then each query's ms or "failed"."""
    run, version = connect(engine, threads, directory)
    print(f"version\t{version}", flush=True)
    for query, sql in queries():
        try:
            if engine == "pivot-cold":
                run("SELECT drop_cache()")
            start = time.perf_counter()
            run(sql)
            print(f"{query}\t{(time.perf_counter() - start) * 1000:.1f}", flush=True)
        except Exception as error:
            print(f"{query}\tfailed", flush=True)
            print(f"{engine} {query}: {str(error)[:200]}", file=sys.stderr)


def run_one(engine, threads, runs, directory, name):
    """Times every query with one engine, best of `runs` fresh processes,
    and writes its file."""
    before = load()
    best, version = {}, ""
    for _ in range(runs):
        out = subprocess.run([sys.executable, __file__, directory, "--machine", name,
                              "--once", f"{engine}:{threads}"],
                             capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            query, value = line.split("\t")
            if query == "version":
                version = value
            elif value != "failed":
                best[query] = min(float(value), best.get(query, float("inf")))
    count = len(list(pathlib.Path(directory).glob("hits_*.parquet")))
    label = "1" if threads == 1 else "N"
    lines = [
        f"# {engine} {version}, {'one thread' if threads == 1 else 'default threads'},"
        f" best of {runs} fresh processes, no warm-up, ms",
        f"# {machine()}, {datetime.date.today()}, {count} files",
        f"# load {before:.2f} before, {load():.2f} after",
    ]
    rows = [f"{query}\t{best[query]:.1f}" if query in best else f"{query}\tfailed"
            for query, _ in queries()]
    print(f"{engine}-{label}: {sum(q in best for q, _ in queries())} queries", file=sys.stderr)
    (results(name) / f"{engine}-{label}.tsv").write_text("\n".join(lines + rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--threads", default="1,N")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--quiet", type=float)
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--once", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.once:
        engine, threads = args.once.split(":")
        return run_once(engine, int(threads), args.directory)
    if args.one:
        engine, threads = args.one.split(":")
        return run_one(engine, int(threads), args.runs, args.directory, args.machine)
    if args.quiet:
        wait_quiet(args.quiet)
    # Each in its own process, so one engine's threads and memory don't
    # affect the next, and Polars reads its thread count fresh.
    for engine in args.engines.split(","):
        for threads in args.threads.split(","):
            one = f"{engine}:{1 if threads == '1' else 0}"
            subprocess.run([sys.executable, __file__, args.directory, "--runs", str(args.runs),
                            "--machine", args.machine, "--one", one], check=True)


if __name__ == "__main__":
    main()
