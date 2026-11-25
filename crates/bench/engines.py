#!/usr/bin/env python3
"""Times other engines on every query of a suite, one thread and as many as
the engine likes, checks each answer against DuckDB's, and writes each to
results/MACHINE/SUITE/ENGINE-THREADS.tsv, with the machine, version, date and
load, so we know where pipitdb stands without running them again. Times from
one machine only compare with times from the same one.

Suites are ours over ClickBench's data, `clickbench`, in
clickbench/queries.tsv; ClickBench's own 43 queries, `clickbench-official`,
in clickbench/official.tsv; and TPC-H and TPC-DS, `tpch` and `tpcds`, in
tpc/, whose files tpc/generate.py makes.

Each engine is timed as pipitdb is: each of RUNS runs is a fresh process
that runs every query once, with no warm-up, as pipitdb has none, and each
query's best is kept, in ms. A time ends when the engine has its result in
its own form, such as Arrow; turning that into Python's values isn't timed.
A query that takes longer than --timeout seconds, or crashes the engine, is
marked so, and the rest run in a fresh process. "1" is one thread,
ClickHouse's parsing and download threads limited too; "N" is the engine's
default.

Each line of a results file is a query and its ms, or why it has none:
`failed` if the engine gave an error or crashed, `wrong` if its rows aren't
DuckDB's, `timeout` if it ran too long. The first run's rows are checked. Rows match DuckDB's as check_queries.py has it,
in any order, and TPC's within its rules for checking answers; where a
LIMIT cuts through rows that tie on its ORDER BY keys, any of them will do. DuckDB's own answers, on as many threads as it likes,
are what's checked against, and are kept in ~/.cache/pipitdb/answers so
they're found once for each query's SQL.

Pivot is a server, so its times include sending the rows to this process,
which in-process engines never pay. pivot-cold clears Pivot's caches of
compressed and decompressed pages before each timed run, as its own
benchmark's cold runs do, so it reads the files as every other engine does:
only cold runs are compared. The OS page cache stays warm for every engine.
Pivot runs only the `clickbench` suite, whose table pivot.sql makes.

Needs `pip install duckdb chdb datafusion polars psycopg[binary]`, and Pivot
from `curl https://pivotlake.io | sh`, in $PIVOT_HOME or ~/.pivot.

Usage: crates/bench/engines.py DIR --suite SUITE --machine NAME
       [--engines duckdb,...] [--threads 1,N] [--runs 3] [--timeout 300]
       [--quiet LOAD]
DIR holds the suite's Parquet files: hits_*.parquet for ClickBench, a file a
table for TPC. --quiet waits until the machine's load average stays below
LOAD before starting.
"""

import argparse
import atexit
import collections
import datetime
import hashlib
import json
import os
import pathlib
import platform
import re
import selectors
import shutil
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from check_queries import QUERIES, order, same  # noqa: E402

ENGINES = ["duckdb", "clickhouse", "datafusion", "polars", "pivot-cold"]
ANSWERS = pathlib.Path.home() / ".cache/pipitdb/answers"


def queries(suite):
    """Each query's name and SQL."""
    for line in QUERIES[suite].read_text().splitlines():
        name, _, sql = line.split("\t")
        yield name, sql


def tables(directory):
    """Each table's Parquet files, by name: a file's name less a trailing
    `_N`, so hits_0.parquet and hits_1.parquet are both `hits`."""
    found = collections.defaultdict(list)
    for path in sorted(pathlib.Path(directory).glob("*.parquet")):
        found[re.sub(r"_\d+$", "", path.stem)].append(str(path.resolve()))
    return found


def machine():
    """The CPU and its cores, as one line."""
    if platform.system() == "Darwin":
        sysctl = lambda key: subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        return f"{sysctl('machdep.cpu.brand_string')}, {sysctl('hw.ncpu')} cores, macOS {platform.mac_ver()[0]}"
    return f"{platform.processor() or platform.machine()}, {os.cpu_count()} cores, {platform.platform()}"


def results(name, suite):
    """The directory of a machine's results for a suite."""
    path = HERE / "results" / name / suite
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


# ClickBench's own queries are in DuckDB's SQL; each engine's names for its
# functions, as in ClickBench's queries for that engine. STRLEN counts bytes.
DIALECTS = {
    "clickhouse": [(r"\bSTRLEN\(", "length(")],
    "datafusion": [(r"\bSTRLEN\(", "octet_length("), (r"\btoDateTime\(", "to_timestamp_seconds(")],
    "polars": [(r"\bSTRLEN\(", "octet_length("),
               (r"\btoDateTime\((\w+)\)", r"CAST(\1 * 1000000 AS TIMESTAMP)")],
}


def dialect(engine, suite, sql):
    """`sql` in `engine`'s names for functions."""
    if suite == "clickbench-official":
        for pattern, replacement in DIALECTS.get(engine, []):
            sql = re.sub(pattern, replacement, sql)
    return sql


def connect(engine, threads, suite, directory):
    """A function running SQL with `engine` and returning its rows, as lists
    of Python values, and the engine's version. ClickBench's own queries read
    EventDate as a date, as its DuckDB setup does, and call ClickHouse's
    toDateTime; ours read the day number."""
    found = tables(directory)
    official = suite == "clickbench-official"
    if engine == "duckdb":
        import duckdb
        con = duckdb.connect()
        if threads:
            con.execute(f"SET threads={threads}")
        select = "* REPLACE (make_date(EventDate) AS EventDate)" if official else "*"
        for name, paths in found.items():
            files = ", ".join(f"'{p}'" for p in paths)
            con.execute(f"CREATE VIEW {name} AS SELECT {select} FROM read_parquet([{files}], binary_as_string=true)")
        if official:
            con.execute("CREATE MACRO toDateTime(t) AS epoch_ms(t * 1000)")
        # A relation's Arrow output saves the copy a cursor's makes.
        return (lambda sql: con.sql(sql).to_arrow_table()), duckdb.__version__
    if engine == "clickhouse":
        import chdb
        from chdb import session
        s = session.Session()
        s.query("CREATE DATABASE IF NOT EXISTS bench ENGINE = Atomic")
        s.query("USE bench")
        select = "* REPLACE (toDate(EventDate) AS EventDate)" if official else "*"
        for name, paths in found.items():
            pattern = paths[0] if len(paths) == 1 else re.sub(r"_\d+\.parquet$", "_*.parquet", paths[0])
            s.query(f"CREATE VIEW {name} AS SELECT {select} FROM file('{pattern}', Parquet)")
        # Standard SQL's nulls for an outer join's missing side and ROLLUP's
        # missing keys.
        settings = " SETTINGS join_use_nulls = 1, group_by_use_nulls = 1"
        if threads:
            settings += (f", max_threads={threads}, max_parsing_threads={threads},"
                         f" max_download_threads={threads}")
        return (lambda sql: s.query(sql + settings, "ArrowTable")), chdb.__version__
    if engine == "datafusion":
        import datafusion
        config = (datafusion.SessionConfig()
                  .set("datafusion.sql_parser.enable_ident_normalization", "false")
                  .set("datafusion.execution.parquet.binary_as_string", "true"))
        if threads:
            config = config.with_target_partitions(threads)
        ctx = datafusion.SessionContext(config)
        for name, paths in found.items():
            source = str(pathlib.Path(paths[0]).parent) + "/" if len(paths) > 1 else paths[0]
            ctx.register_parquet(f"{name}_files" if official else name, source)
            if official:
                ctx.sql(f"CREATE VIEW {name} AS SELECT * REPLACE "
                        f"(arrow_cast(arrow_cast(EventDate, 'Int32'), 'Date32') AS EventDate) "
                        f"FROM {name}_files").collect()
        return (lambda sql: ctx.sql(sql).collect()), datafusion.__version__
    if engine == "polars":
        # Read when polars is imported, so set before.
        if threads:
            os.environ["POLARS_MAX_THREADS"] = str(threads)
        import polars
        frames = {}
        for name, paths in found.items():
            # ClickBench's strings are stored as binary.
            frame = polars.scan_parquet(paths).with_columns(polars.col(polars.Binary).cast(polars.String))
            if official:
                frame = frame.with_columns(polars.col("EventDate").cast(polars.Int32).cast(polars.Date))
            frames[name] = frame
        ctx = polars.SQLContext(frames)
        return (lambda sql: ctx.execute(sql).collect()), polars.__version__
    if engine == "pivot-cold":
        if suite != "clickbench":
            raise ValueError("Pivot runs only the clickbench suite")
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
    schema = (HERE / "clickbench/pivot.sql").read_text().replace("{source}", str(pathlib.Path(directory).resolve()))
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
        # Pivot does. Pivot doesn't send binary rows yet, so its rows aren't
        # checked.
        result = state["con"].pgconn.exec_params(sql.encode(), [])
        if result.status != psycopg.pq.ExecStatus.TUPLES_OK:
            raise RuntimeError(result.error_message.decode())
        return None
    return run, version.removeprefix("pivot ")


def python_rows(result):
    """An engine's result as rows of Python values: an Arrow table, a list
    of Arrow batches, or a Polars frame."""
    if hasattr(result, "rows"):
        return [list(row) for row in result.rows()]
    batches = result if isinstance(result, list) else [result]
    rows = []
    for batch in batches:
        rows.extend(map(list, zip(*(column.to_pylist() for column in batch.columns))))
    return rows


def text(value):
    """A value as check_queries.py compares it: as DuckDB's CLI prints it."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def as_text(rows):
    return [[text(value) for value in row] for row in rows]


def normal(value):
    """A value's text as `same` compares it: dates and times as pipitdb's
    numbers, and floats to 9 significant digits."""
    if len(value) in (10, 19) and value[4:5] == "-" and value[7:8] == "-":
        try:
            day = datetime.datetime.fromisoformat(value).replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            return value
        return str(day.date().toordinal() - EPOCH) if len(value) == 10 else str(int(day.timestamp()))
    if "." in value or "e" in value:
        try:
            return format(float(value), ".9g")
        except ValueError:
            return value
    return value


EPOCH = datetime.date(1970, 1, 1).toordinal()


def contained(got, cursor):
    """Whether each of rows `got` is one of those `cursor` gives, each used
    once. They're read in chunks, so an answer of millions of rows is never
    held whole."""
    need = collections.Counter(tuple(map(normal, row)) for row in got)
    while need and (chunk := cursor.fetchmany(100_000)):
        for row in chunk:
            key = tuple(normal(text(value)) for value in row)
            if need.get(key):
                need[key] -= 1
                if not need[key]:
                    del need[key]
    return not need


def close(got, want):
    """Whether `got` is `want` within TPC's rules for checking answers: a
    ratio or average within 1% once rounded to cents, and a sum within $100.
    Columns' kinds aren't known, so both bound each value. Engines divide
    decimals keeping different digits. Integers, such as counts, must be
    exact."""
    if same(got, want):
        return True
    if "." not in got and "." not in want:
        return False
    try:
        r, v = float(got), float(want)
    except ValueError:
        return False
    return abs(round(r, 2) - v) <= min(100, 0.01 * abs(v))


def matches(got, want, equal=same):
    """Whether rows `got` are `want`'s, in any order, their values `equal`."""
    got, want = sorted(got, key=order), sorted(want, key=order)
    return len(got) == len(want) and all(
        len(g) == len(w) and all(map(equal, g, w)) for g, w in zip(got, want))


LIMIT = re.compile(r"\s+LIMIT\s+(\d+)(?:\s+OFFSET\s+(\d+))?\s*;?\s*$", re.I)
ORDER = re.compile(r"\bORDER\s+BY\s+(.*)$", re.I | re.S)


def ties_only(got, want, sql, reference):
    """Whether rows `got` differ from `want` only where a LIMIT cuts through
    rows that tie on the ORDER BY keys: each is a row of the answer without
    the LIMIT, there are as many, and where the keys are columns of the
    answer, they take the same values. `reference` runs SQL with DuckDB,
    giving a cursor."""
    limit = LIMIT.search(sql)
    if not limit or len(got) != len(want):
        return False
    base = sql[:limit.start()]
    cursor = reference(base)
    names = [d[0].lower() for d in cursor.description]
    if not contained(got, cursor):
        return False
    keys = ORDER.search(base)
    if not keys:
        # Any rows will do.
        return True
    positions = []
    for key in keys.group(1).split(","):
        key = re.sub(r"\s+(ASC|DESC)\b.*$", "", key.strip(), flags=re.I).strip().lower()
        if key not in names:
            # A key that isn't a column: its values can't be compared.
            return True
        positions.append(names.index(key))
    picked = lambda rows: sorted(([row[p] for p in positions] for row in rows), key=order)
    return all(map(lambda g, w: all(map(same, g, w)), picked(got), picked(want)))


def reference_runner(suite, directory):
    """Runs SQL with DuckDB on as many threads as it likes, returning a
    cursor over its rows."""
    import duckdb
    con = duckdb.connect()
    select = "* REPLACE (make_date(EventDate) AS EventDate)" if suite == "clickbench-official" else "*"
    for name, paths in tables(directory).items():
        files = ", ".join(f"'{p}'" for p in paths)
        con.execute(f"CREATE VIEW {name} AS SELECT {select} FROM read_parquet([{files}], binary_as_string=true)")
    if suite == "clickbench-official":
        con.execute("CREATE MACRO toDateTime(t) AS epoch_ms(t * 1000)")

    return lambda sql: con.cursor().execute(sql)


def answer(suite, directory, query, sql, reference):
    """DuckDB's rows for `query`, as text, found once for its SQL and kept."""
    path = ANSWERS / suite / pathlib.Path(directory).resolve().as_posix().strip("/").replace("/", "_") / f"{query}-{hashlib.sha1(sql.encode()).hexdigest()[:12]}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = as_text(reference(sql).fetchall())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def check(suite, directory, query, sql, got, reference):
    """`ok` if rows `got`, as text, are DuckDB's answer to `query`, else
    `wrong`."""
    want = answer(suite, directory, query, sql, reference)
    equal = close if suite in ("tpch", "tpcds") else same
    if matches(got, want, equal) or ties_only(got, want, sql, reference):
        return "ok"
    return "wrong"


def run_once(engine, threads, suite, directory, start, out):
    """Runs every query from the `start`th once with one engine, in this
    process. Prints the engine's version, then each query's ms, and writes
    each query's rows as text to `out`, a line each, if given."""
    run, version = connect(engine, threads, suite, directory)
    print(f"version\t{version}", flush=True)
    rows = open(out, "w") if out else None
    for query, sql in list(queries(suite))[start:]:
        sql = dialect(engine, suite, sql)
        print(f"start\t{query}", flush=True)
        if engine == "pivot-cold":
            run("SELECT drop_cache()")
        begin = time.perf_counter()
        try:
            got = run(sql)
        except Exception as error:
            print(f"failed\t{query}", flush=True)
            print(f"{engine} {query}: {str(error)[:300]}", file=sys.stderr, flush=True)
            continue
        ms = (time.perf_counter() - begin) * 1000
        if rows and got is not None:
            rows.write(json.dumps([query, as_text(python_rows(got))]) + "\n")
            rows.flush()
        print(f"ms\t{query}\t{ms:.1f}", flush=True)


def run_process(engine, threads, args, out):
    """One run of every query with one engine: each query's ms or status.
    A query that crashes the engine or runs longer than `args.timeout`
    seconds is marked so, and the rest run in a fresh process."""
    names = [query for query, _ in queries(args.suite)]
    found, version, start = {}, "", 0
    while start < len(names):
        # Each process writes its rows to its own file.
        rows = f"{out}.{start:04}" if out else ""
        command = [sys.executable, __file__, args.directory, "--suite", args.suite,
                   "--machine", args.machine, "--once", f"{engine}:{threads}:{start}:{rows}"]
        child = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        current, deadline = None, time.monotonic() + args.timeout
        while True:
            if not selector.select(max(0.0, deadline - time.monotonic())):
                child.kill()
                found[current or names[start]] = "timeout"
                break
            line = child.stdout.readline()
            if not line:
                if current is not None:
                    found[current] = "failed"
                break
            kind, *rest = line.rstrip("\n").split("\t")
            if kind == "version":
                version = rest[0]
            elif kind == "start":
                current, deadline = rest[0], time.monotonic() + args.timeout
            else:
                found[rest[0]] = float(rest[1]) if kind == "ms" else kind
                current = None
        child.wait()
        selector.close()
        start = max([names.index(q) + 1 for q in found] + [start + 1])
    return found, version


def rows_of(out):
    """The rows a run wrote, by query."""
    rows = {}
    for path in sorted(pathlib.Path(out).parent.glob(pathlib.Path(out).name + ".*")):
        for line in path.read_text().splitlines():
            query, got = json.loads(line)
            rows[query] = got
        path.unlink()
    return rows


def run_one(engine, threads, args):
    """Times every query with one engine, best of `args.runs` fresh
    processes, checks the rows of the first, and writes its file."""
    before = load()
    reference = reference_runner(args.suite, args.directory)
    out = tempfile.mktemp(prefix="rows-", dir=tempfile.gettempdir())
    best, statuses, version = {}, {}, ""
    for attempt in range(args.runs):
        found, version = run_process(engine, threads, args, out if attempt == 0 else None)
        if attempt == 0:
            rows = rows_of(out)
            for query, sql in queries(args.suite):
                if isinstance(found.get(query), float) and engine != "pivot-cold":
                    try:
                        status = check(args.suite, args.directory, query, sql, rows.get(query, []), reference)
                    except Exception as error:
                        print(f"{engine} {query}: checking: {str(error)[:200]}", file=sys.stderr)
                        status = "unchecked"
                    if status != "ok":
                        found[query] = status
                statuses[query] = found.get(query, "failed")
        for query, value in found.items():
            if isinstance(value, float) and isinstance(statuses.get(query), float):
                statuses[query] = min(statuses[query], value)
    count = sum(len(paths) for paths in tables(args.directory).values())
    label = "1" if threads == 1 else "N"
    lines = [
        f"# {engine} {version}, {'one thread' if threads == 1 else 'default threads'},"
        f" best of {args.runs} fresh processes, no warm-up, ms",
        f"# {machine()}, {datetime.date.today()}, {count} files",
        f"# load {before:.2f} before, {load():.2f} after",
    ]
    rows = [f"{query}\t{statuses[query]:.1f}" if isinstance(statuses[query], float)
            else f"{query}\t{statuses[query]}" for query, _ in queries(args.suite)]
    ok = sum(isinstance(s, float) for s in statuses.values())
    print(f"{args.suite} {engine}-{label}: {ok} of {len(statuses)} queries", file=sys.stderr)
    (results(args.machine, args.suite) / f"{engine}-{label}.tsv").write_text("\n".join(lines + rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--suite", required=True, choices=list(QUERIES))
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--threads", default="1,N")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--quiet", type=float)
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--once", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.once:
        engine, threads, start, out = args.once.split(":", 3)
        return run_once(engine, int(threads), args.suite, args.directory, int(start), out)
    if args.one:
        engine, threads = args.one.split(":")
        return run_one(engine, int(threads), args)
    if args.quiet:
        wait_quiet(args.quiet)
    # Each in its own process, so one engine's threads and memory don't
    # affect the next, and Polars reads its thread count fresh.
    engines = [e for e in args.engines.split(",") if e != "pivot-cold" or args.suite == "clickbench"]
    for engine in engines:
        for threads in args.threads.split(","):
            one = f"{engine}:{1 if threads == '1' else 0}"
            subprocess.run([sys.executable, __file__, args.directory, "--suite", args.suite,
                            "--runs", str(args.runs), "--timeout", str(args.timeout),
                            "--machine", args.machine, "--one", one], check=True)


if __name__ == "__main__":
    main()
