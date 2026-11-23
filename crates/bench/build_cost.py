#!/usr/bin/env python3
"""Measures what a hash join's build costs per row against its probe, on this
machine: the ratio the join planner's `BUILD_COST` stands for.

It makes a table of 1M rows to build and one of 8M to probe, joined on a
unique key and on a key each row shares with about 3 others, with DuckDB.
Filters keep 10%, 40% or all of each side. Each join's time, less its two
scans', is fitted to its rows built, probed and made.

Usage: crates/bench/build_cost.py [DIR]
DIR keeps the tables between runs; a temporary one by default. Needs the
DuckDB CLI on PATH, and builds the `suite` example with `--profile speed`.
"""

import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent.parent.parent
SHARES = (10, 40, 100)
RUNS = 5

TABLES = {
    "build_unique": "SELECT (i * 2654435761) % 16777216 AS bk, i % 100 AS bf, i AS bv"
    " FROM range(1048576) t(i) ORDER BY hash(i)",
    "build_dups": "SELECT (hash(i) % 262144)::BIGINT * 64 AS bk, i % 100 AS bf, i AS bv"
    " FROM range(1048576) t(i) ORDER BY hash(i + 1)",
    "probe_unique": "SELECT (((hash(j) % 1048576)::BIGINT * 2654435761) % 16777216)::BIGINT"
    " AS pk, (hash(j + 7) % 100)::BIGINT AS pf, j AS pv FROM range(8388608) t(j)",
    "probe_dups": "SELECT (hash(j) % 262144)::BIGINT * 64 AS pk,"
    " (hash(j + 7) % 100)::BIGINT AS pf, j AS pv FROM range(8388608) t(j)",
}


def duckdb(sql):
    command = ["duckdb", "-csv", "-noheader", "-c", sql]
    run = subprocess.run(command, capture_output=True, text=True)
    if run.returncode != 0:
        sys.exit(run.stderr)
    return run.stdout


def queries():
    """Each query's name and PipeSQL: each side's scan, and each join."""
    for kind in ("unique", "dups"):
        for share in SHARES:
            for side, column in (("build", "b"), ("probe", "p")):
                yield f"{column.upper()}{share}_{kind}", (
                    f"FROM {side}_{kind} |> WHERE {column}f < {share}"
                    f" |> AGGREGATE count(*), sum({column}v)"
                )
        for b in SHARES:
            for p in SHARES:
                yield f"J{b}_{p}_{kind}", (
                    f"FROM probe_{kind} |> WHERE pf < {p} |> JOIN build_{kind} ON pk = bk"
                    f" |> WHERE bf < {b} |> AGGREGATE count(*), sum(bv)"
                )


def rows(data, kind, b, p):
    """The rows a join builds, probes and makes."""
    build, probe = data / f"build_{kind}.parquet", data / f"probe_{kind}.parquet"
    out = duckdb(
        f"SELECT (SELECT count(*) FROM '{build}' WHERE bf < {b}),"
        f" (SELECT count(*) FROM '{probe}' WHERE pf < {p}),"
        f" (SELECT count(*) FROM '{probe}' JOIN '{build}' ON pk = bk WHERE bf < {b} AND pf < {p})"
    )
    return [int(n) for n in out.strip().split(",")]


def solve(a, y):
    """The least-squares solution of `a x = y`."""
    n = len(a[0])
    m = [
        [sum(r[i] * r[j] for r in a) for j in range(n)] + [sum(r[i] * v for r, v in zip(a, y))]
        for i in range(n)
    ]
    for i in range(n):
        for r in range(n):
            if r != i:
                f = m[r][i] / m[i][i]
                m[r] = [x - f * z for x, z in zip(m[r], m[i])]
    return [m[i][n] / m[i][i] for i in range(n)]


def main():
    data = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
    data.mkdir(parents=True, exist_ok=True)
    for name, sql in TABLES.items():
        if not (data / f"{name}.parquet").exists():
            duckdb(f"COPY ({sql}) TO '{data / name}.parquet' (FORMAT parquet, COMPRESSION snappy)")
    (data / "queries.tsv").write_text("".join(f"{n}\t{q}\t\n" for n, q in queries()))
    build = ["cargo", "build", "-q", "--profile", "speed", "--example", "suite"]
    subprocess.run([*build, "-p", "pipitdb-bench"], cwd=ROOT, check=True)
    suite = ROOT / "target/speed/examples/suite"
    files = sorted(str(f) for f in data.glob("*.parquet"))
    times = {}
    for _ in range(RUNS):
        command = [suite, data / "queries.tsv", *files]
        run = subprocess.run(command, env={"QUIET": "1"}, capture_output=True, text=True)
        for name, value, unit in re.findall(r"^Q(\S+): ([\d.]+)(ms|s|µs) ", run.stderr, re.M):
            ms = float(value) * {"ms": 1, "s": 1000, "µs": 0.001}[unit]
            times[name] = min(times.get(name, ms), ms)
    for kind in ("unique", "dups"):
        a, y = [], []
        for b in SHARES:
            for p in SHARES:
                a.append(rows(data, kind, b, p))
                beyond = times[f"J{b}_{p}_{kind}"] - times[f"B{b}_{kind}"] - times[f"P{p}_{kind}"]
                y.append(beyond * 1e6)
        built, probed, made = solve(a, y)
        print(
            f"{kind:6}: {built:6.1f} ns a row built, {probed:5.1f} probed, {made:5.1f} made:"
            f" building costs {built / probed:.1f} times probing"
        )


if __name__ == "__main__":
    main()
