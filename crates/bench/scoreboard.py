#!/usr/bin/env python3
"""Runs pipitdb on a suite's queries, best of RUNS, checks its rows against
DuckDB's as engines.py does, saves the times to
results/MACHINE/SUITE/pipitdb.tsv with the commit, and prints them beside
every other result saved for that machine and suite: the engines engines.py
ran, one thread and N, and spike.tsv, the prototype pipitdb grew from.

Each is scored as ClickBench scores, over the queries it got right: the
geometric mean of (t + 10) / (best + 10), where best is the fastest of
those compared that got the query right, so 1.0 is best on every query it
ran. A query an engine fails, gets wrong, or can't run isn't in its score,
so each score comes with how many queries it covers.

The machine's load changes times, so run this on an idle machine, or with
--quiet LOAD to wait until the load average stays below LOAD. Each file says
which machine it came from and the load around its run.

Usage: crates/bench/scoreboard.py DIR --suite SUITE --machine NAME
       [--runs 3] [--binary PATH] [--save NAME] [--no-run] [--quiet LOAD]
DIR holds the suite's Parquet files. --binary runs another build, such as
the spike's, and --save names its results file.
"""

import argparse
import datetime
import math
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
TIME = re.compile(r"^Q(\S+): ([\d.]+)(ns|µs|ms|s) ")
SECONDS = {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1.0}
sys.path.insert(0, str(HERE))
from check_queries import QUERIES, rows_by_query  # noqa: E402
from engines import check, load as load_average, machine, reference_runner, results, tables, wait_quiet  # noqa: E402


def queries(suite):
    """Each query's name and whether pipitdb has it."""
    for line in QUERIES[suite].read_text().splitlines():
        name, pipesql, sql = line.split("\t")
        yield name, bool(pipesql), sql


def run(binary, suite, files, runs):
    """Each query's rows from one run, and its best time in ms over `runs`
    runs of every query; a query with no time failed. If the process dies,
    say out of memory, each query after the last it reported runs in a
    process of its own."""
    best, rows = {}, {}
    names = [name for name, written, _ in queries(suite) if written]
    for attempt in range(runs):
        quiet = {"QUIET": "1"} if attempt else {}
        pending = [None]
        while pending:
            only = pending.pop(0)
            env = dict(os.environ, **quiet, **({"QUERY": only} if only else {}))
            out = subprocess.run([binary, str(QUERIES[suite]), *files], capture_output=True,
                                 text=True, env=env)
            if not attempt:
                rows.update(rows_by_query(out.stdout.splitlines()))
            reported = []
            for line in out.stderr.splitlines():
                if line.startswith("Q") and ": " in line:
                    reported.append(line[1:].split(": ")[0])
                if match := TIME.match(line):
                    name, value, unit = match.groups()
                    ms = float(value) * SECONDS[unit] * 1e3
                    best[name] = min(ms, best.get(name, ms))
            if out.returncode != 0 and only is None:
                last = max((names.index(n) for n in reported if n in names), default=-1)
                pending = names[last + 2:]
                print(f"Q{names[last + 1] if last + 1 < len(names) else '?'}: the process died",
                      file=sys.stderr)
    return best, rows


def load(path):
    """A results file: its header lines, and each query's ms or status."""
    header, times = [], {}
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            header.append(line)
            continue
        name, value = line.split("\t")
        times[name] = float(value) if re.fullmatch(r"[\d.]+", value) else value
    return header, times


def score(times, columns, names):
    """ClickBench's score for each column over the queries it got right,
    and how many those are."""
    scores = {}
    for column in columns:
        logs = []
        for name in names:
            ms = times[column].get(name)
            if not isinstance(ms, float):
                continue
            best = min(t for c in columns if isinstance(t := times[c].get(name), float))
            logs.append(math.log((ms + 10) / (best + 10)))
        if logs:
            scores[column] = (math.exp(sum(logs) / len(logs)), len(logs))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--suite", required=True, choices=list(QUERIES))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--binary")
    parser.add_argument("--save", default="pipitdb")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--machine", required=True)
    parser.add_argument("--quiet", type=float)
    args = parser.parse_args()
    directory = results(args.machine, args.suite)
    files = [path for paths in tables(args.directory).values() for path in paths]
    if not args.no_run:
        binary = args.binary
        if not binary:
            subprocess.run(["cargo", "build", "-q", "--profile", "speed", "-p", "pipitdb-bench",
                            "--example", "suite"], cwd=ROOT, check=True)
            binary = ROOT / "target/speed/examples/suite"
        commit = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                                capture_output=True, text=True).stdout.strip()
        built = "build " + (os.path.basename(str(binary)) if args.binary else f"at {commit}")
        if args.quiet:
            wait_quiet(args.quiet)
        before = load_average()
        best, rows = run(binary, args.suite, files, args.runs)
        after = load_average()
        reference = reference_runner(args.suite, args.directory)
        threads = os.environ.get("THREADS", "1")
        lines = [f"# {args.save}, {built}, {'one thread' if threads == '1' else f'THREADS={threads}'},"
                 f" best of {args.runs}, ms",
                 f"# {machine()}, {datetime.date.today()}, {len(files)} files",
                 f"# load {before:.2f} before, {after:.2f} after"]
        for name, written, sql in queries(args.suite):
            if not written:
                status = "unwritten"
            elif name not in best:
                status = "failed"
            else:
                status = check(args.suite, args.directory, name, sql, rows.get(name, []), reference)
            lines.append(f"{name}\t{best[name]:.1f}" if status == "ok" else f"{name}\t{status}")
        (directory / f"{args.save}.tsv").write_text("\n".join(lines) + "\n")

    # pipitdb, the spike, then each engine, one thread before N.
    paths = sorted(directory.glob("*.tsv"), key=lambda p: (
        {"pipitdb": 0, "spike": 1}.get(p.stem, 2), p.stem.endswith("-N"), p.stem))
    times = {}
    for path in paths:
        header, times[path.stem] = load(path)
        print("\n".join(header))
    columns = list(times)
    names = [name for name, _, _ in queries(args.suite)]
    print("\n| query | " + " | ".join(columns) + " |")
    print("|---" * (len(columns) + 1) + "|")
    for name in names:
        cells = []
        for column in columns:
            ms = times[column].get(name, "")
            cells.append(f"{ms:,.1f}" if isinstance(ms, float) else ms)
        print(f"| {name} | " + " | ".join(cells) + " |")

    # Every one-thread result, then pipitdb on one thread against the
    # engines on N.
    for label, chosen in (("one thread", [c for c in columns if not c.endswith("-N")]),
                          ("pipitdb on one thread, engines on N",
                           [c for c in columns if c in ("pipitdb",) or c.endswith("-N")])):
        scores = score(times, chosen, names)
        if len(scores) < 2:
            continue
        ranked = ", ".join(f"{c} {s:.2f} over {n}" for c, (s, n) in sorted(scores.items(), key=lambda x: x[1][0]))
        print(f"\nScore, {label}, of {len(names)} queries: {ranked}")


if __name__ == "__main__":
    main()
