#!/usr/bin/env python3
"""Runs pipitdb's ClickBench queries, best of RUNS, saves the times to
results/pipitdb.tsv with the commit, and prints them beside every other
result saved there: the engines engines.py ran, one thread and N, and
results/spike.tsv, the prototype pipitdb grew from. Also ClickBench's score
for each, over the queries all of them ran: the geometric mean of
(t + 10) / (best + 10), so 1.0 is best on every query.

The machine's load changes times, so run this on an idle machine, and
compare results from the same one: each file says which it came from.

Usage: crates/bench/clickbench/scoreboard.py DIR [--runs 3]
       [--binary PATH] [--save NAME] [--no-run]
DIR holds hits_*.parquet. --binary runs another build, such as the spike's,
and --save names its results file.
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
RESULTS = HERE / "results"
ROOT = HERE.parents[2]
TIME = re.compile(r"^Q(\S+): ([\d.]+)(ns|µs|ms|s) ")
SECONDS = {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1.0}
sys.path.insert(0, str(HERE))
from engines import machine  # noqa: E402


def names():
    return [line.split("\t")[0] for line in (HERE / "queries.tsv").read_text().splitlines()]


def run(binary, files, runs):
    """Each query's best time in ms over `runs` runs of every query."""
    best = {}
    for _ in range(runs):
        out = subprocess.run([binary, *files], capture_output=True, text=True,
                             env=dict(os.environ, QUIET="1"), check=True)
        for line in out.stderr.splitlines():
            if match := TIME.match(line):
                name, value, unit = match.groups()
                ms = float(value) * SECONDS[unit] * 1e3
                best[name] = min(ms, best.get(name, ms))
    return best


def load(path):
    """A results file: its header lines, and each query's ms or None."""
    header, times = [], {}
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            header.append(line)
            continue
        name, ms = line.split("\t")
        times[name] = None if ms == "failed" else float(ms)
    return header, times


def score(times, columns, queries):
    """ClickBench's score for each column, over `queries`."""
    scores = {}
    for column in columns:
        logs = []
        for query in queries:
            best = min(times[c][query] for c in columns)
            logs.append(math.log((times[column][query] + 10) / (best + 10)))
        scores[column] = math.exp(sum(logs) / len(logs))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--binary")
    parser.add_argument("--save", default="pipitdb")
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    files = sorted(str(p) for p in pathlib.Path(args.directory).glob("hits_*.parquet"))
    if not args.no_run:
        binary = args.binary
        if not binary:
            subprocess.run(["cargo", "build", "-q", "--profile", "speed", "-p", "pipitdb-bench",
                            "--example", "clickbench"], cwd=ROOT, check=True)
            binary = ROOT / "target/speed/examples/clickbench"
        commit = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                                capture_output=True, text=True).stdout.strip()
        built = "build " + (os.path.basename(str(binary)) if args.binary else f"at {commit}")
        best = run(binary, files, args.runs)
        lines = [f"# {args.save}, {built}, one thread, best of {args.runs}, ms",
                 f"# {machine()}, {datetime.date.today()}, {len(files)} files"]
        lines += [f"{name}\t{best[name]:.1f}" for name in names() if name in best]
        (RESULTS / f"{args.save}.tsv").write_text("\n".join(lines) + "\n")

    # pipitdb, the spike, then each engine, one thread before N.
    paths = sorted(RESULTS.glob("*.tsv"), key=lambda p: (
        {"pipitdb": 0, "spike": 1}.get(p.stem, 2), p.stem.endswith("-N"), p.stem))
    times = {}
    for path in paths:
        header, times[path.stem] = load(path)
        print("\n".join(header))
    columns = list(times)
    print("\n| query | " + " | ".join(columns) + " |")
    print("|---" * (len(columns) + 1) + "|")
    for name in names():
        cells = []
        for column in columns:
            ms = times[column].get(name)
            cells.append("" if ms is None else f"{ms:,.1f}")
        print(f"| {name} | " + " | ".join(cells) + " |")

    # Scores over the queries pipitdb and every one-thread result ran, then
    # pipitdb one thread against the engines on N.
    for label, chosen in (("one thread", [c for c in columns if not c.endswith("-N")]),
                          ("pipitdb on one thread, engines on N",
                           [c for c in columns if c in ("pipitdb",) or c.endswith("-N")])):
        common = [n for n in names() if all(times[c].get(n) is not None for c in chosen)]
        if len(chosen) < 2 or not common:
            continue
        scores = score(times, chosen, common)
        ranked = ", ".join(f"{c} {s:.2f}" for c, s in sorted(scores.items(), key=lambda x: x[1]))
        print(f"\nScore, {label}, over {len(common)} queries: {ranked}")


if __name__ == "__main__":
    main()
