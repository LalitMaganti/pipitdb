#!/usr/bin/env python3
"""Times two builds of the clickbench example on the same files, taking turns
so a change in the machine's load hits both, and prints each query's best
time for each and the change.

Build each with `cargo build --profile speed -p pipitdb-bench --example
clickbench` and copy target/speed/examples/clickbench aside.

Usage: crates/bench/clickbench/compare.py BEFORE AFTER [--rounds N] FILE...
"""

import argparse
import os
import re
import subprocess

TIME = re.compile(r"^Q(\S+): ([\d.]+)(ns|µs|ms|s) ")
SECONDS = {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1.0}


def times(binary, files):
    """Each query's time in seconds, from one run of every query."""
    env = dict(os.environ, QUIET="1")
    run = subprocess.run([binary, *files], capture_output=True, text=True, env=env, check=True)
    found = {}
    for line in run.stderr.splitlines():
        if match := TIME.match(line):
            number, value, unit = match.groups()
            found[number] = float(value) * SECONDS[unit]
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    best = {"before": {}, "after": {}}
    for _ in range(args.rounds):
        for side in ("before", "after"):
            for number, seconds in times(getattr(args, side), args.files).items():
                best[side][number] = min(seconds, best[side].get(number, seconds))
    print(f"{'query':10} {'before ms':>10} {'after ms':>10} {'change':>8}")
    for number, before in best["before"].items():
        after = best["after"].get(number)
        if after is None:
            print(f"{number:10} {before * 1e3:10.2f} {'failed':>10}")
            continue
        change = (after / before - 1) * 100
        print(f"{number:10} {before * 1e3:10.2f} {after * 1e3:10.2f} {change:+7.1f}%")


if __name__ == "__main__":
    main()
