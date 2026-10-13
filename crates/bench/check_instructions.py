#!/usr/bin/env python3
"""Checks gungraun's instruction counts against those recorded for a machine,
in crates/bench/instructions/MACHINE.json. Counts differ between machines but
not between runs on one, so each machine has its own; CI checks ci.json.

Run after `cargo bench -p pipitdb-bench --bench instructions -- --save-summary=json`.
Fails if a benchmark's count is more than LIMIT_PCT away from the file, in
either direction, so that improvements get recorded too. `--save` records the
counts measured instead.

Usage: crates/bench/check_instructions.py [--machine MACHINE] [--save]
"""

import argparse
import json
import pathlib
import sys

LIMIT_PCT = 1
DIR = pathlib.Path("crates/bench/instructions")

parser = argparse.ArgumentParser()
parser.add_argument("--machine", default="ci")
parser.add_argument("--save", action="store_true")
args = parser.parse_args()
path = DIR / f"{args.machine}.json"

measured = {}
for summary_path in pathlib.Path("target/gungraun").rglob("summary.json"):
    summary = json.loads(summary_path.read_text())
    name = summary["function_name"]
    if summary.get("id"):
        name += "." + summary["id"]
    total = summary["profiles"][0]["data"]["total"]["metrics"]
    measured[name] = total["Ir"]["values"]["new"]
saved = json.dumps(dict(sorted(measured.items())), indent=2) + "\n"

if args.save:
    path.write_text(saved)
    print(f"Saved {len(measured)} counts to {path}.")
    sys.exit(0)

expected = json.loads(path.read_text())
failed = False
for name in sorted(expected.keys() | measured.keys()):
    old, new = expected.get(name), measured.get(name)
    if old is None or new is None:
        print(f"{name}: {old or 'missing'} expected, {new or 'missing'} measured !")
        failed = True
        continue
    change_pct = (new - old) / old * 100
    over = abs(change_pct) > LIMIT_PCT
    print(f"{name}: {new} instructions ({change_pct:+.1f}%){' !' if over else ''}")
    failed |= over

if failed:
    print(f"\nMore than {LIMIT_PCT}% off. If expected, run with --save, or update {path} to:\n")
    print(saved)
    sys.exit(1)
