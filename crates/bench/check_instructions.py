#!/usr/bin/env python3
"""Checks gungraun's instruction counts against crates/bench/instructions.

Run after `cargo bench -p pipitdb-bench --bench instructions -- --save-summary=json`.
Fails if a benchmark's count is more than LIMIT_PCT away from the file, in
either direction, so that improvements get recorded too.
"""

import json
import pathlib
import sys

LIMIT_PCT = 5
EXPECTED_PATH = pathlib.Path("crates/bench/instructions")

expected = {}
for line in EXPECTED_PATH.read_text().splitlines():
    name, count = line.split()
    expected[name] = int(count)

measured = {}
for path in pathlib.Path("target/gungraun").rglob("summary.json"):
    summary = json.loads(path.read_text())
    name = summary["function_name"]
    if summary.get("id"):
        name += "." + summary["id"]
    total = summary["profiles"][0]["data"]["total"]["metrics"]
    measured[name] = total["Ir"]["values"]["new"]

failed = False
for name in sorted(expected.keys() | measured.keys()):
    old, new = expected.get(name), measured.get(name)
    if old is None or new is None:
        print(f"{name}: {old or 'missing'} expected, {new or 'missing'} measured")
        failed = True
        continue
    change_pct = (new - old) / old * 100
    print(f"{name}: {new} instructions ({change_pct:+.1f}%)")
    if abs(change_pct) > LIMIT_PCT:
        failed = True

if failed:
    print(f"\nMore than {LIMIT_PCT}% off. If expected, update {EXPECTED_PATH} to:\n")
    for name in sorted(measured):
        print(f"{name} {measured[name]}")
    sys.exit(1)
