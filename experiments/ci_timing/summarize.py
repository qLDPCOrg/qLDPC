# SPDX-License-Identifier: Apache-2.0

"""Summarize the records of probe_plugin.py.

Temporary, for the CI timing probe on the branch ci/timing-probe, which is never merged.
Usage: python summarize.py PROFILE_OUT [T0], where T0 is the epoch time at which the step started.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
records = {path.stem: json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))}
t0 = float(sys.argv[2]) if len(sys.argv) > 2 else min(r["imported"] for r in records.values())


def rel(value: float | None) -> str:
    return "-" if value is None else f"{value - t0:6.1f}"


def package(name: str) -> str:
    return name.split(".")[0]


print("PROBE phases (seconds after the step started)")
print("PROBE process  import config collect_start collect_end first_test end | cpu compile"
      " galois qldpc other collect_compile rss_mb")
file_wall: dict[str, float] = collections.Counter()
file_compile: dict[str, float] = collections.Counter()
file_worker: dict[str, str] = {}
function_compile: dict[str, float] = collections.Counter()
function_processes: dict[str, set[str]] = collections.defaultdict(set)
tests = []
for name, record in records.items():
    first = min((start for _, start, _ in record["tests"]), default=None)
    by_package: dict[str, float] = collections.Counter()
    collect_compile = 0.0
    for test, function, seconds in record["compiles"]:
        by_package[package(function)] += seconds
        function_compile[function] += seconds
        function_processes[function].add(name)
        if test in ("<startup>", "<collection>"):
            collect_compile += seconds
        else:
            file_compile[test.split("::")[0]] += seconds
    total = sum(by_package.values())
    other = total - by_package["galois"] - by_package["qldpc"]
    print(
        f"PROBE {name:10s} {rel(record['imported'])} {rel(record.get('configured'))}"
        f" {rel(record.get('collection_start'))} {rel(record.get('collection_end'))}"
        f" {rel(first)} {rel(record['end'])} | {record['cpu']:6.1f} {total:6.1f}"
        f" {by_package['galois']:6.1f} {by_package['qldpc']:6.1f} {other:6.1f}"
        f" {collect_compile:6.1f} {record['max_rss_mb']:7.0f}"
    )
    for node_id, start, end in record["tests"]:
        file = node_id.split("::")[0]
        file_wall[file] += end - start
        file_worker[file] = name
        tests.append((end - start, node_id))

print("\nPROBE test files by wall time (worker, wall, compile within)")
for file, wall in sorted(file_wall.items(), key=lambda item: -item[1])[:30]:
    print(f"PROBE file {wall:6.1f} {file_compile[file]:6.1f} {file_worker[file]:4s} {file}")

print("\nPROBE slowest tests")
for seconds, node_id in sorted(tests, reverse=True)[:30]:
    print(f"PROBE test {seconds:6.1f} {node_id[:160]}")

print("\nPROBE compiled functions by total compile time (processes that compiled them)")
for function, seconds in sorted(function_compile.items(), key=lambda item: -item[1])[:30]:
    print(f"PROBE compile {seconds:6.1f} {len(function_processes[function]):2d} {function}")
