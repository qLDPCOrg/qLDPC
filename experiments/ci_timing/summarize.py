# SPDX-License-Identifier: Apache-2.0

"""Print one TIMING line per test process from the records of profile_plugin.py, and totals.

Temporary, for the CI timing experiment on the branch ci/timing-experiment, which is never merged.
Usage: python summarize.py PROFILE_OUT
"""

import json
import pathlib
import sys

records = {
    path.stem: json.loads(path.read_text())
    for path in sorted(pathlib.Path(sys.argv[1]).glob("*.json"))
}
total_cpu = total_compile = 0.0
for name, record in records.items():
    compile_time = sum(seconds for _, _, seconds in record["compiles"])
    if name != "controller":
        total_cpu += record["cpu"]
        total_compile += compile_time
    first = record["first_test"]
    print(
        f"TIMING process {name}: wall={record['wall']:.1f} first_test="
        f"{'-' if first is None else f'{first:.1f}'} cpu={record['cpu']:.1f}"
        f" compile={compile_time:.1f}"
    )
print(f"TIMING workers: cpu={total_cpu:.1f} compile={total_compile:.1f}")
