# Checking the tuning of the exact distance searches

Many defaults of `SearchTuning` in `qldpc.codes.distance._tuning` come from simple heuristics, such as "run an optional step when the work that it acts on reaches the cost of the step".
The [Retuning section of `_tuning.py`](../../src/qldpc/codes/distance/_tuning.py) names the classes of defaults.
These scripts check the tuning, and none of them sets or justifies a value, apart from running searches on one thread:

- [`workloads.py`](workloads.py) names the codes and the workloads that the other scripts run.
- [`constants.py`](constants.py) checks that the orders of magnitude that the heuristics assume still hold on a machine.
- [`trace.py`](trace.py) shows which decisions the searches of one workload make and what each of their stages costs, and compares the decisions of two traces.
- [`sweep.py`](sweep.py) times the current tuning against a proposed one, or the defaults of two checkouts, in paired trials of [`trial.py`](trial.py), to catch gross regressions or bugs when the tuning or the search code changes.

[`benchmark.py`](benchmark.py) compares the exact distance methods on known codes, and plays no part in tuning.
Every script except `workloads.py` describes its outputs and options under `--help`.
The outputs of the checks belong in the description of the pull request that they support, never in `_tuning.py` or the documentation.
[`workloads_test.py`](workloads_test.py) checks that the `qt7` workload keeps the parity checks of earlier releases, so that timings of it stay comparable; CI does not run it, so run `python -m pytest experiments/distance_tuning` after changing `QTCode` or `workloads.py`.

## Checking the orders of magnitude

Run `constants.py` as its `--help` describes:

```bash
python experiments/distance_tuning/constants.py
```

`constants.py` does not check `macwilliams_shared_walk_rank`, which gates a test of whether the X and Z checks of a logical-operator counter span the same space.
Trace `mcount` workloads of counters of rank 14 or more, such as `mcount/qrm:2:6/4`, with the test forced (`--tuning '{"macwilliams_shared_walk_rank": 0}'`).
The default assumes that the test costs about as much as a walk of `2**14` steps, so compare the costliest `self-dual test` with `2**14` times the seconds per step of `walk` events of at least `2**16` steps.

## Tracing decisions

For example, to trace a search under the current tuning and under a proposed one, and to compare the decisions of the two traces:

```bash
python experiments/distance_tuning/trace.py ccdistance/lp13s5/X \
    --output experiments/distance_tuning/results/current.json
python experiments/distance_tuning/trace.py ccdistance/lp13s5/X \
    --tuning '{"automorphism_minimum_states": 100000}' \
    --output experiments/distance_tuning/results/proposed.json
python experiments/distance_tuning/trace.py --compare \
    experiments/distance_tuning/results/current.json experiments/distance_tuning/results/proposed.json
```

`trace.py` says what a trace records, and what `--compare` compares.

## Catching regressions

For example, to time the current tuning against a proposed one on three workloads:

```bash
python experiments/distance_tuning/sweep.py --tag automorphism-states --rounds 6 \
    --workloads ccdistance/lp13s5/X ccdistance/surface:17/X ccdistance/qt7/X \
    --proposed '{"automorphism_minimum_states": 100000}'
```

`sweep.py` says what the summary of a comparison gives and flags, and when to time with `--repeat`.
A flagged disagreement, or a proposed arm much slower than the current one where the traced decisions do not change, points to a bug or a regression.

Before timing:

- Time searches in new single-threaded processes, one at a time, as `sweep.py` and `constants.py` do, since completion tables save less when several searches run at once.
  `sweep.py` sets `NUMBA_NUM_THREADS=1` for its trials, and `trace.py` and `benchmark.py` set `num_threads` to 1, unless a `--tuning` of `trace.py` sets it; set `NUMBA_NUM_THREADS=1` when you run `trial.py` directly.
- Do not edit `src/` while the scripts run, since each process imports it anew.

## Changing a default

The Retuning section of `_tuning.py` says when to change a default.

## Limits

The heuristics have two known weak spots, which can cost time or memory but never change an exact result:

- The cost of matroid partitioning varies widely between families of codes of the same size.
  Its gate, `partition_work_ratio`, compares a formula for that cost with the estimated work of a search, which can far exceed the work done, so a search can partition and turn out small.
- The cost model of completion tables decides at break-even with rough coefficients, so its marginal decisions can go either way.
