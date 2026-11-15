# Checking the tuning of the exact distance searches

Most defaults of `SearchTuning` in `qldpc.codes.distance._tuning` are round values that simple heuristics give, such as "run an optional step once the work that it acts on reaches the cost of the step".
The [Retuning section of `_tuning.py`](../../src/qldpc/codes/distance/_tuning.py) names the classes of defaults and says when to change one.
These scripts check the tuning, and none of them sets or justifies a value:

- [`workloads.py`](workloads.py) names the codes and the workloads that the other scripts run.
- [`constants.py`](constants.py) checks that the orders of magnitude that the heuristics assume still hold on a machine.
- [`trace.py`](trace.py) shows which decisions the searches of one workload make and what each of their stages costs, and compares the decisions of two traces.
- [`sweep.py`](sweep.py) times the current tuning against a proposed one, or the defaults of two checkouts, in paired trials of [`trial.py`](trial.py), to catch gross regressions or bugs when the tuning or the search code changes.

[`benchmark.py`](benchmark.py) compares the exact distance methods on known codes, and plays no part in tuning.
Each script's `--help` gives its options.
`constants.py` and `sweep.py` write their results to `results/`, which Git ignores, and `trace.py` writes a trace to the file that `--output` names.
These outputs belong in the description of the pull request that they support, never in `_tuning.py` or the documentation.

## Checking the orders of magnitude

Run `constants.py` on a quiet machine, with limits on its time and memory:

```bash
python experiments/distance_tuning/constants.py
```

The script first traces a few quick searches.
For each optional step that a derived value gates, it prints the cost of the step in seconds and, under `in units`, in the unit of the field, and, under `derived`, that cost rounded as `round_to_grid` does, beside the default.
It then traces the searches of its cost data set, `COST_DATA_SET`, under four tunings, which can take hours, and prints each coefficient of the cost model of completion tables, under `measured`, beside its default.
It saves each measurement with a fingerprint of the qLDPC source, so it measures again after a commit or a change to `src/`, and resumes an interrupted run.

`constants.py` does not check `macwilliams_shared_walk_rank`, which gates a test of whether the X and Z checks of a counter span the same space.
Trace `mcount` workloads of counters of rank 14 or more, such as `mcount/qrm:2:6/4`, with the test forced (`--tuning '{"macwilliams_shared_walk_rank": 0}'`), and compare the costliest `self-dual test` with the seconds per step of `walk` events of at least `2**16` steps.

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

A trace lists the events of the stages of the searches, such as the families of information sets that a search builds and its decisions to use completion tables or automorphisms, and the seconds of most stages.
`trace.py` says what each event records.
`--compare` says whether two traces make the same decisions, and prints the first that differs.
If the tunings differ only in thresholds and no decision differs, the searches take the same steps.

## Catching regressions

For example, to time the current tuning against a proposed one on three workloads:

```bash
python experiments/distance_tuning/sweep.py --tag automorphism-states --rounds 6 \
    --workloads ccdistance/lp13s5/X ccdistance/surface:17/X ccdistance/qt7/X \
    --proposed '{"automorphism_minimum_states": 100000}'
```

With `--worktrees CURRENT PROPOSED` instead of `--proposed`, the script compares the defaults of two checkouts.
For each workload, its summary gives the median time and peak memory of each tuning, or arm, and the median ratio of their times over rounds, with its spread and, from six rounds on, a confidence interval; it flags arms that disagree on a result.
A disagreement, or a ratio well above one where the traced decisions do not change, points to a bug or a regression; no ratio by itself justifies changing a default.

Before timing:

- Time searches in new single-threaded processes, one at a time, as `sweep.py` and `constants.py` do, since completion tables save less when several searches run at once.
- Do not edit `src/` while the scripts run, since each process imports it anew.
- A trial of one repetition also times work that codes cache, such as the generator matrix of a classical code given by checks, which is the same in every arm and can dominate a trial of a small code; `--repeat 3` leaves it out of the median.
- Codes also remember automorphisms once a search finds them, and whether a search finds them depends on the tuning, so time a change of `automorphism_minimum_states` or `automorphism_minimum_layer_work` with one repetition.

## Changing a default

The Retuning section of `_tuning.py` says when to change a default.

## Limits

Two weak spots of the heuristics are known, which can cost time or memory but never change an exact result:

- The cost of matroid partitioning varies widely between families of codes of the same size.
  Its gate, `partition_work_ratio`, compares a formula for that cost with the estimated work of a search, which can far exceed the work done, so a search can partition and turn out small.
- The cost model of completion tables decides at break-even with rough coefficients, so its marginal decisions can go either way.
