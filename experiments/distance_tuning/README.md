# Checking the tuning of the exact distance searches

Many defaults of `SearchTuning` in `qldpc.codes.distance._tuning` come from simple heuristics, such as "run an optional step when the work that it acts on reaches the cost of the step".
The [Retuning section of `_tuning.py`](../../src/qldpc/codes/distance/_tuning.py) names the classes of defaults.
These scripts check the tuning, and none of them sets or justifies a value:

- [`workloads.py`](workloads.py) names the codes and the workloads that the other scripts run.
- [`constants.py`](constants.py) checks that the orders of magnitude that the heuristics assume still hold on a machine.
- [`trace.py`](trace.py) shows which decisions the searches of one workload make and what each of their stages costs, and compares the decisions of two traces.
- [`sweep.py`](sweep.py) times the current tuning against a proposed one, or the defaults of two checkouts, in paired trials of [`trial.py`](trial.py), to catch gross regressions or bugs when the tuning or the search code changes.

[`benchmark.py`](benchmark.py) compares the exact distance methods on known codes, and plays no part in tuning.
Every script except `workloads.py` lists its options under `--help`.
`constants.py` and `sweep.py` write their results to `results/`, which Git ignores, and `trace.py` writes a trace to the file that `--output` names.
These outputs belong in the description of the pull request that they support, never in `_tuning.py` or the documentation.

## Checking the orders of magnitude

Run `constants.py` on a quiet machine, with limits on its time and memory:

```bash
python experiments/distance_tuning/constants.py
```

The script first traces a few quick searches.
It prints a table of the cost of each optional step that a derived value gates, in seconds and in the unit of the field that gates the step (`in units`), and then a table, by field, of that cost rounded as `round_to_grid` does (`derived`) beside the default.
It then traces the searches of its cost data set, `COST_DATA_SET`, under each tuning of `COST_MODEL_ARMS`, which together can take hours, and prints each coefficient of the cost model of completion tables, under `measured`, beside its default.
Compare each step's `in units` with the default of the field that gates it.

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

A trace lists the events of the searches' stages, such as the families of information sets that a search builds and its decisions to use completion tables or automorphisms, with the seconds of most stages.
`trace.py` says what each event records.
Each `layer` event of a BZ distance search, and each `weight limit` event of a connected-cluster search, has a `problem` field: the index of its problem among those that one call of a distance function searches.
A call that searches both sectors of a CSS code has problem 0 for the X sector and problem 1 for the Z sector, and a call that searches one sector has only problem 0 (`wrap_problems` in `trace.py` says which calls search both).
`--compare`, which ignores `problem` fields, says whether two traces make the same decisions, and prints the first that differs, unless one trace ends before it.
If the tunings differ only in thresholds and no decision differs, the searches take the same steps.

## Catching regressions

For example, to time the current tuning against a proposed one on three workloads:

```bash
python experiments/distance_tuning/sweep.py --tag automorphism-states --rounds 6 \
    --workloads ccdistance/lp13s5/X ccdistance/surface:17/X ccdistance/qt7/X \
    --proposed '{"automorphism_minimum_states": 100000}'
```

With `--worktrees CURRENT PROPOSED` instead of `--proposed`, the script compares the defaults of two checkouts.
Its summary gives, for each workload, the median time and peak memory of each arm (tuning or checkout), and the median ratio of the proposed arm's time to the current arm's over rounds, with its spread and, from six rounds on, a confidence interval.
It flags arms that disagree on a result.
A disagreement, or a ratio well above one where the traced decisions do not change, points to a bug or a regression; no ratio by itself justifies changing a default.

Before timing:

- Time searches in new single-threaded processes, one at a time, as `sweep.py` and `constants.py` do, since completion tables save less when several searches run at once.
- Do not edit `src/` while the scripts run, since each process imports it anew.
- A trial of one repetition also times work that codes cache, such as the generator matrix of a classical code given by checks.
  Such work is the same in every arm, can dominate a trial of a small code, and drops out of the median with `--repeat 3`.
  Coset searches also reuse their setups, so time each setup budget with one repetition.
- Codes also remember automorphisms once a search finds them, and whether a search finds them depends on the tuning, so time a change of `automorphism_minimum_states` or `automorphism_minimum_layer_work` with one repetition.

## Changing a default

The Retuning section of `_tuning.py` says when to change a default.

## Limits

The heuristics have two known weak spots, which can cost time or memory but never change an exact result:

- The cost of matroid partitioning varies widely between families of codes of the same size.
  Its gate, `partition_work_ratio`, compares a formula for that cost with the estimated work of a search, which can far exceed the work done, so a search can partition and turn out small.
- The cost model of completion tables decides at break-even with rough coefficients, so its marginal decisions can go either way.
