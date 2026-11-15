# Checking the tuning of the exact distance searches

The defaults of `SearchTuning` in `qldpc.codes.distance._tuning` are round values that simple heuristics give, such as "run an optional step once the work that it acts on reaches the cost of the step".
The [Retuning section](../../src/qldpc/codes/distance/_tuning.py) of the module docstring of `_tuning` names the classes of defaults and their heuristics.
The scripts of this directory check the tuning, and none of them sets or justifies a value:

- [`constants.py`](constants.py) checks that the orders of magnitude that the heuristics assume still hold on a machine.
- [`trace.py`](trace.py) shows which decisions the searches of one workload make and what each of their stages costs, and compares the decisions of two traces.
- [`sweep.py`](sweep.py) times the current tuning against a proposed one, or the defaults of two checkouts, in paired trials of [`trial.py`](trial.py), to catch gross regressions or bugs when the tuning or the search code changes.
- [`workloads.py`](workloads.py) names the codes and the workloads that these scripts run.

[`benchmark.py`](benchmark.py) compares the exact distance methods on known codes, and plays no part in tuning.
The docstrings of the scripts give their options.
`constants.py` and `sweep.py` write their results to `results/`, which Git ignores, and `trace.py` writes a trace to the file that `--output` names.
These outputs belong in the description of the pull request that they support, never in `_tuning.py` or the documentation.

## Checking the orders of magnitude

Run `constants.py` on a quiet machine, under a guard of time and memory:

```bash
python experiments/distance_tuning/constants.py
```

The script first traces a few quick searches, and prints the cost of each optional step that a derived value gates, in seconds and in the unit of the field that gates it (candidate/set evaluations or states), with the value that the heuristic gives for that cost, under `derived`, beside the default.
It then traces the searches of its cost data set, `COST_DATA_SET`, under four tunings, which can take hours, and prints each coefficient of the cost model of completion tables, under `measured`, beside its default.
Its functions say how it measures each cost.
It saves each measurement with the source that it ran on, so it measures again after any change of `src/`, and resumes an interrupted run.

By the rule of the Retuning section, a default calls for a change only if the cost under `in units` differs from it by more than a factor of about 3.
Neighboring values of 1, 2, or 5 times a power of ten differ by at most a factor of 2.5, so a derived value one step from its default calls for no change.
The coefficients of the cost model compare with their defaults in the same way.

`constants.py` does not check `macwilliams_shared_walk_rank`, which gates a test of whether the X and Z checks of a counter span the same space.
To check its assumption, trace `mcount` workloads with the test forced, as in `trace.py mcount/qgolay/8 --tuning '{"macwilliams_shared_walk_rank": 0}'`, and compare the seconds of each `self-dual test` event with the seconds per step of the `walk` events.

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

A trace lists the events of each stage of the searches, such as the families of information sets that a search builds and its decisions to use completion tables or automorphisms, each with its seconds; the wrappers in `trace.py` say what each event records.
`--compare` lists the decisions that differ between two traces, and if two tunings differ only in thresholds, a search whose decisions agree takes the same steps.

## Catching regressions

For example, to time the current tuning against a proposed one on three workloads:

```bash
python experiments/distance_tuning/sweep.py --tag automorphism-states --rounds 3 \
    --workloads ccdistance/lp13s5/X ccdistance/surface:17/X ccdistance/qt7/X \
    --proposed '{"automorphism_minimum_states": 100000}'
```

With `--worktrees CURRENT PROPOSED` instead of `--proposed`, the script compares the defaults of two checkouts.
Its summary gives, for each workload, the median time and peak memory of each arm and the median ratio of their times, with its spread and a confidence interval, and flags arms that disagree on a result.
A disagreement, or a ratio far from one where the traced decisions do not change, points to a bug or a regression; no ratio by itself justifies changing a default.

Before timing:

- Time searches in new single-threaded processes, one at a time, as `sweep.py` and `constants.py` do, since completion tables save less when several searches run at once.
- Do not edit `src/` while the scripts run, since each process imports it anew.
- A trial of one repetition also times work that codes cache, such as the generator matrix of a classical code given by checks, which is the same in every arm and can dominate a trial of a small code; `--repeat 3` leaves it out of the median.
- Codes also remember automorphisms once a search finds them, and whether a search finds them depends on the tuning, so time a change of when searches ask for them, such as of `automorphism_minimum_states` or `automorphism_minimum_layer_work`, with one repetition.

## Changing a default

The Retuning section of the module docstring of `_tuning` says when to change a default, and the scripts above make the checks that it calls for.

## Limits

Two weak spots of the heuristics are known, which can cost time or memory but never change an exact result:

- The cost of matroid partitioning varies a lot between families of codes of the same size, and its gate, `partition_work_ratio`, compares that cost with the estimated work of a search, which can far exceed the work done, so a search can partition and then turn out to be small.
- The cost model of completion tables decides at break-even with rough coefficients, so its marginal decisions can go either way.
