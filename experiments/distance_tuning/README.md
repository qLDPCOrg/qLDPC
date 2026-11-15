# Checking the tuning of the exact distance searches

The defaults of `SearchTuning` in `qldpc.codes.distance._tuning` are round values that simple heuristics give, such as "run an optional step once the work that it acts on reaches the cost of the step".
The [Retuning section](../../src/qldpc/codes/distance/_tuning.py) of the module docstring of `_tuning` names the classes of defaults and their heuristics, and says when to change a default.
The comment of each field names the class of its default.
The scripts of this directory check the tuning, and none of them sets or justifies a value:

- [`constants.py`](constants.py) checks that the orders of magnitude that the heuristics assume still hold on a machine.
- [`trace.py`](trace.py) shows which decisions the searches of one workload make and what each of their stages costs, and compares the decisions of two traces.
- [`sweep.py`](sweep.py) times the whole current tuning against a whole proposed one, or the defaults of two checkouts, in paired trials of [`trial.py`](trial.py), to catch gross regressions or bugs when the tuning or the search code changes.
- [`workloads.py`](workloads.py) names the codes and the workloads that these scripts run.

[`benchmark.py`](benchmark.py) compares the exact distance methods on known codes, and plays no part in tuning.
The [experiments README](../README.md#distance-tuning) introduces the scripts, and their docstrings give their options.
`constants.py` and `sweep.py` write their results to `results/`, which Git ignores, and `trace.py` writes a trace to the file that `--output` names.
The outputs of the scripts belong in the description of the pull request that they support, never in `_tuning.py` or the documentation.

## Checking the orders of magnitude

Run `constants.py` on a quiet machine, under a guard of time and memory:

```bash
python experiments/distance_tuning/constants.py
```

The script first traces a few quick searches.
It prints a table of the cost of each optional step that a derived value gates, in seconds and in the unit of the field that gates the step: candidate/set evaluations for Brouwer-Zimmermann searches, and states for connected-cluster searches.
It then prints the value that the heuristic of each derived value gives for that cost, under `derived`, beside the default.
Last, it traces the searches of its cost data set, `COST_DATA_SET`, under four tunings, which can take hours, and prints each coefficient of the cost model of completion tables, under `measured`, beside its default.
The functions of `constants.py` say how it measures each cost.

Read the output by the rule of the Retuning section: a default calls for a change only if the cost that its heuristic assumes is off by more than a factor of about 3.
So compare the cost of each step, under `in units`, with the default of the field that gates the step.
Neighboring values of 1, 2, or 5 times a power of ten differ by at most a factor of 2.5, so a derived value one step away from its default calls for no change.
The coefficients of the cost model are rough estimates, and compare with their defaults in the same way.

The script saves each measurement with the commit and the uncommitted changes of `src/` that it ran on.
So it measures every cost again after a commit or a change to `src/`, and an interrupted run resumes where it stopped.

`macwilliams_shared_walk_rank` gates a test of whether the X and Z checks of a counter span the same space, and `constants.py` does not check the assumption of its comment.
To check that assumption, trace `mcount` workloads with the test forced, as in `trace.py mcount/qgolay/8 --tuning '{"macwilliams_shared_walk_rank": 0}'`.
Then compare the seconds of each `self-dual test` event with the seconds per step of the `walk` events.

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

A trace lists the events of each stage of the searches of its workload, such as the families of information sets that a search builds, its decisions to use completion tables or automorphisms, and the layers that it scans, each with its seconds.
`--compare` lists the decisions that differ between two traces of one workload.
If two tunings differ only in thresholds, a search whose decisions do not change takes the same steps, apart from the time of the decisions themselves.
The wrapper of each event in `trace.py` says what the event records, and the docstring of `trace.py` says what the comparison leaves out, such as the sets of each family.

## Catching regressions

For example, to time the whole current tuning against a proposed one on three workloads:

```bash
python experiments/distance_tuning/sweep.py --tag automorphism-states --rounds 3 \
    --workloads ccdistance/lp13s5/X ccdistance/surface:17/X ccdistance/qt7/X \
    --proposed '{"automorphism_minimum_states": 100000}'
```

With `--worktrees CURRENT PROPOSED` instead of `--proposed`, the script compares the defaults of two checkouts.
For each workload, its summary gives the median time of each arm, the median ratio of the time of the proposed arm to that of the current arm, with the spread of the ratios and a confidence interval, and the median peak memory of each arm.
It also flags each workload whose arms disagree on a result.
A disagreement, or a ratio far from one on a workload whose traced decisions do not change, points to a bug or a regression.
No ratio justifies a change of a default by itself (see the Retuning section).

Before timing:

- Time searches in new single-threaded processes, one at a time, as `sweep.py` and `constants.py` do, since completion tables save less when several searches run at once.
- Do not edit `src/` while the scripts run, since each process imports it anew.
- A trial of one repetition also times work that a search caches on its code, such as the generator matrix of a classical code given by checks.
  That work is the same in every arm, and can dominate a trial of a small code.
  It shrinks ratios toward one without reversing them, so time such workloads with `--repeat 3`, whose median leaves it out.
- A code also remembers its automorphisms once a search finds them, and whether a search finds them depends on the tuning.
  So time a change of when searches ask for automorphisms, such as of `automorphism_minimum_states` or `automorphism_minimum_layer_work`, with one repetition, so that every trial pays for asking.

## Changing a default

The Retuning section of the module docstring of `_tuning` says when to change a default.
The scripts of this directory make the checks that the section calls for: `constants.py` checks the orders of magnitude that the heuristics assume, and `trace.py` and `sweep.py` catch the regressions and bugs that a change can bring.

## Limits

Two weak spots of the heuristics are known:

- The cost of matroid partitioning varies a lot between families of codes of the same size.
  Its gate, `partition_work_ratio`, compares that cost with the estimated work of a search, which can far exceed the work that the search does.
  So a search can partition and then turn out to be small.
- The cost model of completion tables decides at break-even with rough coefficients, so its marginal decisions can go either way.

These weak spots can cost searches time or memory, but never change an exact result.
