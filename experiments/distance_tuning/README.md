# Checking the tuning of the exact distance searches

Many defaults of `SearchTuning` in `qldpc.codes.distance._tuning` come from simple heuristics, such as "run an optional step when the work that it acts on reaches the cost of the step".
The [Retuning section of `_tuning.py`](../../src/qldpc/codes/distance/_tuning.py) names the kinds of default and says when to change a default.
[Section 4.4 of the Brouwer-Zimmermann code guide](../../src/qldpc/codes/distance/brouwer_zimmermann_code.md#44-heuristics-and-cost-models) describes the heuristics of Brouwer-Zimmermann searches and their known weak spots.
These scripts check the tuning, and none of them sets or justifies a default:

- [`workloads.py`](workloads.py) names the codes and the workloads that the other scripts run.
- [`constants.py`](constants.py) checks that the orders of magnitude that the heuristics assume still hold on a machine, and measures the coefficients of the cost model of completion tables ([Section 3.9 of the Brouwer-Zimmermann guide](../../src/qldpc/codes/distance/brouwer_zimmermann.md#39-skipping-completions-by-completion-tables); [Section 4.4 of `brouwer_zimmermann_code.md`](../../src/qldpc/codes/distance/brouwer_zimmermann_code.md#44-heuristics-and-cost-models) names them and gives the model).
- [`trace.py`](trace.py) shows which decisions the searches of one workload make and what each of their stages costs, and compares the decisions of two traces.
- [`sweep.py`](sweep.py) times two arms in paired trials, either the current tuning against a proposed one or the defaults of two checkouts, to catch gross regressions or bugs when the tuning or the search code changes.
- [`trial.py`](trial.py) times one workload under one tuning in one process; `sweep.py` runs it for each trial.

[`benchmark.py`](benchmark.py) compares the exact distance methods on known codes, and plays no part in tuning.
Every script except `workloads.py` describes its outputs and options under `--help`.
Report the outputs of the checks in the description of the pull request that they support ([Where documentation belongs](../../docs/source/contributing/standards/writing.rst#where-documentation-belongs)).
[`workloads_test.py`](workloads_test.py) checks that the `qt7` workload keeps the parity checks of earlier releases, so that timings of it stay comparable; CI does not run it, so run `python -m pytest experiments/distance_tuning` after changing `QTCode` or `workloads.py`.

## Checking the orders of magnitude

Run `constants.py`, which takes hours, as its `--help` describes:

```bash
python experiments/distance_tuning/constants.py
```

`constants.py` does not check `macwilliams_shared_walk_rank`, whose description in `_tuning.py` says what it gates.
To check its default of 14:

1. Trace an `mcount` workload whose X and Z checks have equal ranks of 14 or more, such as `mcount/qrm:2:6/4`, with `--tuning '{"macwilliams_shared_walk_rank": 0}'`.
   With this setting, the trace records the test of whether the X and Z checks span the same space, whatever the default.
2. Take the `seconds` of the costliest `self-dual test` event as the cost of the test.
3. For each `walk` event of at least `2**16` steps, divide its `seconds` by its `steps` to estimate the cost of a step (`trace.py` describes these events).
4. Compare the cost of the test with `2**14` times each estimate of the cost of a step.
   If they differ by more than a factor of about 3, the default is off (see the Retuning section of `_tuning.py`).

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

`sweep.py` says what the summary of a comparison gives, when it warns, and when to time with `--repeat`.
A warning that the arms disagree on a result, or a proposed arm much slower than the current one where the traced decisions do not change, points to a bug or a regression.

Before timing:

- Time searches in new single-threaded processes, one at a time, as `sweep.py` and `constants.py` do, since completion tables save less when several searches run at once.
  `trace.py` and `benchmark.py` run their searches with `num_threads=1`.
  When you run `trial.py` directly, limit it to one thread as its `--help` says.
- Do not edit `src/` while the scripts run, since each process imports it anew.
