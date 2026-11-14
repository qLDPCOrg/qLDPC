# Deriving the tuning of the exact distance searches

The derived defaults of `SearchTuning` in `qldpc.codes.distance._tuning` follow from rules and from constants of the machine, not from timing candidates.
The [Retuning section](../../src/qldpc/codes/distance/_tuning.py) of its module docstring states the classes of values and their rules, and the comment of each field names the class of its default.
This directory holds what determines the derived values, and what confirms a change of rule:

- [`constants.py`](constants.py) measures the constants of the machine that the derived values need, except that of `automorphism_minimum_layer_work`, including the five coefficients of the cost model of completion tables, and prints the values that the rules give beside the defaults.
- [`trace.py`](trace.py) records the decisions of the searches of one workload under one tuning, and compares the decisions of two traces.
- [`sweep.py`](sweep.py) times the whole current tuning against a whole proposed one, or the defaults of two checkouts, in paired trials of [`trial.py`](trial.py).
- [`workloads.py`](workloads.py) names the codes and the workloads that these scripts run.

[`benchmark.py`](benchmark.py) is a benchmark of the exact distance methods, which is not part of tuning.
The [experiments README](../README.md#distance-tuning) introduces the scripts, and their docstrings give their options.
The scripts write their results to `results/`, which Git ignores.

## Rules and constants

Every derived value follows from a rule fixed before measuring and, unless it is a limit at equality, from a measured constant of the machine, which `constants.py` measures for every derived value except `automorphism_minimum_layer_work`:

- **Break-even gates.** An optional step, such as building a family of information sets, solving linear programs, or asking for automorphisms, runs once the work that it acts on reaches the cost of the step, the costliest of its measurements.
  Work counts candidate/set evaluations in BZ searches, and states in connected-support searches.
  The value is rounded by ratio to 1, 2, or 5 times a power of ten.
- **Limits at equality.** A limit that compares two estimates in one unit breaks even where they are equal.
- **Measured coefficients.** The coefficients of the cost model of completion tables are constants of the machine, measured on the costs of the layers of the cost data set of the Retuning section.

The other values are budgets of memory, interrupt latency, or setup effort, statistical conventions, or the widest width of blocks allowed, as the Retuning section defines them.
Two fields, `coset_route_minimum_work` and `macwilliams_shared_walk_rank`, are in no class, and their comments say how they were chosen.
`macwilliams_shared_walk_rank` gates a test that lets one walk of a check space serve both sectors of a count by the MacWilliams method.
Its break-even constant lies on the edge between two ranks.
Five runs of an earlier version of `constants.py` traced `mcount/qgolay/8`, `mcount/qrm:2:6/4`, and `mcount/qrm:2:7/1`, with the test forced, and timed the walks and tests of the traces.
In each run, the costliest test took as long as a walk of `2.0 * 10**4` to `3.4 * 10**4` steps, on both sides of `2**14.5`, so breaking even gives 14 in one run and 15 in four.
Every traced test found checks that do not span the same space.

On an Apple M3 Max that also ran other work, on the library code that follows #869 and #870, one candidate/set evaluation of a BZ search took 1.17 to 1.20 ns, the median over five searches of 1.15 to 1.95 ns, and three runs of the script gave these constants, each the costliest of its measurements, and values:

| Field | Value | The step, and its cost |
| --- | --- | --- |
| `partition_work_ratio` | 5 | Matroid partitioning, for each unit `b^2 k^2 n` of its gate: 4.53 to 4.56 ns, or 3.81 to 3.86 evaluations, on `distance/classical:128:80:6/-`, which tries coverages up to 6; `distance/lp13s5/X` with automorphisms off, which tries coverages up to 9, took 3.23 to 3.26 ns, or 2.7 to 2.8 evaluations |
| `completion_bound_minimum_layer_work` | `2 * 10**7` | A prediction of the cost of completion tables, which samples supports: 29.5 to 29.7 ms, or `2.48 * 10**7` to `2.53 * 10**7` evaluations, on layer 4 of `distance/qt7/X`, of 181 rows |
| `overlapping_minimum_work` | `2 * 10**6` | Building a family of overlapping information sets: 2.89 to 3.57 ms, or `2.44 * 10**6` to `2.99 * 10**6` evaluations, on `distance/qt7/X`, of 181 rows; the builds on the 175 rows of `distance/lp13s5/X` took 2.7 to 3.0 ms, and on the 48 rows of `distance/classical:128:80:6/-` 0.5 to 0.6 ms |
| `lp_minimum_work` | `2 * 10**8` | The linear programming of `distance/qt7/X` with automorphisms off, a search of 181 rows that solves 11 programs: 0.21 s, or `1.79 * 10**8` to `1.81 * 10**8` evaluations, for planning and programs |
| `completion_bound_max_cost_percent` | 100 | None: the limit compares two estimates in one unit |
| `automorphism_minimum_layer_work` | `5 * 10**6` | Finding automorphisms and building their images: at most 7.33 ms, on `hadamard/gross`, or about `5.6 * 10**6` evaluations at 1.32 ns, an earlier measurement of nine searches that `constants.py` does not yet repeat |
| `automorphism_minimum_states` | `5 * 10**4` | Asking for automorphisms in a connected-support search, which finds them, verifies them, and orders their orbits: 8.9 to 9.4 ms, on `csdistance/qt7/X`, or `3.4 * 10**4` to `3.7 * 10**4` states at 253 to 263 ns each, the median over six searches of 99 to 828 ns |

The row of `automorphism_minimum_states` comes from six later runs of the script, which gave the other values again.
These constants carry the sizes of the searches measured: predictions, overlapping families, and linear programming cost less on smaller bases, and a state of a connected-support search took 99 to 828 ns across the six searches.
Of the edges of their rounding, the geometric means of neighboring values, the constant of `overlapping_minimum_work` lies 5 to 23% below `3.16 * 10**6`, that of `partition_work_ratio` 20 to 22% above 3.16, that of `completion_bound_minimum_layer_work` 20 to 22% below `3.16 * 10**7`, that of `lp_minimum_work` 27 to 28% above `1.41 * 10**8`, and that of `automorphism_minimum_states` 8 to 18% above `3.16 * 10**4`.
The five coefficients of the cost model keep values measured on the cost data set before `constants.py` measured them, and no run of the script has measured them yet.

## Workflow

1. **Measure the constants.**
   Run `python experiments/distance_tuning/constants.py` on a quiet machine, under a guard of time and memory.
   Its quick measurements take a few minutes, after which it prints the derived values; its traces of the cost data set then take hours.
   It saves each measurement with the commit and the uncommitted changes of `src/` that it ran on, so it measures every constant again after a commit or a change to `src/`, and an interrupted run resumes where it stopped.
2. **Re-derive the values.**
   The script prints each derived value beside its default, and each coefficient of the cost model of completion tables, before rounding, beside its default.
   Round the coefficients, and accept them, as the Retuning section says.
   The values that differ from their defaults, with the defaults of the other fields, make the proposed tuning.
3. **Trace which decisions move.**
   Trace key searches under the current and the proposed tuning, and compare the traces.
   In these examples, the proposed tuning holds the earlier value of `automorphism_minimum_states`, `10**5`, so they repeat the confirmation of its derived value with the arms swapped:

   ```bash
   python experiments/distance_tuning/trace.py csdistance/lp13s5/X --output current.json
   python experiments/distance_tuning/trace.py csdistance/lp13s5/X \
       --tuning '{"automorphism_minimum_states": 100000}' --output proposed.json
   python experiments/distance_tuning/trace.py --compare current.json proposed.json
   ```

   If the tunings differ only in thresholds, a search whose decisions do not change takes the same steps, apart from the time of the decisions themselves.
   The docstring of `trace.py` says what the comparison leaves out, such as the sets of each family.
4. **Confirm by timing the whole tuning.**
   Time the workloads whose decisions move, and controls, under the whole current and the whole proposed tuning:

   ```bash
   python experiments/distance_tuning/sweep.py --tag automorphism-states --rounds 3 \
       --workloads csdistance/lp13s5/X csdistance/surface:17/X csdistance/qt7/X \
       --proposed '{"automorphism_minimum_states": 100000}'
   ```

   With `--worktrees CURRENT PROPOSED` instead of `--proposed`, it compares the defaults of two checkouts.

## Criteria for changing a derived value

- **Change rules or constants, never one value.**
  A value changes only because its rule, a measured constant, or a stated budget changed, and a changed rule applies to every field of its class.
- **Record predictions before measuring.**
  Record the rule, the values that it gives, the decisions that it should change, and how large a difference in time must be to matter, in absolute terms.
- **Trace decisions before timing.**
  Traces show which searches a change can affect; time those, and controls.
- **Compare the whole tuning against the current one.**
  Time the whole proposed tuning against the whole current one, including quick searches and peak memory, and report every comparison made.
  A slowdown calls for changing a rule, never for adjusting one value.
- **Publish a profile for each machine.**
  On another machine, rerun `constants.py`, and publish the values that it derives as that machine's profile, rather than retuning by timing.

## Before measuring

- Time searches in new single-threaded processes, one at a time, as `sweep.py` and `constants.py` do: completion tables save less when several searches run at once.
- Do not edit `src/` while the scripts run, since each process imports it anew.
- A trial of one repetition also times work that a search caches on its code, such as the generator matrix of a classical code given by checks, which is the same in every arm: under a second for most distance workloads, most of an `mcount` trial, and up to minutes for the classical codes of 4000 and 8000 bits.
  It shrinks ratios toward 1 without reversing them; time such workloads with `--repeat 3`, whose median leaves it out.
- A code also remembers its automorphisms once a search finds them, and whether a search finds them depends on the tuning.
  Time a change of when searches ask for automorphisms, such as of `automorphism_minimum_states` or `automorphism_minimum_layer_work`, with one repetition, so that every trial pays for asking.

## Limits

- The rules were chosen while studying the workloads of this directory, and the constants are measured on a few of them, so the derived values carry the sizes of those searches.
- The estimated work of a BZ search, which `overlapping_minimum_work`, `partition_work_ratio`, and `lp_minimum_work` read, can exceed the work done many times over, so these gates break even with the estimated work, not with the work done.
- The cost model of completion tables errs in both directions, so a cost limit at break-even accepts some tables that cost more than they save.
- A gate that a search asks once, such as that of `automorphism_minimum_states`, decides on a prediction of the states of the next weight limit, which can be wrong.

These limits can cost searches time or memory, but never change an exact result.

At commit 1a2c05293 (#871), this page described the timing campaign that chose the earlier values, and recorded the commands of its sweeps.
