# Retuning the exact distance searches

This page walks through a retuning of the fields of `SearchTuning` that Brouwer–Zimmermann (BZ) searches read, like the retuning that set their current values.
The Retuning section of the module docstring of [`qldpc.codes.distance._tuning`](../../src/qldpc/codes/distance/_tuning.py) states how a value is chosen among candidates, and the comment of each field says what to measure to retune it.
The [experiments README](../README.md#distance-tuning) introduces the scripts.
Sweeps write their trials to `results/`, which Git ignores, so this page keeps the command behind each measurement that [`evidence.md`](evidence.md) cites.

## Before measuring

- Time searches in new single-threaded processes, as `sweep.py` does, and run sweeps under a guard of time and memory.
- Completion tables save less when several searches run at once, so time their fields one search at a time (`--jobs 1`), and screen other fields a few at a time.
- Record the commit: `sweep.py` keys trials by it, and by a hash of any uncommitted change to `src/`.
- Do not edit `src/` while a sweep runs, since each trial imports it anew.
- A trial of one repetition also times work that a search caches on its code, such as the generator matrix of a classical code given by checks, which is the same in every arm: under a second for most distance workloads, 5 to 8 and 29 to 39 seconds for the sparse codes of 4000 and 8000 bits, and about 50 and 370 seconds for the classical codes of 4000 and 8000 bits.
  It shrinks ratios toward 1 without reversing them; time such workloads with `--repeat 3`, whose median leaves it out.

## 1. Measure the noise floor

Run a tuning against itself, as `noise-j1`, `noise-j4`, and `noise-j10` below do, and read the 95% intervals of `collect_results.py`.
On an Apple M3 Max, ten rounds gave 95% intervals of ±1% to ±9% one search at a time, ±3.5% in the median, and ±3% to ±8% ten at a time, ±5.6% in the median; two of eleven workloads compared with themselves one at a time had intervals that excluded 1.
Differences within these intervals do not distinguish candidates.

## 2. Work through the fields in order

Earlier fields change what later fields see, so take them in this order:

1. Batches and interrupts: `information_set_batch_exponent`, `scan_chunk_size`, `scan_chunks_per_call`.
   They change only overhead, so they can be settled first.
2. Enumeration: `enumeration_max_dimension`, `enumeration_setup_cost_factor`, `enumeration_batch_exponent`.
   They decide whether a search builds information sets at all.
3. Families of information sets: `max_information_sets`, `disjoint_random_column_orders`, `overlapping_minimum_work`, `overlapping_attempts_per_set`, and the partitioning fields.
   The family fixes the sets that every later stage scans.
4. Linear programming and automorphisms: `lp_minimum_work`, `automorphism_minimum_layer_work`.
   A search that switches to automorphism images discards its family, so these fields change which layers the next ones see.
5. Completion tables: first the threshold `completion_bound_minimum_layer_work`, which decides which layers consider tables, then the cost model (`fit_completion_costs.py`, one search at a time), the limit, the sampling, and the widths.
6. Coset searches: `coset_completion_block_bits`, `max_css_single_pairs`, and the shared family fields, on coset workloads.
7. Caches, with `cache_stats.py`.

Before timing a stage, trace a few searches with `trace.py` to see which decisions its fields can change; a threshold that no workload reaches cannot be tuned on them.
Then, as in section 5, check how much the values matter on long runs.

## 3. Choose workloads that exercise each field

| Fields | Workloads | Why |
| --- | --- | --- |
| Batches | `distance/pp232/X`, `distance/qt7/Z`, `count/gross/12`, `count/pp110/12`, `ccount/...` | Long scans; `ccount` compiles the loop that `scan_chunks_per_call` bounds |
| Enumeration | `distance/classical:<n>:<checks>:2/-`, `distance/sparse:<n>:<k>:<w>:2/-` with 16 to 23 rows and 40 to 8000 bits | Only small bases enumerate; random codes have high distance, sparse ones low |
| Families and partitioning | `distance/classical:128:80:1/-`, `distance/classical:140:90:1/-`, and quantum workloads with `automorphism_minimum_layer_work` at `MAX_VALUE` | Families matter only where automorphisms do not take over, and classical codes have none |
| Automorphisms | `distance/qt7/X`, `distance/qt7/Z`, `distance/lp13s5/X`, `distance/pp232/X` | Codes whose searches switch |
| Completion tables | `distance/qt7/X`, `distance/pp232/X`, `distance/lp13s5/X`, `distance/classical:150:100:1/-`, `count/pp110/12`; `distance/bb288/X` and `distance/pp170/X` for long searches | Large layers of quantum codes, layers of few rows, and steps of counts |
| Coset searches | `coset/...`, `first/...`, `hcoset/...`, `hbound/...`; `coset/pp232/css/10/10/3000` and `coset/qt7/css/10/10/3000` for long searches | The costliest queries use tables; `hbound` on states that are not CSS reads `max_css_single_pairs` |

Several fields are shared by distance searches, counts, and coset searches, so check a candidate on all three.

## 4. Choose values, and verify mechanisms

Choose each value by the rule of the Retuning section, which favors long runs, and report effects of running several searches at once without letting them decide a value.
A mechanism justifies a choice only once it is verified:

- Before each trace or measurement, write down the prediction, and the result that would refute it.
- Confirm a mechanism by an intervention: change the quantity that it names, and check that the outcome moves as predicted.
- Cite trace fields, such as stage times, decisions, and counts, and, for claims about the code, its file and line.
- Label each claim verified, consistent (it fits the data, untested), or conjecture, and state only verified mechanisms in field comments.

For example, the factor `enumeration_setup_cost_factor` stands for the time to set up information sets, divided by the numbers of rows and columns and by the time per enumerated codeword, but that ratio was 0.35 to 97 across codes.
The cause, a sizing of enumeration batches that does not vectorize rows of more than 960 bits, was found in `src/qldpc/codes/distance/_brute_force.py` and confirmed by raising `enumeration_batch_exponent` from 15 to 30, which made each enumerated codeword about 325 times faster on rows of 1000 bits.

## 5. Check the sensitivity of long runs

The long runs, of a minute or more one search at a time, are `distance/bb288/X`, `distance/pp170/X`, `hadamard/gross`, `hadamard/pp110`, `distance/classical:160:110:1/-`, `distance/classical:1000:976:2/-`, the classical codes of 4000 and 8000 bits with 21 to 23 rows, the sparse codes of 4000 bits with 21 to 23 rows when they enumerate, and the coset searches `coset/pp232/css/10/10/3000`, `coset/qt7/css/10/10/3000`, and `coset/lp13s5/physical/7/7/1000`.
The check `c1-gate` (section 7) found two more, `distance/lp13s6/X` and `distance/lp13s6/Z`, which the windows do not cover.
The classical codes of 4000 and 8000 bits spend about 50 and 370 seconds of that in work that does not depend on the tuning (see "Before measuring").
Coset searches read none of the thresholds that `windows.py` covers but `overlapping_minimum_work`, whose families on coset searches `coset_families.py` prints; `resample.py` shows whether a layer near the cost limit decides by its sampling draw.

Trace the long runs, and a set of quick ones, at the defaults, and give the traces to `windows.py`, which prints, for each threshold, the nearest decision on either side of it, for the workloads named by `--long` and for all.
A threshold whose window on long runs is wide, or whose candidates measured alike on long runs, is robust there, and its precise value matters only for quick runs.
A narrow window, or an effect of more than 1.2 times on a long run, makes it sensitive; trace the long runs just past each edge to see what changes.
To compare two tunings on a long run without timing it, trace it under both and run `windows.py --compare` on the two traces: if the tunings differ only in thresholds and every decision is the same, so is the search, apart from the time of the decisions themselves.
Traces record the number of sets of each family, not the sets, so `--compare` may not tell apart searches whose families differ only in their sets, such as families that `max_partition_coverage` changes.
Time a long run, one search at a time and in 4 or more paired rounds, only where a decision differs.
`trace.py` cannot trace `ccount` workloads, whose compiled setup bypasses its wrappers; trace the same counts as `count` workloads.

## 6. Guard against over-fitting

A value over-fits when it suits the workloads, the machine, and the conditions that chose it, but not others.
This section says what guards the values against over-fitting, how far each guard reaches, and which values are most at risk.
[`evidence.md`](evidence.md) gives the measurements behind each value.

### Guards, and how far they reach

- **Windows on long runs.**
  Within the window of a threshold (section 5), every decision of the traced long runs stays the same, so their time depends on the precise value only through the cost of making those decisions.
  The windows are wide for the product of `partition_work_ratio` and `partition_cost_factor` (33 to `1.4 * 10**5`), for `lp_minimum_work` (every value up to `1.17 * 10**13`), for whether `completion_bound_minimum_layer_work` lets a layer use tables (`10**6` to `5.9 * 10**8` supports), and for `overlapping_minimum_work` on distance searches.
  There, quick runs placed the value, and a wrong value costs quick runs only.
  The windows of `enumeration_setup_cost_factor`, `automorphism_minimum_layer_work`, `completion_bound_max_cost_percent`, and `overlapping_minimum_work` on coset searches are narrow (see "Values most at risk").
  No window covers `distance/lp13s6/X` and `distance/lp13s6/Z`, two long runs found after the windows were traced (section 5).
- **Round values inside windows.**
  Where a window holds several candidates that make the same decisions, the value is a round number inside it, not the candidate that happened to time fastest.
  For example, `overlapping_minimum_work` is `2 * 10**4`, inside the window from 16,473 to 34,300 in which every traced coset search builds the same families.
- **Mechanisms checked by intervention.**
  Where a value rests on a mechanism, a trace or the code shows it, and changing the quantity that it names moved the outcome as the mechanism implies.
  But most mechanisms were proposed after the sweeps whose effects they explain.
  The last checks, the sweeps of section 7 from `c2-coverage` on, had their predictions written down first.
  Of those predictions, five were refuted, in whole or in part, by their written criteria, one by the families that it predicted, and one only in its wording; `evidence.md` reports what was found instead.
- **Coefficients measured directly.**
  `partition_cost_factor`, the build costs of completion tables, and `completion_bound_kept_first_row_ratio` agree with direct measurements of the quantities that they stand for, rather than with fits to the total times of searches.
  But the measurements spread.
  Partitioning took 4.3 to 10.5 of the units of `partition_cost_factor`, whose value, 10, lies near the top.
  The build costs fit blocks of 16 columns on quantum codes, not blocks of 8 columns or classical codes.
  The kept ratio ranged from 0.04 to 0.50.
- **A noise floor.**
  Speed does not choose between candidates that measured alike (see the Retuning section, and section 1 for the size of the noise).
  But candidates that make the same decisions on every timed workload measure alike by construction; for those, the windows locate the value.
  Of the 817 comparisons with 95% intervals, about 13 would exclude 1 by chance if their rounds were independent and their candidates equal, and of the 49 compared by their ranges, about 10.
  But 2 of the 33 comparisons of a tuning with itself (section 1) excluded 1, so a candidate set aside by one small difference may have been set aside by chance.
  Rechecked one search at a time, 12 for `information_set_batch_exponent` and 16,384 for `scan_chunk_size` were slower again, and 512 for `completion_bound_samples` measured alike.
- **Checks on workloads that chose nothing.**
  `certify/gross/css/2` and `faultdist/toric:6/Z`, which call the APIs of fault-propagation certificates, chose no value.
  Held out for `overlapping_minimum_work`, they measured alike at `10**4`, `2 * 10**4`, and `10**5` (`c8-usecase`), so the value does them no harm, though they cannot show that it is best.
  The values of the five fields that the last decisions set were timed together, against their values before, on 18 workloads (`c9-decided`), with predictions written first.
  No result differed, and the slowdowns that they found, of two quick searches, are explained in `evidence.md`; one of them led to the cost limit of 90, which was then timed only on those two.

### Values most at risk

These values lie near an edge of a window, rest on a few workloads, or trade one kind of search against another, roughly in order of risk:

- **The partition gate.**
  A search partitions only if its gate value, its estimated work in units of the cost of partitioning, reaches the product of `partition_work_ratio` and `partition_cost_factor`, 5000.
  Partitioning slows searches that switch to automorphism images, such as those of `lp13s<seed>` codes, about 3 to 4 times, and speeds random classical codes up to 1.7 times, but their gate values overlap, so no product separates them.
  5000 lies in the window, from 3022 to 7227, that misclassifies the fewest traced searches, but the gate values of codes with the same parameters scatter from 0 to 24,375, so where they fall is partly chance.
  Deciding on automorphisms before partitioning, a follow-up, would remove the need to fit this product.
- **The width of coset tables.**
  16-column tables make CSS coset searches 2.8 to 29 times faster, and add about 1.5 GiB of memory to the long ones.
  But they make two of three searches of Pauli errors by physical weight 2.7 to 4 times slower, and, with `overlapping_minimum_work` at `2 * 10**4`, a quick search of the moderate code `pp110` 63 milliseconds slower.
  On residual-weight estimates of state-preparation circuits, not in this repository, 16-column tables took longer to build than most estimates spent scanning.
  A width for each kind of search is a follow-up.
- **`overlapping_minimum_work` on coset searches.**
  Its window, from 16,473 to 34,300, comes from the costs of queries of weights 2 and 3 on four codes: the lower edge from `qt7`, and the upper edge from `pp110`.
  Other codes have other edges, so the value suits codes like these, not every code.
  Below the window, the long coset searches took 2.1 to 3.2 times as long; above it, two quick ones 1.5 to 1.75 times as long, four searches at a time.
- **The fields of enumeration.**
  Their measurements were taken while enumeration handled rows of more than 960 bits one codeword at a time (section 4), so their best values will change once that is fixed.
  The window of `enumeration_setup_cost_factor` on long runs is 12.5 to 23.8, and on all traced workloads the window of `enumeration_max_dimension` is 20 alone.
  `allow_enumeration_by_work` is a switch for tests: with it off, `distance/classical:1000:976:2/-` took 1.6 seconds instead of 70.6.
- **The cost limit, and the sampling and costs of completion tables.**
  With 1024 samples, the predictions of the traced long-run layers near the limit leave a gap from 86.1 to 93.9 percent under 40 sampling seeds, and the limit, 90, lies in it, so none of those layers decides by its draw.
  But five layers make that gap, and the cost model errs both ways, by more than two times on some traced layers.
  100, chosen first as a break-even point, made a quick search 1.19 times slower, since a layer of several sets compares the tables of each set with a scan of that set alone, not with their scan in lockstep.
  The lookup costs were kept after one run, and the measured costs of the same lookups varied two- to threefold between sessions.
- **`automorphism_minimum_layer_work`.**
  Its window on long runs is `8.5 * 10**7` to `1.8 * 10**8`, but just past either edge a long search switches to automorphism images one layer sooner or later, a layer of at most 0.22 seconds.
- **`max_css_single_pairs`.**
  It is the smallest power of two that fits the one workload that reads it on a state of more than 255 qubits.

### Other limits

- **One machine, under load.**
  Every timing comes from one Apple M3 Max.
  Most sweeps ran three to eight searches at a time, and the sweeps of one search at a time, including the last checks, ran beside other work, at load averages of up to 19.
- **Older library code.**
  Most sweeps ran on earlier library code, at the commits that `trials.jsonl` records.
  The windows, the refit data set, and the last checks used the current library code.
  The comparisons of the sampling fields and of the block widths, and the comparison of the cost limit three searches at a time, did not.
- **Fields not timed on long runs.**
  `max_information_sets`, `disjoint_random_column_orders`, `overlapping_attempts_per_set`, `max_partition_coverage`, `max_css_single_pairs`, the batch and chunk sizes, the block widths of distance tables, `completion_bound_max_table_bytes`, and the coefficients of the cost model were timed only on runs of seconds, apart from one run of `hadamard/gross` behind the lookup costs.
  The cache sizes, and the fields whose comments say they were chosen by hand, were not chosen by these measurements.

## 7. Reproduce the measurements

The commands below ran on the library code of their time, at the commits that `trials.jsonl` records, and each compares its arms with its first arm, as its command gives it.
That arm is usually the defaults of its time, in some sweeps with `completion_bound_minimum_layer_work` at `3 * 10**7`.
In `final-short-f`, it sets five fields to the values that its command gives.
Some sweeps, such as `p3-noaut`, also set a field in every arm.
The `--base` of each command gives the defaults of its time where they differ from the current ones, so that each command compares the same values on the current code.
The refit data set of the Retuning section is that of the row "Cost coefficients".

Measurements of one search use `trace.py`, `trial.py`, or `fit_completion_costs.py`:

| Measurement | Command |
| --- | --- |
| Partitioning: estimates, costs, and gates | `trace.py distance/lp13s5/X --tuning '{"partition_work_ratio": 10}' --output R1.json`, the same with `"automorphism_minimum_layer_work": 4611686018427387904`, and `trace.py hadamard/gross --output R3.json`; `trace.py <workload> --setup-only` for the gates of other workloads |
| `enumeration_setup_cost_factor` | `trace.py <workload> --setup-only --tuning '{"allow_enumeration_by_size": false, "allow_enumeration_by_work": false}'` for the time of the disjoint family, and `trial.py <workload> --repeat 3 --tuning '{"enumeration_max_dimension": 62}'` for the time per word, for `distance/classical:<n>:<n - k>:2/-` and `distance/sparse:<n>:<k>:<n / 20>:2/-` with `n` of 200, 1000, and 4000 and `k` of 16 and 20 |
| `enumeration_batch_exponent` | `trial.py 'distance/classical:1000:980:2/-' --repeat 3 --tuning '{"enumeration_max_dimension": 62}'`, and with `"enumeration_batch_exponent": 30` |
| `automorphism_minimum_layer_work` at `10**7` | `trace.py distance/qt7/X --tuning '{"automorphism_minimum_layer_work": 10000000}'` |
| Sampling of `completion_bound_minimum_layer_work` | `trace.py <workload> --output <file>` at the defaults, and with automorphisms off, summing the `seconds` of the `predict` events before each `layer` event |
| Predictions of the cost model, and `completion_bound_block_bits` | `trace.py distance/classical:150:100:1/-` and `trace.py distance/pp232/X` for the predictions, and `fit_completion_costs.py collect --repeats 3 --workloads distance/pp232/X 'distance/classical:150:100:1/-'`, then `fit_completion_costs.py fit`, for the measured costs; `trace.py distance/qt7/X --tuning '{"completion_bound_block_bits": 8}'` |
| Cost coefficients | `fit_completion_costs.py collect --repeats 3 --workloads distance/lp13s5/X distance/lp13s8/X distance/qt7/X distance/qt7/Z distance/pp232/X distance/pp110/X distance/gross/X 'distance/classical:150:100:1/-' 'distance/classical:128:80:1/-'`, then `fit_completion_costs.py fit --min-supports 1e8` |
| Windows of the thresholds | the commands below the table |
| Past the edges of the windows | `trace.py <long> --tuning '{"completion_bound_minimum_layer_work": 1000000}'`; `trace.py hadamard/pp110 --tuning '{"automorphism_minimum_layer_work": 80000000}'`, the same for `distance/pp170/X`, and `trace.py distance/bb288/X --tuning '{"automorphism_minimum_layer_work": 200000000}'`, each with `--output` to a file outside the directory of the traces of the windows, and compared with its trace at the defaults by `windows.py --compare`; `trace.py hadamard/gross --tuning '{"completion_bound_lookup_block_cost": 0.9, "completion_bound_lookup_block_cost_per_mib": 0.05}'`, whose layer 6 declines its tables, for the time of that layer without them |
| Layer 7 of `distance/classical:128:80:6/-` at the limits of 90 and 100 | `trace.py 'distance/classical:128:80:6/-' --tuning '{"completion_bound_max_cost_percent": 100}'`, the same with `"completion_bound_samples": 256` added, and at the defaults, compared by `windows.py --compare` and by their `layer` events (`c11`) |
| Gate values of partitioning across codes | `trace.py distance/lp13s<seed>/<sector> --setup-only` for seeds 1 to 10 and sectors `X` and `Z`, and `trace.py 'distance/classical:128:80:<seed>/-' --setup-only` for seeds 2 to 6; a gate value is the `estimated_work` of the `family` event of kind `partitioned` divided by its `unit` |
| Below the window of `enumeration_setup_cost_factor` | `trace.py <w> --setup-only --tuning '{"enumeration_setup_cost_factor": 8}'` for `'distance/classical:8000:7979:2/-'` and `'distance/sparse:8000:21:400:2/-'` |
| Families of `overlapping_minimum_work` | `coset_families.py coset/pp232/css/10/10/3000`, and the same for `coset/qt7/css/10/10/3000` and `coset/lp13s5/physical/7/7/1000`; `coset_families.py <w> --thresholds 20000 24000 30000` for `coset/gross/css/10/10/3000`, `coset/pp110/css/8/8/2000`, `coset/pp232/css/10/10/3000`, and `coset/qt7/css/10/10/3000` |
| Sampling of layers near the cost limit | `resample.py hadamard/gross --weights 6` and `resample.py distance/pp170/X --weights 6 7`, and `resample.py hadamard/pp110 --weights 5` and `resample.py distance/bb288/X --weights 5`, at the defaults of their time (`c10` ran them with `--changes '[]'` at commit `dfa8944b3`, whose limit was 100, and `c12` the first at the current defaults, with the changes of `completion_bound_samples`, `completion_bound_confidence_z`, `completion_bound_kept_first_row_ratio`, and the lookup costs that `evidence.md` names); for the sampling budget, `resample.py hadamard/gross --weights 6 --num-seeds 1 --changes '[{"completion_bound_sampling_work": 25000000}, {"completion_bound_sampling_work": 100000000}]'` |
| Linear programs of `lp_minimum_work` | the `linear program` events of `trace.py <w>` at the defaults |
| Memory of `max_css_single_pairs` | `tracemalloc` around `_tabulate_css_singles` of `_brouwer_zimmermann/coset.py` on 255, 372, and 511 random single-qubit syndromes of full size |
| Cache sizes | `cache_stats.py --cycles 2 coset/gross/css/6/6/100 coset/pp110/css/6/6/100 coset/qt7/physical/5/5/50 hbound/pp232/2/100 distance/qt7/X distance/lp13s5/X count/gross/12` |

Searches without information sets: `trial.py 'distance/classical:<n>:<checks>:2/-' --repeat 2` with and without `--tuning '{"allow_enumeration_by_work": false}'`, reading the second repetition.

These commands trace the workloads behind the windows that `evidence.md` cites, at the defaults, and print the windows; the classical codes that enumerate many words are traced up to their decision to enumerate.
Keep other traces out of `traces/`, since `windows.py` requires its traces to share one tuning.

```bash
trace() { python experiments/distance_tuning/trace.py "$1" "${@:2}" --output "traces/$(echo "$1" | tr '/:' '_-').json"; }
for w in count/{gross,pp110}/12 distance/{bb288,pp170,pp232,lp13s8}/X distance/{gross,pp110}/{X,both} \
    distance/{lp13s5,qt7}/{X,Z} distance/surface:{6,7}/X distance/toric:6/X hadamard/{gross,pp110} \
    distance/classical:{128:80,130:80,140:90,150:100,160:110}:1/- distance/classical:40:{16..24}:2/- \
    distance/classical:200:{176..184}:2/- distance/classical:1000:{980..984}:2/- \
    distance/classical:4000:{3983,3984}:2/- distance/classical:8000:7984:2/- \
    distance/sparse:200:{16..22}:10:2/- distance/sparse:200:24:10:2/- distance/sparse:1000:{16..24}:50:2/- \
    distance/sparse:4000:{16,17,19,20,21,22,23}:200:2/- distance/sparse:8000:{16,20}:400:2/-; do
    trace "$w"
done
for w in distance/classical:1000:{976..979}:2/- distance/classical:4000:{3977..3981}:2/- \
    distance/classical:8000:{7977..7980}:2/-; do
    trace "$w" --setup-only
done
python experiments/distance_tuning/windows.py traces/*.json --long distance/{bb288,pp170}/X \
    hadamard/{gross,pp110} distance/classical:160:110:1/- distance/classical:1000:976:2/- \
    distance/classical:4000:{3977..3979}:2/- distance/classical:8000:{7977..7979}:2/- \
    distance/sparse:4000:{21..23}:200:2/-
```

These sweeps produced the ratios that `evidence.md` cites, and `collect_results.py --tags <tag>` summarizes each.

### noise-j10

```bash
python experiments/distance_tuning/sweep.py \
    --tag noise-j10 \
    --rounds 10 \
    --jobs 10 \
    --timeout 600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads coset/gross/physical/6/6/500 count/pp110/12 coset/pp232/physical/6/6/500 distance/lp13s5/X distance/lp13s8/X distance/qt7/Z coset/gross/css/10/10/3000 distance/qt7/X count/gross/12 distance/pp232/X first/pp232/physical/6/6/2000 \
    --arms '[{}, {}]'
```

### noise-j4

```bash
python experiments/distance_tuning/sweep.py \
    --tag noise-j4 \
    --rounds 10 \
    --jobs 4 \
    --timeout 600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/lp13s8/X distance/lp13s5/X distance/qt7/Z distance/qt7/X count/pp110/12 coset/gross/physical/6/6/500 count/gross/12 distance/pp232/X coset/pp232/physical/6/6/500 coset/gross/css/10/10/3000 first/pp232/physical/6/6/2000 \
    --arms '[{}, {}]'
```

### noise-j1

```bash
python experiments/distance_tuning/sweep.py \
    --tag noise-j1 \
    --rounds 10 \
    --jobs 1 \
    --timeout 600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/lp13s5/X distance/lp13s8/X distance/qt7/X distance/qt7/Z distance/pp232/X count/gross/12 count/pp110/12 coset/gross/physical/6/6/500 coset/pp232/physical/6/6/500 coset/gross/css/10/10/3000 first/pp232/physical/6/6/2000 \
    --arms '[{}, {}]'
```

### p1-isbatch

```bash
python experiments/distance_tuning/sweep.py \
    --tag p1-isbatch \
    --rounds 10 \
    --jobs 6 \
    --timeout 600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/lp13s5/X distance/qt7/Z distance/qt7/X distance/pp232/X \
    --arms '[{"information_set_batch_exponent": 15}, {"information_set_batch_exponent": 12}, {"information_set_batch_exponent": 18}, {"information_set_batch_exponent": 21}]'
```

### p1-chunk

```bash
python experiments/distance_tuning/sweep.py \
    --tag p1-chunk \
    --rounds 10 \
    --jobs 6 \
    --timeout 600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads count/pp110/12 count/gross/12 ccount/pp110/12 ccount/gross/12 \
    --arms '[{"scan_chunk_size": 65536}, {"scan_chunk_size": 4096}, {"scan_chunk_size": 1048576}]'
```

### p1-chunks-per-call

```bash
python experiments/distance_tuning/sweep.py \
    --tag p1-chunks-per-call \
    --rounds 10 \
    --jobs 6 \
    --timeout 600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads ccount/pp110/12 ccount/gross/12 \
    --arms '[{"scan_chunks_per_call": 64}, {"scan_chunks_per_call": 4}, {"scan_chunks_per_call": 1024}]'
```

### p1-isbatch-fine

```bash
python experiments/distance_tuning/sweep.py \
    --tag p1-isbatch-fine \
    --rounds 10 \
    --jobs 1 \
    --timeout 600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/pp232/X distance/qt7/Z \
    --arms '[{"information_set_batch_exponent": 15}, {"information_set_batch_exponent": 13}, {"information_set_batch_exponent": 14}, {"information_set_batch_exponent": 16}, {"information_set_batch_exponent": 17}]'
```

### p1-chunk-fine

```bash
python experiments/distance_tuning/sweep.py \
    --tag p1-chunk-fine \
    --rounds 10 \
    --jobs 1 \
    --timeout 600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads count/gross/12 count/pp110/12 \
    --arms '[{"scan_chunk_size": 65536}, {"scan_chunk_size": 16384}, {"scan_chunk_size": 32768}, {"scan_chunk_size": 131072}]'
```

### p2-enum

```bash
python experiments/distance_tuning/sweep.py \
    --tag p2-enum \
    --rounds 6 \
    --jobs 8 \
    --repeat 3 \
    --timeout 1800 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/classical:40:24:2/- distance/classical:40:22:2/- distance/classical:40:20:2/- distance/classical:40:16:2/- distance/classical:200:184:2/- distance/classical:40:18:2/- distance/sparse:200:16:10:2/- distance/classical:200:182:2/- distance/sparse:200:24:10:2/- distance/sparse:200:18:10:2/- distance/sparse:200:22:10:2/- distance/classical:200:180:2/- distance/sparse:200:20:10:2/- distance/classical:200:178:2/- distance/classical:200:176:2/- distance/classical:1000:984:2/- distance/sparse:1000:24:50:2/- distance/sparse:1000:16:50:2/- distance/sparse:4000:16:200:2/- distance/sparse:1000:20:50:2/- distance/classical:1000:980:2/- distance/surface:6/X distance/sparse:4000:20:200:2/- distance/toric:6/X distance/surface:7/X distance/classical:4000:3984:2/- distance/classical:4000:3980:2/- distance/classical:1000:976:2/- \
    --arms '[{}, {"enumeration_max_dimension": 30}, {"allow_enumeration_by_size": false, "allow_enumeration_by_work": false}]'
```

### p2-confirm

```bash
python experiments/distance_tuning/sweep.py \
    --tag p2-confirm \
    --rounds 6 \
    --jobs 8 \
    --repeat 3 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/classical:40:23:2/- distance/classical:40:22:2/- distance/classical:200:182:2/- distance/sparse:200:17:10:2/- distance/classical:200:183:2/- distance/sparse:200:18:10:2/- distance/sparse:1000:17:50:2/- distance/classical:1000:983:2/- distance/classical:200:181:2/- distance/classical:40:21:2/- distance/sparse:200:19:10:2/- distance/sparse:1000:18:50:2/- distance/classical:1000:982:2/- distance/classical:1000:981:2/- distance/classical:40:20:2/- distance/sparse:200:20:10:2/- distance/classical:200:180:2/- distance/sparse:1000:19:50:2/- distance/classical:40:19:2/- distance/classical:200:179:2/- distance/sparse:200:21:10:2/- distance/classical:1000:980:2/- distance/sparse:1000:20:50:2/- distance/sparse:4000:17:200:2/- distance/sparse:1000:21:50:2/- distance/classical:1000:979:2/- distance/sparse:4000:19:200:2/- distance/classical:4000:3983:2/- distance/classical:4000:3981:2/- distance/sparse:4000:20:200:2/- distance/classical:4000:3980:2/- distance/surface:6/X distance/toric:6/X distance/surface:7/X distance/classical:8000:7984:2/- distance/classical:8000:7980:2/- \
    --arms '[{}, {"enumeration_max_dimension": 18}, {"enumeration_max_dimension": 18, "enumeration_setup_cost_factor": 16}, {"enumeration_max_dimension": 18, "enumeration_setup_cost_factor": 4}]'
```

Its two workloads of 8000 bits, about 7.5 GiB each, ran two at a time, in a second sweep:

```bash
python experiments/distance_tuning/sweep.py \
    --tag p2-confirm \
    --rounds 6 \
    --jobs 2 \
    --repeat 3 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/sparse:8000:16:400:2/- distance/sparse:8000:20:400:2/- \
    --arms '[{}, {"enumeration_max_dimension": 18}, {"enumeration_max_dimension": 18, "enumeration_setup_cost_factor": 16}, {"enumeration_max_dimension": 18, "enumeration_setup_cost_factor": 4}]'
```

### p5-screen

```bash
python experiments/distance_tuning/sweep.py \
    --tag p5-screen \
    --rounds 8 \
    --jobs 3 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/lp13s5/X distance/qt7/Z distance/qt7/X distance/pp232/X distance/classical:128:80:1/- distance/classical:150:100:1/- count/gross/12 count/pp110/12 \
    --arms '[{}, {"completion_bound_max_table_bytes": 0}, {"completion_bound_minimum_layer_work": 0, "completion_bound_max_cost_percent": 4611686018427387904}, {"completion_bound_block_bits": 8}, {"completion_bound_block_bits": 4}]'
```

### p5-gates

```bash
python experiments/distance_tuning/sweep.py \
    --tag p5-gates \
    --rounds 8 \
    --jobs 3 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads count/pp110/12 distance/qt7/X count/gross/12 distance/qt7/Z distance/lp13s5/X distance/pp232/X distance/classical:150:100:1/- \
    --arms '[{}, {"completion_bound_minimum_layer_work": 10000000}, {"completion_bound_minimum_layer_work": 30000000}, {"completion_bound_max_cost_percent": 4611686018427387904}, {"completion_bound_max_cost_percent": 100}]'
```

### p5-load-j1

```bash
python experiments/distance_tuning/sweep.py \
    --tag p5-load-j1 \
    --rounds 8 \
    --jobs 1 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/classical:150:100:1/- \
    --arms '[{}, {"completion_bound_max_cost_percent": 4611686018427387904}]'
```

### p5-load-j6

```bash
python experiments/distance_tuning/sweep.py \
    --tag p5-load-j6 \
    --rounds 12 \
    --jobs 6 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/classical:150:100:1/- \
    --arms '[{}, {"completion_bound_max_cost_percent": 4611686018427387904}]'
```

### p5-isolate-j1

```bash
python experiments/distance_tuning/sweep.py \
    --tag p5-isolate-j1 \
    --rounds 8 \
    --jobs 1 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "max_css_single_pairs": 65536, "completion_bound_minimum_layer_work": 100000000, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads count/pp110/12 count/gross/12 distance/classical:128:80:1/- distance/classical:150:100:1/- distance/pp232/X distance/qt7/X distance/qt7/Z distance/lp13s5/X \
    --arms '[{"completion_bound_minimum_layer_work": 30000000}, {"completion_bound_minimum_layer_work": 30000000, "completion_bound_lookup_block_cost": 0.9, "completion_bound_lookup_block_cost_per_mib": 0.05}, {}]'
```

### p5-sampling

```bash
python experiments/distance_tuning/sweep.py \
    --tag p5-sampling \
    --rounds 8 \
    --jobs 3 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/classical:128:80:1/- count/pp110/12 count/gross/12 distance/qt7/X distance/pp232/X distance/lp13s5/X \
    --arms '[{"completion_bound_minimum_layer_work": 30000000}, {"completion_bound_minimum_layer_work": 30000000, "completion_bound_samples": 128}, {"completion_bound_minimum_layer_work": 30000000, "completion_bound_samples": 512}, {"completion_bound_minimum_layer_work": 30000000, "completion_bound_confidence_z": 1.0}, {"completion_bound_minimum_layer_work": 30000000, "completion_bound_confidence_z": 3.0}]'
```

### p3-aut

```bash
python experiments/distance_tuning/sweep.py \
    --tag p3-aut \
    --rounds 8 \
    --jobs 6 \
    --timeout 1800 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/classical:130:80:1/- distance/classical:128:80:1/- distance/lp13s5/X distance/qt7/Z distance/qt7/X distance/classical:140:90:1/- coset/gross/css/10/10/3000 distance/pp232/X coset/gross/physical/6/6/500 coset/pp232/physical/6/6/500 first/pp232/physical/6/6/2000 \
    --arms '[{"completion_bound_minimum_layer_work": 30000000}, {"completion_bound_minimum_layer_work": 30000000, "max_information_sets": 8}, {"completion_bound_minimum_layer_work": 30000000, "max_information_sets": 32}, {"completion_bound_minimum_layer_work": 30000000, "overlapping_attempts_per_set": 1}, {"completion_bound_minimum_layer_work": 30000000, "overlapping_attempts_per_set": 4}, {"completion_bound_minimum_layer_work": 30000000, "partition_work_ratio": 100}, {"completion_bound_minimum_layer_work": 30000000, "partition_cost_factor": 80}, {"completion_bound_minimum_layer_work": 30000000, "max_partition_coverage": 2}, {"completion_bound_minimum_layer_work": 30000000, "disjoint_random_column_orders": 0}, {"completion_bound_minimum_layer_work": 30000000, "disjoint_random_column_orders": 32}, {"completion_bound_minimum_layer_work": 30000000, "overlapping_minimum_work": 10000}, {"completion_bound_minimum_layer_work": 30000000, "overlapping_minimum_work": 1000000}, {"completion_bound_minimum_layer_work": 30000000, "lp_minimum_work": 100000000}, {"completion_bound_minimum_layer_work": 30000000, "lp_minimum_work": 10000000000}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 10000000}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 1000000000}]'
```

### p3-noaut

```bash
python experiments/distance_tuning/sweep.py \
    --tag p3-noaut \
    --rounds 8 \
    --jobs 6 \
    --timeout 1800 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/gross/X distance/lp13s5/X distance/qt7/X \
    --arms '[{"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904, "max_information_sets": 8}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904, "max_information_sets": 32}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904, "overlapping_attempts_per_set": 1}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904, "overlapping_attempts_per_set": 4}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904, "partition_work_ratio": 100}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904, "partition_cost_factor": 80}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904, "max_partition_coverage": 2}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904, "disjoint_random_column_orders": 0}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904, "disjoint_random_column_orders": 32}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904, "overlapping_minimum_work": 10000}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904, "overlapping_minimum_work": 1000000}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904, "lp_minimum_work": 100000000}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 4611686018427387904, "lp_minimum_work": 10000000000}]'
```

### p3-orders-aut

```bash
python experiments/distance_tuning/sweep.py \
    --tag p3-orders-aut \
    --rounds 8 \
    --jobs 6 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads hbound/pp232/2/2000 coset/gross/physical/6/6/500 coset/pp232/physical/6/6/500 coset/gross/css/10/10/3000 hcoset/gross/css/4/4/500 first/pp232/physical/6/6/2000 distance/qt7/X distance/qt7/Z distance/lp13s5/X distance/pp232/X distance/classical:140:90:1/- \
    --arms '[{"completion_bound_minimum_layer_work": 30000000}, {"completion_bound_minimum_layer_work": 30000000, "disjoint_random_column_orders": 16}, {"completion_bound_minimum_layer_work": 30000000, "disjoint_random_column_orders": 32}, {"completion_bound_minimum_layer_work": 30000000, "disjoint_random_column_orders": 64}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 30000000}, {"completion_bound_minimum_layer_work": 30000000, "automorphism_minimum_layer_work": 300000000}]'
```

### p3-partition

```bash
python experiments/distance_tuning/sweep.py \
    --tag p3-partition \
    --rounds 4 \
    --jobs 4 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/lp13s5/X distance/qt7/X hadamard/gross hadamard/pp110 \
    --arms '[{"completion_bound_minimum_layer_work": 30000000}, {"completion_bound_minimum_layer_work": 30000000, "partition_work_ratio": 30}, {"completion_bound_minimum_layer_work": 30000000, "partition_work_ratio": 100}, {"completion_bound_minimum_layer_work": 30000000, "partition_work_ratio": 4611686018427387904}]'
```

### p6-short

```bash
python experiments/distance_tuning/sweep.py \
    --tag p6-short \
    --rounds 8 \
    --jobs 4 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads coset/pp110/css/8/8/2000 coset/gross/physical/6/6/500 coset/pp232/physical/6/6/500 coset/gross/css/10/10/3000 first/pp232/physical/6/6/2000 hcoset/gross/css/4/4/500 hbound/pp232/2/2000 hbound/pp372/2/2000 distance/qt7/X distance/pp232/X \
    --arms '[{"completion_bound_minimum_layer_work": 30000000}, {"completion_bound_minimum_layer_work": 30000000, "overlapping_minimum_work": 30000}, {"completion_bound_minimum_layer_work": 30000000, "overlapping_minimum_work": 10000}, {"completion_bound_minimum_layer_work": 30000000, "overlapping_minimum_work": 1000}, {"completion_bound_minimum_layer_work": 30000000, "coset_completion_block_bits": 4}, {"completion_bound_minimum_layer_work": 30000000, "coset_completion_block_bits": 16}, {"completion_bound_minimum_layer_work": 30000000, "max_css_single_pairs": 262144}]'
```

### p6-long

```bash
python experiments/distance_tuning/sweep.py \
    --tag p6-long \
    --rounds 4 \
    --jobs 4 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads coset/pp232/css/10/10/3000 coset/lp13s5/physical/7/7/1000 coset/qt7/css/10/10/3000 \
    --arms '[{"completion_bound_minimum_layer_work": 30000000}, {"completion_bound_minimum_layer_work": 30000000, "overlapping_minimum_work": 10000}, {"completion_bound_minimum_layer_work": 30000000, "overlapping_minimum_work": 1000}, {"completion_bound_minimum_layer_work": 30000000, "coset_completion_block_bits": 4}, {"completion_bound_minimum_layer_work": 30000000, "coset_completion_block_bits": 16}]'
```

### p2-factor

```bash
python experiments/distance_tuning/sweep.py \
    --tag p2-factor \
    --rounds 6 \
    --jobs 4 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/sparse:1000:21:50:2/- distance/classical:1000:979:2/- distance/sparse:4000:21:200:2/- distance/sparse:1000:22:50:2/- distance/classical:1000:978:2/- distance/classical:4000:3979:2/- distance/sparse:4000:22:200:2/- distance/classical:4000:3978:2/- distance/classical:1000:977:2/- distance/sparse:1000:23:50:2/- distance/sparse:4000:23:200:2/- distance/classical:4000:3977:2/- distance/classical:8000:7979:2/- distance/classical:8000:7978:2/- distance/classical:8000:7977:2/- \
    --arms '[{}, {"enumeration_setup_cost_factor": 16}]'
```

### p5-costlimit-j1

```bash
python experiments/distance_tuning/sweep.py \
    --tag p5-costlimit-j1 \
    --rounds 6 \
    --jobs 1 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/classical:128:80:1/- distance/classical:140:90:1/- distance/classical:150:100:1/- distance/classical:160:110:1/- distance/qt7/X distance/qt7/Z distance/pp232/X distance/lp13s5/X count/pp110/12 count/gross/12 \
    --arms '[{"completion_bound_minimum_layer_work": 30000000}, {"completion_bound_minimum_layer_work": 30000000, "completion_bound_max_cost_percent": 150}, {"completion_bound_minimum_layer_work": 30000000, "completion_bound_max_cost_percent": 200}]'
```

### accept-isolate

```bash
python experiments/distance_tuning/sweep.py \
    --tag accept-isolate \
    --rounds 8 \
    --jobs 3 \
    --timeout 3600 \
    --base '{"enumeration_setup_cost_factor": 128, "partition_work_ratio": 10, "partition_cost_factor": 20, "completion_bound_minimum_layer_work": 100000000, "max_css_single_pairs": 65536, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000}' \
    --workloads distance/classical:128:80:1/- distance/classical:140:90:1/- distance/classical:150:100:1/- \
    --arms '[{}, {"completion_bound_minimum_layer_work": 30000000}, {"completion_bound_minimum_layer_work": 30000000, "completion_bound_max_cost_percent": 200}]'
```

### final-short-f

```bash
python experiments/distance_tuning/sweep.py \
    --tag final-short-f \
    --rounds 8 \
    --jobs 3 \
    --timeout 3600 \
    --base '{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads distance/classical:128:80:1/- distance/classical:140:90:1/- distance/classical:150:100:1/- distance/sparse:1000:21:50:2/- distance/lp13s5/X distance/lp13s8/X distance/qt7/Z distance/qt7/X distance/pp232/X count/pp110/12 count/gross/12 coset/gross/css/10/10/3000 ccount/pp110/12 coset/pp232/physical/6/6/500 hbound/pp372/2/2000 certify/gross/css/2 faultdist/toric:6/Z \
    --arms '[{"completion_bound_minimum_layer_work": 100000000, "partition_work_ratio": 10, "partition_cost_factor": 20, "enumeration_setup_cost_factor": 128, "max_css_single_pairs": 65536}, {}]'
```

### v-maxdim

```bash
python experiments/distance_tuning/sweep.py \
    --tag v-maxdim \
    --rounds 8 \
    --jobs 1 \
    --repeat 3 \
    --timeout 3600 \
    --base '{"partition_cost_factor": 20, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads distance/classical:200:180:2/- distance/sparse:200:19:10:2/- \
    --arms '[{}, {"enumeration_max_dimension": 18}, {"enumeration_max_dimension": 18, "allow_enumeration_by_work": false}]'
```

### v-partition

```bash
python experiments/distance_tuning/sweep.py \
    --tag v-partition \
    --rounds 10 \
    --jobs 1 \
    --timeout 3600 \
    --base '{"partition_cost_factor": 20, "completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads distance/lp13s8/X distance/lp13s5/X distance/classical:128:80:1/- distance/classical:140:90:1/- \
    --arms '[{}, {"partition_work_ratio": 4611686018427387904}]'
```

### costlimit-j1-f

```bash
python experiments/distance_tuning/sweep.py \
    --tag costlimit-j1-f \
    --rounds 4 \
    --jobs 1 \
    --timeout 1800 \
    --base '{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads distance/classical:160:110:1/- distance/classical:150:100:1/- distance/classical:140:90:1/- \
    --arms '[{}, {"completion_bound_max_cost_percent": 150}, {"completion_bound_max_cost_percent": 200}]'
```

### costlimit-quick-f

```bash
python experiments/distance_tuning/sweep.py \
    --tag costlimit-quick-f \
    --rounds 4 \
    --jobs 1 \
    --timeout 600 \
    --base '{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads distance/lp13s5/X distance/lp13s5/Z distance/lp13s8/X distance/qt7/X distance/qt7/Z \
    --arms '[{}, {"completion_bound_max_cost_percent": 200}]'
```

### c2-coverage

```bash
python experiments/distance_tuning/sweep.py \
    --tag c2-coverage \
    --rounds 10 \
    --jobs 1 \
    --timeout 600 \
    --base '{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads distance/lp13s5/X distance/lp13s8/X distance/classical:128:80:1/- distance/classical:140:90:1/- \
    --arms '[{}, {"max_partition_coverage": 3}, {"max_partition_coverage": 2}]'
```

### c5-isbatch

```bash
python experiments/distance_tuning/sweep.py \
    --tag c5-isbatch \
    --rounds 10 \
    --jobs 1 \
    --timeout 600 \
    --base '{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads distance/pp232/X distance/qt7/Z \
    --arms '[{}, {"information_set_batch_exponent": 11}, {"information_set_batch_exponent": 12}, {"information_set_batch_exponent": 13}, {"information_set_batch_exponent": 14}]'
```

### c5-isbatch-high

```bash
python experiments/distance_tuning/sweep.py \
    --tag c5-isbatch-high \
    --rounds 10 \
    --jobs 1 \
    --timeout 600 \
    --base '{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads distance/pp232/X distance/qt7/Z \
    --arms '[{}, {"information_set_batch_exponent": 16}, {"information_set_batch_exponent": 17}, {"information_set_batch_exponent": 18}, {"information_set_batch_exponent": 20}]'
```

### c5-chunk

```bash
python experiments/distance_tuning/sweep.py \
    --tag c5-chunk \
    --rounds 10 \
    --jobs 1 \
    --timeout 600 \
    --base '{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads count/gross/12 count/pp110/12 \
    --arms '[{}, {"scan_chunk_size": 16384}, {"scan_chunk_size": 32768}]'
```

### c1-gate

```bash
python experiments/distance_tuning/sweep.py \
    --tag c1-gate \
    --rounds 6 \
    --jobs 1 \
    --timeout 600 \
    --base '{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads distance/lp13s1/X distance/lp13s6/X distance/lp13s6/Z distance/lp13s8/X distance/classical:128:80:2/- distance/classical:128:80:3/- distance/classical:128:80:4/- distance/classical:128:80:5/- distance/classical:128:80:6/- \
    --arms '[{}, {"partition_work_ratio": 4611686018427387904}]'
```

### c1-low

```bash
python experiments/distance_tuning/sweep.py \
    --tag c1-low \
    --rounds 6 \
    --jobs 1 \
    --timeout 600 \
    --base '{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads distance/lp13s2/X distance/lp13s10/X \
    --arms '[{}, {"partition_work_ratio": 20}]'
```

### c4-setup

```bash
python experiments/distance_tuning/sweep.py \
    --tag c4-setup \
    --rounds 4 \
    --jobs 1 \
    --repeat 3 \
    --timeout 900 \
    --base '{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads distance/sparse:8000:21:400:2/- \
    --arms '[{}, {"enumeration_setup_cost_factor": 8}]'
```

### c6-samples

```bash
python experiments/distance_tuning/sweep.py \
    --tag c6-samples \
    --rounds 10 \
    --jobs 1 \
    --timeout 600 \
    --base '{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads count/pp110/12 ccount/pp110/12 distance/classical:128:80:1/- distance/lp13s5/X distance/qt7/X distance/pp232/X \
    --arms '[{}, {"completion_bound_samples": 512}, {"completion_bound_samples": 1024}]'
```

### c7-coset

```bash
python experiments/distance_tuning/sweep.py \
    --tag c7-coset \
    --rounds 4 \
    --jobs 1 \
    --timeout 900 \
    --base '{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads coset/pp232/css/10/10/300 coset/qt7/css/10/10/300 \
    --arms '[{}, {"overlapping_minimum_work": 10000}]'
```

### c8-usecase

```bash
python experiments/distance_tuning/sweep.py \
    --tag c8-usecase \
    --rounds 10 \
    --jobs 1 \
    --timeout 600 \
    --base '{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100}' \
    --workloads certify/gross/css/2 faultdist/toric:6/Z \
    --arms '[{}, {"overlapping_minimum_work": 10000}, {"overlapping_minimum_work": 20000}]'
```

### c9-decided

```bash
python experiments/distance_tuning/sweep.py \
    --tag c9-decided \
    --rounds 4 \
    --jobs 1 \
    --timeout 900 \
    --base '{"completion_bound_max_cost_percent": 100}' \
    --workloads coset/gross/css/10/10/3000 coset/pp110/css/8/8/2000 coset/pp232/css/10/10/300 coset/qt7/css/10/10/300 coset/gross/physical/6/6/500 coset/pp232/physical/6/6/500 first/pp232/physical/6/6/2000 hcoset/gross/css/4/4/500 hbound/pp232/2/2000 hbound/pp372/2/2000 certify/gross/css/2 faultdist/toric:6/Z distance/lp13s1/X distance/lp13s5/X distance/classical:128:80:6/- distance/qt7/X distance/pp232/X count/pp110/12 \
    --arms '[{"completion_bound_samples": 256, "coset_completion_block_bits": 8, "overlapping_minimum_work": 100000, "partition_work_ratio": 100, "completion_bound_max_cost_percent": 90}, {}]'
```

### c11-revert

```bash
python experiments/distance_tuning/sweep.py \
    --tag c11-revert \
    --rounds 6 \
    --jobs 1 \
    --timeout 600 \
    --base '{"completion_bound_max_cost_percent": 100}' \
    --workloads distance/classical:128:80:6/- coset/pp110/css/8/8/2000 \
    --arms '[{}, {"completion_bound_max_cost_percent": 90}, {"completion_bound_samples": 256}, {"overlapping_minimum_work": 100000}, {"partition_work_ratio": 100}, {"coset_completion_block_bits": 8}]'
```
