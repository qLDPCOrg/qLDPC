# Retuning the exact distance searches

This page walks through a retuning of the fields of `SearchTuning` that Brouwer-Zimmermann (BZ) searches read, like the retuning that set their current values.
The Retuning section of the module docstring of [`qldpc.codes.distance._tuning`](../../src/qldpc/codes/distance/_tuning.py) states how a value is chosen among candidates, and the comment of each field says what to measure to retune it.
The [experiments README](../README.md#distance-tuning) introduces the scripts.
Sweeps write their trials to `results/`, which Git ignores, so this page keeps the commands of the sweeps behind the current values.

## Before measuring

- Time searches in new single-threaded processes, as `sweep.py` does, and run sweeps under a guard of time and memory.
- Completion tables save less when several searches run at once, so time their fields one search at a time (`--jobs 1`), and screen other fields a few at a time.
- Record the commit: `sweep.py` keys trials by it, and by a hash of any uncommitted change to `src/`.
- Do not edit `src/` while a sweep runs, since each trial imports it anew.
- A trial of one repetition also times work that a search caches on its code, such as the generator matrix of a classical code given by checks, which is the same in every arm: under a second for most distance workloads, but seconds to tens of seconds for the sparse codes of 4000 and 8000 bits, and about a minute and several minutes for the classical codes of 4000 and 8000 bits.
  It shrinks ratios toward 1 without reversing them; time such workloads with `--repeat 3`, whose median leaves it out.

## 1. Measure the noise floor

Run a tuning against itself, as `noise-j1`, `noise-j4`, and `noise-j10` below do, and read the 95% intervals of `collect_results.py`.
On an Apple M3 Max, ten rounds gave 95% intervals of roughly ±1% to ±9% one search at a time and ±3% to ±8% ten at a time, and a few workloads compared with themselves had intervals that excluded 1.
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
| Families and partitioning | `distance/classical:128:80:1/-`, `distance/classical:140:90:1/-`, and quantum workloads with `automorphism_minimum_layer_work` at `MAX_VALUE` | Families shape the scan only where automorphisms do not take over, and classical codes have none |
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

For example, the factor `enumeration_setup_cost_factor` stands for the time to set up information sets, divided by the numbers of rows and columns and by the time per enumerated codeword, but that ratio ranged from below 1 to about 100 across codes.
The main cause, a sizing of enumeration batches that did not vectorize rows of more than 960 bits, was found in `src/qldpc/codes/distance/_brute_force.py` and confirmed by raising `enumeration_batch_exponent` from 15 to 30, which made each enumerated codeword a few hundred times faster on rows of 1000 bits.

## 5. Check the sensitivity of long runs

The long runs, of a minute or more one search at a time, are `distance/bb288/X`, `distance/pp170/X`, `hadamard/gross`, `hadamard/pp110`, `distance/classical:160:110:1/-`, `distance/classical:1000:976:2/-`, the classical codes of 4000 and 8000 bits with 21 to 23 rows, and the coset searches `coset/pp232/css/10/10/3000`, `coset/qt7/css/10/10/3000`, and `coset/lp13s5/physical/7/7/1000`.
The sparse codes of 4000 bits with 21 to 23 rows, which take tens of seconds when they enumerate, are traced with them, and the windows below count them among the long runs (`--long`).
The check `c1-gate` (section 7) found two more long runs, `distance/lp13s6/X` and `distance/lp13s6/Z`, which the windows do not cover.
The classical codes of 4000 and 8000 bits spend about a minute and several minutes of that in work that does not depend on the tuning (see "Before measuring").
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
The values were chosen in-sample, on dozens of workloads, on one Apple M3 Max that often ran other work, and mostly on earlier library code.
These measures limit how much that matters:

- The distance searches and counts among the long runs were traced, and `windows.py` found how far each threshold can move before one of their decisions changes; where that range is broad, the precise value matters only for quick runs.
- Where candidates made no clear difference, the value is a round number among them, not the one that happened to time fastest.
- Where a value rests on a mechanism, changing the quantity that the mechanism names moved the outcome as it implies.
- The last checks, whose sweeps in section 7 have tags of a `c` and a digit, had their predictions written down first, and several were refuted.
- A tuning timed against itself set the noise floor that a difference had to exceed.

These measures reach only so far:

- Some candidates made no clear difference only because the searches timed make the same decisions under them, so those timings do not test the value.
- Many comparisons were made, so a candidate set aside by one small difference may have been set aside by chance.
- The judgment calls, `enumeration_setup_cost_factor`, `partition_work_ratio`, and `coset_completion_block_bits`, trade one kind of search against another, and the first two rest on where the measurements of a few codes happened to fall.
- `overlapping_minimum_work` was chosen with 8-column coset tables, on the query costs of a few CSS codes.
  With the 16-column tables now used, a higher value was a little faster on the one CSS coset search timed again.
- The fields of enumeration were measured while enumeration handled rows of more than 960 bits one word at a time (section 4), before a fix made those rows vectorize.
- The cost limit and the number of samples of completion tables rest on a few long-run layers and searches, and the cost model errs in both directions.
- Two workloads of fault-propagation certificates, used to choose no value, ran no slower with the values decided last, with the cost limit then at 100, but their queries are too cheap to test the values that changed.
- The final set of values was not timed as a whole.

## 7. Reproduce the measurements

The commands below ran on the library code of their time, at the commits that `trials.jsonl` records, and each compares its arms with its first arm, as its command gives it.
That arm is usually the defaults of its time, in some sweeps with `completion_bound_minimum_layer_work` at `3 * 10**7`.
In `final-short-f`, it sets five fields to the values that its command gives.
Some sweeps, such as `p3-noaut`, also set a field in every arm.
The `--base` of each command gives the defaults of its time where they differ from the current ones, so that each command compares the same values on the current code.

Measurements of one search use `trace.py`, `trial.py`, or `fit_completion_costs.py`:

| Measurement | Command |
| --- | --- |
| Partitioning: estimates, costs, and gates | `trace.py distance/lp13s5/X --tuning '{"partition_work_ratio": 10}' --output R1.json`, the same with `"automorphism_minimum_layer_work": 4611686018427387904`, and `trace.py hadamard/gross --output R3.json`; `trace.py <workload> --setup-only` for the gates of other workloads |
| `enumeration_setup_cost_factor` | `trace.py <workload> --setup-only --tuning '{"allow_enumeration_by_size": false, "allow_enumeration_by_work": false}'` for the time of the disjoint family, and `trial.py <workload> --repeat 3 --tuning '{"enumeration_max_dimension": 62}'` for the time per word, for `distance/classical:<n>:<n - k>:2/-` and `distance/sparse:<n>:<k>:<n / 20>:2/-` with `n` of 200, 1000, and 4000 and `k` of 16 and 20 |
| `enumeration_batch_exponent` | `trial.py 'distance/classical:1000:980:2/-' --repeat 3 --tuning '{"enumeration_max_dimension": 62}'`, and with `"enumeration_batch_exponent": 30` |
| `automorphism_minimum_layer_work` at `10**7` | `trace.py distance/qt7/X --tuning '{"automorphism_minimum_layer_work": 10000000}'` |
| Sampling of `completion_bound_minimum_layer_work` | `trace.py <workload> --tuning '{"completion_bound_samples": 256}' --output <file>`, and with automorphisms off, summing the `seconds` of the `predict` events before each `layer` event |
| Predictions of the cost model, and `completion_bound_block_bits` | `trace.py distance/classical:150:100:1/-` and `trace.py distance/pp232/X` for the predictions, and `fit_completion_costs.py collect --repeats 3 --workloads distance/pp232/X 'distance/classical:150:100:1/-'`, then `fit_completion_costs.py fit`, for the measured costs; `trace.py distance/qt7/X --tuning '{"completion_bound_block_bits": 8}'` |
| Cost coefficients | `fit_completion_costs.py collect --repeats 3 --workloads distance/lp13s5/X distance/lp13s8/X distance/qt7/X distance/qt7/Z distance/pp232/X distance/pp110/X distance/gross/X 'distance/classical:150:100:1/-' 'distance/classical:128:80:1/-'`, then `fit_completion_costs.py fit --min-supports 1e8` |
| Windows of the thresholds | the commands below the table |
| Past the edges of the windows | `trace.py <long> --tuning '{"completion_bound_minimum_layer_work": 1000000}'`; `trace.py hadamard/pp110 --tuning '{"automorphism_minimum_layer_work": 80000000}'`, the same for `distance/pp170/X`, and `trace.py distance/bb288/X --tuning '{"automorphism_minimum_layer_work": 200000000}'`, each with `--output` to a file outside the directory of the traces of the windows, and compared with its trace at the defaults by `windows.py --compare`; `trace.py hadamard/gross --tuning '{"completion_bound_lookup_block_cost": 0.9, "completion_bound_lookup_block_cost_per_mib": 0.05, "completion_bound_samples": 256}'`, whose layer 6 declines its tables, for the time of that layer without them |
| Layer 7 of `distance/classical:128:80:6/-` at the limits of 90 and 100 | `trace.py 'distance/classical:128:80:6/-' --tuning '{"completion_bound_max_cost_percent": 100}'`, the same with `"completion_bound_samples": 256` added, and at the defaults, compared by `windows.py --compare` and by their `layer` events |
| Gate values of partitioning across codes | `trace.py distance/lp13s<seed>/<sector> --setup-only` for seeds 1 to 10 and sectors `X` and `Z`, and `trace.py 'distance/classical:128:80:<seed>/-' --setup-only` for seeds 2 to 6; a gate value is the `estimated_work` of the `family` event of kind `partitioned` divided by its `unit` |
| Below the window of `enumeration_setup_cost_factor` | `trace.py <w> --setup-only --tuning '{"enumeration_setup_cost_factor": 8}'` for `'distance/classical:8000:7979:2/-'` and `'distance/sparse:8000:21:400:2/-'` |
| Families of `overlapping_minimum_work` | `coset_families.py coset/pp232/css/10/10/3000`, and the same for `coset/qt7/css/10/10/3000` and `coset/lp13s5/physical/7/7/1000`; `coset_families.py <w> --thresholds 20000 24000 30000` for `coset/gross/css/10/10/3000`, `coset/pp110/css/8/8/2000`, `coset/pp232/css/10/10/3000`, and `coset/qt7/css/10/10/3000` |
| Sampling of layers near the cost limit | `resample.py hadamard/gross --weights 6` and `resample.py distance/pp170/X --weights 6 7`, and `resample.py hadamard/pp110 --weights 5` and `resample.py distance/bb288/X --weights 5`; for the sampling budget, `resample.py hadamard/gross --weights 6 --num-seeds 1 --changes '[{"completion_bound_sampling_work": 25000000}, {"completion_bound_sampling_work": 100000000}]'` |
| Linear programs of `lp_minimum_work` | the `linear program` events of `trace.py <w>` at the defaults |
| Memory of `max_css_single_pairs` | `tracemalloc` around `_tabulate_css_singles` of `_brouwer_zimmermann/coset.py` on 255, 372, and 511 random single-qubit syndromes of full size |
| Cache sizes | `cache_stats.py --cycles 2 coset/gross/css/6/6/100 coset/pp110/css/6/6/100 coset/qt7/physical/5/5/50 hbound/pp232/2/100 distance/qt7/X distance/lp13s5/X count/gross/12` |

Searches without information sets: `trial.py 'distance/classical:<n>:<checks>:2/-' --repeat 2` with and without `--tuning '{"allow_enumeration_by_work": false}'`, reading the second repetition.

These commands trace the workloads behind the windows of section 5, at the defaults, and print the windows; the classical codes that enumerate many words are traced up to their decision to enumerate.
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

These sweeps produced the ratios behind the current values, and `collect_results.py --tags <tag>` summarizes each.

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
