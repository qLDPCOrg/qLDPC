# Evidence for the defaults of BZ searches

This file gives the reason for the default of each field of `SearchTuning` in [`_tuning.py`](../../src/qldpc/codes/distance/_tuning.py) whose comment refers here, and the measurements behind it.
Each measurement names the sweep tag of [section 7 of the README](README.md#7-reproduce-the-measurements), or the script and arguments, that reproduces it, with how many searches ran at a time and in how many rounds.
A ratio is the median, over rounds, of the time of a candidate divided by that of the first arm of its sweep in the same round; brackets give its 95% interval, or with fewer than six rounds its range, and candidates measured alike if that contains 1 on every workload compared.
The window of a threshold is the interval of its values over which every decision of the traced workloads stays the same (`windows.py`); "on the traced long runs" restricts it to the traced long runs of [section 5](README.md#5-check-the-sensitivity-of-long-runs), of a minute or more one search at a time, which do not include `distance/lp13s6/X` and `distance/lp13s6/Z`, long runs found later; quick runs take a few seconds or less.
Every timing comes from one Apple M3 Max, on the library code of qLDPC commits from `15897436` to `eba1dd2f`; "earlier code" marks a sweep that ran on library code older than that of the current values, at the commit that `trials.jsonl` records.
Claims are labeled verified (a trace or the code shows it, and an intervention confirms it), consistent (it fits the data, untested), or conjecture, and each reason names the rule of the Retuning section of the module docstring that chose the value.
[Section 6 of the README](README.md#6-guard-against-over-fitting) says what guards these values against over-fitting.

## allow_enumeration_by_work

Value: True.
Reason: a switch for tests that force a route, not a calibration, so no rule chose it; its measurements show what the route costs.

Evidence:

- `trial.py 'distance/classical:<n>:<checks>:2/-' --repeat 2`, with and without `--tuning '{"allow_enumeration_by_work": false}'`, second repetition, one run each, three at a time: False made `distance/classical:1000:976:2/-` take 1.56 s instead of 70.57 s, and `distance/classical:4000:3977:2/-` 21.28 s instead of 43.78 s (verified).
- The same with False made `distance/classical:8000:7977:2/-` take 99.72 s, against about 80 s for the default search, estimated from the 450 s of `p2-factor` (earlier code), of which about 370 s was work that no tuning changes; this refuted a prediction that False makes it more than 5 times faster.
- One repetition of `trial.py 'distance/classical:1000:976:2/-'` took 66.60 s at the defaults, 1.013 s with `enumeration_batch_exponent` 30, and 2.286 s with both `allow_enumeration` fields False, so enumeration one codeword at a time causes most of its time (verified, intervention).
- At the defaults, the classical codes of 1000 to 8000 bits with 21 to 24 rows enumerate by work, except `distance/classical:8000:7979:2/-`, which enumerates by size (traces of the windows).

Limitations:

- These and the measurements of the other fields of enumeration were taken while enumeration handles rows of more than 960 bits one codeword at a time, since the batch sizing of `_brute_force.py` subtracts the number of 64-bit words of a row from the batch exponent, not its logarithm (see [`enumeration_batch_exponent`](#enumeration_batch_exponent)).
  They should be checked again once that is fixed.
- One run each, three at a time.

Follow-up: fix the batch sizing of enumeration, and time the workloads that enumerate before and after; measure the fields of enumeration again only if one of them gets slower.

## enumeration_max_dimension

Value: 20, chosen by hand.
Reason: chosen by hand, so no rule chose it; 20 is the upper edge of its window on the traced long runs, and its window on all traced workloads is 20 alone (verified, traces).

Evidence:

- `p2-confirm` (eight at a time, the codes of 8000 bits two at a time, 6 rounds, `--repeat 3`, earlier code), 18 with `enumeration_setup_cost_factor` 16 against 20 with 128: `distance/sparse:1000:19:50:2/-` 0.052 [0.043, 0.066] (0.136 s against 2.75 s), and `distance/sparse:1000:20:50:2/-` 0.024 [0.023, 0.026] (0.135 s against 5.46 s).
  That they gain since they then search information sets instead of enumerating rows of 1000 bits one codeword at a time is consistent.
- In the same sweep, `distance/classical:1000:980:2/-` and `distance/classical:1000:981:2/-`, whose routes 18 also changes, measured alike, and the codes of 4000 and 8000 bits with 19 or 20 rows enumerate under both values, since the factor admits them.
- In the same sweep, of the six codes of 40 and 200 bits with 19 or 20 rows, which take milliseconds, 18 made `distance/classical:40:20:2/-` take 0.647 and `distance/sparse:200:20:10:2/-` 0.330 of their time, made `distance/classical:200:180:2/-` take 1.293 and `distance/sparse:200:19:10:2/-` 1.375 times as long, and measured alike on `distance/classical:200:181:2/-` (1.68 [0.83, 1.86]) and `distance/classical:40:21:2/-` (1.13).
- `v-maxdim` (one at a time, 8 rounds, `--repeat 3`, earlier code), 18 against 20: `distance/classical:200:180:2/-` 1.31 [1.18, 1.60], and 0.87 [0.81, 0.91] with `allow_enumeration_by_work` False, so it is slower since it enumerates by work after building information sets (verified).
  `distance/sparse:200:19:10:2/-` 1.41 [1.29, 1.79], and 1.37 [1.35, 2.34] with it False, so information sets search it more slowly regardless (verified).
- At 21, `distance/sparse:4000:21:200:2/-` and `distance/sparse:1000:21:50:2/-` enumerate (traces), which took them 2.1 and 23 times as long as information sets in `p2-factor` (four at a time, 6 rounds, one repetition, earlier code: 19.28 s against 9.39 s, and a ratio of 0.043); the classical codes of 21 rows enumerate by size instead of by work.

Window: on the traced long runs, every value up to 20, its upper edge.
On all traced workloads, 20 alone, between `distance/sparse:200:20:10:2/-` and `distance/classical:1000:980:2/-`, of 20 rows, and `distance/classical:1000:979:2/-` and others, of 21; the traced workloads have bases of every number of rows from 16 to 24.

Limitations:

- The enumeration caveat of [`allow_enumeration_by_work`](#allow_enumeration_by_work).
- 18 favored two medium runs, of 1000 bits, by seconds, and moved quick runs both ways by milliseconds.

Follow-up: as for [`allow_enumeration_by_work`](#allow_enumeration_by_work).

## enumeration_setup_cost_factor

Value: 16.
Reason: rule (2), among 4, 16, and 128: against 128, 16 made four of six sparse codes of 1000 and 4000 bits with 21 to 23 rows 2.1 to 23 times faster and the other two no slower, and random codes of 1000 and 4000 bits at most 1.3 s slower (verified).

Evidence:

- `p2-factor` (four at a time, 6 rounds, one repetition, earlier code), 16 against 128: `distance/sparse:1000:21:50:2/-` 0.043, `distance/sparse:4000:21:200:2/-` 0.483, `distance/sparse:4000:22:200:2/-` 0.309, `distance/sparse:4000:23:200:2/-` 0.176, and `distance/sparse:1000:22:50:2/-` and `distance/sparse:1000:23:50:2/-` 1.001.
- In the same sweep, the random codes of 1000 and 4000 bits were at most 1.3 s slower, the most `distance/classical:4000:3979:2/-`, 1.020 [1.006, 1.034] of 66.5 s, which includes about 50 s of work that no tuning changes.
  Those of 8000 bits measured alike: `distance/classical:8000:7977:2/-` 1.004 [0.991, 1.056], `distance/classical:8000:7978:2/-` 1.014 [0.984, 1.096], and `distance/classical:8000:7979:2/-` 1.001 [0.971, 1.047].
- `final-short-f` (three at a time, 8 rounds), whose first arm sets this factor to 128 among five fields: `distance/sparse:1000:21:50:2/-` 0.042 [0.041, 0.042].
- What the factor stands for, the time to build the disjoint information sets of a code divided by its numbers of rows and columns and by the time to enumerate one codeword, has no single value (verified; the commands of the row `enumeration_setup_cost_factor` of section 7).
  With 16 or 20 rows, it was 0.35 and 0.53 on random codes of 1000 bits, 1.50 and 1.19 on sparse ones, 1.12 and 10.54 on random codes of 4000 bits, 4.62 and 4.05 on sparse ones, and 57 to 97 on codes of 200 bits.
  An enumerated codeword took 5 to 7 ns on codes of 200 bits, and 4.1 to 5.5 µs on codes of 1000 and 4000 bits, whose rows enumerate one codeword at a time (verified, see [`enumeration_batch_exponent`](#enumeration_batch_exponent)).
- Below the window, at 8 (`c4-setup`, one at a time, 4 rounds, `--repeat 3`; and traces), `distance/sparse:8000:21:400:2/-` searches information sets and took 0.590 [0.579, 0.600] of its time (6.95 s against 11.68 s).
  `distance/classical:8000:7979:2/-` then builds its disjoint family, 381 sets in 1.71 s in a trace, before it enumerates by work; this was traced, not timed.
- Above the window of size tests, from 23.83, `distance/classical:8000:7978:2/-` enumerates by size instead of by work, which spares it its family build.
  From 24.97, `distance/sparse:4000:21:200:2/-` enumerates, the first route that changes, which took it 2.1 times as long in `p2-factor`, and `distance/classical:4000:3979:2/-` enumerates by size instead of by work (traces).
- With `enumeration_max_dimension` 18, 4 against 16 (`p2-confirm`, its fourth arm against its third, paired by round, 6 rounds, eight at a time and the codes of 8000 bits two at a time, `--repeat 3`, earlier code; ranges): `distance/sparse:4000:19:200:2/-` 0.693 [0.666, 0.745], `distance/sparse:4000:20:200:2/-` 0.317 [0.295, 0.394], `distance/classical:8000:7980:2/-` 1.997 [1.728, 2.102], `distance/classical:4000:3981:2/-` 1.202 [1.077, 1.327], and `distance/sparse:8000:20:400:2/-` 1.195 [1.010, 1.292].

Window: on the traced long runs and on all traced workloads, every size test stays the same from 12.48 to 23.83, the ratios of `2**rows` to rows times columns of `distance/classical:8000:7979:2/-` and `distance/classical:8000:7978:2/-`, and every route from 12.48 to 24.97, that of `distance/sparse:4000:21:200:2/-`.

Limitations:

- The enumeration caveat of [`allow_enumeration_by_work`](#allow_enumeration_by_work).
- Past either edge of the window of routes, a value helps some codes and slows others.
- `p2-factor` ran one repetition, which times work that no tuning changes and so shrinks ratios toward 1.

Follow-up: as for [`allow_enumeration_by_work`](#allow_enumeration_by_work).

## enumeration_batch_exponent

Value: 15, chosen by hand.
Reason: chosen by hand, so no rule chose it.
Its measurements verify the cause of the enumeration caveat: rows of `w` packed 64-bit words are vectorized over at most `2**(15 + 1 - w)` combinations of rows (`_brute_force.py`), so rows of more than 960 bits enumerate one codeword at a time.

Evidence:

- `trial.py 'distance/classical:1000:980:2/-' --repeat 3 --tuning '{"enumeration_max_dimension": 62}'`, and the same with `"enumeration_batch_exponent": 30`: walls of 4.974, 4.144, and 4.151 s at 15, about 4.0 µs per codeword after the first repetition, and of 0.815, 0.013, and 0.013 s at 30, about 12.3 ns per codeword, so 30 made each codeword about 325 times faster (verified, intervention).
- No value between 15 and 30 was timed.
- At 30, rows of 4 words would take arrays of 4 GiB, and rows of more than 1920 bits would still enumerate one codeword at a time (code).

Limitations: one code, with enumeration forced.

Follow-up: as for [`allow_enumeration_by_work`](#allow_enumeration_by_work).

## max_information_sets

Value: 16.
Reason: rule (3): 16 and 32 measured alike, and 16 builds the smaller families (verified).

Evidence:

- `p3-aut` (six at a time, 8 rounds, earlier code), against 16: 32 changed no workload beyond its interval, and 8 made `distance/classical:140:90:1/-` take 1.46 [1.35, 1.52] times as long.
- `p3-noaut` (the same, with automorphisms off): 32 changed no workload beyond its interval, and 8 made `distance/lp13s5/X` take 1.22 [1.19, 1.24] times as long, `distance/qt7/X` 0.921 of its time, and `distance/gross/X` 0.997.

Limitations: not timed on long runs.

## disjoint_random_column_orders

Value: 8.
Reason: rule (3): 0, 8, and 16 measured alike, and 8 is their middle (verified).

Evidence:

- `p3-aut` (six at a time, 8 rounds, earlier code), against 8: 0 made `coset/gross/physical/6/6/500` take 1.011 [0.972, 1.092] and `distance/qt7/X` 1.017 [1.000, 1.078] times as long, and 32 made them take 1.020 [1.002, 1.042] and 1.026 [1.011, 1.093].
- `p3-orders-aut` (six at a time, 8 rounds, earlier code): 16 measured alike; 32 made `hbound/pp232/2/2000` take 1.047 [1.008, 1.150] times as long, and 64 1.141 [1.085, 1.343].
- So 3 of the 25 comparisons of 32 in `p3-aut`, `p3-noaut`, and `p3-orders-aut`, all six at a time, excluded 1.
  Comparisons of a tuning with itself excluded 1 in 2 of 11 one at a time (`noise-j1`), and in none of 22 four and ten at a time (`noise-j4`, `noise-j10`); none ran six at a time.

Limitations: not timed on long runs.

## overlapping_minimum_work

Value: `2 * 10**4`.
Reason: the maintainer's decision, by rule (1): the value sets the target weight of the families of coset searches, the smallest weight from 2 whose predicted examinations reach it (verified, `coset_families.py` and `_brouwer_zimmermann/coset_routes.py`, and timings on both sides of the window).
`2 * 10**4` is a round value inside the window above 16,473 and up to 34,300, in which the CSS coset searches of `gross`, `pp110`, `pp232`, and `qt7` build the same families (verified: `2 * 10**4`, `2.4 * 10**4`, and `3 * 10**4` give identical fingerprints and tails on every part).
The lower edge is the predicted examinations of a query of weight 2 of the Z part of `qt7`, a large code, and the upper edge those of a query of weight 3 of the Z part of `pp110`, a moderate code.
Coset searches serve mostly moderate codes queried many times, as in certification and the ranking of circuits, so the edge of `pp110` matters most, and `2 * 10**4` lies 1.7 times below it.

Evidence:

- `coset_families.py` on CSS coset searches of the four codes, the predicted examinations of queries of weights 2 and 3 of their Z parts: `pp110` 1,770 and 34,300; `gross` 3,080 and 79,200; `pp232` 10,880 and 529,545; `qt7` 16,473 and 988,443.
- `coset_families.py coset/pp232/css/10/10/3000`, and the same for `coset/qt7/css/10/10/3000` and `coset/lp13s5/physical/7/7/1000`: the Z part of `pp232` aims at weight 2 for values up to 10,880 (a family of 3 sets, predicted work `5.39 * 10**10`), at weight 3 up to 529,545 (14 sets, `1.03 * 10**11`), and at weight 4 above.
  The Z part of `qt7` aims at weight 2 up to 16,473 (3 sets, `1.39 * 10**11`), at weight 3 up to 988,443 (5 sets, `5.28 * 10**10`), and at weight 4 above.
- In the same output, the X part of `pp232` keeps its family up to 204,853, the predicted examinations of its query of weight 7, though its target weight changes above 106,083; the X part of `qt7` keeps one family at every value, and `coset/lp13s5/physical/7/7/1000` one family from `10**3` to `10**7`.
- From `10**6` to `10**7`, the Z parts build byte-identical families to those at `10**4`; this refuted, by fingerprints rather than by timing, a prediction that a higher target weight is faster or alike.
- Below the window, `c7-coset` (one at a time, 4 rounds; ranges), `10**4` against `10**5`: `coset/pp232/css/10/10/300` 3.189 [3.147, 3.239] (41.90 s against 13.28 s), and `coset/qt7/css/10/10/300` 2.402 [2.378, 2.430] (52.95 s against 22.01 s), as predicted before the check (verified).
- `p6-long` (four at a time, 4 rounds, earlier code; ranges), against `10**5`: `10**4` made `coset/pp232/css/10/10/3000` take 3.160 [3.016, 3.309] and `coset/qt7/css/10/10/3000` 2.120 [1.906, 2.131] times as long, and `10**3` 3.141 and 2.083; `coset/lp13s5/physical/7/7/1000` took 1.031 and 1.024.
- Above the window, `p6-short` (four at a time, 8 rounds, earlier code), against `10**5`: `3 * 10**4` made `coset/gross/css/10/10/3000` take 0.578 [0.538, 0.585] and `coset/pp110/css/8/8/2000` 0.666 [0.533, 0.716] of their time (1.9 s against 3.3 s, and 0.135 s against 0.21 s), `10**4` 0.574 and 0.662, and `10**3` 1.002 and 1.002.
  Above 34,300, the Z part of `pp110`, and above 79,200 that of `gross`, lose their target weight of 3.
- Held out from the choice, `c8-usecase` (one at a time, 10 rounds, at load averages of 3.7 to 5.7), against `10**5`: `certify/gross/css/2` (0.347 s) 0.995 [0.971, 1.002] at `10**4` and 0.999 [0.969, 1.009] at `2 * 10**4`, and `faultdist/toric:6/Z` (0.035 s) 0.979 [0.956, 1.002] and 0.975 [0.950, 1.001], alike as predicted.
  That their routes keep the disjoint family is consistent, not traced.
- `hbound/pp232/2/2000`, `hbound/pp372/2/2000`, `first/pp232/physical/6/6/2000`, and `hcoset/gross/css/4/4/500` measured alike at `10**4`, `3 * 10**4`, and `10**5` (`p6-short`, and `coset-m`; earlier code).
- Distance searches and counts decide the same from `8.4 * 10**4` to `1.4 * 10**6` (traces of the windows), and the traced long ones estimate their work at `5.17 * 10**11` or more.
  1000, `10**4`, and `3 * 10**4` changed no timed distance search by more than 3% (`p3-aut` and `p6-short`, earlier code: `distance/qt7/X` 1.03 [1.00, 1.09] at `10**4` in `p3-aut`; `distance/pp232/X` 1.000, 0.997, and 0.995, and `distance/qt7/X` 1.009, 1.013, and 1.006, at `3 * 10**4`, `10**4`, and `10**3` in `p6-short`).
- With the other decided values (`c9-decided`, see [the decided values together](#the-decided-values-together)), `2 * 10**4` left `distance/qt7/X`, `distance/pp232/X`, and `distance/lp13s5/X` within 5%, and made the CSS coset searches of `gross`, `pp232`, and `qt7` 3.5 to 15 times faster, with the width of 16; `coset/pp110/css/8/8/2000` was 1.155 times slower, from that width (see [`coset_completion_block_bits`](#coset_completion_block_bits)).

Window: on the CSS coset searches of `gross`, `pp110`, `pp232`, and `qt7`, above 16,473 and up to 34,300.
On the traced long coset runs, above 16,473 and up to 204,853, and for their Z parts up to 529,545 (`pp232`) and 988,443 (`qt7`).
On distance searches and counts, above `8.368 * 10**4` (`distance/surface:7/X`) and up to `1.437 * 10**6` (`distance/classical:200:179:2/-`); on the traced long distance runs, every value up to `5.17 * 10**11` (`distance/classical:160:110:1/-`).

Limitations:

- The work estimate of the route misranks the families of the Z part of `pp232`: it prefers the slower family, which values up to 10,880 build, predicting `5.39 * 10**10` examinations for it and `1.03 * 10**11` for the family of the window; the value works by never building the slower one.
- `2 * 10**4` lies below the window of distance searches, so it changes decisions of quick distance searches, such as `distance/surface:7/X`; it was timed on distance searches only with the other decided values.
- The long coset runs of 3000 queries were timed only four at a time, on earlier code; on the current code, only their versions with 300 queries were timed.
- The result at `3 * 10**4` ran four at a time, on earlier code.

## overlapping_attempts_per_set

Value: 2.
Reason: rule (3): 1, 2, and 4 measured alike, and 2 leaves a second order when the first repeats an existing set (verified).

Evidence:

- `p3-aut` and `p3-noaut` (six at a time, 8 rounds, earlier code), against 2: 1 and 4 changed none of 14 workloads beyond its interval.

Limitations: not timed on long runs.

## partition_work_ratio

Value: 500, so that its product with `partition_cost_factor`, 10, is 5000.
Reason: the maintainer's decision, by rule (3): no traced product separates the searches that partitioning slows from those that it speeds up, since their gate values overlap.
What separates them is whether a search switches to automorphism images, as the searches of `lp13s<seed>` do and those of random classical codes do not (verified on the traced searches).
5000 is a round value in the interval of products above 3022 and up to 7227, which misclassifies the fewest traced searches: of those that partitioning slows, only `distance/lp13s8/X`, at 24,375, still partitions, and none that it speeds up is skipped.
On the traced searches, 5000 and 1000 decide differently only on `distance/lp13s1/X`.

Evidence:

- A gate value is the estimated work of a search divided by `b * k**2 * n`: the `estimated_work` of the `family` event of kind `partitioned` of `trace.py <w> --setup-only`, divided by its `unit` (current code).
- Gate values of `distance/lp13s<seed>/<sector>` for seeds 1 to 10: 3022 for s1/X; 24,375 for s6/X, s6/Z, and s8/X; 571.7 for s1/Z, s2/X, s2/Z, s5/X, s5/Z, s8/Z, s10/X, and s10/Z; 0.1 for s3, 4.7 for s4, and 0.0 for s9; s7 records no partition event.
- Gate values of `distance/classical:128:80:<seed>/-` for seeds 2 to 6: 16,357, 85,170, 16,357, 137,667, and 7227.
  `distance/qt7/X` has 6895, and measured alike with partitioning never (`p3-partition`, four at a time, 4 rounds, earlier code: 4.05, 4.05, and 4.07 s).
- `c1-gate` (one at a time, 6 rounds), partitioning never against the product 1000: `distance/lp13s1/X` 0.324 [0.304, 0.356] (0.59 s against 1.84 s), and `distance/lp13s8/X` 0.292 [0.266, 0.309] (0.51 s against 1.75 s).
  `distance/lp13s6/X` (167 s) 0.964 [0.930, 1.107] and `distance/lp13s6/Z` (73 s) 0.962 [0.847, 1.055], alike.
  `distance/classical:128:80:<seed>/-` for seeds 2 to 6: 1.168, 1.692, 1.130, 1.590, and 1.717 (0.37 to 2.63 s with partitioning), and for seed 1, 1.19 [1.10, 1.22] (`v-partition`, one at a time, 10 rounds, earlier code).
- `c1-low` (one at a time, 6 rounds), ratio 20, a product of 200, against the product 1000: `distance/lp13s2/X` 3.96 [3.877, 4.002] (1.71 s against 0.43 s), and `distance/lp13s10/X` 3.56 [3.480, 3.642] (1.84 s against 0.51 s).
- Partitioning takes 1.2 to 1.3 s on the `lp13s<seed>` codes, and 0.04 s on `classical:128:80` (traces).
- A search that switches to automorphism images discards the family that partitioning found, which partitioning builds before the switch (verified: trace of `hadamard/gross`, and `_brouwer_zimmermann/distance.py`).
- The work estimate at the gate exceeded the work done 17 times on `distance/classical:160:110:1/-`, 274 times on `distance/classical:150:100:1/-`, and 341 times on `distance/lp13s5/X` (traces of the windows).
  It assumes that the search finds no lighter word, which dominates on the classical codes, and counts supports that completion tables and automorphism images skip, which dominates on `distance/lp13s5/X` (verified).
  So the intent of the ratio, that an attempt that does not pay off adds at most about `1 / partition_work_ratio` to the work of a search, cannot hold.
- On the traced long runs, the gate decides the same for every product above 32.3 and up to 143,490.
  `distance/lp13s6/X` and `distance/lp13s6/Z`, long runs found later, have the gate value 24,375, inside that interval, and measured alike with partitioning off (`c1-gate`).
  `hadamard/gross` and `hadamard/pp110` measured alike from ratio 10 to never (`p3-partition`, factor 20, four at a time, 4 rounds, earlier code: 0.995, 0.999, and 0.988 for `hadamard/gross` and 0.983, 0.972, and 0.965 for `hadamard/pp110` at ratios 30, 100, and never, against 10).
- `c9-decided` (one at a time, 4 rounds; ranges), with the other decided values, against the product 1000 and the values before the decisions: `distance/lp13s1/X` 0.360 [0.354, 0.368] (0.71 s against 2.0 s), as predicted before the check, and `distance/lp13s5/X`, at 571.7, 1.043 [1.019, 1.095].
  `distance/classical:128:80:6/-`, at 7227, partitions under both, and was 1.190 times slower from the cost limit, then at 100 (see [`completion_bound_max_cost_percent`](#completion_bound_max_cost_percent)).

Window: on the traced long runs, every product above 32.3 (`distance/classical:1000:976:2/-`) and up to 143,490 (`distance/classical:160:110:1/-`).
Among the traced gate values of quick runs, 5000 lies between 3022 (`distance/lp13s1/X`) and 6895 (`distance/qt7/X`).

Limitations:

- The partition gate cannot separate the traced searches that partitioning slows from those that it speeds up.
- The product bounds no share of wasted time, since the work estimate overstates the work done.

Follow-up: decide whether to switch to automorphism images before partitioning.

## max_partition_coverage

Value: 4, chosen by hand.
Reason: chosen by hand, so no rule chose it; its effect runs through the partition gate, which cannot separate the traced searches (see [`partition_work_ratio`](#partition_work_ratio)).

Evidence:

- Coverage `c` tries coverages 2 to `c`, so the cost estimate counts `b = 2 + ... + c` copies of the columns, 9 at 4, 5 at 3, and 2 at 2, and lowering `c` raises every gate value by `9 / b` (verified by `c2-coverage`).
- `c2-coverage` (one at a time, 10 rounds), 3 and 2 against 4, at the product 1000: `distance/classical:128:80:1/-` 0.945 [0.928, 0.962] and 1.120 [1.097, 1.142]; `distance/lp13s8/X`, whose attempt finds no family, 0.613 [0.607, 0.634] and 0.396 [0.391, 0.399]; `distance/classical:140:90:1/-` 0.999 and 1.000.
  `distance/lp13s5/X` 2.066 [2.049, 2.101] and 1.345 [1.323, 1.370], since 3 and 2 raise its gate value from 572 to 1029 and 2573, past the product of 1000, so that it partitions.
  A prediction that 3 lies between 4 and 2 on `distance/classical:128:80:1/-` was refuted in its wording, since 3 was faster than 4, though not by its written criterion.
- `p3-aut` (six at a time, 8 rounds, earlier code): 2 made `distance/classical:128:80:1/-` take 1.21 [1.14, 1.28] times as long.

Limitations:

- Measured at the product 1000; at the product of 5000, the gate values of `distance/lp13s5/X` at coverages 3 and 2, 1029 and 2573, stay below the product, so its slowdowns would not recur (consistent, untested).
- Not timed on long runs.

## partition_cost_factor

Value: 10.
Reason: rule (1), a direct measurement of the cost of partitioning in its units, of which the value takes the higher end (verified, two searches).

Evidence:

- `trace.py distance/lp13s5/X --tuning '{"partition_work_ratio": 10}'`, and the same with automorphisms off: partitioning took 1.453 and 1.452 s, at 1.62 and 1.54 ns per evaluation, so 10.0 and 10.5 units.
- Six traces of `hadamard/gross`, at the defaults and controls: partitioning took 0.816 to 1.007 s, at 1.71 to 2.03 ns per evaluation, so 4.34, 4.42, 5.17, 4.39, 4.39, and 5.37 units, of 94,618,368 evaluations each.
- Each trace times an evaluation by the layers of its search that used no tables and had at least `10**7` evaluations.
- At the ratio of 100, the factors 10 and 20 made the same decision on all 22 traced gates (earlier code).

Limitations:

- At the ratio of 500, the products 5000 and 10,000 of the factors 10 and 20 differ on `distance/classical:128:80:6/-` (7227) and `distance/qt7/X` (6895), by the traced gate values (consistent, untested).
- The measurements spread from 4.3 to 10.5 units.

## lp_minimum_work

Value: `10**9`.
Reason: rule (3): `10**8`, `10**9`, and `10**10` measured alike, and `10**9` is their middle, inside the window over which every traced workload decides the same (verified).

Evidence:

- `p3-aut` and `p3-noaut` (six at a time, 8 rounds, earlier code): the three measured alike, except that `10**8` made `distance/gross/X` without automorphisms 21 ms slower (1.110 [1.001, 1.448]).
  Timings separate them only on `distance/gross/X`, at `1.37 * 10**8`: the other timed searches use certificates of disjoint sets, or estimate their work at `3.02 * 10**10` or more.
- The `linear program` events of `trace.py <w>` at the defaults: the long runs `distance/bb288/X`, `hadamard/gross`, and `hadamard/pp110` estimate their work at `5.01 * 10**13`, `1.17 * 10**13`, and `6.23 * 10**13`, and solve 2, 2, and 3 linear programs in 5.2, 6.5, and 9.0 ms.
- On quick runs, `count/gross/12` and `ccount/gross/12` solve 12 in 17.6 to 17.9 ms, `distance/classical:140:90:1/-` 14 in 23.8 ms, `distance/classical:128:80:1/-` 7 in 10.0 ms (2.7% of its time), `distance/classical:130:80:1/-` 9 in 10.7 ms (3.4%), `distance/lp13s5/X` 2 in 6.5 ms (1.07%), and `distance/lp13s5/Z`, `distance/lp13s8/X`, `distance/qt7/X`, and `distance/qt7/Z` 2 each; this refuted a prediction of under 1%.
- `distance/lp13s6/X` and `distance/lp13s6/Z`, long runs found later, estimate their work at `2.18 * 10**12`.

Window: on all traced workloads, above `3.649 * 10**8` (`distance/pp110/both`) and up to `1.956 * 10**9` (`count/gross/12`), so from `3.7 * 10**8` to `1.9 * 10**9`; on the traced long runs, every value up to `1.17 * 10**13`.

## automorphism_minimum_layer_work

Value: `10**8`.
Reason: rule (3): `3 * 10**7`, `10**8`, and `3 * 10**8` measured alike, and `10**8` is their middle; they measured alike since the searches timed with them decide the same from `1.867 * 10**7` (`distance/pp232/X`) to `4.153 * 10**8` (`distance/lp13s5/X`) (verified, traces).

Evidence:

- `p3-orders-aut` (six at a time, 8 rounds, earlier code): `3 * 10**7` and `3 * 10**8` measured alike with `10**8` on `distance/qt7/X`, `distance/qt7/Z`, `distance/lp13s5/X`, and `distance/pp232/X`.
- A search decides once (`_brouwer_zimmermann/distance.py`): `trace.py distance/qt7/X --tuning '{"automorphism_minimum_layer_work": 10000000}'` decided before finding its best word, kept its family, and took 21.04 s against 3.01 s; `p3-aut` (six at a time, 8 rounds, earlier code) gives 7.06 [6.95, 7.77] for `10**7` (verified).
- `p3-aut`: `10**9` made `distance/qt7/X`, `distance/qt7/Z`, and `distance/lp13s5/X` take 1.27, 1.41, and 1.31 times as long.
- Past the edges of the window on the traced long runs (traces compared by `windows.py --compare`): at `8 * 10**7`, `hadamard/pp110` switches before layer 4, skipping 0.12 s, and `distance/pp170/X`, whose layer 5 has `8.30 * 10**7`, before layer 5, skipping 0.22 s; at `2 * 10**8`, `distance/bb288/X` switches before layer 5, after its layer 4 of 0.22 s; all with identical later decisions.

Window: on the traced long runs, above `8.441 * 10**7` (layer 4 of `hadamard/pp110`) and up to `1.823 * 10**8` (layer 4 of `distance/bb288/X`); on all traced workloads, above `8.441 * 10**7` and up to `1.267 * 10**8` (`distance/gross/X`).

Limitations: the window on the traced long runs is narrow, though a layer of at most 0.22 s changes past either edge.

## completion_bound_minimum_layer_work

Value: `3 * 10**7`.
Reason: rule (1): it lies below the `4.5 * 10**7` supports of the steps of `count/pp110/12`, which consider tables only at such values and were then 2.75 times faster (verified, trace and intervention); `10**7` measured alike, three at a time.

Evidence:

- Traces of `count/pp110/12`: its steps have `4.5 * 10**7` supports (59 rows, weight 6), and at `10**8` it records no prediction of tables.
- `p5-isolate-j1` (one at a time, 8 rounds, earlier code), `10**8` against `3 * 10**7`: `count/pp110/12` 2.751 [2.418, 2.899], and `distance/classical:128:80:1/-`, with a layer of `7.36 * 10**7` supports, 1.489 [1.244, 1.541]; the others within their intervals.
- `p5-gates` (three at a time, 8 rounds, earlier code): `10**7` and `3 * 10**7` made `count/pp110/12` take 0.505 [0.431, 0.604] and 0.517 [0.435, 0.739] of its time at `10**8`, so the two measured alike, with a wide interval (consistent).
- Traces at the defaults, with 256 samples: sampling the layers of at least `3 * 10**7` supports that declined tables took 0.34 to 0.55 ms for each set on the classical codes, 1.2 to 1.3 ms on `distance/pp170/X`, 3.1 ms on `hadamard/pp110`, 6.2 to 6.4 ms on `distance/lp13s5/X` and `distance/lp13s8/X`, 6.7 ms on `hadamard/gross`, and 8.0 to 8.1 ms on `distance/qt7/X` and `distance/qt7/Z`, at most 13.1% of the layer (layer 4 of `distance/qt7/X`).
- On the traced long runs, every decision to use tables stays the same from `10**6` supports to `5.92 * 10**8`, layer 5 of `distance/bb288/X`, the smallest layer that uses tables there.
  At `10**6`, each long search considers one more layer shape, two for `distance/classical:160:110:1/-`, predicts 2.8 to 13.2, and declines it; the sampling took 1.5 to 53 ms for each search.

Window: on the traced long runs, the layers considered stay the same above `2.374 * 10**7` (`hadamard/gross`, 156 rows, weight 4) and up to `4.15 * 10**7` (`distance/pp170/X`, 89 rows, weight 5); on all traced workloads, above `2.374 * 10**7` and up to `3.78 * 10**7` (`distance/lp13s5/X`, 175 rows, weight 4).
The layers that use tables stay the same from `10**6` to `5.92 * 10**8` on the traced long runs, and up to `4.51 * 10**7` (`count/pp110/12`) on all traced workloads.

Limitations:

- The share of sampling was measured with 256 samples; with 1024, `c6-samples` found the quick runs that sample such layers at most 7.3% slower in all (see [`completion_bound_samples`](#completion_bound_samples)).
- `10**7` measured alike only three at a time, on earlier code.

## completion_bound_max_cost_percent

Value: 90.
Reason: the maintainer's decision, by rule (1).
With 1024 samples, the predictions of the traced long-run layers near the limit under 40 sampling seeds leave a gap from 86.1 to 93.9 percent, and 90 lies in it, so none of those layers decides by its draw (verified, `resample.py`).
100 was chosen first, as the break-even point of the cost model, but two checks showed that it is not one:

- Each set of a layer gets tables if its own prediction is within the limit, and the layer then scans its sets one at a time if their average, counting 1.0 for each set without tables, is within it too (`is_within_cost_limit` in `_information_sets/completion_bounds.py` and `_choose_tables_of_sets` in `_brouwer_zimmermann/distance.py`).
  But each prediction compares the tables of a set with a scan of that set alone, while the layer would otherwise scan its sets in lockstep.
  At 100, 7 of the 8 sets of layer 7 of `distance/classical:128:80:6/-`, of `7.36 * 10**7` supports, predict 0.982 to 0.997 and get tables, and the layer took 1.177 s one set at a time, against 0.846 to 0.866 s in lockstep at 90 and before the decisions (verified: traces of section 7, `c11`, with interventions on the limit and the samples).
  With 256 samples at 100, one set gets tables, and the layer took 1.114 s one set at a time.
- At 100 with 1024 samples, layer 6 of `distance/pp170/X` predicts 0.939 to 1.053 under 40 seeds and gets tables under 31 of them, so a long run decides by its draw (`c10`), and layer 5 of `hadamard/pp110` predicts 0.989 to 1.159, within the limit under 2.

200 was not chosen.
It made `distance/classical:160:110:1/-`, a long run, 1.40 times faster one search at a time, since the model overestimates the cost of tables on bases of few rows, but it made `distance/classical:140:90:1/-` 1.32 times slower three at a time, and gives tables to layers of several sets whose lockstep scan costs less.

Evidence:

- The predictions of the cost model err both ways (`trace.py` at the defaults, and `fit_completion_costs.py collect`, the rows of section 7 for the cost model; verified).
  It predicted 0.67 for layer 5 of `distance/pp232/X`, which took 0.66 to 0.96 of its time without tables when it used them, in six runs over two sessions.
  It predicted 1.41 for layer 9 of `distance/classical:150:100:1/-`, of 50 rows, which forced tables would have made take 0.55 to 0.70, in six runs, so every limit up to 140 declines them.
  It predicted 0.898 for layer 6 of `hadamard/gross`, which took 0.45 with tables (13.92 s against 30.78 s).
  That load causes the underestimate is a conjecture.
- `p5-gates` (three at a time, 8 rounds, earlier code): 100 against 90 changed no workload beyond its interval.
- `costlimit-j1-f` (one at a time, 4 rounds; ranges), 200 and 150 against 90: 200 made `distance/classical:160:110:1/-`, a long run of 67.71 s, take 0.714 [0.672, 0.720] of its time, 1.40 times faster, and `distance/classical:150:100:1/-`, of 8.78 s, 0.774 [0.715, 0.857], 1.29 times faster; `distance/classical:140:90:1/-` measured alike (0.993 [0.947, 1.015]).
  150 made them take 1.007, 0.772, and 1.000.
- `costlimit-quick-f` (one at a time, 4 rounds; ranges), 200 against 90: `distance/lp13s5/X` 0.969 [0.425, 1.046] (0.533 s), `distance/lp13s5/Z` 1.018 [0.951, 1.073] (0.628 s), `distance/lp13s8/X` 0.986 [0.950, 0.995] (1.743 s), `distance/qt7/X` 0.990 [0.826, 1.100] (2.934 s), and `distance/qt7/Z` 0.979 [0.923, 1.012] (2.018 s), medians within 0.125 s; single rounds changed by up to 0.71 s.
- Earlier sweeps of 200 against 90 (earlier code): `p5-costlimit-j1` (one at a time, 6 rounds) gave 0.85, 0.65, and 0.62 on the classical codes of 140, 150, and 160 bits, and `accept-isolate` (three at a time, 8 rounds) 1.32 [1.15, 1.42] on `distance/classical:140:90:1/-`.
- Traces at 200: `hadamard/gross`, `hadamard/pp110`, and `distance/pp170/X` build the same tables one or two layers sooner, and the layers that change took 0.18 s less to 0.38 s more, while the other layers did the same work; `distance/bb288/X` is unchanged, and `distance/classical:160:110:1/-` uses tables on layers 8 to 10.
- `resample.py`, under seeds 0 to 39 (`c10`, at 100 percent and 1024 samples): layer 6 of `hadamard/gross` predicts 0.732 to 0.861, layer 7 of `distance/pp170/X` 0.702 to 0.799, layer 5 of `distance/bb288/X` 0.640 to 0.729, layer 6 of `distance/pp170/X` 0.939 to 1.053, and layer 5 of `hadamard/pp110` 0.989 to 1.159.
  With 256 samples, layer 6 of `hadamard/gross` predicted 0.707 to 1.058, within 0.90 under 24 seeds.
- `c11-revert` (one at a time, 6 rounds, at load averages near 10), each decided value reverted alone, against the decided values with the limit at 100: 90 made `distance/classical:128:80:6/-` take 0.913 [0.803, 1.025] of its time, and the other reverts measured alike, so timing alone did not single out the limit; the traces above did.

Window: on the traced long runs, with 1024 samples, from 86.1 to 93.9 under every seed of 40; at seed 0, from about 82.6 to 98.0.
With 256 samples, the window was from 89.84 to 103.46, and its lower edge rested on the draw of layer 6 of `hadamard/gross`.

Limitations:

- The gap was found with 40 seeds of five traced long-run layers; other layers can predict inside it.
- The cost model errs in both directions, so a layer near the limit may decline tables that would pay, or use tables that cost more.
- The final set was not timed as a whole: `c9-decided` ran with the limit at 100, and only `distance/classical:128:80:6/-` and `coset/pp110/css/8/8/2000` were timed again with it at 90 (`c11-revert`).

## completion_bound_block_bits

Value: 16.
Reason: rule (1): with 8, the cost model declines the tables of the costliest layers of `distance/qt7/X`, which is then as slow as without tables (verified, trace and forced tables).

Evidence:

- On every layer of the width data set of at least `5 * 10**8` supports, 16 skipped more supports, in less time, than 8 (earlier code).
- `p5-screen` (three at a time, 8 rounds, earlier code), against 16: 8 made `distance/qt7/X`, `distance/pp232/X`, and `count/gross/12` take 13.35, 2.16, and 2.20 times as long, 4 made them take 13.35, 2.18, and 2.45, and disabling tables 13.77, 2.14, and 2.36.
- `trace.py distance/qt7/X --tuning '{"completion_bound_block_bits": 8}'`, and forced tables of 8-column blocks (`fit_completion_costs.py`): the model predicted 1.318 for layer 5 and 1.255 for layer 6, which forced tables made take 1.14 and 0.99 of their time without tables.

Limitations: not timed on long runs.

## completion_bound_fallback_block_bits

Value: 8, chosen by hand.
Reason: chosen by hand, so no rule chose it.

Evidence:

- Forced on the six layers of the width data set of at least `10**9` supports that the refit data set includes (`fit_completion_costs.py ... fit --min-supports 1e8`), tables of 8-column blocks took 1.135 of the time without tables on layer 5 of `distance/lp13s5/X`, 1.194 on layer 5 of `distance/lp13s8/X`, and 1.138 on layer 5 of `distance/qt7/X`.
  The model predicts 1.273, 1.324, and 1.318 for those three, and declines their tables.
  The other three took 0.988 (layer 6 of `distance/qt7/X`, which the model also declines), 0.417, and 0.159 (layers 5 and 6 of `distance/qt7/Z`).

Limitations: not timed on long runs.

## coset_completion_block_bits

Value: 16.
Reason: the maintainer's decision, by rule (2): a large gain on long CSS coset searches, at a loss on Pauli searches by physical weight, including the long run `coset/lp13s5/physical/7/7/1000`, and about 1.5 GiB more peak memory.
The maintainer prioritizes CSS codes; searches by physical weight are a niche.

Evidence:

- `p6-long` (four at a time, 4 rounds, earlier code; ranges), against 8: 16 made `coset/qt7/css/10/10/3000` take 0.034, `coset/pp232/css/10/10/3000` 0.127, and `coset/lp13s5/physical/7/7/1000` 3.996 of their time, at peak resident memory of 0.27 to 1.83 GiB; 4 made them take 1.41 to 1.89 times as long.
- `p6-short` (four at a time, 8 rounds, earlier code), against 8: 16 made `coset/gross/css/10/10/3000` 2.8 times and `coset/gross/physical/6/6/500` 2.2 times faster, and `coset/pp232/physical/6/6/500` take 2.75 times as long; 4 made `coset/gross/css/10/10/3000` and `coset/gross/physical/6/6/500` take 2.87 and 1.61 times as long.
- On residual-weight estimates of state-preparation circuits of 37, 81, and 95 qubits, not in the repository, tables of 16-column blocks took longer to build than most estimates spent scanning.
- `c9-decided` (one at a time, 4 rounds; ranges), with the other decided values, against 8 with the values before the decisions: `coset/qt7/css/10/10/300` 0.067 [0.061, 0.081] (1.5 s against 22.7 s), `coset/pp232/css/10/10/300` 0.220 [0.202, 0.223] (3.0 s against 13.7 s), `coset/gross/css/10/10/3000` 0.286 [0.266, 0.302], and `coset/gross/physical/6/6/500` 0.597 [0.475, 0.708]; `coset/pp232/physical/6/6/500` 2.895 [2.857, 2.913] (5.5 s against 1.9 s), and `first/pp232/physical/6/6/2000` 2.723 [2.707, 2.745] (21.5 s against 7.9 s).
  Peak resident memory after the search rose from about 0.26 GiB to 1.75 and 1.83 GiB on the CSS coset searches of `pp232` and `qt7`, and to 1.2 GiB on `coset/gross/physical/6/6/500`, but not on those of `pp232` by physical weight.
  `certify/gross/css/2`, `faultdist/toric:6/Z`, `hbound/pp232/2/2000`, `hbound/pp372/2/2000`, and `hcoset/gross/css/4/4/500` measured within 4%.
- `c11-revert` (one at a time, 6 rounds), against the decided values: width 8 made `coset/pp110/css/8/8/2000` take 0.651 [0.640, 0.694] of its time (0.116 s against 0.179 s, at 261 MiB against 421), and the threshold `10**5` 0.874 [0.797, 0.887].
  So on this moderate code, 16-column tables cost about 63 ms once `2 * 10**4` gives more of its queries routes by tails (verified, intervention), and made it 1.155 times slower than before the decisions in `c9-decided`.

Limitations:

- Searches of Pauli errors by physical weight are 2.7 to 4 times slower, and `coset/pp110/css/8/8/2000`, of a moderate code, 63 ms slower.
- The residual-weight estimates above cannot be reproduced from the repository.
- The long runs of 3000 queries were measured four at a time, on earlier code.

Follow-up: a separate width for each kind of search.

## completion_bound_samples

Value: 1024.
Reason: the maintainer's decision, by rule (1): more samples make the decisions of layers near the cost limit depend less on the sampling draw (verified, `resample.py`).
With 1024, every traced long-run layer near the limit of 90 decides the same under 40 sampling seeds, while with 256, layer 6 of `hadamard/gross`, 14 s with tables and 31 s without, used its tables under only 24 of them.

Evidence:

- `resample.py hadamard/gross --weights 6` and `resample.py distance/pp170/X --weights 6 7`, and the same for layer 5 of `hadamard/pp110` and of `distance/bb288/X`, under seeds 0 to 39: the seeds within the limit at 256, 512, and 1024 samples were 24, 34, and 40 for layer 6 of `hadamard/gross`; 1, 0, and 0 for layer 6 of `distance/pp170/X`; 39, 40, and 40 for its layer 7; 0, 0, and 0 for layer 5 of `hadamard/pp110`; and 40, 40, and 40 for layer 5 of `distance/bb288/X`.
- Layer 6 of `hadamard/gross` predicts 0.707 to 1.058 under these seeds with 256 samples, and 0.732 to 0.861 with 1024, 0.826 at seed 0; at seed 0, 128 samples make it predict 0.933, 512 0.880, and 4096 0.819.
- `c6-samples` (one at a time, 10 rounds), against 256: 512 measured alike on six quick runs.
  1024 made `ccount/pp110/12` take 1.049 [1.009, 1.114] times as long (0.706 s against 0.660 s), `count/pp110/12` 1.023 [1.004, 1.069] (0.677 s against 0.661 s), and `distance/lp13s5/X` 1.073 [1.029, 1.118] (0.590 s against 0.567 s), so 16 to 46 ms more; `distance/classical:128:80:1/-` 1.003, `distance/pp232/X` 1.038 [0.956, 1.051], and `distance/qt7/X` 1.030 [0.960, 1.051].
  A prediction that no run is more than 10% slower held, though `distance/lp13s5/X` exceeded the predicted 5%.
- `p5-sampling` (three at a time, 8 rounds, earlier code), against 256: 128 measured alike, and 512 made `count/pp110/12` take 1.066 [1.029, 1.107] times as long.
- With 256 samples, sampling took 1 to 13 ms on the cost data set.
- `resample.py hadamard/gross --weights 6` at the decided values (`c12`): 0.826 at seed 0, within 0.90 under all 40 seeds; at seed 0, 128 samples make it predict 0.933 and decline its tables, 256 0.898, 512 0.880, and 4096 0.819.

Limitations:

- At a limit of 100, layer 6 of `distance/pp170/X` decides by its draw even with 1024 samples (31 of 40 seeds within; `c10`), so the reason holds for the limit of 90.
- 1024 costs quick runs up to 46 ms; a number of samples that grows with the layer could spare them, which was not tried.
- Not timed on long runs.

## completion_bound_sampling_work

Value: `5 * 10**7`, chosen by hand.
Reason: chosen by hand, so no rule chose it.

Evidence:

- `resample.py hadamard/gross --weights 6 --num-seeds 1 --changes '[{"completion_bound_sampling_work": 25000000}, {"completion_bound_sampling_work": 100000000}]'`: layer 6 predicts 0.8984 at `2.5 * 10**7`, `5 * 10**7`, and `10**8`, so the budget does not limit its samples (verified).

Limitations: checked with 256 samples, not with 1024.

## completion_bound_confidence_z

Value: 2.0.
Reason: rule (3): 1.0, 2.0, and 3.0 measured alike on quick runs, and 2.0 is a conventional confidence of about 98%.

Evidence:

- `p5-sampling` (three at a time, 8 rounds, earlier code), against 2.0: 1.0 and 3.0 changed no workload beyond its interval.
- `resample.py hadamard/gross --weights 6` at seed 0 (`c12`): with 1024 samples, 1.0 makes layer 6 predict 0.786, and 3.0 0.868, both within the limit of 90; with 256 samples, 3.0 gave 0.991, which declined the tables of that layer, 13.9 s with them and 30.8 s without.

Limitations: not timed on long runs.

## completion_bound_lookup_block_cost

This section also covers `completion_bound_lookup_block_cost_per_mib`.

Value: 2.0, and 0.04 for each MiB.
Reason: rule (1): fitted to the cost data set, in which lookups cost 2 to 8 scanned supports per block (earlier code); a refit on the refit data set did not reproduce, so the measurements favor neither, and 2.0 and 0.04 stand.

Evidence:

- `fit_completion_costs.py collect ...`, then `fit_completion_costs.py fit --min-supports 1e8` (the row "Cost coefficients" of section 7): the refit gives 0.787 scanned supports per block, plus 0.041 for each MiB of tables.
  It rests on lookups that cost 0.46 to 0.85 scanned supports per block (median 0.72) on 9 layers with tables of under 1 MiB, and 1.24 to 7.57 (median 4.14) on 9 layers with tables of 18 to 113 MiB.
- For the 46 MiB of tables of `distance/pp232/X`, the refit predicts 2.7 scanned supports per block, and 2.0 and 0.04 predict 3.8.
  The layers of that search cost 2.5 to 4.5 in the session of the refit, and 6.0 to 9.4 in another (verified).
- With 0.9 and 0.05, from an earlier refit, and 256 samples, layer 6 of `hadamard/gross` declined its tables at the limit of 90: in traces of one batch, three at a time, that layer took 30.8 s instead of 13.9 s, and the search 179.0 s instead of 170.0 s.
  With 1024 samples, 0.9 and 0.05 make it predict 0.866 and keep its tables (`c12`).
  The refit of 0.787 and 0.041 makes that layer predict 0.816 and keep its tables.
- The refit makes layer 6 of `distance/pp170/X` predict 0.896 and use its tables, which was not timed.

Limitations:

- One traced long run, `hadamard/gross`.
- The measured costs of the same lookups varied two- to threefold between sessions.

## completion_bound_kept_first_row_ratio

Value: 0.35.
Reason: rule (1), a direct measurement: the median of the ratios on the 7 layers of the refit data set that scan one set with blocks of 16 columns and skip at most 99% of their supports (verified).

Evidence:

- `fit_completion_costs.py fit --min-supports 1e8` (the row "Cost coefficients" of section 7): with blocks of 16 columns, `distance/lp13s5/X` 0.35, `distance/lp13s8/X` 0.32, `distance/pp232/X` 0.49 and 0.38, `distance/qt7/X` 0.36, `distance/qt7/Z` 0.31, and the last layer of `distance/classical:150:100:1/-` 0.04, so 0.04 to 0.49; the layers 6 of `distance/qt7/X` and `distance/qt7/Z`, which skip 0.998 and 0.997, are left out.
- With blocks of 8 columns, 9 layers gave 0.38 to 0.75, median 0.65.
- The cost data set gave 0.31 to 0.50 (earlier code).
- `resample.py hadamard/gross --weights 6` at seed 0 (`c12`): with 1024 samples, 0.31 makes layer 6 predict 0.778, and 0.38 0.862, both within the limit of 90; with 256 samples, 0.38 gave 0.939, which declined its tables.

Limitations:

- The ratios spread from 0.04 to 0.50.
- Not timed on long runs.

## completion_bound_table_byte_cost

This section also covers `completion_bound_build_step_cost`.

Value: 0.1 for each byte, and 0.2 for each step, chosen by hand.
Reason: chosen by hand, so no rule chose them; a direct measurement agrees with them for blocks of 16 columns, the default width, on quantum codes (verified).

Evidence:

- `measure_build` of `fit_completion_costs.py`, applied to each run of each layer with a build in the data of the row "Cost coefficients" of section 7: with blocks of 16 columns, the two predicted 0.97 to 1.00 of the measured cost of the 9 builds of `distance/lp13s5/X`, 0.95 to 0.98 of the 9 of `distance/lp13s8/X`, 0.94 to 1.07 of the 6 of `distance/pp232/X`, and 0.94 to 1.05 of the 18 of `distance/qt7/X` and `distance/qt7/Z`, so 0.94 to 1.07 over 42 builds.
- They predicted 0.80 to 1.98 of the 24 builds of `distance/classical:150:100:1/-`.
- With blocks of 8 columns, they predicted 0.43 to 0.53 on the same quantum codes, and 0.49 to 0.82 on that classical code.

Limitations:

- They fit blocks of 16 columns on quantum codes, not blocks of 8 columns or the classical code.
- Not timed on long runs.

## information_set_batch_exponent

Value: 15.
Reason: rule (3): 15 is the smallest value beyond which larger ones were no slower, one at a time, and smaller values were slower by the overhead of returning to Python (verified, intervention).

Evidence:

- `c5-isbatch` (one at a time, 10 rounds), 11 to 14 against 15: `distance/pp232/X` 1.200 [1.185, 1.208], 1.096 [1.091, 1.132], 1.047 [1.034, 1.058], and 1.025 [1.015, 1.033]; `distance/qt7/Z` 1.070 [1.048, 1.090], 1.038 [1.014, 1.044], 1.009 [1.003, 1.024], and 1.006 [0.986, 1.019].
  This refuted a prediction, by its written criterion, that 11 to 15 measure alike.
- `c5-isbatch-high` (one at a time, 10 rounds), 16, 17, 18, and 20 against 15: `distance/pp232/X` 0.979 [0.892, 1.021], 0.961 [0.929, 0.992], 0.993 [0.933, 1.047], and 0.980 [0.939, 1.026]; `distance/qt7/Z` 0.994, 1.000, 0.995, and 1.019, all intervals containing 1.
  So 17 on `distance/pp232/X` was faster, in one of eight comparisons.
- At 15, a call of a layer without tables takes about 0.04 to 0.09 ms, at the 1.2 to 2.7 ns per support of the layers of at least `10**7` supports without tables of the traced searches.

Limitations: not timed on long runs.

## scan_chunk_size

Value: 65,536.
Reason: rule (3): twice 32,768, the smallest value that measured alike with it one at a time, a margin chosen by hand; smaller values were slower.

Evidence:

- `c5-chunk` (one at a time, 10 rounds), against 65,536: 16,384 made `count/gross/12` and `count/pp110/12` take 1.009 [1.001, 1.058] and 1.030 [1.008, 1.087] times as long, and 32,768 0.994 [0.990, 1.060] and 1.009 [0.966, 1.060].
  This refuted a prediction, by its written criterion, that 16,384 measures alike.
- `p1-chunk-fine` (one at a time, 10 rounds, earlier code): 131,072 measured alike (0.993), and 16,384 made `count/pp110/12` take 1.016 [1.001, 1.023] times as long.
- `p1-chunk` (six at a time, 10 rounds, earlier code): `2**20` measured alike, and 4096 made `count/pp110/12` take 1.090 [1.062, 1.114] times as long.

Limitations: not timed on long runs.

## scan_chunks_per_call

Value: 64.
Reason: rule (3): powers of two from 4 to 1024 measured alike, and 64 lies between them.

Evidence:

- `p1-chunks-per-call` (six at a time, 10 rounds, earlier code), against 64: 4 and 1024 made `ccount/gross/12` take 1.001 and 1.007, and `ccount/pp110/12` 1.002 and 0.998, of its time.

Limitations: not timed on long runs.

## max_css_single_pairs

Value: `2**18`.
Reason: rule (1): the table makes each query of weight one a lookup instead of a match over the qubits, and `2**18` is the smallest power of two that fits the `373**2` pairs of the 372-qubit state of `hbound/pp372/2/2000`, the only workload whose searches read it on a state of more than 255 qubits (verified, intervention on the field).

Evidence:

- `p6-short` (four at a time, 8 rounds, earlier code), `2**18` against `2**16`: `hbound/pp372/2/2000` 0.58 [0.51, 0.64]; `final-short-f` (three at a time, 8 rounds), against the first arm, which sets `2**16`: 0.589 [0.578, 0.597].
  Single runs took 0.539 s at `2**18`, 0.901 s at `2**16`, and 0.962 s with the table disabled.
- `tracemalloc` around `_tabulate_css_singles` of `_brouwer_zimmermann/coset.py` on random full-size syndromes: 11.1 MiB for `2**16` pairs on 255 qubits, 28.4 MiB for 139,129 pairs on 372 qubits, and 65.8 MiB for `2**18` pairs on 511 qubits (verified).
- On the other workloads that read it, every table fits under both values.

Limitations: one workload reads it on a state of more than 255 qubits.

## Cache sizes

The cache sizes of `_tuning.py` are chosen by hand.
These measurements describe them at their values; they did not choose them.

Evidence:

- `cache_stats.py --cycles 2 coset/gross/css/6/6/100 coset/pp110/css/6/6/100 coset/qt7/physical/5/5/50 hbound/pp232/2/100 distance/qt7/X distance/lp13s5/X count/gross/12`, which runs each workload once before emptying the caches: `BINOMIAL_CACHE_SIZE` had 39 hits and 8 misses, so these seven workloads used 8 entries.
- In the same run, `CSS_SEARCH_CACHE_SIZE` and `PAULI_SEARCH_CACHE_SIZE` had 2 hits and 2 misses each.
  A missed pair of the 144-qubit state of `coset/gross/css/6/6/100` took about 0.002 s to set up, the difference of 0.009 s and 0.007 s rounded to milliseconds, and a missed search of the 343-qubit state of `coset/qt7/physical/5/5/50` 0.25 s, from 0.324 s and 0.075 s.

Limitations: the memory of an entry of `CSS_SEARCH_CACHE_SIZE` or `PAULI_SEARCH_CACHE_SIZE`, which may hold completion tables, was not measured.

## Completion tables under concurrent load

The module docstring of `_tuning.py` refers here.
These effects decide no value.

Evidence:

- With tables wherever they are considered, `completion_bound_max_cost_percent` at `MAX_VALUE`, against 90, `distance/classical:150:100:1/-` took 0.746 [0.730, 0.763] of its time one at a time (`p5-load-j1`, 8 rounds), 0.903 [0.896, 0.941] six at a time (`p5-load-j6`, 12 rounds), and 1.304 [1.079, 1.445] three at a time (`p5-gates`, 8 rounds) at load averages of 17 to 20, all on earlier code (verified).
- That contention for caches causes this is a conjecture.

## The decided values together

`c9-decided` (one search at a time, 4 rounds; ranges; load averages of 8 to 15) timed the values of five fields together, `completion_bound_samples` 1024, `coset_completion_block_bits` 16, `overlapping_minimum_work` `2 * 10**4`, `partition_work_ratio` 500, and `completion_bound_max_cost_percent` at 100, since revisited to 90, against their values before the decisions, 256, 8, `10**5`, 100, and 90, on 18 workloads, with predictions written first.
No result differed.

- Faster: `coset/qt7/css/10/10/300` 0.067, `coset/pp232/css/10/10/300` 0.220, `coset/gross/css/10/10/3000` 0.286, `distance/lp13s1/X` 0.360, and `coset/gross/physical/6/6/500` 0.597.
- Slower: `coset/pp232/physical/6/6/500` 2.895 and `first/pp232/physical/6/6/2000` 2.723, by physical weight, as predicted; `distance/classical:128:80:6/-` 1.190, from the limit at 100 (see [`completion_bound_max_cost_percent`](#completion_bound_max_cost_percent)); and `coset/pp110/css/8/8/2000` 1.155, from the width (see [`coset_completion_block_bits`](#coset_completion_block_bits)).
  The last two refuted predictions of the check.
- Within 5%: `certify/gross/css/2`, `faultdist/toric:6/Z`, `hbound/pp232/2/2000`, `hbound/pp372/2/2000`, `hcoset/gross/css/4/4/500`, `distance/lp13s5/X`, `distance/qt7/X`, `distance/pp232/X`, and `count/pp110/12`.

With the limit at 90, only `distance/classical:128:80:6/-` and `coset/pp110/css/8/8/2000` were timed again (`c11-revert`).

## Known limitations

These were found during the retuning, and are not pursued:

- The work estimate of a search, which several gates compare with costs, exceeded the work done 17 to 341 times on three traced searches (see [`partition_work_ratio`](#partition_work_ratio)); whether it explains the other gates is untested.
- The cost model of completion tables overestimates the cost on bases of few rows, as on layer 9 of `distance/classical:150:100:1/-` (1.41 predicted, 0.55 to 0.70 measured); a limit of 200 made `distance/classical:160:110:1/-` 1.4 times faster one search at a time, but `distance/classical:140:90:1/-` 1.32 times slower three at a time.
- Layers of several sets compare the tables of each set with a scan of that set alone, not with their lockstep scan (see [`completion_bound_max_cost_percent`](#completion_bound_max_cost_percent)); at the limit of 90, no traced layer loses by it.
- A distance search decides once whether to switch to automorphism images; deciding before it finds its best word made `distance/qt7/X` 7 times slower at `automorphism_minimum_layer_work` of `10**7`, but no traced search loses by it at the default.
- The work estimate of the routes of coset searches prefers the slower family of the Z part of `pp232` (see [`overlapping_minimum_work`](#overlapping_minimum_work)); no traced search loses by it at the default.
- 1024 samples cost quick runs up to 46 ms (see [`completion_bound_samples`](#completion_bound_samples)).
