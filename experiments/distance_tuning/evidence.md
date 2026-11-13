# Evidence for the defaults of BZ searches

This file gives the reason for the default of each field of `SearchTuning` in [`_tuning.py`](../../src/qldpc/codes/distance/_tuning.py) whose comment refers here, and the measurements behind it.
Its numbers are rough: they come from one Apple M3 Max, mostly on earlier library code, and timings of the same work varied up to two- or threefold between sessions.
Each claim names the sweep tag of [section 7 of the README](README.md#7-reproduce-the-measurements), or the script and the row of its table, that reproduces it, and says how many searches ran at a time where that matters.
Candidates "measured alike" when their difference lay within the noise of their comparison, and a window is the range of a threshold over which traced decisions stay the same (`windows.py`); long runs take a minute or more one search at a time, quick runs a few seconds or less, and "the traced long runs" are those of [section 5](README.md#5-check-the-sensitivity-of-long-runs).
Each claimed reason or mechanism is labeled verified (a trace or the code shows it, and an intervention confirms it), consistent (it fits the data, untested), or conjecture, and each reason names the rule of the Retuning section of the module docstring that chose the value.
[Section 6 of the README](README.md#6-guard-against-over-fitting) says what guards these values against over-fitting.

## allow_enumeration_by_work

Value: True.
Reason: a switch for tests that force a route, not a calibration, so no rule chose it.
Evidence: `trial.py` with and without the field (the line "Searches without information sets" of section 7), one run each, three at a time: False made a random code of 1000 bits about 40 times faster and one of 4000 bits about twice as fast, but one of 8000 bits somewhat slower, against an estimate of its default time.
On the code of 1000 bits, raising `enumeration_batch_exponent` to 30 made it faster still, so enumeration one codeword at a time causes most of its time (verified, intervention).
Limitations: one run each; this and the other measurements of enumeration were taken while enumeration handles rows of more than 960 bits one codeword at a time (see [`enumeration_batch_exponent`](#enumeration_batch_exponent)), and should be checked again once that is fixed.

## enumeration_max_dimension

Value: 20, chosen by hand, so no rule chose it.
Reason: it is the largest value at which no traced long run enumerates where information sets are faster (verified, traces).
Evidence: traces (the commands of the windows in section 7) show that every value other than 20 changes the route of some traced search; at 21, sparse codes of 1000 and 4000 bits with 21 rows enumerate, which took them about 20 and 2 times as long as information sets in `p2-factor` (four at a time, earlier code).
In `p2-confirm` (eight at a time, earlier code), 18 made sparse codes of 1000 bits with 19 and 20 rows tens of times faster, by seconds, since they then avoid enumerating rows of 1000 bits one codeword at a time (consistent), and moved quick runs both ways by milliseconds; `v-maxdim` (one at a time, earlier code) traced two of those quick slowdowns, of about 1.3 to 1.4 times, to their routes (verified).
Limitations: the enumeration caveat of [`allow_enumeration_by_work`](#allow_enumeration_by_work), under which lower values favor codes of about 1000 bits.

## enumeration_setup_cost_factor

Value: 16.
Reason: rule (2), among 4, 16, and 128: against 128, a large gain on sparse codes of 1000 and 4000 bits with 21 to 23 rows, at a loss of at most about a second on random codes of those lengths (verified).
Trade-off: higher values enumerate more long codes of low dimension, which slows sparse codes far more than it helps random ones, while rows of more than 960 bits enumerate one codeword at a time.
Evidence: `p2-factor` (four at a time, earlier code), 16 against 128: most of those sparse codes ran from about 2 to 20 times faster and the rest alike, random codes of 1000 and 4000 bits at most about a second slower, and those of 8000 bits alike.
`p2-confirm` (eight at a time, earlier code), 4 against 16, made some codes of 4000 and 8000 bits up to about 3 times faster and others up to about twice as slow; `c4-setup` (one at a time), at 8, made a sparse code of 8000 bits almost twice as fast.
What the factor stands for, the time to set up information sets divided by the numbers of rows and columns and by the time per enumerated codeword, ranged from below 1 to about 100 across codes (verified; the row `enumeration_setup_cost_factor` of section 7), since long rows enumerate one codeword at a time.
Window: every traced search takes the same route from about 12 to 25; past either edge, a value helps some codes and slows others, under the enumeration caveat of [`allow_enumeration_by_work`](#allow_enumeration_by_work).

## enumeration_batch_exponent

Value: 15, chosen by hand, so no rule chose it.
Its measurements verify the cause of the enumeration caveat: rows of `w` packed 64-bit words are vectorized over at most `2**(15 + 1 - w)` combinations of rows (`_brute_force.py`), so rows of more than 960 bits enumerate one codeword at a time.
Evidence: `trial.py` with enumeration forced on a random code of 1000 bits (the row `enumeration_batch_exponent` of section 7): at 30, each codeword took a few hundred times less time than at 15, nanoseconds instead of microseconds (verified, intervention).
Limitations: one code, with enumeration forced; no value between 15 and 30 was timed, and at 30, rows of 4 words would take arrays of 4 GiB, and rows of more than 1920 bits would still enumerate one codeword at a time (code).
Follow-up: fix the batch sizing of enumeration, and time the workloads that enumerate before and after.

## max_information_sets

Value: 16.
Reason: rule (3): 16 and 32 measured alike, and 16 builds the smaller families.
Evidence: `p3-aut` and `p3-noaut` (quick runs, six at a time, earlier code): 32 measured alike with 16, and 8 moved quick searches both ways, slowing some by up to about 1.5 times.

## disjoint_random_column_orders

Value: 8.
Reason: rule (3): 0, 8, and 16 measured alike, and 8 is their middle.
Evidence: `p3-aut`, `p3-noaut`, and `p3-orders-aut` (quick runs, six at a time, earlier code): 32 and 64 made a few quick searches from a few percent to about 15% slower; differences this small also appeared when a tuning was compared with itself (`noise-j1`), so they may be chance.

## overlapping_minimum_work

Value: `2 * 10**4`.
Reason: rule (1), the maintainer's decision: on a coset search, the value sets the target weight of its other families, the smallest weight from 2 whose queries predict at least that many examinations (verified: `coset_families.py`, `_brouwer_zimmermann/coset_routes.py`, and timings on both sides).
`2 * 10**4` is a round value in the narrow range, roughly from `1.6 * 10**4` to `3.4 * 10**4`, over which the CSS coset searches of four codes build the same families (verified by the fingerprints of `coset_families.py`).
Coset searches serve mostly moderate codes queried many times, as in certification and the ranking of circuits, so the upper edge, set by a moderate code, matters most.
Evidence: `coset_families.py` (the row "Families of `overlapping_minimum_work`" of section 7): the lower edge is the predicted cost of a query of weight 2 of a large code, such as `qt7`, and the upper edge that of a query of weight 3 of a moderate code, such as `pp110`.
Below the range, `c7-coset` (one at a time) and `p6-long` (four at a time, earlier code): `10**4` made the long CSS coset searches of two large codes about 2 to 3 times slower than `10**5`, as predicted before `c7-coset` (verified).
Above it, `p6-short` (four at a time, earlier code): `10**5` made quick CSS coset searches of two moderate codes about 1.5 to 2 times slower than `3 * 10**4`.
Held out from the choice, `c8-usecase` (one at a time): two workloads of fault-propagation certificates measured alike at `10**4`, `2 * 10**4`, and `10**5`, as predicted; that their routes keep the disjoint family is consistent, not traced.
Distance searches make the same traced decisions from about `10**5` to `10**6`, below which the value lies, so it changes decisions of some quick distance searches; values from `10**3` to `10**5` timed within a few percent of each other on them (`p3-aut`, `p6-short`, earlier code).
Limitations: the range comes from the query costs of four CSS codes, and other codes may place it elsewhere; the route's own work estimate prefers the slower family of one large code, which values below the range build, so the value works there by never building it.

## overlapping_attempts_per_set

Value: 2.
Reason: rule (3): 1, 2, and 4 measured alike on quick runs (`p3-aut` and `p3-noaut`, six at a time, earlier code), and 2 leaves a second order when the first repeats an existing set; it was not timed on long runs.

## partition_work_ratio

Value: 500, so that its product with `partition_cost_factor` is 5000.
Reason: rule (3), the maintainer's decision: no product separates the searches that partitioning slows from those that it speeds up, since their gate values overlap, and 5000 is a round value that misclassifies the fewest traced searches.
Trade-off: a lower product partitions more searches, which speeds up some random classical codes and wastes time on searches that switch to automorphism images, since the switch discards the family (verified: a trace and `_brouwer_zimmermann/distance.py`).
Evidence: `trace.py --setup-only` on lifted-product codes of ten seeds and random classical codes of five (the row "Gate values of partitioning across codes" of section 7): the gate values of codes with the same parameters scatter over several orders of magnitude, so where they fall is partly chance.
`c1-gate` and `c1-low` (one at a time): partitioning made some quick searches of lifted-product codes about 3 to 4 times slower and quick searches of random classical codes of 128 bits up to about 1.7 times faster; two long runs of lifted-product codes measured alike with it off.
In `c9-decided`, with the other decided values, 5000 made a quick lifted-product search that it no longer partitions about 3 times faster than 1000, as predicted.
Traces of three searches (the commands of the windows): the work estimate at the gate exceeded the work done tens to hundreds of times, since it assumes that a search finds no lighter word and counts supports that completion tables and automorphism images skip (verified), so the product bounds no share of wasted time.
Window: the traced long runs decide the same for every product from about 30 to `10**5`.
Limitations: the gate cannot separate the traced searches, and the value rests on where the gate values of a few codes fell.
Follow-up: decide whether to switch to automorphism images before partitioning.

## max_partition_coverage

Value: 4, chosen by hand, so no rule chose it.
Reason: lowering it raises every gate value, so its effect runs through the partition gate (see [`partition_work_ratio`](#partition_work_ratio); verified by `c2-coverage`).
Evidence: `c2-coverage` (one at a time, at the product 1000): 3 and 2 moved quick searches both ways, by up to about 2.5 times, mostly by changing which searches pass the gate; a prediction that 3 lies between 4 and 2 on a random classical code was refuted in its wording.
Limitations: measured at the product 1000, not 5000, and not on long runs.

## partition_cost_factor

Value: 10.
Reason: rule (1), a direct measurement of the cost of partitioning in its units, of which the value takes the higher end (verified, two searches).
Evidence: `trace.py` of a lifted-product code and of a long run of the gross code (the row "Partitioning: estimates, costs, and gates" of section 7): partitioning took about 10 units on the first and about 4 to 5 on the second.
Limitations: the measurements spread about twofold, over two searches; with a factor of 20, the gate would decide differently on two quick searches (consistent, untested).

## lp_minimum_work

Value: `10**9`.
Reason: rule (3): `10**8`, `10**9`, and `10**10` measured alike, and `10**9` is their middle.
Evidence: `p3-aut` and `p3-noaut` (six at a time, earlier code): the three measured alike but for tens of milliseconds on one quick search; the other timed searches estimate their work far from these values.
The `linear program` events of `trace.py` at the defaults show a few linear programs per search, of milliseconds each, at most a few percent of a quick run, which refuted a prediction of under 1%.
Window: every traced workload decides the same from about `4 * 10**8` to `2 * 10**9`, and the traced long runs at every value up to about `10**13`.

## automorphism_minimum_layer_work

Value: `10**8`.
Reason: rule (3): `3 * 10**7`, `10**8`, and `3 * 10**8` measured alike, and `10**8` is their middle; they measured alike since the searches timed with them decide the same over that range (verified, traces).
Evidence: `p3-orders-aut` and `p3-aut` (six at a time, earlier code): the three measured alike on four quick searches of quantum codes, and `10**9` made three of them about 1.3 to 1.4 times slower.
A search decides once whether to switch (`_brouwer_zimmermann/distance.py`): at `10**7`, a quick search of a quantum Tanner code decided before it found its best word, kept its family, and ran about 7 times slower (verified: the row "`automorphism_minimum_layer_work` at `10**7`" of section 7, and `p3-aut`).
Window: on the traced long runs, from about `8 * 10**7` to `2 * 10**8`, narrow, though past either edge a long search switches only one layer sooner or later, a layer of a fraction of a second, with the same later decisions (the row "Past the edges of the windows").

## completion_bound_minimum_layer_work

Value: `3 * 10**7`.
Reason: rule (1): the threshold only bounds the cost of sampling a layer, since the cost model decides whether larger layers use tables; at `3 * 10**7`, the steps of a count of a moderate code, such as `count/pp110/12`, of about `4.5 * 10**7` supports, consider tables and run about 3 times faster (verified, trace and intervention).
Evidence: `p5-isolate-j1` (one at a time, earlier code), `10**8` against `3 * 10**7`: that count ran about 3 times slower, and a quick search of a random classical code with a layer of about `7 * 10**7` supports about 1.5 times slower; `p5-gates` (three at a time, earlier code): `10**7` measured alike.
Traces at the defaults with 256 samples (the row "Sampling of `completion_bound_minimum_layer_work`" of section 7): sampling a layer that declined tables took from under a millisecond to several milliseconds for each set, at most about 13% of a layer.
Window: the traced long runs make the same decisions to use tables from about `10**6` to `6 * 10**8` supports, so the value matters only for quick runs.

## completion_bound_max_cost_percent

Value: 90.
Reason: rule (1), the maintainer's decision: with 1024 samples, the predictions of the traced long-run layers near the limit leave a gap of several percent around 90 under all 40 sampling seeds tried, so none of those layers decides by its draw (verified, `resample.py`).
Trade-off: a higher limit gives tables to more layers, which pays where the cost model overestimates their cost, and loses where it underestimates it, or where the layer would otherwise scan several sets in lockstep.
Evidence: each set of a layer gets tables if its own prediction is within the limit, but that prediction compares its tables with a scan of that set alone, while the layer would otherwise scan its sets in lockstep (`_choose_tables_of_sets` in `_brouwer_zimmermann/distance.py`).
So at 100, a layer of a quick search of a random classical code gave tables to most of its sets, predicted just under 1.0, and ran about 1.4 times slower than its lockstep scan at 90 (verified: the row "Layer 7 of `distance/classical:128:80:6/-` at the limits of 90 and 100" of section 7, with interventions on the limit and the samples); in `c9-decided` the search was about 1.2 times slower, though timing alone (`c11-revert`) did not single out the limit.
`resample.py` at 100 (the row "Sampling of layers near the cost limit"): one long-run layer predicts on both sides of 1.0 under different seeds, so at 100 a long run decides by its draw.
`trace.py` and `fit_completion_costs.py` (the rows of the cost model in section 7): the cost model errs both ways, by about twofold on some traced layers (verified); that load causes the underestimates is a conjecture.
At 200, `costlimit-j1-f` (one at a time) made a long run of a random classical code about 1.4 times faster, since the model overestimates the cost of tables on bases of few rows (consistent), but `accept-isolate` (three at a time, earlier code) made a quick one about 1.3 times slower.
Limitations: the gap rests on a few traced long-run layers, and other layers can predict inside it; the decided values were not timed together with the limit at 90, and only the two searches that slowed in `c9-decided` were timed again (`c11-revert`).

## completion_bound_block_bits

Value: 16.
Reason: rule (1): with 8, the cost model declines the tables of the costliest layers of a quick search of a quantum Tanner code, which is then as slow as without tables (verified, trace and forced tables, the row "Predictions of the cost model, and `completion_bound_block_bits`" of section 7).
Evidence: on the large layers of the width data set, 16 skipped more supports, in less time, than 8 (earlier code); in `p5-screen` (quick runs, three at a time, earlier code), 8 and 4 made three quick runs on quantum codes from about 2 to more than 10 times slower, about as slow as without tables.

## completion_bound_fallback_block_bits

Value: 8, chosen by hand.
Evidence: `fit_completion_costs.py` (the row "Cost coefficients"), forced on the largest layers of the width data set: 8-column tables cost more than none on about half of them, by up to about 20%, and the cost model declines those tables; on the others they saved up to most of the time.

## coset_completion_block_bits

Value: 16.
Reason: rule (2), the maintainer's decision: a large gain on long CSS coset searches, at a loss on searches of Pauli errors by physical weight and more peak memory; the maintainer prioritizes CSS codes, and searches by physical weight are a niche.
Evidence: `p6-long` (four at a time, earlier code), against 8: the long CSS coset searches of two large codes ran about 8 and 30 times faster, and the long search by physical weight about 4 times slower; 4 was slower than 8 on all three.
`c9-decided` (one at a time), with the other decided values: CSS coset searches ran about 3 to 15 times faster, and searches by physical weight about 3 times slower on one code but faster on another; peak memory rose by about 1.5 GiB on the long CSS searches.
`c11-revert` (one at a time): on a quick CSS coset search of a moderate code, 16-column tables cost tens of milliseconds once `overlapping_minimum_work` gives more of its queries routes by tails (verified, intervention).
Limitations: on residual-weight estimates of state-preparation circuits, not in the repository, 16-column tables took longer to build than most estimates spent scanning; the long runs of 3000 queries were measured four at a time, on earlier code.
Follow-up: a coset table width for each kind of search.

## completion_bound_samples

Value: 1024.
Reason: rule (1), the maintainer's decision: more samples make the decisions of layers near the cost limit depend less on the sampling draw (verified, `resample.py`).
Evidence: `resample.py` on the traced long-run layers near the limit, under 40 seeds (the row "Sampling of layers near the cost limit"): with 1024 samples every layer decides the same under every seed, while with 256 one layer, which tables make about twice as fast, declined them under about 4 seeds in 10.
In `c6-samples` (one at a time), against 256, 512 measured alike, and 1024 made quick runs up to about 7% slower; a predicted bound of 5% was exceeded on one search, though the criterion of 10% held.
Limitations: it costs quick runs that sample layers up to a few tens of milliseconds; at a limit of 100, one long-run layer decides by its draw even with 1024 samples, so the reason holds for the limit of 90; not timed on long runs.

## completion_bound_sampling_work

Value: `5 * 10**7`, chosen by hand.
Evidence: `resample.py` with half and twice the budget (the row "Sampling of layers near the cost limit"): a traced long-run layer predicted the same, so the budget did not limit its samples (verified, with 256 samples only).

## completion_bound_confidence_z

Value: 2.0.
Reason: rule (3): 1.0, 2.0, and 3.0 measured alike on quick runs (`p5-sampling`, three at a time, earlier code), and 2.0 is a conventional confidence of about 98%.
Evidence: `resample.py` on a traced long-run layer near the limit: with 1024 samples, 1.0 and 3.0 keep its decision, though with 256, 3.0 declined its tables.

## completion_bound_lookup_block_cost

Value: 2.0, and 0.04 for each MiB for `completion_bound_lookup_block_cost_per_mib`, which this section also covers.
Reason: rule (1): fitted to the cost data set, in which lookups cost from about 2 to 8 scanned supports per block (earlier code); a refit on the refit data set gave lower costs, but the measured costs of the same lookups varied two- to threefold between sessions, so the measurements favor neither, and 2.0 and 0.04 stand.
Evidence: `fit_completion_costs.py` (the row "Cost coefficients"): the refit gives about 0.8 scanned supports per block plus about 0.04 for each MiB; lookups in small tables cost under one scanned support per block, and in tables of tens of MiB from about 1 to 8.
`resample.py` on a traced long-run layer near the limit, with 1024 samples: 2.0 and 0.04, the refit, and an earlier refit of 0.9 and 0.05 all keep its tables; with 256 samples, the earlier refit declined them, which made that layer about twice as slow in traces three at a time.
Limitations: one traced long run; the refit would also make another long-run layer use its tables, which was not timed.

## completion_bound_kept_first_row_ratio

Value: 0.35.
Reason: rule (1), a direct measurement: about the median of the ratios on the layers of the refit data set that scan one set with blocks of 16 columns (verified).
Evidence: `fit_completion_costs.py fit` (the row "Cost coefficients"): those ratios spread from below 0.1 to about 0.5, most between 0.3 and 0.5; blocks of 8 columns gave higher ratios, and the cost data set about 0.3 to 0.5 (earlier code).
`resample.py` on a traced long-run layer near the limit, with 1024 samples: 0.31 and 0.38 keep its decision.
Limitations: the ratios spread widely, and the value was not timed on long runs.

## completion_bound_table_byte_cost

Value: 0.1 for each byte, and 0.2 for each step for `completion_bound_build_step_cost`, which this section also covers, both chosen by hand.
Reason: chosen by hand; a direct measurement agrees with them for blocks of 16 columns, the default width, on quantum codes (verified).
Evidence: `measure_build` of `fit_completion_costs.py` on the data of the row "Cost coefficients": they predicted the builds of 16-column blocks on four quantum codes within about 10%, between about 0.8 and 2 times the cost on a random classical code, and about half the cost of 8-column blocks.

## information_set_batch_exponent

Value: 15.
Reason: rule (3): 15 is the smallest value beyond which larger ones were not clearly faster, one search at a time, and smaller values were slower by the overhead of returning to Python (verified, intervention).
Evidence: `c5-isbatch` and `c5-isbatch-high` (quick runs, one at a time): 11 to 14 made two searches up to about 20% slower, the more so the smaller the value, which refuted a prediction that they measure alike; 16 to 20 measured alike with 15 but for one small difference.

## scan_chunk_size

Value: 65,536.
Reason: rule (3): twice 32,768, the smallest value that measured alike with it one search at a time, a margin chosen by hand.
Evidence: `c5-chunk` (quick runs, one at a time): 16,384 made counts a few percent slower, which refuted a prediction that it measures alike; in `p1-chunk-fine` and `p1-chunk` (earlier code), larger values measured alike, and 4096 made a count about 9% slower.

## scan_chunks_per_call

Value: 64.
Reason: rule (3): powers of two from 4 to 1024 measured alike on compiled counts (`p1-chunks-per-call`, six at a time, earlier code), and 64 lies between them.

## max_css_single_pairs

Value: `2**18`.
Reason: rule (1): the table makes each query of weight one a lookup instead of a match over the qubits, and `2**18` is the smallest power of two that fits the pairs of the one workload whose searches read it on a state of more than 255 qubits, `hbound/pp372/2/2000` (verified, intervention on the field).
Evidence: `p6-short` and `final-short-f` (three or four at a time): against `2**16`, it made that workload almost twice as fast; on the other workloads that read it, every table fits under both values.
Limitations: the value fits one workload; a full table on 511 qubits takes tens of MiB (`tracemalloc`, the row "Memory of `max_css_single_pairs`"), so the cache of `PAULI_SEARCH_CACHE_SIZE` can hold about half a GiB of tables.

## Cache sizes

The cache sizes of `_tuning.py` are chosen by hand; these measurements describe them at their values.
Evidence: `cache_stats.py` (the row "Cache sizes") on seven workloads: the binomial cache used a few entries, and the caches of CSS and Pauli searches hit about half the time; a missed setup took milliseconds for a CSS pair and a fraction of a second for a search by physical weight of a large state; the memory of an entry of the caches of searches, which may hold completion tables, was not measured.

## Completion tables under concurrent load

These effects decide no value.
With tables wherever they are considered, a quick search of a random classical code ran about 1.3 times faster one at a time (`p5-load-j1`), about 1.1 times faster six at a time (`p5-load-j6`), and about 1.3 times slower three at a time under heavy load (`p5-gates`), all on earlier code (verified); that contention for caches causes this is a conjecture.

## The decided values together

`c9-decided` (one search at a time, under load) timed five decided values together, with the cost limit then at 100, against their values before the decisions, on 18 workloads, with predictions written first, and no result changed.
As predicted, the CSS coset searches of three codes ran from about 3 to 15 times faster, and the searches by physical weight of one code about 3 times slower, while about half the workloads, including those of fault-propagation certificates, measured within about 5%.
Two quick runs slowed, which refuted predictions, and were traced: a search of a random classical code, about 20% slower, from the cost limit at 100 (see [`completion_bound_max_cost_percent`](#completion_bound_max_cost_percent)), and a CSS coset search of a moderate code, about 15% slower, from the coset table width (see [`coset_completion_block_bits`](#coset_completion_block_bits)).
The final set, with the limit back at 90, was not timed as a whole; only those two searches were timed again (`c11-revert`).

## The merge of main

This branch merged `main` at `32a3f717e`, which changed no field or default of `SearchTuning`, and a check written before the merge found that it changes no search decision: `trace.py` of 8 distance searches and counts, compared by `windows.py --compare`, makes the same decisions before and after, and `coset_families.py` gives 5 coset searches the same families.

## Known limitations

- The work estimate of a search, which several gates compare with costs, can exceed the work done tens to hundreds of times (see [`partition_work_ratio`](#partition_work_ratio)).
- The cost model of completion tables errs both ways, by about twofold on some layers, and overestimates the cost of tables on bases of few rows.
- Layers of several sets compare the tables of each set with a scan of that set alone, not with their lockstep scan; at the limit of 90, no traced layer loses by it.
- A distance search decides once whether to switch to automorphism images, so deciding too early can make it several times slower; no traced search loses by it at the default.
- The work estimate of the routes of coset searches prefers the slower family of one large code (see [`overlapping_minimum_work`](#overlapping_minimum_work)); no traced search loses by it at the default.
