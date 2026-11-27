# Brouwer-Zimmermann distance search in qLDPC: the source code and its requirements

This file holds the parts of the guide to the Brouwer-Zimmermann (BZ) search on the source code and the correctness requirements; [brouwer_zimmermann.md](brouwer_zimmermann.md) holds the rest, which explains the algorithm and its extensions, and the references.
Like code comments, this file calls a search basis a **set** for short.

<a name="sec-source-code"></a>
## Guide to the source code

[README.md](README.md) says what each module of this directory holds, in "[Where to look](README.md#where-to-look)", and in which levels the modules import each other, in "[How the modules depend on each other](README.md#how-the-modules-depend-on-each-other)".
The docstrings of the modules and of their functions say which code does what, and cite the sections of this guide that explain it.
A flow that spans modules is described in the docstring of the module that coordinates it.
For example, [_brouwer_zimmermann/distance.py](_brouwer_zimmermann/distance.py) says how a BZ distance search prepares, exits, and advances by layers.

The thresholds, limits, and coefficients of the heuristics are fields of `SearchTuning` in [_tuning.py](_tuning.py): [Heuristics and cost models](#sec-heuristics) gives the reasons for those of BZ searches.

<a name="sec-coset-searches"></a>
### Coset searches

`CosetLeaderSearch` of [coset_leaders.py](coset_leaders.py) applies BZ to cosets `u + V`, unless it is built with `method="connected_cluster"`.
A query of `CosetLeaderSearch`, whether the coset of a syndrome has a word within a maximum weight, runs one or two **coset searches**.
They are a search of Pauli errors or, for the CSS weight `max(|x|, |z|)` on a CSS state, searches of the X and Z parts of the errors, each in a coset of a classical code ([problems.md](problems.md#the-coset-leader-search)).

Each family of search bases of a coset search is packed with its full rows, and with the certificate of its ranks and maximum coverage ([Families for coset searches](brouwer_zimmermann.md#sec-coset-families)).
That certificate takes no weight divisor, even when the disjoint packing of [Heuristics and cost models](#sec-heuristics) infers one for `V`, since the words of a coset need not share it ([Rounding the certificate to a proven weight multiple](brouwer_zimmermann.md#sec-weight-multiples)).
If `V` is even, a search can round its bounds up to the parity of a coset.
It does so when some coset is odd, which holds exactly when some word whose syndrome has one set bit is odd.
In a search of Pauli errors, an error weighs at least half the Hamming weight of its binary word, so the search has the **weight factor** `η = 2`, by which it divides its bounds, rounding up, after any rounding to parity; a search of X or Z parts has `η = 1`.

A query starts from the **word of its syndrome**, the sum of the words with unit syndromes ([_cosets.py](_cosets.py)) for the set bits of its syndrome.
It reduces that word to the offset of a search basis only when it first reaches the basis, since many queries end at an early search basis of layer zero.
It stops at its **stopping step**, the first step whose bound exceeds the maximum weight, unless it finds a light enough word sooner.
[_information_sets/certificates.py](_information_sets/certificates.py) finds stopping steps by a binary search of the steps, whose bounds never decrease, and also adds up the work of the steps ([Work estimates and exhaustive enumeration](brouwer_zimmermann.md#sec-work-estimates)).

Each family of a coset search keeps the bounds of its steps in a **planning buffer**: one table for each parity of the cosets if the search rounds to parity, and one otherwise, and a query takes the table of its coset's parity.
Each table has an extra row that caches one stopping step.
For a nonnegative maximum weight, a query reuses the cached step if that step is also the stopping step of the weight, and otherwise finds the step and caches it.
Callers share the coset searches, which a cache keeps, with the planning buffers of their families.
The batch loops of coset searches run without the global interpreter lock ([Behavior spanning files](#sec-cross-file-behavior)), so each call writes only buffers that no other running call uses.
A batch of `find_first_not_within_max_weight` gives each of its workers, one on one thread, its own copy of the planning buffer, whose cached step serves only the queries of that worker; the copy holds a few small tables.
The copy starts from the initial step `(0, 1)`, rather than the cached step of the shared buffer, since NumPy may copy without the global interpreter lock while a single query writes that step; so the copy depends on nothing that another thread writes.
A batch of `get_min_weights` caches no steps, and writes none of the buffers that it shares.
A single query calls its compiled search from Python without releasing the global interpreter lock, so its compiled calls hold the lock while they read and write the shared buffer.
Two single queries interleave those reads and writes only when the search runs as plain Python or in free-threaded Python.
A cached step is written as two entries, so an interleaved read can pair the layer of one step with the number of completed sets of another.
Every step that the binary search of steps returns, like the initial `(0, 1)`, has a layer of the table and from one to all of its sets completed, so any such pair `(w, j)` is a valid entry of the bounds, which never change, and so is `[w, j - 1]`.
A query reads each entry once and reuses the pair only if those bounds show that it is the stopping step, which is unique since the bounds never decrease; otherwise it finds the step itself.

A single query first looks up the words of weight at most one.
The lookup settles the query if it finds such a word, or if the maximum weight is below two.
Otherwise, and for every batch query, the query takes the **route** of its maximum weight: a family of search bases, with its packed rows and planning buffer, which [_brouwer_zimmermann/coset_routes.py](_brouwer_zimmermann/coset_routes.py) chooses once for each maximum weight ([Heuristics and cost models](#sec-heuristics)).
A batch query takes the route once for the whole batch, and only then splits the batch across threads ([Behavior spanning files](#sec-cross-file-behavior)).
For the CSS weight on a CSS state, `CosetLeaderSearch` gives the X parts and the Z parts of a batch to their own searches and routes.

On a **route by tails** ([Heuristics and cost models](#sec-heuristics)), the queries weigh words by their tails ([Packed enumeration with pivot-column deletion](brouwer_zimmermann.md#sec-packed-enumeration)).
They pass the family's **record of tails** to the compiled searches, which Numba compiles separately for searches with and without a record, so the choice costs nothing for each word.
The record holds the rank of each search basis, the tails of its rows, and the tails of the offsets of the words with unit syndromes on it.
Reducing a word to its offset is linear, so a query finds the tail of its offset by summing these tails for the set bits of its syndrome, without reading the pivots.
The record also holds the completion tables that the routes of the family ask for.
The routes build the tails of a family, and the tables of each search basis, once, for the first route that needs them.
They build new tables in the array of a new record, which copies the earlier tables; the old record keeps those until the routes replace it.
A query by tails skips completions by tables in the layers of two or more rows of a search basis that has tables, and scans its other layers without tables.
It rebuilds its lightest word once, with the full rows, from the word's search basis and support.
The search of Pauli errors by CSS weight has no routes by tails, since the CSS weight of a word depends on which of its deleted pivot columns lie in the X part.

<a name="sec-heuristics"></a>
### Heuristics and cost models

Every family gets the certificate of its own verified ranks and maximum coverage, so the heuristics below change how much work a search does, not whether its certificate holds; with a cutoff above one, they can change which witness at or below the cutoff it returns ([Behavior spanning files](#sec-cross-file-behavior)).
Searches read the heuristics' fields as README.md's [Tuning](README.md#tuning) section says.
A heuristic must also keep the requirements on threads of [Correctness requirements](#sec-requirements), such as "Decide only from what a layer leaves the same on any number of threads".
Some fields are thresholds of gates on optional steps, such as building a family of search bases, asking for automorphisms, or predicting the cost of completion tables.
A step can save at most the work that it acts on, so its gate lets it run once an estimate of that work, in candidate/set evaluations, reaches the cost of the step, which the threshold need match only to an order of magnitude.
The Retuning section of [_tuning.py](_tuning.py) defines the other classes of fields, such as budgets and rough coefficients, and says how the defaults are set and when to change them.

Distance searches and counts build their families in the stages of [_information_sets/basis_families.py](_information_sets/basis_families.py): a disjoint packing, its pruning, an overlapping family, and a family of bounded coverage by matroid partitioning.

- **The disjoint packing** packs at most `2^r - 1` information sets greedily in natural column order.
  The layers of one set already reach every nonzero combination of rows, so a layer of more sets would examine more candidates than there are words, and the limit keeps the memory of the packing from growing with the square of the length.
  If it packs fewer information sets than `min(2^r - 1, floor(N / r))`, it tries up to `disjoint_random_column_orders` random orders from a fixed seed, a budget of setup effort since each costs another packing, and keeps the family with the lexicographically largest tuple of ranks.
  It also infers the divisor `D = 2` of [Rounding the certificate to a proven weight multiple](brouwer_zimmermann.md#sec-weight-multiples).
- **Pruning** loses nothing: it removes a search basis of rank deficiency `d = r - r_i`, whose term `b_i(t)` is zero for `t < d`, when the information sets alone certify `U` by layer `d`.
- **The overlapping stage** builds search bases on column orders that put the least-used columns first, with ties broken by a random number generator, and keeps the cheapest family built.
  Each run of the stage gets its own generator with the same fixed seed, so every run builds the same family from the same input.
  It makes up to `overlapping_attempts_per_set` attempts, each a row reduction, for each set that it may add, up to a family of `max_information_sets` sets.
  A repeated pivot set still consumes an attempt.
  Both fields are budgets of setup effort, since more sets can avoid costly layers but take more attempts.
  It builds nothing if the family that it is given has no information set, or at least `max_information_sets` of them, or information sets that use every nonzero column, or if no family of at most `max_information_sets` sets could do less work.
  Its gate, `overlapping_minimum_work` on the estimated work of the given family, breaks even with the cost of the attempts.
  The routes of coset searches (below) gate the stage with `coset_route_minimum_work` instead, so they run it with a minimum work of zero.
- **The partitioning stage** finds the largest number of information sets, up to `m = max_information_sets`, in which no column appears more than `b` times.
  It uses the matroid partitioning of Section 5.1 of Lisoněk and Trummer ([References](brouwer_zimmermann.md#sec-references)) in [_information_sets/matroid_partition.py](_information_sets/matroid_partition.py).
  It may try each coverage `b` from two up to `ceil(m*r/N)`, the least at which `m` sets could fit by counting their pivots.
  A search takes the family of least work, if it does less work than the family of the earlier stages.
  Partitioning at `b` finds at most `k_b = min(m, floor(b*N/r))` sets, since at most `floor(b*N/r)` fit within coverage `b` by counting their pivots.
  A family of `j` information sets puts some column in at least `ceil(j*r/N)` of them, so at each step its coverage-divided certificate is at most that of `j` sets of that coverage.
  The work up to a step depends only on the step and the number of sets, so the family does at least the work of those `j` sets.
  Let `W(k)` be the least such work over `j` from one to `k`, which the stage computes.
  Partitioning skips each `b` at which `W(k_b)` is at least the work of the best family so far, at first the family of the earlier stages.
  No family that it finds at such a `b` could replace that family, so the search takes the same family as if partitioning tried every `b`.
  The cost of trying every `b` from two up to `ceil(m*r/N)` is roughly proportional to `S`, the sum of `b^2*r^2*N` over them, which grows with the cube of `m`.
  So a search tries partitioning only if some family of at most `m` sets could do less work, and the search's estimated work is at least `partition_work_ratio * S`.
  `partition_work_ratio` stands for the cost of one unit of `S` in evaluations.
  The gate charges all of `S`, though partitioning may skip some `b`.
  This gate is one of the two known weak spots of the heuristics.
  The actual cost of one unit of `S` varies widely, even between codes of the same size.
  The estimated work of a search can also far exceed the work that it does ([Work estimates and exhaustive enumeration](brouwer_zimmermann.md#sec-work-estimates)), so a search can pay for partitioning and then need little work.
  The limit for experiments `C_max = partition_max_coverage` caps `m` at `floor(C_max*N/r)`, the largest number of sets that fit within coverage `C_max` by counting their pivots.
  So partitioning tries no coverage above `C_max`.
  At `C_max = 1`, partitioning instead builds the maximum disjoint family of the routes of coset searches (below), but with quotient labels, which the routes omit.
  It prunes that family as the pruning stage does.
  `m` does not limit that family, and `S` is `r^2*N`.
  The test of whether some family could do less work considers the disjoint families that partitioning can build.
  At coverage one, partitioning, like a packing, keeps a rank-deficient block only below `2^r - 1` information sets, so such a family holds at most `min(ceil(N/r), 2^r - 1)` sets, counting that block.
  It does no less work than as many information sets of coverage one, whose certificate is at least as large at each step, since the work up to a step depends only on the step and the number of sets.
  Let `f = min(floor(N/r), 2^r - 1)`, the largest number of information sets that fit, up to the limit of a packing.
  The **disjoint rank bound** is `f` information sets and, if `f < 2^r - 1` and `N > f*r`, a rank-deficient block of rank `N - f*r`.
  By counting pivots, it bounds the ranks of every maximum disjoint family lexicographically.
  Partitioning skips coverage one if the family `F` of the earlier stages is disjoint and has the ranks of this bound, pruned for `U` as the pruning stage prunes a family.
  Every maximum disjoint family then has `f` information sets and a block of rank at most `N - f*r`, but at least that of any block of `F` (see the routes of coset searches below).
  Pruning drops a block of lower rank if it drops one of rank `N - f*r`, since the lower rank gives a larger deficiency `d`, and information sets that certify `U` by one layer also certify it by every later one.
  Once pruned, a maximum disjoint family therefore has the ranks of `F`, does the same work, and cannot replace `F`.
- **Automorphism families** ([_information_sets/automorphism_families.py](_information_sets/automorphism_families.py)) row reduce in natural column order and balance the shares of the orbits ([Images of an information set under automorphisms](brouwer_zimmermann.md#sec-automorphisms)).
  The gate `automorphism_minimum_layer_work` of [Images of an information set under automorphisms](brouwer_zimmermann.md#sec-automorphisms) breaks even with the cost of finding automorphisms, verifying them, and building their family.
- **The routes of coset searches** ([_brouwer_zimmermann/coset_routes.py](_brouwer_zimmermann/coset_routes.py)) choose the route of each maximum weight `ω` of a coset search ([Coset searches](#sec-coset-searches)).
  The **predicted examinations** of a query at `ω` with a family, without completion tables, are one more than the largest work at `ω` over its planning buffer's tables, which bounds the words that the query examines or skips ([Work estimates and exhaustive enumeration](brouwer_zimmermann.md#sec-work-estimates)).
  1. If the predicted examinations with the disjoint family are below `coset_route_minimum_work`, the route is the disjoint family.
  2. Otherwise, the routes try once to build two families for `ω_T`, the least weight from two at which the predicted examinations with the disjoint family reach the threshold, so that they do not depend on the order of queries.
     The first, the **maximum disjoint family**, which matroid partitioning with coverage one finds, holds the largest number of disjoint information sets, up to the limit of the disjoint packing.
     If it holds fewer than that limit, it also holds a rank-deficient block of the largest rank on the other columns, if any are left ([Search bases and rank-deficient blocks](brouwer_zimmermann.md#sec-search-bases)).
     It is kept if it has more information sets, or a rank-deficient block of larger rank, than the greedy packing.
     Edmonds' matroid partitioning algorithm maximizes the number of columns in its independent column sets.
     All but the last of these sets are information sets, so no other choice of as many information sets leaves a block of larger rank.
     A greedy packing can miss information sets.
     For example, the columns `e1 = (1, 0)`, `e2 = (0, 1)`, `e1 + e2`, and `e1 + e2` hold two disjoint information sets, but a greedy packing in natural order takes `e1` and `e2`, and leaves a rank-deficient block of rank one.
     The second is a family of the overlapping stage, aimed at the Hamming weight `η ω_T + 1`.
     It starts from the first family if that was kept, and otherwise from the disjoint family, and it is kept if it does less work than the family that it starts from.
  3. The route is the family of least work at `ω`, summed over its planning buffer's tables, and on ties the disjoint family, then the maximum disjoint one.
     A search for minimum weights up to `ω` takes the route of `ω`, or of the number of qubits if that is smaller, since no coset has a larger minimum weight.
  4. In a search by Hamming weight ([Terms](README.md#terms) in README.md), the route is a route by tails if its own predicted examinations also reach the threshold.
     Its queries then weigh words by their tails, and skip completions by tables on blocks of `coset_completion_block_bits` columns, for every search basis that they scan in layers of two or more rows, if the tables fit in `coset_completion_max_table_bytes`.
     The threshold and this limit replace the minimum layer work and the cost model of the completion tables of distance searches.

  Building the two families costs a matroid partitioning and the row reductions of an overlapping family; weighing by tails adds a fixed cost to each query, for the tail of its offset in each search basis that it reaches and the rebuild of its lightest word; and tables cost memory and a build.
  Only queries that examine many words repay these costs.
  The threshold also sets the aim `ω_T`, which no break-even rule covers, so its default is chosen by hand.
- **The choice of completion tables** ([_information_sets/completion_bounds.py](_information_sets/completion_bounds.py)) decides, by the steps of [Skipping completions by completion tables](brouwer_zimmermann.md#sec-completion-tables), whether a distance search uses tables on a search basis in a layer, or a count on a step.
  Its gate `completion_bound_minimum_layer_work` breaks even with the cost of the prediction, whose sampling stays within the budget of setup effort `completion_bound_sampling_work`.
  The prediction takes the share that tables skip at the lower end of a Wilson score interval of `completion_bound_confidence_z` standard errors, or as zero with fewer than `completion_bound_min_samples` samples, conventions that keep sampling error from choosing tables that cost more than they save.
  The limit `completion_bound_max_cost_percent` compares two estimates in one unit, so it breaks even at 100 percent.
  The rough coefficients of its cost model are the costs `completion_bound_lookup_block_cost`, `completion_bound_lookup_block_cost_per_mib`, `completion_bound_table_byte_cost`, and `completion_bound_build_step_cost`, in scans of one support of a distance search, and the multiplier `completion_bound_kept_first_row_ratio`; a count scans full rows and labels, which cost more, so the model is conservative for it.
  In a layer of `w` rows of a search basis of `r` rows, the model predicts `k (1 - q) binom(r, w - 1) + binom(r, w - 2)` lookups, where `k` is the multiplier and `q` is the predicted share of supports that the tables skip.
  The script `experiments/distance_tuning/constants.py` measures `k` by solving this for `k`, with the lookups that it counts and the share `q` that the tables skipped: `(lookups - binom(r, w - 2)) / max(1, (1 - q) binom(r, w - 1))`.
  The model's decisions near break-even rest on these rough coefficients and can go either way, which is the other known weak spot of the heuristics.
  The widths `completion_bound_block_bits` and `coset_completion_block_bits` default to the widest that the tables accept, since a wider block never gives a weaker bound.
  Sampling only predicts the value of tables; every skip rests on a computed bound.
- **The estimates of work** of [Work estimates and exhaustive enumeration](brouwer_zimmermann.md#sec-work-estimates) and of an automorphism family ([Images of an information set under automorphisms](brouwer_zimmermann.md#sec-automorphisms)) both come from the plans of steps of [_information_sets/certificates.py](_information_sets/certificates.py).

<a name="sec-cross-file-behavior"></a>
### Behavior spanning files

- **Cutoff.**
  The docstring of `get_distance_from_generators` in [distances.py](distances.py) states the contracts of `cutoff` and of its early exit, which the other distance functions share.
  The module docstring of [_brouwer_zimmermann/distance.py](_brouwer_zimmermann/distance.py) lists where a BZ search checks `cutoff`.
- **Weight limit.**
  The docstring of `get_distance_from_generators` also states the contract of `max_weight`.
  The functions pass `max_weight+1` to BZ as an upper bound, as though a witness of that weight were known, so that it stops once its certificate reaches that bound, and clamp the results of the other methods to `max_weight+1`.
  They also cap `cutoff` at `max_weight`, since a higher cutoff would end the search at once on the assumed witness.
  Inside the searches, a sector without nontrivial words counts as `n+1`, which no word on `n` bits or qubits weighs, so that it never lowers the minimum over sectors.
- **Units.**
  Binary input (`symplectic=False`) is a classical problem, or a pure-X or pure-Z problem with one bit per qubit, and symplectic input (`symplectic=True`) has general Pauli rows.
  For symplectic input, BZ maps rows by `T` ([Pauli weight as Hamming weight](brouwer_zimmermann.md#sec-pauli-weight)), doubles `cutoff` and the bound of a weight limit, passes the proven weight divisor two ([Rounding the certificate to a proven weight multiple](brouwer_zimmermann.md#sec-weight-multiples)), and halves the result, while `method="brute_force"` uses symplectic weights and the original `cutoff`.
- **Independence.**
  For every BZ and brute-force search of the distance functions of [distances.py](distances.py), including those that `method="auto"` runs through [_auto.py](_auto.py), [distances.py](distances.py) makes the generators of the caller independent.
  It reduces them with [qldpc/_binary.py](../../_binary.py), or builds the adapted basis of [_code_matrices.py](_code_matrices.py) when the caller gives `S`.
- **Resumable layers.**
  A compiled scan of a distance search returns after a batch of at most `2^search_chunk_exponent` supports, or, with completion tables, that many supports and lookups together.
  Counting lookups keeps the scan responsive while it skips most supports.
  Successive batches resume from the position of the scan, its support and the XORs of its prefixes.
  On several threads, each thread has its own position, and a chunk of each thread holds at most `num_threads` times a batch of supports and lookups.
  Each thread holds the global interpreter lock between its chunks, so chunks that grow with the number of threads keep the share of time under the lock about constant, at the cost of a longer wait for a keyboard interrupt.
  Each advance of the search of a sector scans one layer, or its leading search bases.
  Once the search bases that an advance scans have completed a layer, it may raise the lower bound of the search ([The stopping certificate](brouwer_zimmermann.md#sec-stopping-certificate)), but it moves on to the next layer only once every search basis has.
  Before a costly layer, an advance may get automorphisms and switch to an automorphism family, which restarts the search at layer two and may raise the lower bound ([Images of an information set under automorphisms](brouwer_zimmermann.md#sec-automorphisms)).
  If a scan stops at the cutoff or raises an exception, the advance leaves the lower bound and the next layer as they were before that scan, but keeps any earlier switch.
  A later advance restarts the layer from its first support, unless the search has finished, as it has after a stop at the cutoff.
  If an advance raises, a witness that the interrupted scan found can stay out of the `U` that the sectors share, so build a new search to search again.
- **Threads.**
  A search scans a layer of a set on `num_threads` threads if the layer has `w ≥ 3` rows and more than `num_threads` batches of supports.
  A search resolves `num_threads` once, from the tuning; README.md's [Threads](README.md#threads) gives its default.
  Any other layer, and every layer on one thread, is scanned on the calling thread.
  A **work item** is the set of supports of the layer whose first `k` rows are fixed.
  `k` is 2 if `w ≥ 4` and fixing one row leaves fewer items for each thread than a constant of [_threads.py](_threads.py) sets, and 1 otherwise.
  Since `k ≤ w - 2`, every completion of a prefix that completion tables skip ([Skipping completions by completion tables](brouwer_zimmermann.md#sec-completion-tables)) lies in the item of the prefix, so the items partition the supports that a scan of the layer weighs or skips.
  The items are numbered in lexicographic order, the order in which the scans step through a layer ([_information_sets/support_stepping.py](_information_sets/support_stepping.py)).
  Their numbers of supports vary widely, and earlier items tend to hold more, so ranges of equal numbers of items would not balance the threads.
  The threads, the calling thread among them, claim ranges of consecutive items in increasing order, weighted by the number of supports of each item, through [_threads.py](_threads.py), whose docstrings say how guided self-scheduling sizes the ranges.
  A thread scans its range with a compiled scan that Numba compiles without the global interpreter lock, and which stops once a support passes the last item of the range.
  Splitting depends on the order of the items: a range starts at its first item's first support, the end of a range is tested in the same order, and completion tables take pairs of rows in scan order.
  Starting a range sets the support and the prefix XORs only up to the depth at which the range starts, and leaves the other entries stale, so a scan must read no entry beyond its depth.
  The threads share one copy of the rows, labels, and tables of the set.
  They scan one set at a time, in the order of the sets, so each advance and its certificate are unchanged.
  Each thread lowers `U` under a lock after each chunk, and starts its next chunk from the `U` of that moment, which skips more than an older `U` would.
  Every value of `U` is the weight of an eligible word of some sector, or the caller's upper bound, so whichever value a chunk starts from, it skips only words that cannot lower `U`.
  After the layer, `U` is the least of its value before the layer and the weights of the eligible words of the layer, as on one thread.
  So every later decision, such as completion tables and the switch to an automorphism family, is that of one thread.
  The thread that brings `U` to the cutoff stops the others after their chunks.
  The witness, and so the result of an early exit, can then depend on the timing of the threads, though its weight still lies between the distance and the cutoff.
  An exception in any thread stops the others after their chunks, and so does one in the calling thread while it starts or waits for them, such as a keyboard interrupt.
  The exception is raised again once all threads have returned; only in narrow windows, which a comment in [_threads.py](_threads.py) describes, can a further interrupt end the search while threads still run.
  The process waits for them at exit, unless the search ran on a daemon thread, whose threads are daemons too.
- **Coset batches on threads.**
  `CosetLeaderSearch` fixes `num_threads` when it is built, and with it a **split work** of `num_threads << search_chunk_exponent` units.
  The comment of `search_chunk_exponent` in [_tuning.py](_tuning.py) defines these units for the BZ and connected-cluster methods.
  `find_first_not_within_max_weight` and `get_min_weights` of `CosetLeaderSearch` run each batch of a BZ coset search through [_threads.py](_threads.py).
  The calling thread searches the batch alone, in order, in one call of the compiled batch loop of its query, which stops once the work of its queries reaches the split work.
  So a batch of cheap queries, and any batch on one thread, runs on the calling thread in one call, and costs about as much as one call of the loop.
  Once a batch has shown that it is costly, the rest of it is split into contiguous blocks, about a fixed number for each thread, a constant of [_threads.py](_threads.py).
  The threads, the calling thread among them, claim ranges of blocks in increasing order, and search the blocks one at a time with the same loop.
  Each worker has its own buffer for the lightest word; Numba compiles the batch loops of both queries without the global interpreter lock, and the call on the calling thread compiles them before any thread starts.
  The query takes its route before the threads start, so no thread builds tails or tables, and the threads share the tables of the route, apart from the copies of its planning buffer that a batch gives its workers ([Coset searches](#sec-coset-searches)).
  The answer for a syndrome depends only on the code, the maximum weight, and the tuning ([Correctness requirements](#sec-requirements)), so the weights of `get_min_weights` are those of one thread.
  In `find_first_not_within_max_weight`, a syndrome without a word within the maximum weight ends the batch.
  On the calling thread, before the split, it ends the query.
  After the split, [_threads.py](_threads.py) keeps a limit, a block index that only falls.
  A block with a syndrome that ends the batch lowers the limit to the index of that block.
  The threads skip every block after the limit, including the unstarted blocks of ranges that they have claimed.
  A block runs in one compiled call, which never checks the limit: once it starts, it runs until its end or its first syndrome that ends the batch, even if the limit falls below its index meanwhile.
  Every block before the limit runs, so the first syndrome in batch order without a word within the maximum weight is the one that one thread finds.
  The returned bound and lightest word for that syndrome are also those of one thread.
  For the CSS weight on a CSS state, each batch query runs two batches in this way, one of X parts and then one of Z parts.
  In `find_first_not_within_max_weight`, the batch of Z parts holds the syndromes before the first whose X part has no word within the maximum weight, or every syndrome if all X parts have such a word.
  In `get_min_weights`, the batch of Z parts holds the syndromes whose X parts have a word within the maximum weight.
  An exception before the split, while no other thread runs, reaches the caller at once; one after it, in any thread, such as a keyboard interrupt in the calling thread, reaches the caller as the docstrings of [_threads.py](_threads.py) say, and the threads stop within their ranges, except in the narrow windows that they describe.
- **CSS sectors.**
  Each sector of a joint search keeps its own lower bound and the certificate of its own family ([Sharing witnesses between CSS sectors](brouwer_zimmermann.md#sec-shared-witnesses)).
- **Automorphisms.**
  The distance functions take automorphisms as an array or as a function, and pass them to their BZ searches as a function that calls the function of the caller at most once.
  `get_distance_exact` of `QuditCode`, `CSSCode`, and `ClassicalCode` in `src/qldpc/codes/common.py` pass a function that returns the permutation automorphisms that the code knows.
  The code caches these automorphisms, so it looks for them with bliss (`src/qldpc/external/_bliss.py`) at most once.
  The docstrings in `src/qldpc/codes/common.py` say which automorphisms a code knows and which it verifies before returning them.
  [Where automorphisms come from](brouwer_zimmermann.md#sec-automorphism-sources) explains where they come from.
  `ClassicalCode.get_distance_exact` passes none when it measures the distance from a vector, whose coset an automorphism of the code need not preserve.
  A search calls the function only at the points of [Images of an information set under automorphisms](brouwer_zimmermann.md#sec-automorphisms), and uses the generators only after [_information_sets/automorphism_families.py](_information_sets/automorphism_families.py) verifies them; for a CSS code, they must map the row spaces of both parity-check matrices onto themselves.
  For a general Pauli problem, [distances.py](distances.py) lifts them to the coordinates of `T`, except for Pauli strings that `get_distance_pauli` searches as a CSS code.

<a name="sec-requirements"></a>
## Correctness requirements

A change may use other representations, certificates, and search strategies, but must keep these requirements for the nontrivial problems of [The problem](brouwer_zimmermann.md#sec-problem):

- **Search `V \ S`.**
  Transformations must preserve `V` and `S`, or an equivalent representation of `V \ S`, such as the label test of [Quotient labels for the stabilizer test](brouwer_zimmermann.md#sec-quotient-labels).
- **Keep weights in one unit.**
  Candidate weights, certificates, cutoffs, and results must agree, which includes restoring the pivot ones ([Packed enumeration with pivot-column deletion](brouwer_zimmermann.md#sec-packed-enumeration)) and converting between Pauli and Hamming weights ([Behavior spanning files](#sec-cross-file-behavior)).
- **Use only justified certificates.**
  A certificate `L` must hold for every eligible word lighter than `U`, all of which are unexamined, and need not hold for heavier words, such as skipped ones, so a search reports the bound `min(L, U)`, not `L`.
  It relies on block ranks, coverage, and completed layers ([Search bases and rank-deficient blocks](brouwer_zimmermann.md#sec-search-bases), [The stopping certificate](brouwer_zimmermann.md#sec-stopping-certificate), and [Overlapping search bases](brouwer_zimmermann.md#sec-overlapping-bases)), and may overestimate coverage but never underestimate it; a coset search may use any family of its code, with that family's ranks and coverage.
  Rounding needs a congruence of every eligible weight: a common divisor, or, only in a coset search, where every word is eligible, the parity of a coset of an even code ([Rounding the certificate to a proven weight multiple](brouwer_zimmermann.md#sec-weight-multiples)).
  A certificate from another CSS sector needs a justification in the sector where it is applied, and a certificate from the images of an information set, as of an automorphism family, needs verified automorphisms and every examined eligible word to lower `U` ([Images of an information set under automorphisms](brouwer_zimmermann.md#sec-automorphisms)).
- **Skip only words that a bound proves heavy enough:** at least `U` in a distance search ([Skipping completions by completion tables](brouwer_zimmermann.md#sec-completion-tables)).
  Completion bounds need a minimum over every remaining pair of rows; disjoint blocks; exactly `w` pivot ones, as only in an information set, or the coset scans' bound on pivot ones; and the rows of their own search basis in scan order ([Skipping completions by completion tables](brouwer_zimmermann.md#sec-completion-tables)).
  Coset tails must keep every column but the pivots ([Packed enumeration with pivot-column deletion](brouwer_zimmermann.md#sec-packed-enumeration)).
  A coset query of maximum weight `ω` skips at `h(ω)`, never `h(ω) - 1`, and reports `R` only if it skipped a word.
- **Find minimum weights of cosets exactly up to `ω`.**
  A search may stop only at a step whose bound reaches `U`, or within a step once `U` is at most the bound of the step before or the floor of the search: `w_x` in the search of the Z part for the CSS weight ([Minimum weights of cosets](brouwer_zimmermann.md#sec-coset-min-weights)), and zero otherwise.
  Its scans may keep only words lighter than its incumbent, and its tables may skip only at the incumbent or at the weight of a lighter word that the scan has found ([Minimum weights of cosets](brouwer_zimmermann.md#sec-coset-min-weights)).
- **Keep the strict stop of counting.**
  A count of the words of weight `t` must also exhaust the words of weight exactly `t`, so it stops only at a step whose bound exceeds `t`, never at equality, and skips only completions heavier than `t` ([Skipping completions by completion tables](brouwer_zimmermann.md#sec-completion-tables)).
  A word of weight `t` is counted at the first step that visits it, and steps run in order, so a count that stops within a layer counts each such word once.
  The certificate of a count uses the ranks and actual coverage of its family, and the target `t+1` at which it aims the stages of basis families only aims them.
- **Keep coset queries reproducible.**
  The family and tables that answer a query must depend only on the code, the maximum weight, and the tuning, not on earlier queries, since the certifier's counterexamples report a query's word and bound.
  A query therefore uses tables only on the search bases that its own route asks for.
  The record of tails that it reads may also hold tables of other search bases ([Coset searches](#sec-coset-searches) and [Heuristics and cost models](#sec-heuristics)).
  - Since the batch loops of coset searches run without the global interpreter lock, every call of one, on any number of threads, writes only buffers that no other running call uses ([Coset searches](#sec-coset-searches)).
    A call writes only buffers that its own worker owns, and its own rows of an output that the calls of a batch share; the docstrings of the loops in [_brouwer_zimmermann/coset.py](_brouwer_zimmermann/coset.py) say which buffers each writes.
    A copy of the planning buffer that a worker owns starts from the first step, and no call writes the shared planning buffer.
    A test of [_brouwer_zimmermann/coset_test.py](_brouwer_zimmermann/coset_test.py) runs batches with every array of the search read-only, so that numba rejects a write to one in any branch of compiled code, and checks that the arrays are unchanged.
    Review checks the writes that this misses.
    Numba compiles some writes to read-only arrays anyway, such as those through `.flat`, `np.ascontiguousarray`, `np.asfortranarray`, `np.nditer`, or `np.fill_diagonal`.
    The test does not mark the buffers that a batch makes for its workers: the lightest word and the copy of the planning buffer of each worker, and the shared weights whose rows the calls write.
    The test also misses paths that the tests do not run with the arrays read-only, and, in plain Python, writes that tests do not run.
    Review checks that each worker writes only its own buffers, and each call only its own rows.
  - The routes of [_brouwer_zimmermann/coset_routes.py](_brouwer_zimmermann/coset_routes.py) choose each route once, under a lock, and there build the families other than the disjoint one, once, and each record of tails, and a batch query takes its route before its threads start ([Behavior spanning files](#sec-cross-file-behavior)).
    A record of tails never changes once a query may read it: a route that needs new completion tables gets a new record of its family, which copies the earlier tables and their layout ([Coset searches](#sec-coset-searches)).
- **Count only work without the global interpreter lock in the split of coset batches.**
  The batch loops count as work only what they do without the global interpreter lock, in the units that the comment of `search_chunk_exponent` in [_tuning.py](_tuning.py) defines ([Behavior spanning files](#sec-cross-file-behavior)).
  That work decides when [_threads.py](_threads.py) moves the rest of a batch to threads, so this rule keeps the speed of the split, not its answers, which are those of one thread on any split.
- **Keep the scans of a layer splittable across threads** ([Behavior spanning files](#sec-cross-file-behavior), "Threads").
  The scans are the compiled scans of a whole layer and of a range of work items of a distance search, in [_brouwer_zimmermann/affine_search.py](_brouwer_zimmermann/affine_search.py), which a change to their shared body reaches; the coset scans split no layer.
  A scan's position is its support, the XORs of its prefixes, and its depth, the number of rows it has fixed, and it reads no entry of the support or of the prefix XORs beyond that depth.
  Besides its position, it carries only values derived from its position and from data that no thread writes during the layer, such as the number of pivot rows among its fixed rows; the counts of its work; and its copy of `U`, which starts at the `U` it is given and only falls.
  It must visit the supports of each work item together, in the lexicographic order in which the distance search numbers the items, and skip only supports that share the first `w - 2` rows of its current support.
  Unless it stops at the cutoff, each chunk is exact from any starting `U'`.
  It returns the minimum of `U'` and the weights of the eligible words of its supports, and skips only words of weight at least its current `U`.
  Tests on threads and work items check that work items cover a layer once, in scan order.
  They also compare searches and scans on threads with those on one thread: their distances, their bounds after each layer, and, from a shared `U` that the layer does not lower, the totals of supports weighed and skipped and of bounds looked up.
  Tests of results can miss a race or an abort, so each of these rules says what checks it:
  - The compiled scans decide to skip or prune only from their position, the values they carry, and data that no thread writes during the layer.
    Review checks this.
  - The compiled scans write only to arrays that their caller gives the thread that runs them, such as its support and prefix XORs.
    A test of [_brouwer_zimmermann/distance_test.py](_brouwer_zimmermann/distance_test.py) runs the threads with every array of the search read-only, so that numba rejects a write to one in any branch of compiled code, and checks that the arrays are unchanged.
    It misses the same kinds of writes as the test of coset batches under "Keep coset queries reproducible".
    It also does not mark the arrays that each thread of a split layer gets for itself: the thread's support and prefix XORs, and the work items that it scans.
    Review checks the writes that the test misses, and that each thread writes only its own arrays.
  - Other code that threads run writes shared state only under a lock, through thread-safe objects, or under its own key.
    The shared `U`, the claims of items, the stop event, and the result of a row of a coset batch that ends the batch are written in these ways.
    Review checks this.
  - The shared `U` is lowered only as a minimum, under the lock.
    Review checks this.
  - Code that threads run calls no numba-parallel code, and reads the tuning only from its search.
    The section "Working here" of [README.md](README.md#working-here) states these rules for the whole package, why they hold, and which tests and reviews check them.
- **Decide only from what a layer leaves the same on any number of threads** ([Behavior spanning files](#sec-cross-file-behavior), "Threads").
  After the layer of a set, `U` and the lower bound are those of one thread, and so are the estimates derived from them, such as the work of the next layer and the work of an automorphism family.
  Every later decision, such as the sets that a layer scans, completion tables, the switch to an automorphism family, and the choices of `method="auto"` (section "Choosing a method" of [problems.md](problems.md#choosing-a-method)), reads nothing else that the layers leave.
  None may read what depends on the timing of the threads, such as `U` during the layer of a set, how far the layer of a set has got, or the counts of supports that a split layer weighs, looks up, and skips, which depend on when `U` falls.
  A split layer that stops at the cutoff leaves a `U` that can depend on that timing, but the search then ends, with `U` as the result of an early exit.
  The lower bound counts the layer of a set only once every work item of that layer has run, as an advance of the search does, which raises the lower bound only after it has scanned each set that it counts.
  A test of [_brouwer_zimmermann/distance_test.py](_brouwer_zimmermann/distance_test.py) compares the bounds, best weights, and switches after each layer with those of one thread.
- **Preserve the return contract.**
  Return `max_weight+1` if every nontrivial word is heavier than `max_weight`, and otherwise the exact minimum, unless `cutoff`, which `max_weight` caps, permits an early exit at a known eligible witness of weight at most `cutoff` ([Behavior spanning files](#sec-cross-file-behavior)).

Tests check these requirements, against exhaustive enumeration where they can, and each such test cites this section in its docstring.
Helper tests check current interfaces, and tests of heuristics check that each heuristic decides chosen cases as intended.
So neither makes a packing order or optimization a requirement, and a deliberate change of a heuristic may need new cases.
