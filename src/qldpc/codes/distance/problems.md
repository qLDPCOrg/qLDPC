# How the package turns problems into searches

This guide, for contributors, explains how the package turns problems of [the grid](README.md#the-grid) into searches or other computations: which sectors a distance search of a CSS code searches, how `method="auto"` chooses the method of each distance search, how the coset-leader search answers its queries with coset searches, and how the MacWilliams method counts logical operators from the weight distributions of the checks.
[brouwer_zimmermann.md](brouwer_zimmermann.md) and [connected_cluster.md](connected_cluster.md) explain the searches themselves.

## Searching one sector of a CSS code

The distance of a CSS code is the smaller of the distances of its two sectors, so a search of the distance normally searches both.
Let `G_X` and `G_Z` be the X-type and Z-type gauge generators of the code, which are its checks if it is a stabilizer code, and let `S_X` and `S_Z` be the spaces of its X-type and Z-type stabilizers.
The X sector holds the words of `S_Z^⊥` outside `rowspan(G_X)`, and the Z sector those of `S_X^⊥` outside `rowspan(G_Z)` (Section 3.1 of `connected_cluster.md`).
`_search_css` in `distances.py` searches only the X sector if it finds a qubit permutation `π` that maps `rowspan(G_X)` onto `rowspan(G_Z)` and `rowspan(G_Z)` onto `rowspan(G_X)`.
The two sectors then have the same distance, so the result is the same, and so are the meanings of `cutoff` and `upper_bound`, since one sector has a word of a given weight exactly when the other does.

The proof: `π` preserves weights and dot products, so it maps the orthogonal complement of a space onto that of its image.
The stabilizers are the gauge operators that commute with every gauge operator of the other type, `S_Z = rowspan(G_Z) ∩ rowspan(G_X)^⊥` and `S_X = rowspan(G_X) ∩ rowspan(G_Z)^⊥`.
So `π(S_Z) = rowspan(G_X) ∩ rowspan(G_Z)^⊥ = S_X`, and `π` maps a word of the X sector, in `S_Z^⊥` but not in `rowspan(G_X)`, to a word in `S_X^⊥` but not in `rowspan(G_Z)`, a word of the Z sector of the same weight.
The inverse of `π` swaps the row spaces too, and so maps the Z sector into the X sector, which makes the map onto.
For a stabilizer code, `S_X = rowspan(G_X)` and `S_Z = rowspan(G_Z)`, and the proof is the same.

Only `_search_css` looks for such a permutation, in the `has_sector_map` that it passes to `_search`.
It calls `get_sector_map` in `_sector_maps.py`, which finds only two kinds.
The first is the identity, which swaps the row spaces if they are equal.
For the second, color the X-type and Z-type gauge generators of the Tanner graph differently.
With the optional pybliss package, bliss compares the canonical forms of this graph and of the graph with the colors swapped, and an isomorphism between them gives the second kind.
Such an isomorphism maps the nonzero X-type gauge generators, as listed and counting repeats, one to one onto the nonzero Z-type ones, and the Z-type ones onto the X-type ones, so its permutation of the qubits swaps the row spaces.
So bliss misses a permutation that swaps the row spaces but does not map the two lists onto each other, as when only one type has a redundant generator, or when the generators of one type are not images of those of the other.
Without pybliss, the lookup finds only the identity.
A search that misses a permutation searches both sectors, and its result stays the same.
`maps_sectors` checks every candidate from bliss by rank tests over GF(2) before a search uses it.
The fixed candidates below skip these tests, since the lookup accepts one only if it maps the two lists onto each other, which already makes it swap the row spaces.
Brute force never looks, so that it searches each sector as given: it is meant for small codes and for independent checks of the other methods, and `auto` takes it only when enumerating the sectors is cheap.

Cheap necessary tests come before the costly steps.
Unequal ranks rule out every permutation.
A few random words of `rowspan(G_Z)` test whether `rowspan(G_X)` may hold `rowspan(G_Z)` before the rank test of the identity, and the weights of the gauge generators and the X-degrees and Z-degrees of the qubits must match under swapped colors before bliss runs.
If pybliss is installed, the lookup first tries the reversal of the qubits, a cyclic shift by half of them, and a quarter turn of a square grid of qubits, as fixed candidates, and takes the first that maps the two lists of nonzero gauge generators onto each other, as the permutation of an isomorphism that bliss would find does.
It tries them only with pybliss, since each stands in for a map that bliss would find.
The reversal maps the lists of bivariate bicycle and generalized bicycle codes onto each other, so for them the lookup does not run bliss.
The lookup takes milliseconds on codes of hundreds of qubits.
On codes of thousands of qubits, it takes tens of milliseconds when the cheap tests reject, about a tenth of a second when it ends in the identity, and up to a few tenths of a second when bliss runs.
The switch `sector_maps` of `SearchTuning` turns the lookup off (see the docstring of `_tuning.py`).
Code classes that promise equal distances with `promise_equal_distance_xz`, such as `BBCode`, search one sector already through `CSSCode.get_distance_exact`; this lookup covers other codes, and the functions that take matrices.

Other exact distance tools use the same idea.
DistQLDPC, at commit fae495b, finds a map of the X sector onto the Z sector by a search for an isomorphism of its Tanner graph that exchanges the colors of the X and Z checks ([cc_aut.cc, lines 537–571](https://github.com/guluchen/DistQLDPC/blob/fae495bce77d6fff2cbb308aaca0676b45c8e98e/src/cc/cc_aut.cc#L537-L571)), and verifies it before it searches one sector ([cc_driver.cc, lines 172–193](https://github.com/guluchen/DistQLDPC/blob/fae495bce77d6fff2cbb308aaca0676b45c8e98e/src/cc/cc_driver.cc#L172-L193) and [316–330](https://github.com/guluchen/DistQLDPC/blob/fae495bce77d6fff2cbb308aaca0676b45c8e98e/src/cc/cc_driver.cc#L316-L330)).
QEDist, at commit 71a0f0c, takes a map that its user supplies or that its constructor of lifted-product codes proposes ([constructors.py, lines 103–116](https://github.com/yeon-lee/QEDist/blob/71a0f0c1020ffa48f6839e15c1a24b6642a8480e/cssdist/constructors.py#L103-L116)), verifies it when it builds a `CSSCode` ([code.py, lines 59–67](https://github.com/yeon-lee/QEDist/blob/71a0f0c1020ffa48f6839e15c1a24b6642a8480e/cssdist/code.py#L59-L67)), and then certifies the distance by searching one sector ([audit.py, lines 39–43](https://github.com/yeon-lee/QEDist/blob/71a0f0c1020ffa48f6839e15c1a24b6642a8480e/cssdist/audit.py#L39-L43)).
Qubitserf, at commit 04b9349, tests whether the identity, the reversal of the qubits, or a cyclic shift by half of them swaps the row spaces ([cc.cpp, lines 105–120](https://github.com/Quantinuum/Qubitserf/blob/04b9349a2d421815c35b2a3d53878167e7237f47/cpp/distfind/cc.cpp#L105-L120)).

## Choosing a method

Each problem's method type has its own `"auto"`, which chooses among that problem's methods: the coset searches of `CosetLeaderSearch` take the BZ method for it (`_get_coset_search_method` of `coset_leaders.py`), and so do counts (`_get_logical_count_method` of `logical_counts.py`).
For distances, `method="auto"` is the default of the distance functions, of `get_distance` and `get_distance_exact` of the code classes, and of `get_fault_distance`.
It chooses a method for each search, in `_search` of `distances.py`, which every distance function calls.
A code class that searches other codes, as `HGPCode` searches its seed codes, passes the method down, so each of those searches chooses too.
The rule aims to avoid a method that is very slow on a search, not to find the fastest one, and it decides from the shapes of the sectors and from the progress of the searches, without fitted thresholds.

1. Brute force runs if `is_cheap_to_enumerate` finds that it examines at most `auto_brute_force_max_words` packed 64-bit integers.
   Brute force runs on one thread, whatever `num_threads` is, at under a nanosecond per integer, so unlike that of the other methods its cost is known before it runs.
   The default, `2**20`, is about a millisecond of enumeration, the fixed cost of a BZ search of a small code.
   On larger spaces, a BZ search is usually faster (see "Refining the choice").
2. Otherwise, a search of binary words that excludes no nonzero word, as for a classical code, takes the BZ method, and so does every search of a function that takes generators, since it has no checks.
   A connected-cluster search can be much slower than BZ on classical codes whose checks are only moderately sparse.
3. Every other search, as of a quantum code or of the faults of a circuit, runs `search_by_cost` in `_auto.py`.
   It advances a connected-cluster search one weight limit at a time, through `ConnectedClusterSearch`, and counts work in BZ candidate/set evaluations, a state being `connected_cluster_state_evaluations` of them.
   The expected work of a limit is the work that the search has spent plus the predicted work of that limit.
   - In its free phase, the search runs on until the expected work of its next limit exceeds `auto_free_work`.
     Below that, fixed costs that the estimates leave out, such as setup, dominate the costs that a switch would compare.
     `SupportSearch._predict_states` predicts `MAX_VALUE - 1` states for a first limit that `cutoff` raises, so such a limit ends the free phase at once.
   - `_probe` then runs a BZ search by whole layers within `auto_probe_work`.
     It skips the costlier stages of setup: overlapping and partitioned families, and so the linear programs of [Section 3.1 of the BZ guide](brouwer_zimmermann.md#31-overlapping-search-bases), which only those take.
     A probe that gets automorphisms may still solve the program over orbits of [Section 3.7](brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms).
     If the probe finishes, its result is the answer.
     Otherwise, its shared upper bound `U`, the weight of a nontrivial word that it has found or the caller's upper bound if that is less, caps the connected-cluster search, which then stops once its lower bounds reach `U`.
     The search of a sector can hold a lighter witness from its setup, which joins `U` only after the probe advances that search.
   - Before each limit whose expected work exceeds `auto_free_work`, starting with the limit that ended the free phase, `_should_switch` may switch to a BZ search with the full tuning and the upper bound `U`.
     The level of a limit is the weight that it certifies, one weight divisor above it.
     The search switches if its spent and projected work to certify the level of its next limit exceeds the estimated work of BZ.
     Otherwise, it switches if its spent and projected work to reach `U` exceeds that of BZ, but only while the search of its next limit grows faster per unit of weight than BZ's would at the level of that limit.
     The projection to `U` extrapolates that growth over every limit below `U`, so it calls for a switch only while the growth test agrees.
     The comparison at the level projects only up to the level of the next limit, and needs no growth test.
     The search can switch only once the search of its next limit has a growth, which takes at least two limits without a witness.
     Before that, a first limit that `cutoff` raises has the prediction `MAX_VALUE - 1`, which says nothing about its cost, and after one limit the prediction assumes no growth.
     So a search with a cutoff, whose first limit is the cutoff, runs through it and one more limit before it can switch.
     With a cutoff within about two weight divisors of the distance, it may never switch.

The choice keeps every result exact, and `cutoff` and `upper_bound` mean what they mean for each method.
Since `U` is the weight of a nontrivial word or the caller's upper bound, a search with `U` as its upper bound returns the smaller of the distance and the caller's upper bound, as a search without it does.
When the connected-cluster search switches, a BZ search of the same sectors, in the same order, replaces it, and takes nothing from it but `U`.
For Pauli strings, that BZ search searches their images under `symplectic_to_hamming` (Section 2.7 of `brouwer_zimmermann.md`) and halves the weight that it finds.
The tuning of the probe changes only its family of information sets and their certificate, which no result depends on.
Which method runs can still change which word an early exit at `cutoff` finds, and so can the timing of the threads of a BZ search on several threads.

The estimates are crude.
The connected-cluster search predicts the states of a limit by `SupportSearch._predict_states` until two successive limits without a witness have visited states from the same number of roots.
It then extrapolates the states of its last limit without a witness by the growth of its states per unit of weight (`ConnectedClusterSearch.get_growth`).
`_BrouwerZimmermannCosts` estimates the work of BZ from the rank `r` and width `n` of each basis alone: a family that covers the columns evenly certifies about `(t + 1) n / r` after layer `t`, which scans `C(r, t)` supports of each set.
It counts one set per layer, as with an automorphism family, which underestimates a family of several sets.
It also leaves out completion tables, which can skip most supports of a costly layer, so it can overestimate the work of a hard BZ search.
To reach `U`, the projection counts half of the states predicted for the limit at weight `U` of the search of the next limit, as for a search that must find a witness of weight `U` itself.
A search with `U` as its upper bound stops before that limit, so this overestimates the connected-cluster work.
These two overestimates push the choice in opposite directions, so a refinement that removes one should remove the other.

### Refining the choice

The known ways for the rule to lose are these:

- A classical code whose checks are sparse enough for a connected-cluster search to be much faster takes BZ.
- A classical code of high relative distance whose space exceeds `auto_brute_force_max_words` takes BZ, which can be slower than enumerating it, as on a BCH code of length 127 and dimension 29, on which BZ takes about 1.4 times as long on one thread.
- A quantum code on which BZ is much faster pays for the connected-cluster search up to the switch, the probe, and the setup of a BZ search that starts again, since the switch discards the probe.
- A quantum code on which the connected-cluster search would win switches to BZ if the estimates make BZ look cheaper than it is, as when a basis has columns that its information sets cannot cover evenly.
  The comparison at the level of the next limit does not ask for faster growth, so such an estimate alone can make it switch.
- A search with a cutoff near the distance may never switch (see above).
  It then pays for the whole connected-cluster search unless its probe finishes.

A better rule could use:

- the work of each layer that a BZ search plans (`get_layer_work`), which counts the sets of its family, in place of the closed form;
- the work that the completion tables of a BZ search are predicted to skip;
- a projection to `U` that stops before the limit at `U`;
- a race that interleaves the two searches;
- more features of the sectors, such as the weights of their checks.

It could also resume the probe rather than start a new BZ search.
It must read from a search only what is the same on any number of threads, as Section 5 of `brouwer_zimmermann_code.md` and Section 5 of `connected_cluster.md` require ("Decide only from what a layer leaves the same on any number of threads" in the first).
For example, the counts of supports that a BZ layer on threads weighs and skips depend on the timing of its threads, but the predictions of its completion tables do not.
To check a refinement, record its choice and its predicted cost for each code before you time it, and then time it on codes other than those it was designed on, against each method alone.
A refinement should never make a choice that is much slower, in absolute time, than the one it replaces.

## The coset-leader search

`CosetLeaderSearch` answers each query with one or two coset searches, each over the cosets of a binary code.
For a single query, each looks up words of weight at most one in a table, then answers with `MinWeightBound`.

- In general, one search of the Pauli errors, modulo the stabilizers, answers a query.
- On a CSS state, where pure-X and pure-Z stabilizers generate its stabilizer group, two classical searches answer a query by CSS weight, the larger of the weights of the X and Z parts of an error, on `n` columns each rather than `2n`.
  The pure-X and pure-Z stabilizers of a state meet only in the identity, so they generate its stabilizer group, of dimension `n`, exactly if their dimensions add up to `n`.
  The X parts of the pure-X stabilizers span a code `C_X`, and the Z parts of the pure-Z stabilizers a code `C_Z`.
  Multiplying an error by a stabilizer adds a word of `C_X` to its X part `x` and, independently, a word of `C_Z` to its Z part `z`, so the smallest CSS weight with a given syndrome is the larger of the smallest weights in `x + C_X` and `z + C_Z`.
  `CosetLeaderSearch` maps the given syndrome to the syndromes of `x` for the pure-Z stabilizers and of `z` for the pure-X stabilizers.
  Each pure-Z stabilizer is a product of given stabilizers, so the syndrome bit of `x` for it, which is that of the error, is the sum of the bits of the given syndrome for those factors, and likewise for `z`.
  `get_min_weights` searches the Z part of an error only if its X part is within `max_weight`, with the weight of the X part as the floor of that search.

By default, each coset search is a Brouwer-Zimmermann (BZ) search, `BrouwerZimmermannCosetSearch` in `_brouwer_zimmermann/coset.py`, whose `_search_coset` scans the information sets of its code (Section 4.1 of `brouwer_zimmermann_code.md`).
The BZ search of Pauli errors is cached by stabilizer matrix (`PAULI_SEARCH_CACHE_SIZE` in `_tuning.py`), since its syndromes are for the given stabilizers.
Those of X and Z parts are cached by stabilizer group (`CSS_SEARCH_CACHE_SIZE`), so `CosetLeaderSearch` objects for different generators of one state share them.
`find_first_not_within_max_weight` returns the bound that a single BZ search found for the first syndrome without an error within `max_weight`, or, with two BZ searches, the bound that `get_min_weight_bound` gives for that syndrome.
`get_min_weights` answers a batch in one compiled loop per coset search (`BrouwerZimmermannCosetSearch.get_min_weights`; Section 3.10 of `brouwer_zimmermann.md`).
Both batch queries search their batches with compiled loops, `_find_first_not_within_max_weight` and `_find_min_weights`, and can move the rest of a costly batch to `num_threads` threads, the field of the tuning that `CosetLeaderSearch` reads when it is built ([Threads](README.md#threads)).
The calling thread starts the batch alone, and a batch of cheap queries stays there.
The split work of a search is `num_threads * 2**search_chunk_exponent` units of work, one for each query and one for each word that it weighs after the word of its syndrome.
Once the queries of a batch on the calling thread reach the split work, `run_blocks` of `_threads.py` gives the rest of the batch to the threads in blocks, each searched with the same compiled loop (Section 4.5 of `brouwer_zimmermann_code.md`).
The answers are those of one thread: the weights, and the first index without an error within `max_weight` with its bound and error.

With `method="connected_cluster"`, each `CosetLeaderSearch` builds its own coset searches, `ConnectedClusterCosetSearch` in `_connected_cluster/coset.py`, which make their checks sparser with `get_sparser_rows` and run `SupportSearch` from the syndrome of a query (Section 3.8 of `connected_cluster.md`); their checks are the stabilizers for Pauli errors, the pure-Z stabilizers for X parts, and the pure-X stabilizers for Z parts.
This method weighs errors by CSS weight only on CSS states (Section 3.8 of `connected_cluster.md`).
The class docstring of `ConnectedClusterCosetSearch` states the lower bounds of its answers.
`find_first_not_within_max_weight` and `get_min_weights` try increasing weight limits (`find_first_cluster_bound` and `get_min_cluster_weights` of `_connected_cluster/coset.py`), which can find light errors sooner than one search at `max_weight`.
They move a costly batch to threads with `run_blocks` too, with the same split work, in which the work of a query is the number of states that its searches visit; a query that looks up its error does none.
Their answers are those of one thread (Section 3.11 of `connected_cluster.md`).
For the CSS weight, the bound that `find_first_not_within_max_weight` returns holds an error for each part, and for a part within `max_weight`, that error can differ from the one that `get_min_weight_bound` gives.

The table "Routes and the names that control them" of [Exact-search performance and tuning](../../../../docs/source/exact_search_tuning.rst) names the fields that configure the searches of `CosetLeaderSearch`.

## The MacWilliams method

For a stabilizer code on `n` qubits with checks `H_X` and `H_Z`, `method="macwilliams"` of [_counting/macwilliams.py](_counting/macwilliams.py) counts from the weight distributions `A_X` of `row(H_X)` and `A_Z` of `row(H_Z)`.
The nontrivial X logicals are the words of `ker(H_Z)` outside `row(H_X)`.
Since `ker(H_Z)` is the dual of `row(H_Z)`, the MacWilliams identity ([References](#references)) gives their number of weight `w`:

$$
N_X(w) = 2^{-\operatorname{rank}(H_Z)} \sum_j K_w(j)\,A_Z(j) - A_X(w),
$$

where the [binary Krawtchouk polynomial](https://en.wikipedia.org/wiki/Kravchuk_polynomials) `K_w(j)` is the sum of `(-1)^(x·y)` over the words `x` of weight `w`, for any `y` of weight `j`.
The Z counts exchange X and Z.
A Gray-code walk over the `2^r` words of a row space of rank `r` gives its distribution `A`, so the cost is exponential in the ranks of the checks rather than in the number of logical qubits.
Since `|K_w(j)| ≤ binom(n, w)` and `A` sums to `2^r`, no Krawtchouk value, entry of `A`, product `K_w(j) A(j)`, partial sum of these products, or divisor `2^r` exceeds `2^r binom(n, w)` in magnitude.
So the sum of these products and its division by `2^r` are exact in signed 64-bit integers at every weight `w` with `2^r binom(n, w) < 2^63`.
qLDPC uses 64-bit integers below the first weight that fails this bound, and Python integers from there on.

## References

- J. MacWilliams, "A theorem on the distribution of weights in a systematic code," *Bell System Technical Journal* 42(1), 79-94, 1963 ([doi:10.1002/j.1538-7305.1963.tb04003.x](https://doi.org/10.1002/j.1538-7305.1963.tb04003.x)): the MacWilliams identity of [The MacWilliams method](#the-macwilliams-method).
