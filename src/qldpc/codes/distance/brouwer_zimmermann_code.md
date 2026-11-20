# Brouwer-Zimmermann distance search in qLDPC: the source code and its requirements

This file holds Sections 4 and 5 of the guide to the Brouwer-Zimmermann (BZ) search; [brouwer_zimmermann.md](brouwer_zimmermann.md) holds Sections 1 to 3 and the references, and its Sections 2 and 3 explain the algorithm and its extensions.

## 4. Guide to the source code

This section maps the concepts of Sections [2](brouwer_zimmermann.md#2-the-brouwer-zimmermann-algorithm) and [3](brouwer_zimmermann.md#3-extensions-and-optimizations) to modules and identifiers.
The thresholds, limits, and coefficients of the heuristics are fields of `SearchTuning` in [_tuning.py](_tuning.py): [Section 4.4](#44-heuristics-and-cost-models) gives the reasons for those of BZ searches.

### 4.1 Module map and layering

[README.md](README.md) maps the modules of this directory and the levels in which they import each other; this section says how the searches use them.

The distance functions of [core.py](core.py) run the BZ search of [_brouwer_zimmermann/distance.py](_brouwer_zimmermann/distance.py), the exhaustive enumeration of [_brute_force.py](_brute_force.py), or the connected-cluster search of [_connected_cluster/search.py](_connected_cluster/search.py).
For BZ, they build an adapted basis ([Section 3.4](brouwer_zimmermann.md#34-quotient-labels-for-the-stabilizer-test)) for each sector, and pass the sectors to `get_distance_brouwer_zimmermann`, both sectors of a CSS code at once in `get_distance_css`, or only the X sector if `_search_css` finds a qubit permutation that maps each sector onto the other.
`_search_css` finds only the identity and, with pybliss, permutations that map the nonzero X-type gauge generators, as listed, one to one onto the Z-type ones and back (section "Searching one sector of a CSS code" of [README.md](README.md)).
[_information_sets/automorphism_families.py](_information_sets/automorphism_families.py) verifies automorphisms and builds the families of [Section 3.7](brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms), with the certificates of [_information_sets/certificates.py](_information_sets/certificates.py).

[logical_counts.py](logical_counts.py) builds its families with the stages of [_information_sets/basis_families.py](_information_sets/basis_families.py) that distance searches use ([Section 4.4](#44-heuristics-and-cost-models)), in `_select_basis_family` and `count_words` of [_counting/scan.py](_counting/scan.py), under the stricter stopping rule of [Section 5](#5-correctness-requirements).
To find its default weight, the distance, a count runs a distance search from the disjoint families that it keeps.
The setup functions of distance searches and counts can be compiled on request through [_compilation.py](../../_compilation.py).
Every caller of `get_overlapping_family` ([Section 4.4](#44-heuristics-and-cost-models)) gets its `OverlappingFamilyPolicy`, with a fixed-seed generator, from `build_overlapping_family_policy`: the distance search and the counts through `get_overlapping_family_policy`, which reads the tuning, and `CosetRoutes` of [_brouwer_zimmermann/coset_routes.py](_brouwer_zimmermann/coset_routes.py).
[_information_sets/completion_bounds.py](_information_sets/completion_bounds.py) chooses and builds the tables of [Section 3.9](brouwer_zimmermann.md#39-skipping-completions-by-completion-tables) for distance searches and counts, and sizes and builds them for the routes by tails of coset searches (below).

The distance search and `CosetLeaderSearch` scan with the compiled entry points of [_brouwer_zimmermann/affine_search.py](_brouwer_zimmermann/affine_search.py), which Numba compiles from one body, `_scan`, separately for each set of constant flags, so no flag costs a branch for each word:

- `scan_set` scans one set of a distance search, with or without completion tables.
- `scan_set_items` scans a range of work items of a layer of such a set, as `scan_set` does, on one of several threads, from the first support that `start_set_items` sets ([Section 4.5](#45-behavior-spanning-files)).
  The distance search runs the ranges through `run_work_items` of [_threads.py](_threads.py).
- `scan_coset_layer` and `scan_coset_layer_by_css_weight` scan one layer of one set in a coset search, and keep its lightest word.
- `scan_coset_layer_by_tails` does so by the tails of the words ([Section 3.5](brouwer_zimmermann.md#35-packed-enumeration-with-pivot-column-deletion)), and keeps the support of the lightest word.
- `scan_coset_layer_by_tails_with_tables` also skips, in a layer of two or more rows, the completions whose bound reaches the lesser of a skip bound and the best weight so far.
  A query skips at `h(ω)` ([Section 3.9](brouwer_zimmermann.md#39-skipping-completions-by-completion-tables)), and a search for minimum weights at its incumbent ([Section 3.10](brouwer_zimmermann.md#310-minimum-weights-of-cosets)).

Each search packs its problem once, before it scans, in an `AffineInstance` of [_brouwer_zimmermann/affine_instance.py](_brouwer_zimmermann/affine_instance.py): the packed rows, labels, pivots, and ranks of its sets, and the certificate of their steps, whose weight divisor rounds its bounds.
`get_distance_instance` packs a basis family without its pivot columns ([Section 3.5](brouwer_zimmermann.md#35-packed-enumeration-with-pivot-column-deletion)), and a distance search builds another instance when it switches to an automorphism family, or a copy with the certificate of [Section 3.7](brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms), "Images of the blocks of a family", once it has automorphisms.

`CosetLeaderSearch` of [coset_leaders.py](coset_leaders.py), which the fault-propagation certifier of `qldpc.circuits.certification` uses, applies BZ to cosets `u + V`, unless it is built with `method="connected_cluster"`.
A query of `CosetLeaderSearch`, whether the coset of a syndrome has a word within a maximum weight, runs one or two **coset searches**.
They are a search of Pauli errors or, for the CSS weight `max(|x|, |z|)` on a CSS state, searches by Hamming weight of the X and Z parts of the errors, each in a coset of a classical code ([README.md](README.md#the-coset-leader-search)).

Each family of search bases of a coset search has a **coset instance**: `get_coset_instance` packs the full rows of the disjoint sets of the code, and `get_coset_family_instance` those of any other family, with the certificate of its ranks and maximum coverage.
That certificate takes no weight divisor, even when `get_disjoint_family` infers one for `V`, since the words of a coset need not share it ([Section 3.2](brouwer_zimmermann.md#32-rounding-the-certificate-to-a-proven-weight-multiple)).
If `V` is even, `AffineInstance.get_weight_bounds` rounds the bounds up to a parity that it is given, and `get_search_tables` asks for that rounding when some coset is odd, which holds exactly when some word whose syndrome has one set bit is odd.
In a search of Pauli errors, an error weighs at least half the Hamming weight of its binary word, so the instance has the weight factor `η = 2`, by which `get_weight_bounds` divides its bounds, rounding up, after any rounding to parity; a search of X or Z parts has `η = 1`.

A query starts from the **word of its syndrome**, the sum of the words with unit syndromes (`get_unit_syndrome_words`) for the set bits of its syndrome.
It stops at its **stopping step**, the first step whose bound exceeds the maximum weight, unless it finds a light enough word sooner.
Each family of a coset search keeps the bounds of its steps in a **planning buffer**: one table for each parity of the cosets if the search rounds to parity, and one otherwise, and a query takes the table of its coset's parity.
Each table has an extra row that caches a stopping step; `get_stopping_step` reuses it for a nonnegative maximum weight exactly when it is that weight's stopping step.
Callers share the searches that `get_pauli_search` and `get_css_searches` cache, with the planning buffers of their families.
The batch loops `_find_first_not_within_max_weight` and `_find_min_weights` run without the global interpreter lock ([Section 4.5](#45-behavior-spanning-files)), so each call writes only buffers that no other running call uses.
`BrouwerZimmermannCosetSearch.find_first_not_within_max_weight` gives each worker of a batch, one on one thread, its own copy of the planning buffer (`_get_own_planning`), whose cached step serves only the queries of that worker; the copy holds a few small tables.
The copy starts from the initial step `(0, 1)`, rather than the cached step of the shared buffer, since NumPy may copy without the global interpreter lock while a single query writes that step; so the copy depends on nothing that another thread writes.
`_find_min_weights` caches no steps, and writes none of the buffers that it shares.
A single query, `get_min_weight_bound`, calls `_search_coset` from Python as a compiled entry point without `nogil`, so its compiled calls hold the global interpreter lock while they read and write the shared buffer.
Two single queries interleave those reads and writes only when `_search_coset` runs as plain Python or in free-threaded Python.
A cached step is written as two entries, so an interleaved read can pair the layer of one step with the number of completed sets of another.
Every step that `find_stopping_step` returns, like the initial `(0, 1)`, has a layer of the table and from one to all of its sets completed, so any such pair `(w, j)` is a valid entry of the bounds, which never change, and so is `[w, j - 1]`.
`get_stopping_step` reads each entry once and reuses the pair only if those bounds show that it is the stopping step, which is unique since the bounds never decrease; otherwise it finds the step itself.
`test_cached_stopping_steps_are_reused_exactly_while_they_stay_first` of [_brouwer_zimmermann/coset_tables_test.py](_brouwer_zimmermann/coset_tables_test.py) plants every such pair.

`find_stopping_step` of [_information_sets/certificates.py](_information_sets/certificates.py) finds stopping steps by a binary search of the steps, whose bounds never decrease, and `plan_steps` also adds up their work ([Section 3.6](brouwer_zimmermann.md#36-work-estimates-and-exhaustive-enumeration)).
The distance search plans each layer with `Certificate.get_num_sets_to_certify`, which calls `Certificate.plan`; `plan` uses `find_stopping_step` when the certificate solves no linear programs, and otherwise checks every step of the layer but the last in order with `certifies`.

`BrouwerZimmermannCosetSearch.get_min_weight_bound` first looks up the words of weight at most one, and below a maximum weight of two returns the lower bound two without a search, or three for an odd coset in a search of X or Z parts that rounds to parity.
Otherwise a query takes the **route** of its maximum weight: a family of search bases, with its coset instance and planning buffer, which `CosetRoutes` of [_brouwer_zimmermann/coset_routes.py](_brouwer_zimmermann/coset_routes.py) chooses once for each maximum weight ([Section 4.4](#44-heuristics-and-cost-models)).
`find_first_not_within_max_weight` takes the route once for a batch, and for the CSS weight on a CSS state, `CosetLeaderSearch` gives the X parts and the Z parts of a batch to their own searches and routes.
Both batch queries split a batch across threads, after they take the route ([Section 4.5](#45-behavior-spanning-files)).
`get_min_weights` of `CosetLeaderSearch` finds the minimum weights of [Section 3.10](brouwer_zimmermann.md#310-minimum-weights-of-cosets) through `BrouwerZimmermannCosetSearch.get_min_weights` and `_find_min_weights`, which finds its stopping steps with `find_stopping_step` and leaves the cached steps to the queries.

On a **route by tails** ([Section 4.4](#44-heuristics-and-cost-models)), the queries weigh words by their tails, and pass the family's **record of tails**, `CosetTails`, as a plain tuple of its fields to the compiled searches, such as `_search_coset`, in place of None; Numba compiles each search separately for a record and for None, so the choice costs nothing for each word.
The record holds the tails of the rows of each search basis and of the words with unit syndromes, from which a query sums the tail of its offset, and the completion tables that the routes of the family ask for.
`CosetRoutes` builds the tails of a family, and the tables of each search basis, once, for the first route that needs them (`get_coset_tails`, `CompletionTables.build`).
It builds new tables in the array of a new record, which copies the earlier tables; the old record keeps those until `CosetRoutes` replaces it.
A query by tails scans with `scan_coset_layer_by_tails_with_tables` the layers of two or more rows of a search basis that has tables, and the others with `scan_coset_layer_by_tails`.
It rebuilds its lightest word once, with the full rows, from the word's search basis and support.
The search of Pauli errors by CSS weight has no routes by tails, since the CSS weight of a word depends on which of its deleted pivot columns lie in the X part.

`BrouwerZimmermannCosetSearch`, `_search_coset`, and `_find_min_weights` are in [_brouwer_zimmermann/coset.py](_brouwer_zimmermann/coset.py), and the tables of steps and tails, such as `get_search_tables`, `get_stopping_step`, and `CosetTails`, in [_brouwer_zimmermann/coset_tables.py](_brouwer_zimmermann/coset_tables.py).

### 4.2 Flow and exit points

A public entry point validates inputs and chooses a method.
For BZ, `get_distance_brouwer_zimmermann` builds a `BrouwerZimmermannSearch`, which prepares a search of each sector.
Each call of its `advance` method advances the search of one remaining sector by one layer, in round-robin order for a joint CSS search ([Section 3.3](brouwer_zimmermann.md#33-sharing-witnesses-between-css-sectors)).
Preparation returns a result, or a `_SectorSearch` whose `advance` method searches one layer `t`, or only its first `j` search bases if step `(t, j)` certifies `U` ([Section 2.5](brouwer_zimmermann.md#25-the-stopping-certificate)).
A result of preparation can include a witness from another sector or the bound of a weight limit ([Section 4.5](#45-behavior-spanning-files)), so it need not be the distance of its sector.

| Input or method in [core.py](core.py) | Outcome |
| --- | --- |
| Trivial input | Returns `NO_DISTANCE`, whatever `max_weight` is, for an empty classical code, a quantum problem with no logical rows, or `V = S` |
| `method="brute_force"` | Runs exhaustive enumeration over an independent basis, adapted to `S ⊆ V` for a quantum problem |

`_prepare_sector_search` considers the following exits in order.
The disjoint family, its pruning ([Section 4.4](#44-heuristics-and-cost-models)), and any overlapping or partitioned replacement come from the stages of [_information_sets/basis_families.py](_information_sets/basis_families.py); the exits between those stages belong to the distance search.

| Preparation stage | Outcome |
| --- | --- |
| Initial row witness | Returns if `U ≤ cutoff`, where `U` includes the lightest eligible row and any supplied bound (from another sector or a weight limit) |
| Rows of the disjoint family | Returns if checking the rows of every search basis gives `U ≤ cutoff` |
| Certificate of the disjoint family | Returns if `L(1) ≥ U` |
| Rows of each replacement family, overlapping and then partitioned, if selected | Returns if `U ≤ cutoff` |
| Certificate of each replacement family | Returns if `L(1) ≥ U` |
| Switch at the end of setup | If setup asked for automorphisms before partitioning and found some, the search bounds with the images of its family's blocks, then considers switching to their family ([Section 3.7](brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms)); returns if then `U ≤ cutoff` or the lower bound reaches `U` |
| Ready search | Returns a `_SectorSearch` that scans from layer two |

| Search location | Outcome |
| --- | --- |
| `_SectorSearch.advance` | Returns after one layer, or after the leading search bases of a layer that certify `U`, or earlier if `U ≤ cutoff`; a search may first get automorphisms, then bound its words with the images of its family's blocks, which may finish it, and then switch to an automorphism family ([Section 3.7](brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms)); a finished search does no further work |
| `_SectorSearch.finished` | Marks a search finished if `U ≤ cutoff`, `L ≥ U`, or every layer has been enumerated (`next_weight > dimension`) |
| `BrouwerZimmermannSearch.finished` | Marks the search finished once the shared `U ≤ cutoff`, or the search of every sector has finished; `get_distance_brouwer_zimmermann` then returns `U` |

The tests `test_brouwer_zimmermann_setup_exits`, `test_brouwer_zimmermann_setup_cutoffs`, `test_rows_of_an_automorphism_family_can_reach_the_cutoff`, `test_a_switch_at_the_end_of_setup_can_certify_the_distance`, and `test_searches_that_setup_settles_are_finished_at_once` in [_brouwer_zimmermann/distance_test.py](_brouwer_zimmermann/distance_test.py) exercise these exits.

### 4.3 Concept-to-code glossary

| Concept | Code |
| --- | --- |
| Upper bound `U` | `best`, or `upper_bound` as a parameter |
| Lower certificate `L` | `lower_bound` of a search, computed by `Certificate.lower_bound(completed_weight)` from the certificate's `dimension`, `ranks`, `weight_divisor`, and `max_coverage` |
| Certificate within a layer `L(t, j)` | `Certificate.step_lower_bound(weight, num_completed)` and the table of `get_step_lower_bounds`, which `Certificate.get_weight_bounds` rounds and divides for coset searches; `get_num_sets_to_certify`, `plan`, `find_stopping_step`, and `plan_steps` find stopping steps ([Section 4.1](#41-module-map-and-layering)) |
| Certificate from linear programming | `get_lp_lower_bound`, which a `Certificate` with `pivot_blocks` combines with the coverage-divided certificate; `BasisFamily.get_search_certificate` adds the blocks to costly searches of overlapping families, and `_switch_to_automorphism_family` adds the blocks and `column_orbits`, for the program over orbits, once a search has verified automorphisms |
| Family of search bases | `BasisFamily` of [_information_sets/basis_families.py](_information_sets/basis_families.py), which stacks the search bases into arrays and gives their `Certificate`; packed in an `AffineInstance`, and, in coset searches, with the tables of `CosetSearchTables` and `CosetTails` of [_brouwer_zimmermann/coset_tables.py](_brouwer_zimmermann/coset_tables.py) |
| Automorphisms, generators of `Γ` | `automorphisms`, whose row `i` maps column `j` to column `automorphisms[i, j]`, as returned by `get_automorphisms`; `_known_permutation_automorphisms` and `_get_permutation_automorphisms` of `QuditCode` and of `ClassicalCode` |
| Automorphism family | `get_automorphism_family` of [_information_sets/automorphism_families.py](_information_sets/automorphism_families.py), which returns a `BasisFamily` of the scanned set and the `Certificate` of all of its images; `_switch_to_automorphism_family` adopts them, and sets `uses_automorphism_family` of the search |
| Last completed layer `t`, next layer | `completed_weight`, `next_weight` |
| Divisor `D` | `Certificate.weight_divisor`, in Hamming units of the searched basis |
| Maximum coverage `C` | `Certificate.max_coverage` |
| Dimension `r`, block rank `r_i` | `dimension`, `ranks` (`BasisFamily.ranks`, `AffineInstance.ranks`) |
| Search basis and its block | One set of a `BasisFamily`: its entries of `generators`, `labels`, `pivots`, and `ranks`; the recorded block is the pivot subset, as explained below |
| Number of coordinates `N` | The length of `columns` (or `nonzero_columns`), the array of nonzero columns, in the functions that build families of search bases |
| Stored width `N_stored` | `basis.shape[1]` before pivot deletion, including any all-zero columns |
| Labels `M` | `labels`, which has no columns for a classical problem |
| Eligible rows | `get_eligible_rows` of [_binary.py](_binary.py): every row of a classical basis; rows with nonzero labels for a quantum problem |
| Support, prefix XORs | `support` and `prefixes` in `_SectorSearch._scan_set` and in the scans of [_brouwer_zimmermann/affine_search.py](_brouwer_zimmermann/affine_search.py), which step through a layer with `advance_support` of [_information_sets/support_stepping.py](_information_sets/support_stepping.py) |
| Packed generators | `generators` of the `AffineInstance` of a distance search: `uint64` words, after deleting pivot columns. `widths` counts the valid words for each search basis. |
| Packed labels | `labels` of the `AffineInstance` of a distance search: `uint64` quotient-label words, without pivot-column deletion |
| Batch of a compiled scan | `batch_size`, equal to `1 << search_chunk_exponent` supports, or supports and lookups together in a scan with completion tables, and `num_threads` times as many on threads ([Section 4.5](#45-behavior-spanning-files)) |
| Work item of a layer, and the threads that scan it ([Section 4.5](#45-behavior-spanning-files)) | `_WorkItems` and `_SectorSearch._scan_set_on_threads` of [_brouwer_zimmermann/distance.py](_brouwer_zimmermann/distance.py), `start_set_items` and `scan_set_items`, and `run_work_items` of [_threads.py](_threads.py); `num_threads` of a search |
| Split work, and the blocks of a batch of coset queries that threads search ([Section 4.5](#45-behavior-spanning-files)) | `run_blocks` of [_threads.py](_threads.py), which runs a batch on the calling thread until its work reaches the split work, and the rest in blocks with `run_work_items`; `_find_first_on_threads` and `_find_min_weights_on_threads` of [_brouwer_zimmermann/coset.py](_brouwer_zimmermann/coset.py); `num_threads` and `split_work` of a `CosetLeaderSearch` |
| Candidate/set evaluation | The unit of the work of `plan_steps`, which `Certificate.work` counts |
| Completion tables ([Section 3.9](brouwer_zimmermann.md#39-skipping-completions-by-completion-tables)) | `CompletionTables` of [_information_sets/completion_bounds.py](_information_sets/completion_bounds.py), chosen by `select_completion_tables` and kept, with the index of their set, in `completion_tables` and `completion_tables_set` of a distance search, or in `CosetTails` of a coset search |

For the rank-deficient block of [Section 2.4](brouwer_zimmermann.md#24-search-bases-and-rank-deficient-blocks), the implementation records only its `rank` independent pivot columns in its row of `BasisFamily.pivots`, padded with -1, which give the same term `b_i(t)`, and which coverage and pivot-column deletion use.

### 4.4 Heuristics and cost models

Every family gets the certificate of its own verified ranks and maximum coverage, so the heuristics below change how much work a search does, not whether its certificate holds; with a cutoff above one, they can change which witness at or below the cutoff it returns ([Section 4.5](#45-behavior-spanning-files)).
Searches read the heuristics' fields as README.md's [Tuning](README.md#tuning) section says.
A heuristic must also keep the requirements on threads of [Section 5](#5-correctness-requirements), such as "Decide only from what a layer leaves the same on any number of threads".
Some fields are thresholds of gates on optional steps, such as building a family of search bases, solving linear programs, asking for automorphisms, or predicting the cost of completion tables.
A step can save at most the work that it acts on, so its gate lets it run once an estimate of that work, in candidate/set evaluations, reaches the cost of the step, which the threshold need match only to an order of magnitude.
The Retuning section of [_tuning.py](_tuning.py) defines the other classes of fields, such as budgets and rough coefficients, and says how the defaults are set and when to change them.

- **`get_disjoint_family`** packs at most `2^r - 1` information sets greedily in natural column order.
  The layers of one set already reach every nonzero combination of rows, so a layer of more sets would examine more candidates than there are words, and the limit keeps the memory of the packing from growing with the square of the length.
  If it packs fewer information sets than `min(2^r - 1, floor(N / r))`, it tries up to `disjoint_random_column_orders` random orders from a fixed seed, a budget of setup effort since each costs another packing, and keeps the family with the lexicographically largest tuple of ranks.
  It also infers the divisor `D = 2` of [Section 3.2](brouwer_zimmermann.md#32-rounding-the-certificate-to-a-proven-weight-multiple).
- **`prune_family`** loses nothing: it removes a search basis of rank deficiency `d = r - r_i`, whose term `b_i(t)` is zero for `t < d`, when the information sets alone certify `U` by layer `d`.
- **`get_overlapping_family`** builds search bases on column orders that put the least-used columns first, with ties broken by a generator that `build_overlapping_family_policy` seeds afresh for every call, and keeps the cheapest family built.
  It makes up to `overlapping_attempts_per_set` attempts, each a row reduction, for each set that it may add, up to a family of `max_information_sets` sets, and a repeated pivot set still consumes an attempt; both fields are budgets of setup effort, since more sets can avoid costly layers but take more attempts.
  It builds nothing if the family that it is given has no information set, or at least `max_information_sets` of them, or information sets that use every nonzero column, or if no family of at most `max_information_sets` sets could do less work.
  Its gate, `overlapping_minimum_work` on the estimated work of the given family, breaks even with the cost of the attempts.
  `CosetRoutes` passes a minimum work of zero, since `coset_route_minimum_work` gates its families instead.
- **`get_partitioned_family`** finds the most information sets, up to `m = max_information_sets`, in which no column appears more than `b` times, by the matroid partitioning of Section 5.1 of Lisoněk and Trummer ([References](brouwer_zimmermann.md#references)) in `_information_sets/matroid_partition.py`.
  It may try each coverage `b` from two up to `ceil(m*r/N)`, the least at which `m` sets could fit by counting their pivots.
  A search takes the family of least work, if it does less work than the family of the earlier stages.
  Partitioning at `b` finds at most `m_b = min(m, floor(b*N/r))` sets, since at most `floor(b*N/r)` fit within coverage `b` by counting their pivots.
  A family of `j` information sets puts some column in at least `ceil(j*r/N)` of them, so at each step its coverage-divided certificate is at most that of `j` sets of that coverage.
  The work up to a step depends only on the step and the number of sets, so the family does at least the work of those `j` sets.
  `_estimate_optimistic_work` returns `W(k)`, the least such work over `j` from one to `k`.
  Partitioning skips each `b` at which `W(m_b)` is at least the work of the best family so far, at first the family of the earlier stages.
  No family that it finds at such a `b` could replace that family, so the search takes the same family as if partitioning tried every `b`.
  Trying every `b` from two up to `ceil(m*r/N)` costs roughly the sum of `b^2*r^2*N` over them, times a factor that varies between families of codes; the sum grows with the cube of `m`.
  So a search tries partitioning only if some family of at most `m` sets could do less work, and the search's estimated work is at least `partition_work_ratio` times that sum, a ratio that stands for the cost of one unit of the sum in evaluations.
  The gate charges that whole sum, though partitioning may skip some `b`.
  With `B = partition_max_coverage`, a limit for experiments, `m` is capped at `floor(B*N/r)`, the most sets that fit within coverage `B` by counting their pivots, so it tries no coverage above `B`.
  At `B = 1`, partitioning instead builds the maximum disjoint family of `get_maximum_disjoint_family` (see `CosetRoutes` below), pruned by `prune_family`.
  `m` does not limit that family, and the sum is `r^2*N`.
  The test of whether some family could do less work considers the disjoint families that partitioning can build.
  At coverage one, partitioning, like a packing, keeps a rank-deficient block only below `2^r - 1` information sets, so such a family holds at most `min(ceil(N/r), 2^r - 1)` sets, counting that block.
  It does no less work than as many information sets of coverage one, whose certificate is at least as large at each step, since the work up to a step depends only on the step and the number of sets.
  Let `f = min(floor(N/r), 2^r - 1)`, the most information sets that fit, up to the limit of a packing.
  The **disjoint rank bound** is `f` information sets and, if `f < 2^r - 1` and `N > f*r`, a rank-deficient block of rank `N - f*r`.
  By counting pivots, it bounds the ranks of every maximum disjoint family lexicographically.
  Partitioning skips coverage one if the family `F` of the earlier stages is disjoint and has the ranks of this bound, pruned for `U` as `prune_family` prunes a family.
  Every maximum disjoint family then has `f` information sets and a block of rank at most `N - f*r`, but at least that of any block of `F` (see `CosetRoutes`).
  Pruning drops a block of lower rank if it drops one of rank `N - f*r`, since the lower rank gives a larger deficiency `d`, and information sets that certify `U` by one layer also certify it by every later one.
  Once pruned, a maximum disjoint family therefore has the ranks of `F`, does the same work, and cannot replace `F`.
- **`get_search_certificate`** adds the linear programs of [Section 3.1](brouwer_zimmermann.md#31-overlapping-search-bases) to searches of overlapping families whose estimated work reaches `lp_minimum_work`, a gate that breaks even with their cost, since each costs far more than examining a word; the work estimates that choose families use only the coverage-divided certificate.
  The rational dual weights `y` have denominators of at most `lp_dual_max_denominator`, a convention: every positive limit is sound by weak duality, a smaller one can round away part of the bound, and a larger one lengthens the exact arithmetic.
- **`get_automorphism_family`** row reduces in natural column order and balances the shares of the orbits in `_balance_orbit_shares` ([Section 3.7](brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms)).
  The gate `automorphism_minimum_layer_work` of [Section 3.7](brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms) breaks even with the cost of finding automorphisms, verifying them, and building their family.
  The program over orbits of [Section 3.7](brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms) has no gate of its own.
  A search gets automorphisms only before costly work: before matroid partitioning that passes `partition_work_ratio`, or before a layer of at least `automorphism_minimum_layer_work` evaluations ([Section 3.7](brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms)).
  The program, with a variable for each orbit that the blocks meet, costs milliseconds against such work.
- **`CosetRoutes`** chooses the route of each maximum weight `ω` of a coset search ([Section 4.1](#41-module-map-and-layering)).
  The **predicted examinations** of a query at `ω` with a family, without completion tables, are one more than the largest work at `ω` over its planning buffer's tables, which bounds the words that the query examines or skips ([Section 3.6](brouwer_zimmermann.md#36-work-estimates-and-exhaustive-enumeration)).
  1. If the predicted examinations with the disjoint family are below `coset_route_minimum_work`, the route is the disjoint family.
  2. Otherwise, `CosetRoutes` tries once to build two families, at `ω_T`, the least weight from two at which the predicted examinations with the disjoint family reach the threshold, so that they do not depend on the order of queries.
     The first, which `get_maximum_disjoint_family` finds by matroid partitioning with coverage one, holds the most disjoint information sets, up to the limit of `get_disjoint_family`, and below that limit a rank-deficient block of the largest rank ([Section 2.4](brouwer_zimmermann.md#24-search-bases-and-rank-deficient-blocks)) on the other columns.
     It is kept if it has more information sets, or a rank-deficient block of larger rank, than the greedy packing.
     Edmonds' matroid partitioning algorithm maximizes the number of columns in its sets, all but the last of which are information sets, so no other choice of as many information sets leaves a block of larger rank.
     A greedy packing can miss sets: the columns `e1`, `e2`, `e1 + e2`, and `e1 + e2` hold two disjoint information sets, but a greedy packing in natural order takes `e1` and `e2`, and leaves a rank-deficient block of rank one.
     The second, an overlapping family of `get_overlapping_family` aimed at the Hamming weight `η ω_T + 1`, is kept if it does less work than the family that it starts from: the first, if kept, and otherwise the disjoint family.
  3. The route is the family of least work at `ω`, summed over its tables, and on ties the disjoint family, then the maximum disjoint one.
     A search for minimum weights up to `ω` takes the route of `ω`, or of the number of qubits if that is smaller, since no coset has a larger minimum weight.
  4. In a search by Hamming weight, the route is a route by tails if its own predicted examinations also reach the threshold.
     Its queries then weigh words by their tails, and skip completions by tables on blocks of `coset_completion_block_bits` columns, for every search basis that they scan in layers of two or more rows, if the tables fit in `coset_completion_max_table_bytes`; the threshold and this limit replace the minimum layer work and the cost model of `select_completion_tables`.
     Since a route by tails builds these tables without predicting whether they pay, this limit defaults lower than `completion_bound_max_table_bytes`.
     That field limits the tables of distance searches and counts, which build only the tables that their cost model accepts.

  Building the two families costs a matroid partitioning and the row reductions of an overlapping family; weighing by tails adds a fixed cost to each query, for the tail of its offset in each search basis and the rebuild of its lightest word; and tables cost memory and a build.
  Only queries that examine many words repay these costs.
  The threshold also sets the aim `ω_T`, which no break-even rule covers, so its default is chosen by hand.
- **`select_completion_tables`** decides, by the steps of [Section 3.9](brouwer_zimmermann.md#39-skipping-completions-by-completion-tables), whether a distance search uses tables on an information set in a layer, or a count on a step.
  Its gate `completion_bound_minimum_layer_work`, which `considers_tables` checks, breaks even with the cost of the prediction, whose sampling stays within the budget of setup effort `completion_bound_sampling_work`.
  The prediction takes the share that tables skip at the lower end of a Wilson score interval of `completion_bound_confidence_z` standard errors, or as zero with fewer than `completion_bound_min_samples` samples, conventions that keep sampling error from choosing tables that cost more than they save.
  The limit `completion_bound_max_cost_percent` compares two estimates in one unit, so it breaks even at 100 percent.
  The rough coefficients of `predict_completion_cost` are the costs `completion_bound_lookup_block_cost`, `completion_bound_lookup_block_cost_per_mib`, `completion_bound_table_byte_cost`, and `completion_bound_build_step_cost`, in scans of one support of a distance search, and the multiplier `completion_bound_kept_first_row_ratio`; a count scans full rows and labels, which cost more, so the model is conservative for it.
  The widths `completion_bound_block_bits` and `coset_completion_block_bits` default to sixteen, the widest that `CompletionTables.build` accepts, since a wider block never gives a weaker bound.
  Sampling only predicts the value of tables; every skip rests on a computed bound.
- **`Certificate.work` and `estimate_automorphism_family_work`** are the estimates of Sections [3.6](brouwer_zimmermann.md#36-work-estimates-and-exhaustive-enumeration) and [3.7](brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms), both of which `plan_steps` computes.

### 4.5 Behavior spanning files

- **Cutoff.**
  The docstring of `get_distance_from_generators` in [core.py](core.py) states the contracts of `cutoff` and of an early exit, which the other distance functions share.
  The default `cutoff=1` gives the exact distance of a nontrivial problem, which is at least one, and a `cutoff` of zero or less never exits early; `_scan_cutoff` clamps negative values to `-1` for the compiled scan.
  [Section 4.2](#42-flow-and-exit-points) lists where `cutoff` is checked.
- **Weight limit.**
  The same docstring states the contract of `max_weight`.
  The functions pass `upper_bound=max_weight+1` to BZ, as though a witness of that weight were known, so that it stops once its certificate reaches that bound, and clamp the results of the other methods to `max_weight+1`.
  They also cap `cutoff` at `max_weight`, since a higher cutoff would end the search at once on the assumed witness.
  Inside the searches, a sector without nontrivial words counts as `n+1`, which no word on `n` bits or qubits weighs, so that it never lowers the minimum over sectors.
- **Units.**
  Binary input (`symplectic=False`) is a classical problem, or a pure-X or pure-Z problem with one bit per qubit, and symplectic input (`symplectic=True`) has general Pauli rows.
  For symplectic input, BZ maps rows by `T` ([Section 2.7](brouwer_zimmermann.md#27-pauli-weight-as-hamming-weight)), doubles `cutoff` and the bound of a weight limit, passes `weight_divisor=2`, and halves the result, while `method="brute_force"` uses symplectic weights and the original `cutoff`.
- **Independence.**
  `_get_nested_basis` in `core.py` makes the generators of a caller independent, with `get_independent_rows`, or with `get_nested_code_basis` when the caller gives `S`, for every BZ and brute-force search of `_search_generators` and `_search_checks`, including those of `search_by_cost`; `_search_css` calls `get_nested_code_basis` itself for its BZ searches and those of `search_by_cost`.
- **Resumable layers.**
  A compiled scan of a distance search returns after at most `batch_size` supports, or supports and lookups together with completion tables, so that it stays responsive while it skips most supports, and successive batches reuse its `support` and `prefixes` arrays.
  On several threads, each thread has its own arrays, and a chunk of each thread holds at most `batch_size * num_threads` supports and lookups.
  Each thread holds the global interpreter lock between its chunks, so chunks that grow with the number of threads keep the share of time under the lock about constant, at the cost of a longer wait for a keyboard interrupt.
  Once the search bases that `advance` scans have completed a layer, it may raise `lower_bound` ([Section 2.5](brouwer_zimmermann.md#25-the-stopping-certificate)), but it increments `next_weight` only once every search basis has.
  Before a costly layer, `advance` may get automorphisms, then bound its words with the images of the blocks of its family, which may raise `lower_bound`, and switch to an automorphism family, which sets `next_weight` to two and may raise `lower_bound` ([Section 3.7](brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms)).
  If a scan stops at the cutoff or raises an exception, `advance` leaves `lower_bound` and `next_weight` as they were before that scan, but keeps any earlier switch.
  A later call restarts the layer from its first support, unless the search has finished, as it has after a stop at the cutoff.
  If `BrouwerZimmermannSearch.advance` raises, a witness that the interrupted scan found can stay out of the shared `U`, so build a new search to search again.
- **Threads.**
  A search scans a layer of a set on `num_threads` threads if the layer has `w ≥ 3` rows and more than `num_threads * batch_size` supports.
  `BrouwerZimmermannSearch` resolves `num_threads` once from the tuning.
  If the field is `None`, it takes the environment variable `QLDPC_NUM_THREADS`, or else `numba.config.NUMBA_NUM_THREADS` (README.md, "Threads").
  Otherwise, and always on one thread, it scans the layer on the calling thread with `scan_set`.
  A **work item** is the set of supports of the layer whose first `k` rows are fixed: `k = 1`, or `k = 2` if fixing one row leaves fewer than 32 items for each thread and `w ≥ 4`.
  Since `k ≤ w - 2`, every completion of a prefix that completion tables skip ([Section 3.9](brouwer_zimmermann.md#39-skipping-completions-by-completion-tables)) lies in the item of the prefix, so the items partition the supports that a scan of the layer weighs or skips.
  The items are numbered in lexicographic order, the order in which `advance_support` steps through a layer.
  Their numbers of supports vary widely, and earlier items tend to hold more.
  The threads, the calling thread among them, claim ranges of consecutive items by guided self-scheduling, weighted by the number of supports of each item: each range holds the fewest next items whose supports reach about `1 / (2 * num_threads)` of the supports that no thread has claimed, so a heavy item can make a range overshoot that share.
  So the ranges shrink toward the end of the layer, which balances the threads, and the claims number at most about `2 * num_threads * ln(W / m) + 1`, for the `W` supports of the layer and the `m` supports of its lightest item.
  Ranges of equal numbers of items would not do, since earlier items tend to hold more supports.
  A thread scans its range with `scan_set_items`, which Numba compiles without the global interpreter lock, and which stops once a support passes the last item of the range.
  Splitting depends on the order of the items: `start_set_items` starts a range at its first item's first support, the end of a range is tested in the same order, and completion tables take pairs of rows in scan order.
  `start_set_items` sets `support` and `prefixes` only up to the depth at which a range starts, and leaves the other entries stale, so a scan must read no entry beyond its depth.
  The threads share one copy of the rows, labels, and tables of the set.
  They scan one set at a time, in the order of the sets, so `advance` and its certificate are unchanged.
  Each thread lowers `U` under a lock after each chunk, and starts its next chunk from the `U` of that moment, which skips more than an older `U` would.
  Every value of `U` is the weight of an eligible word of some sector, or the caller's upper bound, so whichever value a chunk starts from, it skips only words that cannot lower `U`.
  After the layer, `U` is the least of its value before the layer and the weights of the eligible words of the layer, as on one thread.
  So every later decision, such as completion tables and the switch to an automorphism family, is that of one thread.
  The thread that brings `U` to the cutoff stops the others after their chunks.
  The witness, and so the result of an early exit, can then depend on the timing of the threads, though its weight still lies between the distance and the cutoff.
  An exception in any thread stops the others after their chunks, and so does one in the calling thread while it starts or waits for them, such as a keyboard interrupt.
  The exception is raised again once all threads have returned, except in narrow windows that only a signal handler could close (see `run_work_items` in `_threads.py`).
  A further interrupt in such a window can end the search while threads still run.
  If it arrives before any thread has told the others to stop, they run until the layer ends; otherwise, until their chunks end.
  The process waits for them at exit, unless the search ran on a daemon thread, whose threads are daemons too.
- **Coset batches on threads.**
  `CosetLeaderSearch` fixes `num_threads` when it is built, and with it a **split work** of `num_threads * batch_size`, as much as each thread of a distance search on `num_threads` threads scans in one chunk.
  `find_first_not_within_max_weight` and `get_min_weights` of `BrouwerZimmermannCosetSearch` run a batch through `run_blocks` of [_threads.py](_threads.py).
  The calling thread searches the batch alone, in order, in one call of the compiled loop, `_find_first_not_within_max_weight` or `_find_min_weights`, which stops once the work of its queries reaches the split work: one unit for each query, and one for each word that it weighs after the word of its syndrome.
  So a batch of cheap queries, and any batch on one thread, runs on the calling thread in one call, and costs about as much as one call of the loop.
  Once a batch has shown that it is costly, `run_blocks` splits the rest of it into contiguous blocks, about 16 for each thread, which the threads, the calling thread among them, claim in ranges, in increasing order, with `run_work_items`, and search one at a time with the same loop.
  Each worker has its own buffer for the lightest word; Numba compiles both loops without the global interpreter lock, and the call on the calling thread compiles them before any thread starts.
  The query takes its route before the threads start, so no thread builds tails or tables, and the threads share the tables of the route, apart from the copies of its planning buffer that `find_first_not_within_max_weight` gives each worker ([Section 4.1](#41-module-map-and-layering)).
  The answer for a syndrome depends only on the code, the maximum weight, and the tuning ([Section 5](#5-correctness-requirements)), so the weights of `get_min_weights` are those of one thread.
  In `find_first_not_within_max_weight`, a syndrome without a word within the maximum weight ends the batch.
  On the calling thread, before the split, it ends the query.
  In a block, `run_items_until_first` of `_threads.py`, which `run_blocks` calls, lowers a limit to the block, and the threads skip each block after the limit, including the rest of a range that they have claimed.
  The limit only falls, and every row before it runs, so the least index found, with its bound and the lightest word of its own search, is that of one thread.
  For the CSS weight on a CSS state, the X pass, the Z pass on the syndromes before the index of the X pass, and the passes of `get_min_weights` each run their batch in this way.
  An exception before the split, while no other thread runs, reaches the caller at once; one after it, in any thread, such as a keyboard interrupt in the calling thread, reaches the caller as the docstring of `run_work_items` says, and the threads stop within their ranges, except in the narrow windows that it describes.
- **CSS sectors.**
  Each sector of a joint search keeps its `lower_bound` and the `certificate` of its `instance` ([Section 3.3](brouwer_zimmermann.md#33-sharing-witnesses-between-css-sectors)).
- **Automorphisms.**
  The distance functions take automorphisms as an array or as a function, and pass them to their BZ searches as a function that calls the function of the caller at most once.
  `get_distance_exact` of `CSSCode` and of `QuditCode` in `src/qldpc/codes/common.py` pass a function that returns `_known_permutation_automorphisms` of the code, which the code caches, so bliss runs at most once for a code.
  It holds the automorphisms that bliss (`src/qldpc/external/_bliss.py`) finds among each set of checks from `_get_permutation_automorphism_search_matrices`, followed by the generators from `_get_permutation_automorphisms`, without repeats or the identity.
  The code verifies the generators from `_get_permutation_automorphisms`, and those that bliss finds in sets of checks other than the parity checks, before it appends them.
  `get_distance_exact` of `ClassicalCode` passes `_known_permutation_automorphisms` in the same way, with bliss searching the parity check matrix and the generators from `_get_permutation_automorphisms` appended, unless it measures the distance from a vector, whose coset an automorphism of the code need not preserve.
  A search calls the function only at the points of [Section 3.7](brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms), and uses the generators only after `verify_automorphisms` accepts them; for a CSS code, they must map the row spaces of both parity-check matrices onto themselves.
  For a general Pauli problem, `_get_lifted_automorphisms` lifts them to the coordinates of `T`, except for Pauli strings that `get_distance_pauli` searches as a CSS code.

## 5. Correctness requirements

A change may use other representations, certificates, and search strategies, but must keep these requirements for the nontrivial problems of [Section 2.1](brouwer_zimmermann.md#21-the-problem):

- **Search `V \ S`.**
  Transformations must preserve `V` and `S`, or an equivalent representation of `V \ S`, such as the label test of [Section 3.4](brouwer_zimmermann.md#34-quotient-labels-for-the-stabilizer-test).
- **Keep weights in one unit.**
  Candidate weights, certificates, cutoffs, and results must agree, which includes restoring the pivot ones ([Section 3.5](brouwer_zimmermann.md#35-packed-enumeration-with-pivot-column-deletion)) and converting between Pauli and Hamming weights ([Section 4.5](#45-behavior-spanning-files)).
- **Use only justified certificates.**
  A certificate `L` must hold for every eligible word lighter than `U`, all of which are unexamined, and need not hold for heavier words, such as skipped ones, so a search reports the bound `min(L, U)`, not `L`.
  It relies on block ranks, coverage, and completed layers (Sections [2.4](brouwer_zimmermann.md#24-search-bases-and-rank-deficient-blocks), [2.5](brouwer_zimmermann.md#25-the-stopping-certificate), and [3.1](brouwer_zimmermann.md#31-overlapping-search-bases)), and may overestimate coverage but never underestimate it; a coset search may use any family of its code, with that family's ranks and coverage.
  Rounding needs a congruence of every eligible weight: a common divisor, or, only in a coset search, where every word is eligible, the parity of a coset of an even code ([Section 3.2](brouwer_zimmermann.md#32-rounding-the-certificate-to-a-proven-weight-multiple)).
  A certificate from another CSS sector needs a justification in the sector where it is applied, and a certificate from the images of blocks, as of an automorphism family or the program over orbits, needs verified automorphisms and every examined eligible word to lower `U` ([Section 3.7](brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms)).
- **Skip only words that a bound proves heavy enough:** at least `U` in a distance search ([Section 3.9](brouwer_zimmermann.md#39-skipping-completions-by-completion-tables)).
  Completion bounds need a minimum over every remaining pair of rows; disjoint blocks; exactly `w` pivot ones, as only in an information set, or the coset scans' bound on pivot ones; and the rows of their own search basis in scan order ([Section 3.9](brouwer_zimmermann.md#39-skipping-completions-by-completion-tables)).
  Coset tails must keep every column but the pivots ([Section 3.5](brouwer_zimmermann.md#35-packed-enumeration-with-pivot-column-deletion)).
  A coset query of maximum weight `ω` skips at `h(ω)`, never `h(ω) - 1`, and reports `R` only if it skipped a word.
- **Find minimum weights of cosets exactly up to `ω`.**
  A search may stop only at a step whose bound reaches `U`, or within a step once `U` is at most the bound of the step before or the floor of the search: `w_x` in the search of the Z part for the CSS weight ([Section 3.10](brouwer_zimmermann.md#310-minimum-weights-of-cosets)), and zero otherwise.
  Its scans may keep only words lighter than its incumbent, and its tables may skip only at the incumbent or at the weight of a lighter word that the scan has found ([Section 3.10](brouwer_zimmermann.md#310-minimum-weights-of-cosets)).
- **Keep the strict stop of counting.**
  A count of the words of weight `t` must also exhaust the words of weight exactly `t`, so it stops only at a step whose bound exceeds `t`, never at equality, and skips only completions heavier than `t` ([Section 3.9](brouwer_zimmermann.md#39-skipping-completions-by-completion-tables)).
  A word of weight `t` is counted at the first step that visits it, and steps run in order, so a count that stops within a layer counts each such word once.
  The certificate of a count uses the ranks and actual coverage of its family, and the `upper_bound=t+1` that it passes to the family stages only aims them.
- **Keep coset queries reproducible.**
  The family and tables that answer a query must depend only on the code, the maximum weight, and the tuning, not on earlier queries, since the certifier's counterexamples report a query's word and bound.
  A query therefore reads only the tables that its own route asks for (Sections [4.1](#41-module-map-and-layering) and [4.4](#44-heuristics-and-cost-models)).
  - Since the batch loops run without the global interpreter lock, every call of one, on any number of threads, writes only buffers that no other running call uses ([Section 4.1](#41-module-map-and-layering)).
    A call of `_find_first_not_within_max_weight` writes only its worker's lightest word and its worker's copy of the planning buffer, reset to the first step.
    A call of `_find_min_weights` writes only its own buffer for the lightest word and the weights of its own rows.
    Neither writes the shared planning buffer.
    `test_concurrent_batches_share_a_search` runs its batches with every array of the search read-only, so that numba rejects a write to one in any branch of compiled code, and checks that the arrays are unchanged.
    Review checks the writes that this misses.
    Numba compiles some writes to read-only arrays anyway, such as those through `.flat`, `np.ascontiguousarray`, `np.asfortranarray`, `np.nditer`, or `np.fill_diagonal`.
    The test does not mark the buffers that a batch makes for its workers: the lightest word `best` and the copy of the planning buffer of each worker of `_find_first_on_threads`, and the rows of the shared `weights` of `_find_min_weights_on_threads`.
    The test also misses paths that the tests do not run with the arrays read-only, and, in plain Python, writes that tests do not run.
    Review checks that each worker writes only its own buffers, and each call only its own rows.
  - `CosetRoutes` chooses each route once, under `_ROUTE_LOCK`, and there builds the families other than the disjoint one, once, and each record of tails, and a batch query takes its route before its threads start ([Section 4.5](#45-behavior-spanning-files)).
    A record of tails never changes once a query may read it: a route that needs new completion tables gets a new record of its family, which copies the earlier tables and their layout ([Section 4.1](#41-module-map-and-layering)).
- **Count only work without the global interpreter lock in the split of coset batches.**
  The batch loops count as work only what they do without the global interpreter lock, in the units of [Section 4.5](#45-behavior-spanning-files): one for each query, and one for each word that it weighs after the word of its syndrome.
  That work decides when `run_blocks` moves the rest of a batch to threads, so this rule keeps the speed of the split, not its answers, which are those of one thread on any split.
- **Keep the scans of a layer splittable across threads** ([Section 4.5](#45-behavior-spanning-files), "Threads").
  The scans are `scan_set` and `scan_set_items` of [_brouwer_zimmermann/affine_search.py](_brouwer_zimmermann/affine_search.py), which a change to `_scan` reaches; the coset scans split no layer.
  A scan's position is `support`, `prefixes`, and its depth, the number of rows it has fixed, and it reads no entry of `support` or `prefixes` beyond that depth.
  Besides its position, it carries only values derived from its position and from data that no thread writes during the layer, such as the number of pivot rows among its fixed rows; the counts of its work; and its copy of `U`, which starts at the `U` it is given and only falls.
  It must visit the supports of each work item together, in the order in which `_WorkItems` numbers the items, and skip only supports that share the first `w - 2` rows of its current support.
  Each chunk is exact from any starting `U'`: it returns the minimum of `U'` and the weights of the eligible words of its supports, and skips only words of weight at least its current `U`.
  The tests on threads and work items below check that work items cover a layer once, in scan order, and compare searches and scans on threads with those on one thread: their distances, their bounds after each layer, and, from a shared `U` that the layer does not lower, the totals of supports weighed and skipped and of bounds looked up.
  Tests of results can miss a race or an abort, so each of these rules says what checks it:
  - The compiled scans decide to skip or prune only from their position, the values they carry, and data that no thread writes during the layer.
    Review checks this.
  - The compiled scans write only to arrays that their caller gives the thread that runs them, such as `support` and `prefixes`.
    `test_searches_on_threads_advance_as_searches_on_one_thread` runs the threads with every array of the search read-only, so that numba rejects a write to one in any branch of compiled code, and checks that the arrays are unchanged.
    Review checks the writes that this misses.
    Numba compiles some writes to read-only arrays anyway, such as those through `.flat`, `np.ascontiguousarray`, `np.asfortranarray`, `np.nditer`, or `np.fill_diagonal`.
    The test does not mark the arrays that each thread of `_scan_set_on_threads` gets for itself: its `support` and `prefixes`, and the work items that it scans.
    The test also misses paths that the tests do not run with the arrays read-only, and, in plain Python, writes that tests do not run.
    Review checks that each thread writes only its own arrays.
  - Other code that threads run writes shared state only under a lock, through thread-safe objects, or under its own key, as the shared `U`, the claims of items, the stop event, and the result of a row of a coset batch that ends the batch are written.
    Review checks this.
  - The shared `U` is lowered only as a minimum, under the lock.
    Review checks this.
  - Code that threads run calls no numba-parallel code (`parallel=True` or `prange`, such as some functions of galois; see `run_work_items` in `_threads.py`), and reads the tuning only from its search, never through `get_tuning`, since new threads start with the default tuning.
    `test_nothing_asks_numba_for_parallel_code` of `src/qldpc/_compiled_code_rules_test.py` checks that qLDPC compiles none of its own functions numba-parallel.
    `test_always_compiled_functions_use_nothing_from_outside_qldpc` of `src/qldpc/_compiled_code_rules_test.py` keeps galois's functions out of compiled code.
    The fixture `check_code_on_threads` of [conftest.py](conftest.py) fails each test of this package whose threads call numba-parallel code from Python, or read the tuning through `get_tuning`.
    Review checks such calls on paths that tests do not run on threads.
- **Decide only from what a layer leaves the same on any number of threads** ([Section 4.5](#45-behavior-spanning-files), "Threads").
  After the layer of a set, `U` and the lower bound are those of one thread, and so are the estimates derived from them, such as `get_layer_work` and `estimate_automorphism_family_work`.
  Every later decision, such as the sets that a layer scans, completion tables, the switch to an automorphism family, and the choices of `method="auto"` (section "Choosing a method" of [README.md](README.md)), reads nothing else that the layers leave.
  None may read what depends on the timing of the threads, such as `U` during the layer of a set, how far the layer of a set has got, or the counts of supports that a split layer weighs, looks up, and skips, which depend on when `U` falls.
  A split layer that stops at the cutoff leaves a `U` that can depend on that timing, but the search then ends, with `U` as the result of an early exit.
  The lower bound counts the layer of a set only once every work item of that layer has run, as in `advance`, which raises it only after `_SectorSearch._scan_set` has scanned each set that it counts.
  `test_searches_on_threads_advance_as_searches_on_one_thread` compares the bounds, best weights, and switches after each layer with those of one thread.
- **Preserve the return contract.**
  Return `max_weight+1` if every nontrivial word is heavier than `max_weight`, and otherwise the exact minimum, unless `cutoff`, which `max_weight` caps, permits an early exit at a known eligible witness of weight at most `cutoff` ([Section 4.5](#45-behavior-spanning-files)).

These tests check the requirements, against exhaustive enumeration where they can:

- [_information_sets/basis_families_test.py](_information_sets/basis_families_test.py): `test_family_certificates` and `test_overlapping_family_certificates`, certificates with rank-deficient blocks, parity, and overlaps.
- [_information_sets/automorphism_families_test.py](_information_sets/automorphism_families_test.py): `test_automorphism_families_certify_like_the_family_of_all_images`.
- [_information_sets/certificates_test.py](_information_sets/certificates_test.py): `test_programs_over_orbits_bound_every_word_that_meets_every_image`, the program over orbits against enumerated words.
- [_information_sets/completion_bounds_test.py](_information_sets/completion_bounds_test.py): `test_bounds_are_exact_relaxations`, completion bounds against enumerated completions.
- [_brouwer_zimmermann/distance_test.py](_brouwer_zimmermann/distance_test.py): `test_brouwer_zimmermann_sector_certificates`, joint searches with distinct sector certificates; `test_automorphism_family_searches_scan_every_layer_they_certify` and `test_automorphism_family_searches_scan_their_new_set_from_layer_two`, distances that skipping a layer would miss; `test_searches_ask_for_automorphisms_only_before_costly_layers`; `test_images_of_blocks_bound_a_search_once_it_has_automorphisms`, the bound of the images of the blocks of a family against the program over every image; `test_completion_tables_keep_distances_exact`; for searches on threads, `test_searches_on_threads_find_the_distances_that_one_thread_finds`, `test_threads_find_the_lightest_words_of_layers_split_across_them`, `test_layers_of_a_set_of_deficient_rank_are_exact_on_threads`, `test_searches_on_threads_advance_as_searches_on_one_thread`, `test_scans_on_threads_weigh_look_up_and_skip_what_one_thread_does`, and `test_searches_on_threads_stop_at_the_cutoff_within_a_layer`; and, for work items, `test_work_items_fix_the_first_rows_of_supports_in_lexicographic_order`.
- [_brouwer_zimmermann/affine_search_test.py](_brouwer_zimmermann/affine_search_test.py): `test_scan_of_one_information_set`, which weighs or skips each support once; `test_coset_tail_scans_with_tables_skip_only_completions_that_reach_the_skip_bound`, including rank deficiency three; and, for scans on threads, `test_scans_of_the_work_items_of_a_layer_add_up_to_the_scan_of_the_layer`, which checks that the scans of the ranges of work items of a layer, run in order, find the best weight and the counts of the scan of the layer, and that scans on three threads of the items that fix one row, from the best weight that the layer ends with, count what the scan of the layer from that weight does.
- [_thread_rules_test.py](_thread_rules_test.py): `test_entry_points_without_the_lock_are_named_in_section_5_of_their_guides`, which reads the code; and `test_workers_that_call_numba_parallel_code_or_read_the_tuning_fail`, which checks the fixture `check_code_on_threads` of [conftest.py](conftest.py).
- [_threads_test.py](_threads_test.py): for work items on threads, `test_every_item_runs_once` and `test_every_item_runs_when_a_share_of_the_weight_rounds_away`, which check that threads claim and run every work item of a layer once; and, for coset batches, `test_costly_batches_move_the_rest_to_threads_in_contiguous_blocks`, `test_blocks_after_the_first_that_ends_a_batch_are_skipped_by_its_thread`, and `test_a_block_that_ends_a_batch_after_a_later_one_lowers_the_limit`, the split and the falling limit of `run_blocks`.
- [_brouwer_zimmermann/coset_test.py](_brouwer_zimmermann/coset_test.py): `test_searches_by_tails_answer_as_searches_of_whole_words`, the words, skips, and bounds `R` of queries with tables; `test_min_weights_of_cosets_match_brute_force`; `test_min_weight_searches_above_max_weight_examine_as_queries`; `test_answers_do_not_depend_on_earlier_queries`; and, for batches on threads, `test_batches_on_threads_answer_as_on_one_thread`, `test_an_earlier_heavy_coset_that_ends_last_is_the_answer`, `test_an_earlier_heavy_coset_recorded_before_a_later_one_is_the_answer`, and `test_the_word_of_an_earlier_heavy_coset_survives_a_later_search`, which find the first heavy coset in every order of the threads; `test_each_worker_searches_with_its_own_copy_of_the_planning_buffer`, `test_copies_of_planning_buffers_share_nothing_and_start_from_the_first_step`, and `test_concurrent_batches_share_a_search`, which check the buffers that workers write; and `test_costly_batches_move_to_threads_once_their_work_reaches_the_split_work` and `test_batches_within_the_split_work_run_on_the_calling_thread`, which check the work that the loops count.
- [_brouwer_zimmermann/coset_routes_test.py](_brouwer_zimmermann/coset_routes_test.py): the tables that routes by tails ask for and keep, also when threads choose routes at once.
- [coset_leaders_test.py](coset_leaders_test.py): `test_min_weights_match_exhaustive_minima`.
- [logical_counts_test.py](logical_counts_test.py) and [_counting/scan_test.py](_counting/scan_test.py): `test_counting_continues_at_an_equal_lower_bound` and `test_skipping_chunks_stop_at_a_lighter_nontrivial_word`, counts that a stop or a skip at equality would get wrong.
- [core_test.py](core_test.py): `test_brouwer_zimmermann_random_cross_checks`, on binary and general Pauli problems; `test_max_weight`, on BZ and brute force; and `test_max_weight_caps_cutoff`.

Helper tests check current interfaces, and tests of heuristics check that each heuristic decides chosen cases as intended.
So neither makes a packing order or optimization a requirement, and a deliberate change of a heuristic may need new cases.
