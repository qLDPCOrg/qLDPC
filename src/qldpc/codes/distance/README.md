# Exact distances and logical-operator counts: a guide for contributors

This directory is the package `qldpc.codes.distance`.
It computes exact distances of binary codes, counts the logical operators of a given weight in binary CSS codes, and finds light Pauli errors with given syndromes on stabilizer states.
Users reach it through the public functions and classes below, which take plain arrays, and through the `get_distance_exact` methods of the code classes in `src/qldpc/codes/common.py`, which call the distance functions for binary codes.
`brouwer_zimmermann.md` and `connected_cluster.md` explain the Brouwer-Zimmermann and connected-cluster searches, map them to the source code, explain their heuristics, and list the requirements that every change to them must keep; for the Brouwer-Zimmermann search, the last three are in `brouwer_zimmermann_code.md`.

## The grid

The package solves a few problems, each by several methods, which its `method` argument names, on one thread or several.
Each cell of the table below says whether a problem takes a method, and how it runs:

- **threads:** on one thread, and its costly work on `num_threads` threads (see [Threads](#threads)); for coset leaders, the batch queries, while a single query runs on one thread;
- **one thread:** always on one thread;
- **not yet:** not supported, but it could be;
- **not by design:** not supported, for the reason below the table.

| Problem | `brute_force` | `brouwer_zimmermann` | `connected_cluster` | `macwilliams` | `auto` |
| --- | --- | --- | --- | --- | --- |
| Distance | one thread | threads | threads | not by design | threads |
| Coset weight of a vector | one thread | threads | threads | not by design | threads |
| Coset leaders | not by design | threads | threads | not by design | threads |
| Logical counts | one thread | one thread | not by design | one thread | not yet |

- **Distance** is the minimum weight of a nontrivial word: the functions of `core.py`, the `get_distance_exact` methods of the code classes, and `get_fault_distance` in `src/qldpc/circuits/certification/fault_distance.py`.
- **Coset weight of a vector** is the minimum distance between a vector and a word of a binary classical code, which `ClassicalCode.get_distance_exact(vector=...)` in `src/qldpc/codes/common.py` finds by a distance search.
- **Coset leaders** are the lightest Pauli errors with a given syndrome on a stabilizer state: `CosetLeaderSearch` in `coset_leaders.py`.
- **Logical counts** are the numbers of logical operators of a given weight in a CSS code: `CSSLogicalCounter` in `logical_counts.py`.
  A count by `brouwer_zimmermann` that finds its default weight runs a distance search, which can use threads.

The cells that are empty, or on one thread only:

- Coset leaders by brute force: tests check the coset searches against the exhaustive minima of `get_exhaustive_minima` in `conftest.py`, and a library method would serve only states of a dozen qubits.
- Counts by `connected_cluster`: the search finds words whose supports the checks connect, and a count above the distance must also count other words.
- `macwilliams` for distances and coset problems: it needs a CSS stabilizer code whose X and Z checks have rank at most 63, and has no form for cosets; counts already give the distance that it finds.
- Brute force on threads: `auto` takes it only for at most `auto_brute_force_max_words` packed words, about a millisecond.
- Not yet: counts, single coset queries, and the MacWilliams method on threads; and `auto` for counts.

Where the code, guide, and tests of each supported cell live:

| Cell | Code | Guide | Tests |
| --- | --- | --- | --- |
| Distance, `brute_force` | `_brute_force.py` | Section 3.6 of `brouwer_zimmermann.md` | `_brute_force_test.py` |
| Distance, `brouwer_zimmermann` | `_brouwer_zimmermann/distance.py` | Sections 2 and 3 of `brouwer_zimmermann.md`, and `brouwer_zimmermann_code.md` | `_brouwer_zimmermann/distance_test.py` |
| Distance, `connected_cluster` | `_connected_cluster/search.py` | Sections 3.1 to 3.7, 3.9, and 3.11 of `connected_cluster.md` | `_connected_cluster/search_test.py` |
| Distance, `auto` | `core.py` and `_auto.py` | [Choosing a method](#choosing-a-method) | `core_test.py` and `_auto_test.py` |
| Coset weight of a vector, every method | `src/qldpc/codes/common.py` | the docstring of `ClassicalCode.get_distance_exact` | `src/qldpc/codes/common_test.py` |
| Coset leaders, `brouwer_zimmermann` and `auto` | `_brouwer_zimmermann/coset.py`, `_brouwer_zimmermann/coset_routes.py`, and `_brouwer_zimmermann/coset_tables.py` | Sections 3.9 and 3.10 of `brouwer_zimmermann.md`, and Section 4.5 of `brouwer_zimmermann_code.md` | `_brouwer_zimmermann/coset_test.py` and `coset_leaders_test.py` |
| Coset leaders, `connected_cluster` | `_connected_cluster/coset.py` | Sections 3.8 and 3.11 of `connected_cluster.md` | `_connected_cluster/coset_test.py` and `coset_leaders_test.py` |
| Logical counts, `brute_force` and `brouwer_zimmermann` | `_counting/scan.py` | Section 3.11 of `brouwer_zimmermann.md` | `_counting/scan_test.py` and `logical_counts_test.py` |
| Logical counts, `macwilliams` | `_counting/macwilliams.py` | "The MacWilliams method" in Section 3.11 of `brouwer_zimmermann.md` | `_counting/macwilliams_test.py` |

`_grid_test.py` checks the first table against the methods that each problem takes, and runs every supported cell, in each of its modes, on small instances against brute force.
A new problem, method, or mode adds its cells to both tables, and to `_grid_test.py`.

## Public names

`__init__.py` re-exports these names, and everything else in this directory is private:

- `get_distance_from_generators`, `get_distance_from_checks`, `get_distance_css`, `get_distance_pauli`, `DistanceMethod`, and `NoDistance`, whose only member `NoDistance.NAN` is also exported as `NO_DISTANCE`, from `core.py`.
- `CSSLogicalCounter`, `CSSLogicalCounts`, `count_logical_ops_css`, `LogicalCountMethod`, and `compile_search_setup`, from `logical_counts.py`.
- `CosetLeaderSearch`, from `coset_leaders.py`, and `MinWeightBound`, its bound for one syndrome, from `min_weight_bounds.py`.

`qldpc.codes` re-exports only `NoDistance`, `NO_DISTANCE`, and `DistanceMethod`, which the code classes take or return.
The documentation cites the other names at `qldpc.codes.distance`.

## Terms

- **Sector:** a set of words that a search examines, such as the logical operators of one type of a CSS code.
  A search of a classical code, of a qubit code that is not CSS, or of a circuit examines one sector.
  A Brouwer-Zimmermann search takes a basis of them, labeled to tell nontrivial words, such as logical operators, from trivial ones, such as stabilizers.
  A connected-cluster search takes checks whose kernel holds them and a matrix whose row space holds the trivial ones.
- **Information set:** in most of the source, a basis of a sector, row reduced on `rank` pivot columns: its first `rank` rows are the identity there and its other rows zero.
  In a full set, `rank` is the dimension of the basis; in a partial set it is smaller.
  `brouwer_zimmermann.md` and a few docstrings of `_brouwer_zimmermann/distance.py` call this basis a search basis or a set, and use "information set" only for the pivot columns of a full set (Sections 2.3 and 2.4).
- **Basis family** (`BasisFamily`): the information sets that one Brouwer-Zimmermann search or count uses together, stacked into arrays.
- **Certificate** (`Certificate`): a lower bound on the weight of the words that a Brouwer-Zimmermann search has neither visited nor skipped, which tells the search when it can stop; with completion tables, or with the images of blocks under automorphisms, it bounds only the words lighter than the best weight `U` of the search (Sections 3.7 and 3.9 of `brouwer_zimmermann.md`).
- **Connected-cluster search** (`SupportSearch`): the depth-first search that `method="connected_cluster"` selects, which grows supports through the unsatisfied rows of a check matrix `H` to find light binary or symplectic words outside the row space of an excluded matrix `G`, or light vectors with a given syndrome.
  `"auto"` runs it for the distances of quantum codes and the fault distances of circuits, unless enumerating is cheap (see "Choosing a method").
- **Automorphism:** in this package, a permutation of the coordinates of a search that maps the words it searches, and those it excludes, onto themselves.
  The public functions take only permutation automorphisms of a code, which permute qudits and apply no single-qudit gates, and lift them to the coordinates of a Pauli problem (Section 2.7 of `brouwer_zimmermann.md`).
  An automorphism of a quantum code can also apply single-qudit Clifford gates (see `get_transversal_automorphism_group` in `qldpc.circuits`); exchanging the coordinates of one qubit in a Pauli problem applies a single-qubit Clifford gate, but qLDPC never supplies such permutations.
- **Scan:** the enumeration of the combinations of rows of each information set, layer by layer, a layer holding the combinations of a given number of rows.
  Counts have their own scan, in `_counting/scan.py`, and the Brouwer-Zimmermann distance and coset searches (see "The coset-leader search") use the compiled scans of `_brouwer_zimmermann/affine_search.py`.
  Every scan steps through a layer with `advance_support` in `_information_sets/support_stepping.py`.

## Layout

- `core.py` and `logical_counts.py` are the public interfaces of the distances and the counts.
- `coset_leaders.py` is the public interface of the coset-leader search, and `min_weight_bounds.py` defines `MinWeightBound`, which its coset searches build too.
- `_auto.py` holds the parts of the automatic choice of a method, `method="auto"`, that compare the costs of the methods (see "Choosing a method").
- `_brouwer_zimmermann/` holds the Brouwer-Zimmermann searches, which scan information sets, and `_connected_cluster/` the connected-cluster searches, which grow connected supports.
  Each holds the distance search of its method, and the coset searches with which the coset-leader search answers queries by that method, with their batch queries.
  `_brute_force.py` holds exhaustive enumeration.
- `_counting/` holds the word scan of the counts, the local Clifford gates of deformed counters, and the MacWilliams method.
- `_information_sets/` holds the families of information sets and their certificates, which the Brouwer-Zimmermann searches and the counts share.
  The method name `"information_sets"` is an alias of `"brouwer_zimmermann"`, for distances and counts alike.
- `_cosets.py` holds the parts that the coset searches of both methods share.
- `_sector_maps.py` finds the qubit permutations that let a distance search of a CSS code search one sector (see "Searching one sector of a CSS code").
- `_tuning.py` holds the settings of the searches.
- `_threads.py` runs the work items of a search, or the blocks of a batch of queries, on several threads, and yields the outputs of work items in order on the calling thread, for the fault lister of `src/qldpc/circuits/certification/`.
- `_binary.py` holds the binary input checks, binary products, nested bases, the stabilizers of subsystem codes, and the symplectic-to-Hamming map.
  The code constructors in `src/qldpc/codes/common.py` also use its `infer_field`, which finds the field of rows given in lists or tuples.

## Where to look

| To change… | Look in (this directory) |
| --- | --- |
| A distance method, or the validation of method names | `core.py`, whose validation `coset_leaders.py` also uses; the early checks of `src/qldpc/circuits/certification/fault_propagation.py` and `src/qldpc/circuits/certification/fault_distance.py`; where `logical_counts.py` and `get_distance_exact` in `src/qldpc/codes/common.py` dispatch on the method name, and where `get_distance` validates it; and the `brute_force` flag that `logical_counts.py` passes to `_counting/scan.py` |
| The automatic choice of a method (`method="auto"`): which searches take which method, and when a connected-cluster search switches to the BZ method | `core.py`, where the distance functions dispatch on the method; `_auto.py`, which estimates and compares the costs; and `_get_coset_search_method` of `coset_leaders.py`, where coset searches take BZ |
| The counter, its deformed counters, or `compile_search_setup` | `logical_counts.py` |
| The coset-leader search, which `src/qldpc/circuits/certification/` and `src/qldpc/experimental/circuits/benchmarking/residual_weights.py` use, and how its queries split into coset searches | `coset_leaders.py` |
| The Brouwer-Zimmermann (BZ) distance search | `_brouwer_zimmermann/distance.py` |
| The compiled scans of layers that the BZ distance search and the BZ coset searches share, one body specialized by constant flags | `_brouwer_zimmermann/affine_search.py` |
| The instances of BZ distance searches and BZ coset searches: their packed information sets, and the certificate of these sets | `_brouwer_zimmermann/affine_instance.py` |
| The BZ coset searches of the coset-leader search | `_brouwer_zimmermann/coset.py` |
| The tables of BZ coset searches: the bounds of their steps and their stopping steps, and the tails and completion tables of their routes by tails | `_brouwer_zimmermann/coset_tables.py` |
| The routes of BZ coset searches: the family of information sets for each maximum weight, the gate that chooses it, and the routes by tails | `_brouwer_zimmermann/coset_routes.py` |
| The connected-cluster search, of distances and of vectors with a given syndrome | `_connected_cluster/search.py` |
| The connected-cluster coset searches of the coset-leader search, and their batch queries | `_connected_cluster/coset.py` |
| How the connected-cluster searches make their checks sparser with the same span, and find the light sums of pairs of checks and of known words that distance searches add | `_connected_cluster/sparse_checks.py` |
| Exhaustive enumeration, which the `brute_force` method runs, with its packed symplectic weights | `_brute_force.py` |
| The scan of a count, or how its basis families follow local gates | `_counting/scan.py` |
| The local Clifford gates of deformed counters | `_counting/local_cliffords.py` |
| The MacWilliams counting method | `_counting/macwilliams.py` |
| The bound for one syndrome that the coset-leader search and its coset searches give (`MinWeightBound`) | `min_weight_bounds.py` |
| Which sectors a distance search of a CSS code searches, and the qubit permutations that map one sector onto the other | `_search_css` of `core.py`, and `_sector_maps.py`, which calls `get_matrix_isomorphism` of `src/qldpc/external/_bliss.py` |
| The parts that the coset searches of both methods share: words with unit syndromes, sums of the rows that a support selects, and tables of the errors of weight zero or one | `_cosets.py` |
| How basis families are built | `_information_sets/basis_families.py` |
| Certificates, and how searches plan their steps: where they stop, and the work of the steps up to there | `_information_sets/certificates.py` |
| The step from one support of a layer to the next, which every scan of layers shares | `_information_sets/support_stepping.py` |
| Automorphism families, the images of one information set under automorphisms of a code; and the check of those automorphisms and the orbits that they generate, which the connected-cluster search uses too | `_information_sets/automorphism_families.py` |
| Completion tables, which let a BZ scan skip the supports whose completions by two more rows are too heavy, and the choice of whether to build them | `_information_sets/completion_bounds.py` |
| Information sets with bounded column coverage, by matroid partitioning | `_information_sets/matroid_partition.py` |
| The settings that a search reads when it starts (`SearchTuning` and `use_tuning`), and the constants that size caches when qLDPC is imported | `_tuning.py` |
| How a search runs its work items on threads (`run_work_items`), or until the first item that ends its work (`run_items_until_first`); how a batch of queries splits into blocks on threads (`run_blocks`); how the outputs of work items come back in order (`iterate_in_order`); and the default number of threads | `_threads.py`; how the BZ distance search splits a layer into work items is in `_brouwer_zimmermann/distance.py`, how the connected-cluster distance search splits a weight limit, in `_connected_cluster/search.py`, how BZ coset queries search their blocks, in `_brouwer_zimmermann/coset.py`, and how connected-cluster coset queries do, in `_connected_cluster/coset.py` |
| Binary input checks, binary products, nested bases, the stabilizers that `get_css_stabilizers` and `get_pauli_stabilizers` derive from gauge generators, or the map of symplectic vectors to binary vectors of twice their weight | `_binary.py` |
| Bit packing, bit counts, row reduction, or products of packed rows over GF(2), which other subpackages share | `src/qldpc/_binary.py` |
| How numba code is compiled or cached | `src/qldpc/_compilation.py` |

## How the modules depend on each other

From the top:

1. `logical_counts.py`, which takes `NoDistance` from `core.py`, and `coset_leaders.py`, which takes `DistanceMethod` from it.
2. `core.py`.
3. `_auto.py`, the automatic choice of a method.
4. The methods `_brouwer_zimmermann/`, `_connected_cluster/`, `_counting/`, and `_brute_force.py`.
5. `_information_sets/`, `_cosets.py`, and `min_weight_bounds.py`, which the methods build on, and `_sector_maps.py`, which `core.py` uses.
   The connected-cluster search uses `infer_weight_divisor`, `verify_automorphisms`, and `get_orbits` of `_information_sets/`.
6. `_tuning.py`.
7. `_binary.py` and `_threads.py`.

A top-level module or subpackage imports only from lower levels, and a module also from its own subpackage.
`_layering_test.py` checks this.
Within a subpackage, a module imports only the modules of the subpackage that it builds on, which the lines below give.
The `_LAYERS` of the subpackage's `_layering_test.py` places its modules in layers consistent with these dependencies, and that test checks the imports against the layers.
In `_information_sets/`, `basis_families` builds on `certificates` and `matroid_partition`, `automorphism_families` on `basis_families` and `certificates`, and `completion_bounds` on `certificates`.
`support_stepping` builds on no other module of `_information_sets/`.
In `_brouwer_zimmermann/`, `distance` and `coset` build on `affine_instance` and `affine_search`, and `coset` also on `coset_routes` and `coset_tables`.
`coset_routes` builds on `coset_tables`, and both build on `affine_instance`.
In `_connected_cluster/`, `search` builds on `sparse_checks`, and `coset` on both.
In `_counting/`, `scan` builds on `local_cliffords`, and `macwilliams` on no other module of `_counting/`.
Modules of every level may also import `src/qldpc/_binary.py`, the bit-packed linear algebra over GF(2) that other subpackages share.
`src/qldpc/codes/common.py` imports this package, so `logical_counts.py` imports the code classes of `codes.common` inside the functions that use them.
Modules elsewhere in qLDPC, such as `src/qldpc/codes/common.py` and the deprecation registry in `src/qldpc/_deprecations/core.py`, use private names of this package, so grep `src/qldpc` for a private name before you move or rename it.

## Searching one sector of a CSS code

The distance of a CSS code is the smaller of the distances of its two sectors, so a search of the distance normally searches both.
Let `G_X` and `G_Z` be the X-type and Z-type gauge generators of the code, which are its checks if it is a stabilizer code, and let `S_X` and `S_Z` be the spaces of its X-type and Z-type stabilizers.
The X sector holds the words of `S_Z^⊥` outside `rowspan(G_X)`, and the Z sector those of `S_X^⊥` outside `rowspan(G_Z)` (Section 3.1 of `connected_cluster.md`).
`_search_css` in `core.py` searches only the X sector if it finds a qubit permutation `π` that maps `rowspan(G_X)` onto `rowspan(G_Z)` and `rowspan(G_Z)` onto `rowspan(G_X)`.
The two sectors then have the same distance, so the result is the same, and so are the meanings of `cutoff` and `upper_bound`, since one sector has a word of a given weight exactly when the other does.

The proof: `π` preserves weights and dot products, so it maps the orthogonal complement of a space onto that of its image.
The stabilizers are the gauge operators that commute with every gauge operator of the other type, `S_Z = rowspan(G_Z) ∩ rowspan(G_X)^⊥` and `S_X = rowspan(G_X) ∩ rowspan(G_Z)^⊥`.
So `π(S_Z) = rowspan(G_X) ∩ rowspan(G_Z)^⊥ = S_X`, and `π` maps a word of the X sector, in `S_Z^⊥` but not in `rowspan(G_X)`, to a word in `S_X^⊥` but not in `rowspan(G_Z)`, a word of the Z sector of the same weight.
The inverse of `π` swaps the row spaces too, and so maps the Z sector into the X sector, which makes the map onto.
For a stabilizer code, `S_X = rowspan(G_X)` and `S_Z = rowspan(G_Z)`, and the proof is the same.

Only `_search_css` looks for such a permutation, with `get_sector_map` in `_sector_maps.py`, and it finds only two kinds.
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

`method="auto"` is the default of the distance functions, of `get_distance` and `get_distance_exact` of the code classes, and of `get_fault_distance`.
It chooses a method for each search, in `_search_css`, `_search_checks`, and `_search_generators` of `core.py`.
A code class that searches other codes, as `HGPCode` searches its seed codes, passes the method down, so each of those searches chooses too.
The coset searches of `CosetLeaderSearch` take the BZ method for `"auto"` (`_get_coset_search_method` of `coset_leaders.py`).
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
It must read from a search only what is the same on any number of threads, as Section 5 of each guide requires ("Decide only from what a layer leaves the same on any number of threads" in `brouwer_zimmermann_code.md`).
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
Both batch queries search their batches with compiled loops, `_find_first_not_within_max_weight` and `_find_min_weights`, and can move the rest of a costly batch to `num_threads` threads, the field of the tuning that `CosetLeaderSearch` reads when it is built ([Threads](#threads)).
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

## Tuning

A value that a search reads when it starts, and that changes how long the search takes or how much memory it uses but no result that it guarantees, is a field of `SearchTuning` in `_tuning.py`, unless its best value does not depend materially on the user, the code, or the machine (below).
Such values include a threshold that chooses a route, a limit, a coefficient of a cost model, a batch size, and a switch that turns a step off.
They are not arguments of public functions, whose arguments say what to compute and by which method, such as `cutoff`, `max_weight`, and `method`.

- A search reads the active tuning once, when it starts, and passes the values that it needs down as arguments, including to compiled code, so that one search never mixes two tunings.
  The distance functions of `core.py` read it for a distance search, `CSSLogicalCounter.count_logical_ops` for a count, and `CosetLeaderSearch` when it is built.
  `BrouwerZimmermannSearch` and `ConnectedClusterSearch`, which `get_distance_brouwer_zimmermann` and `get_distance_connected_cluster` build, read it only if their callers pass none.
  A cache that outlives a search is keyed by the values that it depends on, as the fields of `_CosetTuning` in `_brouwer_zimmermann/coset.py` key the cached searches of `CosetLeaderSearch`.
- Only a value that qLDPC fixes before any search starts is a constant in `_tuning.py`, such as the size of a cache, which `functools.lru_cache` fixes at import.
  Code reads cache sizes as `_tuning.NAME`.
  Constants fixed by the algorithm or by native integer widths, such as `WORK_CAP`, and the seeds of random generators stay in the modules that use them, except `MAX_VALUE`, the largest value of an integer field.
  The seeds make results reproducible rather than tune anything, though an early exit of a BZ search on several threads can still depend on the timing of its threads.
  A speed-only value that users may plausibly want to change to run faster is a field.
  Other speed-only constants, whose best value does not depend materially on the user, the code, or the machine, stay in their modules:
  - `ITEMS_PER_THREAD` and `BLOCKS_PER_THREAD` in `_threads.py`, which only balance threads at the end of the work
  - `_MAX_ITEMS` and `_MAX_FIXED_WORDS` in `_connected_cluster/search.py`, which bound the items of a frontier and the compiled versions of its search
  - `_MIN_FOUR_RUSSIANS_SIZE` in `_binary.py`, the crossover of a kernel that searches call only in their setup
  - `_NUM_RANDOM_WORDS` in `_sector_maps.py`, the size of a cheap prefilter
- Each field and constant has its own `#:` comment directly above it, for Sphinx to render, with no colon on its first line, since Sphinx reads text before such a colon as a type.
  A field's comment says what the field controls, with its unit and special values, and the class and rule of its default, as the "Retuning" section of the module docstring defines them, or why a field in no class has its default; where a guide section or a cost model explains its heuristic, the comment points there.
  A constant's comment says what it controls; the "Retuning" section says how the sizes of caches were chosen.
  Comments, docstrings, and guides in this package follow the [writing standard's rule on measurements](../../../../docs/source/contributing/standards/writing.rst#where-documentation-belongs).
- Change a default only as the "Retuning" section allows.
- [Exact-search performance and tuning](../../../../docs/source/exact_search_tuning.rst) renders these comments after hand-written tables that lead from symptoms and routes to the values; update the tables, and for a constant the list of constants, when you add, rename, or remove a value.
  `_tuning_test.py` checks that the table of routes names every field and every constant except `MAX_VALUE`, that the page lists every constant, and that each value has its own comment.
  The documentation build fails on a broken reference into `_tuning`.
- The module docstring of `_tuning.py` says which results a tuning can change.
- To force a route or exercise a value in a test, call the `tune` fixture of `src/qldpc/codes/conftest.py`, as in `tune(overlapping_minimum_work=0)`, rather than patching module attributes.
- If the implementation cannot safely support a field's full range, narrow it in the field's metadata rather than adding code for unsupported values.
  A field whose default is `None`, such as `num_threads`, also accepts `None`.

### Threads

A BZ distance search scans each costly layer of a set on `num_threads` threads, which share one copy of the set's completion tables (Section 4.5 of `brouwer_zimmermann_code.md`).
A connected-cluster distance search searches each costly weight limit on `num_threads` threads, which give the same witness and count of states as a search on one thread (Section 3.11 of `connected_cluster.md`).
On any number of threads, it searches in chunks of `2**search_chunk_exponent` states, between which it takes keyboard interrupts, and a thread stops a work item after its current chunk once an earlier item has a witness.
With either method, the batch queries of `CosetLeaderSearch`, `find_first_not_within_max_weight` and `get_min_weights`, move the rest of a costly batch to `num_threads` threads ([The coset-leader search](#the-coset-leader-search)).
A `CosetLeaderSearch` fixes its number of threads when it is built.
By default, `num_threads` is `None`, which takes the value of the environment variable `QLDPC_NUM_THREADS` when qLDPC was imported, if it was set, and otherwise `numba.config.NUMBA_NUM_THREADS`.
`QLDPC_NUM_THREADS` is the recommended control, and must be a positive integer.
Numba never reads it.
Numba's parallel code, such as some of galois's, uses numba's own count, which numba reads from the environment variable `NUMBA_NUM_THREADS` when it is imported, or else from the CPUs available to the process.
So `QLDPC_NUM_THREADS` sets numba's count too only if qLDPC is imported before numba, and so before galois: qLDPC then sets `NUMBA_NUM_THREADS` to the same value.
If numba was imported first with another count, qLDPC warns that numba keeps it; to avoid that, also set `NUMBA_NUM_THREADS` to the same value before Python starts.
The docstring of `src/qldpc/_num_threads.py` gives the other warnings.
Numba does not read the CPU quotas of containers, so in a container, set `QLDPC_NUM_THREADS` to its CPU quota rounded down, or to 1 if the quota is below 1.
`numba.set_num_threads` changes only the count of the calling thread's numba parallel code, not `QLDPC_NUM_THREADS` or `numba.config.NUMBA_NUM_THREADS`, nor the threads of searches.
`NUMBA_NUM_THREADS` alone still sets the count of searches when `QLDPC_NUM_THREADS` is not set.
To keep searches on one core, set `QLDPC_NUM_THREADS=1` before Python starts, which reaches every thread and worker process, or run them within `use_tuning(num_threads=1)`.
BLAS threads follow the BLAS library's own variables, such as `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` (Accelerate on macOS), or `OMP_NUM_THREADS`, which the sweep script of `experiments/distance_tuning` sets.
New threads and spawned processes normally start with the default tuning (see the module docstring of `_tuning.py`), so a program that runs searches in a pool of workers must enter `use_tuning` in the function that each worker runs.
That covers only the searches built there: a `CosetLeaderSearch` that the workers share must be built within `use_tuning(num_threads=1)`.
Programs that run searches in several processes or threads at once should cap the count in one of these ways, since each search would otherwise start a thread for each CPU.
Other searches, queries, and counts run on one thread, except the BZ distance search that a count may run first to find its default weight.

## Compiled code

Most searches spend their time in functions that numba compiles.
Two decorators in `src/qldpc/_compilation.py` mark them, and its docstring says when to use each:

- `@always_compiled` marks the per-word loops and their helpers.
  They are compiled on first use and cached on disk, in a directory named by a hash of qLDPC's source, so the first run after an edit anywhere in qLDPC compiles again.
  A small helper that hot loops call, such as `advance_support`, is marked `@always_compiled(inline=True)`, so that compiled callers include its code rather than pay for a call.
- `@compiled_on_request` marks parts of the set-up of searches, which run as plain Python unless a program calls `compile_search_setup()`.

An entry point that several threads run at once, such as `scan_set_items` in `_brouwer_zimmermann/affine_search.py` or `_search_supports` in `_connected_cluster/search.py`, is marked `@always_compiled(nogil=True)`, so that its compiled code, and every compiled function that it calls, runs without Python's global interpreter lock.
The docstring of `src/qldpc/_compilation.py` states the rules for that code.

Numba looks up the functions that compiled code calls when compiling it, so compiled code may call NumPy and compiled functions, but not plain Python functions of qLDPC.
Tests in `src/qldpc/_compiled_code_rules_test.py` check further rules, and the name of a failing test says which one was broken.

Set `QLDPC_DISABLE_COMPILATION=1` before importing qLDPC to run all of this code as plain Python while debugging.
It affects only qLDPC, whereas `NUMBA_DISABLE_JIT=1` also runs other packages that use numba, such as galois, as plain Python, which makes them slow too.

## Working here

- Tests sit beside their modules, and each module must be fully covered by its own test file.
  `python checks/pytest_.py src/qldpc/codes/distance` runs the tests.
  `python checks/coverage_.py src/qldpc/codes/distance/<module>.py` checks one module's coverage.
  The validation commands in `docs/source/contributing/index.rst` list the other checks.
- `conftest.py` holds the fixtures and helpers that several test files share, such as `enumerate_logical_ops`, which counts logical operators by enumeration.
- Coverage cannot trace compiled code, so the coverage check runs compiled functions as plain Python, and CI runs these tests again with them compiled (see `docs/source/contributing/standards/testing.rst`).
- `python -m pytest src/qldpc/codes/distance --compile-search-setup` runs the tests with the set-up code of searches compiled.
  Some tests request the `plain_search_setup` fixture of `conftest.py`, whose docstring says which tests need it.
- A change to the scans or how they step, to pruning, to the work items, or to any code that threads run must keep the requirements on threads in Section 5 of the guide of each method (`brouwer_zimmermann_code.md` or `connected_cluster.md`), and pass the tests on threads that the guide lists.
  So must a change to data that threads read, such as completion tables, the routes of coset searches, and the tables of a connected-cluster search, or to a decision that a search makes from the results of a layer or a weight limit.
  Code that threads run reads the tuning only from its search, never through `get_tuning()`, since new threads start with the default tuning (see the module docstring of `_tuning.py`).
- While each test of this package runs, the fixture `check_code_on_threads` of `conftest.py` fails it if its threads read the tuning through `get_tuning()` or call numba-parallel code from Python.
  `_thread_rules_test.py` checks that Section 5 of the guides names each entry point compiled with `nogil=True`.
  `test_nothing_asks_numba_for_parallel_code` of `src/qldpc/_compiled_code_rules_test.py` checks that no function of qLDPC is compiled with numba's `parallel` option.
  Four tests on threads, which Section 5 of each guide names, run with every array of their search read-only, through `check_shared_arrays_on_threads` or `shared_arrays_read_only` of `conftest.py`, so that numba rejects a write to one in any branch of compiled code.
  Section 5 of each guide lists the writes that this misses, which review checks.
- `_readme_test.py` fails if "Where to look" omits a module, if "Layout" omits a top-level module or subpackage, or if this page names a file that does not exist.
