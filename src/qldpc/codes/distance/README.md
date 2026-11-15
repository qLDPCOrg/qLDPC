# Exact distances and logical-operator counts: a guide for contributors

This directory is the package `qldpc.codes.distance`.
It computes exact distances of binary codes, counts the logical operators of a given weight in binary CSS codes, and finds light Pauli errors with given syndromes on stabilizer states.
Users reach it through the public functions and classes below, which take plain arrays, and through the `get_distance_exact` methods of the code classes in `src/qldpc/codes/common.py`, which call the distance functions for binary codes.
`brouwer_zimmermann.md` explains the Brouwer-Zimmermann algorithm, maps its concepts to names in the code, describes its heuristics for choosing basis families, and lists the correctness requirements that every change to a Brouwer-Zimmermann search must keep.
`connected_cluster.md` explains the search of sparse binary codes over supports that are connected in a Tanner graph, maps it to the source code, says what its cost depends on, and lists its correctness requirements.

## Public names

`__init__.py` re-exports these names, and everything else in this directory is private:

- `get_distance_from_generators`, `get_distance_from_checks`, `get_distance_css`, `get_distance_pauli`, `DistanceMethod`, and `NoDistance`, whose only member `NoDistance.NAN` is also exported as `NO_DISTANCE`, from `core.py`.
- `CSSLogicalCounter`, `CSSLogicalCounts`, `count_logical_ops_css`, `LogicalCountMethod`, and `compile_search_setup`, from `logical_counts.py`.
- `CosetLeaderSearch`, from `coset_leaders.py`, and `MinWeightBound`, the answer of its queries, from `min_weight_bounds.py`.

`qldpc.codes` re-exports only `NoDistance`, `NO_DISTANCE`, and `DistanceMethod`, which the code classes take or return.
The documentation cites the other names at `qldpc.codes.distance`.

## Terms

- **Sector:** the words of one type that a search examines, such as the logical operators of one type of a CSS code.
  A Brouwer-Zimmermann search takes a basis of them, with labels that tell nontrivial words from trivial ones, such as logical operators from stabilizers.
  A connected-cluster search takes the parity checks whose kernel holds them, and a matrix whose row space has the trivial ones.
- **Information set:** in most of the source, the basis of a sector, row reduced on `rank` pivot columns, where `rank` is at most the dimension of the basis, so that its first `rank` rows form an identity matrix on those columns and its other rows are zero there.
  In a full set, `rank` is the dimension of the basis.
  A partial set, whose `rank` is smaller, goes by the same name.
  `brouwer_zimmermann.md`, and a few docstrings of `_brouwer_zimmermann/distance.py`, call this basis a search basis or a set, and use "information set" only for the pivot columns of a full set (Sections 2.3 and 2.4).
- **Basis family** (`BasisFamily`): the information sets that one Brouwer-Zimmermann search or count uses together, stacked into arrays.
- **Certificate** (`Certificate`): a lower bound on the weight of the words that a Brouwer-Zimmermann search has neither visited nor skipped, which tells the search when it can stop.
  With completion tables, it bounds only the words lighter than the lightest word that the search has found (see Section 3.9 of `brouwer_zimmermann.md`).
- **Connected-cluster search** (`SupportSearch`): a depth-first search for light words that satisfy the rows of a binary matrix `H` of parity checks and lie outside the row space of an excluded matrix `G`.
  It grows supports, the sets of variables on which words are nonzero, only through unsatisfied checks.
  The words are binary, or symplectic for qubit codes that are not CSS.
  `method="connected_cluster"` selects this search, which also finds light vectors with a given syndrome.
- **Scan:** the enumeration of the combinations of rows of each information set, layer by layer, where a layer holds the combinations of a given number of rows.
  Counts have their own scan, in `_counting/scan.py`.
  The Brouwer-Zimmermann distance search in `_brouwer_zimmermann/distance.py` and the Brouwer-Zimmermann coset searches in `_brouwer_zimmermann/coset.py` use the compiled scans of `_brouwer_zimmermann/affine_search.py`.
  Every scan steps through a layer with `advance_support` in `_information_sets/support_stepping.py`.

## Layout

- `core.py` and `logical_counts.py` are the public interfaces of the distances and the counts.
- `coset_leaders.py` is the public interface of the coset-leader search, and `min_weight_bounds.py` defines the answer of its queries, which its coset searches build too.
- `_brouwer_zimmermann/` holds the Brouwer-Zimmermann searches, which scan information sets, and `_connected_cluster/` the connected-cluster searches, which grow connected supports.
  Each holds the distance search of its method, and the coset searches with which the coset-leader search answers queries by that method.
  `_brute_force.py` holds exhaustive enumeration.
- `_counting/` holds the word scan of the counts, the local Clifford gates of deformed counters, and the MacWilliams method.
- `_information_sets/` holds the families of information sets and their certificates, which the Brouwer-Zimmermann searches and the counts share.
  The method name `"information_sets"` is an alias of `"brouwer_zimmermann"`, for distances and counts alike.
- `_cosets.py` holds the parts that the coset searches of both methods share.
- `_tuning.py` holds the settings of the searches.
- `_binary.py` holds the binary input checks, nested bases, the stabilizers of subsystem codes, and the symplectic-to-Hamming map.
  The code constructors in `src/qldpc/codes/common.py` also use its `infer_field`, which finds the field of rows given in lists or tuples.

## Where to look

| To change… | Look in (this directory) |
| --- | --- |
| A distance method, or the validation of method names | `core.py`, whose validation `coset_leaders.py` also uses; the early checks of `src/qldpc/circuits/certification/fault_propagation.py` and `src/qldpc/circuits/certification/fault_distance.py`; where `logical_counts.py` and `get_distance_exact` in `src/qldpc/codes/common.py` dispatch on the method name, and where `get_distance` validates it; and the `brute_force` flag that `logical_counts.py` passes to `_counting/scan.py` |
| The counter, its deformed counters, or `compile_search_setup` | `logical_counts.py` |
| The coset-leader search, which `src/qldpc/circuits/certification/` and `src/qldpc/experimental/circuits/benchmarking/residual_weights.py` use: its queries of the least weight of the Pauli errors with a given syndrome, and how a query splits into coset searches | `coset_leaders.py` |
| The Brouwer-Zimmermann (BZ) distance search | `_brouwer_zimmermann/distance.py` |
| The compiled scans of layers that the BZ distance search and the BZ coset searches share, one body specialized by constant flags | `_brouwer_zimmermann/affine_search.py` |
| The instances of BZ distance searches and BZ coset searches: their packed information sets, and the certificate of these sets | `_brouwer_zimmermann/affine_instance.py` |
| The BZ coset searches of the coset-leader search | `_brouwer_zimmermann/coset.py` |
| The tables of BZ coset searches: the bounds of their steps and their stopping steps, and the tails and completion tables of their routes by tails | `_brouwer_zimmermann/coset_tables.py` |
| The routes of BZ coset searches: the family of information sets that answers the queries of each maximum weight, the gate that chooses it, and the routes by tails, whose queries weigh words by their tails and skip completions by tables | `_brouwer_zimmermann/coset_routes.py` |
| The connected-cluster search, of distances and of vectors with a given syndrome | `_connected_cluster/search.py` |
| The connected-cluster coset searches of the coset-leader search | `_connected_cluster/coset.py` |
| How the connected-cluster searches make their checks sparser with the same span | `_connected_cluster/sparse_checks.py` |
| Exhaustive enumeration, which the `brute_force` method runs, with its packed symplectic weights | `_brute_force.py` |
| The scan of a count, or how its basis families follow local gates | `_counting/scan.py` |
| The local Clifford gates of deformed counters | `_counting/local_cliffords.py` |
| The MacWilliams counting method | `_counting/macwilliams.py` |
| The answers of the queries of the coset-leader search and of its coset searches (`MinWeightBound`) | `min_weight_bounds.py` |
| The parts that the coset searches of both methods share: words with unit syndromes, packed rows, sums of the rows that a support selects, and tables of the errors of weight zero or one | `_cosets.py` |
| How basis families are built | `_information_sets/basis_families.py` |
| Certificates, and how searches plan their steps: where they stop, and the work of the steps up to there | `_information_sets/certificates.py` |
| The step from one support of a layer to the next, which every scan of layers shares | `_information_sets/support_stepping.py` |
| Automorphism families, the images of one information set under automorphisms of a code; and the check of those automorphisms and the orbits that they generate, which the connected-cluster search uses too | `_information_sets/automorphism_families.py` |
| Completion tables, which bound the weights of the words that complete a BZ support with two more rows, so that a scan can skip the supports whose words are too heavy for its search; and the choice of whether to build them | `_information_sets/completion_bounds.py` |
| Information sets with bounded column coverage, by matroid partitioning | `_information_sets/matroid_partition.py` |
| The settings that a search reads when it starts (`SearchTuning` and `use_tuning`), such as the thresholds that choose routes, the coefficients of cost models, memory limits, and batch sizes; and the constants that size caches when qLDPC is imported | `_tuning.py` |
| Binary input checks, nested bases, the stabilizers that `get_css_stabilizers` and `get_pauli_stabilizers` derive from gauge generators, or the map of symplectic vectors to binary vectors of twice their weight | `_binary.py` |
| Bit packing, bit counts, or row reduction over GF(2), which other subpackages share | `src/qldpc/_binary.py` |
| How numba code is compiled or cached | `src/qldpc/_compilation.py` |

## How the modules depend on each other

From the top:

1. `logical_counts.py`, which takes `NoDistance` from `core.py`, and `coset_leaders.py`, which takes `DistanceMethod` from it.
2. `core.py`.
3. The methods `_brouwer_zimmermann/`, `_connected_cluster/`, `_counting/`, and `_brute_force.py`.
4. `_information_sets/`, `_cosets.py`, and `min_weight_bounds.py`, which the methods build on.
   The connected-cluster search uses `infer_weight_divisor`, `verify_automorphisms`, and `get_orbits` of `_information_sets/`.
5. `_tuning.py`.
6. `_binary.py`.

A top-level module or subpackage imports only from lower levels, and a module also from its own subpackage.
`_layering_test.py` checks this.
Within a subpackage, the order is a convention.
In `_information_sets/`, `basis_families` builds on `certificates` and `matroid_partition`, `automorphism_families` on `basis_families` and `certificates`, and `completion_bounds` on `certificates`.
In `_brouwer_zimmermann/`, `distance` and `coset` build on `affine_instance` and `affine_search`, and `coset` also on `coset_routes` and `coset_tables`.
`coset_routes` builds on `coset_tables`, and both build on `affine_instance`.
In `_connected_cluster/`, `coset` builds on `search` and `sparse_checks`, and in `_counting/`, `scan` on `local_cliffords`.
Modules of every level may also import `src/qldpc/_binary.py`, the bit-packed linear algebra over GF(2) that other subpackages share.
`src/qldpc/codes/common.py` imports this package, so `logical_counts.py` imports the code classes of `codes.common` inside the functions that use them.
Modules elsewhere in qLDPC, such as `src/qldpc/codes/common.py` and the deprecation registry in `src/qldpc/_deprecations/core.py`, use private names of this package, so grep `src/qldpc` for a private name before you move or rename it.

## The coset-leader search

`CosetLeaderSearch` answers each query with one or two coset searches, each of the cosets of one binary code.

- In general, one search of the Pauli errors, modulo the stabilizers, answers a query.
- For the CSS weight on a CSS state, two classical searches cover the X parts and the Z parts of the errors separately, on `n` columns each rather than `2n`.
  A pure-X stabilizer has no Z part, a pure-Z stabilizer has no X part, and a stabilizer state is CSS if pure-X and pure-Z stabilizers generate its stabilizer group.
  The X parts of the pure-X stabilizers span a binary code `C_X`, and the Z parts of the pure-Z stabilizers a binary code `C_Z`.
  Multiplying an error by a stabilizer adds a word of `C_X` to its X part `x` and a word of `C_Z` to its Z part `z`, independently.
  So the smallest CSS weight of an error with a given syndrome is the larger of the smallest Hamming weights in the cosets `x + C_X` and `z + C_Z`.
  The coset of an X part is labeled by its syndrome for the pure-Z stabilizers, and that of a Z part by its syndrome for the pure-X stabilizers.
  `CosetLeaderSearch` maps a syndrome for the given stabilizers to these two syndromes.

The coset searches answer their own queries with `MinWeightBound` too, whose representative is then a word of their code: a packed Pauli error, or an X or Z part.
Each coset search first looks up, in a table, a word of weight at most one with the syndrome of the query.

By default, each coset search is a Brouwer-Zimmermann (BZ) search, `BrouwerZimmermannCosetSearch` in `_brouwer_zimmermann/coset.py`, whose `_search_coset` scans the information sets of its code (Section 4.1 of `brouwer_zimmermann.md`).
The BZ search of the Pauli errors depends on the given stabilizers, since its syndromes are for them, so it is cached for the most recently used stabilizer matrices (see `PAULI_SEARCH_CACHE_SIZE` in `_tuning.py`).
The two BZ searches of X and Z parts depend only on the stabilizer group, so they are cached for the most recently used groups (see `CSS_SEARCH_CACHE_SIZE` in `_tuning.py`), and the `CosetLeaderSearch` objects for different generating sets of one state share them.
With one BZ search, `find_first_not_within_max_weight` returns the bound that this search found for the first syndrome without an error within `max_weight`, so that syndrome is not searched again.
With two BZ searches, it returns the bound that `get_min_weight_bound` gives for that syndrome.
`get_min_weights` answers the syndromes of a batch in one compiled loop for each coset search, with `BrouwerZimmermannCosetSearch.get_min_weights` (Section 3.10 of `brouwer_zimmermann.md`).
For the CSS weight, it searches the Z part of a class only if its X part is within the maximum weight, with the weight of the X part as the floor of that search.

With `method="connected_cluster"`, each coset search is a `ConnectedClusterCosetSearch` in `_connected_cluster/coset.py`, which runs `SupportSearch` of `_connected_cluster/search.py` from the syndrome of a query (Section 3.8 of `connected_cluster.md`).
Each `CosetLeaderSearch` builds its own connected-cluster coset searches.
A connected-cluster coset search grows supports through its unsatisfied checks: the stabilizers for Pauli errors, the pure-Z stabilizers for X parts, and the pure-X stabilizers for Z parts.
It first replaces its checks with sparser checks that span the same space, with `get_sparser_rows` of `_connected_cluster/sparse_checks.py`, since dense checks slow it (Section 3.8 of `connected_cluster.md`).
The CSS weight is not a sum over qubits, so this method weighs errors by CSS weight only on CSS states, by searching their X and Z parts separately.
The class docstring of `ConnectedClusterCosetSearch` states the lower bounds of its answers.
`find_first_not_within_max_weight` searches each syndrome that the table lacks with increasing weight limits (`_get_batch_cluster_bound`), which can find light errors sooner than one search at `max_weight`.
So for the CSS weight, the representative of a part within `max_weight` can differ from the one that `get_min_weight_bound` gives.
`get_min_weights` also tries increasing weight limits (`_get_min_cluster_weight`; Section 3.8 of `connected_cluster.md`).

`CosetLeaderSearch` reads the active tuning when it is built, and the module docstring of `_tuning.py` says what a tuning can change about its answers.

- `disjoint_random_column_orders` configures every BZ search.
- `max_css_single_pairs` configures the BZ searches by CSS weight that are not split into two classical searches.
- `coset_route_minimum_work`, `max_information_sets`, and `overlapping_attempts_per_set` configure the families of information sets with which a BZ search answers costly queries (see `_brouwer_zimmermann/coset_routes.py`).
  The threshold `coset_route_minimum_work` also decides which queries of the BZ searches by physical weight, and of the classical searches, take routes by tails.
  These queries weigh words without the pivot columns of each set of their family, and skip the words that tables of bounds prove to weigh more than `max_weight`.
  `coset_completion_block_bits` and `completion_bound_max_table_bytes` configure these tables.
- `max_sparsifying_rounds` and `sparsifying_rounds_without_improvement` configure how the connected-cluster coset searches make their checks sparser.

## Tuning

Every value that changes only how long a search takes or how much memory it uses, and that a search can read when it starts, such as a threshold that chooses a route, a limit, a coefficient of a cost model, an attempt count, or a batch size, is a field of `SearchTuning` in `_tuning.py`.
The arguments of the public functions say what to compute and by which method, such as `cutoff`, `max_weight`, and `method`; do not add arguments that only tune a search.
Section 4.4 of `brouwer_zimmermann.md` explains the heuristics of the Brouwer-Zimmermann searches and the fields that they use, and Sections 3.5, 3.8, and 3.9 of `connected_cluster.md` explain the heuristics and fields of the connected-cluster searches.

- A search reads the active tuning once, when it starts, and passes the values that it needs down to the stages and loops that use them.
  The distance functions of `core.py` read it for a distance search, `CSSLogicalCounter.count_logical_ops` for a count, and `CosetLeaderSearch` when it is built.
  `get_distance_brouwer_zimmermann_many` and `get_distance_connected_cluster` read it only if their callers pass none.
  Pass these values as arguments, including to compiled code, and do not read the tuning anywhere else, so that one search never mixes two tunings.
  A value that the code keeps in a cache that outlives a search also keys that cache, as the fields of `_CosetTuning` in `_brouwer_zimmermann/coset.py` key the cached searches of `CosetLeaderSearch`.
- Only a value that qLDPC fixes before any search starts, such as the size of a cache, which `functools.lru_cache` fixes at import, is a constant in `_tuning.py`, read as `_tuning.NAME`.
  Constants fixed by the algorithm or by native integer widths, such as `WORK_CAP`, and the fixed seeds of random generators, which make results reproducible rather than tune anything, stay in the modules that use them.
- Document each field and constant in its own `#:` comment directly above it, so that Sphinx renders it.
  A field's comment says three things, and no more.
  First, it says what the field controls, in what unit, and what special values, such as zero and `MAX_VALUE`, do.
  Second, it states the heuristic behind the default in a sentence: the class of the default and the rule of that class, as the "Retuning" section of the module docstring defines them, or, for a field in no class, why the default was chosen.
  Third, it points to the section of `brouwer_zimmermann.md` or `connected_cluster.md` that explains this heuristic.
  A longer comment needs a reason, such as an interaction with another field that a reader could not guess.
  A constant's comment says what the constant controls and why its value was chosen.
  Keep each comment readable without knowing the method that uses the value.
  State each fact in one place: point to the guide for the reasoning behind a heuristic, and to a formula that the code computes, such as a cost model, rather than restating them.
  The "Retuning" section of the module docstring holds what several values share: the classes of defaults, and the rule for changing a default.
  Do not put a colon on the first line of a comment, since Sphinx then reads the text before it as a type.
- Comments, docstrings, and guides in this package give no measurements: no times, ratios, counts, or percentages from runs, and no machines or workloads as their evidence.
  The rule of a gate, a threshold that turns on an optional step, may state the cost of the step that its heuristic assumes, as a round order of magnitude.
  Measurements of a change to the tuning belong in the description of its pull request.
- Change a default only as the "Retuning" section of the module docstring of `_tuning.py` allows.
- [Exact-search performance and tuning](../../../../docs/source/exact_search_tuning.rst) renders the fields and constants from their comments, after hand-written tables that lead from symptoms and routes to the names of the values.
  When you add, rename, or remove a field or constant, update those tables, and for a constant the list of constants on the page.
  `_tuning_test.py` checks that the route map of that page names every field and every constant except `MAX_VALUE`, that the page lists every constant, and that each field and constant has its own `#:` comment.
  The documentation build fails on a broken reference into `_tuning`.
- A new heuristic whose value could change a result that a search guarantees does not belong in `SearchTuning`.
  The module docstring of `_tuning.py` says which results a tuning can change, and how.
- To force a route or exercise a value in a test, call the `tune` fixture of `src/qldpc/codes/conftest.py`, as in `tune(overlapping_minimum_work=0)`, rather than patching module attributes.
- For a field whose full range the implementation cannot safely support, narrow its range in its metadata rather than adding code for unsupported values.

## Compiled code

Most searches spend their time in functions that numba compiles.
Two decorators in `src/qldpc/_compilation.py` mark them, and its docstring says when to use each:

- `@always_compiled` marks the per-word loops and their helpers.
  They are compiled on first use and cached on disk, in a directory named by a hash of qLDPC's source, so the first run after an edit anywhere in qLDPC compiles again.
  A small helper that hot loops call, such as `advance_support`, is marked `@always_compiled(inline=True)`, so that compiled callers include its code rather than pay for a call.
- `@compiled_on_request` marks parts of the set-up of searches, which run as plain Python unless a program calls `compile_search_setup()`.

Numba looks up the functions that compiled code calls when it compiles it, so compiled code may call NumPy and compiled functions, but not plain Python functions of qLDPC.
Tests in `src/qldpc/_compilation_test.py` check further rules, and the name of a failing test says which one was broken.

Set `QLDPC_DISABLE_COMPILATION=1` before importing qLDPC to run all of this code as plain Python while debugging.
It affects only qLDPC, whereas `NUMBA_DISABLE_JIT=1` also runs other packages that use numba, such as galois, as plain Python, which makes them slow too.

## Working here

- Tests sit beside their modules, and each module must be fully covered by its own test file.
  `python checks/pytest_.py src/qldpc/codes/distance` runs the tests.
  `python checks/coverage_.py src/qldpc/codes/distance/<module>.py` checks one module's coverage.
  The validation commands in `docs/source/contributing/index.rst` list the other checks.
- `conftest.py` holds the fixtures and helpers that several test files share, such as a check of counts against brute-force enumeration.
- Coverage cannot trace compiled code, so the coverage check runs compiled functions as plain Python, and CI runs these tests again with them compiled (see `docs/source/contributing/standards/testing.rst`).
- `python -m pytest src/qldpc/codes/distance --compile-search-setup` runs the tests with the set-up code of searches compiled.
  Some tests request the `plain_search_setup` fixture of `conftest.py`, whose docstring says which tests need it.
- `_readme_test.py` fails if this page omits a module, or names a file that does not exist.
