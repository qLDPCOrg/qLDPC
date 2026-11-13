# Exact distances and logical-operator counts: a guide for contributors

This directory is the package `qldpc.codes.distance`.
It computes exact distances of binary codes, counts the logical operators of a given weight in binary CSS codes, and finds light Pauli errors with given syndromes on stabilizer states.
Users reach it through the public functions and classes below, which take plain arrays, and through the `get_distance_exact` methods of the code classes in `src/qldpc/codes/common.py`, which call the distance functions for binary codes.
`brouwer_zimmermann.md` explains the Brouwer-Zimmermann algorithm, maps its concepts to names in the code, describes its heuristics for choosing basis families, and lists the correctness requirements that every change to a Brouwer-Zimmermann search must keep.
`connected_support.md` explains the search of sparse binary codes over supports that are connected in a Tanner graph, maps it to the code, says what its cost depends on, and lists its correctness requirements.

## Public names

`__init__.py` re-exports these names, and everything else in this directory is private:

- `get_distance_from_generators`, `get_distance_from_checks`, `get_distance_css`, `get_distance_pauli`, `DistanceMethod`, and `NoDistance`, whose only member `NoDistance.NAN` is also exported as `NO_DISTANCE`, from `core.py`.
- `CSSLogicalCounter`, `CSSLogicalCounts`, `count_logical_ops_css`, `LogicalCountMethod`, and `compile_search_setup`, from `logical_counts.py`.
- `CosetLeaderSearch`, from `coset_leaders.py`, and `MinWeightBound`, the answer of its queries, from `min_weight_bounds.py`.

`qldpc.codes` re-exports only `NoDistance`, `NO_DISTANCE`, and `DistanceMethod`, which the code classes take or return, and docs cite the other names at `qldpc.codes.distance`.

## Terms

- **Sector:** the words of one type that a search examines, such as the logical operators of one type of a CSS code.
  A Brouwer-Zimmermann search takes a basis of them, with labels that tell nontrivial words from trivial ones, such as logical operators from stabilizers.
  A connected-support search takes the parity checks whose kernel holds them, and a matrix whose row space has the trivial ones.
- **Information set:** in the code, the basis of a sector, row reduced so that a block of its columns forms an identity matrix.
  `brouwer_zimmermann.md` calls this basis a search basis, and uses "information set" for the block of columns itself.
- **Basis family** (`BasisFamily`): the information sets that one Brouwer-Zimmermann search or count uses together, stacked into arrays.
- **Certificate** (`Certificate`): a lower bound on the weight of the words that a Brouwer-Zimmermann search has neither visited nor skipped, which tells the search when it can stop.
  With completion tables, it bounds only the words lighter than the best weight (see Section 3.9 of `brouwer_zimmermann.md`).
- **Connected-support search** (`SupportSearch`): the depth-first search of the supports that grow through unsatisfied parity checks, for a binary matrix of checks `H` and an excluded matrix `G`, whose words are binary, or symplectic for qubit codes that are not CSS.
  It is another exact method, selected by `method="connected_support"`, and it also finds light vectors with a given syndrome.
- **Scan:** the enumeration of the combinations of rows of each information set, layer by layer.
  Counting scans in `_counting/scan.py`.
  The Brouwer-Zimmermann distance search in `_brouwer_zimmermann/distance.py` and the Brouwer-Zimmermann coset searches in `_brouwer_zimmermann/coset.py` scan with the compiled scans of `_brouwer_zimmermann/affine_search.py`.
  All of them step through a layer with `advance_support` in `_information_sets/support_stepping.py`.

## Layout

- `core.py` and `logical_counts.py` are the public interfaces of the distances and the counts.
- `coset_leaders.py` is the public interface of the coset-leader search, and `min_weight_bounds.py` defines the answer of its queries, which its coset searches build too.
- `_brouwer_zimmermann/` holds the Brouwer-Zimmermann searches, which scan information sets, and `_connected_support/` the searches over connected supports.
  Each holds its distance search, and the coset searches with which the coset-leader search answers its queries by that method.
  Exhaustive enumeration fits in one module, `_brute_force.py`.
- `_counting/` holds the word scan of the counts, the local Clifford gates of deformed counters, and the MacWilliams method.
- `_information_sets/` holds the families of information sets and their certificates, which the Brouwer-Zimmermann searches and the counts share.
  The method name `"information_sets"` is an alias of `"brouwer_zimmermann"`, for distances and counts alike.
- `_cosets.py` holds what the coset searches of both methods share.
- `_tuning.py` holds the settings of the searches, and `_binary.py` the binary input checks, nested bases, the stabilizers of subsystem codes, and the symplectic-to-Hamming map.
  The code constructors in `src/qldpc/codes/common.py` also use its `infer_field`, which finds the field of rows given in lists or tuples.

## Where to look

| To change… | Look in (this directory) |
| --- | --- |
| A distance method, or the validation of method names | `core.py`, whose validation `coset_leaders.py` also uses; the early checks of `src/qldpc/circuits/certification/fault_propagation.py` and `src/qldpc/circuits/certification/fault_distance.py`; and where `logical_counts.py`, `_counting/scan.py`, `get_distance_exact`, and `get_distance` in `src/qldpc/codes/common.py` dispatch on the method name |
| The counter, its deformed counters, or `compile_search_setup` | `logical_counts.py` |
| The queries of the minimum weights of syndrome classes of Pauli errors (coset-leader weights), and how a query splits into coset searches, used by `src/qldpc/circuits/certification/` and `src/qldpc/experimental/circuits/benchmarking/residual_weights.py` | `coset_leaders.py` |
| The Brouwer-Zimmermann (BZ) distance search | `_brouwer_zimmermann/distance.py` |
| The compiled scans of layers that the BZ distance search and the BZ coset-leader search share, one body specialized by constant flags | `_brouwer_zimmermann/affine_search.py` |
| The packed information sets of a BZ distance search or a BZ coset-leader search, and the certificate of their steps | `_brouwer_zimmermann/affine_instance.py` |
| The BZ coset searches of the coset-leader search | `_brouwer_zimmermann/coset.py` |
| The tables of the steps and tails of BZ coset searches | `_brouwer_zimmermann/coset_tables.py` |
| The families of information sets with which a BZ coset-leader search answers costly queries, the gate that chooses them for each maximum weight, and the routes whose queries weigh words by their tails and skip completions by tables | `_brouwer_zimmermann/coset_routes.py` |
| The connected-support search, of distances and of vectors with a given syndrome | `_connected_support/search.py` |
| The connected-support coset searches of the coset-leader search | `_connected_support/coset.py` |
| Exhaustive enumeration, for small codes, fallbacks, and the `brute_force` method, with its packed symplectic weights | `_brute_force.py` |
| The scan of a count, or how its basis families follow local gates | `_counting/scan.py` |
| The local Clifford gates of deformed counters | `_counting/local_cliffords.py` |
| The MacWilliams counting method | `_counting/macwilliams.py` |
| The answers of the queries of the coset-leader search and of its coset searches (`MinWeightBound`) | `min_weight_bounds.py` |
| The parts that the coset searches of both methods share: words with unit syndromes, packed rows, sums of the rows that a support selects, and tables of the errors of weight zero or one | `_cosets.py` |
| How basis families are built | `_information_sets/basis_families.py` |
| Certificates, and how searches plan their steps: where they stop, and the work of the steps up to there | `_information_sets/certificates.py` |
| The step from one support of a layer to the next, which every scan of layers shares | `_information_sets/support_stepping.py` |
| Automorphism families: the images of one information set under automorphisms of a code, and the check of those automorphisms, which the connected-support search uses too | `_information_sets/automorphism_families.py` |
| Completion tables, which bound the words that complete the first rows of a BZ support with two more rows: a distance search skips the supports whose words cannot be lighter than its best word, a count those whose words all weigh more than its target weight, and a BZ coset query on a route by tails those whose words all weigh more than its maximum weight. A BZ search for the minimum weight of a coset skips those too, and those whose words cannot be lighter than its lightest word so far. Also the choice of whether to build them | `_information_sets/completion_bounds.py` |
| Information sets with bounded column coverage, by matroid partitioning | `_information_sets/matroid_partition.py` |
| The settings that a search reads when it starts, such as the thresholds that choose between exhaustive enumeration, the kinds of information-set families, and completion tables, the threshold of states before which a connected-support search asks for automorphisms, the coefficients of cost models, memory limits, and batch sizes (`SearchTuning`, `use_tuning`), and the constants that size caches when qLDPC is imported | `_tuning.py` |
| Binary input checks, nested bases, the stabilizers that `get_css_stabilizers` and `get_pauli_stabilizers` derive from gauge generators, or the map of symplectic vectors to binary vectors of twice their weight | `_binary.py` |
| Bit packing, bit counts, or row reduction over GF(2), which other subpackages share | `src/qldpc/_binary.py` |
| How numba code is compiled or cached | `src/qldpc/_compilation.py` |

## How the modules depend on each other

From the top:

1. `logical_counts.py`, which takes `NoDistance` from `core.py`, and `coset_leaders.py`, which takes `DistanceMethod` from it.
2. `core.py`.
3. The methods `_brouwer_zimmermann/`, `_connected_support/`, and `_counting/`.
4. `_information_sets/`, `_cosets.py`, `min_weight_bounds.py`, and `_brute_force.py`, which the methods build on.
   The connected-support search uses `infer_weight_divisor`, `verify_automorphisms`, and `get_orbits` of `_information_sets/`, and the Brouwer-Zimmermann distance search uses `_brute_force.py` for its exhaustive routes.
5. `_tuning.py`.
6. `_binary.py`.

A top-level module or subpackage imports only from lower levels, and a module also from its own subpackage.
`_layering_test.py` checks this.
Within a subpackage, the order is a convention.
In `_information_sets/`, `basis_families` builds on `certificates` and `matroid_partition`, `automorphism_families` on `basis_families` and `certificates`, and `completion_bounds` on `certificates`.
In `_brouwer_zimmermann/`, `distance` and `coset` build on `affine_instance` and `affine_search`, and `coset` also on `coset_routes` and `coset_tables`.
`coset_routes` builds on `coset_tables`, and both build on `affine_instance`.
In `_connected_support/`, `coset` builds on `search`, and in `_counting/`, `scan` on `local_cliffords`.
Modules of every level may also import `src/qldpc/_binary.py`, the bit-packed linear algebra over GF(2) that other subpackages share.
`src/qldpc/codes/common.py` imports this package, so `logical_counts.py` imports the code classes of `codes.common` inside the functions that use them.
Modules elsewhere in qLDPC, such as `src/qldpc/codes/common.py` and the deprecation registry in `src/qldpc/_deprecations/core.py`, use private names of this package, so grep `src/qldpc` for a private name before you move or rename it.

## The coset-leader search

`CosetLeaderSearch` answers each query with one or two coset searches, each of the cosets of one binary code.

- In general, one search of the Pauli errors, modulo the stabilizers, answers a query.
  Its syndromes are for the given stabilizers, so it is cached for the most recently used stabilizer matrices (see `PAULI_SEARCH_CACHE_SIZE` in `_tuning.py`).
- For the CSS weight on a CSS state, two classical searches cover the X parts and the Z parts of the errors separately, on `n` columns each rather than `2n`.
  A pure-X stabilizer has no Z part, a pure-Z stabilizer has no X part, and a stabilizer state is CSS if pure-X and pure-Z stabilizers generate its stabilizer group.
  The X parts of the pure-X stabilizers span a binary code `C_X`, and the Z parts of the pure-Z stabilizers a binary code `C_Z`.
  Multiplying an error by a stabilizer adds a word of `C_X` to its X part `x` and a word of `C_Z` to its Z part `z`, independently, so the smallest CSS weight of an error with a given syndrome is the larger of the smallest Hamming weights in the cosets `x + C_X` and `z + C_Z`.
  The coset of an X part is labeled by its syndrome for the pure-Z stabilizers, and that of a Z part by its syndrome for the pure-X stabilizers, and the search maps a syndrome for the given stabilizers to these two syndromes.
  The two coset searches depend only on the stabilizer group, so they are cached for the most recently used groups (see `CSS_SEARCH_CACHE_SIZE` in `_tuning.py`), and searches for different generating sets of one state share them.

The coset searches answer their own queries with `MinWeightBound` too, whose representative is then a word of their code: a packed Pauli error, or an X or Z part.
Each coset search first looks up the words of weight at most one in a table.

By default, each coset search is a Brouwer-Zimmermann (BZ) search, `BrouwerZimmermannCosetSearch` in `_brouwer_zimmermann/coset.py`, whose `_search_coset` scans the information sets of its code (Section 4.1 of `brouwer_zimmermann.md`).
With one BZ search, `find_first_not_within_max_weight` returns the bound that the batch search found, so the syndrome is not searched again, and with two, the bound that `get_min_weight_bound` gives.
`get_min_weights` answers the syndromes of a batch in one compiled loop for each coset search, with `BrouwerZimmermannCosetSearch.get_min_weights` (Section 3.10 of `brouwer_zimmermann.md`).
For the CSS weight, it searches the Z part of a class only if its X part is within the maximum weight, with the weight of the X part as the floor of that search.

With `method="connected_support"`, each coset search is a `ConnectedSupportCosetSearch` in `_connected_support/coset.py`, which runs `SupportSearch` of `_connected_support/search.py` from the syndrome of a query (Section 3.8 of `connected_support.md`).
It searches through the unsatisfied stabilizers for Pauli errors, the pure-Z stabilizers for X parts, and the pure-X stabilizers for Z parts.
It is fast if its checks are sparse.
Given stabilizers, such as row-reduced ones, can be dense, so it first replaces its checks with sparser ones that span the same space, which can fall short of the sparsest generators.
The CSS weight is not a sum over qubits, so it weighs errors by CSS weight only on CSS states, whose X and Z parts it searches separately.
A syndrome with an error of weight at most one gets the weight of its lightest errors as a lower bound.
For any other syndrome, a query with an error within `max_weight` gives the lower bound two, and a query with none gives the larger of two and `max_weight + 1`.
The BZ searches often give higher bounds.
`find_first_not_within_max_weight` searches each syndrome that the table lacks with increasing weight limits (`_get_batch_support_bound`), which can find light errors sooner than one search at `max_weight`.
So for the CSS weight, the representative of a part within `max_weight` can differ from the one that `get_min_weight_bound` gives.
`get_min_weights` also tries increasing weight limits (`_get_min_support_weight`), and starts the search of a Z part at the weight of its X part.

`CosetLeaderSearch` reads the active tuning when it is built, and the module docstring of `_tuning.py` says what a tuning can change about its answers.

- `disjoint_random_column_orders` configures every BZ search.
- `max_css_single_pairs` configures the BZ searches by CSS weight that are not split into two classical searches.
- `overlapping_minimum_work`, `max_information_sets`, and `overlapping_attempts_per_set` configure the families of information sets with which a BZ search answers costly queries (see `_brouwer_zimmermann/coset_routes.py`).
  The threshold `overlapping_minimum_work` also decides which queries of the BZ searches by physical weight, and of the classical searches, weigh words without the pivot columns of each set of their family, and skip the words that tables of bounds prove to weigh more than `max_weight`.
  `coset_completion_block_bits` and `completion_bound_max_table_bytes` configure these tables.
- `max_sparsifying_rounds` and `sparsifying_rounds_without_improvement` configure how the connected-support searches make their checks sparser.

## Tuning

Every value that changes only how long a search takes or how much memory it uses, and that a search can read when it starts, such as a threshold that chooses a route, a limit, a coefficient of a cost model, an attempt count, or a batch size, is a field of `SearchTuning` in `_tuning.py`.
The arguments of the public functions say what to compute and by which method, such as `cutoff`, `max_weight`, and `method`; do not add arguments that only tune a search.
Section 4.4 of `brouwer_zimmermann.md` says where each heuristic of a Brouwer-Zimmermann search uses its fields, and Sections 3.5 and 3.8 of `connected_support.md` where the connected-support searches use theirs.

- A search reads the active tuning once, when it starts, and passes the values that it needs down to the stages and loops that use them.
  The distance functions of `core.py` read it for a distance search, `CSSLogicalCounter.count_logical_ops` for a count, and `CosetLeaderSearch` when it is built; `get_distance_brouwer_zimmermann_many` and `get_distance_connected_support` read it only if their callers pass none.
  Pass these values as arguments, including to compiled code, and do not read the tuning anywhere else, so that one search never mixes two tunings.
  A value that the code keeps in a cache that outlives a search also keys that cache, as the fields of `_CosetTuning` in `_brouwer_zimmermann/coset.py` key the cached searches of `CosetLeaderSearch`.
- Only a value that qLDPC fixes before any search starts, such as the size of a cache, which `functools.lru_cache` fixes at import, is a constant in `_tuning.py`, read as `_tuning.NAME`.
  Constants fixed by the algorithm or by native integer widths, such as `WORK_CAP`, and the fixed seeds of random generators, which make results reproducible rather than tune anything, stay in the modules that use them.
- Document each field and constant in its own `#:` comment directly above it, so that Sphinx renders it.
  Say, in this order, what it decides and in what unit; how a smaller or larger value changes a search, including the effects of its smallest and largest values; how its default was chosen and on what data, or that this is not recorded, in one clause; and what to measure to retune it.
  Keep each comment readable without knowing the method that uses the value, and within a hundred words.
  State each fact in one place: link to a formula that the code computes, such as a cost model, rather than restating it, and put procedures that several values share, such as fitting a cost model, in the "Retuning" section of the module docstring.
  Do not put a colon on the first line of a comment, since Sphinx then reads the text before it as a type.
- [Exact-search performance and tuning](../../../../docs/source/exact_search_tuning.rst) renders the fields and constants from their comments, after hand-written tables that lead from symptoms and routes to their names.
  When you add, rename, or remove a field or constant, update that map, and for a constant the list of constants on the page.
  `_tuning_test.py` checks that the map names every field and every constant except `MAX_VALUE`, that the page lists every constant, and that each field and constant has its own `#:` comment, and the documentation build fails on a broken reference into `_tuning`.
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
