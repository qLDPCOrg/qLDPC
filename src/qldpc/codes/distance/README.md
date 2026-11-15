# Exact distances and logical-operator counts: a guide for contributors

This directory is the package `qldpc.codes.distance`.
It computes exact distances of binary codes, counts the logical operators of a given weight in binary CSS codes, and finds light Pauli errors with given syndromes on stabilizer states.
Users reach it through the public functions and classes below, which take plain arrays, and through the `get_distance_exact` methods of the code classes in `src/qldpc/codes/common.py`, which call the distance functions for binary codes.
`brouwer_zimmermann.md` and `connected_cluster.md` explain the Brouwer-Zimmermann and connected-cluster searches, map them to the source code, explain their heuristics, and list the requirements that every change to them must keep.

## Public names

`__init__.py` re-exports these names, and everything else in this directory is private:

- `get_distance_from_generators`, `get_distance_from_checks`, `get_distance_css`, `get_distance_pauli`, `DistanceMethod`, and `NoDistance`, whose only member `NoDistance.NAN` is also exported as `NO_DISTANCE`, from `core.py`.
- `CSSLogicalCounter`, `CSSLogicalCounts`, `count_logical_ops_css`, `LogicalCountMethod`, and `compile_search_setup`, from `logical_counts.py`.
- `CosetLeaderSearch`, from `coset_leaders.py`, and `MinWeightBound`, its bound for one syndrome, from `min_weight_bounds.py`.

`qldpc.codes` re-exports only `NoDistance`, `NO_DISTANCE`, and `DistanceMethod`, which the code classes take or return.
The documentation cites the other names at `qldpc.codes.distance`.

## Terms

- **Sector:** the words of one type that a search examines, such as the logical operators of one type of a CSS code.
  A Brouwer-Zimmermann search takes a basis of them, labeled to tell nontrivial words, such as logical operators, from trivial ones, such as stabilizers.
  A connected-cluster search takes checks whose kernel holds them and a matrix whose row space holds the trivial ones.
- **Information set:** in most of the source, a basis of a sector, row reduced on `rank` pivot columns: its first `rank` rows are the identity there and its other rows zero.
  In a full set, `rank` is the dimension of the basis; in a partial set it is smaller.
  `brouwer_zimmermann.md` and a few docstrings of `_brouwer_zimmermann/distance.py` call this basis a search basis or a set, and use "information set" only for the pivot columns of a full set (Sections 2.3 and 2.4).
- **Basis family** (`BasisFamily`): the information sets that one Brouwer-Zimmermann search or count uses together, stacked into arrays.
- **Certificate** (`Certificate`): a lower bound on the weight of the words that a Brouwer-Zimmermann search has neither visited nor skipped, which tells the search when it can stop; with completion tables, it bounds only the words lighter than the best weight `U` of the search (Section 3.9 of `brouwer_zimmermann.md`).
- **Connected-cluster search** (`SupportSearch`): the depth-first search that `method="connected_cluster"` selects, which grows supports through the unsatisfied rows of a check matrix `H` to find light binary or symplectic words outside the row space of an excluded matrix `G`, or light vectors with a given syndrome.
- **Scan:** the enumeration of the combinations of rows of each information set, layer by layer, a layer holding the combinations of a given number of rows.
  Counts have their own scan, in `_counting/scan.py`, and the Brouwer-Zimmermann distance and coset searches (see "The coset-leader search") use the compiled scans of `_brouwer_zimmermann/affine_search.py`.
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
| The coset-leader search, which `src/qldpc/circuits/certification/` and `src/qldpc/experimental/circuits/benchmarking/residual_weights.py` use, and how its queries split into coset searches | `coset_leaders.py` |
| The Brouwer-Zimmermann (BZ) distance search | `_brouwer_zimmermann/distance.py` |
| The compiled scans of layers that the BZ distance search and the BZ coset searches share, one body specialized by constant flags | `_brouwer_zimmermann/affine_search.py` |
| The instances of BZ distance searches and BZ coset searches: their packed information sets, and the certificate of these sets | `_brouwer_zimmermann/affine_instance.py` |
| The BZ coset searches of the coset-leader search | `_brouwer_zimmermann/coset.py` |
| The tables of BZ coset searches: the bounds of their steps and their stopping steps, and the tails and completion tables of their routes by tails | `_brouwer_zimmermann/coset_tables.py` |
| The routes of BZ coset searches: the family of information sets for each maximum weight, the gate that chooses it, and the routes by tails | `_brouwer_zimmermann/coset_routes.py` |
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
| Completion tables, which let a BZ scan skip the supports whose completions by two more rows are too heavy, and the choice of whether to build them | `_information_sets/completion_bounds.py` |
| Information sets with bounded column coverage, by matroid partitioning | `_information_sets/matroid_partition.py` |
| The settings that a search reads when it starts (`SearchTuning` and `use_tuning`), and the constants that size caches when qLDPC is imported | `_tuning.py` |
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
Within a subpackage, the layering below is a convention.
In `_information_sets/`, `basis_families` builds on `certificates` and `matroid_partition`, `automorphism_families` on `basis_families` and `certificates`, and `completion_bounds` on `certificates`.
In `_brouwer_zimmermann/`, `distance` and `coset` build on `affine_instance` and `affine_search`, and `coset` also on `coset_routes` and `coset_tables`.
`coset_routes` builds on `coset_tables`, and both build on `affine_instance`.
In `_connected_cluster/`, `coset` builds on `search` and `sparse_checks`, and in `_counting/`, `scan` on `local_cliffords`.
Modules of every level may also import `src/qldpc/_binary.py`, the bit-packed linear algebra over GF(2) that other subpackages share.
`src/qldpc/codes/common.py` imports this package, so `logical_counts.py` imports the code classes of `codes.common` inside the functions that use them.
Modules elsewhere in qLDPC, such as `src/qldpc/codes/common.py` and the deprecation registry in `src/qldpc/_deprecations/core.py`, use private names of this package, so grep `src/qldpc` for a private name before you move or rename it.

## The coset-leader search

`CosetLeaderSearch` answers each query with one or two coset searches, each over the cosets of a binary code.
Each answers a single query with `MinWeightBound`, after it looks up words of weight at most one in a table.

- In general, one search of the Pauli errors, modulo the stabilizers, answers a query.
- On a CSS state, where pure-X and pure-Z stabilizers generate its stabilizer group, two classical searches answer a query by CSS weight, the larger of the weights of the X and Z parts of an error, on `n` columns each rather than `2n`.
  The X parts of the pure-X stabilizers span a code `C_X`, and the Z parts of the pure-Z stabilizers a code `C_Z`.
  Multiplying an error by a stabilizer adds a word of `C_X` to its X part `x` and, independently, a word of `C_Z` to its Z part `z`, so the smallest CSS weight with a given syndrome is the larger of the smallest weights in `x + C_X` and `z + C_Z`.
  `CosetLeaderSearch` maps the given syndrome to the syndromes of `x` for the pure-Z stabilizers and of `z` for the pure-X stabilizers.
  `get_min_weights` searches the Z part of an error only if its X part is within `max_weight`, with the weight of the X part as the floor of that search.

By default, each coset search is a Brouwer-Zimmermann (BZ) search, `BrouwerZimmermannCosetSearch` in `_brouwer_zimmermann/coset.py`, whose `_search_coset` scans the information sets of its code (Section 4.1 of `brouwer_zimmermann.md`).
The BZ search of Pauli errors is cached by stabilizer matrix (`PAULI_SEARCH_CACHE_SIZE` in `_tuning.py`), since its syndromes are for the given stabilizers.
Those of X and Z parts are cached by stabilizer group (`CSS_SEARCH_CACHE_SIZE`), so `CosetLeaderSearch` objects for different generators of one state share them.
`find_first_not_within_max_weight` returns the bound that a single BZ search found for the first syndrome without an error within `max_weight`, or, with two BZ searches, the bound that `get_min_weight_bound` gives for that syndrome.
`get_min_weights` answers a batch in one compiled loop per coset search (`BrouwerZimmermannCosetSearch.get_min_weights`; Section 3.10 of `brouwer_zimmermann.md`).

With `method="connected_cluster"`, each `CosetLeaderSearch` builds its own coset searches, `ConnectedClusterCosetSearch` in `_connected_cluster/coset.py`, which make their checks sparser with `get_sparser_rows` and run `SupportSearch` from the syndrome of a query (Section 3.8 of `connected_cluster.md`); their checks are the stabilizers for Pauli errors, the pure-Z stabilizers for X parts, and the pure-X stabilizers for Z parts.
This method weighs errors by CSS weight only on CSS states (Section 3.8 of `connected_cluster.md`).
The class docstring of `ConnectedClusterCosetSearch` states the lower bounds of its answers.
`find_first_not_within_max_weight` and `get_min_weights` try increasing weight limits (`_get_batch_cluster_bound` and `_get_min_cluster_weight`), which can find light errors sooner than one search at `max_weight`.
For the CSS weight, the bound that `find_first_not_within_max_weight` returns holds an error for each part, and the error of a part within `max_weight` can differ from the one that `get_min_weight_bound` gives.

The table "Routes and the names that control them" of [Exact-search performance and tuning](../../../../docs/source/exact_search_tuning.rst) names the fields that configure the searches of `CosetLeaderSearch`.

## Tuning

A value that changes how long a search takes or how much memory it uses, but no result that the search guarantees, and that a search reads when it starts, such as a threshold that chooses a route, a limit, a coefficient of a cost model, or a batch size, is a field of `SearchTuning` in `_tuning.py`, not an argument of a public function.

- A search reads the active tuning once, when it starts, and passes the values that it needs down as arguments, including to compiled code, so that one search never mixes two tunings.
  The distance functions of `core.py` read it for a distance search, `CSSLogicalCounter.count_logical_ops` for a count, and `CosetLeaderSearch` when it is built; `get_distance_brouwer_zimmermann_many` and `get_distance_connected_cluster` read it only if their callers pass none.
  A cache that outlives a search is keyed by the values that it depends on, as the fields of `_CosetTuning` in `_brouwer_zimmermann/coset.py` key the cached searches of `CosetLeaderSearch`.
- Only a value that qLDPC fixes before any search starts, such as the size of a cache, which `functools.lru_cache` fixes at import, is a constant in `_tuning.py`, read as `_tuning.NAME`.
  Constants fixed by the algorithm or by native integer widths, such as `WORK_CAP`, and the seeds of random generators stay in the modules that use them, except `MAX_VALUE`, the largest value of an integer field.
- Each field and constant has its own `#:` comment directly above it, for Sphinx to render, with no colon on its first line, since Sphinx reads text before such a colon as a type.
  A field's comment says what the field controls, with its unit and special values, and the class and rule of its default, as the "Retuning" section of the module docstring defines them, or why a field in no class has its default; where a guide section or a cost model explains its heuristic, the comment points there.
  A constant's comment says what it controls; the "Retuning" section says how the sizes of caches were chosen.
  Comments, docstrings, and guides in this package give no measurements; those belong in pull request descriptions.
  The comment of a derived value, a gate on an optional step, may state the cost of the step that its heuristic assumes, as a round order of magnitude.
- Change a default only as the "Retuning" section allows.
- [Exact-search performance and tuning](../../../../docs/source/exact_search_tuning.rst) renders these comments after hand-written tables that lead from symptoms and routes to the values; update the tables, and for a constant the list of constants, when you add, rename, or remove a value.
  `_tuning_test.py` checks that the table of routes names every field and every constant except `MAX_VALUE`, that the page lists every constant, and that each value has its own comment.
  The documentation build fails on a broken reference into `_tuning`.
- A heuristic whose value could change a result that a search guarantees does not belong in `SearchTuning`; the module docstring of `_tuning.py` says which results a tuning can change.
- To force a route or exercise a value in a test, call the `tune` fixture of `src/qldpc/codes/conftest.py`, as in `tune(overlapping_minimum_work=0)`, rather than patching module attributes.
- If the implementation cannot safely support a field's full range, narrow it in the field's metadata rather than adding code for unsupported values.

## Compiled code

Most searches spend their time in functions that numba compiles.
Two decorators in `src/qldpc/_compilation.py` mark them, and its docstring says when to use each:

- `@always_compiled` marks the per-word loops and their helpers.
  They are compiled on first use and cached on disk, in a directory named by a hash of qLDPC's source, so the first run after an edit anywhere in qLDPC compiles again.
  A small helper that hot loops call, such as `advance_support`, is marked `@always_compiled(inline=True)`, so that compiled callers include its code rather than pay for a call.
- `@compiled_on_request` marks parts of the set-up of searches, which run as plain Python unless a program calls `compile_search_setup()`.

Numba looks up the functions that compiled code calls when compiling it, so compiled code may call NumPy and compiled functions, but not plain Python functions of qLDPC.
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
