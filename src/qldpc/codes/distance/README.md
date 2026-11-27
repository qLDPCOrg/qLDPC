# Exact distances and logical-operator counts: a guide for contributors

This directory is the package `qldpc.codes.distance`.
It computes exact distances of binary codes, counts the logical operators of a given weight in binary CSS codes, and finds light Pauli errors with given syndromes on stabilizer states.
Users reach it through its public functions and classes, which take plain arrays ([Public names](#public-names)).
Users also reach it through the `get_distance_exact` methods of the code classes in `src/qldpc/codes/common.py`.
These methods call the distance functions for binary codes.
`problems.md` explains how the package turns the problems below into searches, and how the MacWilliams method counts.
`brouwer_zimmermann.md` and `connected_cluster.md` explain the Brouwer-Zimmermann and connected-cluster searches, map them to the source code, explain their heuristics, and list the requirements that every change to them must keep.
For the Brouwer-Zimmermann search, `brouwer_zimmermann_code.md` holds the map, the heuristics, and the requirements.

## The grid

The package solves a few problems, each by several methods, on one thread or several.
The `method` argument of its functions and classes names the method.
Each cell of the table below says whether a problem takes a method, and how it runs:

- **threads:** the costly work runs on `num_threads` threads (see [Threads](#threads)); for coset leaders, only batch queries do, and a single query runs on one thread;
- **one thread:** on one thread, whatever `num_threads` is, except as the notes below say;
- **not yet:** not supported, but it could be;
- **not by design:** not supported, for the reason below the table.

| Problem | `brute_force` | `brouwer_zimmermann` | `connected_cluster` | `macwilliams` | `auto` |
| --- | --- | --- | --- | --- | --- |
| Distance | one thread | threads | threads | not by design | threads |
| Coset weight of a vector | one thread | threads | threads | not by design | threads |
| Coset leaders | not by design | threads | threads | not by design | threads |
| Logical counts | one thread | one thread | not by design | one thread | one thread |

- **Distance** is the minimum weight of a nontrivial word: the functions of `distances.py`, the `get_distance_exact` methods of the code classes, and `get_fault_distance` in `src/qldpc/circuits/certification/fault_distance.py`.
  `connected_cluster` needs parity checks, so `get_distance_from_generators` does not take it.
- **Coset weight of a vector** is the minimum distance between a vector and a word of a binary classical code, which `ClassicalCode.get_distance_exact(vector=...)` finds by a distance search, through `_get_coset_weight` of `distances.py`.
- **Coset leaders** are the lightest Pauli errors with a given syndrome on a stabilizer state: `CosetLeaderSearch` in `coset_leaders.py`.
- **Logical counts** are the numbers of logical operators of a given weight in a CSS code: `CSSLogicalCounter` in `logical_counts.py`.
  A count by `brouwer_zimmermann` or `auto` that finds its default weight runs a distance search, which can use threads.

The cells that are empty, or on one thread only:

- Coset leaders by brute force: tests check the coset searches against the exhaustive minima of `get_exhaustive_minima` in `conftest.py`, and a library method would serve only states of a dozen qubits.
- Counts by `connected_cluster`: the search finds one word of each weight limit and prunes the others, rather than enumerating them, and a count above the distance must also count words whose supports the checks do not connect.
- `macwilliams` for distances and coset problems: it needs a CSS stabilizer code whose X and Z checks have rank at most 63, and has no form for cosets; counts already give the distance that it finds.
- Brute force on threads: it serves small codes and tests, and `auto` takes it only for at most `auto_brute_force_max_words` packed 64-bit integers, about a millisecond.
- Not yet: counts, single coset queries, and the MacWilliams method on threads.

Where the code, guide, and tests of each supported cell live:

| Cell | Code | Guide | Tests |
| --- | --- | --- | --- |
| Distance, `brute_force` | `_brute_force.py` | [Work estimates and exhaustive enumeration](brouwer_zimmermann.md#sec-work-estimates) in `brouwer_zimmermann.md` | `_brute_force_test.py` |
| Distance, `brouwer_zimmermann` | `_brouwer_zimmermann/distance.py` | [The Brouwer-Zimmermann algorithm](brouwer_zimmermann.md#sec-algorithm) and [Extensions and optimizations](brouwer_zimmermann.md#sec-extensions) in `brouwer_zimmermann.md`, and `brouwer_zimmermann_code.md` | `_brouwer_zimmermann/distance_test.py` |
| Distance, `connected_cluster` | `_connected_cluster/search.py` | Sections 3.1 to 3.7, 3.9, 3.11, 4, and 5 of `connected_cluster.md` | `_connected_cluster/search_test.py` |
| Distance, `auto` | `distances.py` and `_auto.py` | [Choosing a method](problems.md#choosing-a-method) in `problems.md` | `distances_test.py` and `_auto_test.py` |
| Coset weight of a vector, every method | `_get_coset_weight` of `distances.py` | For `connected_cluster`, "The distance of a vector from a classical code" in Section 3.8 of `connected_cluster.md`.  For the others, the vector `v` is the only generator and the words of the code `C` are excluded, which is the problem that [The problem](brouwer_zimmermann.md#sec-problem) in `brouwer_zimmermann.md` states.  Over GF(2), the words spanned by `v` and `C` but outside `C` are the coset `v + C` | `distances_test.py` and `src/qldpc/codes/common_test.py` |
| Coset leaders, `brouwer_zimmermann` and `auto` | `_brouwer_zimmermann/coset.py`, `_brouwer_zimmermann/coset_routes.py`, and `_brouwer_zimmermann/coset_tables.py` | [The coset-leader search](problems.md#the-coset-leader-search) in `problems.md`, [Families for coset searches](brouwer_zimmermann.md#sec-coset-families), [Rounding the certificate to a proven weight multiple](brouwer_zimmermann.md#sec-weight-multiples), [Skipping completions by completion tables](brouwer_zimmermann.md#sec-completion-tables), and [Minimum weights of cosets](brouwer_zimmermann.md#sec-coset-min-weights) in `brouwer_zimmermann.md`, and [Coset searches](brouwer_zimmermann_code.md#sec-coset-searches), [Heuristics and cost models](brouwer_zimmermann_code.md#sec-heuristics), [Behavior spanning files](brouwer_zimmermann_code.md#sec-cross-file-behavior), and [Correctness requirements](brouwer_zimmermann_code.md#sec-requirements) in `brouwer_zimmermann_code.md` | `_brouwer_zimmermann/coset_test.py` and `coset_leaders_test.py` |
| Coset leaders, `connected_cluster` | `_connected_cluster/coset.py` | [The coset-leader search](problems.md#the-coset-leader-search) in `problems.md`, and Sections 3.8, 3.11, 4, and 5 of `connected_cluster.md` | `_connected_cluster/coset_test.py` and `coset_leaders_test.py` |
| Logical counts, `brute_force`, `brouwer_zimmermann`, and `auto` | `logical_counts.py` and `_counting/scan.py` | [Counting logical operators](brouwer_zimmermann.md#sec-counting) in `brouwer_zimmermann.md` | `_counting/scan_test.py` and `logical_counts_test.py` |
| Logical counts, `macwilliams` | `logical_counts.py` and `_counting/macwilliams.py` | [The MacWilliams method](problems.md#the-macwilliams-method) in `problems.md` | `_counting/macwilliams_test.py` and `logical_counts_test.py` |

`_grid_test.py` checks the first table against the methods that each problem takes, and runs every supported cell, in each of its modes, on small instances against brute force.
A new problem, method, or mode adds its cells to both tables, and to `_grid_test.py`.

## Public names

`__init__.py` re-exports the names that its `__all__` lists, from `distances.py`, `logical_counts.py`, `coset_leaders.py`, and `min_weight_bounds.py`.
Everything else in this directory is private.

`qldpc.codes` re-exports only `NoDistance`, `NO_DISTANCE`, and `DistanceMethod`, which the code classes take or return.
The documentation cites the other names at `qldpc.codes.distance`.

## Terms

- **Sector:** a space of words that a search or count examines separately for nontrivial words.
  For example, the X sector of a CSS stabilizer code holds the X-type operators that commute with the Z stabilizers, and the trivial words of this sector are the X stabilizers.
  [Searching one sector of a CSS code](problems.md#searching-one-sector-of-a-css-code) in `problems.md` gives the sectors of a CSS subsystem code.
  A search of a classical code, of a qubit code that is not CSS, or of a circuit examines one sector.
- **Search basis** and **information set:** a search basis is a basis that a Brouwer-Zimmermann search or count scans.
  It spans a sector, or, in a coset search, the binary code whose cosets the search examines ([Families for coset searches](brouwer_zimmermann.md#sec-coset-families) in `brouwer_zimmermann.md`).
  It is row reduced on a block of columns.
  If the block has rank `r_i`, the first `r_i` rows of the basis are the identity on `r_i` pivot columns of the block, and the other rows are zero on the block.
  The block is an information set if its rank is the dimension of the basis, and rank-deficient otherwise ([Systematic bases and information sets](brouwer_zimmermann.md#sec-information-sets) and [Search bases and rank-deficient blocks](brouwer_zimmermann.md#sec-search-bases) in `brouwer_zimmermann.md`).
  Code comments and `brouwer_zimmermann_code.md` call a search basis a set for short, a full set if its block is an information set and a partial set otherwise.
  Identifiers such as `_information_sets/` and `max_information_sets`, and the phrase "family of information sets", say "information sets" but also mean the search bases row reduced on them.
- **Basis family** (`BasisFamily`): the search bases that one Brouwer-Zimmermann search or count uses together, stacked into arrays.
- **Certificate** (`Certificate`): a lower bound on the weight of the words that a Brouwer-Zimmermann search has neither examined nor skipped ([The stopping certificate](brouwer_zimmermann.md#sec-stopping-certificate) and [Overlapping search bases](brouwer_zimmermann.md#sec-overlapping-bases) in `brouwer_zimmermann.md`).
  The search can stop once the certificate reaches `U`, the weight of the lightest word that it has found.
  With completion tables, or with the images of an information set under automorphisms, the certificate bounds only the words lighter than `U` ([The certificate of an automorphism family](brouwer_zimmermann.md#sec-automorphism-certificate) and [Certificates and contracts](brouwer_zimmermann.md#sec-completion-contracts) in `brouwer_zimmermann.md`).
- **Connected-cluster search** (`SupportSearch`): the depth-first search that `method="connected_cluster"` selects.
  It grows supports through the unsatisfied rows of a check matrix `H`.
  It finds light binary or symplectic words outside the row space of an excluded matrix `G`, or light vectors with a given syndrome.
  Unless enumerating is cheap, `"auto"` runs it for most distance searches of quantum codes and circuits (steps 2 and 3 of [Choosing a method](problems.md#choosing-a-method) in `problems.md`).
- **Automorphism:** in this package, a permutation of the coordinates of a search that maps the words it searches, and those it excludes, onto themselves.
  The public functions take only permutation automorphisms of a code, which permute qudits and apply no single-qudit gates, and lift them to the coordinates of a Pauli problem ([Pauli weight as Hamming weight](brouwer_zimmermann.md#sec-pauli-weight) in `brouwer_zimmermann.md`).
  An automorphism of a quantum code can also apply single-qudit Clifford gates (see `get_transversal_automorphism_group` in `qldpc.circuits`).
  In a Pauli problem, exchanging the coordinates of one qubit applies a single-qubit Clifford gate, but qLDPC never supplies such permutations.
- **Scan:** the enumeration of the combinations of rows of each search basis, layer by layer, a layer holding the combinations of a given number of rows.
  A scan names each combination by its **support**, the set of its row indices ([Packed enumeration with pivot-column deletion](brouwer_zimmermann.md#sec-packed-enumeration) in `brouwer_zimmermann.md`); elsewhere, as in a connected-cluster search, the support of a word is the set of its nonzero coordinates.
  Counts have their own scan, in `_counting/scan.py`, and the Brouwer-Zimmermann distance and coset searches (see [The coset-leader search](problems.md#the-coset-leader-search) in `problems.md`) use the compiled scans of `_brouwer_zimmermann/affine_search.py`.
  Every scan steps through a layer with `advance_support` in `_information_sets/support_stepping.py`.
- **Search by Hamming weight:** a Brouwer-Zimmermann coset search of `CosetLeaderSearch` that weighs its binary words by their Hamming weight, up to a fixed factor.
  The searches by physical weight are searches by Hamming weight, since they search the images of Pauli errors under the map `T` of [Pauli weight as Hamming weight](brouwer_zimmermann.md#sec-pauli-weight) in `brouwer_zimmermann.md`.
  The Hamming weight of the image of an error is twice the physical weight of the error.
  The searches of X parts and of Z parts, into which `CosetLeaderSearch` splits a query by CSS weight on a CSS stabilizer state, are searches by Hamming weight too ([The coset-leader search](problems.md#the-coset-leader-search) in `problems.md`).
  A search by CSS weight on a stabilizer state that is not CSS weighs a Pauli error by `max(|x|, |z|)`, where `x` and `z` are its X and Z parts, so it is not a search by Hamming weight.
  Of the coset searches, only the searches by Hamming weight can weigh words by their tails and skip completions by completion tables ([Coset searches](brouwer_zimmermann_code.md#sec-coset-searches) in `brouwer_zimmermann_code.md`).

## Layout

- `distances.py` and `logical_counts.py` are the public interfaces of the distances and the counts.
- `coset_leaders.py` is the public interface of the coset-leader search.
- `min_weight_bounds.py` defines `MinWeightBound`, the answer of a coset-leader query for one syndrome.
- `_auto.py` holds the parts of the automatic choice of a method, `method="auto"`, that compare the costs of the methods (see [Choosing a method](problems.md#choosing-a-method) in `problems.md`).
- `_brouwer_zimmermann/` and `_connected_cluster/` hold the Brouwer-Zimmermann and connected-cluster searches.
  Each holds the distance search of its method, and the coset searches that answer the single and batch queries of the coset-leader search by that method.
  `_brute_force.py` holds exhaustive enumeration.
- `_counting/` holds the word scan of the counts, the local Clifford gates of deformed counters, and the MacWilliams method.
- `_information_sets/` holds what the scans of search bases share: basis families and their certificates, automorphism families, completion tables, and the step from one support of a layer to the next.
  The Brouwer-Zimmermann searches and the counts use it, and the connected-cluster search a few of its functions (see [How the modules depend on each other](#how-the-modules-depend-on-each-other)).
- `_cosets.py` holds the parts that the coset searches of both methods share.
  `coset_leaders.py` and `_connected_cluster/sparse_checks.py` also use them.
- `_sector_maps.py` finds the qubit permutations that let a distance search of a CSS code search one sector (see [Searching one sector of a CSS code](problems.md#searching-one-sector-of-a-css-code) in `problems.md`).
- `_tuning.py` holds the settings of the searches.
- `_threads.py` runs the work items of a search, or the blocks of a batch of queries, on several threads.
  For the fault lister of `src/qldpc/circuits/certification/`, it also yields the outputs of work items in order on the calling thread.
- `_code_matrices.py` holds the binary input checks, nested bases, the stabilizers of subsystem codes, and the symplectic-to-Hamming map.
  Modules elsewhere in qLDPC use some of its helpers, such as `infer_field`, with which the code constructors in `src/qldpc/codes/common.py` find the field of rows given in lists or tuples.

## Where to look

| To change… | Look in (this directory) |
| --- | --- |
| A method, or the validation of method names | the method type of each problem and its validator: `DistanceMethod` in `distances.py`, `CosetMethod` in `coset_leaders.py`, and `LogicalCountMethod` in `logical_counts.py`; the early checks of `src/qldpc/circuits/certification/fault_propagation.py` and `src/qldpc/circuits/certification/fault_distance.py`; where `logical_counts.py`, `_get_coset_weight` of `distances.py`, and `get_distance_exact` in `src/qldpc/codes/common.py` dispatch on the method name, and where `get_distance` validates it; and the `brute_force` flag that `logical_counts.py` passes to `_counting/scan.py` |
| The automatic choice of a method (`method="auto"`): which searches take which method, and when a connected-cluster search switches to the BZ method | `distances.py`, where the distance functions dispatch on the method; `_auto.py`, which estimates and compares the costs; `_get_coset_search_method` of `coset_leaders.py`, where coset searches take BZ; and `_get_logical_count_method` of `logical_counts.py`, where counts take BZ |
| The counter, its deformed counters, or `compile_search_setup` | `logical_counts.py` |
| The coset-leader search, which `src/qldpc/circuits/certification/` and `src/qldpc/experimental/circuits/benchmarking/residual_weights.py` use, and how its queries split into coset searches | `coset_leaders.py` |
| The Brouwer-Zimmermann (BZ) distance search | `_brouwer_zimmermann/distance.py` |
| The compiled scans of layers that the BZ distance search and the BZ coset searches share, one body specialized by constant flags | `_brouwer_zimmermann/affine_search.py` |
| The instances of BZ distance searches and BZ coset searches: their packed search bases, and the certificate of these bases | `_brouwer_zimmermann/affine_instance.py` |
| The BZ coset searches of the coset-leader search | `_brouwer_zimmermann/coset.py`, which holds names such as `BrouwerZimmermannCosetSearch`, `_search_coset`, and `_find_min_weights` |
| The tables of BZ coset searches: the bounds of their steps and their stopping steps, and the tails and completion tables of their routes by tails | `_brouwer_zimmermann/coset_tables.py`, which holds names such as `get_search_tables`, `get_stopping_step`, and `CosetTails` |
| The routes of BZ coset searches: which basis family the queries of each maximum weight take, and whether they weigh words by their tails | `_brouwer_zimmermann/coset_routes.py` |
| The connected-cluster search, of distances and of vectors with a given syndrome | `_connected_cluster/search.py` |
| The connected-cluster coset searches of the coset-leader search, and their batch queries | `_connected_cluster/coset.py` |
| How the connected-cluster searches make their checks sparser with the same span, and find the light sums of pairs of checks and of known words that distance searches add | `_connected_cluster/sparse_checks.py` |
| Exhaustive enumeration, which the `brute_force` method runs, with its packed symplectic weights | `_brute_force.py` |
| The scan of a count, or how its basis families follow local gates | `_counting/scan.py` |
| The local Clifford gates of deformed counters | `_counting/local_cliffords.py` |
| The MacWilliams counting method | `_counting/macwilliams.py` |
| The bound for one syndrome that the coset-leader search and its coset searches give (`MinWeightBound`) | `min_weight_bounds.py` |
| Which sectors a distance search of a CSS code searches, and the qubit permutations that map one sector onto the other | `_search_css` and `_search` of `distances.py`, and `_sector_maps.py`, which calls `get_matrix_isomorphism` of `src/qldpc/external/_bliss.py` |
| The parts that the coset searches of both methods share: words with unit syndromes, sums of the rows that a support selects, and tables of the errors of weight zero or one | `_cosets.py` |
| How basis families are built | `_information_sets/basis_families.py` |
| Certificates, and how searches plan their steps: where they stop, and the work of the steps up to there | `_information_sets/certificates.py` |
| The step from one support of a layer to the next, which every scan of layers shares | `_information_sets/support_stepping.py` |
| Automorphism families, the images of one information set under automorphisms of a code; and the check of those automorphisms and the orbits that they generate, which the connected-cluster search uses too | `_information_sets/automorphism_families.py` |
| Completion tables, which let a BZ scan skip the supports whose completions by two more rows are too heavy, and the choice of whether to build them | `_information_sets/completion_bounds.py` |
| Information sets with bounded column coverage, by matroid partitioning | `_information_sets/matroid_partition.py` |
| The settings that a search reads when it starts (`SearchTuning` and `use_tuning`), and the constants that size caches when qLDPC is imported | `_tuning.py` |
| How a search runs its work items on threads (`run_work_items`), or until the first item that ends its work (`run_items_until_first`); how a batch of queries splits into blocks on threads (`run_blocks`); how the outputs of work items come back in order (`iterate_in_order`); and the default number of threads | `_threads.py`; how the BZ distance search splits a layer into work items is in `_brouwer_zimmermann/distance.py`, how the connected-cluster distance search splits a weight limit, in `_connected_cluster/search.py`, how BZ coset queries search their blocks, in `_brouwer_zimmermann/coset.py`, and how connected-cluster coset queries do, in `_connected_cluster/coset.py` |
| Binary input checks, nested bases, the stabilizers that `get_css_stabilizers` and `get_pauli_stabilizers` derive from gauge generators, or the map of symplectic vectors to binary vectors of twice their weight | `_code_matrices.py` |
| Bit packing, bit counts, row reduction, or binary products over GF(2), packed or not, which other subpackages share | `src/qldpc/_binary.py` |
| How numba code is compiled or cached | `src/qldpc/_compilation.py` |

## How the modules depend on each other

From the top:

1. `coset_leaders.py`, and `logical_counts.py`, which imports `NO_DISTANCE` and `NoDistance` from `distances.py`.
2. `distances.py`.
3. `_auto.py`, the automatic choice of a method.
4. The methods `_brouwer_zimmermann/`, `_connected_cluster/`, `_counting/`, and `_brute_force.py`.
5. `_information_sets/`, `_cosets.py`, and `min_weight_bounds.py`, which the methods build on, and `_sector_maps.py`, which `distances.py` uses.
   The connected-cluster search uses `infer_weight_divisor`, `verify_automorphisms`, and `get_orbits` of `_information_sets/`.
6. `_tuning.py`.
7. `_code_matrices.py` and `_threads.py`.

A top-level module or subpackage imports only from lower levels.
A module in a subpackage may also import from that subpackage.
`_layering_test.py` checks this with `check_import_layering` of `src/qldpc/conftest.py`.
The docstring of `check_import_layering` gives further rules, such as that no module imports the root of the package.
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

## Tuning

A value that a search reads when it starts, and that changes how long the search takes or how much memory it uses but no result that it guarantees, is a field of `SearchTuning` in `_tuning.py`, unless its best value does not depend materially on the user, the code, or the machine (below).
Such values include a threshold that chooses a route, a limit, a coefficient of a cost model, a batch size, and a switch that turns a step off.
They are not arguments of public functions, whose arguments say what to compute and by which method, such as `cutoff`, `max_weight`, and `method`.

- A search reads the active tuning once, when it starts, and passes the values that it needs down as arguments, including to compiled code, so that one search never mixes two tunings.
  The distance functions of `distances.py` read it for a distance search, `CSSLogicalCounter.count_logical_ops` for a count, and `CosetLeaderSearch` when it is built.
  `BrouwerZimmermannSearch` and `ConnectedClusterSearch`, which `get_distance_brouwer_zimmermann` and `get_distance_connected_cluster` build, read it only if their callers pass none.
  A cache that outlives a search is keyed by the values that it depends on, as the fields of `_CosetTuning` in `_brouwer_zimmermann/coset.py` key the cached searches of `CosetLeaderSearch`.
- Only a value that qLDPC fixes before any search starts is a constant in `_tuning.py`, such as the size of a cache, which `functools.lru_cache` fixes at import.
  Code reads cache sizes as `_tuning.NAME`.
  Constants fixed by the algorithm or by native integer widths, such as `WORK_CAP`, and the seeds of random generators stay in the modules that use them, except `MAX_VALUE`, the largest value of an integer field.
  The seeds make results reproducible rather than tune anything.
  Speed-only constants whose best value does not depend materially on the user, the code, or the machine stay in their modules:
  - `ITEMS_PER_THREAD` and `BLOCKS_PER_THREAD` in `_threads.py`, which only balance threads at the end of the work
  - `RANGES_PER_THREAD` and `OUTPUTS_PER_THREAD` in `_threads.py`, which balance the threads of `iterate_in_order` and bound the outputs that they hold
  - `_MAX_ITEMS` and `_MAX_FIXED_WORDS` in `_connected_cluster/search.py`, which bound the items of a frontier and the compiled versions of its search
  - `_MIN_FOUR_RUSSIANS_SIZE` in `src/qldpc/_binary.py`, the crossover of a kernel that searches call only in their setup
  - `_NUM_RANDOM_WORDS` in `_sector_maps.py`, the size of a cheap prefilter
- Each field and constant has its own `#:` comment directly above it, for Sphinx to render.
  The first line of a comment has no colon, since Sphinx reads text before such a colon as a type.
  A field's comment says what the field controls, with its unit, its special values, and its range if its metadata narrows it.
  The comment also names the kind of the default and how the value was found, or says why the default was chosen by hand, as the "Retuning" section of the module docstring requires.
  Where a guide section or a cost model explains the field's heuristic, the comment points there.
  A constant's comment says what it controls; the "Retuning" section says how the sizes of caches were chosen.
  Comments, docstrings, and guides in this package follow the [writing standard's rule on measurements](../../../../docs/source/contributing/standards/writing.rst#where-documentation-belongs).
- Change a default only as the "Retuning" section allows; [experiments/distance_tuning](../../../../experiments/distance_tuning/README.md) says how to check one.
- [Exact-search performance and tuning](../../../../docs/source/exact_search_tuning.rst) renders the module's values from these comments, after hand-written tables that lead from symptoms and routes to the values.
  The tables are selective; add a value to them when it is how a reader would fix a symptom or control a route.
  `_tuning_test.py` checks that each value has its own comment.
  The documentation build fails on a broken reference into `_tuning`.
- The module docstring of `_tuning.py` says which results a tuning can change.
- To force a route or exercise a value in a test, call the `tune` fixture of `src/qldpc/codes/conftest.py`, as in `tune(overlapping_minimum_work=0)`, rather than patching module attributes.
- If the implementation cannot safely support a field's full range, narrow it in the field's metadata, and in the `narrowed` table of `test_tunings_are_validated` in `_tuning_test.py`, rather than adding code for unsupported values.
  A field whose default is `None`, such as `num_threads`, also accepts `None`.

### Threads

A BZ distance search scans each costly layer of a search basis on `num_threads` threads, which share one copy of the completion tables of that basis ([Behavior spanning files](brouwer_zimmermann_code.md#sec-cross-file-behavior) in `brouwer_zimmermann_code.md`).
A connected-cluster distance search searches each costly weight limit on `num_threads` threads, which give the same witness and count of states as a search on one thread (Section 3.11 of `connected_cluster.md`).
With either method, the batch queries of `CosetLeaderSearch`, `find_first_not_within_max_weight` and `get_min_weights`, move the rest of a costly batch to `num_threads` threads ([The coset-leader search](problems.md#the-coset-leader-search) in `problems.md`).
A `CosetLeaderSearch` fixes its number of threads when it is built.
By default, `num_threads` is `None`, which takes the value of the environment variable `QLDPC_NUM_THREADS` when qLDPC was imported, if it was set, and otherwise `numba.config.NUMBA_NUM_THREADS`.
`QLDPC_NUM_THREADS` is the recommended control, and must be a positive integer.
Numba never reads it.
Numba's parallel code, such as some of galois's, uses numba's own count, which numba normally reads from the environment variable `NUMBA_NUM_THREADS` when it is imported, or else from the CPUs available to the process.
So `QLDPC_NUM_THREADS` sets numba's count too only if qLDPC is imported before numba, and so before galois: qLDPC then sets `NUMBA_NUM_THREADS` to the same value.
If numba was imported first with another count, qLDPC warns that numba keeps it; to avoid that, also set `NUMBA_NUM_THREADS` to the same value before Python starts.
The docstring of `src/qldpc/_num_threads.py` gives the other warnings.
`numba.set_num_threads` changes only the count of the calling thread's numba parallel code, not `QLDPC_NUM_THREADS` or `numba.config.NUMBA_NUM_THREADS`, nor the threads of searches.
Programs that run searches in several processes or threads at once should keep each search on one thread, since each search would otherwise start a thread for each CPU; the row on cores of [Exact-search performance and tuning](../../../../docs/source/exact_search_tuning.rst) says how, in pools of workers and in containers.
Neither `QLDPC_NUM_THREADS` nor `num_threads` limits BLAS threads, which follow the BLAS library's own variables, such as `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` (Accelerate on macOS), or `OMP_NUM_THREADS`.
Exhaustive enumeration, single coset queries, and counts run on one thread, except the BZ distance search with which a count may find its default weight.
[Working here](#working-here) gives the rules for code that threads run, and the tests that check them.

## Compiled code

Most searches spend their time in functions that numba compiles.
Two decorators in `src/qldpc/_compilation.py` mark them, and its docstring says when to use each:

- `@always_compiled` marks the per-word loops and their helpers.
  They are compiled on first use and cached on disk, in a directory named by a hash of qLDPC's source, so the first run after an edit anywhere in qLDPC compiles again.
  A small helper that hot loops call, such as `advance_support`, is marked `@always_compiled(inline=True)`, so that compiled callers include its code rather than pay for a call.
- `@compiled_on_request` marks parts of the set-up of searches, which run as plain Python unless a program calls `compile_search_setup()`.
  Their compiled code is cached in the same directory.

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
- A change to the scans or how they step, to pruning, to the work items, or to any code that threads run must keep the requirements on threads of the guide of each method ([Correctness requirements](brouwer_zimmermann_code.md#sec-requirements) in `brouwer_zimmermann_code.md`, or Section 5 of `connected_cluster.md`), and pass the tests on threads that cite those requirements or that `connected_cluster.md` names.
  So must a change to data that threads read, such as completion tables, the routes of coset searches, and the tables of a connected-cluster search, or to a decision that a search makes from the results of a layer or a weight limit.
  Code that threads run reads the tuning only from its search, never through `get_tuning()`, since new threads can start with the default tuning (see the module docstring of `_tuning.py`).
  Code that threads run also calls no numba-parallel code, such as some functions of galois, since that can abort the process (see `run_work_items` in `_threads.py`).
- While each test of this package runs, the fixture `check_code_on_threads` of `conftest.py` fails it if its threads read the tuning through `get_tuning()` or call numba-parallel code from Python.
  Review checks such calls on paths that tests do not run on threads.
  `_thread_rules_test.py` checks that the docstring of each entry point compiled with `nogil=True` cites the correctness requirements of the guide of its method, `brouwer_zimmermann_code.md` or `connected_cluster.md`.
  `test_nothing_asks_numba_for_parallel_code` of `src/qldpc/_compiled_code_rules_test.py` checks that no function of qLDPC is compiled with numba's `parallel` option.
  `test_always_compiled_functions_use_nothing_from_outside_qldpc` of the same file checks that the functions marked `@always_compiled` use no names imported from packages other than qLDPC, NumPy, and numba, such as functions of galois.
  Four tests on threads run with every array of their search read-only, through `check_shared_arrays_on_threads` or `shared_arrays_read_only` of `conftest.py`.
  Numba then rejects a write to such an array in any branch of compiled code.
  Two of these tests cite [Correctness requirements](brouwer_zimmermann_code.md#sec-requirements) in `brouwer_zimmermann_code.md`, and Section 5 of `connected_cluster.md` names the other two.
  Each guide lists the writes that its tests miss.
  Review checks those writes.
- `_readme_test.py` fails if "Where to look" omits a module, if "Layout" omits a top-level module or subpackage, or if this page or `problems.md` names, in a code span, a Python file of this package that does not exist.
