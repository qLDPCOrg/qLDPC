# Exact distances and logical-operator counts: a guide for contributors

This directory is the package `qldpc.codes.distance`.
It computes exact distances of binary codes, counts the logical operators of a given weight in binary CSS codes, and finds light Pauli errors with given syndromes on stabilizer states.
Users reach it through the public functions and classes below, which take plain arrays, and through the `get_distance_exact` methods of the code classes in `src/qldpc/codes/common.py`, which call the distance functions for binary codes.
`problems.md` explains how the package turns the problems below into searches, and how the MacWilliams method counts.
`brouwer_zimmermann.md` and `connected_cluster.md` explain the Brouwer-Zimmermann and connected-cluster searches, map them to the source code, explain their heuristics, and list the requirements that every change to them must keep; for the Brouwer-Zimmermann search, the last three are in `brouwer_zimmermann_code.md`.

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
- Brute force on threads: it serves small codes and tests, and `auto` takes it only for at most `auto_brute_force_max_words` packed words, about a millisecond.
- Not yet: counts, single coset queries, and the MacWilliams method on threads.

Where the code, guide, and tests of each supported cell live:

| Cell | Code | Guide | Tests |
| --- | --- | --- | --- |
| Distance, `brute_force` | `_brute_force.py` | Section 3.6 of `brouwer_zimmermann.md` | `_brute_force_test.py` |
| Distance, `brouwer_zimmermann` | `_brouwer_zimmermann/distance.py` | Sections 2 and 3 of `brouwer_zimmermann.md`, and `brouwer_zimmermann_code.md` | `_brouwer_zimmermann/distance_test.py` |
| Distance, `connected_cluster` | `_connected_cluster/search.py` | Sections 3.1 to 3.7, 3.9, 3.11, 4, and 5 of `connected_cluster.md` | `_connected_cluster/search_test.py` |
| Distance, `auto` | `distances.py` and `_auto.py` | [Choosing a method](problems.md#choosing-a-method) in `problems.md` | `distances_test.py` and `_auto_test.py` |
| Coset weight of a vector, every method | `_get_coset_weight` of `distances.py` | For `connected_cluster`, "The distance of a vector from a classical code" in Section 3.8 of `connected_cluster.md`.  For the others, the vector `v` is the only generator and the words of the code `C` are excluded, which is the problem of Section 2.1 of `brouwer_zimmermann.md`.  Over GF(2), the words spanned by `v` and `C` but outside `C` are the coset `v + C` | `distances_test.py` and `src/qldpc/codes/common_test.py` |
| Coset leaders, `brouwer_zimmermann` and `auto` | `_brouwer_zimmermann/coset.py`, `_brouwer_zimmermann/coset_routes.py`, and `_brouwer_zimmermann/coset_tables.py` | [The coset-leader search](problems.md#the-coset-leader-search) in `problems.md`, Sections 3.9 and 3.10 of `brouwer_zimmermann.md`, and Section 4.5 of `brouwer_zimmermann_code.md` | `_brouwer_zimmermann/coset_test.py` and `coset_leaders_test.py` |
| Coset leaders, `connected_cluster` | `_connected_cluster/coset.py` | [The coset-leader search](problems.md#the-coset-leader-search) in `problems.md`, and Sections 3.8, 3.11, 4, and 5 of `connected_cluster.md` | `_connected_cluster/coset_test.py` and `coset_leaders_test.py` |
| Logical counts, `brute_force`, `brouwer_zimmermann`, and `auto` | `logical_counts.py` and `_counting/scan.py` | Section 3.11 of `brouwer_zimmermann.md` | `_counting/scan_test.py` and `logical_counts_test.py` |
| Logical counts, `macwilliams` | `logical_counts.py` and `_counting/macwilliams.py` | [The MacWilliams method](problems.md#the-macwilliams-method) in `problems.md` | `_counting/macwilliams_test.py` and `logical_counts_test.py` |

`_grid_test.py` checks the first table against the methods that each problem takes, and runs every supported cell, in each of its modes, on small instances against brute force.
A new problem, method, or mode adds its cells to both tables, and to `_grid_test.py`.

## Public names

`__init__.py` re-exports these names, and everything else in this directory is private:

- `get_distance_from_generators`, `get_distance_from_checks`, `get_distance_css`, `get_distance_pauli`, `DistanceMethod`, and `NoDistance`, whose only member `NoDistance.NAN` is also exported as `NO_DISTANCE`, from `distances.py`.
- `CSSLogicalCounter`, `CSSLogicalCounts`, `count_logical_ops_css`, `LogicalCountMethod`, and `compile_search_setup`, from `logical_counts.py`.
- `CosetLeaderSearch` and `CosetMethod`, from `coset_leaders.py`, and `MinWeightBound`, its bound for one syndrome, from `min_weight_bounds.py`.

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
- **Certificate** (`Certificate`): a lower bound on the weight of the words that a Brouwer-Zimmermann search has neither visited nor skipped, which tells the search when it can stop; with completion tables, or for the images of an information set under automorphisms, it bounds only the words lighter than the best weight `U` of the search (Sections 3.7 and 3.9 of `brouwer_zimmermann.md`).
- **Connected-cluster search** (`SupportSearch`): the depth-first search that `method="connected_cluster"` selects, which grows supports through the unsatisfied rows of a check matrix `H` to find light binary or symplectic words outside the row space of an excluded matrix `G`, or light vectors with a given syndrome.
  `"auto"` runs it for the distances of quantum codes and the fault distances of circuits, unless enumerating is cheap (see [Choosing a method](problems.md#choosing-a-method) in `problems.md`).
- **Automorphism:** in this package, a permutation of the coordinates of a search that maps the words it searches, and those it excludes, onto themselves.
  The public functions take only permutation automorphisms of a code, which permute qudits and apply no single-qudit gates, and lift them to the coordinates of a Pauli problem (Section 2.7 of `brouwer_zimmermann.md`).
  An automorphism of a quantum code can also apply single-qudit Clifford gates (see `get_transversal_automorphism_group` in `qldpc.circuits`); exchanging the coordinates of one qubit in a Pauli problem applies a single-qubit Clifford gate, but qLDPC never supplies such permutations.
- **Scan:** the enumeration of the combinations of rows of each information set, layer by layer, a layer holding the combinations of a given number of rows.
  Counts have their own scan, in `_counting/scan.py`, and the Brouwer-Zimmermann distance and coset searches (see [The coset-leader search](problems.md#the-coset-leader-search) in `problems.md`) use the compiled scans of `_brouwer_zimmermann/affine_search.py`.
  Every scan steps through a layer with `advance_support` in `_information_sets/support_stepping.py`.

## Layout

- `distances.py` and `logical_counts.py` are the public interfaces of the distances and the counts.
- `coset_leaders.py` is the public interface of the coset-leader search, and `min_weight_bounds.py` defines `MinWeightBound`, which its coset searches build too.
- `_auto.py` holds the parts of the automatic choice of a method, `method="auto"`, that compare the costs of the methods (see [Choosing a method](problems.md#choosing-a-method) in `problems.md`).
- `_brouwer_zimmermann/` holds the Brouwer-Zimmermann searches, which scan information sets, and `_connected_cluster/` the connected-cluster searches, which grow connected supports.
  Each holds the distance search of its method, and the coset searches with which the coset-leader search answers queries by that method, with their batch queries.
  `_brute_force.py` holds exhaustive enumeration.
- `_counting/` holds the word scan of the counts, the local Clifford gates of deformed counters, and the MacWilliams method.
- `_information_sets/` holds the families of information sets and their certificates, which the Brouwer-Zimmermann searches and the counts share.
- `_cosets.py` holds the parts that the coset searches of both methods share.
- `_sector_maps.py` finds the qubit permutations that let a distance search of a CSS code search one sector (see [Searching one sector of a CSS code](problems.md#searching-one-sector-of-a-css-code) in `problems.md`).
- `_tuning.py` holds the settings of the searches.
- `_threads.py` runs the work items of a search, or the blocks of a batch of queries, on several threads, and yields the outputs of work items in order on the calling thread, for the fault lister of `src/qldpc/circuits/certification/`.
- `_code_matrices.py` holds the binary input checks, nested bases, the stabilizers of subsystem codes, and the symplectic-to-Hamming map.
  The code constructors in `src/qldpc/codes/common.py` also use its `infer_field`, which finds the field of rows given in lists or tuples.

## Where to look

| To change… | Look in (this directory) |
| --- | --- |
| A method, or the validation of method names | the method type of each problem and its validator: `DistanceMethod` in `distances.py`, `CosetMethod` in `coset_leaders.py`, and `LogicalCountMethod` in `logical_counts.py`; the early checks of `src/qldpc/circuits/certification/fault_propagation.py` and `src/qldpc/circuits/certification/fault_distance.py`; where `logical_counts.py`, `_get_coset_weight` of `distances.py`, and `get_distance_exact` in `src/qldpc/codes/common.py` dispatch on the method name, and where `get_distance` validates it; and the `brute_force` flag that `logical_counts.py` passes to `_counting/scan.py` |
| The automatic choice of a method (`method="auto"`): which searches take which method, and when a connected-cluster search switches to the BZ method | `distances.py`, where the distance functions dispatch on the method; `_auto.py`, which estimates and compares the costs; `_get_coset_search_method` of `coset_leaders.py`, where coset searches take BZ; and `_get_logical_count_method` of `logical_counts.py`, where counts take BZ |
| The counter, its deformed counters, or `compile_search_setup` | `logical_counts.py` |
| The coset-leader search, which `src/qldpc/circuits/certification/` and `src/qldpc/experimental/circuits/benchmarking/residual_weights.py` use, and how its queries split into coset searches | `coset_leaders.py` |
| The Brouwer-Zimmermann (BZ) distance search | `_brouwer_zimmermann/distance.py` |
| The compiled scans of layers that the BZ distance search and the BZ coset searches share, one body specialized by constant flags | `_brouwer_zimmermann/affine_search.py` |
| The instances of BZ distance searches and BZ coset searches: their packed information sets, and the certificate of these sets | `_brouwer_zimmermann/affine_instance.py` |
| The BZ coset searches of the coset-leader search | `_brouwer_zimmermann/coset.py`, which holds names such as `BrouwerZimmermannCosetSearch`, `_search_coset`, and `_find_min_weights` |
| The tables of BZ coset searches: the bounds of their steps and their stopping steps, and the tails and completion tables of their routes by tails | `_brouwer_zimmermann/coset_tables.py`, which holds names such as `get_search_tables`, `get_stopping_step`, and `CosetTails` |
| The routes of BZ coset searches: the family of information sets for each maximum weight, the gate that chooses it, and the routes by tails | `_brouwer_zimmermann/coset_routes.py` |
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

1. `coset_leaders.py`, and `logical_counts.py`, which takes `NoDistance` from `distances.py`.
2. `distances.py`.
3. `_auto.py`, the automatic choice of a method.
4. The methods `_brouwer_zimmermann/`, `_connected_cluster/`, `_counting/`, and `_brute_force.py`.
5. `_information_sets/`, `_cosets.py`, and `min_weight_bounds.py`, which the methods build on, and `_sector_maps.py`, which `distances.py` uses.
   The connected-cluster search uses `infer_weight_divisor`, `verify_automorphisms`, and `get_orbits` of `_information_sets/`.
6. `_tuning.py`.
7. `_code_matrices.py` and `_threads.py`.

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
  The seeds make results reproducible rather than tune anything, though an early exit of a BZ search on several threads can still depend on the timing of its threads.
  A speed-only value that users may plausibly want to change to run faster is a field.
  Other speed-only constants, whose best value does not depend materially on the user, the code, or the machine, stay in their modules:
  - `ITEMS_PER_THREAD` and `BLOCKS_PER_THREAD` in `_threads.py`, which only balance threads at the end of the work
  - `_MAX_ITEMS` and `_MAX_FIXED_WORDS` in `_connected_cluster/search.py`, which bound the items of a frontier and the compiled versions of its search
  - `_MIN_FOUR_RUSSIANS_SIZE` in `src/qldpc/_binary.py`, the crossover of a kernel that searches call only in their setup
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
With either method, the batch queries of `CosetLeaderSearch`, `find_first_not_within_max_weight` and `get_min_weights`, move the rest of a costly batch to `num_threads` threads ([The coset-leader search](problems.md#the-coset-leader-search) in `problems.md`).
A `CosetLeaderSearch` fixes its number of threads when it is built.
By default, `num_threads` is `None`, which takes the value of the environment variable `QLDPC_NUM_THREADS` when qLDPC was imported, if it was set, and otherwise `numba.config.NUMBA_NUM_THREADS`.
`QLDPC_NUM_THREADS` is the recommended control, and must be a positive integer.
Numba never reads it.
Numba's parallel code, such as some of galois's, uses numba's own count, which numba normally reads from the environment variable `NUMBA_NUM_THREADS` when it is imported, or else from the CPUs available to the process.
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
