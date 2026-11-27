# Exact distances and logical-operator counts: a guide for contributors

This directory is the package `qldpc.codes.distance`.
It computes exact distances of binary codes, counts the logical operators of a given weight in binary CSS codes, and finds light Pauli errors with given syndromes on stabilizer states.
Users reach it through its public functions and classes, which take plain arrays ([Public names](#sec-public-names)).
Users also reach it through the `get_distance_exact` methods of the code classes in `src/qldpc/codes/common.py`.
These methods call the distance functions for binary codes.
`problems.md` explains how the package turns the problems below into searches, and how the MacWilliams method counts.
`brouwer_zimmermann.md` and `connected_cluster.md` explain the Brouwer-Zimmermann and connected-cluster searches, map them to the source code, explain their heuristics, and list the requirements that every change to them must keep.
For the Brouwer-Zimmermann search, `brouwer_zimmermann_code.md` holds the map, the heuristics, and the requirements.

<a name="sec-grid"></a>
## The grid

The package solves a few problems, each by several methods, on one thread or several.
The `method` argument of its functions and classes names the method.
Each cell of the table below says whether a problem takes a method, and how it runs:

- **threads:** the costly work runs on `num_threads` threads (see [Threads](#sec-threads)); for coset leaders, only batch queries do, and a single query runs on one thread;
- **one thread:** on one thread, whatever `num_threads` is, except as the notes below say;
- **not yet:** not supported, but it could be;
- **not by design:** not supported, for the reason below the table.

| Problem | `brute_force` | `brouwer_zimmermann` | `connected_cluster` | `macwilliams` | `auto` |
| --- | --- | --- | --- | --- | --- |
| Distance | one thread | threads | threads | not by design | threads |
| Coset weight of a vector | one thread | threads | threads | not by design | threads |
| Coset leaders | not by design | threads | threads | not by design | threads |
| Logical counts | one thread | one thread | not by design | one thread | one thread |

- **Distance** is the minimum weight of a nontrivial word: the functions of `distances.py`, and the code that calls them elsewhere in qLDPC, such as the `get_distance_exact` methods of the code classes.
  `connected_cluster` needs parity checks, so `get_distance_from_generators` does not take it.
- **Coset weight of a vector** is the minimum distance between a vector and a word of a binary classical code, which `ClassicalCode.get_distance_exact(vector=...)` finds by a distance search of `distances.py`.
- **Coset leaders** are the lightest Pauli errors with a given syndrome on a stabilizer state: `CosetLeaderSearch` in `coset_leaders.py`.
- **Logical counts** are the numbers of logical operators of a given weight in a CSS code: `CSSLogicalCounter` in `logical_counts.py`.
  A count by `brouwer_zimmermann` or `auto` that finds its default weight runs a distance search, which can use threads.

The cells that are empty, or on one thread only:

- Coset leaders by brute force: tests check the coset searches against exhaustive minima that `conftest.py` computes, and a library method would serve only states of a dozen qubits.
- Counts by `connected_cluster`: the search finds one word of each weight limit and prunes the others, rather than enumerating them, and a count above the distance must also count words whose supports the checks do not connect.
- `macwilliams` for distances and coset problems: it needs a CSS stabilizer code whose X and Z checks have rank at most 63, and has no form for cosets; counts already give the distance that it finds.
- Brute force on threads: it serves small codes and tests, and `auto` takes it only for at most `auto_brute_force_max_words` packed 64-bit integers, about a millisecond.
- Not yet: counts, single coset queries, and the MacWilliams method on threads.

Where the code, guide, and tests of each supported cell live:

| Cell | Code | Guide | Tests |
| --- | --- | --- | --- |
| Distance, `brute_force` | `_brute_force.py` | [Work estimates and exhaustive enumeration](brouwer_zimmermann.md#sec-work-estimates) in `brouwer_zimmermann.md` | `_brute_force_test.py` |
| Distance, `brouwer_zimmermann` | `_brouwer_zimmermann/distance.py` | [The Brouwer-Zimmermann algorithm](brouwer_zimmermann.md#sec-algorithm) and [Extensions and optimizations](brouwer_zimmermann.md#sec-extensions) in `brouwer_zimmermann.md`, and `brouwer_zimmermann_code.md` | `_brouwer_zimmermann/distance_test.py` |
| Distance, `connected_cluster` | `_connected_cluster/search.py` | [The problem](connected_cluster.md#sec-problem), [Light words are connected](connected_cluster.md#sec-connectivity), [The depth-first search](connected_cluster.md#sec-depth-first-search), [Pruning](connected_cluster.md#sec-pruning), [Symmetries](connected_cluster.md#sec-symmetries), [From a search to the distance](connected_cluster.md#sec-search-to-distance), [Qubit codes that are not CSS](connected_cluster.md#sec-non-css), [Sparser stabilizers of subsystem codes](connected_cluster.md#sec-sparser-stabilizers), [Threads](connected_cluster.md#sec-threads), [Guide to the source code](connected_cluster.md#sec-source-code), and [Correctness requirements](connected_cluster.md#sec-requirements) in `connected_cluster.md` | `_connected_cluster/search_test.py` |
| Distance, `auto` | `distances.py` and `_auto.py` | [Choosing a method](problems.md#sec-choosing-a-method) in `problems.md` | `distances_test.py` and `_auto_test.py` |
| Coset weight of a vector, every method | `distances.py` | For `connected_cluster`, "The distance of a vector from a classical code" in [Vectors with a given syndrome](connected_cluster.md#sec-syndromes) in `connected_cluster.md`.  For the others, the vector `v` is the only generator and the words of the code `C` are excluded, which is the problem that [The problem](brouwer_zimmermann.md#sec-problem) in `brouwer_zimmermann.md` states.  Over GF(2), the words spanned by `v` and `C` but outside `C` are the coset `v + C` | `distances_test.py` and `src/qldpc/codes/common_test.py` |
| Coset leaders, `brouwer_zimmermann` and `auto` | `_brouwer_zimmermann/coset.py`, `_brouwer_zimmermann/coset_routes.py`, and `_brouwer_zimmermann/coset_tables.py` | [The coset-leader search](problems.md#sec-coset-leader-search) in `problems.md`, [Families for coset searches](brouwer_zimmermann.md#sec-coset-families), [Rounding the certificate to a proven weight multiple](brouwer_zimmermann.md#sec-weight-multiples), [Skipping completions by completion tables](brouwer_zimmermann.md#sec-completion-tables), and [Minimum weights of cosets](brouwer_zimmermann.md#sec-coset-min-weights) in `brouwer_zimmermann.md`, and [Coset searches](brouwer_zimmermann_code.md#sec-coset-searches), [Heuristics and cost models](brouwer_zimmermann_code.md#sec-heuristics), [Behavior spanning files](brouwer_zimmermann_code.md#sec-cross-file-behavior), and [Correctness requirements](brouwer_zimmermann_code.md#sec-requirements) in `brouwer_zimmermann_code.md` | `_brouwer_zimmermann/coset_test.py` and `coset_leaders_test.py` |
| Coset leaders, `connected_cluster` | `_connected_cluster/coset.py` | [The coset-leader search](problems.md#sec-coset-leader-search) in `problems.md`, and [Vectors with a given syndrome](connected_cluster.md#sec-syndromes), [Threads](connected_cluster.md#sec-threads), [Guide to the source code](connected_cluster.md#sec-source-code), and [Correctness requirements](connected_cluster.md#sec-requirements) in `connected_cluster.md` | `_connected_cluster/coset_test.py` and `coset_leaders_test.py` |
| Logical counts, `brute_force`, `brouwer_zimmermann`, and `auto` | `logical_counts.py` and `_counting/scan.py` | [Counting logical operators](brouwer_zimmermann.md#sec-counting) in `brouwer_zimmermann.md` | `_counting/scan_test.py` and `logical_counts_test.py` |
| Logical counts, `macwilliams` | `logical_counts.py` and `_counting/macwilliams.py` | [The MacWilliams method](problems.md#sec-macwilliams) in `problems.md` | `_counting/macwilliams_test.py` and `logical_counts_test.py` |

`_grid_test.py` checks the first table against the methods that each problem takes, and runs every supported cell, in each of its modes, on small instances against brute force.
A new problem, method, or mode adds its cells to both tables, and to `_grid_test.py`.

<a name="sec-public-names"></a>
## Public names

`__init__.py` re-exports the names that its `__all__` lists, from `distances.py`, `logical_counts.py`, `coset_leaders.py`, and `min_weight_bounds.py`.
Everything else in this directory is private.

`qldpc.codes` re-exports only `NoDistance`, `NO_DISTANCE`, and `DistanceMethod`, which the code classes take or return.
The documentation cites the other names at `qldpc.codes.distance`.

<a name="sec-terms"></a>
## Terms

- **Sector:** a space of words that a search or count examines separately for nontrivial words.
  For example, the X sector of a CSS stabilizer code holds the X-type operators that commute with the Z stabilizers, and the trivial words of this sector are the X stabilizers.
  [Searching one sector of a CSS code](problems.md#sec-one-sector) in `problems.md` gives the sectors of a CSS subsystem code.
  A search of a classical code, of a qubit code that is not CSS, or of a circuit examines one sector.
- **Search basis** and **information set:** a search basis is a basis that a Brouwer-Zimmermann search or count scans.
  It spans a sector, or, in a coset search, the binary code whose cosets the search examines ([Families for coset searches](brouwer_zimmermann.md#sec-coset-families) in `brouwer_zimmermann.md`).
  It is row reduced on a block of columns.
  If the block has rank `r_i`, the first `r_i` rows of the basis are the identity on `r_i` pivot columns of the block, and the other rows are zero on the block.
  The block is an information set if its rank is the dimension of the basis, and rank-deficient otherwise ([Systematic bases and information sets](brouwer_zimmermann.md#sec-information-sets) and [Search bases and rank-deficient blocks](brouwer_zimmermann.md#sec-search-bases) in `brouwer_zimmermann.md`).
  Code comments and `brouwer_zimmermann_code.md` call a search basis a set for short, a full set if its block is an information set and a partial set otherwise.
  Identifiers such as `_information_sets/` and `max_information_sets`, and the phrase "family of information sets", say "information sets" but also mean the search bases row reduced on them.
- **Basis family:** the search bases that one Brouwer-Zimmermann search or count uses together, stacked into arrays (`_information_sets/basis_families.py`).
- **Certificate:** a lower bound on the weight of the words that a Brouwer-Zimmermann search has neither examined nor skipped ([The stopping certificate](brouwer_zimmermann.md#sec-stopping-certificate) and [Overlapping search bases](brouwer_zimmermann.md#sec-overlapping-bases) in `brouwer_zimmermann.md`; `_information_sets/certificates.py`).
  The search can stop once the certificate reaches `U`, the weight of the lightest word that it has found.
  With completion tables, or with the images of an information set under automorphisms, the certificate bounds only the words lighter than `U` ([The certificate of an automorphism family](brouwer_zimmermann.md#sec-automorphism-certificate) and [Certificates and contracts](brouwer_zimmermann.md#sec-completion-contracts) in `brouwer_zimmermann.md`).
- **Connected-cluster search:** the depth-first search of `_connected_cluster/search.py`, which `method="connected_cluster"` selects.
  It grows supports through the unsatisfied rows of a check matrix `H`.
  It finds light binary or symplectic words outside the row space of an excluded matrix `G`, or light vectors with a given syndrome.
  Unless enumerating is cheap, `"auto"` runs it for most distance searches of quantum codes and circuits (steps 2 and 3 of [Choosing a method](problems.md#sec-choosing-a-method) in `problems.md`).
- **Automorphism:** in this package, a permutation of the coordinates of a search that maps the words it searches, and those it excludes, onto themselves.
  The public functions take only permutation automorphisms of a code, which permute qudits and apply no single-qudit gates, and lift them to the coordinates of a Pauli problem ([Pauli weight as Hamming weight](brouwer_zimmermann.md#sec-pauli-weight) in `brouwer_zimmermann.md`).
  An automorphism of a quantum code can also apply single-qudit Clifford gates (see `get_transversal_automorphism_group` in `qldpc.circuits`).
  In a Pauli problem, exchanging the coordinates of one qubit applies a single-qubit Clifford gate, but qLDPC never supplies such permutations.
- **Scan:** the enumeration of the combinations of rows of each search basis, layer by layer, a layer holding the combinations of a given number of rows.
  A scan names each combination by its **support**, the set of its row indices ([Packed enumeration with pivot-column deletion](brouwer_zimmermann.md#sec-packed-enumeration) in `brouwer_zimmermann.md`); elsewhere, as in a connected-cluster search, the support of a word is the set of its nonzero coordinates.
  Counts have their own scan, in `_counting/scan.py`, and the Brouwer-Zimmermann distance and coset searches (see [The coset-leader search](problems.md#sec-coset-leader-search) in `problems.md`) use the compiled scans of `_brouwer_zimmermann/affine_search.py`.
  Every scan steps from one support of a layer to the next with the compiled step of `_information_sets/support_stepping.py`.
- **Search by Hamming weight:** a Brouwer-Zimmermann coset search of `CosetLeaderSearch` that weighs its binary words by their Hamming weight, up to a fixed factor.
  The searches by physical weight are searches by Hamming weight, since they search the images of Pauli errors under the map `T` of [Pauli weight as Hamming weight](brouwer_zimmermann.md#sec-pauli-weight) in `brouwer_zimmermann.md`.
  The Hamming weight of the image of an error is twice the physical weight of the error.
  The searches of X parts and of Z parts, into which `CosetLeaderSearch` splits a query by CSS weight on a CSS stabilizer state, are searches by Hamming weight too ([The coset-leader search](problems.md#sec-coset-leader-search) in `problems.md`).
  A search by CSS weight on a stabilizer state that is not CSS weighs a Pauli error by `max(|x|, |z|)`, where `x` and `z` are its X and Z parts, so it is not a search by Hamming weight.
  Of the coset searches, only the searches by Hamming weight can weigh words by their tails and skip completions by completion tables ([Coset searches](brouwer_zimmermann_code.md#sec-coset-searches) in `brouwer_zimmermann_code.md`).

<a name="sec-layout"></a>
## Layout

- `distances.py` and `logical_counts.py` are the public interfaces of the distances and the counts.
- `coset_leaders.py` is the public interface of the coset-leader search.
- `min_weight_bounds.py` defines `MinWeightBound`, the answer of a coset-leader query for one syndrome.
- `_auto.py` holds the parts of the automatic choice of a method, `method="auto"`, that compare the costs of the methods (see [Choosing a method](problems.md#sec-choosing-a-method) in `problems.md`).
- `_brouwer_zimmermann/` and `_connected_cluster/` hold the Brouwer-Zimmermann and connected-cluster searches.
  Each holds the distance search of its method, and the coset searches that answer the single and batch queries of the coset-leader search by that method.
  `_brute_force.py` holds exhaustive enumeration.
- `_counting/` holds the word scan of the counts, the local Clifford gates of deformed counters, and the MacWilliams method.
- `_information_sets/` holds what the scans of search bases share: basis families and their certificates, automorphism families, completion tables, and the step from one support of a layer to the next.
  The Brouwer-Zimmermann searches and the counts use it, and the connected-cluster search a few of its functions, on weight divisors and automorphisms.
- `_cosets.py` holds the parts that the coset searches of both methods share.
  `coset_leaders.py` and `_connected_cluster/sparse_checks.py` also use them.
- `_sector_maps.py` finds the qubit permutations that let a distance search of a CSS code search one sector (see [Searching one sector of a CSS code](problems.md#sec-one-sector) in `problems.md`).
- `_tuning.py` holds the settings of the searches.
- `_threads.py` runs the work items of a search, or the blocks of a batch of queries, on several threads.
- `_code_matrices.py` holds the binary input checks, nested bases, the stabilizers of subsystem codes, and the symplectic-to-Hamming map.

<a name="sec-where-to-look"></a>
## Where to look

The table below says where to start for some common kinds of change.
It is not a list of every module: the docstring of each module says what it holds, and a search for the label of a section of a guide, such as `sec-coset-leader-search`, finds the code that the section explains.

| To change… | Start in (this directory) |
| --- | --- |
| The methods that a problem takes, or how their names are validated | `distances.py`, `coset_leaders.py`, and `logical_counts.py`, which define the method type of each problem |
| The automatic choice of a method (`method="auto"`) | `distances.py` and `_auto.py`, after [Choosing a method](problems.md#sec-choosing-a-method) in `problems.md` |
| Which sectors a distance search of a CSS code searches | `distances.py` and `_sector_maps.py`, after [Searching one sector of a CSS code](problems.md#sec-one-sector) in `problems.md` |
| The counts, the deformed counters, or `compile_search_setup` | `logical_counts.py`, and `_counting/` for the scan of a count, the local Clifford gates of deformed counters, and the MacWilliams method |
| The coset-leader search, and how its queries split into coset searches | `coset_leaders.py`, after [The coset-leader search](problems.md#sec-coset-leader-search) in `problems.md` |
| The Brouwer-Zimmermann (BZ) distance search | `_brouwer_zimmermann/distance.py`, and `_brouwer_zimmermann/affine_search.py` and `_brouwer_zimmermann/affine_instance.py` for the compiled scans and the instances that it shares with the BZ coset searches |
| The BZ coset searches | `_brouwer_zimmermann/coset.py`, which runs their queries, `_brouwer_zimmermann/coset_routes.py`, which chooses their routes, and `_brouwer_zimmermann/coset_tables.py`, which builds their tables |
| The connected-cluster searches | `_connected_cluster/search.py` for distances and vectors with a given syndrome, `_connected_cluster/coset.py` for coset searches, and `_connected_cluster/sparse_checks.py` for sparser checks |
| Exhaustive enumeration, which the `brute_force` method runs | `_brute_force.py` |
| Basis families, certificates, automorphism families, completion tables, or the step from one support of a layer to the next | `_information_sets/` |
| The parts that the coset searches of both methods share | `_cosets.py` |
| The bound for one syndrome that the coset-leader search and its coset searches give (`MinWeightBound`) | `min_weight_bounds.py` |
| The settings of the searches | `_tuning.py`, and [Tuning](#sec-tuning) below |
| How searches run on threads | `_threads.py`, and the module of each search for how that search splits its work |
| Binary input checks, nested bases, the stabilizers of subsystem codes, or the map of symplectic vectors to binary vectors | `_code_matrices.py` |
| Bit-packed linear algebra over GF(2), which other subpackages share | `src/qldpc/_binary.py` |
| How numba code is compiled or cached | `src/qldpc/_compilation.py`, and [Compiled code](#sec-compiled-code) below |

<a name="sec-layers"></a>
## How the modules depend on each other

The main levels of the package, from the top:

1. `coset_leaders.py` and `logical_counts.py`.
2. `distances.py`.
3. `_auto.py`, the automatic choice of a method.
4. The methods `_brouwer_zimmermann/`, `_connected_cluster/`, `_counting/`, and `_brute_force.py`.
5. `_information_sets/`, `_cosets.py`, and `min_weight_bounds.py`, which the methods build on, and `_sector_maps.py`, which `distances.py` uses.
6. `_tuning.py`.
7. `_code_matrices.py` and `_threads.py`.

A top-level module or subpackage imports only from lower levels.
A module in a subpackage may also import from lower layers of that subpackage.
The `_layering_test.py` of the package, and that of each subpackage, holds the complete order of the modules and subpackages that it checks in its `_LAYERS`, and checks the imports against it through a helper of `src/qldpc/conftest.py`.
The docstring of that helper gives further rules, such as that no module imports the root of the package.
Modules of every level may also import `src/qldpc/_binary.py`, the bit-packed linear algebra over GF(2) that other subpackages share.
`src/qldpc/codes/common.py` imports this package, so `logical_counts.py` imports the code classes of `qldpc.codes` inside the functions that use them.
Modules elsewhere in qLDPC, such as `src/qldpc/codes/common.py`, use private names of this package, so grep `src/qldpc` for a private name before you move or rename it.

<a name="sec-tuning"></a>
## Tuning

A value that a search reads when it starts, and that changes how long the search takes or how much memory it uses but no result that it guarantees, is a field of `SearchTuning` in `_tuning.py`, unless its best value does not depend materially on the user, the code, or the machine (below).
Such values include a threshold that chooses a route, a limit, a coefficient of a cost model, a batch size, and a switch that turns a step off.
They are not arguments of public functions, whose arguments say what to compute and by which method, such as `cutoff`, `max_weight`, and `method`.

- A search reads the active tuning once, when it starts, and passes the values that it needs down as arguments, including to compiled code, so that one search never mixes two tunings.
  The distance functions of `distances.py` read it for a distance search, `CSSLogicalCounter.count_logical_ops` for a count, and `CosetLeaderSearch` when it is built.
  The distance searches of `_brouwer_zimmermann/distance.py` and `_connected_cluster/search.py` read it only if their callers pass none.
  A cache that outlives a search is keyed by the values that it depends on, as in `_brouwer_zimmermann/coset.py`, whose cache of BZ coset searches is keyed by the fields of the tuning that their setup reads.
- Only a value that qLDPC fixes before any search starts is a constant in `_tuning.py`, such as the size of a cache, which `functools.lru_cache` fixes at import.
  Code reads cache sizes as `_tuning.NAME`.
  Constants fixed by the algorithm or by native integer widths, such as the caps that keep estimates of work within a native integer, and the seeds of random generators stay in the modules that use them.
  The exception is the largest value of an integer field of `SearchTuning`, a constant of `_tuning.py`.
  The seeds make results reproducible rather than tune anything.
  Speed-only constants whose best value does not depend materially on the user, the code, or the machine also stay in their modules, such as the numbers of work items, blocks, and ranges for each thread in `_threads.py`, which only balance the threads.
- Each field and constant has its own `#:` comment directly above it, for Sphinx to render.
  The first line of a comment has no colon, since Sphinx reads text before such a colon as a type.
  A field's comment says what the field controls, with its unit, its special values, and its range if its metadata narrows it.
  The comment also names the kind of the default and how the value was found, or says why the default was chosen by hand, as the "Retuning" section of the module docstring requires.
  Where a guide section or a cost model explains the field's heuristic, the comment points there.
  A constant's comment says what it controls; the "Retuning" section says how the sizes of caches were chosen.
  Comments, docstrings, and guides in this package follow the [writing standard's rule on measurements](../../../../docs/source/contributing/standards/writing.rst#writing-documentation).
- Change a default only as the "Retuning" section allows; [experiments/distance_tuning](../../../../experiments/distance_tuning/README.md) says how to check one.
- [Exact-search performance and tuning](../../../../docs/source/exact_search_tuning.rst) renders the module's values from these comments, after hand-written tables that lead from symptoms and routes to the values.
  The tables are selective; add a value to them when it is how a reader would fix a symptom or control a route.
  `_tuning_test.py` checks that each value has its own comment.
  The documentation build fails on a broken reference into `_tuning`.
- The module docstring of `_tuning.py` says which results a tuning can change.
- To force a route or exercise a value in a test, change fields of `SearchTuning` through the fixture of `src/qldpc/codes/conftest.py` that changes the active tuning until the test ends, rather than patching module attributes.
- If the implementation cannot safely support a field's full range, narrow it in the field's metadata, and in the table of narrowed fields in `_tuning_test.py`, rather than adding code for unsupported values.
  A field whose default is `None`, such as `num_threads`, also accepts `None`.

<a name="sec-threads"></a>
### Threads

A BZ distance search scans each costly layer of a search basis on `num_threads` threads, which share one copy of the completion tables of that basis ([Behavior spanning files](brouwer_zimmermann_code.md#sec-cross-file-behavior) in `brouwer_zimmermann_code.md`).
A connected-cluster distance search searches each costly weight limit on `num_threads` threads, which give the same witness and count of states as a search on one thread ([Threads](connected_cluster.md#sec-threads) in `connected_cluster.md`).
With either method, the batch queries of `CosetLeaderSearch`, `find_first_not_within_max_weight` and `get_min_weights`, move the rest of a costly batch to `num_threads` threads ([The coset-leader search](problems.md#sec-coset-leader-search) in `problems.md`).
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
[Working here](#sec-working-here) gives the rules for code that threads run, and the tests that check them.

<a name="sec-compiled-code"></a>
## Compiled code

Most searches spend their time in functions that numba compiles.
Two decorators in `src/qldpc/_compilation.py` mark them as always compiled or compiled on request, and its docstring says when to use each:

- Always-compiled functions are the per-word loops and their helpers.
  They are compiled on first use and cached on disk, in a directory named by a hash of qLDPC's source, so the first run after an edit anywhere in qLDPC compiles again.
  A small helper that hot loops call, such as the step from one support of a layer to the next, is compiled with `inline=True`, so that compiled callers include its code rather than pay for a call.
- Compiled-on-request functions are parts of the set-up of searches, which run as plain Python unless a program calls `compile_search_setup()`.
  Their compiled code is cached in the same directory.

An entry point that several threads run at once, such as the scan of a chunk of work items of a layer in `_brouwer_zimmermann/affine_search.py`, or the compiled depth-first search in `_connected_cluster/search.py`, is always compiled with `nogil=True`, so that its compiled code, and every compiled function that it calls, runs without Python's global interpreter lock.
The docstring of `src/qldpc/_compilation.py` states the rules for that code.

Numba looks up the functions that compiled code calls when compiling it, so compiled code may call NumPy and compiled functions, but not plain Python functions of qLDPC.
Tests in `src/qldpc/_compiled_code_rules_test.py` check further rules, and the name of a failing test says which one was broken.

Set `QLDPC_DISABLE_COMPILATION=1` before importing qLDPC to run all of this code as plain Python while debugging.
It affects only qLDPC, whereas `NUMBA_DISABLE_JIT=1` also runs other packages that use numba, such as galois, as plain Python, which makes them slow too.

<a name="sec-working-here"></a>
## Working here

- Tests sit beside their modules, and each module must be fully covered by its own test file.
  `python checks/pytest_.py src/qldpc/codes/distance` runs the tests.
  `python checks/coverage_.py src/qldpc/codes/distance/<module>.py` checks one module's coverage.
  The validation commands in `docs/source/contributing/index.rst` list the other checks.
- `conftest.py` holds the fixtures and helpers that several test files share, such as a count of logical operators by enumeration.
- Coverage cannot trace compiled code, so the coverage check runs compiled functions as plain Python, and CI runs these tests again with them compiled (see `docs/source/contributing/standards/testing.rst`).
- `python -m pytest src/qldpc/codes/distance --compile-search-setup` runs the tests with the set-up code of searches compiled.
  Some tests request the fixture of `conftest.py` that runs the set-up code of searches as plain Python even then, and its docstring says which tests need it.
- A change to the scans or how they step, to pruning, to the work items, or to any code that threads run must keep the requirements on threads of the guide of each method ([Correctness requirements](brouwer_zimmermann_code.md#sec-requirements) in `brouwer_zimmermann_code.md`, or [Correctness requirements](connected_cluster.md#sec-requirements) in `connected_cluster.md`), and pass the tests on threads that cite those requirements.
  So must a change to data that threads read, such as completion tables, the routes of coset searches, and the tables of a connected-cluster search, or to a decision that a search makes from the results of a layer or a weight limit.
  Code that threads run takes the tuning only from its search, and never reads the active tuning of `_tuning.py` itself, since new threads can start with the default tuning (see the module docstring of `_tuning.py`).
  Code that threads run also calls no numba-parallel code, such as some functions of galois, since that can abort the process (see `_threads.py`).
- While each test of this package runs, a fixture of `conftest.py` fails it if the workers of `_threads.py` read the active tuning or call numba-parallel code from Python.
  Review checks such calls on paths that tests do not run on threads.
  `_thread_rules_test.py` checks that the docstring of each entry point compiled with `nogil=True` cites the correctness requirements of the guide of its method, `brouwer_zimmermann_code.md` or `connected_cluster.md`.
  `src/qldpc/_compiled_code_rules_test.py` checks that no function of qLDPC is compiled with numba's `parallel` option, and that always-compiled functions use no names imported from packages other than qLDPC, NumPy, and numba, such as functions of galois.
  Some tests on threads of each method run with every array of their search read-only, through helpers of `conftest.py`.
  Numba then rejects a write to such an array in any branch of compiled code.
  Each of these tests cites the correctness requirements of the guide of its method, [Correctness requirements](brouwer_zimmermann_code.md#sec-requirements) in `brouwer_zimmermann_code.md` or [Correctness requirements](connected_cluster.md#sec-requirements) in `connected_cluster.md`.
  Each guide lists the writes that these tests miss.
  Review checks those writes.
- "Layout" and "Where to look" are selective maps, so a new module needs no entry in them.
  `_readme_test.py` fails if this page or `problems.md` names, in a code span, a Python file of this package that does not exist.
