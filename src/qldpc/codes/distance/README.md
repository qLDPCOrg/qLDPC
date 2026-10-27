# Exact distances and logical-operator counts: a guide for contributors

This directory is the package `qldpc.codes.distance`.
It computes exact distances of binary codes, and counts the logical operators of a given weight in binary CSS codes.
Users reach it through the `get_distance_exact` method of binary codes in `src/qldpc/codes/common.py`, and through `CSSLogicalCounter`.
`brouwer_zimmermann.md` explains the algorithm, maps its concepts to names in the code, describes its heuristics for choosing basis families, and lists the correctness requirements that every change to a search must keep.

## Public names

`__init__.py` re-exports these names, and everything else in this directory is private:

- `get_distance_classical`, `get_distance_quantum`, `DistanceMethod`, and `DistanceBackend`, from `core.py`.
- `CSSLogicalCounter`, `CSSLogicalCounts`, `count_logical_ops_css`, `LogicalCountMethod`, and `compile_search_setup`, from `logical_counts.py`.

## Terms

- **Sector:** a basis of the words that a search examines, with labels that tell nontrivial words from trivial ones, such as the logical operators of one type of a CSS code from its stabilizers.
- **Information set:** the basis of a sector, row reduced so that a block of its columns forms an identity matrix.
  `brouwer_zimmermann.md` calls this a search basis.
- **Basis family** (`BasisFamily`): the information sets that one search uses together, stacked into arrays.
- **Certificate** (`Certificate`): a lower bound on the weight of the words that a search has not yet visited, which tells the search when it can stop.
- **Scan:** the enumeration of the combinations of rows of each information set, layer by layer.
  Counting scans in `_counting/scan.py`, and the distance search in `_brouwer_zimmermann.py`.
  Both scans and `_bounded_weight_search.py` step through a layer with `advance_support` in `_information_sets.py`.

## Where to look

| To change… | Look in (this directory) |
| --- | --- |
| A distance method, or the validation of method and backend names | `core.py`, and where `logical_counts.py`, `_counting/scan.py`, and `get_distance_exact` in `src/qldpc/codes/common.py` dispatch on the method name |
| The Brouwer–Zimmermann (BZ) distance search | `_brouwer_zimmermann.py` |
| How basis families and their certificates are built | `_information_sets.py` |
| Exhaustive enumeration, for small codes, fallbacks, and the `brute_force` method | `_brute_force.py` |
| Minimum-weight queries for syndrome classes of Pauli errors, used only by `src/qldpc/circuits/benchmarking/fault_propagation.py` | `_bounded_weight_search.py` |
| The counter, its deformed counters, or `compile_search_setup` | `logical_counts.py` |
| The scan of a count, or how its basis families follow local gates | `_counting/scan.py` |
| The local Clifford gates of deformed counters | `_counting/local_cliffords.py` |
| The MacWilliams counting method | `_counting/macwilliams.py` |
| Binary algebra, bit packing, or bit counts | `_binary.py` |
| How numba code is compiled or cached | `src/qldpc/_compilation.py` |

## How the modules depend on each other

From the top:

1. `core.py` and `logical_counts.py`, the public interfaces.
2. `_brouwer_zimmermann.py`, `_brute_force.py`, `_bounded_weight_search.py`, and the `_counting` package, the searches.
3. `_information_sets.py`, the basis families that the searches share.
4. `_binary.py`, the foundation.

A module imports only modules of its own level or below, except that `_counting/scan.py` takes `DistanceMethod` from `core.py` for an annotation.
`src/qldpc/codes/common.py`, `src/qldpc/circuits/benchmarking/fault_propagation.py` and `src/qldpc/circuits/benchmarking/_fault_enumeration.py`, and the deprecation registry in `src/qldpc/_deprecations/core.py` use private names of this package, so check them before you move or rename anything.

## Compiled code

Most searches spend their time in functions that numba compiles.
Two decorators in `src/qldpc/_compilation.py` mark them, and its docstring says when to use each:

- `@always_compiled` marks the per-word loops and their helpers.
  They are compiled on first use and cached on disk, in a directory named by a hash of qLDPC's source, so the first run after an edit anywhere in qLDPC compiles again.
  A small helper that hot loops call, such as `advance_support`, is marked `@always_compiled(inline=True)`, so that compiled callers include its code rather than pay for a call.
- `@compiled_on_request` marks parts of the set-up in `_information_sets.py` and `_counting/scan.py`, which run as ordinary Python unless a program calls `compile_search_setup()`.

Numba looks up the functions that compiled code calls when it compiles it, so compiled code may call NumPy and compiled functions, but not plain Python functions of qLDPC.
Tests in `src/qldpc/_compilation_test.py` check four further rules, and the name of a failing test says which one was broken:

- Only `src/qldpc/_compilation.py` asks numba for a disk cache, so use `@always_compiled`, not `@njit(cache=True)`.
- An always-compiled function may not call a compiled-on-request one.
- Always-compiled code may use nothing from outside qLDPC except NumPy and numba, because the hash covers only qLDPC's source.
- Other modules must call a compiled-on-request function through its module, and never import it by name.

Set `NUMBA_DISABLE_JIT=1` to run all of this code as plain Python while debugging.

## Working here

- Tests sit beside their modules, and each module must be fully covered by its own test file.
  `python checks/pytest_.py src/qldpc/codes/distance` runs the tests, `python checks/coverage_.py src/qldpc/codes/distance/<module>.py` checks one module's coverage, and the `AGENTS.md` at the root of the repository lists the other checks.
- Coverage cannot see compiled code, so tests also run compiled functions as plain Python.
  `conftest.py` holds the fixtures for this, and the helpers that several test files share, such as a check of counts against brute-force enumeration.
- `readme_test.py` fails if this page omits a module, or names a file that does not exist.
