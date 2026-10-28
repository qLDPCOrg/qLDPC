# Agent and contributor guide

This guide is for people and agents changing qLDPC.
For a human-facing explanation of the data model and package relationships, start with the [library map](docs/source/library_map.rst).
For exact signatures and per-object literature, use the source docstrings or the generated [API reference](https://qldpc.readthedocs.io/autoapi/index.html).

## What to trust when documents disagree

For what the code does, use this order:

1. Current implementation and its invariant tests.
2. Public docstrings and the `__all__` lists in package `__init__.py` files.
3. The [library map](docs/source/library_map.rst) and executable [examples](examples/).
4. Historical discussion, issues, or review notes.

For how code should be written, this guide takes precedence over existing code: code that breaks a rule here is debt to fix, not a precedent to copy.
Re-derive behavior from the current branch before documenting or changing it.
Do not copy transient project history, machine-specific paths, or local-session instructions into production files.

## Orientation

The source tree under `src/qldpc/` mirrors the library: each package and module starts with a docstring whose first line summarizes it, and the [library map](docs/source/library_map.rst) explains how the packages relate.
Canonical executable notebooks live in [`examples/`](examples/), Sphinx pages and the library map in [`docs/source/`](docs/source/), local wrappers for CI's quality gates in [`checks/`](checks/), and untested research scripts in [`experiments/`](experiments/).

The main code hierarchy is:

```text
AbstractCode
|- ClassicalCode
`- QuditCode
   `- CSSCode
```

The methods in `codes/common.py` share cached and mutable state for standard form, logical and gauge operators, parameters, and code transformations.
Do not split them into mixins merely to reduce the file length.
Each decoder backend defines its builder and its specification helper (with `decoder_spec` or `observable_decoder_spec`) beside its implementation, under `decoders/external/` or `decoders/custom/`.
Plumbing that one subpackage uses is an underscore name in that subpackage, and plumbing that several subpackages share lives in private modules (see [Public API and private code](#public-api-and-private-code)).

## Public API and private code

- A name is public if it is broadly useful to researchers and engineers, and private if it is plumbing that makes qLDPC work.
  Where a name is used, such as in other modules or in notebooks, does not decide this.
- Plain names defined in public modules (no underscore anywhere in the import path) are public API.
  Keep them working, and retire a released public name through a tested shim rather than breaking its import (see [Deprecations and future changes](#deprecations-and-future-changes)).
- [`src/qldpc/__init__.py`](src/qldpc/__init__.py) exports subpackages, and each subpackage's `__all__` lists the public names that it re-exports.
  Re-export only what typical users reach for, leave specialized public tools at their module path, give each public object one canonical public path, and put utilities that several packages share in a neutral top-level module such as `qldpc.math`.
  Preserve public import paths when moving implementation code.
- Module-level underscore names, and a class's underscore attributes and methods, may be used anywhere in their top-level subpackage, such as `qldpc.codes` or `qldpc.decoders`, but not outside it.
  Each top-level module, such as `qldpc.math`, is its own scope.
- Private modules, with an underscore anywhere in their import path, exist only for plumbing shared across subpackages or for a substantial standalone topic.
  Name them by topic, never `_common`, `_util`, `_helpers`, or `_misc`, and use plain names for what other modules use.
- [`qldpc/_utils.py`](src/qldpc/_utils.py) holds Python-level plumbing only: lazy imports, warning stack levels, and docstring formatting.
- Public objects are defined in public modules; never re-export a private module's object as public API.
- Everything under [`src/qldpc/experimental/`](src/qldpc/experimental/) is unstable, can change without a deprecation period, and is exempt from these private-name rules.

## Code style and organization

- Every source file starts with `# SPDX-License-Identifier: Apache-2.0`, a blank line, and a module docstring whose first line summarizes the module.
- Docstrings follow the Google convention, and lines are at most 100 columns long.
- Python comments and docstrings use ordinary paragraphs: separate sentences with two spaces and fill lines to the 100-column limit.
- Keep one prose sentence per physical source line in `.rst`, `.md`, and Markdown notebook cells.
  Put `First sentence.` and `Second sentence.` on separate source lines; leave code blocks, equations, and table syntax intact.
- Preserve a module's existing section headings and organization when adding code, and add a section when new functionality does not fit an existing one.
- Start each section with a line of 100 `#` characters followed by a one-line `# title` comment.
- Put important user-facing APIs first within modules and sections, and keep related public entry points together.
- Within each section, put plain names before underscore names, except where code that runs at import time (a base class, decorator, default argument, or module-level call) needs an underscore name first.
  Put a helper in the section of the code that mainly uses it, or in a final helper section if several sections share it, and do not move existing helpers merely because their usage changes.
  Private constants and types may sit near the imports.
- Within a class, put public methods first (for decoders, the decode methods) and private helper methods after them.
- [`code_style_test.py`](src/qldpc/code_style_test.py) checks license headers, section headers, and the order of plain and underscore names.

## Core invariants

### Fields and arrays

- Code arithmetic happens in a `galois.FieldArray`.
- Extension-field storage integers are not ordinary integers modulo the field order.
  Do not use `% field.order` as field coercion; construct or preserve values through the field class.
- Preserve the concrete field when copying or transforming arrays.
  Code equality and compatibility often require the same field class, not merely arrays with equal integer views.
- A `RingArray` has one base `GroupRing`.
  NumPy operations must reject arrays from incompatible rings, including arrays supplied through keyword arguments such as `out=`.

### Codes and Pauli conventions

- A classical parity-check matrix `H` defines words satisfying `H @ word == 0`.
- A general quantum check is a symplectic row `[X | Z]` of even length.
  Use [`qldpc.math.symplectic_conjugate`](src/qldpc/math.py) rather than hand-writing sign and half-order conventions.
- A CSS code stores X-type and Z-type checks separately.
  Commuting stabilizer checks satisfy `H_x @ H_z.T == 0` in the code's field.
- For a subsystem code, constructor check rows generate the gauge group and need not commute.
  Stabilizers come from the center.
  Do not decode a subsystem code as though all gauge generators were stabilizers.
- `CSSCode(..., promise_equal_distance_xz=True)` is trusted, not verified.
  A false promise can make distance results wrong and dependent on which sector was computed first.
- Built-in families may cache construction- or literature-supplied parameters, so test them against independent oracles (see [Tests](#tests)).
- [`codes/distance/`](src/qldpc/codes/distance/) is explicitly binary.
  Use field-aware code methods and oracles for nonbinary constructions.

### Decoders

- Build decoders from typed specifications, as in `decoders.bp_osd(...).build(pcm_or_dem)` or `decoders.mwpm(...).build_observable_decoder(dem)`.
- Specifications of external decoders use their backends' own option names, such as `error_priors`, `observable_error_matrix`, or `error_probabilities`.
  These differences between backends are intended, not inconsistencies to fix.
- Keep error decoders (`decode_errors`, with `decode` as an alias) distinct from observable decoders (`decode_observables`).
  Code that consumes a user-supplied error decoder coerces it with `_as_error_decoder` from [`decoders/protocols.py`](src/qldpc/decoders/protocols.py), or outside `qldpc.decoders` with `build_error_decoder` (below), and calls `decode_errors`.
- A method that decodes a matrix it constructs itself must reject prebuilt decoders with `reject_prebuilt_decoder` from `decoders/construction/_resolution.py`.
- Code-capacity estimators resolve their `decoder=`, `decoder_x=`, and `decoder_z=` inputs with `get_code_capacity_decoder` from [`codes/_code_capacity.py`](src/qldpc/codes/_code_capacity.py).
  It always yields an observable decoder: on binary codes, specifications that predict observables natively build it from a DEM; otherwise inferred errors are projected onto logical predictions.
- Keep `build_error_decoder` in `decoders/construction/_resolution.py` specific to error decoders.
  Dispatch on explicit capabilities (`compile_decoder_for_dem`, `decode_observables`, the `ErrorDecoder` protocol), never on output length.
- Only decoders that declare erasure support may append an erasure flag.
  They append that flag as the last entry of each inferred error; unsupported decoders must reject `add_erasure_bit=True`.
- Detector-error-model decomposition indices and remaps must remain valid after cancellation and simplification.
  Test malformed components, not only happy-path Stim models.
- Sliding-window decoders infer time from detector coordinates by a heuristic described in [`decoders/sinter/window.py`](src/qldpc/decoders/sinter/window.py).
  Do not change that fallback; pass an explicit `detector_to_time` mapping for models with other layouts.

### Circuits

- Circuit and tableau constructors are qubit-only and should use the `_restrict_to_qubits` guard from [`circuits/common.py`](src/qldpc/circuits/common.py).
- `get_encoding_circuit` constructs a valid encoder but is not fault-tolerant.
  Do not present it as a fault-tolerant state-preparation procedure.
- Keep data, check, reference, and ancilla roles explicit through `QubitIDs` and bookkeeping types.
  Do not infer roles from coincident integer indices.
- qLDPC memory-circuit detectors use coordinates such as `(round, 0, check_index)`.
  A later monotone coordinate can enumerate checks rather than time.
- Noise-model operation immunity and qubit immunity are separate controls; preserve both.

### External systems and caching

- GAP helpers may start subprocesses, read standard input, use the clipboard, access the network, or clone missing packages.
  Keep these side effects explicit in the docstrings of the public entry points that can trigger them, and in error messages.
- Expensive reusable computations should use `qldpc.cache.use_disk_cache(cache_name)`.
  Do not hide errors by returning a default value that looks successful.
- Numba code uses `always_compiled` or `compiled_on_request` from `qldpc/_compilation.py`, never `njit(cache=True)`; that module's docstring says which to use, and `_compilation_test.py` enforces its rules.

## Tests

- Every module needs its own co-located `*_test.py`: the coverage gate requires each module to be fully covered by its own test file, and does not gate a module without one.
  Reach a hard-to-reach branch with a focused test rather than a broad scenario.
- Tests run with network sockets disabled.
  Mock only external boundaries such as web, subprocess, clipboard, prompt, and optional-package paths, or a module's own thin wrappers around them, never the logic under test, and do not add a live-service test.
- Disk caches are bypassed while pytest is imported, so tests must not depend on cache persistence.
- Pytest turns every `DeprecationWarning` and `FutureWarning` into an error, so tests use the modern API, and test deprecated usage and future-change warnings with `pytest.warns` (see [Deprecations and future changes](#deprecations-and-future-changes)).
  This applies to warnings from other packages too; avoid the call that triggers one, or add a narrow `ignore` entry to `filterwarnings` in `pyproject.toml`.
- A test should guard behavior that matters: results, invariants, documented errors, and contracts between modules.
  Do not pin implementation choices such as internal call order, cache layouts, or incidental error-message wording.
  A test may call a private helper of the module that it tests to check the helper's contract or reach coverage, and may set private state of objects to construct a case, but should not assert on private state.
- Keep each test minimal: use the smallest code, field, or circuit that exercises the behavior, do not parametrize over cases that take the same path, and delete tests implied by others.
- Prefer independent oracles (rank, commutation, row-space identities, or brute force on small instances) over values that the code under test computed or cached.
  Checking a stored value against an independent expected value, such as one from the literature, is not tautological; deriving the expected value from the code under test is.
- Keep tests that reproduce results from the literature, and cite the source in the test.
- Tests run in random order with a per-run random seed (pytest-randomly), so they must not depend on order.
  Seed random inputs from `pytestconfig.getoption("randomly_seed")` for reproducibility, and make sure that the assertions hold for every seed; otherwise use a fixed seed.
- Check that a new or rewritten test fails when the behavior it guards is broken.

## Common change recipes

### Add or change a code construction

1. Put a classical family in [`codes/classical.py`](src/qldpc/codes/classical.py) and a quantum or CSS family in [`codes/quantum.py`](src/qldpc/codes/quantum.py), next to its closest base or sibling.
2. Re-export public names from [`codes/__init__.py`](src/qldpc/codes/__init__.py).
3. Add a linked literature reference and document the field, subsystem, distance, and validation assumptions in the class docstring.
4. Add co-located tests with independent invariants: rank-derived dimension, CSS orthogonality, symplectic commutation, row-space identities, known small distances, or exact cross-family equivalence.

### Change code-core algebra

1. Read the cached-state interactions in [`codes/common.py`](src/qldpc/codes/common.py) before changing standard form, logicals, stabilizers, gauges, dimension, or distance.
2. Test both CSS and non-CSS paths, stabilizer and subsystem paths, `k=0`, and at least one odd characteristic when signs matter.
3. Preserve public imports and invalidate or transfer cached values deliberately.

### Add or adapt a decoder

Follow the [adding a decoder guide](docs/source/adding_decoders.rst).
In addition, do not change generic resolution for an ordinary backend, and decide and test batch behavior, nonbinary support, detector-error-model support, and erasure signaling explicitly, with direct syndrome and error reproductions.

### Change a circuit workflow

1. Keep matrix, qubit-role, measurement-record, detector-record, and Stim-circuit bookkeeping in sync.
2. Test noiseless detector determinism and observable behavior before adding noise.
3. Exercise repeated rounds and seam behavior; a locally valid circuit fragment can still become wrong when initialization, QEC cycles, and readout are joined.
4. State computational and fault-tolerance limitations in the relevant function or class docstring.

### Change an external integration

1. Separate parsing/encoding logic from process, network, clipboard, and prompt mechanics.
2. Test both the callable-tool path and the absent/manual/error paths without live resources.
3. For finite fields, round-trip values through the external system's actual element encoding; do not assume integer storage conventions match.
4. Report nonzero return codes, standard error, malformed output, and missing data clearly.

### Add documentation or an example

1. Put explanations that span packages in [`docs/source/library_map.rst`](docs/source/library_map.rst), exact API behavior in docstrings, and executable workflows in [`examples/`](examples/).
2. The notebooks under `docs/source/examples/` are links to the canonical files under `examples/`.
   Edit the canonical notebook, not the documentation-tree link.
3. Keep README and the library map selective; AutoAPI lists the public symbols.
4. When a public limitation changes, update the relevant function or class docstring and every README or guide that repeats it in the same pull request.
5. Run the strict documentation build before considering the change complete.

## Deprecations and future changes

Never break callers of a released public API without warning.
Removing or renaming a name, argument, or mode is a deprecation: old usage keeps working with a `DeprecationWarning`, and fails once the deprecation is removed.
Making the same call silently behave differently, as when a default changes, is a future change, which warns with a `FutureWarning`, also while its old behavior is phased out.

- Add a shim only for a name that was public in a released version, at its released path: listed in a package `__all__`, or used in released examples or docs.
  Names added since the last release may change without one.
- Deprecation code lives in the private package [`qldpc/_deprecations/`](src/qldpc/_deprecations/), whose module docstrings explain its mechanics: register renamed and moved names in `ALIASES`, deprecated functions that warn when called in `SILENT_ALIASES`, and deprecated methods in `install_deprecated_methods`.
  Code in that package may use other subpackages' underscore names, to reproduce old behavior.
- Elsewhere, handle deprecated arguments, modes, and calling conventions, and future changes, in small blocks that call `_deprecations` helpers such as `warn_deprecated`, `warn_future_change`, and `defunct`, and move any substantial logic into `qldpc/_deprecations/`.
  Mark each such block, and the parameter that it handles, with `# DEPRECATED: see qldpc._deprecations` or `# FUTURE CHANGE: see qldpc._deprecations`.
- The library and its docs, including the README, are written as if deprecated APIs do not exist, and tests use deprecated APIs only to test them.
  No `__all__` lists a deprecated name, and no `TYPE_CHECKING` alias exposes one, so that star imports do not warn and type checkers reject deprecated names.
  A docstring mentions a deprecated option or old value in at most one `Args` entry, which names its replacement.
- Warning messages and docs say "in a future release" rather than naming a version.
- A future change takes three steps, at least one minor release apart:
  1. Give the keyword that selects the behavior (a new keyword, or the one whose default changes) a default of `None` that keeps the old behavior and warns; explicit values select the old or the new behavior without a warning.
     Describe both behaviors in the docstring, and the change in a "Future changes" section of the [library map](docs/source/library_map.rst), which is added when needed and removed when empty.
  2. Make `None` select the new behavior without a warning, keep the old value working with a `FutureWarning` that it will be removed, update the docstring, and remove the library-map entry.
  3. Make the old value raise `_deprecations.defunct(...)`, mark the block `DEPRECATED`, and rename its tests `test_deprecated_*`.
     If the new value is the only one left, also deprecate the keyword.

  From step 1 until step 3, qLDPC code, tests, and notebooks pass the new value explicitly.
- Name new tests of inline deprecations `test_deprecated_*`, and tests of future changes `test_future_change_*`.
  [`_deprecations/core_test.py`](src/qldpc/_deprecations/core_test.py) tests every registry entry automatically; add a case there for each new deprecated method.
- To remove deprecations, grep for `_deprecations` and, case-insensitively, for `deprecat`, and delete the deprecated code, `Args` notes, and tests, but leave `FUTURE CHANGE` blocks alone.

## Validation commands

Install the development environment with `python -m pip install -e '.[dev]'`.
CI's exact environments and commands are in [`.github/workflows/`](.github/workflows/).

| Command | Purpose |
| --- | --- |
| `python checks/pytest_.py [paths]` | Pytest suite, or the tests under the given paths; does not execute notebooks by default |
| `python -m pytest path/to/module_test.py::test_name -v` | One focused test |
| `python checks/pytest_.py --notebook examples/decoders.ipynb examples/basics.ipynb examples/noise_models.ipynb` | Execute the notebooks that CI executes, with their dependencies installed |
| `python checks/format_.py --check` / `--fix` | Check, or apply, Ruff and `pyproject.toml` formatting |
| `python checks/lint_.py` | Ruff lint |
| `python checks/mypy_.py` | mypy with the repository's typing settings, over source and tests |
| `python checks/coverage_.py` | 100% statement coverage of each module by its own test file, as in CI |
| `python checks/build_docs.py` | Strict Sphinx and notebook documentation build |
| `python checks/all_.py` | Combined gate: formatting, lint, mypy, tests with overall 100% coverage, and (on full runs) the docs build; it does not run the per-module coverage gate or `pyproject.toml` formatting |

Run checks on what you changed while iterating, and save repository-wide runs for a finished change.
Every check script except `build_docs.py` accepts explicit files, as in `python checks/coverage_.py src/qldpc/codes/common.py`, or `-i/--incremental [REVISION ...]`.
The incremental mode checks only the files changed since the merge base with `upstream/main`, `origin/main`, or `main` (the first that exists), or with the given revisions.
For example, `python checks/coverage_.py -i` runs the per-module coverage gate for changed modules only, which is much faster than a full run.
`format_.py` always checks, or with `--fix` rewrites, `pyproject.toml` as well, whatever files it is given.
Prefer the individual scripts for incremental runs: `all_.py -i` measures overall coverage over only the selected tests, so shared fixtures in `conftest.py` files can fail it spuriously.
Before merging, run `python checks/all_.py` and `python checks/coverage_.py`.

The docs build renders saved notebook outputs without executing code cells.
Refresh outputs by executing the entire canonical notebook in an offline, editable installation before changing a notebook example.
Pass an absolute path to this checkout's `src` directory in `PYTHONPATH` if a notebook runner starts its kernel from `examples/`.
The TQEC notebook (`examples/logical_error_rates/specialized/decoding_tqec_circuits.ipynb`) requires Python 3.11--3.13 because its pinned TQEC dependency does not support Python 3.14.

Some check wrappers discover files through Git.
Add new source and test files to the index before relying on the full gate to include them.
Do not weaken a failure with `noqa`, `type: ignore`, or coverage exclusions unless the exceptional condition is real and documented.
