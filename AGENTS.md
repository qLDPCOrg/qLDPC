# Agent and contributor guide

This guide is for people and agents changing qLDPC.
For a human-facing explanation of the data model and package relationships, start with the [library map](docs/source/library_map.rst).
For exact signatures and per-object literature, use the source docstrings or the generated [API reference](https://qldpc.readthedocs.io/autoapi/index.html).

## What to trust when documents disagree

When sources disagree, use this order:

1. Current implementation and its invariant tests.
2. Public docstrings and the `__all__` lists in package `__init__.py` files.
3. The [library map](docs/source/library_map.rst) and executable [examples](examples/).
4. Historical discussion, issues, or review notes.

Re-derive behavior from the current branch before documenting or changing it.
Do not copy transient project history, machine-specific paths, or local-session instructions into production files.

## Public API and compatibility

- [`src/qldpc/__init__.py`](src/qldpc/__init__.py) exports subpackages rather than flattening their classes and functions.
- Each stable subpackage has an explicit `__all__` in its `__init__.py`.
  Preserve those import paths when moving implementation code.
- Treat imports from ordinary `qldpc.*` packages as public and keep them working.
  Use a tested `DeprecationWarning` shim for a necessary rename or move rather than breaking an import.
  For a renamed class or other module-level name, resolve the old name with [`qldpc._util.get_deprecated_alias`](src/qldpc/_util.py) from a module-level `__getattr__`, both in the defining module and in any `__init__.py` that re-exports it, and keep the old name in `__all__`.
  The old name then refers to the same object as the new one, so `isinstance`, subclassing, and unpickling keep working.
- Everything under [`src/qldpc/experimental/`](src/qldpc/experimental/) is explicitly unstable and can change without a deprecation period.
  Do not infer that this weaker guarantee applies elsewhere.
- Keep complete lists of public symbols in `__all__` and AutoAPI.
  Human-written docs should explain what packages do and show representative tasks, not duplicate a class catalogue.

## Repository map

| Area | What it does | Tests and examples |
| --- | --- | --- |
| [`src/qldpc/codes/common.py`](src/qldpc/codes/common.py) | `AbstractCode`, `ClassicalCode`, `QuditCode`, and `CSSCode`; logicals, stabilizers, distance, concatenation, and error-rate interfaces | [`common_test.py`](src/qldpc/codes/common_test.py), [`monte_carlo_test.py`](src/qldpc/codes/monte_carlo_test.py) |
| [`src/qldpc/codes/classical.py`](src/qldpc/codes/classical.py) | Classical code families | [`classical_test.py`](src/qldpc/codes/classical_test.py), [`basics.ipynb`](examples/basics.ipynb) |
| [`src/qldpc/codes/quantum.py`](src/qldpc/codes/quantum.py) | Quantum, CSS, subsystem, product, and geometric code families | [`quantum_test.py`](src/qldpc/codes/quantum_test.py), [`bivariate_bicycle_codes.ipynb`](examples/bivariate_bicycle_codes.ipynb) |
| [`src/qldpc/codes/distance.py`](src/qldpc/codes/distance.py) | Exact binary classical and quantum distance enumeration | [`distance_test.py`](src/qldpc/codes/distance_test.py) |
| [`src/qldpc/abstract/`](src/qldpc/abstract/) | Groups, group rings, `RingArray`, semisimple linear algebra, and Wedderburn--Artin transforms | Co-located `*_test.py` files in the same directory |
| [`src/qldpc/math.py`](src/qldpc/math.py) | Symplectic and finite-field array helpers | [`math_test.py`](src/qldpc/math_test.py) |
| [`src/qldpc/objects.py`](src/qldpc/objects.py) | Pauli labels, graph nodes, Cayley complexes, and chain complexes | [`objects_test.py`](src/qldpc/objects_test.py) |
| [`src/qldpc/decoders/`](src/qldpc/decoders/) | Decoder protocol/adapters, implementations, DEM arrays, retrieval, Sinter, and windowed decoding | Co-located tests plus [`logical_error_rates/`](examples/logical_error_rates/) |
| [`src/qldpc/circuits/`](src/qldpc/circuits/) | Stim circuits, bookkeeping, encoders, memory experiments, noise, benchmarking, and transversal operations | Co-located tests plus [`noise_models.ipynb`](examples/noise_models.ipynb) and [`transversal_gates.ipynb`](examples/transversal_gates.ipynb) |
| [`src/qldpc/external/`](src/qldpc/external/) | GAP, GUAVA, QDistRnd, GroupNames, and code-database integrations | Co-located tests use controlled substitutes for processes, input, and network access |
| [`src/qldpc/cache.py`](src/qldpc/cache.py) | Persistent disk-cache helpers for expensive computations | [`cache_test.py`](src/qldpc/cache_test.py) |
| [`src/qldpc/experimental/`](src/qldpc/experimental/) | Unstable research implementations | Co-located tests and [`examples/experimental/`](examples/experimental/) |
| [`examples/`](examples/) | Canonical executable notebooks and small helper scripts | Sphinx links to these files; do not edit generated or duplicate notebook copies |
| [`docs/source/`](docs/source/) | Sphinx pages and the library map | Strict build through [`checks/build_docs.py`](checks/build_docs.py) |
| [`checks/`](checks/) | Local wrappers for the repository's quality gates | Mirrors the commands used by CI |

The main code hierarchy is:

```text
AbstractCode
|- ClassicalCode
`- QuditCode
   `- CSSCode
```

The methods in `codes/common.py` share cached and mutable state for standard form, logical and gauge operators, parameters, and code transformations.
Do not split them into mixins merely to reduce the file length.
Free-standing Monte Carlo helpers already live in `codes/monte_carlo.py`.

## Core invariants

### Fields and arrays

- Code arithmetic happens in a `galois.FieldArray`.
  The default field is `GF(2)`.
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
- Built-in families may cache construction- or literature-supplied parameters.
  A test that compares `get_code_params()` only with those same cached values is tautological; use an independent rank, commutation, row-space, or bounded-distance oracle.
- [`codes/distance.py`](src/qldpc/codes/distance.py) is explicitly binary.
  Use field-aware code methods and oracles for nonbinary constructions.

### Decoders

- [`decoders.get_decoder`](src/qldpc/decoders/retrieval.py) defaults to GUF for a nonbinary `FieldArray` and BP+OSD otherwise.
- Configure named error decoders with typed helpers such as `decoders.bp_lsd(...)`, and pass the resulting `DecoderSpec` as `decoder=`.
  A one-argument custom constructor is also accepted.
- A prebuilt error decoder is accepted only where the caller knows the matrix being decoded.
  Reject it with `retrieval._reject_prebuilt_decoder(decoder, reason)` wherever a method decodes a matrix that it constructs, such as an effective check matrix, a window, or a simplified detector error model.
- Keep error decoders (`ErrorDecoder`: `decode`, syndrome -> inferred error) distinct from observable decoders (`ObservableDecoder`: `decode_observables`, syndrome -> observable flips).
  A decoder may be both, like `RelayBPDecoder`, but a `decode` method must never return observable flips.
  Validate the output length of an error decoder against the matrix or detector error model it decodes, as `retrieval._match_error_decoder_to_dem` does.
- `SinterDecoder` is the Sinter-facing observable decoder.
  It uses a native observable decoder when a `DecoderSpec` supports one (`DecoderSpec.predicts_observables_natively`), and otherwise converts inferred errors into observable flips; window decoders always need error decoders.
- Keep each typed helper's options and defaults in sync with the decoder it configures; `retrieval_test.py` checks this.
- Only decoders that declare erasure support may append an erasure flag.
  They append that flag as the last entry of each inferred error; unsupported decoders must reject `add_erasure_bit=True`.
- Detector-error-model decomposition indices and remaps must remain valid after cancellation and simplification.
  Test malformed components, not only happy-path Stim models.
- Sliding-window time inference is heuristic.
  Preserve the first nondecreasing detector coordinate convention, and use an explicit `detector_to_time` mapping when a model follows another layout.

### Circuits

- Circuit and tableau constructors are qubit-only and should use the existing `restrict_to_qubits` guard.
- `get_encoding_circuit` constructs a valid encoder but is not fault-tolerant.
  Do not present it as a fault-tolerant state-preparation procedure.
- Transversal-gate search enumerates automorphisms and can be exponential.
- Keep data, check, reference, and ancilla roles explicit through `QubitIDs` and bookkeeping types.
  Do not infer roles from coincident integer indices.
- qLDPC memory-circuit detectors use coordinates such as `(round, 0, check_index)`.
  A later monotone coordinate can enumerate checks rather than time.
- Noise-model operation immunity and qubit immunity are separate controls; preserve both.

### External systems, caching, and tests

- GAP helpers may start subprocesses, read standard input, use the clipboard, access the network, or clone missing packages.
  Keep these side effects explicit in docstrings and error messages.
- Tests run with network sockets disabled.
  Mock optional web, subprocess, clipboard, and prompt paths; do not add a live-service test.
- Disk caches are bypassed while pytest is imported.
  Tests must not depend on cache persistence.
- Expensive reusable computations should use `qldpc.cache.use_disk_cache()`.
  Do not hide errors by returning a default value that looks successful.
- Statement coverage is gated at 100%.
  Co-located `*_test.py` files are the strong convention, although a thin helper can be covered through its consumer's test module.

## Common change recipes

### Add or change a code construction

1. Put a classical family in [`codes/classical.py`](src/qldpc/codes/classical.py) and a quantum or CSS family in [`codes/quantum.py`](src/qldpc/codes/quantum.py), next to its closest base or sibling.
2. Re-export public names from [`codes/__init__.py`](src/qldpc/codes/__init__.py).
3. Add a linked literature reference and document the field, subsystem, distance, and validation assumptions in the class docstring.
4. Add co-located tests with independent invariants: rank-derived dimension, CSS orthogonality, symplectic commutation, row-space identities, known small distances, or exact cross-family equivalence.
5. Verify a regression test has teeth by checking that it fails when the intended mechanism is removed or perturbed.

### Change code-core algebra

1. Read the cached-state interactions in [`codes/common.py`](src/qldpc/codes/common.py) before changing standard form, logicals, stabilizers, gauges, dimension, or distance.
2. Test both CSS and non-CSS paths, stabilizer and subsystem paths, `k=0`, and at least one odd characteristic when signs matter.
3. Preserve public imports and invalidate or transfer cached values deliberately.

### Add or adapt a decoder

1. Implement the protocol in [`decoders/custom.py`](src/qldpc/decoders/custom.py) or the relevant adapter module.
2. Add retrieval wiring in [`decoders/retrieval.py`](src/qldpc/decoders/retrieval.py) and exports in [`decoders/__init__.py`](src/qldpc/decoders/__init__.py).
3. Decide and test batch behavior, nonbinary support, detector-error-model support, and erasure signaling explicitly.
4. Use direct syndrome/error reproductions in addition to factory-selection tests.

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
3. Keep README and the library map selective.
   AutoAPI provides the complete list of public symbols.
4. When a public limitation changes, update the relevant function or class docstring and every README or guide that repeats it in the same pull request.
5. Run the strict documentation build before considering the change complete.

## Validation commands

Install the development environment:

```bash
python -m pip install -e '.[dev]'
```

Use the smallest targeted command while iterating, then the full gate before merging:

| Command | Purpose |
| --- | --- |
| `python checks/all_.py` | Complete repository gate: formatting, lint, strict mypy, tests, 100% coverage, and docs |
| `python checks/pytest_.py` | Full pytest and notebook test suite |
| `python checks/pytest_.py src/qldpc/codes/` | Tests for one package |
| `python -m pytest src/qldpc/codes/quantum_test.py::test_name -v` | One focused test |
| `python checks/format_.py --check` | Verify Ruff and `pyproject.toml` formatting |
| `python checks/format_.py` | Apply repository formatting |
| `python checks/lint_.py` | Ruff lint |
| `python checks/mypy_.py` | Strict mypy over source and tests |
| `python checks/coverage_.py` | Modular 100% statement-coverage gate |
| `python checks/build_docs.py` | Strict Sphinx and notebook documentation build |

Some check wrappers discover files through Git.
Add new source and test files to the index before relying on the full gate to include them.
Do not weaken a failure with `noqa`, `type: ignore`, or coverage exclusions unless the exceptional condition is real and documented.
