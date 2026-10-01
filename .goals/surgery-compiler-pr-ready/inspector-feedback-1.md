# Inspector verdict — iteration 1

## Verdict: FAIL

The implementation and branch integration are currently sound in the exercised cases, and the
complete repository gate passes. However, two acceptance-criterion regressions are not pinned by
meaningful tests. The branch is not yet independently merge-ready because both gaps permit the
specific semantics called out by the goal to regress while the committed tests remain green.

## Blocking findings

### 1. The HGP nonzero-logical test does not verify logical-index resolution

`test_compile_interblock_pair_hgp_nonzero_logical` requests logical indices 3 and 5, but only checks
the merged dimension, source-data count, round count, and noiseless detectors. All four assertions
also pass when the request is changed to indices 0 and 0:

- selected and zero-index requests both report `(dimension, source_data_qubits, syndrome_rounds) =
  (31, 116, 1)`;
- both produce no detector events in the test's two noiseless shots;
- the circuits and resolved operators are nevertheless different.

The production result already exposes `resolved_operators`, so the test should compare each entry
directly with `code.get_logical_ops(Pauli.X)[3]` and `[5]` (and preferably establish that they differ
from index 0). As written, replacing both indexed lookups in `_compiler.py` with `[0]` would leave
this regression green. This does not meet the goal's requirement for meaningful HGP nonzero
logical-index regression coverage.

### 2. Structural intra-code rejection is tested only with different-sized codes

`test_explicit_intracode_structure_uses_code_equality_not_identity` correctly verifies acceptance of
two distinct but equal Steane objects. Its rejection fixture compares Steane with `SurfaceCode(3)`,
whose matrix dimensions and data-qubit count differ. Consequently, a weakened check that compared
only sizes/shapes would satisfy the committed test without validating code structure.

Add a rejection fixture with the same field and matrix shapes/data count but unequal check matrices
(for example, a consistently column-permuted Steane code). The current implementation correctly
rejects such a probe with `ValueError("...structurally identical...")`; the missing piece is a
regression that pins that behavior.

## Independent verification performed

### Branch, ancestry, scope, and repository state

- `HEAD`: `07dd61f327cbf704436622d00f9f737d587e1019`
- `origin/main`: `4b223db4de4754e16001b69d983e4c3fe285415f`
- merge base equals `origin/main`; divergence is `0 behind / 1 ahead`.
- `origin/main` is an ancestor of `HEAD`.
- No merge, rebase, cherry-pick, or revert state is present.
- No remote-tracking ref contains the Builder commit.
- The production diff against `origin/main` is limited to:
  - `src/qldpc/experimental/surgery/_compiler.py`
  - `src/qldpc/experimental/surgery/_compiler_test.py`
  - `src/qldpc/experimental/surgery/circuit.py`
  - `src/qldpc/experimental/surgery/circuit_test.py`
- The only other changed paths are goal-process artifacts.
- Both new compiler files are tracked and visible to Git discovery.
- `git diff --check origin/main...HEAD` and repository consistency checks pass.
- There are no non-ignored untracked files. Ignored development/build caches are present, but no
  generated artifact is included in the diff.

### Implementation and semantic probes

- Raw Stim measurement-record reconstruction passes all eight Steane `XX`/`ZZ` parity truth-table
  cases, including odd parity. Diagnostic observable 0 equals the direct destructive observable 1.
- Explicit block identity works when both disjoint blocks refer to the same `CSSCode` object.
- HGP indices 3 and 5 resolve to the corresponding distinct logical operator vectors in the current
  implementation.
- A same-sized, column-permuted Steane fixture is rejected by the explicit intra-code structural
  path.
- Static resource identities hold for source data, gadget ancillas, adapter data, syndrome
  ancillas, peak physical qubits, rounds, and one logical outcome.
- The public export tuple remains exactly the same ten names.
- AST signature comparison against `origin/main` reports all four public circuit signatures
  unchanged.

### Legacy byte/hash compatibility

I executed the `origin/main` circuit module in memory and compared UTF-8 Stim circuit text and
merged check matrices against the current public builders using identical fixtures:

| Fixture | SHA-256 of circuit text | Result |
| --- | --- | --- |
| Single Steane X, 2 rounds | `fb55828675f4bb390d546ff37acbee04c53019a94c934b7bb029790ccb0e4b98` | identical |
| Inter-code Steane Z×Z, 2 rounds | `9337fb3a713d3875ffd214823015709f79a089e1f62abb909c79b2bdbc550092` | identical |
| Intra-code Webster X×X, 1 round | `2bb2b11bed867faa70f6495aaecf41f8d9e2bac7299bbb4c19a233deacf0648d` | identical |

### Quality gates

- Focused compiler/seam run: **18 passed**.
- `.venv/bin/python checks/all_.py`: **passed** with exit code 0, including formatting, lint,
  strict mypy, tests/coverage, notebooks, and documentation checks.

## Required next iteration

Add only the two focused regression assertions/fixtures described above, rerun the targeted tests
and `python checks/all_.py`, and preserve the otherwise clean implementation and scope.
