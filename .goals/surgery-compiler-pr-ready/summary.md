# Surgery compiler PR-readiness summary

## Outcome

`feat/surgery-compiler-slice` is synchronized with `origin/main`, fully validated, and independently
approved as a pull-request candidate.

## Acceptance criteria

- Current `origin/main` is an ancestor of the feature branch; the branch is not behind and no Git
  operation is in progress.
- The production diff is confined to the private compiler, its tests, and the minimal circuit
  structural seam and regressions.
- The private lowerer compiles one same-basis inter-block logical `XX` or `ZZ` measurement through
  the Webster universal-adapter backend.
- Explicit block identity, logical initialization and resolution, result and diagnostic metadata,
  backend metadata, physical allocation, and exact static resource counts are retained.
- The ten-name public surgery export and public circuit-builder signatures are unchanged.
- Representative public single-, inter-code, and intra-code circuits are byte-identical to
  `origin/main`.
- Steane parity semantics, same-code-object disjoint blocks, HGP indices 3 and 5, diagnostics,
  validation, structural equality, and resource invariants have direct regression coverage.
- The complete `python checks/all_.py` gate passes, including formatting, Ruff, strict mypy, tests,
  100% coverage, notebooks, and documentation.
- No unrelated or generated artifact is included, nothing was pushed, and `main` was not modified.

## Iteration history

### Iteration 1 — FAIL

The implementation and complete quality gate passed, but the Inspector found that two tests did not
pin the intended semantics:

1. The HGP fixture requested logical indices 3 and 5 without asserting that the resolved operators
   were those indices.
2. The incompatible intra-code fixture used differently sized codes, so it did not prove that full
   check-matrix structure was compared.

### Iteration 2 — PASS

The Builder added direct resolved-operator assertions and a same-field, same-size, same-shape
column-permuted Steane rejection fixture. The Inspector applied in-memory weakened implementations
for both cases and confirmed that the strengthened tests fail under those mutations. Focused tests
and the complete repository gate passed again.

## Inspector findings resolved

- A lowerer that silently substituted logical index 0 is now caught.
- A structural check weakened to compare only field, dimensions, and shapes is now caught.
- No remaining high-confidence correctness, compatibility, type-safety, testing, or scope defect
  was reported.

## Recommendations

- Keep the compiler private until a second backend clarifies the stable abstraction.
- Use the next production slice to extract composable experiment parts through a concrete
  two-measurement logical-teleportation workload.
- Continue distinguishing raw logical eigenvalues, detector-observable flips, physical QEC rounds,
  batch latency, and amortized throughput.
- Preserve direct semantic or mutation-style checks when an aggregate dimension or noiseless
  detector assertion could pass for the wrong logical operator.
