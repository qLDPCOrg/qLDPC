# Inspector verdict — iteration 2

## Verdict: PASS

The two iteration-1 blockers are now pinned by focused regressions, including direct mutation
probes of the weakened implementations described in the prior feedback. The complete branch remains
focused and compatible, and the full repository gate passes. I found no unresolved
high-confidence correctness, compatibility, type-safety, testing, or scope defect.

## Iteration-1 blocker verification

### 1. HGP nonzero logical indices are pinned

`test_compile_interblock_pair_hgp_nonzero_logical` now compares
`compiled.resolved_operators` directly with logical X operators 3 and 5 and establishes that both
differ from operator 0.

I loaded an in-memory source mutant that replaced both indexed logical lookups with index 0. The
new resolved-operator assertion failed. Thus the regression catches the exact weakening identified
in iteration 1.

### 2. Same-sized unequal code structures are pinned

`test_explicit_intracode_structure_uses_code_equality_not_identity` now constructs a
column-permuted Steane code with the same field object, check-matrix shapes, and data-qubit count,
while confirming that its structure differs. It requires the explicit intra-code path to reject the
pair with the structural-identity error.

I loaded an in-memory source mutant that accepted equal field, shapes, and data-qubit counts without
comparing matrix contents. That mutant accepted the fixture, so the new `pytest.raises` assertion
failed as required.

## Acceptance-criterion verification

- `origin/main` at `4b223db4de4754e16001b69d983e4c3fe285415f` is the merge base and an
  ancestor of Builder HEAD `1b2cc25744c00cb8e537435b5cff52b4964995bf`; divergence before this
  Inspector commit is `0 behind / 3 ahead`. No merge, rebase, cherry-pick, or revert state exists.
- The production diff is limited to `_compiler.py`, `_compiler_test.py`, `circuit.py`, and
  `circuit_test.py`. All other changed paths are goal-process artifacts. Both compiler files are
  tracked, and `git diff --check origin/main...HEAD` passes.
- The private lowerer retains explicit block identities, initialization, selected logical
  operators, named result and diagnostic metadata, backend metadata, and exact static resource
  identities. Steane `XX`/`ZZ` raw parity truth tables, diagnostic observables, same-code-object
  disjoint blocks, HGP indices, validation failures, and resource invariants remain covered.
- The public export list remains the same ten names. AST comparison shows all four public circuit
  function signatures unchanged.
- Executing the `origin/main` circuit module in memory produced byte-identical circuit text and
  identical merged check matrices for representative single Steane X, inter-code Steane Z×Z, and
  intra-code Webster X×X fixtures.
- The focused compiler and circuit test run passes: **137 passed**.
- `.venv/bin/python checks/all_.py` passes with exit code 0, including formatting, lint, strict
  mypy, tests and coverage, notebooks, and documentation.
- No non-ignored untracked files or unrelated tracked artifacts are present. No remote-tracking ref
  contains the Builder HEAD.

## Copy-ready pull-request framing

**Title:** Add a private lattice-surgery pair compiler slice

**Summary:**

- Lower one same-basis inter-block logical pair measurement through the existing Webster universal
  adapter with explicit block and result metadata.
- Add exact resource accounting and a private structural seam while preserving legacy public
  surgery behavior.
- Cover parity semantics, logical-index resolution, block structure, diagnostics, validation, and
  resource invariants with focused regressions.
