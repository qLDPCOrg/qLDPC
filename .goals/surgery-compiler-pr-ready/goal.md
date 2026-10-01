# Goal: Prepare the surgery compiler slice PR

## User Request

Determine whether `feat/surgery-compiler-slice` is ready to merge into `main`. If it is not,
make it merge/PR-ready, then provide a GitHub title and high-level summary that can be copied and
pasted.

## Refined Goal

Prepare `feat/surgery-compiler-slice` as a focused, current, independently verified pull-request
candidate against `origin/main`. Preserve the intended private compiler vertical slice and legacy
public surgery behavior, resolve any integration issues introduced by the 30 newer mainline commits,
run the repository's complete quality gate, and commit the finished branch without pushing it.

## Acceptance Criteria

- [ ] The branch contains current `origin/main` and has no unresolved merge or rebase state.
- [ ] The production diff is limited to the private compiler slice in
      `src/qldpc/experimental/surgery/_compiler.py`, its co-located tests, and the minimal explicit
      structural seam and regressions in `circuit.py` and `circuit_test.py`; goal-process artifacts
      under `.goals/surgery-compiler-pr-ready/` are the only additional files.
- [ ] The private compiler lowers one same-basis inter-block logical `XX` or `ZZ` pair measurement
      through the existing Webster/universal-adapter backend with explicit block identity,
      initialization, resolved logical operators, result metadata, backend metadata, and exact
      static resource accounting.
- [ ] The ten-name public `qldpc.experimental.surgery` export remains unchanged, public function
      signatures remain compatible, and existing public circuit output remains unchanged for
      representative single-, inter-code, and intra-code fixtures.
- [ ] Steane raw parity truth tables, same-code-object/disjoint-block behavior, HGP nonzero logical
      indices, validation failures, diagnostic observables, and resource invariants have meaningful
      regression coverage.
- [ ] `python checks/all_.py` passes in the declared development environment after all intended new
      files are included in Git discovery.
- [ ] The final worktree has no unrelated or generated artifact litter, and the implementation is
      committed locally on `feat/surgery-compiler-slice`; nothing is pushed.
- [ ] A fresh independent inspector reports PASS with no unresolved high-confidence correctness,
      compatibility, type-safety, testing, or scope defects.

## Scope Boundaries

**In scope:**
- Synchronizing the feature branch with current `origin/main`.
- Fixing conflicts or regressions directly caused by that synchronization or the compiler slice.
- Private logical-block and logical-Pauli request/result records.
- One complete same-basis inter-block pair-measurement lowerer.
- Exact static resource and allocation metadata.
- Focused and full-repository validation.
- A local feature commit suitable for opening a pull request.

**Out of scope:**
- Public compiler exports or a stable public compiler IR.
- Teleportation, injection, feed-forward, batching, or scheduling.
- Intra-block compiler products, mixed-basis products, `Y`, or products with more than two factors.
- A second backend or backend selector.
- Claims of fast, parallel, amortized, distance-preserving, or fault-tolerant compilation beyond
  what the existing surgery backend establishes.
- Changes to the separate `surgery-review` research worktree.
- Pushing, opening the pull request, modifying `main`, or deleting worktrees, branches, or refs.

## Applicable Project Conventions

**Quality gate command:**
- `python checks/all_.py`

**Commit convention:**
- Recent repository commits use concise imperative, sentence-case summaries.
- Goal iterations use `type(scope): [B/I] description` with the required role marker.
- Assisted-by trailer required: `Assisted-by: OpenAI:GPT-5.6 Luna` for Builder commits and
  `Assisted-by: OpenAI:GPT-5.6 Sol` for Inspector commits.
- Repository-required co-author trailer:
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

**Guidelines:**
- `AGENTS.md`

**Rules:**
- Preserve ordinary `qldpc.*` public imports and explicit `__all__` contracts.
- Keep compiler APIs private until semantics and a second backend justify publication.
- Use co-located `*_test.py` files and independent algebraic or circuit-semantic invariants.
- Maintain 100% statement coverage.
- New files must be included in Git discovery before relying on modular quality gates.
- Preserve field and logical-operator semantics; do not infer logical block identity from Python
  object identity.
- Use the smallest targeted checks while iterating, then run the full gate before completion.
- Do not edit generated notebook copies or introduce generated artifacts.
