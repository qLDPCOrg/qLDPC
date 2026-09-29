# Goal summary: Use libgap for GAP integration

## Outcome

The `gap` branch now provides an optional direct GAP backend through the maintained
`passagemath-gap` distribution while preserving qLDPC's executable and manual fallbacks.
Independent inspection passed after thirteen iterations, including live Python 3.14 provider
validation and test-isolation fixes that ensure mocked tests never call a system GAP executable.

## Acceptance criteria

- **Dedicated worktree and branch:** `/Users/D648438/src/qLDPC-gap` is on `gap`, based on recorded
  `main` SHA `40a571e455242e68e1a6f8e67ebaee310ad818b0`.
- **Direct libgap execution:** qLDPC initializes `sage.all__sagemath_gap`, imports the canonical
  `sage.libs.gap.libgap` API, and evaluates complete workloads as one valid
  `CallFuncList(function() ... end, [])` expression without spawning GAP.
- **Supported dependency:** the `gap` extra uses `passagemath-gap>=10.6.48,<10.7` on Python below
  3.15. The obsolete `gappy-system` dependency is not used.
- **Compatibility:** public GAP, group, and code signatures remain unchanged. Output parsing,
  caching, finite-field encoding, errors, and downstream helpers retain their behavior.
- **Fallbacks:** binding absence preserves executable and manual execution. Missing embedded
  packages use the executable installation path when available or raise an explicit error.
- **Package handling:** direct availability checks accept PassageMath's installed package paths as
  well as boolean-like responses. GUAVA and repository overrides such as QDistRnd remain supported.
- **Packaging and documentation:** dependency metadata, the README, API documentation, and library
  map explain installation, backend selection, fallbacks, and retained limitations.
- **Verification:** tests cover direct evaluation, errors, package handling, provider absence,
  executable/manual fallback, and downstream parsing. Mocked tests strictly prevent accidental
  subprocess use; the separate installed-provider contract exercises real libgap in CI.
- **Quality gates:** targeted tests and `.venv/bin/python checks/all_.py` pass from the
  worktree-local environment, including formatting, lint, strict mypy, 100% coverage, tests, and
  documentation.
- **Live CI:** Continuous Integration run `36584345483` passed at Builder SHA `c84aae77`, including
  17,789 statements with zero misses. Conditional Tests run `36584345500` passed on Python 3.10 and
  Python 3.14. The subsequent test-only guard commit was independently reviewed and passed the full
  local gate.
- **Branch readiness:** changes are scoped to issue #275 and the Goal audit record.

## Iteration history

1. **FAIL:** The initial dependency pin lacked the imported module, excluded Python 3.14, and did
   not preserve package-install behavior.
2. **FAIL:** Stable imports and package fallback passed, but the installed-provider contract could
   skip an incompatible import.
3. **FAIL:** Live dependency coverage exposed invalid multi-statement input to `libgap.eval`.
4. **PASS:** A single-expression wrapper fixed evaluation and all local gates passed.
5. **PASS:** CI-only optional-dependency coverage was corrected without hiding broken providers.
6. **FAIL:** A proposed convenience import was rejected because upstream marks it interactive-only.
7. **FAIL:** The initializer remained unsupported and provider-present fallback tests leaked.
8. **FAIL:** The supported import order passed locally but still required live Python 3.14 evidence.
9. **FAIL:** Live libgap imported successfully, but package availability returned an installed path
   and a fallback test reached a real executable.
10. **FAIL:** Provider tests passed after accepting package paths, but another provider-present test
    leaked into the executable path and reduced coverage to 99.89%.
11. **FAIL:** A rerun reproduced the same `FileNotFoundError`, identifying incomplete test isolation
    rather than a product coverage defect.
12. **FAIL:** The absence-path mock was corrected, but inspection requested strict subprocess guards
    for both backend branches.
13. **PASS:** Strict guards, direct and fallback behavior, provider contracts, all local gates, and
    exact-SHA live CI evidence were independently verified.

## Key issues resolved

- Adopted PassageMath's supported library initialization order instead of interactive-only imports.
- Covered qLDPC's supported Python versions with explicit dependency metadata.
- Honored libgap's single-command contract while retaining assignments, loops, and repeated output.
- Preserved GUAVA and QDistRnd behavior across direct and executable backends.
- Accepted PassageMath's non-empty installed package paths as successful availability responses.
- Kept installed-but-unimportable providers as hard failures rather than skips.
- Made fallback tests provider-independent and added strict guards against real GAP subprocess calls.
- Reproduced and resolved Python 3.14 CI failures without lowering the 100% coverage threshold.
- Performed all local setup and validation in the dedicated worktree-local virtual environment.

## Recommendations

- Monitor `passagemath-gap` releases and revisit the `<10.7` upper bound only after running the live
  adapter contract against the newer release line.
- Keep the real-extra CI path required; mock-only coverage did not detect the import, evaluation, or
  package-return-value differences.
- Retain strict subprocess guards in mocked tests so provider availability cannot silently change
  which backend a test exercises.
- Consider a platform matrix for the optional GAP backend if support expands beyond Linux CI.

## Suggested squash command

```bash
git reset --soft 40a571e455242e68e1a6f8e67ebaee310ad818b0
git commit -m 'feat(external): use libgap for GAP integration

Users can enable direct GAP execution with the supported gap extra while retaining existing
fallback behavior when the binding or embedded packages are unavailable.

Assisted-by: OpenAI:GPT-5.6 Luna

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>'
```
