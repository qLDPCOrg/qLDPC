# Goal summary: Use libgap for GAP integration

## Outcome

The `gap` branch now provides an optional direct GAP backend through the maintained
`passagemath-gap` distribution and preserves qLDPC's existing subprocess and manual fallbacks.
Independent inspection passed after five Builder/Inspector iterations, including a follow-up fix
for environment-dependent CI coverage, and the branch is ready for its pull request into `main`.

## Acceptance criteria

- **Dedicated worktree and branch:** `/Users/D648438/src/qLDPC-gap` is on `gap`, based on recorded
  `main` SHA `40a571e455242e68e1a6f8e67ebaee310ad818b0`.
- **Direct libgap execution:** qLDPC prefers `sage.libs.gap.libgap.libgap` and evaluates complete
  workloads as one valid `CallFuncList(function() ... end, [])` expression without spawning GAP.
- **Supported dependency:** the `gap` extra uses `passagemath-gap` 10.6, which provides wheels for
  Python 3.10 through 3.14; `gappy-system` is not used.
- **Compatibility:** public GAP, group, and code signatures remain unchanged. Output parsing,
  caching, finite-field encoding, errors, and downstream helpers retain their behavior.
- **Fallbacks:** binding absence preserves subprocess/manual execution. A package unavailable in
  embedded GAP deliberately uses the executable installation path or raises an explicit error.
- **Package handling:** direct availability checks cover bundled packages such as GUAVA; repository
  overrides such as QDistRnd remain supported by the executable fallback.
- **Packaging and documentation:** dependency metadata, the README, API documentation, and library
  map explain installation, backend selection, and retained limitations.
- **Verification:** mock tests reject the malformed multi-statement regression. The CI coverage job
  installs the real `gap` extra and tests raw libgap evaluation, package queries, and qLDPC's actual
  direct adapter.
- **Quality gates:** targeted external tests and `python checks/all_.py` pass from the worktree-local
  Python 3.14 virtual environment, including formatting, lint, strict mypy, 100% coverage, tests,
  and documentation.
- **Branch readiness:** changes are scoped to issue #275 and the Goal audit record.

## Iteration history

1. **FAIL:** The initial dependency pin predated the imported convenience module, excluded Python
   3.14, and did not preserve package-install behavior. The local environment also did not match CI.
2. **FAIL:** Stable imports, supported markers, package fallback, and full gates were fixed, but the
   installed dependency contract could still skip an incompatible import.
3. **FAIL:** CI began installing and exercising the real dependency, revealing that qLDPC passed
   invalid multi-statement input to the single-command `libgap.eval` API.
4. **PASS:** The adapter wrapped complete programs as one valid GAP expression, strengthened the
   mock regression, exercised qLDPC's real adapter in CI, and passed all quality gates.
5. **PASS:** The first live-extra CI run exposed an environment-dependent uncovered skip path. The
   optional absence branch was excluded from coverage without allowing installed binding failures
   to skip, and the full 100% coverage gate passed.

## Key issues resolved

- Selected the stable `sage.libs.gap.libgap` import exposed by the declared release line.
- Covered every qLDPC-supported Python version with truthful dependency metadata.
- Preserved GUAVA and QDistRnd package behavior across direct and executable backends.
- Prevented optional-dependency tests from hiding installed-but-broken imports.
- Corrected direct evaluation to honor libgap's single-command contract while supporting qLDPC's
  split loops, assignments, and repeated output.
- Kept the optional-dependency test coverage-stable in both installed and absent environments while
  continuing to fail on an installed-but-unimportable binding.
- Recreated validation in a worktree-local environment matching repository CI.

## Recommendations

- Monitor `passagemath-gap` releases and revisit the `<10.7` upper bound only after running the live
  adapter contract test against the newer line.
- Keep the real-extra CI path required; mock-only coverage did not detect the original import and
  evaluation-contract defects.
- Consider adding a separate platform matrix for the optional GAP backend if future changes expand
  beyond the current Linux CI coverage.

## Suggested squash command

```bash
git reset --soft 40a571e455242e68e1a6f8e67ebaee310ad818b0
git commit -m 'feat(external): use libgap for GAP integration

Users can enable direct GAP execution with the supported gap extra while retaining existing
fallback behavior when the binding or embedded packages are unavailable.

Assisted-by: OpenAI:GPT-5.6 Luna

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>'
```
