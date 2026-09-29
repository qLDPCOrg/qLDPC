# Iteration 1 inspection: FAIL

## Verdict

The branch is not ready for a pull request. The advertised GAP extra cannot activate the new
backend with the dependency version selected by the branch, Python 3.14 receives no GAP
dependency, and the required full quality gate fails.

## Blocking findings

### 1. The pinned distribution predates the imported module

`pyproject.toml` selects `passagemath-gap>=10.5.49,<10.6`, while `_get_libgap()` imports
`passagemath_gap`.

Those two choices are incompatible:

- PyPI published `passagemath-gap` 10.5.49 on 2025-11-01. Its own example imports
  `sage.all__sagemath_gap`, and its wheel metadata supports Python 3.9 through 3.13.
- Upstream passagemath added the top-level `passagemath_*` import packages in commit
  `60c700d4f76c9390150944b1d09218782217d4d4` on 2025-11-16, after 10.5.49 was published.
- The stable API already present in the selected release is
  `from sage.libs.gap.libgap import libgap`.

Consequently, installing the declared extra can still leave `_get_libgap()` returning `None`
after catching `ImportError`. qLDPC then falls back to the command-line/manual paths and does not
provide the direct backend the goal requires. The tests miss this because they patch
`_get_libgap()` or provide an artificial `passagemath_gap` module rather than importing the API
from the selected distribution.

This blocks the direct-backend, maintained-installation-path, dependency/API-compatibility, and
PR-readiness criteria.

### 2. The extra is empty on a supported Python version

qLDPC declares Python 3.10 through 3.14 support, but the dependency has the marker
`python_version<'3.14'`. On Python 3.14, `pip install 'qldpc[gap]'` therefore installs no GAP
binding at all. Current passagemath 10.8 releases advertise Python 3.11 through 3.14 support,
while its 10.6 line retains Python 3.10 support. The metadata and documentation neither provide
a version split nor disclose that the documented extra does nothing on 3.14.

### 3. Required quality gates do not pass

Using only `/Users/D648438/src/qLDPC-gap/.venv`:

- `.venv/bin/python checks/all_.py` **failed** in strict mypy.
- A direct rerun of `.venv/bin/python checks/mypy_.py` reported **150 errors in 29 files**, not
  the Builder-reported 68.
- None of those 150 diagnostics is in either changed Python file (`external/gap.py` or
  `external/gap_test.py`). All reported files are unchanged from the initial SHA, and the mypy
  configuration is unchanged, so this iteration did not introduce those diagnostics. This does
  not satisfy the explicit full-gate criterion, however.
- `.venv/bin/python checks/coverage_.py` also **failed** at 99.69% (54 missed statements), all in
  unchanged code. The goal requires 100%.

The higher reproduced mypy count appears environment/dependency-sensitive; the inspected
environment used mypy 1.7.1, numpy 1.26.4, scipy 1.14.1, pytest 9.1.1, and checks-superstaq
0.5.67. Regardless of whether these failures predate the implementation, the immutable goal
explicitly requires a passing full gate and a PR-ready branch, so they cannot be waived.

### 4. Direct-backend package behavior is incomplete

When libgap is present but a package is absent, `require_package()` now always raises
`ModuleNotFoundError`. It never uses the retained repository installation path and never falls
back to a callable GAP executable. This particularly affects the custom QDistRnd repository used
by `external.codes`: the `repo` argument becomes message-only under libgap. The goal explicitly
requires package discovery/loading and supported installation behavior for packages used by the
group/code helpers. Either the embedded GAP package path/install mechanism must be supported, or
the backend-selection/fallback semantics and limitation must be redesigned and tested.

## Acceptance-criterion assessment

| Criterion | Result | Evidence |
| --- | --- | --- |
| Dedicated `gap` branch from recorded `main` SHA | PASS | `merge-base(HEAD, main)` and `main` are both `40a571e…`; branch is `gap`. |
| Direct libgap execution without spawning `gap` | FAIL | Selected release does not provide the imported top-level module; no real-binding test. |
| No `gappy-system`; supported maintained install | FAIL | No `gappy-system` reference was added, but the declared extra/backend API pairing is unusable. |
| Public signatures/parsing/cache/field/error compatibility | PARTIAL | Signatures and legacy fallback code are retained; external parsing tests pass, but the real direct backend was not exercised. |
| Binding-absent manual/subprocess fallback and explicit errors | PASS | Existing paths remain and targeted tests cover their success/error behavior. |
| GAP package discovery/loading/install behavior | FAIL | Direct mode only checks availability and disables the repository install path/fallback. |
| Packaging metadata and user documentation | FAIL | Metadata excludes 3.14 and pins a release incompatible with the import; docs promise `qldpc[gap]` without this limitation. |
| No-network direct/fallback/downstream tests | PARTIAL | 19 external tests pass with sockets disabled, but all direct tests use mocks that conceal the actual import incompatibility. |
| Targeted tests, format, lint, mypy, full gate | FAIL | Targeted external tests, format, lint, and docs pass; mypy, coverage, and `checks/all_.py` fail. |
| Intentional clean PR-ready branch | FAIL | Scope is otherwise focused, but functional and gate blockers remain. |

## Passing checks

- `.venv/bin/python checks/pytest_.py src/qldpc/external/`: 19 passed.
- `.venv/bin/python checks/format_.py --check`: passed.
- `.venv/bin/python checks/lint_.py`: passed.
- `.venv/bin/python checks/build_docs.py`: passed.
- `git diff --check`: passed.
- No implementation files were modified during inspection.

## Required next iteration

1. Select a passagemath version strategy that covers qLDPC's Python 3.10–3.14 range and import
   the API actually exposed by every selected release (prefer the stable
   `sage.libs.gap.libgap.libgap` path unless verified otherwise).
2. Add an integration test against the installed optional dependency, or at minimum a faithful
   import-contract test that would fail with the declared 10.5.49 distribution.
3. Define and test package loading/installation or a deliberate executable fallback for
   QDistRnd and the other required packages.
4. Make the documented extra truthful on every supported Python version.
5. Resolve the repository gate/environment baseline so mypy, 100% coverage, and
   `.venv/bin/python checks/all_.py` genuinely pass before requesting another inspection.
