# Iteration 2 inspection: FAIL

## Verdict

Iteration 1's dependency, Python-version, package-fallback, and quality-gate blockers are
substantially resolved. The branch is still not ready for a pull request because the new
import-contract test skips on the exact `ImportError` that it is meant to detect. In the required
worktree-local environment, `passagemath-gap` is absent, that test skipped, and the complete gate
therefore passed without importing or evaluating through the declared real distribution.

## Blocking finding

### The import-contract test still conceals an import-contract mismatch

`test_installed_libgap_import_contract()` does this:

```python
try:
    module = importlib.import_module("sage.libs.gap.libgap")
except (ImportError, OSError):
    pytest.skip("passagemath-gap is not installed")
```

This does not establish whether the distribution is installed. If `passagemath-gap` is installed
but no longer exposes `sage.libs.gap.libgap`, the test catches that contract failure and reports a
skip rather than a failure. This is precisely how iteration 1's incompatible import could have
remained hidden. The rest of the direct-backend tests patch `_get_libgap()` with `MockLibGap`, so
they also cannot detect a mismatch between the dependency metadata and the imported API.

The current `.venv` demonstrates the coverage gap:

- `importlib.metadata` reports `passagemath-gap` is not installed.
- The targeted external suite reports **19 passed, 1 skipped**.
- The skipped test is the only test that attempts the real import.
- The full quality gate also succeeds with that skip, so the declared optional dependency's actual
  import and `libgap.eval` API are not exercised by any required check.

The test should independently detect distribution presence (for example with
`importlib.metadata.version("passagemath-gap")`) and skip only when the distribution is absent.
When it is present, an import failure must fail. A faithful non-live contract test or a dedicated
extra-enabled CI environment should additionally prevent the normal gate from being entirely
mock-based. The declared release's real `eval` and package-query API should be exercised when the
extra is installed.

An attempt to install the declared extra into only this worktree's `.venv` could not proceed:
the configured package registry returned 403, and direct PyPI artifact access was also blocked
with 403. No shared or global environment was touched. This environmental restriction does not
excuse a test whose exception handling converts an installed-but-broken contract into a skip.

## Iteration 1 blocker verification

### Stable import/API against the declared release: source and metadata fixed; real execution unverified

- `pyproject.toml` now selects `passagemath-gap>=10.6.48,<10.7`.
- `_get_libgap()` now imports the established `sage.libs.gap.libgap` module and reads its `libgap`
  object, rather than importing the later `passagemath_gap` convenience module.
- PyPI metadata for the actual 10.6.48 release documents its Cython interface at
  `sage.libs.gap.libgap` and describes the 10.6 series as supporting Python 3.10--3.14.
- Source behavior uses the expected `libgap.eval` and `function_factory` interfaces.

The pairing is credible and correct by release metadata, but the local gate did not execute the
real binding. Thus the implementation blocker is fixed, while the verification blocker remains.

### Python 3.10--3.14 dependency support: resolved

The dependency marker `python_version<'3.15'` evaluates true for each of Python 3.10, 3.11, 3.12,
3.13, and 3.14. PyPI publishes 10.6.48 wheels for all five CPython versions on supported macOS,
manylinux, and musllinux architectures. The release metadata also explicitly lists those five
Python versions. The README's statement about the selected release line is truthful.

### GUAVA/QDistRnd and fallback behavior: resolved

- Package availability is queried directly through libgap.
- Packages already bundled with the binding, including GUAVA in the declared 10.6.48 wheel, stay
  on the direct backend.
- If GUAVA, QDistRnd, or another package is absent from libgap and a `gap` executable exists,
  `require_package()` records that package, checks/installs it through the executable path, and
  sends subsequent GAP commands to the executable so `LoadPackage` sees the installed package.
- The repository override for QDistRnd remains intact.
- A failed or declined executable installation removes the package from the fallback set and
  propagates an explicit error.
- When no executable is available, a missing embedded package raises `ModuleNotFoundError` with
  package-install guidance rather than returning a success-shaped value.

The generic implementation covers both GUAVA and QDistRnd. Unit tests exercise direct availability,
QDistRnd executable installation, absent-executable errors, package-query errors, and legacy
subprocess behavior. An independent mock probe also confirmed the same fallback for GUAVA.

### Complete quality gate: resolved

All checks were run from `/Users/D648438/src/qLDPC-gap/.venv`:

- `.venv/bin/python checks/pytest_.py src/qldpc/external/`: **19 passed, 1 skipped**.
- `.venv/bin/python checks/format_.py --check`: passed.
- `.venv/bin/python checks/lint_.py`: passed.
- `.venv/bin/python checks/mypy_.py`: passed with no issues in 103 source files.
- `.venv/bin/python checks/all_.py`: passed (706 collected tests plus formatting, lint, strict
  mypy, 100% coverage, and documentation checks).
- `git diff --check`: passed.

The previous baseline mypy and coverage failures are gone in the existing local environment. The
gate result is genuine, but it does not close the import-contract gap because the optional test
skips.

## Acceptance-criterion assessment

| Criterion | Result | Evidence |
| --- | --- | --- |
| Dedicated `gap` branch from recorded `main` SHA | PASS | Branch is `gap`; `main` and merge-base are both recorded SHA `40a571e…`. |
| Direct libgap execution without spawning `gap` | PARTIAL | Source prioritizes libgap and mock tests prove no subprocess call, but no real binding evaluation ran. |
| No `gappy-system`; supported maintained install | PASS | No obsolete dependency; declared stable 10.6 line has 3.10--3.14 wheels and the stable import path. |
| Public signatures/parsing/cache/field/error compatibility | PASS | Public signatures remain, downstream external tests pass, and fallback/cache/error paths remain explicit. |
| Binding-absent manual/subprocess fallback and explicit errors | PASS | Legacy fallback tests pass; missing embedded packages now fail explicitly or use the executable. |
| GAP package discovery/loading/install behavior | PASS | Direct package query plus executable fallback covers bundled GUAVA and external QDistRnd behavior. |
| Packaging metadata and user documentation | PASS | Extra and docs consistently describe 10.6, Python support, direct use, and fallback limitations. |
| No-network direct/fallback/downstream tests | FAIL | Behavioral mocks are comprehensive, but the sole real import test can skip an installed-but-incompatible API. |
| Targeted tests, format, lint, mypy, full gate | PASS | Every required command exits successfully in `.venv`; one optional import-contract test skips. |
| Intentional clean PR-ready branch | FAIL | Changes are focused and committed, but the remaining verification defect blocks PR readiness. |

## Required iteration 3

1. Make the import-contract test skip only when the `passagemath-gap` distribution itself is
   absent; if the distribution is present but its documented module or `libgap` object cannot be
   imported, the test must fail.
2. Ensure a required test validates the declared real API contract without allowing the normal
   all-mock gate to conceal a dependency/import mismatch. Prefer an extra-enabled CI job or a
   faithful contract test that is tied to the declared release metadata.
3. When the real extra is available, exercise at least a minimal direct evaluation and package
   availability query, not only object identity.
