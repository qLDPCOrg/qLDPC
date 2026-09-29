# Iteration 8 inspection: FAIL

## Verdict

Iteration 8 resolves the two code-level findings from iteration 7. The new import order is an
authoritative PassageMath library workaround, and the fallback tests are now independent of an
installed provider. However, the required live Python 3.14 `passagemath-gap` run has not been
performed for `dc227ad6`; the local gate skips the real-provider contract, while the latest remote
coverage job still tests the older direct import and fails. The CI circular-import fix therefore
remains unverified against the binary provider.

## Authoritative import contract

The sequence is supported for non-interactive library code:

```python
import sage.all__sagemath_gap
from sage.libs.gap.libgap import libgap
```

- PassageMath commit `f35b8847b4226074d84c4645ea2c6eacab69e70b` is titled
  `src/sage/features/gap.py: Work around import cycle` and adds exactly the first import before the
  specific `libgap` import.
- This is inside `sage.features.gap.GapPackage._is_present`, an ordinary library feature method,
  not an interactive aggregate namespace or README example.
- The exact 10.6.48 release commit
  `788ff01f07b768f6b0226718e43a0914f0b04f2d` contains that sequence.
- PyPI currently has exactly one release in qLDPC's declared
  `passagemath-gap>=10.6.48,<10.7` range: 10.6.48. Thus the complete presently resolvable range
  contains the workaround.
- qLDPC now reproduces the upstream order with `importlib.import_module`, then consumes the object
  from the specific `sage.libs.gap.libgap` module. It no longer consumes the interactive-only
  `passagemath_gap` re-export.

This satisfies the support-contract concern from iterations 6 and 7.

## Dependency audit

- 10.6.48 metadata requires Python 3.10--3.14 and publishes CPython 3.10--3.14 macOS and Linux
  wheels.
- Its runtime metadata includes version-matched `passagemath-environment` and
  `passagemath-categories`; the former supplies `sage.features.gap`, and the latter supplies the
  aggregate's category initialization. The `passagemath-gap` manifest includes
  `sage.all__sagemath_gap` and `sage.libs.gap`.
- The prior Python 3.14 installation job `109411184585` successfully completed
  `uv sync --extra dev --extra gap`, independently confirming that the declared extra resolves and
  installs in CI.
- Neither metadata nor source adds or uses `gappy-system`.

## Remaining blocker: no live run of the fix

The most recent remote CI run is still `36569955148` at `3a3caaf3`. Its Python 3.14.7 coverage job
`109411339287` fails while testing the old cold import of `sage.libs.gap.libgap`. There is no CI run
for `dc227ad6`, which is local and five commits ahead of `origin/gap`.

The worktree-local Python 3.14.3 environment does not contain `passagemath-gap`. The configured
package index reports no matching distribution, and direct PyPI wheel retrieval returns HTTP 403.
Consequently, `test_installed_libgap_import_contract` skips locally. The upstream source makes the
fix well-founded, but neither mocks nor source inspection proves that qLDPC's complete real-extra
coverage job is green. This leaves iteration 7's explicit live-CI requirement unmet.

## Independent behavior checks

All Python commands used `/Users/D648438/src/qLDPC-gap/.venv`.

- A working-provider `sys.modules` simulation, with distribution metadata exposed as 10.6.48, ran
  all of `gap_test.py`: **6 passed**. This includes the real contract-test path and confirms the
  previous provider-dependent `test_is_installed` failure is fixed.
- Running the subprocess/manual/package-fallback tests while a working provider was importable:
  **3 passed**.
- Separate first-import and second-import broken-provider probes confirmed that the installed
  contract propagates `ImportError` rather than skipping, while runtime `_get_libgap()` returns
  `None` and permits fallback.
- `.venv/bin/python checks/pytest_.py src/qldpc/external/`:
  **19 passed, 1 skipped** (the absent real provider).
- `.venv/bin/python checks/all_.py`: **passed**, including formatting, lint, strict mypy, tests,
  100% statement coverage, and documentation.
- `git diff --check` passed. The branch is `gap`, its merge base is the recorded
  `40a571e455242e68e1a6f8e67ebaee310ad818b0`, and the pre-inspection tree was clean.

## Required iteration 9

Publish or otherwise execute `dc227ad6` in the real-extra Python 3.14 workflow and require both the
non-skipping raw import/adapter contract and the complete coverage job to pass. No further code
change is indicated unless that live run exposes another provider-specific failure.
