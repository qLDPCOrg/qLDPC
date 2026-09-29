# Iteration 7 inspection: FAIL

## Verdict

Iteration 7 now obtains `libgap` from the correct specific module, but the preceding
`passagemath_gap` import is still not an authoritative non-interactive library contract. More
immediately, a successful provider import makes the existing fallback tests fail, so the proposed
fix cannot make the real-extra coverage job green.

## Blocking findings

### 1. The initialization sequence is not a supported library contract

The declared range currently resolves to PassageMath 10.6.48. PyPI marks that release for Python
3.10--3.14 and publishes CPython 3.14 wheels, so the package/version selection itself is sound.
Upstream does not, however, support this sequence for library code:

```python
import passagemath_gap
from sage.libs.gap.libgap import libgap
```

Authoritative upstream evidence says the opposite:

- PassageMath PR #1811 introduced the `passagemath_*` modules and explicitly says they are for
  interactive use, while libraries should use specific `sage.*` imports.
- In 10.6.48, `passagemath_gap/__init__.py` only wildcard-imports
  `sage.all__sagemath_gap`. That aggregate imports categories, geometry, groups, GAP elements, and
  finally `sage.libs.gap.libgap`; it is not a documented initialization API.
- The 10.6.48 README's `passagemath_gap` example runs in IPython and is therefore consistent with
  PR #1811's interactive-only contract.
- The cited commit `788ff01f...` is a release-version update. It contains the aggregate source but
  does not authorize using that interactive module as a library initializer.
- PassageMath issue #1604 acknowledges public-API import-cycle defects and records a particular
  aggregate workaround for another import. It does not establish `passagemath_gap` as the supported
  initializer for libgap.

Importing the aggregate first is credible as a workaround: it imports the same specific libgap
module after establishing a broader order, so it likely avoids the observed cold-import cycle.
Credibility is not an upstream-supported contract, especially for a dependency that qLDPC intends
to advertise as its maintained non-interactive backend.

### 2. A working provider breaks the installed-extra test run

The coverage job installs `--extra gap`, but several legacy tests still assume `_get_libgap()`
returns `None`. For example, `test_is_installed()` clears `is_callable()` and mocks only
`subprocess.run`; line 144 then asserts GAP is not callable. If iteration 7 makes the provider work,
`is_callable()` correctly returns `True` before consulting the subprocess, and the assertion fails.

I reproduced this independently by placing working `passagemath_gap` and
`sage.libs.gap.libgap` module objects in `sys.modules` and running only that test:

```text
FAILED src/qldpc/external/gap_test.py::test_is_installed
E   AssertionError
>   assert not external.gap.is_callable()
```

The same provider-dependent assumptions occur in subprocess-oriented `get_output` tests. In
addition, once imports succeed, no unit test executes `_get_libgap()`'s `ImportError`/`OSError`
return, so the real-extra 100% coverage run would lose coverage of that fallback line unless an
explicit mock covers it.

The last published CI run (`36569955148`, Python 3.14.7) failed on the original direct-import
circular import. The iteration-7 commit is intentionally unpushed, so no remote result proves the
new sequence. Even if the sequence fixes that import, the provider-success failure above prevents
the job from passing.

## Preserved behavior

- The installed-but-broken contract test remains strict. With distribution metadata present and
  either import raising `ImportError`, the error propagates rather than becoming a skip.
- `_get_libgap()` still returns `None` for `ImportError` and `OSError`, including failure of the
  second specific import.
- Existing executable and manual-fallback tests pass when the optional provider is absent.
- The direct adapter, single-expression GAP wrapper, package fallback, downstream parsing,
  dependency metadata, and documentation remain otherwise consistent with the goal.

## Validation

All Python commands used `/Users/D648438/src/qLDPC-gap/.venv` (Python 3.14.3):

- `checks/pytest_.py src/qldpc/external/gap_test.py`: **5 passed, 1 skipped**.
- `checks/pytest_.py src/qldpc/external/`: **19 passed, 1 skipped**.
- `checks/all_.py`: **passed**, including formatting, lint, strict mypy, coverage, tests, and docs.
- Working-provider simulation: **failed** at `gap_test.py:144`, as described above.
- Explicit missing/broken-provider probes: **passed**.
- `git diff --check` passed; the branch base is the recorded `40a571e...`.

The real optional dependency is unavailable from the configured local package index, and public
PyPI wheel access returned HTTP 403, so the real binary wheel could not be executed locally.

## Required iteration 8

1. Obtain and cite an upstream-supported non-interactive initialization path for libgap (or an
   upstream confirmation/fix for the direct-import cycle); do not present the interactive aggregate
   as a supported library initializer based only on its implementation order.
2. Make subprocess/manual tests provider-independent by explicitly mocking `_get_libgap()` as
   absent where that is the scenario, and add explicit first- and second-import exception tests so
   fallback coverage remains stable when the real provider works.
3. Run the real-extra Python 3.14 CI job and require both the raw import/adapter contract test and
   the complete 100% coverage gate to pass.
