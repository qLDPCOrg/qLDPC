# Iteration 6 inspection: FAIL

## Verdict

The Builder found an import that is present in the pinned release and is very likely to avoid the
Python 3.14 initialization cycle, but it is not PassageMath's supported contract for library code.
Upstream introduced the `passagemath_*` aggregate imports for interactive use and explicitly directs
libraries to specific `sage.*` imports. qLDPC must not replace the intended library API with an
interactive convenience re-export without documenting and isolating that initialization workaround.

## Blocking finding

### `passagemath_gap.libgap` is documented for interactive use, not library use

The exact 10.6.48 release does contain `passagemath_gap`; therefore iteration 1's finding about
10.5.49 no longer applies:

- PyPI currently has only one release satisfying `>=10.6.48,<10.7`: 10.6.48.
- Its metadata requires Python 3.10--3.14 and publishes CPython 3.10--3.14 wheels.
- The release commit `788ff01f07b768f6b0226718e43a0914f0b04f2d` contains
  `pkgs/sagemath-gap/passagemath_gap/__init__.py`, and the package manifest includes that directory.
- The 10.6.48 README demonstrates `from passagemath_gap import *`; the module re-exports
  `sage.all__sagemath_gap`, which in turn exports `libgap`.

However, the upstream change that created these packages, PassageMath PR #1811 / commit
`60c700d4f76c9390150944b1d09218782217d4d4`, states:

> Like the existing import names `sage.all__*`, they are meant for interactive use, not library use,
> which should use specific imports `from sage.some.name.space import ...`

For libgap, that specific API is `sage.libs.gap.libgap.libgap`. The Builder's new docstring calls
that API “internal” and makes qLDPC depend directly on an aggregate interactive namespace's
re-export. This is contrary to the upstream maintainer's stated import contract, so the branch is
not yet ready to advertise this as the supported integration.

The likely correct compatibility shape is to use the 10.6.48 aggregate import only to establish the
required split-distribution initialization order, then obtain `libgap` from
`sage.libs.gap.libgap`. The workaround and its reason should be explicit and covered by the real
extra-enabled test. An upstream issue or documented reference would also make the workaround
auditable.

## CI failure analysis

Job `109411339287` ran Python 3.14.7 with 10.6.48 and failed while cold-importing
`sage.libs.gap.libgap`, ending in:

```text
ImportError: cannot import name 'is_MPolynomial' from partially initialized module
'sage.rings.polynomial.multi_polynomial' (most likely due to a circular import)
```

The exact 10.6.48 `sage.all__sagemath_gap` source initializes categories, permutation-group
elements, geometry/groups, and `sage.libs.gap.element` before importing the specific libgap object.
That ordering provides a credible explanation for why importing `passagemath_gap` first avoids the
reported cold-import cycle. The local environment does not contain PassageMath, so the real import
test skipped and this inspector could not execute the fix against the binary wheel. The next CI run
must remain the non-skipping proof.

## Fallback and failure-semantics verification

- `_get_libgap()` still returns `None` for `ImportError` and `OSError`, preserving the optional
  subprocess/manual fallback.
- A module that imports but has no `libgap` attribute also returns `None`, as before.
- `test_installed_libgap_import_contract()` checks distribution presence separately. An absent
  distribution skips, while an installed distribution whose import raises `ImportError` fails the
  test. An independent monkeypatch probe confirmed both outcomes.
- Direct evaluation and package-query errors remain explicit `ValueError`/`ModuleNotFoundError`
  paths, and the package-to-executable fallback tests remain unchanged and passing.

## Validation

All Python commands used `/Users/D648438/src/qLDPC-gap/.venv`:

- `.venv/bin/python checks/pytest_.py src/qldpc/external/gap_test.py`: **5 passed, 1 skipped**.
- `.venv/bin/python checks/pytest_.py src/qldpc/external/`: **19 passed, 1 skipped**.
- `.venv/bin/python checks/all_.py`: **passed**, including format, lint, strict mypy, coverage,
  tests, and documentation.
- Independent absence/broken-install probes passed.
- `git diff --check` passed; the branch base remains the recorded `40a571e…`.

## Required iteration 7

1. Preserve PassageMath's library contract: after any required aggregate initialization, import the
   object from `sage.libs.gap.libgap`, rather than consuming the interactive aggregate's re-export.
2. Explain the initialization workaround without mischaracterizing the specific upstream API as
   internal.
3. Keep the installed-distribution test non-skipping in CI and make it assert the intended
   initialization-plus-specific-import sequence before exercising qLDPC's real adapter.
