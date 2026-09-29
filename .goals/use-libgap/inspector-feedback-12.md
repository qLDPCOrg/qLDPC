# Inspector feedback — iteration 12

## Verdict: PASS

I independently reviewed the complete diff from recorded initial SHA
`40a571e455242e68e1a6f8e67ebaee310ad818b0` through Builder HEAD
`c84aae7701d95147d401a423350ff46cf575618a`. The `gap` branch has that initial
SHA as its merge base, and the product changes are limited to the GAP adapter,
its tests, optional-dependency metadata, CI, and directly related documentation.

## Verification

- The direct backend follows PassageMath's library import order:
  `sage.all__sagemath_gap` followed by `sage.libs.gap.libgap`. PassageMath's
  cited `sage.features.gap` source uses the same order.
- Direct commands are wrapped into one valid
  `CallFuncList(function() ... end, [])` expression. Unit tests verify output,
  loops, assignments, errors, and that this path does not call `subprocess.run`.
- Missing bindings retain executable and manual fallbacks. Missing embedded
  packages either switch to the verified executable backend or raise an
  explicit error; failures do not produce success-shaped values.
- Public GAP/group/code signatures are unchanged. Existing parsing,
  finite-field, cache, package, and exception tests pass.
- The `gap` extra uses `passagemath-gap>=10.6.48,<10.7` for Python below 3.15.
  PyPI metadata for 10.6.48 declares Python 3.10–3.14 support. No product or
  dependency use of `gappy-system` exists.
- README and library-map documentation explain installation, backend priority,
  executable/manual fallback, embedded-package limitations, and Windows limits.
- Test isolation is sound: subprocess results are mocked, and the only live GAP
  contract is the installed-libgap test. Running external tests with
  `PATH=/usr/bin:/bin` (no `gap` command available) produced **19 passed,
  1 skipped**; the skip is the locally absent optional provider.
- The installed-provider test skips only when distribution metadata is absent.
  Once installed, import, `1 + 1`, GUAVA availability, and qLDPC direct output
  are mandatory assertions.

## Quality gates and CI

- Local `.venv/bin/python checks/all_.py`: exit 0; formatting, Ruff, strict
  mypy, pytest/coverage, and documentation all passed.
- Continuous Integration run `36584345483` is successful at exact Builder SHA
  `c84aae7701d95147d401a423350ff46cf575618a`. Jobs `109460522069`,
  `109460674161`, `109460674199`, and `109460674264` all passed. Coverage
  reports **17,789 statements, 0 missed, 100.00%**.
- Conditional Tests run `36584345500` is successful at the same SHA. Jobs
  `109460525227` (Python 3.10) and `109460525241` (Python 3.14) passed.
- The CI coverage job explicitly installs the `gap` extra, so the strict live
  provider contract cannot take its distribution-absent skip.

All acceptance criteria are satisfied. No product-code changes were made by
this inspection.

While this inspection was running, Builder iteration 13 landed as
`b7baa517553ee7e993020937b7c3799d2148d1b0`, adding only stricter subprocess
guards to `test_automorphism` and advancing the process status. I reviewed that
delta, ran the focused test (**1 passed**), and reran the complete local gate
successfully. It strengthens the isolation conclusion and does not change the
product or the exact-SHA CI evidence above.
