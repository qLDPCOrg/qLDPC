# Iteration 3 inspection: FAIL

## Verdict

Iteration 2's import-contract blocker is fixed: distribution presence is detected independently,
the real dependency is installed in the required coverage job, and the live test calls both
`libgap.eval` and `TestPackageAvailability`. The branch is still not ready for a pull request
because qLDPC's actual direct evaluation adapter constructs a GAP program that the real
`libgap.eval` API cannot execute.

## Blocking finding

### The direct adapter passes invalid multi-statement input to `libgap.eval`

`_get_output_libgap()` removes every command's trailing semicolons and joins those commands between
several other top-level GAP statements. For example, an independent probe of the current code for
`("x := 1;;", "Print(x);;")` produced:

```gap
__qldpc_output__ := "";
__qldpc_output___stream := OutputTextString(__qldpc_output__, false);
SetPrintFormattingStatus(__qldpc_output___stream, false);
x := 1
PrintTo(__qldpc_output___stream, x)
CloseStream(__qldpc_output___stream);
__qldpc_output__
```

The two transformed user commands have no statement terminators. More fundamentally, the selected
libgap implementation documents `Gap.eval()` as accepting one GAP command without its trailing
semicolon. Its `gap_eval()` appends one final semicolon and explicitly raises
`GAPError("can only evaluate a single statement")` when `GAP_EvalString` reports multiple statement
evaluations. The adapter supplies assignments, stream setup, user commands, stream close, and the
result expression as multiple top-level statements.

The mock `MockLibGap.eval()` accepts any string, so
`test_get_output_libgap()` cannot detect either incompatibility. The new live test evaluates only
the raw expression `"1 + 1"` and queries a raw GAP function; it does not call
`external.gap.get_output()` or `_get_output_libgap()` against the real binding. Consequently, the
CI job will verify that libgap itself works while still allowing qLDPC's direct execution path to be
broken. Downstream group/code commands are especially affected because they contain several
assignments and split `for ... do ... od` blocks.

This blocks direct libgap execution, preserved downstream behavior, real-backend evaluation
coverage, and PR readiness.

## Iteration 2 blocker verification

- **Distribution detection:** fixed. Only `PackageNotFoundError` from
  `importlib.metadata.version("passagemath-gap")` causes a skip. An installed distribution with a
  broken import now fails.
- **Required real dependency path:** fixed. The coverage job runs
  `uv sync --extra gap --extra relay-bp --group ci-test`; the test job does not depend on the
  separate installation-check environment.
- **Live API calls:** fixed as requested. When installed, the test imports the real module, evaluates
  `1 + 1`, and calls `TestPackageAvailability("guava")`.
- **Workflow behavior:** valid. The YAML parses, both `uv sync` and `uv run` support the supplied
  extra/group options, coverage remains the invoked gate, and pytest still enforces
  `--disable-socket`. PyPI metadata for 10.6.48 confirms Python 3.10--3.14 support, the documented
  libgap interface, and bundled GUAVA.

The worktree-local environment cannot install `passagemath-gap` because its configured registry
returns HTTP 403, so the live test skipped locally. No shared or global environment was modified.
The required CI coverage job now provides the non-skipping path.

## Acceptance-criterion assessment

| Criterion | Result | Evidence |
| --- | --- | --- |
| Dedicated `gap` branch from recorded `main` SHA | PASS | Branch and merge-base are the recorded `40a571e…`. |
| Direct libgap execution without spawning `gap` | FAIL | The real `eval` contract rejects the adapter's malformed multi-statement program. |
| No `gappy-system`; maintained supported install | PASS | No obsolete dependency; the 10.6.48 extra supports Python 3.10--3.14. |
| Public parsing/cache/field/error compatibility | FAIL | Public signatures remain, but actual direct group/code evaluation cannot execute. |
| Binding-absent fallback and explicit errors | PASS | Manual/subprocess and explicit package-error tests pass. |
| GAP package discovery/loading/install behavior | PASS | Real package querying is now required in CI; executable fallback remains covered. |
| Packaging metadata and user documentation | PASS | Metadata and docs consistently describe the supported extra and fallbacks. |
| No-network direct/fallback/downstream tests | FAIL | CI exercises raw real APIs, but actual qLDPC direct evaluation remains mock-only and broken. |
| Targeted tests, format, lint, mypy, full gate | PASS | Targeted: 19 passed, 1 skipped; `checks/all_.py` exits successfully. |
| Intentional clean PR-ready branch | FAIL | Scope is focused, but the direct adapter blocker remains. |

## Validation performed

- `.venv/bin/python checks/pytest_.py src/qldpc/external/`: **19 passed, 1 skipped**.
- `.venv/bin/python checks/all_.py`: **passed** (format, lint, mypy, tests, 100% coverage, docs).
- Workflow YAML parsed successfully; declared `uv` flags were checked against the installed CLI.
- `git diff --check`: passed.
- No `gappy-system` reference exists.
- Only the pre-existing orchestrator update to `status.json` was uncommitted before this report.

## Required iteration 4

1. Make `_get_output_libgap()` use the real single-command libgap contract while preserving output
   capture for complete multi-command programs, including split loops used by code helpers.
2. Add a test that runs qLDPC's `get_output()` (not only raw `libgap.eval`) against the real binding
   in the extra-enabled CI path and verifies representative printed output.
3. Keep the mock tests, but make the fake reject malformed or unsupported eval input so this
   regression cannot be hidden when the optional dependency is absent.
