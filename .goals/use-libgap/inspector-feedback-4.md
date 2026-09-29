# Iteration 4 inspection: PASS

## Verdict

Iteration 3's direct-evaluation blocker is resolved. The adapter now sends one GAP expression to
`libgap.eval`, preserves assignments and split control flow, captures every rewritten `Print` call,
and closes the output stream before returning. The branch satisfies the goal and is ready for
orchestrator finalization.

## Single-command wrapper verification

The generated shape is:

```gap
CallFuncList(function()
local __qldpc_output__, __qldpc_output___stream;
__qldpc_output__ := "";
__qldpc_output___stream := OutputTextString(__qldpc_output__, false);
SetPrintFormattingStatus(__qldpc_output___stream, false);
values := [1, 2];;
for value in values do
PrintTo(__qldpc_output___stream, value);;
od;;
PrintTo(__qldpc_output___stream, "!");;
CloseStream(__qldpc_output___stream);
return __qldpc_output__;
end, [])
```

This is one expression under passagemath's real semantics: its `gap_eval` appends the terminating
semicolon and rejects more than one statement result. `CallFuncList(function() ... end, [])` is
standard GAP syntax, and GAP's own sources use the same local string, `OutputTextString`,
`SetPrintFormattingStatus`, `PrintTo`, `CloseStream`, and return pattern.

Commands ending in `do`, `then`, `else`, or `repeat` remain open control-flow lines; ordinary
unterminated commands receive a semicolon; existing single and double semicolons remain intact.
The tests cover assignment, multiple prints, and a loop split across arguments. Existing nested
loops from the code adapter therefore remain syntactically intact.

Output state is local to the wrapper. Normal execution closes the stream before returning its
string. Binding errors are chained into the established `ValueError` with the original commands,
and subprocess nonzero/stderr behavior remains explicit. The mock now rejects the old top-level
multi-statement shape, and an independent probe confirmed that rejection.

The extra-enabled coverage job runs `test_installed_libgap_import_contract`. That test independently
detects the installed distribution, imports its real API, evaluates through raw libgap, queries
GUAVA, and then calls qLDPC's real `external.gap.get_output` adapter with assignment, split loop,
and repeated output.

## Acceptance-criterion assessment

| Criterion | Result | Evidence |
| --- | --- | --- |
| Dedicated branch and recorded base | PASS | Branch is `gap`; `main` and merge-base are `40a571e…`. |
| Direct libgap without spawning GAP | PASS | libgap is selected first; direct tests assert no subprocess call; extra-enabled CI exercises the real adapter. |
| Maintained supported install | PASS | No production `gappy-system` use; `passagemath-gap` 10.6.48 declares Python 3.10--3.14 support and wheels. |
| Public behavior and compatibility | PASS | Public signatures are unchanged; parsing, cache, finite-field, group, and code tests pass. |
| Optional fallback and explicit errors | PASS | Subprocess/manual paths and missing-package/error paths remain tested and explicit. |
| Package handling | PASS | Real GUAVA availability is checked in CI; missing embedded packages deliberately switch to the executable or raise. |
| Packaging and documentation | PASS | The `gap` extra, supported marker, README, API docstring, and library map agree. |
| Required tests | PASS | Mock regression has teeth; real API CI calls qLDPC's adapter; all downstream external tests pass. |
| Quality gates | PASS | Targeted external tests and the complete local gate pass. |
| Focused PR-ready branch | PASS | Product changes are issue-related; only the orchestrator status update was uncommitted before this report. |

## Validation performed

- `.venv/bin/python checks/pytest_.py src/qldpc/external/`: **19 passed, 1 skipped**.
- `.venv/bin/python checks/all_.py`: **passed** (706 tests, formatting, lint, strict mypy,
  100% coverage, and docs).
- Workflow YAML parsed successfully; its `uv --extra`/`--group` flags and coverage invocation were
  verified.
- The installed-release test skipped locally because `passagemath-gap` is unavailable from the
  configured registry. Both the configured registry and direct PyPI artifact fetch returned 403;
  no shared environment was touched. The required CI job installs the extra and cannot skip an
  installed-but-broken API.
- Public signatures match the initial SHA, branch scope is focused, and the worktree had no product
  changes from inspection.
