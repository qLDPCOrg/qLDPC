# Goal: Use libgap for GAP integration

## User Request

Working in a new worktree and branch off of `main`, named `gap`, fix
https://github.com/qLDPCOrg/qLDPC/issues/275. Do not stop until the branch is ready to open a
pull request into `main`. Use only worktree-local virtual environments so shared environments
used by other agents are not modified. End with a copy/paste-ready pull request title and
GitHub pull request summary.

## Refined Goal

Replace qLDPC's subprocess-first GAP execution with a direct libgap-backed integration using
the maintained `passagemath-gap` distribution, while preserving the existing public GAP,
group, and code APIs and their documented optional/fallback behavior. The implementation must
work across qLDPC's supported Python range, must not depend on the obsolete and currently
unbuildable `gappy-system` package, and must include packaging, tests, and user documentation
needed for a reviewable pull request into `main`.

All setup, dependency installation, and validation must use an isolated virtual environment
inside `/Users/D648438/src/qLDPC-gap`; do not activate, install into, or otherwise modify a
shared environment.

## Acceptance Criteria

- [ ] The `gap` branch exists in the dedicated `/Users/D648438/src/qLDPC-gap` worktree and is
      based on the `main` commit recorded in `status.json`.
- [ ] qLDPC can use the direct libgap interface supplied by `passagemath-gap` for GAP
      evaluation; when that binding is available, supported GAP operations do not spawn the
      `gap` command-line program.
- [ ] The implementation does not add or use `gappy-system`, and the supported installation
      path is usable on qLDPC's supported Python versions without relying on the build failure
      described in issue #275.
- [ ] Existing public signatures and return behavior in `qldpc.external.gap`,
      `qldpc.external.groups`, and `qldpc.external.codes` remain compatible, including command
      output parsing, disk-cache behavior, finite-field encoding, and established exception
      semantics.
- [ ] The optional/manual fallback remains usable when the direct binding is unavailable;
      missing optional functionality is surfaced explicitly rather than silently returning
      success-shaped results.
- [ ] GAP package discovery/loading and any supported installation behavior continue to work
      with the direct backend, including the packages used by group and code helpers.
- [ ] Packaging metadata exposes the maintained libgap dependency in the repository's
      conventional way, and installation/user documentation clearly explains how to enable
      and use GAP support and any retained limitations.
- [ ] Tests cover direct-backend availability, evaluation, errors, package handling, absence of
      the binding, fallback behavior, and unchanged downstream group/code parsing. Tests must
      not require network access.
- [ ] Targeted external-module tests, formatting, linting, strict mypy, and the repository's
      full `python checks/all_.py` quality gate pass from the worktree-local environment.
- [ ] The final branch contains only intentional issue-related and Goal process changes, has no
      uncommitted changes, and is ready for a pull request into `main`.

## Scope Boundaries

**In scope:**
- Direct libgap integration using the `passagemath-gap` distribution.
- The GAP execution and package-loading adapter plus tightly coupled group/code call sites.
- Optional-dependency metadata, tests, and directly related installation/API documentation.
- Preserving a practical non-libgap fallback where required for compatibility.
- Validation exclusively in a worktree-local virtual environment.

**Out of scope:**
- Rewriting unrelated group theory, coding theory, decoder, circuit, or lattice-surgery logic.
- Adding new GAP-powered features unrelated to replacing the execution backend.
- Fixing general Windows GAP support tracked separately by issue #294.
- Vendoring or repairing the abandoned `gappy-system` project.
- Modifying any shared virtual environment or unrelated worktree.
- Opening or publishing the pull request; the requested deliverable is a branch ready to open
  one, plus copy/paste-ready title and summary.

## Applicable Project Conventions

**Quality gate command:**
- `.venv/bin/python checks/all_.py`
- Targeted tests should use `.venv/bin/python checks/pytest_.py src/qldpc/external/` or a
  narrower existing selector before the full gate.

**Environment setup:**
- Create `/Users/D648438/src/qLDPC-gap/.venv` with a local Python interpreter.
- Install/sync all dependencies into that local environment only. Never use a shared active
  environment or global `pip`.

**Commit convention:**
- Follow the repository's observed conventional-commit style where practical.
- Goal iteration commits must retain the required `[B]` or `[I]` role marker.
- Builder trailer: `Assisted-by: OpenAI:GPT-5.6 Luna`
- Inspector trailer: `Assisted-by: OpenAI:GPT-5.6 Sol`
- User-requested final PR preparation must also retain the repository-required co-author trailer
  if commits are rewritten.

**Guidelines:**
- `/Users/D648438/src/qLDPC-gap/AGENTS.md`
- `/Users/D648438/src/qLDPC-gap/CLAUDE.md`
- `/Users/D648438/src/qLDPC-gap/.github/workflows/continuous-integration.yaml`
- No `CONSTITUTION.md`, `.agents/guidelines/`, or `.github/guidelines/` exists.

**Rules:**
- Source lives under `src/qldpc/`; tests are co-located as `<module>_test.py`.
- Keep public exports explicit and preserve compatibility or add deprecation shims.
- Ruff line length is 100, mypy runs in strict mode, tests disable network sockets, and
  statement coverage must remain 100%.
- Use the existing check wrappers; the full gate runs format, lint, mypy, pytest/coverage, and
  documentation checks.
