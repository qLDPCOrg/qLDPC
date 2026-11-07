# Agent and contributor guide

This guide is for people and agents changing qLDPC.
Before proposing, implementing, or reviewing changes, read the [project purpose and goals](README.md#-goals) and keep them in mind.
Use them to guide scope, design, validation, and explanation.
Apply the goals to the requested work rather than using them to justify unrelated changes.

A change need not advance every goal or include a tutorial.
Make significant tradeoffs explicit, and keep the supported scope maintainable.
The [standards](docs/source/contributing/standards/index.rst) state the rules that a change must satisfy, one topic per standard.
The [contributing page](docs/source/contributing/index.rst) explains how to set up, validate, and review a change.
Before editing, read the standards that its [kinds-of-change table](docs/source/contributing/index.rst#kinds-of-change) lists for the change.
For a human-facing explanation of the data model and package relationships, start with the [library map](docs/source/library_map.rst).
For exact signatures and per-object literature, use the source docstrings or the generated [API reference](https://qldpc.readthedocs.io/autoapi/index.html).

## What to trust when documents disagree

For what the code does, use this order:

1. Current implementation and its invariant tests.
2. Public docstrings and the `__all__` lists in package `__init__.py` files.
3. The [library map](docs/source/library_map.rst) and executable [examples](examples/).
4. Historical discussion, issues, or review notes.

For how code should be written, this guide and the standards take precedence over existing code: code that breaks a rule is debt to fix, not a precedent to copy.
Re-derive behavior from the current branch before documenting or changing it.
Do not copy transient project history, machine-specific paths, or local-session instructions into production files.

## Orientation

Canonical executable notebooks live in [`examples/`](examples/), Sphinx pages and the library map in [`docs/source/`](docs/source/), local wrappers for CI's quality gates in [`checks/`](checks/), and untested research scripts in [`experiments/`](experiments/).
Read the [orientation](docs/source/contributing/index.rst#orientation) and the [code standard](docs/source/contributing/standards/code.rst) before reorganizing code.

## Public API and private code

Plain names defined in public modules are public API, even when they are not re-exported.
Preserve released public import paths and respect top-level subpackage boundaries for underscore names.
Before adding or moving names, read the [public/private rules](docs/source/contributing/standards/code.rst#public-api-and-private-code), including the exceptions for experimental and deprecation code.

## Code style and organization

Read the [code standard](docs/source/contributing/standards/code.rst), including its [simplicity and clarity rules](docs/source/contributing/standards/code.rst#simplicity-and-clarity), before changing source.
Source files need an SPDX license header and a module docstring.
Sections use a line of exactly 100 `#` characters and a title; plain names precede underscore names unless import-time dependencies require otherwise.
[`code_style_test.py`](src/qldpc/code_style_test.py) checks these layout rules.
Follow the [prose style rules](docs/source/contributing/standards/writing.rst#prose-style) when writing comments, docstrings, or documentation.

## Core invariants

Read the applicable sections of the [implementation standard](docs/source/contributing/standards/implementation.rst) and the [decoder standard](docs/source/contributing/standards/decoders.rst) before changing algebra, decoding, circuits, or integrations.

- Preserve concrete finite fields and compatible group rings; extension-field values are not integers modulo the field order.
- Distinguish subsystem gauge generators from stabilizers, and preserve symplectic conventions.
- Keep error inference distinct from observable prediction, and dispatch on explicit decoder capabilities.
- Circuit tools are qubit-only, and a valid encoding circuit is not necessarily fault-tolerant.
- Treat supplied parameters and constructor promises as trusted inputs, not independent evidence.
- Make external side effects explicit and report failures rather than returning success-shaped defaults.

## Tests

Read the [testing standard](docs/source/contributing/standards/testing.rst) before adding or rewriting tests.
Every module needs a co-located `*_test.py` that fully covers it.
Use small, independent oracles and guard scientific behavior, not incidental implementation details.
Mock external boundaries, never the logic under test.
Tests must work offline, without persistent result caches or order dependence, and treat deprecation and future-change warnings as errors.
Check that a new or rewritten test fails when its guarded behavior is broken.

## Deprecations and future changes

Never break callers of a released public API without warning.
Read the [deprecation standard](docs/source/contributing/standards/deprecations.rst) before changing a public name, argument, mode, or default.
Deprecations use `DeprecationWarning`; changes that would silently alter the same call use `FutureWarning`.
Future changes move through an old default that warns, a new default that warns on explicit old usage, and defunct old usage, at least one minor release apart.
Keep substantial compatibility code in [`qldpc/_deprecations/`](src/qldpc/_deprecations/).

## Reviews and pull requests

When asked for a **standard review**, follow [reviewing changes](docs/source/contributing/reviewing.rst).
Agents run a standard review before reporting a change as ready for a pull request, unless told otherwise.
Follow the [pull request standard](docs/source/contributing/standards/pull_requests.rst) when preparing a PR description.

## Changes to standards and agent guidance

Do not change `AGENTS.md` or any file under [`docs/source/contributing/`](docs/source/contributing/) on your own, including mechanical edits such as updating a renamed path.
Change these files only as a maintainer requested or approved, and show the maintainer the exact diff before committing it.
When you notice a pattern worth codifying, such as a repeated request or a recurring review finding, propose the wording and where it belongs instead of applying it.

## Validation commands

Install the development environment with `python -m pip install -e '.[dev]'`.
Read the [validation details](docs/source/contributing/index.rst#validation-commands) for incremental checks, notebook environments, and coverage caveats.
CI's exact environments and commands are in [`.github/workflows/`](.github/workflows/).

| Command | Purpose |
| --- | --- |
| `python checks/pytest_.py [paths]` | Pytest suite, or the tests under the given paths; does not execute notebooks by default |
| `python -m pytest path/to/module_test.py::test_name -v` | One focused test |
| `python checks/pytest_.py --notebook examples/decoders.ipynb examples/basics.ipynb examples/noise_models.ipynb` | Execute the notebooks that CI executes, with their dependencies installed; left to CI |
| `python checks/format_.py --check` / `--fix` | Check, or apply, Ruff and `pyproject.toml` formatting |
| `python checks/lint_.py` | Ruff lint |
| `python checks/mypy_.py` | mypy with the repository's typing settings, over source and tests |
| `python checks/coverage_.py` | 100% statement coverage of each module by its own test file, as in CI |
| `python checks/build_docs.py` | Strict Sphinx and notebook documentation build |
| `python checks/all_.py` | Convenience for people, not agents: formatting, lint, mypy, tests with overall 100% coverage, and (on full runs) the docs build; it does not run the per-module coverage gate or `pyproject.toml` formatting |

While drafting, run only the tests of the changed modules and the formatting, lint, and type checks on changed files.
Run the [finishing checks](docs/source/contributing/index.rst#validation-commands), the full format, lint, type, coverage, and documentation checks, only when a maintainer explicitly asks to finish or finalize a pull request, or to make it ready to merge or ready for review.
Never start them on your own; you may report that a change looks ready for them.
Run the checks individually; do not run `checks/all_.py`, which repeats the test run.
Leave notebook and GAP integration tests to CI.
Do not weaken a failure with `noqa`, `type: ignore`, or coverage exclusions unless the exceptional condition is real and documented.
