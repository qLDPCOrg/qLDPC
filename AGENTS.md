# Agent and contributor guide

This guide is for people and agents changing qLDPC.
Before proposing, implementing, or reviewing changes, read the [project purpose and goals](README.md#-goals) and keep them in mind.
Use them to guide scope, design, validation, and explanation.
Apply the goals to the requested work rather than using them to justify unrelated changes.

A change need not advance every goal or include a tutorial.
Make significant tradeoffs explicit, and keep the supported scope maintainable.
The [contributor reference](docs/source/contributing.rst) contains the detailed rules and procedures; read its applicable sections before editing.
For a human-facing explanation of the data model and package relationships, start with the [library map](docs/source/library_map.rst).
For exact signatures and per-object literature, use the source docstrings or the generated [API reference](https://qldpc.readthedocs.io/autoapi/index.html).

## What to trust when documents disagree

For what the code does, use this order:

1. Current implementation and its invariant tests.
2. Public docstrings and the `__all__` lists in package `__init__.py` files.
3. The [library map](docs/source/library_map.rst) and executable [examples](examples/).
4. Historical discussion, issues, or review notes.

For how code should be written, this guide and the contributor reference take precedence over existing code: code that breaks a rule is debt to fix, not a precedent to copy.
Re-derive behavior from the current branch before documenting or changing it.
Do not copy transient project history, machine-specific paths, or local-session instructions into production files.

## Orientation

Canonical executable notebooks live in [`examples/`](examples/), Sphinx pages and the library map in [`docs/source/`](docs/source/), local wrappers for CI's quality gates in [`checks/`](checks/), and untested research scripts in [`experiments/`](experiments/).
Read the [orientation](docs/source/contributing.rst#orientation) before reorganizing code or changing shared code-core state.

## Public API and private code

Plain names defined in public modules are public API, even when they are not re-exported.
Preserve released public import paths and respect top-level subpackage boundaries for underscore names.
Before adding or moving names, read the [public/private rules](docs/source/contributing.rst#public-api-and-private-code), including the exceptions for experimental and deprecation code.

## Code style and organization

Read the [style and organization rules](docs/source/contributing.rst#code-style-and-organization) before changing source or documentation.
Source files need an SPDX license header and a module docstring.
Sections use a line of exactly 100 `#` characters and a title; plain names precede underscore names unless import-time dependencies require otherwise.
[`code_style_test.py`](src/qldpc/code_style_test.py) checks these layout rules.
Follow the [prose style rules](docs/source/contributing.rst#prose-style) when writing comments, docstrings, or documentation.

## Core invariants

Read the applicable [implementation requirements](docs/source/contributing.rst#core-invariants) before changing algebra, decoding, circuits, or integrations.

- Preserve concrete finite fields and compatible group rings; extension-field values are not integers modulo the field order.
- Distinguish subsystem gauge generators from stabilizers, and preserve symplectic conventions.
- Keep error inference distinct from observable prediction, and dispatch on explicit decoder capabilities.
- Circuit tools are qubit-only, and a valid encoding circuit is not necessarily fault-tolerant.
- Treat supplied parameters and constructor promises as trusted inputs, not independent evidence.
- Make external side effects explicit and report failures rather than returning success-shaped defaults.

## Tests

Read the [testing rules](docs/source/contributing.rst#tests) before adding or rewriting tests.
Every module needs a co-located `*_test.py` that fully covers it.
Use small, independent oracles and guard scientific behavior, not incidental implementation details.
Mock external boundaries, never the logic under test.
Tests must work offline, without persistent result caches or order dependence, and treat deprecation and future-change warnings as errors.
Check that a new or rewritten test fails when its guarded behavior is broken.

## Common change recipes

Read the relevant recipe before starting:

| Change | Required reference |
| --- | --- |
| Code construction | [Construction recipe](docs/source/contributing.rst#add-or-change-a-code-construction) |
| Code-core algebra or cached state | [Algebra recipe](docs/source/contributing.rst#change-code-core-algebra) |
| Decoder | [Decoder recipe](docs/source/contributing.rst#add-or-adapt-a-decoder) and [adding a decoder](docs/source/adding_decoders.rst) |
| Circuit workflow | [Circuit recipe](docs/source/contributing.rst#change-a-circuit-workflow) |
| External integration | [Integration recipe](docs/source/contributing.rst#change-an-external-integration) |
| Documentation or example | [Documentation recipe](docs/source/contributing.rst#add-documentation-or-an-example) |

## Deprecations and future changes

Never break callers of a released public API without warning.
Read the [full compatibility procedure](docs/source/contributing.rst#deprecations-and-future-changes) before changing a public name, argument, mode, or default.
Deprecations use `DeprecationWarning`; changes that would silently alter the same call use `FutureWarning`.
Future changes move through an old default that warns, a new default that warns on explicit old usage, and defunct old usage, at least one minor release apart.
Keep substantial compatibility code in [`qldpc/_deprecations/`](src/qldpc/_deprecations/).

## Pull requests

Follow the [PR-summary guidance](docs/source/contributing.rst#pull-request-summaries) when preparing a PR description.

## Validation commands

Install the development environment with `python -m pip install -e '.[dev]'`.
Read the [validation details](docs/source/contributing.rst#validation-commands) for incremental checks, notebook environments, and coverage caveats.
CI's exact environments and commands are in [`.github/workflows/`](.github/workflows/).

| Command | Purpose |
| --- | --- |
| `python checks/pytest_.py [paths]` | Pytest suite, or the tests under the given paths; does not execute notebooks by default |
| `python -m pytest path/to/module_test.py::test_name -v` | One focused test |
| `python checks/pytest_.py --notebook examples/decoders.ipynb examples/basics.ipynb examples/noise_models.ipynb` | Execute the notebooks that CI executes, with their dependencies installed |
| `python checks/format_.py --check` / `--fix` | Check, or apply, Ruff and `pyproject.toml` formatting |
| `python checks/lint_.py` | Ruff lint |
| `python checks/mypy_.py` | mypy with the repository's typing settings, over source and tests |
| `python checks/coverage_.py` | 100% statement coverage of each module by its own test file, as in CI |
| `python checks/build_docs.py` | Strict Sphinx and notebook documentation build |
| `python checks/all_.py` | Combined gate: formatting, lint, mypy, tests with overall 100% coverage, and (on full runs) the docs build; it does not run the per-module coverage gate or `pyproject.toml` formatting |

Run checks on what you changed while iterating, and save repository-wide runs for a finished change.
Before merging, run `python checks/all_.py` and `python checks/coverage_.py`.
Do not weaken a failure with `noqa`, `type: ignore`, or coverage exclusions unless the exceptional condition is real and documented.
