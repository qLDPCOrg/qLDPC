# Working on qLDPC

Before proposing, implementing, or reviewing changes, read the [project purpose and goals](README.md#-goals), and use them to guide scope, design, validation, and explanation.
Apply the goals to the requested work rather than using them to justify unrelated changes.
A change need not advance every goal or include a tutorial.

The [contributing guide](docs/source/contributing/index.rst) explains how to set up, validate, and review a change, and the [standards](docs/source/contributing/standards/index.rst) state the rules that a change must satisfy.
For the data model and how the packages relate, start with the [library map](docs/source/library_map.rst).
For exact signatures and per-object literature, use the source docstrings or the generated [API reference](https://qldpc.readthedocs.io/autoapi/index.html).

## What to trust when documents disagree

For what the code does, use this order:

1. Current implementation and its invariant tests.
2. Public docstrings and the `__all__` lists in package `__init__.py` files.
3. The library map and executable [examples](examples/).
4. Historical discussion, issues, or review notes.

For how code should be written, the standards take precedence over existing code.
Re-derive behavior from the current branch before documenting or changing it.

## Before editing

Read the standards that the [kinds-of-change table](docs/source/contributing/index.rst#kinds-of-change) lists for the change.
Read the [orientation](docs/source/contributing/index.rst#orientation) before reorganizing code.

## Reviews

When asked for a **standard review**, follow [reviewing changes](docs/source/contributing/reviewing.rst).
Run a standard review before reporting a change as ready for a draft pull request, unless told otherwise.
When asked for a pull request description, follow the [pull request standard](docs/source/contributing/standards/pull_requests.rst).

## When to run checks

The [validation commands](docs/source/contributing/index.rst#validation-commands) define the drafting and finishing checks.
On your own, run only the drafting checks.
Run the finishing checks only when a maintainer explicitly asks to finish, finalize, or make a pull request ready to merge or ready for review.
Examples of such requests are "finish this PR" and "run the finishing checks".
You may report that a change appears ready for the finishing checks.
Run the finishing checks individually rather than through `checks/all_.py`, whose test run duplicates the one in `coverage_.py`.
Leave `pytest_.py --notebook` runs and the GAP integration tests to CI.
As part of drafting, refresh the saved outputs of a notebook that you change, even though CI executes notebooks.

## Changes to the standards and this file

Do not change this file or any file under [`docs/source/contributing/`](docs/source/contributing/) on your own, including mechanical edits such as updating a renamed path.
Change these files only as a maintainer requested or approved, and show the maintainer the exact diff before committing it.
When you notice a pattern worth codifying, such as a repeated request or a recurring review finding, propose the wording and where it belongs instead of applying it.
