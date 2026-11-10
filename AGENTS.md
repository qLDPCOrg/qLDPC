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

When asked to run a **standard review**, coordinate it as [reviewing changes](docs/source/contributing/reviewing.rst) describes.
Run a standard review only when asked; you may suggest one.
As a reviewer within a standard review, follow the coordinator's instructions, and end your report without a review status.
Otherwise, end every report on a change with its review status, as in "standard review: not run" or "standard review: 2 rounds; pre-existing problems fixed: 1; open decisions: none".
Count every pre-existing problem that the review fixed, and list each one in the report, so that the maintainer sees each change they did not ask for.
When asked for a pull request description, follow the [pull request standard](docs/source/contributing/standards/pull_requests.rst).
Use judgment to point out when a change would be easier to review as a sequence of smaller pull requests, such as when it mixes concerns that could merge separately or its review stops converging.

## When to run checks

The [validation commands](docs/source/contributing/index.rst#validation-commands) define three tiers of checks: anytime, as needed, and finishing.
Run anytime checks freely, and as-needed checks when writing, reviewing, or verifying a change.
Run the finishing checks only when a maintainer asks you to finish a change, as in "finish this PR", or to run them.
Such a request includes leaving you to finish a change on your own, such as while the maintainer is away, since then no one will read CI's results before you report.
Otherwise, the maintainer usually starts CI and reads its results, which make the finishing checks unnecessary.
So report a change ready to push once its anytime and as-needed checks pass and any review that ran is done.
Run checks individually, never through `checks/all_.py`.
Leave repository-wide test runs, notebook runs, and the GAP integration tests to CI.

## Protected files

These files state the rules for working on qLDPC, or enforce them:

- this file;
- the "Goals" and "AI usage" sections of [`README.md`](README.md);
- every file under [`docs/source/contributing/`](docs/source/contributing/), [`checks/`](checks/), and [`.github/`](.github/);
- the tests that enforce the standards: [`code_style_test.py`](src/qldpc/code_style_test.py), [`_compilation_test.py`](src/qldpc/_compilation_test.py), and [`contributor_docs_test.py`](src/qldpc/contributor_docs_test.py).

They change only through a patch that a maintainer approves, including changes that a maintainer asks for.
The exception is a change that the maintainer explicitly says to apply directly.

1. Write the changes to protected files as a `.patch` file outside the repository, such as in a session directory, and give the maintainer a link to it.
2. If the maintainer asks for revisions, write a new patch, which may replace the old one, and link it again.
3. Once the maintainer approves a patch, apply exactly that patch with `git apply`, and commit it.
   If it no longer applies, regenerate it and ask for approval again, rather than editing protected files by hand.

Updates that a change elsewhere forces on protected files, such as a renamed path that they cite, also go through a patch, which you may present together with that change.
Changes to other files follow the usual workflow.

When you notice a pattern worth codifying, such as a repeated request or a recurring review finding, describe it and the rule you would propose.
Write a patch for it only once the maintainer wants the rule.
