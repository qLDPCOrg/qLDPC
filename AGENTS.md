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

Coordinate each **full review** or **quick review** as [reviewing changes](docs/source/contributing/reviewing.rst) describes.
As a reviewer, follow the coordinator's instructions, and end your report without a review status.
Otherwise, end every report on a change with its review status, as in "review: not run", "quick review: passed after 1 round; open decisions: none", "full review: passed after 2 rounds; pre-existing problems fixed: 1; open decisions: none", or "full review: not passed after 4 rounds; open decisions: none".
Count the pre-existing problems that the review fixed, as in the examples.
When asked for a pull request description, follow the [pull request standard](docs/source/contributing/standards/pull_requests.rst).
Use judgment to point out when a change would be easier to review as a sequence of smaller pull requests, such as when it mixes concerns that could merge separately or its review stops converging.

## Working on a change

1. While writing a change, run the anytime and as-needed checks of the [validation commands](docs/source/contributing/index.rst#validation-commands).
2. When the change is ready, run the review that [reviewing changes](docs/source/contributing/reviewing.rst) calls for.
   Skip it if the change has passed that review or a full review of the whole change, and every revision since has been [local](docs/source/contributing/reviewing.rst#rounds) and evidently correct.
   If the review does not pass, stop and report.
3. Run the extended checks only if the maintainer asks for them by name, or has said that they will be away while the change is completed.
   Handle their failures as the validation commands describe.
4. Hand the change off as [handing off a change](#handing-off-a-change) describes.

A request to finish a change, as in "finish this PR", asks for steps 2 to 4.
Run checks individually, never through `checks/all_.py`.
Run tests over the whole repository only as part of the extended checks.
Leave notebook runs and the GAP integration tests to CI.

## Handing off a change

Hand a change to the maintainer to push only once it has passed its review and all repository changes for it, including new files, are committed.
When handing off a change, give the following for each of its pull requests:

1. Its review status, as [reviews](#reviews) describes.
   Then list each pre-existing problem that the review fixed, each fix left as a suggestion, each unreviewed revision, and each open decision, as [reviewing changes](docs/source/contributing/reviewing.rst) describes.
2. One shell block that pushes the branch and then opens GitHub's page for a new pull request, as in:

   ```bash
   cd <worktree> && git push -u origin <branch> && \
     open "https://github.com/qLDPCOrg/qLDPC/compare/main...<branch>?expand=1"
   ```

3. The title, alone in a code block.
4. The description, in a `markdown` code block, following the [pull request standard](docs/source/contributing/standards/pull_requests.rst).
   If the description contains code blocks, use a longer fence for it, such as four backticks.

If the change has more than one pull request, say whether they depend on each other and in which order they should merge.

## Protected files

These files state the rules for working on qLDPC, or enforce them:

- this file;
- the "Goals" and "AI usage" sections of [`README.md`](README.md);
- every file under [`docs/source/contributing/`](docs/source/contributing/), [`checks/`](checks/), and [`.github/`](.github/);
- the tests that enforce the standards: [`code_style_test.py`](src/qldpc/code_style_test.py), [`_compilation_test.py`](src/qldpc/_compilation_test.py), and [`contributor_docs_test.py`](src/qldpc/contributor_docs_test.py).

They change only through a patch that a maintainer approves, including changes that a maintainer asks for.
The exception is a change that the maintainer explicitly says to apply directly.

1. Write the changes to protected files as a `.patch` file outside the repository, such as in a session directory, and give the maintainer a link to it.
   Say whether a review has seen the patch as it stands, and name any part that none has.
2. If the maintainer asks for revisions, write a new patch, which may replace the old one, and link it again.
3. Once the maintainer approves a patch, apply exactly that patch with `git apply`, and commit it.
   If it no longer applies, regenerate it and ask for approval again, rather than editing protected files by hand.

Updates that a change elsewhere forces on protected files, such as a renamed path that they cite, also go through a patch, which you may present together with that change.
Changes to other files follow the usual workflow.

When you notice a pattern worth codifying, such as a repeated request or a recurring review finding, describe it and the rule you would propose.
Write a patch for it only once the maintainer wants the rule.
