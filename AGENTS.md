# Working on qLDPC

Read the [project goals](README.md#-goals) before proposing, implementing, or reviewing a change, and let them guide its scope, design, validation, and explanation.
Apply them to the requested work, not to justify unrelated changes; a change need not advance every goal or include a tutorial.

The [contributing guide](docs/source/contributing/index.rst) explains how to set up, validate, and review a change.
Before editing, read the [standards](docs/source/contributing/standards/index.rst) that its [kinds-of-change table](docs/source/contributing/index.rst#kinds-of-change) lists for the change.
For how the packages relate, start with the [library map](docs/source/library_map.rst); for exact signatures and per-object literature, use the docstrings or the [API reference](https://qldpc.readthedocs.io/autoapi/index.html).

## What to trust when documents disagree

For what the code does, use this order:

1. Current implementation and its invariant tests.
2. Public docstrings and the `__all__` lists in package `__init__.py` files.
3. The library map and executable [examples](examples/).
4. Historical discussion, issues, or review notes.

For how code should be written, the [standards](docs/source/contributing/standards/index.rst) take precedence over existing code.
Re-derive behavior from the current branch before documenting or changing it, and check whether the code already does what a request asks.

## Reviews

Run each **full review** or **quick review** as [reviewing changes](docs/source/contributing/reviewing.rst) describes.
As a reviewer, follow the coordinator's instructions, and give no review status.
Otherwise, end every report on a change with its review status, as in "review: not run", "quick review: passed after 1 round; pre-existing problems fixed: 0; open decisions: none", or "full review: not passed after 4 rounds; pre-existing problems fixed: 1; open decisions: 2".
Point out when a change would be easier to review as a sequence of smaller pull requests, such as when it mixes concerns that could merge separately or its review stops converging.

## Decisions and recommendations

Recommend, present decisions, and weigh new process as [decisions and recommendations](docs/source/contributing/decisions.rst) describes, including its rules on [adding process](docs/source/contributing/decisions.rst#adding-process).

## Working on a change

1. While writing a change, run the drafting checks of the [validation commands](docs/source/contributing/index.rst#validation-commands).
2. When the change is ready, run the review that [reviewing changes](docs/source/contributing/reviewing.rst) calls for, unless the change has passed that review or a full review of the whole change, and every revision since has been [local](docs/source/contributing/reviewing.rst#rounds) and evidently correct.
   If the review does not pass, stop and report.
3. Run the extended checks only if the maintainer asks for them by name, or has said that they will be away while the change is completed.
   Fix each failure that the change causes, then return to step 2.
   Report any other failure, such as one that also occurs on the main branch, as an open decision.
4. Hand the change off as [handing off a change](#handing-off-a-change) describes.

A request to finish a change, as in "finish this PR", asks for steps 2 to 4.
Run checks individually, never through `checks/all_.py`, and run tests over the whole repository only in the extended checks.
Leave notebook runs and the GAP integration tests to CI.

## Handing off a change

Hand a change to the maintainer to push only once it has passed its review and everything for it, including new files, is committed.
For each of its pull requests, give in your message, not only in a file:

1. its review status, as [reviews](#reviews) describes, then each pre-existing problem that the review fixed, each fix left as a suggestion, each unreviewed revision, and each open decision;
2. one shell block that pushes the branch and then opens GitHub's page for a new pull request, as in:

   ```bash
   cd <worktree> && git push -u origin <branch> && \
     open "https://github.com/qLDPCOrg/qLDPC/compare/main...<branch>?expand=1"
   ```

3. the title, alone in a code block;
4. the description, following the [pull request standard](docs/source/contributing/standards/pull_requests.rst), in a `markdown` code block whose fence is longer than any fence inside it.

A request for a pull request's information, even "in the standard format", asks for this list.
If there are several pull requests, say whether they depend on each other and in which order they should merge.

## Protected files

These files state the rules for working on qLDPC, or enforce them:

- this file;
- the "Goals" and "AI usage" sections of [`README.md`](README.md);
- every file under [`docs/source/contributing/`](docs/source/contributing/), [`checks/`](checks/), and [`.github/`](.github/);
- the tests that enforce the standards: [`_code_style_test.py`](src/qldpc/_code_style_test.py), [`_compilation_test.py`](src/qldpc/_compilation_test.py), and [`contributor_docs_test.py`](src/qldpc/contributor_docs_test.py).

Unless the maintainer says to apply a change to them directly, change them only through a patch that the maintainer approves.
This holds even when the maintainer asked for the change, or when a change elsewhere forces it, such as renaming a path that they cite.
Write the patch outside the repository, link it, and say which parts of it no review has seen.
Once the maintainer approves it, apply exactly that patch with `git apply`, and commit it; if it no longer applies, regenerate it and ask again.
