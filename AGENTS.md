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
If the person you work for asks for [standard triage](docs/source/contributing/decisions.rst#standard-triage) by name, run it as it describes; settle decisions by it only when they ask you to.

## Working on a change

1. While writing a change, run the drafting checks of the [validation commands](docs/source/contributing/index.rst#validation-commands).
2. When the change is ready, run the review that [reviewing changes](docs/source/contributing/reviewing.rst) calls for, unless the change has passed that review or a full review of the whole change, and every revision since has been [local](docs/source/contributing/reviewing.rst#rounds) and evidently correct.
   If you work for a person who is not a maintainer, run one only if they ask, since maintainers review each pull request.
   If the review does not pass, stop and report, unless the person you work for has explicitly waived it for the hand-off, as [handing off a change](#handing-off-a-change) describes.
3. Run the [extended checks](docs/source/contributing/index.rst#validation-commands) only if the person you work for asks for them by name, or has said that they will be away while the change is completed.
   Fix each failure that the change causes, then return to step 2.
   Report any other failure, such as one that also occurs on the main branch, as an open decision.
4. Hand the change off as [handing off a change](#handing-off-a-change) describes.

A request to finish a change, as in "finish this PR", asks for steps 2 to 4.
Run checks individually, never through `checks/all_.py`, and run tests over the whole repository only in the extended checks.
Leave notebook runs and the GAP integration tests to CI.

## Handing off a change

Hand a pull request to the person you work for, who pushes it, only once the following conditions hold for its changes and for those of each pull request that it builds on and that has unpushed commits.
Each of these changes is committed, new files included.
Each has also passed any review that step 2 calls for, unless that person has explicitly waived that review for the hand-off; a request for the pull request or its information does not waive it.
A hand-off claims that these conditions hold.
For a pull request that you may not hand off because a review has not passed, give no push command, title, or description, and report the review as [reviewing changes](docs/source/contributing/reviewing.rst) describes.
For each pull request that you hand off, give in your message, not only in a file:

1. its review status, as [reviews](#reviews) describes, then each pre-existing problem that the review fixed, each fix left as a suggestion, each unreviewed revision, each decision that standard triage settled, and each open decision;
2. one shell block that pushes the branch and then opens GitHub's page for a new pull request, as in:

   ```bash
   cd <worktree> && git push -u origin <branch> && \
     open "https://github.com/qLDPCOrg/qLDPC/compare/main...<branch>?expand=1"
   ```

   Without write access, the block pushes to a fork and opens `compare/main...<fork owner>:<branch>`.

3. the title, alone in a code block;
4. the description, following the [pull request standard](docs/source/contributing/standards/pull_requests.rst), in a `markdown` code block whose fence is longer than any fence inside it.

A request for a pull request's information, even "in the standard format", asks for this list if the pull request may be handed off.
If there are several pull requests, say whether they depend on each other and in which order they should merge.

## Protected files

These files state the rules for working on qLDPC, or enforce them:

- this file;
- the "Goals" and "AI usage" sections of [`README.md`](README.md);
- every file under [`docs/source/contributing/`](docs/source/contributing/), [`checks/`](checks/), and [`.github/`](.github/);
- the tests that enforce the standards: [`_code_style_test.py`](src/qldpc/_code_style_test.py), [`_compiled_code_rules_test.py`](src/qldpc/_compiled_code_rules_test.py), and [`contributor_docs_test.py`](src/qldpc/contributor_docs_test.py).

If you work for a person who is not a maintainer, change them only when that person asks or a change elsewhere forces you to.
Otherwise, a maintainer approves each change to them before you hand it off, unless they say to apply it directly:

- **A pull request that exists to change protected files:** the maintainer inspects its branch, which is easier than reading a patch.
  Make the change on the branch, and run its review.
  Then give the maintainer a command to copy and paste that changes to its worktree, as in `cd <worktree>`, and ask them to inspect the branch.
  Hand the pull request off only once they say that they have reviewed it, and ask again after any later revision.
- **A pull request about something else that also changes protected files:** the maintainer approves a patch, which shows those changes apart from the rest.
  This holds even when the maintainer asked for the change, or when a change elsewhere forces it, such as renaming a cited path.
  Write the patch outside the repository, link it, and say which parts of it no review has seen.
  Once the maintainer approves it, apply exactly that patch with `git apply`, and commit it; if it no longer applies, regenerate it and ask again.
