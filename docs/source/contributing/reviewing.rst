Reviewing changes
=================

In a standard review, independent reviewers examine a change from several angles, and the change is revised in response.
Review and revision repeat in rounds while a round finds significant problems.
A problem is significant if fixing it changes behavior, an interface, a test, or what documentation says, rather than only polishing wording.

To request a standard review, name it, as in "run a standard review of this branch".
A request can narrow or extend the review, as in "standard review of the tests and prose".
Instructions in the request take precedence over this page.

Scope
-----

A standard review covers the changes of a pull request, or of a branch since its merge base with the main branch, including uncommitted changes.
Find the merge base the same way as the incremental (``-i``) mode of the :ref:`validation commands <validation-commands>`.
Reviewers read changed files in full, not only the diff.

Coordinator and reviewers
-------------------------

The person or agent who runs a standard review is its coordinator.
The coordinator starts the reviewers and revises the change, but does not review it.
A reviewer is a person or agent who did not write the change and who examines it critically from one angle.
An agent coordinator runs each reviewer as a separate subagent with its own context.
If it cannot, it says so in its report rather than reviewing the change itself.

The coordinator gives each reviewer:

* the purpose of the change and its scope;
* the reviewer's angle and the standard that defines it;
* the decisions made so far, such as those of earlier rounds, so that reviewers do not reopen them;
* these instructions: report findings without editing the change, run only the drafting checks of the :ref:`validation commands <validation-commands>`, with ``format_.py --check`` rather than ``--fix``, and check whether the change requires updates elsewhere, such as to callers, to docstrings or guides that repeat a changed limitation, or to the :doc:`library map </library_map>` and examples.

A finding gives a location, the problem, why it matters, and a suggested fix.

Angles
------

A standard review has one reviewer for each of these angles that the change can affect:

* **Correctness:** results, edge cases, error handling, and documented contracts, checked against independent oracles where possible.
  Also check the :doc:`implementation standard <standards/implementation>` for a change to the areas it covers, and the :doc:`decoder standard <standards/decoders>` for a change to decoders or decoder plumbing.
  For a refactor, check that behavior is unchanged.
* **Clarity:** names and commented paragraphs of code, following :ref:`simplicity and clarity <simplicity-and-clarity>` in the code standard.
  Also check that comments and docstrings match what the code does.
* **Simplicity:** the rule to keep code only as complex as its uses require, in :ref:`simplicity and clarity <simplicity-and-clarity>`.
* **Code standard:** the rest of the :doc:`code standard <standards/code>`, including public and private names, module layout, and import layers.
* **Testing standard:** the :doc:`testing standard <standards/testing>`.
* **Writing standard:** the :doc:`writing standard <standards/writing>`, applied to every changed comment, docstring, document, and notebook.
* **Deprecations:** the :doc:`deprecation standard <standards/deprecations>`, for a change to a public name, argument, mode, or default.

An angle applies only to changes in its area, even when a request asks for all angles.
The :doc:`pull request standard <standards/pull_requests>` is not a review angle; it applies when preparing a pull request description.

One reviewer may cover up to two angles of a small change, such as a change to one module or to prose only.

Rounds
------

The coordinator runs each round as follows:

1. Start the reviewers together, so that none sees the findings of another.
   Each round reviews the whole change.
   A later round may skip an angle that no revision since the previous round can affect.
2. Fix clear problems, and verify each fix with the drafting checks.
   If the coordinator did not write the change, as for another contributor's pull request, report fixes as suggestions instead of making them.
   The review then ends after one round.
3. Leave decisions to the person the coordinator works for.
   Decisions include design changes, public API changes, changes to the standards or to this protocol, and disagreements with a reviewer.
   Record each open decision, continue with the other fixes, and list the open decisions in the report.
4. Start another round with fresh reviewers if the round found significant problems, unless three rounds have run.
   If significant problems remain as open decisions, report them and ask how to proceed before starting another round.

After the last round, report what each round found and changed, and list the decisions that remain open.
A standard review does not run the finishing checks of the :ref:`validation commands <validation-commands>`, which belong to finishing a pull request.
If the three-round limit ended the review, ask whether to start another round.
