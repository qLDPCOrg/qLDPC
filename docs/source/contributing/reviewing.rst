Reviewing changes
=================

In a review, independent reviewers examine a change from several angles, and the change is revised in response.
Review and revision repeat in :ref:`rounds <review-rounds>`.
qLDPC has two kinds of review:

* A **full review** gives each angle that the change can affect its own reviewer, except as the :ref:`grouping rules <review-grouping>` allow.
  It suits a change where a missed problem could give a wrong result that tests are unlikely to catch.
  Examples are changes to code constructions, code-core algebra or cached state, decoders, circuit or tableau tools, and scientific claims in documentation.
  It also suits a change to a public name, argument, mode, or default, and a change too large for one reviewer to read in full.
* A :ref:`quick review <quick-review>` has a single reviewer, who covers every angle, and at most two rounds.
  It suits most other small changes.

To request a review, name it, as in "run a full review of this branch" or "run a quick review of this branch".
A request can narrow or extend the review, as in "full review of the tests and prose".
Instructions in the request take precedence over this page.

Scope
-----

A review covers the changes of a pull request, or of a branch since its merge base with the main branch, including uncommitted changes.
Find the merge base the same way as the incremental (``-i``) mode of the :ref:`validation commands <validation-commands>`.
Reviewers read changed files in full, not only the diff.

Coordinator and reviewers
-------------------------

The person or agent who runs a review is its coordinator.
The coordinator starts the reviewers and revises the change, but does not review it.
A reviewer is a person or agent who did not write the change and who examines it critically from one or more angles.
An agent coordinator runs each reviewer as a separate subagent with its own context.
If it cannot, it says so in its report rather than reviewing the change itself.

The coordinator gives each reviewer:

* the purpose of the change and its scope;
* the reviewer's angles and the standards that define them;
* the decisions made so far, such as those of earlier rounds, so that reviewers do not reopen them;
* the output of the checks that the coordinator ran for the round, as step 1 of the :ref:`rounds <review-rounds>` describes;
* these instructions:

  * Report findings without editing the change.
  * Give each finding a location, the problem, why it matters, and a suggested fix.
  * Mark a finding significant if it asks for a closer look at an angle, as the next instruction describes, or if the change has or causes one of these problems:

    * a wrong result;
    * a broken contract;
    * an unhandled edge case that gives a wrong result or fails silently;
    * an interface or design problem;
    * claimed behavior that no test verifies;
    * documentation that states something false.

    Mark every other finding about the change minor.
  * If you cover several angles and one of them needs a closer look than you can give it, report that in a finding that names the angle.
  * Report problems that existed before the change separately, as pre-existing problems.
    A problem that the change causes in unchanged code, such as a broken caller, is a finding about the change.
  * Do not rerun the checks whose output you were given.
    Run other drafting checks of the :ref:`validation commands <validation-commands>` only as your angles need, such as a targeted test of an edge case, and run ``format_.py`` only with ``--check``.
  * Check whether the change requires updates elsewhere, such as to callers, to docstrings or guides that repeat a changed limitation, or to the :doc:`library map </library_map>` and examples.

Angles
------

A review covers each of these angles that the change can affect:

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

.. _review-grouping:

Grouping angles
---------------

In a full review, the coordinator may group angles under one reviewer within these limits:

* Clarity, Simplicity, Code standard, Writing standard, and Deprecations may share reviewers in any grouping.
* Correctness and Testing standard may share a reviewer with each other, but not with other angles.
* Correctness has a reviewer of its own for a change to a code construction, code-core algebra or cached state, a decoder, or a circuit or tableau tool.

The coordinator may also split an angle among several reviewers, such as one Correctness reviewer for each subsystem of a large change.
If a reviewer asks for a closer look at an angle, that angle has a reviewer of its own in the next round, and that reviewer examines the whole change.

.. _quick-review:

Quick review
------------

A quick review follows the :ref:`rounds <review-rounds>` below, with these differences:

* Each round has a single reviewer, who covers every angle that the change, or in the second round its fixes, can affect.
* The second round reviews only the fixes of the first round and what they affect.
* The review ends after two rounds.
* If the reviewer asks for a closer look at an angle, the review ends without passing.

If a quick review does not pass, suggest a full review.

.. _review-rounds:

Rounds
------

The coordinator runs each round as follows:

1. Run the anytime checks and ``coverage_.py -i`` of the :ref:`validation commands <validation-commands>`, unless they already ran after the last revision.
   Give their output to every reviewer.
   Start the reviewers together, so that none sees the findings of another.
   In a full review, the first two rounds examine the whole change, except as step 5 allows.
   Later rounds review the revisions since the previous round and what they affect.
   A round after the first may skip an angle that no revision since the previous round can affect.
2. Fix clear findings about the change, and verify each fix with the drafting checks.
   If the coordinator did not write the change, as for another contributor's pull request, report fixes as suggestions instead of making them.
   The review then ends after one round.
3. Leave decisions to the person the coordinator works for.
   Decisions include design changes, public API changes, changes to the standards or to this protocol, and disagreements with a reviewer.
   Record each open decision, continue with the other fixes, and list the open decisions in the report.
   Present each decision in plain terms: whether a silent wrong result is at risk, whether a call that works now breaks, or whether only runtime or an error message changes.
   Then give the options, their costs, and a recommendation.
4. Fix a minor pre-existing problem in the change if its fix is local and needs none of these decisions, such as a misleading message, a missing check, or an untested edge case.
   Like other fixes, they are revisions that any later round reviews.
   Report every such fix explicitly, under its own heading, so that the person the coordinator works for sees each change that they did not ask for.
   What to do about any other pre-existing problem that can give a wrong result or fail silently is a decision.
   Recommend fixing it in the change, in a follow-up pull request, or in a small pull request that precedes the change.
   List the remaining pre-existing problems in the report, without raising them as decisions.
5. Start another round with fresh reviewers if the round had a significant finding, unless four rounds have run, or two in a quick review.
   In a full review, the next round may review only the fixes if every significant finding was fixed by a local, evidently correct edit, and no reviewer asked for a closer look at an angle.
   Edits that change logic, an interface, or how parts of the code interact are not local.
   Minor findings and pre-existing problems do not start another round.
   If a significant problem needs a decision, or, from the third round on, needs more than a local fix outside the revisions, report it and ask how to proceed, such as by splitting the change, before starting another round.

A review passes when a round has no significant finding.
After the last round, report whether the review passed, each round's reviewers and their angles, what each round found and changed, the pre-existing problems that the review fixed, and the decisions that remain open.
A review does not run the finishing checks of the :ref:`validation commands <validation-commands>`, which belong to finishing a pull request.
If the review ends without passing, such as at its round limit, say so, and ask how to proceed, such as with another round or by splitting the change.
