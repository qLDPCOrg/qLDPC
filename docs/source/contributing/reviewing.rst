Reviewing changes
=================

In a review, one or more independent reviewers examine a change from several angles, and the change is revised in response.
Review and revision repeat in :ref:`rounds <review-rounds>`.
qLDPC has two kinds of review:

* A **full review** gives each angle that the change can affect its own reviewer, except as the :ref:`grouping rules <review-grouping>` allow.
  It suits a high-risk change: one that could alter results, behavior, or documented claims in a way that tests would likely miss.
  Such changes are most common in code constructions, code-core algebra or cached state, decoders, circuit or tableau tools, external integrations, caching, and scientific claims in documentation.
  A full review also suits a change too large for one reviewer to read in full, and a change to a public name, argument, mode, or default.
* A :ref:`quick review <quick-review>` has at most two rounds, each with one reviewer.
  It suits every other change in any area.
  Examples are changes to comments or docstrings that make no new claims, renames of private names, and test-only changes.

Run the kind of review that a request names, as in "run a quick review of this branch".
If no request names a kind, as in "review this branch" or when no one asked for a review, run a full review for a change that suits one, and a quick review otherwise.
A kind that a request names holds for the rest of the work on the change, and so does an instruction not to run a full review.
The exception is a revision that gives the change a new reason to suit a full review, as when work on one decoder also changes a code construction.
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
* the decisions made so far, so that reviewers do not reopen them;
* the findings of earlier rounds and how each was handled, so that reviewers can check the fixes;
* the output of the checks that step 1 of :ref:`each round <review-rounds>` describes;
* these instructions:

  * Report findings without editing the change.
  * Give each finding a location, the problem, why it matters, and a suggested fix.
  * Mark a finding significant only if the change has or causes one of these problems:

    * a wrong result;
    * a broken contract;
    * an unhandled edge case that gives a wrong result or fails silently;
    * an interface or design problem;
    * claimed behavior that no test verifies;
    * documentation that states something false.

    Mark every other finding about the change minor.
  * If you cannot read in full the changes in the area of one of your angles, or cannot check them as that angle requires, ask for a closer look at that angle.
    Make the request separately from your findings, and say what you could not read or check.
    No angle requires a check other than the drafting checks.
  * Report problems that existed before the change separately, as pre-existing problems.
    A problem that the change causes in unchanged code, such as a broken caller, is a finding about the change.
  * Do not rerun the checks whose output you were given.
    Run only those other drafting checks of the :ref:`validation commands <validation-commands>` that your angles need, such as a targeted test of an edge case.
    Run ``format_.py`` only with ``--check``.
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

Assigning angles to reviewers
-----------------------------

The coordinator may group angles under one reviewer within these limits:

* Clarity, Simplicity, Code standard, Writing standard, and Deprecations may share a reviewer in any combination.
* Correctness and Testing standard may share a reviewer only with each other.
* Correctness has a reviewer of its own for a high-risk change.

Whenever a reviewer's share of the change is too large to read in full, regroup that reviewer's angles or split the share, such as with one Correctness reviewer for each subsystem.

If a reviewer asks for a closer look at an angle, split that reviewer's share of the change for that angle among added reviewers in the same round, before revising the change.
Brief them like the other reviewers, and add the reason for the request.
If no split can meet a request, as in a quick review or when no reviewer can run a check that the angle requires, the request is a significant finding that needs a decision.
A later round that reviews the same share splits it at least as finely.

.. _review-rounds:

Rounds
------

The coordinator runs each round as follows:

1. Run the anytime checks and ``coverage_.py -i`` of the :ref:`validation commands <validation-commands>`, unless they already ran on the change in its current state.
   Give no reviewer the findings of another reviewer in the same round.
   The first two rounds review the whole change.
   Later rounds review the revisions since the previous round started and what they affect.
   A round after the first may skip an angle that no revision since the previous round started can affect.
2. Fix clear findings about the change, and verify each fix with the drafting checks.
   A revision, such as a fix, is local if it is not high-risk and changes no interface and no rule for working on qLDPC.
   No round reviews the final round's fixes.
   So in the final round, such as one with no significant finding or at the round limit, make only local, evidently correct fixes.
   Report any other fix as a suggestion.
   If the coordinator is not permitted to revise the change, as for another contributor's pull request, report every fix as a suggestion, and end the review after one round.
3. Leave decisions to the person the coordinator works for.
   Decisions include design changes, public API changes, changes to the standards or to this protocol, and disagreements with a reviewer.
   Record each open decision, continue with the other fixes, and list the open decisions in the report.
   Present each decision in plain terms: whether a silent wrong result is at risk, whether a call that works now breaks, or whether only runtime or an error message changes.
   Then give the options, their costs, and a recommendation.
4. Fix a minor pre-existing problem in the change, such as a misleading message, a missing check, or an untested edge case, if its fix is local, evidently correct, and needs no decision.
   Report every such fix explicitly, under its own heading, so that the person the coordinator works for sees each change that they did not ask for.
   What to do about any other pre-existing problem that can give a wrong result or fail silently is a decision.
   Recommend fixing it in the change, in a follow-up pull request, or in a small pull request that precedes the change.
   List the remaining pre-existing problems in the report, without raising them as decisions.
5. Start another round with fresh reviewers if the round had a significant finding, unless four rounds have run.
   Minor findings and pre-existing problems do not start another round.
   If a significant problem needs a decision, report it and ask how to proceed, such as by splitting the change, before starting another round.
   From the third round on, do the same for a significant problem that needs a non-local fix outside the reviewed revisions.

A review passes when a round has no significant finding.
After the final round, report:

* whether the review passed;
* each round's reviewers and their angles;
* what each round found and changed;
* each request for a closer look and how it was handled;
* the fixes left as suggestions;
* the pre-existing problems that the review fixed;
* each unreviewed revision, such as the final round's fixes;
* the decisions that remain open.

If the review ends without passing, such as at its round limit, say so and ask how to proceed.
Options include another round, a full review after a quick review, and splitting the change.

.. _quick-review:

Quick review
------------

A quick review follows the rest of this page, with these differences:

* Each round has one reviewer, who covers all of the round's angles.
* The second round reviews only the revisions since the first round started and what they affect.
* The review ends after two rounds.
