Reviewing changes
=================

In a standard review, independent reviewers examine a change from several angles, and the change is revised in response.
Review and revision repeat in rounds, as :ref:`review-rounds` describes.

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
* these instructions:

  * Report findings without editing the change.
  * Give each finding a location, the problem, why it matters, and a suggested fix.
  * Mark a finding significant only if the change has or causes one of these problems:

    * a wrong result;
    * a broken contract;
    * an unhandled edge case;
    * an interface or design problem;
    * claimed behavior that no test verifies;
    * documentation that states something false.

    Mark every other finding about the change minor.
  * Report problems that existed before the change separately, as pre-existing problems.
    A problem that the change causes in unchanged code, such as a broken caller, is a finding about the change.
  * Run only the drafting checks of the :ref:`validation commands <validation-commands>`, and run ``format_.py`` only with ``--check``.
  * Check whether the change requires updates elsewhere, such as to callers, to docstrings or guides that repeat a changed limitation, or to the :doc:`library map </library_map>` and examples.

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

.. _review-rounds:

Rounds
------

The coordinator runs each round as follows:

1. Start the reviewers together, so that none sees the findings of another.
   The first two rounds review the whole change.
   Later rounds review the revisions since the previous round and what they affect.
   A round after the first may skip an angle that no revision since the previous round can affect.
2. Fix clear findings about the change, and verify each fix with the drafting checks.
   If the coordinator did not write the change, as for another contributor's pull request, report fixes as suggestions instead of making them.
   The review then ends after one round.
3. Leave decisions to the person the coordinator works for.
   Decisions include design changes, public API changes, changes to the standards or to this protocol, and disagreements with a reviewer.
   Record each open decision, continue with the other fixes, and list the open decisions in the report.
4. Fix a minor pre-existing problem in the change if its fix is local and needs none of these decisions, such as a misleading message, a missing check, or an untested edge case.
   Like other fixes, they are revisions that any later round reviews.
   Report every such fix explicitly, under its own heading, so that the person the coordinator works for sees each change that they did not ask for.
   What to do about any other pre-existing problem is a decision.
   Recommend fixing it in the change, in a follow-up pull request, or in a small pull request that precedes the change.
5. Start another round with fresh reviewers if the round had a significant finding, unless four rounds have run.
   If every fix of a significant finding was a local, evidently correct edit, the next round may review only those fixes, with one reviewer covering up to two angles.
   Edits that change logic, an interface, or how parts of the code interact are not local.
   The coordinator judges each fix.
   Minor findings and pre-existing problems do not start another round.
   From the third round on, neither does a significant problem outside the revisions that needs more than a local fix; report it and recommend how to proceed instead.
   If significant problems remain as open decisions, report them and ask how to proceed before starting another round.

A standard review passes when a round has no significant finding.
After the last round, report whether the review passed, what each round found and changed, the pre-existing problems that the review fixed, and the decisions that remain open.
A standard review does not run the finishing checks of the :ref:`validation commands <validation-commands>`, which belong to finishing a pull request.
If the review ends without passing, such as at the four-round limit, say so, and ask how to proceed, such as with another round or by splitting the change.
