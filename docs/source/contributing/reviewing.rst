Reviewing changes
=================

Maintainers and agents review changes as this page describes; other contributors need not read it.
In a review, reviewers who did not write a change examine it from several angles, and the change is revised in response, in :ref:`rounds <review-rounds>`.
A change is **high-risk** if it could alter results or behavior in a way that tests would likely miss, or if it adds or corrects a documented claim about results, guarantees, or limitations.
Such changes are most common in code constructions, code-core algebra or cached state, decoders, circuit or tableau tools, external integrations, and caching.
qLDPC has two kinds of review:

* A **full review** gives each angle that the change can affect its own reviewer, except as the :ref:`grouping rules <review-grouping>` allow.
  It suits a high-risk change, a change to what the rules for working on qLDPC require, a change too large for one reviewer to read in full, and a change to a public name, argument, mode, or default.
* A :ref:`quick review <quick-review>` has at most two rounds, each with one reviewer.
  It suits every other change, such as one that changes only tests, renames private names, or rewords, moves, or cuts documentation without adding or correcting a claim.

Run the kind of review that a request names, and otherwise the kind that the change suits.
A kind that a request names, and an instruction not to run a full review, hold for the rest of the work on the change, unless a revision gives the change a new reason to suit a full review.
Instructions in the request, such as to narrow or extend the review, take precedence over this page.
Where this page does not settle a case, choose by the risk of a wrong result that tests would miss, and say in the report what you chose and why.

Scope
-----

A review covers the changes of a pull request, or of a branch since its merge base with the main branch, including uncommitted changes.
Find the merge base as the incremental (``-i``) mode of the :ref:`validation commands <validation-commands>` does.
Reviewers read changed files in full, not only the diff.

Coordinator and reviewers
-------------------------

The person or agent who runs a review, its coordinator, starts the reviewers and revises the change, but does not review it.
An agent coordinator runs each reviewer as a separate subagent with its own context, or says in its report that it could not.

The coordinator gives each reviewer:

* the purpose and scope of the change and of the review, in the requester's words where they gave them;
* the reviewer's angles and the standards that define them;
* the decisions made so far, so that reviewers do not reopen them;
* the findings of earlier rounds and how each was handled;
* the output of the :ref:`round's checks <review-rounds>`;
* these instructions:

  * Report findings without editing the change, each with a location, the problem, why it matters, and a suggested fix.
  * Mark a finding significant only if the change has or causes one of these problems:

    * a wrong result, a broken contract, or an unhandled edge case that gives a wrong result or fails silently;
    * an interface or design problem;
    * claimed behavior that no test verifies;
    * documentation that states something false, or something that could mislead a reader about what the code does or guarantees;
    * documentation that leaves out a behavior or limitation without which a reader could get a wrong or materially misleading result without an error.

    Mark every other finding minor.
    For a change to the rules for working on qLDPC, a gap or ambiguity in the rules is significant exactly when a contributor who follows them in good faith would plausibly miss it and do something that a maintainer would have to correct.
  * Ask for documentation only where a standard calls for it; for a behavior that the writing standard says to state, name the reasonable expectation, its source, and the wrong result that it would give.
  * Report problems that existed before the change separately, as pre-existing problems; a problem that the change causes in unchanged code, such as a broken caller, is a finding about the change.
  * Check whether the change needs updates elsewhere, such as to callers, to documents that repeat a changed limitation, or to the :doc:`library map </library_map>` and examples.
  * Do not rerun the checks whose output you were given, and run only the other drafting checks that your angles need, with ``format_.py`` only in ``--check`` mode.
    No angle requires a check beyond the drafting checks.
  * If you cannot read your share of the change in full, or check it as an angle requires, ask for a closer look, separately from your findings, and say what you could not read or check.

Angles
------

A review covers each of these angles that the change can affect, and each angle covers only the changes in its area, even when a request asks for all angles:

* **Correctness:** results, edge cases, error handling, and documented contracts, checked against independent oracles where possible; the :doc:`implementation <standards/implementation>` and :doc:`decoder <standards/decoders>` standards in their areas; and, for a refactor, unchanged behavior.
  For a change to the rules for working on qLDPC, it also covers what a good-faith reader would do under them.
* **Clarity:** names and commented paragraphs of code, as :ref:`simplicity and clarity <simplicity-and-clarity>` describes, and whether comments and docstrings match the code.
* **Simplicity:** code only as complex as its uses require, as the same section describes.
* **Code standard:** the rest of the :doc:`code standard <standards/code>`.
* **Testing standard:** the :doc:`testing standard <standards/testing>`.
* **Writing standard:** the :doc:`writing standard <standards/writing>`, for every changed comment, docstring, document, and notebook.
  Read the changed prose cold first, as a reader who has not seen the code, and note each reference without an antecedent and each sentence that such a reader would have to reread; then check the prose against the code.
* **Deprecations:** the :doc:`deprecation standard <standards/deprecations>`, for a change to a public name, argument, mode, or default.

.. _review-grouping:

Assigning angles to reviewers
-----------------------------

Clarity, Simplicity, Code standard, Writing standard, and Deprecations may share a reviewer in any combination.
Correctness and Testing standard may share a reviewer only with each other, and Correctness has a reviewer of its own for a high-risk change.
Give each reviewer a share of the change that it can read in full, splitting a large share, such as by subsystem.

If a reviewer asks for a closer look at an angle, split its share for that angle among added reviewers in the same round, before revising the change, and brief them with the reason; later rounds split that share at least as finely.
If no split can meet the request, as in a quick review or when no reviewer can run a check that the angle requires, the request is a significant finding that needs a decision.

.. _review-rounds:

Rounds
------

A review runs in rounds until one has no significant finding, which passes the review, or until four rounds have run; minor findings and pre-existing problems start no round.
Each round has fresh reviewers, who do not see each other's findings, and starts with the anytime checks and ``coverage_.py -i`` of the :ref:`validation commands <validation-commands>`, unless they already ran on the change as it stands.
The first two rounds review the whole change; later rounds review the revisions since the previous round started, and what they affect, and may skip an angle that those revisions cannot affect.
A revision is local if it is not high-risk and changes no interface and no rule for working on qLDPC.

After each round, the coordinator:

* fixes clear findings about the change, and verifies each fix with the drafting checks;
* fixes a minor pre-existing problem in the change, such as a misleading message or an untested edge case, if the fix is local, evidently correct, and needs no decision;
* reports each such fix under its own heading;
* leaves decisions to the person it works for, presents each as :doc:`decisions and recommendations <decisions>` describes, and continues with the other fixes;
* if a significant problem needs a decision, or, from the third round on, needs a non-local fix outside the reviewed revisions, reports it and asks how to proceed before another round, such as by splitting the change.

Decisions, unless already made, include design and public API changes, changes to the standards or to this page, disagreements with a reviewer, fixes that the rules on :ref:`adding process <decisions-adding-process>` make into decisions, and what to do about any other pre-existing problem that can give a wrong result or fail silently.
No round reviews the final round's fixes, so in the final round make only local, evidently correct fixes, and report any other fix as a suggestion.
Where possible, a final-round fix to prose keeps to wording that a reviewer saw.
A coordinator who may not revise the change, as for another contributor's pull request, reports every fix as a suggestion and ends the review after one round.

The report gives:

* whether the review passed, and, if not, a question on how to proceed, such as with another round, a full review after a quick review, or a split;
* each round's reviewers, angles, findings, and changes, and each request for a closer look and how it was handled;
* the fixes left as suggestions;
* the pre-existing problems that the review fixed, and the others;
* each unreviewed revision, such as the final round's fixes;
* the open decisions.

.. _quick-review:

Quick review
------------

A quick review follows the rest of this page, except that:

* each round has one reviewer, who covers all of the round's angles and runs a further drafting check only to confirm or rule out a specific problem;
* the second round reviews only the revisions since the first round started, and what they affect;
* the review ends after two rounds.
