Decisions and recommendations
=============================

Maintainers and agents follow this page; other contributors need not read it.
It covers adding process, recommendations to whoever decides or to another contributor, a standard format for presenting decisions, and standard triage, in which independent agents recommend how to settle decisions without asking.
A report of requested work, such as a hand-off or a review's findings with their suggested fixes, is not a recommendation unless it asks its recipient to choose or to take another step.

.. _decisions-adding-process:

Adding process
--------------

Process is what is added to how work is done: rules, review steps, checks, experiments, measurement protocols, and the scripts and tests that serve them.
It excludes the library, the tests of the library's behavior, and what a standard already requires.
Process can grow to serve itself, as when each fix to a benchmark protocol adds a rule to cover a gap that the previous rule left.

* Before adding or recommending process that neither the request nor the rules call for, say in one sentence, in the recommendation or the report, what the work as a whole was requested for, and check that the addition serves that purpose, not only its own.
* A fix that would add process to guard against a problem that other process caused or left open is a decision for the person who requested the work, even if they delegated other decisions.
  Offer removing or simplifying that other process as one of the decision's options.
* A rule for working on qLDPC earns its place only if, without it, contributors would do something that a maintainer would have to correct.
  Prefer a principle with an example to a procedure, and sharpening or cutting a rule to adding one.
  In a review of a change to the rules, a gap that a good-faith reader would resolve correctly is not worth a new clause.

Principles for recommendations
------------------------------

Constraints
~~~~~~~~~~~

Consider only options that meet these constraints in what they change; if none does, redesign or escalate.

* **Correctness:** for supported inputs, a correct result or an explicit failure, never a mathematically wrong or materially misleading result.
* **Transparency:** each result that users see says whether it is exact, a bound, an estimate, or a trusted input, with the assumptions and limits needed to interpret it.
* **Standards:** the :doc:`standards <standards/index>` are met, unless changing a standard is the decision.
* **Task limits:** the limits that the task sets, such as a memory ceiling, are met.

Values
~~~~~~

Among the options that meet the constraints, the first of these values on which they :ref:`materially differ <decisions-material>` decides, whatever the later values say:

* **Capability:** the :ref:`key uses <decisions-key-uses>` that users can carry out within practical time and memory.
* **Simplicity:** the lasting code, API, state, and concepts that the option adds, for the code and for its users.
* **Reproducibility:** whether someone can reconstruct the inputs and configuration, repeat the work, and check it.
* **Performance:** the absolute time and memory that it saves or costs on the key uses.
* **Compatibility:** the existing uses of public behavior that must change, and the cost of migrating them.

A recommendation says what decided it, where possible in a sentence that it already needs: a value, by name; a constraint that leaves one option; or, when the evidence cannot separate the options, :ref:`reversibility <decisions-reversibility>`.
It adds two lines when it has something to say in them, and the :ref:`standard format <decisions-standard-format>` always has them:

* **Tension:** the values or evidence that pull the other way, and what the choice costs on them; or "none".
* **Independent opinion:** "not warranted", "suggested", or "obtained", with whether it agreed.
  One is warranted when correctness is disputed, the evidence for a consequential choice is weak, an error would be costly or hard to undo, or the recommender has a stake in an option beyond having written or proposed it.
  Whoever gives it gets the problem, the options, the constraints, the key uses, and the evidence, but not the recommendation; so a review whose reviewers saw the recommendation is not one.

Evidence
~~~~~~~~

* State each claim that matters to the decision with its scope and source, and say whether the evidence verifies it, supports it, is merely consistent with it, leaves it conjectural, or refutes it.
  A fix needs nothing more when a test fails before it and passes after it.
* Match the claim with evidence that does not reuse the reasoning or code under test: a derivation or a primary source for a scientific guarantee, and a proof, an independent oracle, or brute force on small cases for an implementation.
  Support a claim about time or memory with representative measurements, saying which factors, such as the machine, the load, and a random seed, were varied and which were fixed.
* Before measuring to choose between options, write down what result would change the choice.
  An explanation of observed behavior, or an extrapolation from it, stays a conjecture until a test that it could have failed supports it.
* When several candidates, thresholds, or explanations were tried, say how many, and how the reported one was chosen.

.. _decisions-key-uses:

* **Key uses** are those that a maintainer or the project names, or that are observed in practice; cite the source of each, and treat a use that a contributor proposes as a conjecture until it is confirmed.

.. _decisions-material:

* A difference is **material** only if the evidence separates it from the noise and its absolute time, memory, or count matters on a key use.
  For example, a slowdown of an exact distance search is material if it exceeds both about 20% and 0.2 s per search.
* Judge the current choice by the same standards as the others: it gets neither a veto nor a free pass.

.. _decisions-checks:

* Before recommending, run the cheapest check whose plausible outcomes could reverse the recommendation.

.. _decisions-reversibility:

* When the evidence cannot separate the options, prefer a staged or reversible step that loses nothing material; otherwise gather evidence or escalate.

.. _decisions-standard-format:

The standard format
-------------------

Present a decision in the standard format to the person who decides it, when they ask for "the standard format", or when the decision affects many modules, risks a wrong result, would be hard to undo, trades off what only that person can weigh, or changes a public name, argument, mode, or default.
A decision to fix a reproduced bug without changing a public contract does not need the format.
Present any other decision as a plain question with choices: name the recommended choice, and say in a few plain words what a wrong choice would risk.
Between contributors who do not decide, as from a reviewer to a review's coordinator, use whatever form serves the recipient.

The format has these parts, in order, each as long as the stakes call for, omitting any that does not apply:

* **Question:** the decision, as one question.
* **Context:** what the decision is about, why it arises now, and what it affects.
* **Risk:** in plain terms, whether a silent wrong result is at risk, whether a call that works now breaks, or whether only runtime or an error message changes; and how hard a wrong choice would be to undo.
* **Options:** a table with a row for each option: what it does, its advantages, and its costs.
* **Recommendation:** the option and what decided it, with the **Tension** and **Independent opinion** lines.
* **Ask:** one question, with the options as its choices.

Give each decision its own Question and Ask; a report that leads into several decisions may first list them briefly.
Keep the presentation high-level and plain, with technical detail and a few rounded, absolute numbers only where they could change the choice.
Put supporting detail, such as the evidence for each claim, after the Ask or in a linked file.

.. _decisions-standard-triage:

Standard triage
---------------

Standard triage settles, without asking the person who decides, the decisions that you would otherwise present to them.
It applies only when the person you work for asks for it by name, and only to the work that they ask it for, until they end it.
It does not apply because they are away or a question went unanswered.

For each such decision, or for a few closely related ones together:

1. Write a brief that presents the decision as the :ref:`standard format <decisions-standard-format>` would, with its scope, risk, and options, and with the constraints, key uses, and evidence, but without your recommendation.
2. Give the brief to at least two independent agents, each with fresh context and without the others' answers.
   Ask each to evaluate the decision critically, such as whether the brief misstates its scope or risk, misses or misdescribes an option, or rests on weak evidence.
   Then ask each to recommend an option and explain it, citing what decided it, such as a constraint, a value, a standard, a project goal, or the evidence, and to state its tension, as the principles for recommendations describe.
3. If they disagree with each other or with you, reconcile them: find the claim or value on which they split, and settle it by evidence, such as the cheapest check that could decide it, or by the principles above, such as the order of values or reversibility.
   You may give each the other's argument and ask for a reply.
   If they still disagree, leave the decision open.
4. Take the reconciled recommendation, unless triage leaves the decision open, and continue the work.
   Where practical, make each decision in a commit of its own, so that it can be reverted alone.

Triage leaves open each decision to:

* change public behavior, such as a public name, signature, argument, mode, default, or result, or deprecate or remove one, except to fix a reproduced bug without changing a public contract;
* weaken a guarantee, such as by loosening a test, tolerance, or check, by skipping a test or marking it as expected to fail, or by changing whether a result is exact, a bound, or an estimate;
* change how work on qLDPC is done, such as by adding the process that the rules on :ref:`adding process <decisions-adding-process>` cover, or by applying a patch to a protected file;
* change the dependencies or the build;
* go beyond the request, such as by starting work that it did not ask for or by exceeding the limits that it set;
* act outside the local repository, such as by pushing or by commenting on an issue or pull request, or rewrite or delete work that is not yours;
* waive a review or a check that the rules call for.

Triage still prepares a recommendation for each decision that it leaves open.
Continue with the work that an open decision does not block, and stop the work that it blocks.
For example, if a review finds that a fix would change a public default, leave that fix open, and finish and review the rest of the change.

In the hand-off or the report on the work, present each decision that triage settled or left open in the standard format.
Give the agents' recommendations, the reasons and tensions that they state, and how any disagreement was reconciled; and for a settled decision, the option taken and how to undo it, such as the commit to revert.
