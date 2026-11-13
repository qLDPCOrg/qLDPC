Decisions and recommendations
=============================

Contributors, whether people or agents, often recommend a choice to the person who decides it, usually a maintainer.
They also recommend choices to each other, as when a reviewer recommends one of several fixes to a review's coordinator.
This page states the principles that every recommendation follows, in any form, and a standard format for presenting a complex or risky decision to the person who decides it.
A report of requested work that is already done, such as a handoff or a review's findings with their suggested fixes, is not a recommendation, unless it asks its recipient to choose among options or to take an additional step.

Principles for recommendations
------------------------------

Constraints
~~~~~~~~~~~

An option must meet these constraints to be considered at all.
If no option meets them all, redesign or escalate; never use the values to pick the least bad violation.
The constraints apply to what an option changes, not to a pre-existing problem that it leaves for another pull request.

* **Correctness:** for supported inputs, the option does not return a mathematically wrong or materially misleading result instead of a correct result or an explicit failure.
* **Transparency:** where a user encounters a result, it says whether it is exact, a bound, an estimate, or a trusted input, with the assumptions and limits needed to interpret it.
* **Standards:** the option meets the :doc:`standards <standards/index>`, unless changing a standard is itself the decision.
* **Task limits:** the option meets any limits that the task sets, such as a memory ceiling or a required capability.

Values
~~~~~~

The values and the rules of evidence below compare options on the :ref:`key uses <decisions-key-uses>`: the uses that a maintainer or the project names, or that are observed in practice.
Among the options that meet the constraints, the values apply in order: the highest value on which the options :ref:`materially differ <decisions-material>` decides, whatever the lower values say (a lexicographic order).
Cite values and named rules by name, never by number.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Question to put to each option
   * - **Capability**
     - Which of the key uses can users carry out with one option but not the other, within practical limits of time and memory?
   * - **Simplicity**
     - How much lasting code, API, and state, and how many concepts, does it add, for the code and for its users?
   * - **Reproducibility**
     - Can someone reconstruct the inputs and configuration, repeat the work, and check it independently?
   * - **Performance**
     - On the key uses, what absolute time and memory does it save or cost?
   * - **Compatibility**
     - For public behavior, how many existing uses must change, and at what migration cost?

What a recommendation states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every recommendation states what decided it: the deciding value by name, or, if no value decides, the constraint that leaves only one eligible option, or the rule :ref:`Reversibility as a tie-breaker <decisions-reversibility>` below.
A sentence that the recommendation already needs can say this.
Apply the other principles without saying that you applied them; state only what they ask you to state.
A recommendation also gives these lines when it has something to say in them, and the :ref:`standard format <decisions-standard-format>` always gives them:

* **Tension:** the values or evidence that pull the other way, by name, and what the choice costs on them; or "none".
* **Independent opinion:** "not warranted", "suggested", or "obtained", with whether it agreed.

An independent opinion is warranted when correctness is disputed, the evidence for a consequential choice is weak, an error would be costly or hard to undo, or the recommender has a stake in one option.
Having written or proposed an option is not by itself a stake.
The rule :ref:`Proportionate checks <decisions-checks>` below says how to obtain one, and what does not count as one.

Evidence
~~~~~~~~

These rules of evidence always apply:

* State each claim that matters to the decision with its scope, its source, and how well it is established.
  Evidence of correctness must not be circular.
* Judge measurements in absolute terms on the key uses.
* Do not credit an explanation of observed behavior until a test that it could fail has supported it.

The full rules below apply only when a recommendation depends on any of these:

* establishing that a scientific guarantee applies;
* deciding whether measurements are representative, or whether their differences are material;
* a causal explanation, or an extrapolation from observed behavior;
* a disputed or materially uncertain claim whose resolution could change the recommendation.

Examples include choosing a default or a heuristic from benchmarks, recommending a change because of a material performance claim, choosing among live hypotheses of a bug's cause, and resolving a disputed claim of correctness.
Otherwise, the constraints, the values, and the three rules above suffice, together with the definitions that they link to.
A fix needs only those three rules when a test fails before it and passes after it.

**Evidence for claims**

* **Prospective, falsifiable tests.**
  Before a new measurement is used to choose, record the key use, the outcome that would change the choice, and, where feasible, a practical threshold for a material difference, in a durable record made before measuring, such as a file, a commit, or a test assertion.
  Before crediting a mechanism or an extrapolation, subject it to a severe test: record a nontrivial prediction that distinguishes it from a live alternative, and test it on a new condition or by intervention.
  A post-hoc or patched explanation remains a conjecture until a new prediction is recorded and supported.
  A prediction is nontrivial if the setup and the observations so far do not already force it, some plausible result would count against it, and it distinguishes the claim from a live alternative.
  Static checks, regression tests, reproductions of known results, and retrievals of existing evidence need no separate prediction; a test assertion written before the test runs records one.
* **Matched evidence.**
  Match the evidence to the claim:

  * a derivation or a primary source for a scientific guarantee;
  * a proof, an independent oracle, or a non-circular invariant or test for the correctness of an implementation;
  * representative measurements for time and memory;
  * discriminating evidence for a causal mechanism.

  An oracle is independent only if it does not reuse the reasoning or implementation under test.
  For factors that are likely to affect a measurement but are not under test, such as the machine, the load, the order of runs, and a random seed, report which were varied and which were fixed.
  A trace establishes what executed, not what caused an effect.
  A mechanism raises confidence that an effect generalizes, but does not enlarge the effect or override contrary measurements.
* **Calibrated claims.**
  When stating how well a claim is established, say whether the evidence verifies it within its scope, supports it, is merely consistent with it, leaves it conjectural, or refutes it.
  Do not label incidental facts.

**Comparing options**

.. _decisions-key-uses:

* **Sourced key uses.**
  Judge capability, practicality, and representativeness on the key uses, and cite the source of each.
  Treat a use that a contributor proposes as a conjecture until it is confirmed, rather than substituting the costliest conceivable case.

.. _decisions-material:

* **Material effects in absolute terms.**
  A difference counts only when the evidence separates it from the relevant uncertainty and its absolute time, memory, or count matters on a key use; ratios are supplementary.
  Report a contrary effect judged immaterial only when it lies near the threshold or its materiality could reasonably be disputed.
  When several candidates, thresholds, or explanations were tried, say how many, and how the reported one was chosen.
* **Symmetric comparison.**
  Apply the same constraints, standards of evidence, and values to the current and the proposed choices.
  The current choice gets neither a veto nor a free pass; its demonstrated reliability, its existing validation, and its migration cost count through the values.

**Residual uncertainty**

.. _decisions-checks:

* **Proportionate checks.**
  Before recommending, run the shortest available check whose plausible outcomes could reverse the recommendation.
  When an independent opinion is warranted, suggest one, or obtain one if your task allows.
  Give whoever gives it the problem, the options, the constraints, the key uses, and the evidence, but not the recommendation.
  A review whose reviewers saw the recommendation is not an independent opinion of it.

.. _decisions-reversibility:

* **Reversibility as a tie-breaker.**
  When the evidence cannot distinguish the options under the values, prefer a staged or reversible step if it introduces no known material loss under them.
  Otherwise, gather evidence or escalate.

.. _decisions-standard-format:

The standard format
-------------------

Present a decision to the person who decides it in this standard format when it is sufficiently complex, technical, wide-reaching, or risky, or when that person asks for "the standard format".
Examples are changes to a public name, argument, mode, or default.
Others are choices that affect many modules, risk a wrong result, or would be hard to undo, and trade-offs that only that person can weigh.
A decision to fix a reproduced bug without changing a public contract does not need the format.
Present any other decision as a plain question with choices: name the recommended choice, and say in a few plain words what a wrong choice would risk, as the Risk part below does.
A recommendation to a contributor who does not decide it, as from a reviewer to a review's coordinator, follows the principles above, in whatever form serves the recipient.

The format has these parts, in order, at a length proportional to the decision's stakes; omit a part that does not apply:

* **Question:** the decision, as one question.
* **Context:** what the decision is about, why it arises now, and what it affects, such as files, public names, users, or other decisions.
* **Risk:** in plain terms, whether a silent wrong result is at risk, whether a call that works now breaks, or whether only runtime or an error message changes; and how hard a wrong choice would be to undo.
* **Options:** a table with a row for each option, giving what it does, its advantages, and its costs.
* **Recommendation:** the option recommended and what decided it, with the **Tension** and **Independent opinion** lines.
* **Ask:** one question, with the options as its choices.

Give each decision its own Question and Ask, rather than merging decisions into one question.
A report that leads into several decisions presents each in turn, and may first summarize them all briefly, as a review's report lists its open decisions.

By default, keep the presentation high-level and the language plain.
Include technical detail, and a few rounded numbers in absolute terms, only where they could change the choice.
When a decision turns on a technical point, say so in the Question or the Context, and give the least detail needed to judge it.
Put supporting detail, including the statements that the rules of evidence call for, after the Ask, or in a linked file, and define each term once.
