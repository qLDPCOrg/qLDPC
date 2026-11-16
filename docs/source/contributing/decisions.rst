Decisions and recommendations
=============================

Maintainers and agents follow this page; other contributors need not read it.
It covers adding process, recommendations to whoever decides or to another contributor, and a standard format for presenting a complex or risky decision.
A report of requested work, such as a hand-off or a review's findings, is not a recommendation unless it asks its recipient to choose or to take another step.

.. _decisions-adding-process:

Adding process
--------------

Process is what is added to how work is done, beyond the library, the tests of its behavior, and what a standard requires: rules, review steps, checks, experiments, measurement protocols, and the scripts and tests that serve them.
Process can grow to serve itself, as when each fix to a benchmark protocol adds a rule to cover a gap that the previous rule left.

* Before adding or recommending process that neither the request nor the rules call for, say in one sentence, in the recommendation or the report, what the work as a whole was requested for, and check that the addition serves that purpose, not only its own.
* A fix that would add process to guard against a problem that other process caused or left open is a decision for the person who requested the work, even if they delegated other decisions.
  Offer removing or simplifying that other process as one of its options.
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

Among those options, the first of these values on which they :ref:`materially differ <decisions-material>` decides, whatever the later values say:

* **Capability:** the :ref:`key uses <decisions-key-uses>` that users can carry out within practical time and memory.
* **Simplicity:** the lasting code, API, state, and concepts that the option adds, for the code and for its users.
* **Reproducibility:** whether someone can reconstruct the inputs and configuration, repeat the work, and check it.
* **Performance:** the absolute time and memory that it saves or costs on the key uses.
* **Compatibility:** the existing uses of public behavior that must change, and the cost of migrating them.

A recommendation says what decided it, in a sentence that it already needs where possible: a value, by name; a constraint that leaves one option; or, when the evidence cannot separate the options, :ref:`reversibility <decisions-reversibility>`.
When it has something to say in them, and always in the standard format, it adds two lines:

* **Tension:** the values or evidence that pull the other way, and what the choice costs on them; or "none".
* **Independent opinion:** "not warranted", "suggested", or "obtained", with whether it agreed.
  One is warranted when correctness is disputed, the evidence for a consequential choice is weak, an error would be costly or hard to undo, or the recommender has a stake in an option beyond having proposed it.
  Whoever gives it gets the problem, the options, the constraints, the key uses, and the evidence, but not the recommendation; so a review whose reviewers saw the recommendation is not one.

Evidence
~~~~~~~~

* State each claim that matters to the decision with its scope and source, and say whether the evidence verifies it, supports it, is merely consistent with it, leaves it conjectural, or refutes it.
  A fix that a test shows failing before and passing after needs nothing more.
* Match the evidence to the claim, independently of what it checks: a derivation or a primary source for a scientific guarantee; a proof, an independent oracle, or brute force on small cases for an implementation; and representative measurements for time and memory, saying which factors, such as the machine, the load, and a random seed, were varied and which were fixed.
* Before a measurement chooses between options, write down what result would change the choice.
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

Present a decision to the person who decides it in the standard format when they ask for "the standard format", or when the decision changes a public name, argument, mode, or default, affects many modules, risks a wrong result, would be hard to undo, or trades off what only that person can weigh.
A decision to fix a reproduced bug without changing a public contract does not need the format.
Present any other decision as a plain question with choices that names the recommended choice and says in a few plain words what a wrong choice would risk.
Between contributors who do not decide, as from a reviewer to a review's coordinator, use whatever form serves the recipient.

The format has these parts, in order, each as long as the stakes call for, omitting any that does not apply:

* **Question:** the decision, as one question.
* **Context:** what the decision is about, why it arises now, and what it affects.
* **Risk:** in plain terms, whether a silent wrong result is at risk, whether a call that works now breaks, or whether only runtime or an error message changes; and how hard a wrong choice would be to undo.
* **Options:** a table with a row for each option: what it does, its advantages, and its costs.
* **Recommendation:** the option and what decided it, with the **Tension** and **Independent opinion** lines.
* **Ask:** one question, with the options as its choices.

Give each decision its own Question and Ask; a report that leads into several decisions may first list them briefly.
Keep the presentation high-level and plain, with technical detail and a few rounded, absolute numbers only where they could change the choice, and put supporting detail after the Ask or in a linked file.
