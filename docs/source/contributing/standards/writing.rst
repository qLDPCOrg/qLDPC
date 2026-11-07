Writing standard
================

This standard covers prose in comments, docstrings, documentation, and example notebooks, and where each kind of documentation belongs.
The :ref:`code standard <simplicity-and-clarity>` covers names and the comments that outline long blocks of code.

Prose style
-----------

* Choose material for a document's audience and purpose.
  Keep docstrings focused on documented objects and contributor guidance focused on contribution decisions; omit tangents and prose that merely restates implementation details.
* Write and review prose by purpose.
  For each paragraph, docstring section, or comment, ask what its reader needs from it, and whether that reader would miss it if it were gone.
  If not, delete it, or point to where the information lives.
  If so, state its message in one line, then write the simplest text that gets that message across.
* Prefer short sentences with one idea each.
  Split a sentence with stacked clauses, parenthetical asides, or chains of semicolons rather than compressing it.
* Describe what code does, not what it does not do or how it used to work.
  Give readers a sense of what happens, and point to where details live rather than repeating them.
  Check that comments and docstrings match what the code does, and update them when the code changes.
* In substantial scientific exposition, introduce a problem and explain why it matters before describing implementation details.
  Worked examples also need enough context to explain the question they answer and what their results establish.
  Include background only where it helps intended readers.
* Introduce or explicitly identify an object, quantity, or concept before using a definite reference or pronoun.
  A clear name or defining description is sufficient; identical wording need not appear earlier.

  * **Unclear without prior context:** ``Run a simulation and check the estimate.``
  * **Clear:** ``Run a simulation to estimate a failure rate, then check that estimate.``

* Use the serial (Oxford) comma before the conjunction in a list of three or more items, as in ``X, Y, and Z``.
* Docstrings follow the Google convention.
* Python comments and docstrings use ordinary paragraphs: separate sentences with two spaces and fill lines to the 100-column limit.
* Keep one prose sentence per physical source line in repository ``.rst`` and ``.md`` files and notebook Markdown cells.
  Put ``First sentence.`` and ``Second sentence.`` on separate source lines; leave code blocks, equations, and table syntax intact.
  For PR-summary text, use the paragraph formatting in :ref:`pull-request-summaries` instead.

Documentation and examples
--------------------------

* Put rules that a change must satisfy in the :doc:`standards <index>`, one topic per standard, and state each rule in exactly one standard.
  Guides explain how qLDPC works and how to do things, and link to the standards rather than restating them.
* Put explanations that span packages in the :doc:`library map </library_map>`, exact API behavior in docstrings, and executable workflows in ``examples/``.
* The notebooks under ``docs/source/examples/`` are links to the canonical files under ``examples/``.
  Edit the canonical notebook, not the documentation-tree link.
* Keep README and the library map selective; AutoAPI lists the public symbols.
* When a public limitation changes, update the relevant function or class docstring and every README or guide that repeats it in the same pull request.
* Run the strict documentation build before considering the change complete.

Use the smallest meaningful example and explain the scientific choices and limitations, with pointers to the relevant literature and implementation.
Make examples easy to adapt to a new question rather than requiring users to adopt a complete workflow.

Example notebooks
~~~~~~~~~~~~~~~~~

Example notebooks should help readers understand and adapt a scientific workflow, not merely show that API calls run.
Use these criteria when writing new notebooks and reviewing existing ones:

* **Purpose:** introduce the question an example answers and why it is useful before presenting implementation details.
  Explain why the chosen code, model, or comparison helps answer that question.
  Orient introductions toward readers exploring the possibilities, not only those who have already adopted qLDPC or chosen a workflow.
* **Framing:** state what an example enables without inventing a mistaken approach or assuming the reader needs correction.
  Show readers what they can explore or accomplish, rather than prescribing what they should do.
  Describe the choices made in an example as choices for that example; reserve imperatives for procedural steps and genuine requirements.
  Avoid straw-man motivation: a contrast such as "do X rather than Y" needs a reason to discuss Y beyond making X sound useful.
  Retain contrasts that clarify scientific assumptions, genuine alternatives, or limitations.
* **Background and assumptions:** introduce the concepts, quantities, and notation needed to follow the example before using them.
  State the scientific assumptions that affect its meaning, without turning a focused example into a textbook chapter.
* **Terminology:** define metrics and any conditioning, such as on accepted shots, where they are introduced, then use concise names consistently.
  Retain qualifiers only where they distinguish quantities.
* **Interpretation:** explain how outputs, assertions, and figures answer the question.
  Distinguish demonstrations of an interface from scientific checks, and exact results from estimates or bounds.
  State what a result does not establish when that affects its interpretation.
* **Figures:** explain shared plot conventions, such as reference lines and uncertainty bands, before the first figure that uses them.

Review the prose, code, and saved outputs together from the intended reader's perspective; successful execution alone does not establish that an example is clear or meaningful.
Keep explanations focused on the ideas and decisions, rather than narrating each API call.
Related cells may share an introduction or interpretation, and a few focused sentences can be enough.
