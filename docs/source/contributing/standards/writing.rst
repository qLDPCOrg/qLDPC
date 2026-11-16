Writing standard
================

This standard covers prose in comments, docstrings, documentation, and example notebooks, and where each kind of documentation belongs.
The :ref:`code standard <simplicity-and-clarity>` covers names and the comments that outline long blocks of code.

Prose style
-----------

* Write and review prose by purpose.
  For each paragraph, docstring section, or comment, ask what its reader needs from it, and whether that reader would miss it if it were gone.
  If not, delete it, or point to where the information lives.
  If so, state its message in one line, then write the simplest text that gets that message across, without tangents or restated implementation details.
* Fix unclear text by cutting or restructuring it before adding words.
* A docstring says what its object does, and what its arguments and results mean where the signature and summary line leave that unclear.
  Beyond that, state a behavior when, and only when, a reader who knows the subject's standard terms could reasonably expect something else, and so get a wrong or materially misleading result without an error.
  An expectation is reasonable if it has a source, such as the literature, a similar function, or a name.
  Also state limitations and security risks, and what other standards call for, such as assumptions and side effects.
  For example, say that the distance of a subsystem code is dressed, since the literature also defines a bare distance, but not that an empty batch of syndromes decodes to an empty batch of errors.
* Prefer short sentences with one idea each.
  Split a sentence with stacked clauses, parenthetical asides, or chains of semicolons rather than compressing it.
* Describe what code does, not how it used to work.
  Give readers a sense of what happens, and point to where details live rather than repeating or counting them.
  Check that comments and docstrings match what the code does, and update them when the code changes.
* In substantial scientific exposition, introduce a problem and explain why it matters before describing implementation details, and include background only where it helps intended readers.
* Introduce or explicitly identify an object, quantity, or concept before using a definite reference or pronoun.
  A clear name or defining description is sufficient; identical wording need not appear earlier.

  * **Unclear without prior context:** ``Run a simulation and check the estimate.``
  * **Clear:** ``Run a simulation to estimate a failure rate, then check that estimate.``

* When a text mentions an object, quantity, or concept more than once by a long description, give it a short name or a symbol, such as one that the code or the literature uses.
  Then use that name, rather than repeating a long description or chaining noun phrases.

  * **Repeated noun phrases:** ``If the rank of the checks equals the rank of their X parts plus the rank of their Z parts, ...``
  * **Named once:** ``Let H_X and H_Z be the X and Z parts of the checks H.  If rank(H) = rank(H_X) + rank(H_Z), ...``

* Use the serial (Oxford) comma before the conjunction in a list of three or more items, as in ``X, Y, and Z``.
* Docstrings use the section format of the Google convention, such as ``Args:`` and ``Returns:``.
* Python comments and docstrings use ordinary paragraphs: separate sentences with two spaces and fill lines to the 100-column limit.
* Keep one prose sentence per physical source line in repository ``.rst`` and ``.md`` files and notebook Markdown cells.
  Put ``First sentence.`` and ``Second sentence.`` on separate source lines; leave code blocks, equations, and table syntax intact.
  For a pull request description, use the paragraph formatting in the :doc:`pull request standard <pull_requests>` instead.

.. _writing-documentation:

Where documentation belongs
---------------------------

* Put each rule that a change must satisfy in exactly one :doc:`standard <index>`; each standard covers one topic.
  In guides, which explain how qLDPC works and how to make a change, link to the relevant standard rather than restating its rules.
* ``AGENTS.md`` holds how agents behave, such as when they act on their own and which changes need approval, and may remind agents of a rule elsewhere in one sentence that links to it.
  Put anything that people need in a guide or a standard, never only in ``AGENTS.md``.
* Keep each page focused on one topic or task, and small enough to read whole; split a page that outgrows that.
* Put explanations that span packages in the :doc:`library map </library_map>`, exact API behavior in docstrings, and executable workflows in ``examples/``.
* In a package whose directory holds guides to it, such as ``codes/distance/``, only those guides give correctness arguments for its code, such as why a bound holds.
  Docstrings and comments in the package point to a guide's section rather than repeat its argument, and may state in a clause what that section proves.
* Mention a measurement that bears on a choice in the code only when it is the reason for the choice: when the code would do otherwise without it, as when one way of building stim circuits is much faster than another.
  Then give a rounded factor or figure, such as "about 10 times faster", without machines or pull requests.
  Measurements that only check a choice made on other grounds, such as a heuristic, go in the pull request description when review needs them; a heuristic may still state the rough costs that it assumes.
* The notebooks under ``docs/source/examples/`` are links to the canonical files under ``examples/``.
  Edit the canonical notebook, not the documentation-tree link.
* After changing a notebook example, refresh its saved outputs by executing the entire canonical notebook in an offline, editable installation.
* Keep README and the library map selective; AutoAPI lists the public symbols.
* When a public limitation changes, update the relevant function or class docstring and every README or guide that repeats it in the same pull request.

.. _writing-notebooks:

Example notebooks
~~~~~~~~~~~~~~~~~

An example notebook helps readers understand and adapt a scientific workflow, not merely shows that API calls run.
Use the smallest meaningful example, make it easy to adapt to a new question, and point to the relevant literature and implementation.

* **Purpose:** introduce the question that the example answers, and why it is useful, before implementation details, for readers still exploring what is possible.
* **Framing:** say what the example enables, and describe its choices as choices for that example.
  Avoid straw-man contrasts and assumptions that the reader needs correcting, but keep contrasts that clarify assumptions, genuine alternatives, or limitations.
* **Background and assumptions:** introduce the concepts, quantities, notation, and scientific assumptions that the example needs, without turning it into a textbook chapter.
* **Terminology:** define metrics and any conditioning, such as on accepted shots, where they first appear, then use short names consistently.
* **Interpretation:** explain how outputs, assertions, and figures answer the question.
  Distinguish demonstrations of an interface from scientific checks, and exact results from estimates or bounds.
  Say what a result does not establish when that affects its meaning.
* **Figures:** explain shared plot conventions, such as reference lines and uncertainty bands, before the first figure that uses them.

Review the prose, code, and saved outputs together as the intended reader would; running without errors does not make an example clear or meaningful.
Explain ideas and decisions rather than narrating each API call; related cells may share a few focused sentences.
