Pull request standard
=====================

This standard covers pull request descriptions.

Use a pull request description as a guide to review, not a second set of documentation or an execution log.
Write a description for a reader who has read no other pull request: link merged pull requests by number, and explain any context that another pull request provides rather than referring to it, as in "the next pull request".
Give numbers, such as test counts, timings, or speedups, only when review needs them, as for a change made for speed, and do not take measurements only for a description.
The description of a change to the rules for working on qLDPC says how it changes their total length, and which rule, if any, it replaces.
Normally use the following structure, in proportion to the pull request, and omit sections that add no information:

* **Motivation or Background:** why the change is needed, and the context needed to review it.
* **Changes:** what the whole pull request does, grouping related changes rather than listing commits, and pointing to important details rather than repeating changed documentation.
* **Testing or Validation, when useful:** how consequential decisions were made, tested, or validated, such as alternatives compared, independent controls, experiments, or manual exploration.
  Do not list routine CI checks or results; mention a check that also runs in CI only when its design or interpretation matters, and then say what it establishes.
* **Additional notes, when needed:** material tradeoffs, limitations, compatibility considerations, or review questions that fit nowhere above.

Write the prose of a description in paragraphs, with two spaces between sentences and no line breaks merely at sentence boundaries.
