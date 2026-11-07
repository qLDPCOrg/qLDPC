Pull request standard
=====================

This standard covers pull request summaries.

Use a PR summary as a guide to review, not a second set of documentation or an execution log.
Normally use the following structure:

* **Motivation or Background:** explain why the change is needed and provide context necessary to review it.
* **Changes:** outline what the full PR does, grouping related changes rather than listing individual commits.
  Point readers toward important details instead of reproducing explanations from changed documentation.
* **Testing or Validation, when useful:** explain how consequential decisions were made, tested, or validated, such as alternatives compared, independent controls, experiments, or manual exploration.
  Do not list routine CI checks or results.
  A check that also runs in CI can deserve mention when its design or interpretation matters; explain what it establishes and why it was chosen, not merely that it passed.
* **Additional notes, when needed:** include material tradeoffs, limitations, compatibility considerations, or review questions that do not fit naturally above.

Write PR-summary prose in paragraphs, with two spaces between sentences within each paragraph.
Do not insert line breaks merely at sentence boundaries; the one-sentence-per-source-line convention is for repository documentation, not PR summaries.

Use bulleted lists for independent points and numbered lists when order matters.
Keep the structure proportional to the PR, and omit sections that add no information.
