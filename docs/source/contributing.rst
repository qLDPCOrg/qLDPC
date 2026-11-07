Contributing
============

Start with the `agent and contributor guide <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md>`_ for project guidance, essential rules, and task-specific reading pointers.
The detailed rules are organized by topic in the following guides.
Read the guides relevant to a change before editing, and review the change against them.

* :doc:`Code guide <code_guide>`: public and private names, code style and organization, and simplicity and clarity.
* :doc:`Writing guide <writing_guide>`: prose style for comments, docstrings, and documentation, and criteria for documentation and example notebooks.
* :doc:`Testing guide <testing_guide>`: what tests guard and how they run.
* :doc:`Implementation guide <implementation_guide>`: requirements for algebra, decoders, circuits, and external systems, and recipes for common changes.
* :doc:`Deprecations and future changes <deprecations>`: how to change a released public API.
* :doc:`Adding a decoder <adding_decoders>`: decoder contracts and backend contributions.
* :doc:`Reviewing changes <reviewing>`: the standard review, which checks a change against these guides.

This page covers the repository layout, development setup, validation, and pull request summaries.

Orientation
-----------

The source tree under ``src/qldpc/`` mirrors the library: each package and module starts with a docstring whose first line summarizes it, and the :doc:`library map <library_map>` explains how the packages relate.
Canonical executable notebooks live in ``examples/``, Sphinx pages and the library map in ``docs/source/``, local wrappers for CI's quality gates in ``checks/``, and untested research scripts in ``experiments/``.

.. _development-installation:

Development installation
------------------------

For work on qLDPC itself, clone the repository and install it in editable mode with its development dependencies:

.. code-block:: bash

   git clone https://github.com/qLDPCOrg/qLDPC.git
   cd qLDPC
   python -m pip install -e '.[dev]'

.. _validation-commands:

Validation commands
-------------------

CI's exact environments and commands are in ``.github/workflows/``.
Use the `validation commands in the agent and contributor guide <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md#validation-commands>`_.

Run checks on what you changed while iterating, and save repository-wide runs for a finished change.
While iterating, the tests of the changed modules usually suffice.
Run the coverage gates, the documentation build, and the full test suite once review is done, as in the last step of a :doc:`standard review <reviewing>`.
Every check script except ``build_docs.py`` accepts explicit files, as in ``python checks/coverage_.py src/qldpc/codes/common.py``, or ``-i/--incremental [REVISION ...]``.
The incremental mode checks only the files changed since the merge base with ``upstream/main``, ``origin/main``, or ``main`` (the first that exists), or with the given revisions.
For example, ``python checks/coverage_.py -i`` runs the per-module coverage gate for changed modules only, which is much faster than a full run.
``format_.py`` always checks, or with ``--fix`` rewrites, ``pyproject.toml`` as well, whatever files it is given.
Prefer the individual scripts for incremental runs: ``all_.py -i`` measures overall coverage over only the selected tests, so shared fixtures in ``conftest.py`` files can fail it spuriously.
Before merging, run ``python checks/all_.py`` and ``python checks/coverage_.py``.

The docs build renders saved notebook outputs without executing code cells.
Refresh outputs by executing the entire canonical notebook in an offline, editable installation before changing a notebook example.
Pass an absolute path to this checkout's ``src`` directory in ``PYTHONPATH`` if a notebook runner starts its kernel from ``examples/``.
The TQEC notebook (``examples/logical_error_rates/specialized/decoding_tqec_circuits.ipynb``) requires Python 3.11--3.13 because its pinned TQEC dependency does not support Python 3.14.

Some check wrappers discover files through Git.
Add new source and test files to the index before relying on the full gate to include them.

.. _pull-request-summaries:

Pull request summaries
----------------------

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
