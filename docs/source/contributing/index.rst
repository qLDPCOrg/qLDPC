Contributing
============

Contributing in brief
---------------------

1. Read the `project goals <https://github.com/qLDPCOrg/qLDPC#-goals>`_, and let them guide your change's scope, design, validation, and explanation.
2. Install qLDPC as the :ref:`development installation <development-installation>` describes.
3. Find your kind of change in the :ref:`kinds-of-change table <kinds-of-change>`, and read the standards that it lists before editing.
4. While you work, run the drafting checks of the :ref:`validation commands <validation-commands>`.
5. Open a pull request, with a description that follows the :doc:`pull request standard <standards/pull_requests>`.
6. Maintainers review each pull request against the :doc:`standards <standards/index>`, often with AI agents, as :doc:`reviewing changes <reviewing>` describes.
   You need not run that review yourself.

The README states how `qLDPC uses AI assistants <https://github.com/qLDPCOrg/qLDPC#ai-usage>`_.
AI agents follow `AGENTS.md <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md>`_, which states what they may do without a maintainer's request.

.. _kinds-of-change:

Kinds of change
---------------

Every change follows the :doc:`code <standards/code>`, :doc:`writing <standards/writing>`, and :doc:`testing <standards/testing>` standards.
Some kinds of change have further requirements:

.. list-table::
   :header-rows: 1

   * - Change
     - Further requirements
   * - Code construction
     - Implementation standard: :ref:`code constructions <implementation-code-constructions>` and :ref:`algebra and code state <implementation-algebra>`
   * - Code-core algebra or cached state
     - Implementation standard: :ref:`algebra and code state <implementation-algebra>`
   * - Decoder
     - :doc:`Decoder standard <standards/decoders>`
   * - Circuit or tableau tool
     - Implementation standard: :ref:`circuits <implementation-circuits>`
   * - External integration or caching
     - Implementation standard: :ref:`external systems and caching <implementation-external-systems>`
   * - Documentation or example
     - Writing standard: :ref:`where documentation belongs <writing-documentation>` and :ref:`example notebooks <writing-notebooks>`
   * - Public name, argument, mode, or default
     - :doc:`Deprecation standard <standards/deprecations>`

Orientation
-----------

The source tree under ``src/qldpc/`` mirrors the library: each package and module starts with a docstring whose first line summarizes it, and the :doc:`library map </library_map>` explains how the packages relate.
Canonical executable notebooks live in ``examples/``, Sphinx pages and the library map in ``docs/source/``, local wrappers for CI's quality gates in ``checks/``, and research code in ``experiments/``.

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

.. list-table::
   :header-rows: 1

   * - Command
     - Purpose
   * - ``python checks/pytest_.py [paths]``
     - The pytest suite, or the tests under the given paths; does not execute notebooks by default
   * - ``python -m pytest src/qldpc/codes/common_test.py::test_name -v``
     - One focused test
   * - ``python checks/pytest_.py --notebook examples/decoders.ipynb examples/basics.ipynb examples/noise_models.ipynb``
     - Execute the notebooks that CI executes, with their dependencies installed
   * - ``python checks/format_.py --check`` or ``--fix``
     - Check, or apply, Ruff and ``pyproject.toml`` formatting
   * - ``python checks/lint_.py``
     - Ruff lint
   * - ``python checks/mypy_.py``
     - mypy with the repository's typing settings, over source and tests
   * - ``python checks/coverage_.py``
     - Per-module tests and coverage, as in CI: the coverage gate that the :doc:`testing standard <standards/testing>` describes
   * - ``python -m pytest src/qldpc --compiled-module-tests``
     - Compiled run, as in CI: the test files of the modules that define compiled functions, with the always-compiled functions compiled
   * - ``python -m pytest src/qldpc/codes/distance --compile-search-setup``
     - Compiled run, as in CI: the distance tests, with the compiled-on-request functions compiled too
   * - ``python checks/build_docs.py``
     - Strict Sphinx and notebook documentation build
   * - ``python checks/all_.py``
     - Formatting, lint, mypy, ``coverage_.py``, and (on full runs) the docs build

Checks come in three tiers by cost.
The first two tiers are the drafting checks, which run while a change is in progress, including during a :doc:`review <reviewing>` and in a draft pull request.
The third tier is the extended checks.

* **Anytime:** ``format_.py --check``, ``lint_.py``, and ``mypy_.py`` over the whole repository.
  To apply formatting, run ``format_.py --fix`` on changed files, as in ``python checks/format_.py --fix -i``.
  Whatever files it is given, ``format_.py`` also checks ``pyproject.toml``, and without ``--check`` it rewrites that file.
* **As needed:** when writing, reviewing, or verifying a change, run targeted tests that cover it.
  Select tests by function, or run the test file of a changed module, as in ``python -m pytest src/qldpc/codes/common_test.py::test_name``.
  Run ``pytest_.py`` only on selected paths, since over the whole repository it largely repeats the test run of ``coverage_.py``.
  When targeted tests do not settle a question, such as whether a module's own tests fully cover it, or pass and run fast as plain Python, run ``coverage_.py`` on that module and its test file.
  For example, ``python checks/coverage_.py src/qldpc/codes/common.py src/qldpc/codes/common_test.py`` checks the coverage of one module and its test file.
* **Extended:** the checks that CI runs on every pull request, other than the notebook runs: the anytime checks, ``coverage_.py`` over the whole repository, the compiled runs, and ``build_docs.py``.

Refreshing the saved outputs of a changed notebook, as the :doc:`writing standard <standards/writing>` requires, is part of writing the change, not a check.

CI runs on pull requests that are ready for review, not on drafts.
It runs the checks that need extra dependencies or platforms, such as the notebook and GAP integration tests, so these need not run locally.
It also makes the compiled runs.
Run these locally only in the extended checks, or as targeted tests of a change to compiled code.
To re-balance the coverage shards, run the "Store test durations" workflow from the Actions tab, then ``python checks/_fetch_test_durations.py``, and commit the updated ``.test_durations``.

Every check script except ``build_docs.py`` accepts explicit files, as in ``python checks/coverage_.py src/qldpc/codes/common.py``, or ``-i/--incremental [REVISION ...]``.
The incremental mode checks only the files changed since the merge base with ``upstream/main``, ``origin/main``, or ``main`` (the first that exists), or with the given revisions.
For example, ``python checks/coverage_.py -i`` runs the per-module coverage gate for changed files only, which is much faster than a full run.
Check scripts discover files through Git, so stage new source and test files with ``git add`` before running any check on them, including incremental runs.

The docs build renders saved notebook outputs without executing code cells.
Pass an absolute path to this checkout's ``src`` directory in ``PYTHONPATH`` if a notebook runner starts its kernel from ``examples/``.
The TQEC notebook (``examples/logical_error_rates/specialized/decoding_tqec_circuits.ipynb``) requires Python 3.11--3.13 because its pinned TQEC dependency does not support Python 3.14.

.. toctree::
   :hidden:

   reviewing
   decisions
   standards/index
