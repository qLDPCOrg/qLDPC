Contributing
============

Let the `project goals <https://github.com/qLDPCOrg/qLDPC#-goals>`_ guide each change's scope, design, validation, and explanation.
The README states how `qLDPC uses AI assistants <https://github.com/qLDPCOrg/qLDPC#ai-usage>`_.
AI agents follow `AGENTS.md <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md>`_, which states what they may do without a maintainer's request.

Contributor documentation comes in two kinds:

* :doc:`Standards <standards/index>` state the rules that a change must satisfy, one topic per standard.
* Guides, such as this page, explain how qLDPC works and how to make a change, and link to the standards rather than restating them.

Read the standards that apply to a change before editing, and check the change against them in a :doc:`standard review <reviewing>`.

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
   * - Pull request description
     - :doc:`Pull request standard <standards/pull_requests>`

Orientation
-----------

The source tree under ``src/qldpc/`` mirrors the library: each package and module starts with a docstring whose first line summarizes it, and the :doc:`library map </library_map>` explains how the packages relate.
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
     - Per-module tests and coverage, as in CI: each module under ``src/`` needs its own test file, which must pass and alone cover 100% of the module's statements.
       It skips the tests marked ``compiled``.
   * - ``python checks/build_docs.py``
     - Strict Sphinx and notebook documentation build
   * - ``python checks/all_.py``
     - Formatting, lint, mypy, tests with overall 100% coverage, and (on full runs) the docs build

Checks come in three tiers by cost.
The first two tiers are the drafting checks, which run while a change is in progress, including during a :doc:`standard review <reviewing>` and in a draft pull request.
The third tier is the finishing checks.

* **Anytime:** ``format_.py --check``, ``lint_.py``, and ``mypy_.py`` over the whole repository.
  To apply formatting, run ``format_.py --fix`` on changed files, as in ``python checks/format_.py --fix -i``.
  Whatever files it is given, ``format_.py`` also checks ``pyproject.toml``, and without ``--check`` it rewrites that file.
* **As needed:** when writing, reviewing, or verifying a change, run targeted tests that cover it.
  Select tests by function, or run the test file of a changed module, as in ``python -m pytest src/qldpc/codes/common_test.py::test_name``.
  Run ``pytest_.py`` only on selected paths, since over the whole repository it largely repeats the test run of ``coverage_.py``.
  When targeted tests do not settle a question, such as whether a module's own tests fully cover it, run ``coverage_.py`` on that module and its test file.
  For example, ``python checks/coverage_.py src/qldpc/codes/common.py src/qldpc/codes/common_test.py`` checks the coverage of one module and its test file.
* **Finishing:** local checks that catch, before CI runs, the failures that a change most likely causes.
  CI runs them too, over the whole repository, so they are needed only when no one will read CI's results soon, such as when a change is finished while the maintainers are away.
  In that case, check the finished change and the code that it could affect before its pull request is marked ready for review.
  Run the anytime checks, ``coverage_.py -i``, the tests of closely related code, such as callers and subclasses of changed code, and ``build_docs.py`` if the change edits documentation, notebooks, or docstrings.
  Leave repository-wide test runs to CI.
  After fixing a failure, rerun the failed checks.
  If a standard review ran, ask the maintainer whether to review a significant fix in another round.
  If the maintainer is away, report that question as an open decision.

Refreshing the saved outputs of a changed notebook, as the :doc:`writing standard <standards/writing>` requires, is part of writing the change, not a check.

CI runs on pull requests that are ready for review, not on drafts.
It runs the checks that need extra dependencies or platforms, such as the notebook and GAP integration tests, so these need not run locally.
It also runs the tests marked ``compiled``, which ``coverage_.py`` skips, so these need not run locally either.

``all_.py`` runs most checks in one command.
It skips the per-module coverage gate and ``pyproject.toml`` formatting.
Running it alongside ``coverage_.py`` runs most tests twice.
Every check script except ``build_docs.py`` accepts explicit files, as in ``python checks/coverage_.py src/qldpc/codes/common.py``, or ``-i/--incremental [REVISION ...]``.
The incremental mode checks only the files changed since the merge base with ``upstream/main``, ``origin/main``, or ``main`` (the first that exists), or with the given revisions.
For example, ``python checks/coverage_.py -i`` runs the per-module coverage gate for changed files only, which is much faster than a full run.
Check scripts discover files through Git, so stage new source and test files with ``git add`` before running any check on them, including incremental runs.
For incremental runs, prefer the individual scripts: ``all_.py -i`` measures overall coverage over only the selected tests, so shared fixtures in ``conftest.py`` files can fail it spuriously.

The docs build renders saved notebook outputs without executing code cells.
Pass an absolute path to this checkout's ``src`` directory in ``PYTHONPATH`` if a notebook runner starts its kernel from ``examples/``.
The TQEC notebook (``examples/logical_error_rates/specialized/decoding_tqec_circuits.ipynb``) requires Python 3.11--3.13 because its pinned TQEC dependency does not support Python 3.14.

.. toctree::
   :hidden:

   reviewing
   standards/index
