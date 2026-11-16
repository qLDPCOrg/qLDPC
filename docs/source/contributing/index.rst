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
The first two, the drafting checks, run while a change is in progress, including during a :doc:`review <reviewing>` and in a draft pull request.

* **Anytime:** ``format_.py --check``, ``lint_.py``, and ``mypy_.py`` over the whole repository.
  To apply formatting to changed files, run ``python checks/format_.py --fix -i``.
  ``format_.py`` always checks ``pyproject.toml`` as well, and without ``--check`` it rewrites that file.
* **As needed:** targeted tests of the change, selected by function or by the test file of a changed module.
  Run ``pytest_.py`` only on selected paths, since over the whole repository it repeats the test run of ``coverage_.py``.
  When targeted tests do not settle whether a module's own tests cover it fully, pass, and run fast as plain Python, run ``coverage_.py`` on the module and its test file.
* **Extended:** the anytime checks, ``coverage_.py`` over the whole repository, the compiled runs, and ``build_docs.py``: the checks that CI runs, other than the notebook runs.

CI runs on pull requests that are ready for review, not on drafts.
It also runs the checks that need extra dependencies or platforms, such as the notebook and GAP integration tests, so these need not run locally.
Make the compiled runs locally only in the extended checks, or to test a change to compiled code.
To re-balance the coverage shards, run the "Store test durations" workflow from the Actions tab, then ``python checks/_fetch_test_durations.py``, and commit the updated ``.test_durations``.

Every check script except ``build_docs.py`` accepts explicit files, or ``-i/--incremental [REVISION ...]``, which checks only the files changed since the merge base with ``upstream/main``, ``origin/main``, or ``main`` (the first that exists), or with the given revisions.
Check scripts find files through Git, so stage new files with ``git add`` before checking them.

Refreshing the saved outputs of a changed notebook, as the :doc:`writing standard <standards/writing>` requires, is part of writing the change, not a check; the docs build renders those outputs without running code.
If a notebook runner starts its kernel from ``examples/``, put the absolute path of this checkout's ``src`` directory in ``PYTHONPATH``.
The TQEC notebook (``examples/logical_error_rates/specialized/decoding_tqec_circuits.ipynb``) requires Python 3.11--3.13, since its pinned TQEC dependency does not support Python 3.14.

.. toctree::
   :hidden:

   reviewing
   decisions
   standards/index
