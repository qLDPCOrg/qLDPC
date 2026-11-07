Contributing
============

Start with the `agent and contributor guide <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md>`_ for project guidance, essential rules, and task-specific reading pointers.
Contributor documentation comes in two kinds:

* :doc:`Standards <standards/index>` state the rules that a change must satisfy, one topic per standard.
* Guides, such as this page, explain how to work on qLDPC and link to the standards rather than restating them.

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
     - :doc:`Decoder standard <standards/decoders>`, and the :doc:`guide to adding a decoder </adding_decoders>`
   * - Circuit or tableau tool
     - Implementation standard: :ref:`circuits <implementation-circuits>`
   * - External integration or caching
     - Implementation standard: :ref:`external systems and caching <implementation-external-systems>`
   * - Documentation or example
     - Writing standard: :ref:`documentation and examples <writing-documentation>` and :ref:`example notebooks <writing-notebooks>`
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
Use the `validation commands in the agent and contributor guide <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md#validation-commands>`_.

While iterating, run checks on what you changed; the tests of the changed modules usually suffice.
Run the coverage gates, the documentation build, and the full test suite once review is done, as in the last step of a :doc:`standard review <reviewing>`.
Every check script except ``build_docs.py`` accepts explicit files, as in ``python checks/coverage_.py src/qldpc/codes/common.py``, or ``-i/--incremental [REVISION ...]``.
The incremental mode checks only the files changed since the merge base with ``upstream/main``, ``origin/main``, or ``main`` (the first that exists), or with the given revisions.
For example, ``python checks/coverage_.py -i`` runs the per-module coverage gate for changed modules only, which is much faster than a full run.
``format_.py`` always checks, or with ``--fix`` rewrites, ``pyproject.toml`` as well, whatever files it is given.
Prefer the individual scripts for incremental runs: ``all_.py -i`` measures overall coverage over only the selected tests, so shared fixtures in ``conftest.py`` files can fail it spuriously.
Before merging, run ``python checks/all_.py`` and ``python checks/coverage_.py``.

The docs build renders saved notebook outputs without executing code cells, so refresh them as the :doc:`writing standard <standards/writing>` describes.
Pass an absolute path to this checkout's ``src`` directory in ``PYTHONPATH`` if a notebook runner starts its kernel from ``examples/``.
The TQEC notebook (``examples/logical_error_rates/specialized/decoding_tqec_circuits.ipynb``) requires Python 3.11--3.13 because its pinned TQEC dependency does not support Python 3.14.

Some check wrappers discover files through Git.
Add new source and test files to the index before relying on the full gate to include them.

.. toctree::
   :hidden:

   reviewing
   standards/index
