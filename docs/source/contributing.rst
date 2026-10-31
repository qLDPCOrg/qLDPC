Contributor reference
=====================

Start with the `agent and contributor guide <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md>`_ for project guidance, essential rules, and task-specific reading pointers.
This reference contains the detailed contribution rules and procedures.
Read the sections relevant to your change before editing.

Orientation
-----------

The source tree under ``src/qldpc/`` mirrors the library: each package and module starts with a docstring whose first line summarizes it, and the :doc:`library map <library_map>` explains how the packages relate.
Canonical executable notebooks live in ``examples/``, Sphinx pages and the library map in ``docs/source/``, local wrappers for CI's quality gates in ``checks/``, and untested research scripts in ``experiments/``.

The methods in ``codes/common.py`` share cached and mutable state for standard form, logical and gauge operators, parameters, and code transformations.
Do not split them into mixins merely to reduce the file length.
Plumbing that one subpackage uses is an underscore name in that subpackage, and plumbing that several subpackages share lives in private modules (see `Public API and private code`_).

Public API and private code
---------------------------

* A name is public if it is broadly useful to researchers and engineers, and private if it is plumbing that makes qLDPC work.
  Where a name is used, such as in other modules or in notebooks, does not decide this.
* Plain names defined in public modules (no underscore anywhere in the import path) are public API.
  Keep them working, and retire a released public name through a tested shim rather than breaking its import (see `Deprecations and future changes`_).
* ``src/qldpc/__init__.py`` exports subpackages, and each subpackage's ``__all__`` lists the public names that it re-exports.
  Re-export only what typical users reach for, leave specialized public tools at their module path, give each public object one canonical public path, and put utilities that several packages share in a neutral top-level module such as ``qldpc.math``.
  Preserve public import paths when moving implementation code.
* Module-level underscore names, and a class's underscore attributes and methods, may be used anywhere in their top-level subpackage, such as ``qldpc.codes`` or ``qldpc.decoders``, but not outside it.
  Each top-level module, such as ``qldpc.math``, is its own scope.
* Private modules, with an underscore anywhere in their import path, exist only for plumbing shared across subpackages or for a substantial standalone topic.
  Name them by topic, never ``_common``, ``_util``, ``_helpers``, or ``_misc``, and use plain names for what other modules use.
* ``qldpc/_utils.py`` holds Python-level plumbing only: lazy imports, warning stack levels, and docstring formatting.
* Public objects are defined in public modules; never re-export a private module's object as public API.
* Everything under ``src/qldpc/experimental/`` is unstable, can change without a deprecation period, and is exempt from these private-name rules.

Code style and organization
---------------------------

* Every source file starts with ``# SPDX-License-Identifier: Apache-2.0``, a blank line, and a module docstring whose first line summarizes the module.
* Keep Python source lines at most 100 columns long.
* Preserve a module's existing section headings and organization when adding code, and add a section when new functionality does not fit an existing one.
* Start each section with a line of 100 ``#`` characters followed by a one-line ``# title`` comment.
* Put important user-facing APIs first within modules and sections, and keep related public entry points together.
* Within each section, put plain names before underscore names, except where code that runs at import time (a base class, decorator, default argument, or module-level call) needs an underscore name first.
  Put a helper in the section of the code that mainly uses it, or in a final helper section if several sections share it, and do not move existing helpers merely because their usage changes.
  Private constants and types may sit near the imports.
* Within a class, put public methods first (for decoders, the decode methods) and private helper methods after them.
* ``src/qldpc/code_style_test.py`` checks license headers, section headers, and the order of plain and underscore names.
* The top-level modules and subpackages of ``qldpc.circuits`` and ``qldpc.decoders`` import each other in layers: each imports only from itself and from lower layers, and never from the root of its package.
  ``_LAYERS`` in the package's ``layering_test.py`` lists the layers and is the only place to update when adding a top-level module or subpackage; changes inside a subpackage need no edits.
  Imports in functions and under ``TYPE_CHECKING`` count; tests and ``conftest.py`` files may import from any layer.
  To keep an import that breaks the layers, end its line (for an import that spans several lines, its first line) with ``# layering-exception: <reason>``; the check fails if such a comment is no longer needed.
  To layer another package, add a ``layering_test.py`` to its directory that calls ``check_import_layering`` from ``src/qldpc/conftest.py``.

Prose style
~~~~~~~~~~~

* Choose material for a document's audience and purpose.
  Keep docstrings focused on documented objects and contributor guidance focused on contribution decisions; omit tangents and prose that merely restates implementation details.
* In substantial scientific exposition, introduce a problem and explain why it matters before describing implementation details.
  Worked examples also need enough context to explain the question they answer and what their results establish.
  Include background only where it helps intended readers.
* Introduce or explicitly identify an object, quantity, or concept before using a definite reference or pronoun.
  A clear name or defining description is sufficient; identical wording need not appear earlier.

  * **Unclear without prior context:** ``Run a simulation and check the estimate.``
  * **Clear:** ``Run a simulation to estimate a failure rate, then check that estimate.``

* Docstrings follow the Google convention.
* Python comments and docstrings use ordinary paragraphs: separate sentences with two spaces and fill lines to the 100-column limit.
* Keep one prose sentence per physical source line in repository ``.rst`` and ``.md`` files and notebook Markdown cells.
  Put ``First sentence.`` and ``Second sentence.`` on separate source lines; leave code blocks, equations, and table syntax intact.
  For PR-summary text, use the paragraph formatting in `Pull request summaries`_ instead.

.. _core-invariants:

Implementation requirements
---------------------------

Use the :doc:`library map <library_map>` and source docstrings for the data model and scientific conventions.
The requirements here concern changes to their implementation.

Algebra and code state
~~~~~~~~~~~~~~~~~~~~~~

* Preserve concrete ``galois.FieldArray`` types when copying or transforming code data.
  Do not coerce extension-field values with ``% field.order``.
* Reject incompatible group rings in NumPy operations, including arrays supplied through keyword arguments such as ``out=``.
* Use ``qldpc.math.symplectic_conjugate`` rather than duplicating sign and half-order conventions.
* Do not route nonbinary codes through the binary-only ``codes/distance/`` routines.
* Preserve stabilizer-based syndrome construction in subsystem decoding, and include a focused subsystem regression test when changing that path.

Decoder integration
~~~~~~~~~~~~~~~~~~~

Follow the contracts and maintainer requirements in the :doc:`adding a decoder guide <adding_decoders>`.
Reuse the existing construction and resolution helpers rather than duplicating capability dispatch.

Circuits
~~~~~~~~

* Use the ``_restrict_to_qubits`` guard from ``circuits/common.py`` in circuit and tableau constructors.
* Keep data, check, reference, and ancilla roles explicit through ``QubitIDs`` and bookkeeping types.
  Do not infer roles from coincident integer indices.
* Preserve memory-circuit detector-coordinate semantics; a monotone check index must not be interpreted as time.
* Noise-model operation immunity and qubit immunity are separate controls; preserve both.

External systems and caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Document subprocess, standard-input, clipboard, network, and package-installation side effects in the public entry points that can trigger them, and in error messages.
* Expensive reusable computations should use ``qldpc.cache.use_disk_cache(cache_name)``.
  Do not hide errors by returning a default value that looks successful.
* Numba code uses ``always_compiled`` or ``compiled_on_request`` from ``qldpc/_compilation.py``, never ``njit(cache=True)``; that module's docstring says which to use, and ``_compilation_test.py`` enforces its rules.

Tests
-----

* Every module needs its own co-located ``*_test.py``: the coverage gate requires each module to be fully covered by its own test file, and does not gate a module without one.
  Reach a hard-to-reach branch with a focused test rather than a broad scenario.
* Tests run with network sockets disabled.
  Mock only external boundaries such as web, subprocess, clipboard, prompt, and optional-package paths, or a module's own thin wrappers around them, never the logic under test, and do not add a live-service test.
* Disk caches are bypassed while pytest is imported, so tests must not depend on cache persistence.
* Pytest turns every ``DeprecationWarning`` and ``FutureWarning`` into an error, so tests use the modern API, and test deprecated usage and future-change warnings with ``pytest.warns`` (see `Deprecations and future changes`_).
  This applies to warnings from other packages too; avoid the call that triggers one, or add a narrow ``ignore`` entry to ``filterwarnings`` in ``pyproject.toml``.
* A test should guard behavior that matters: results, invariants, documented errors, and contracts between modules.
  Do not pin implementation choices such as internal call order, cache layouts, or incidental error-message wording.
  A test may call a private helper of the module that it tests to check the helper's contract or reach coverage, and may set private state of objects to construct a case, but should not assert on private state.
* Keep each test minimal: use the smallest code, field, or circuit that exercises the behavior, do not parametrize over cases that take the same path, and delete tests implied by others.
* Prefer independent oracles (rank, commutation, row-space identities, or brute force on small instances) over values that the code under test computed or cached.
  Checking a stored value against an independent expected value, such as one from the literature, is not tautological; deriving the expected value from the code under test is.
* Keep tests that reproduce results from the literature, and cite the source in the test.
* Tests run in random order with a per-run random seed (pytest-randomly), so they must not depend on order.
  Seed random inputs from ``pytestconfig.getoption("randomly_seed")`` for reproducibility, and make sure that the assertions hold for every seed; otherwise use a fixed seed.
* Check that a new or rewritten test fails when the behavior it guards is broken.

Common change recipes
---------------------

Add or change a code construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Put a classical family in ``codes/classical.py`` and a quantum or CSS family in ``codes/quantum.py``, next to its closest base or sibling.
2. Re-export public names from ``codes/__init__.py``.
3. Add a linked literature reference and document the field, subsystem, distance, and validation assumptions in the class docstring.
4. Add co-located tests with independent invariants: rank-derived dimension, CSS orthogonality, symplectic commutation, row-space identities, known small distances, or exact cross-family equivalence.
   Do not validate construction-supplied or cached parameters by reading those same values back.

Change code-core algebra
~~~~~~~~~~~~~~~~~~~~~~~~

1. Read the cached-state interactions in ``codes/common.py`` before changing standard form, logicals, stabilizers, gauges, dimension, or distance.
2. Test both CSS and non-CSS paths, stabilizer and subsystem paths, ``k=0``, and at least one odd characteristic when signs matter.
3. Preserve public imports and invalidate or transfer cached values deliberately.

Add or adapt a decoder
~~~~~~~~~~~~~~~~~~~~~~

Follow the :doc:`adding a decoder guide <adding_decoders>`.
Use its backend-contribution checklist for a maintained integration, and its shared-plumbing requirements when changing decoder construction or resolution.

Change a circuit workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Keep matrix, qubit-role, measurement-record, detector-record, and Stim-circuit bookkeeping in sync.
2. Test noiseless detector determinism and observable behavior before adding noise.
3. Exercise repeated rounds and seam behavior; a locally valid circuit fragment can still become wrong when initialization, QEC cycles, and readout are joined.
4. State computational and fault-tolerance limitations in the relevant function or class docstring.

Change an external integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Separate parsing/encoding logic from process, network, clipboard, and prompt mechanics.
2. Test both the callable-tool path and the absent/manual/error paths without live resources.
3. For finite fields, round-trip values through the external system's actual element encoding; do not assume integer storage conventions match.
4. Report nonzero return codes, standard error, malformed output, and missing data clearly.

Add documentation or an example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Put explanations that span packages in the :doc:`library map <library_map>`, exact API behavior in docstrings, and executable workflows in ``examples/``.
2. The notebooks under ``docs/source/examples/`` are links to the canonical files under ``examples/``.
   Edit the canonical notebook, not the documentation-tree link.
3. Keep README and the library map selective; AutoAPI lists the public symbols.
4. When a public limitation changes, update the relevant function or class docstring and every README or guide that repeats it in the same pull request.
5. Run the strict documentation build before considering the change complete.

Use the smallest meaningful example and explain the scientific choices and limitations, with pointers to the relevant literature and implementation.
Make examples easy to adapt to a new question rather than requiring users to adopt a complete workflow.

Example notebooks
^^^^^^^^^^^^^^^^^

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

Deprecations and future changes
-------------------------------

Never break callers of a released public API without warning.
Removing or renaming a name, argument, or mode is a deprecation: old usage keeps working with a ``DeprecationWarning``, and fails once the deprecation is removed.
Making the same call silently behave differently, as when a default changes, is a future change, which warns with a ``FutureWarning``, also while its old behavior is phased out.

* Add a shim only for a name that was public in a released version, at its released path: listed in a package ``__all__``, or used in released examples or docs.
  Names added since the last release may change without one.
* Deprecation code lives in the private package ``qldpc/_deprecations/``, whose module docstrings explain its mechanics: register renamed and moved names in ``ALIASES``, deprecated functions that warn when called in ``SILENT_ALIASES``, and deprecated methods in ``install_deprecated_methods``.
  Code in that package may use other subpackages' underscore names, to reproduce old behavior.
* Elsewhere, handle deprecated arguments, modes, and calling conventions, and future changes, in small blocks that call ``_deprecations`` helpers such as ``warn_deprecated``, ``warn_future_change``, and ``defunct``, and move any substantial logic into ``qldpc/_deprecations/``.
  Mark each such block, and the parameter that it handles, with ``# DEPRECATED: see qldpc._deprecations`` or ``# FUTURE CHANGE: see qldpc._deprecations``.
* The library and its docs, including the README, are written as if deprecated APIs do not exist, and tests use deprecated APIs only to test them.
  No ``__all__`` lists a deprecated name, and no ``TYPE_CHECKING`` alias exposes one, so that star imports do not warn and type checkers reject deprecated names.
  A docstring mentions a deprecated option or old value in at most one ``Args`` entry, which names its replacement.
* Warning messages and docs say "in a future release" rather than naming a version.
* A future change takes three steps, at least one minor release apart:

  1. Give the keyword that selects the behavior (a new keyword, or the one whose default changes) a default of ``None`` that keeps the old behavior and warns; explicit values select the old or the new behavior without a warning.
     Describe both behaviors in the docstring, and the change in a "Future changes" section of the :doc:`library map <library_map>`, which is added when needed and removed when empty.
  2. Make ``None`` select the new behavior without a warning, keep the old value working with a ``FutureWarning`` that it will be removed, update the docstring, and remove the library-map entry.
  3. Make the old value raise ``_deprecations.defunct(...)``, mark the block ``DEPRECATED``, and rename its tests ``test_deprecated_*``.
     If the new value is the only one left, also deprecate the keyword.

  From step 1 until step 3, qLDPC code, tests, and notebooks pass the new value explicitly.
* Name new tests of inline deprecations ``test_deprecated_*``, and tests of future changes ``test_future_change_*``.
  ``_deprecations/core_test.py`` tests every registry entry automatically; add a case there for each new deprecated method.
* To remove deprecations, grep for ``_deprecations`` and, case-insensitively, for ``deprecat``, and delete the deprecated code, ``Args`` notes, and tests, but leave ``FUTURE CHANGE`` blocks alone.

.. _development-installation:

Development installation
------------------------

For work on qLDPC itself, clone the repository and install it in editable mode with its development dependencies:

.. code-block:: bash

   git clone https://github.com/qLDPCOrg/qLDPC.git
   cd qLDPC
   python -m pip install -e '.[dev]'

Validation commands
-------------------

CI's exact environments and commands are in ``.github/workflows/``.
Use the `validation commands in the agent and contributor guide <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md#validation-commands>`_.

Run checks on what you changed while iterating, and save repository-wide runs for a finished change.
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
Do not weaken a failure with ``noqa``, ``type: ignore``, or coverage exclusions unless the exceptional condition is real and documented.

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
