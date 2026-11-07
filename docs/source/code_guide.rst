Code guide
==========

This guide covers which names are public, how source files are organized, and how to keep code simple and clear.
The :doc:`writing guide <writing_guide>` covers the prose of comments and docstrings.

Public API and private code
---------------------------

* A name is public if it is broadly useful to researchers and engineers, and private if it is plumbing that makes qLDPC work.
  Where a name is used, such as in other modules or in notebooks, does not decide this.
* Plain names defined in public modules (no underscore anywhere in the import path) are public API.
  Keep them working, and retire a released public name through a tested shim rather than breaking its import (see :doc:`deprecations`).
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
* The methods in ``codes/common.py`` share cached and mutable state for standard form, logical and gauge operators, parameters, and code transformations.
  Do not split them into mixins merely to reduce the file length.
* ``src/qldpc/code_style_test.py`` checks license headers, section headers, and the order of plain and underscore names.
* The top-level modules and subpackages of ``qldpc.circuits`` and ``qldpc.decoders`` import each other in layers: each imports only from itself and from lower layers, and never from the root of its package.
  ``_LAYERS`` in the package's ``layering_test.py`` lists the layers and is the only place to update when adding a top-level module or subpackage; changes inside a subpackage need no edits.
  Imports in functions and under ``TYPE_CHECKING`` count; tests and ``conftest.py`` files may import from any layer.
  To keep an import that breaks the layers, end its line (for an import that spans several lines, its first line) with ``# layering-exception: <reason>``; the check fails if such a comment is no longer needed.
  To layer another package, add a ``layering_test.py`` to its directory that calls ``check_import_layering`` from ``src/qldpc/conftest.py``.
* Do not weaken a failure with ``noqa``, ``type: ignore``, or coverage exclusions unless the exceptional condition is real and documented.

.. _simplicity-and-clarity:

Simplicity and clarity
----------------------

* Keep code only as complex as its uses require.
  Avoid storing data that can be derived from other stored data, accepting arguments that are unused or derivable from other arguments, and adding options, parameters, or code paths that nothing uses.
  Reuse existing implementations, such as ``qldpc._binary``, rather than duplicating them, and remove dead code and comments that describe removed designs.
  Deliberate exceptions are fine when a clear benefit, such as a performance gain at negligible cost, justifies them; explain the reason in a comment.
* Choose names that describe what they hold or do.
  Make a name as specific as the thing it names: a general name for a general tool, and a specific name for a special case.
* Split a long block of code into paragraphs separated by blank lines.
  Start each paragraph with a comment that says what it does.
  A reader should be able to follow the outline of the code from these comments alone.
