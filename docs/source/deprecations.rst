Deprecations and future changes
===============================

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
