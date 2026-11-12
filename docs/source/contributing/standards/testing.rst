Testing standard
================

This standard covers what tests should guard and the conditions under which they run.

* Every module under ``src/``, other than an ``__init__.py`` or ``conftest.py`` file, needs its own co-located ``*_test.py``.
  The coverage gate fails a module without one.
  It requires each module to be fully covered by its own test file, and each test file to be fully covered by itself.
  The gate runs the selected test files together on parallel pytest workers, and credits a module only with the lines that its own test file runs in the test process, not in subprocesses (see the module ``_coverage_contexts`` of checks-superstaq).
  Between test files on one worker, it resets the repository's shared fixtures and qLDPC's ``functools`` caches, but not other module-level state.
  Reach a hard-to-reach branch with a focused test rather than a broad scenario.
* The coverage gate runs qLDPC's compiled functions as plain Python, so that coverage can trace them.
  So every test must be fast as plain Python.
  CI also runs some test files with these functions compiled, in the compiled runs of the :ref:`validation commands <validation-commands>`.
  A test of behavior that only compiled code has must also pass as plain Python, and must be in a test file that a compiled run selects, unless it compiles its own functions even in the coverage gate, as ``src/qldpc/_compilation_test.py`` does.
* Tests run with network sockets disabled.
  Mock only external boundaries such as web, subprocess, clipboard, prompt, and optional-package paths, or a module's own thin wrappers around them, never the logic under test, and do not add a live-service test.
* Disk caches are bypassed while pytest is imported, so tests must not depend on cache persistence.
* Pytest turns every ``DeprecationWarning`` and ``FutureWarning`` into an error, so tests use the modern API, and test deprecated usage and future-change warnings with ``pytest.warns`` (see :doc:`deprecations`).
  This applies to warnings from other packages too; avoid the call that triggers one, or add a narrow ``ignore`` entry to ``filterwarnings`` in ``pyproject.toml``.
* A test should guard behavior that matters: results, invariants, documented errors, and contracts between modules.
  Do not pin implementation choices such as internal call order, cache layouts, or incidental error-message wording.
  A test may call a private helper of the module that it tests to check the helper's contract or reach coverage, and may set private state of objects to construct a case, but should not assert on private state.
* Keep tests minimal, both individually and in total.
  Use the smallest code, field, or circuit that exercises the behavior.
  Keep a test, parameter value, or looped input to the code under test only if it could catch a bug that no other case would.
  A change that reuses tested code needs at most a small test of the new integration, and none if existing tests already cover it.
* Prefer independent oracles (rank, commutation, row-space identities, or brute force on small instances) over values that the code under test computed or cached.
  Checking a stored value against an independent expected value, such as one from the literature, is not tautological; deriving the expected value from the code under test is.
* Keep tests that reproduce results from the literature, and cite the source in the test.
* Tests run in random order with a per-run random seed (pytest-randomly), so they must not depend on order.
  Seed random inputs from ``pytestconfig.getoption("randomly_seed")`` for reproducibility, and make sure that the assertions hold for every seed; otherwise use a fixed seed.
* Check that a new or rewritten test fails when the behavior it guards is broken.
