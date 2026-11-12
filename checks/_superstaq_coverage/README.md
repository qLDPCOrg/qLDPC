# Copies of the coverage check of checks-superstaq

This package holds verbatim copies of two modules of checks-superstaq, which its latest release lacks:

* `coverage_.py`, from `checks-superstaq/checks_superstaq/coverage_.py`;
* `_coverage_contexts.py`, from `checks-superstaq/checks_superstaq/_coverage_contexts.py`.

They come from [Infleqtion/client-superstaq#1458](https://github.com/Infleqtion/client-superstaq/pull/1458), at commit `6f90c6c2`.
`coverage_.py` takes `check_utils` from the installed checks-superstaq.

* Never edit the copies.
  `__init__.py` checks their hashes on import, so the coverage check fails if they change.
  Report a problem upstream instead.
  Once it is fixed there, copy the files anew, and update their git blob hashes in `UPSTREAM_HASHES` of `__init__.py` and the commit in this file.
* `__init__.py` and `ruff.toml` belong to qLDPC.
  `__init__.py` puts `checks/` on `PYTHONPATH`, so that the pytest processes of the coverage check can import the plugin `_superstaq_coverage._coverage_contexts`, which `checks/coverage_.py` names.
  `ruff.toml` turns qLDPC's lint rules off for the copies, which follow upstream's style.
  The copies also pass qLDPC's formatter; if a new copy does not, extend `ruff.toml` rather than reformat the copy.

`checks/coverage_.py` says how to switch to a release of checks-superstaq that includes these modules.
