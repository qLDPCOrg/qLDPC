# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "qLDPC"
copyright = "2023 The qLDPC Authors and Infleqtion Inc."  # pylint:disable=redefined-builtin
author = "Michael A. Perlin"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# see https://www.sphinx-doc.org/en/master/usage/extensions/index.html
extensions = [
    "autoapi.extension",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",
]

# use the pre-executed outputs in notebooks
nbsphinx_execute = "never"

# generate stub.rst files automatically
autosummary_generate = False

# fix fox mathjax v3
# https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.mathjax
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

templates_path = ["_templates"]
exclude_patterns = ["modules.rst"]

autoapi_dirs = [
    "../../src/qldpc",
]
autoapi_type = "python"

# Note: "imported-members" is intentionally omitted.  Including it re-documents
# every object on each parent package's page (e.g. classes defined in
# qldpc/codes/quantum.py would also appear on the qldpc.codes page), which both
# roughly doubles the number of pages/warnings and produces ambiguous
# cross-references.  Each object is instead documented once, on the page of the
# module that defines it.
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

autoapi_ignore = ["*_test.py", "*/checks/*.py", "*conftest.py"]

autoapi_member_order = "groupwise"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

