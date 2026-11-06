# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
from __future__ import annotations

import importlib
import os
import sys
import typing

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "qLDPC"
# Sphinx renders this as "(c) Copyright {copyright}."
copyright = (  # pylint:disable=redefined-builtin
    "2023-May 2024 The qLDPC Authors and Infleqtion Inc.; June 2024-present The qLDPC Authors"
)
author = "Michael A. Perlin"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# see https://www.sphinx-doc.org/en/master/usage/extensions/index.html
extensions = [
    "autoapi.extension",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",
]

# Render "Attributes:" sections as :ivar: fields.  Otherwise napoleon emits an attribute directive
# for each entry, which duplicates the attribute that AutoAPI documents for the dataclass field.
napoleon_use_ivar = True

# use the pre-executed outputs in notebooks
nbsphinx_execute = "never"

# Pin MathJax to v3.  nbsphinx requests MathJax on every notebook page, so leaving the major version
# to Sphinx's default would silently change how the published notebooks typeset their inline TeX.
# https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.mathjax
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

autoapi_dirs = [
    "../../src/qldpc",
]
autoapi_type = "python"

# Note: "imported-members" is intentionally omitted.  Including it re-documents every object on each
# parent package's page (e.g. classes defined in qldpc/codes/quantum/surface.py would also appear on
# the qldpc.codes and qldpc.codes.quantum pages), which both roughly doubles the number of
# pages/warnings and produces ambiguous cross-references.  Each object is instead documented once,
# on the page of the module that defines it.
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

autoapi_ignore = ["*_test.py", "*conftest.py"]

# Keep AutoAPI's generated .rst files on disk through the whole build.  With the default (False),
# AutoAPI removes them before the HTML writer copies page sources (html_copy_source), which makes
# the build fail intermittently with "autoapi/.../index.rst does not exist" under recent Sphinx.
# (These files land in docs/source/autoapi, which is gitignored.)
autoapi_keep_files = True

autoapi_member_order = "groupwise"


def _skip_decoder_specification_helpers(
    app: object, what: str, name: str, obj: object, skip: bool | None, options: object
) -> bool | None:
    """Skip static AutoAPI entries of generated decoder specification helpers.

    Helpers such as ``bp_osd = decoder_spec(...)`` are assignments, so AutoAPI, which reads source
    statically, would render them as data with no signature.  decoders.rst documents them instead
    with autodoc, which inspects their runtime signatures and docstrings.
    """
    if what != "data":
        return skip
    from qldpc.decoders.construction.specs import DecoderSpec

    module_name, _, attribute = name.rpartition(".")
    value = getattr(importlib.import_module(module_name), attribute, None)
    return_annotation = getattr(value, "__annotations__", {}).get("return")
    if typing.get_origin(return_annotation) is DecoderSpec:
        return True
    return skip


def _resolve_package_reexports(
    app: typing.Any, env: typing.Any, node: typing.Any, contnode: typing.Any
) -> typing.Any:
    """Resolve references to names that a qLDPC package re-exports from one of its modules.

    AutoAPI documents each object only in the module that defines it (see autoapi_options), but type
    annotations name objects by the paths that code imports them from, such as qldpc.objects.Pauli.
    This handler resolves such a reference to the documented object that the package re-exports.
    """
    if node.get("refdomain") != "py":
        return None
    package_name, _, attribute = node.get("reftarget", "").rpartition(".")
    if not package_name.startswith("qldpc"):
        return None
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        return None
    if attribute not in getattr(package, "__all__", ()):
        return None

    objects = env.get_domain("py").objects
    targets = [
        name
        for name in objects
        if name.startswith(f"{package_name}.")
        and name.endswith(f".{attribute}")
        and getattr(sys.modules.get(name.rpartition(".")[0]), attribute, None)
        is getattr(package, attribute)
    ]
    if len(targets) != 1:
        return None
    from sphinx.util.nodes import make_refnode

    entry = objects[targets[0]]
    return make_refnode(
        app.builder, node["refdoc"], entry.docname, entry.node_id, contnode, targets[0]
    )


def setup(app: typing.Any) -> None:
    """Register Sphinx event handlers."""
    app.connect("autoapi-skip-member", _skip_decoder_specification_helpers)
    app.connect("missing-reference", _resolve_package_reexports)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
