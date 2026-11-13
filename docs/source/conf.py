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
    statically, would render them as data with no signature.  decoders/reference.rst documents them
    instead with autodoc, which inspects their runtime signatures and docstrings.
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
    annotations and docstrings name objects by the paths that code imports them from, such as
    qldpc.objects.Pauli.  This handler resolves such a reference, or a reference to a member of such
    an object, such as qldpc.abstract.GroupMember.to_matrix, to the documented object.
    """
    if node.get("refdomain") != "py":
        return None
    names = node.get("reftarget", "").split(".")
    if names[0] != "qldpc":
        return None
    # split the target into a package, a name that it re-exports, and the name of a member, if any
    for index in range(len(names) - 1, 0, -1):
        try:
            package = importlib.import_module(".".join(names[:index]))
            break
        except ImportError:
            pass
    else:
        return None
    attribute = names[index]
    if attribute not in getattr(package, "__all__", ()):
        return None

    objects = env.get_domain("py").objects
    definitions = [
        name
        for name in objects
        if name.startswith(f"{package.__name__}.")
        and name.endswith(f".{attribute}")
        and getattr(sys.modules.get(name.rpartition(".")[0]), attribute, None)
        is getattr(package, attribute)
    ]
    if len(definitions) != 1:
        return None
    target = ".".join([definitions[0], *names[index + 1 :]])
    if target not in objects:
        return None
    from sphinx.util.nodes import make_refnode

    entry = objects[target]
    return make_refnode(app.builder, node["refdoc"], entry.docname, entry.node_id, contnode, target)


_PYTHON_DOCS = "https://docs.python.org/3/library"

# The documentation of the external objects that the API reference names, keyed by the names that it
# uses.  The strict docs build must not need network access, so it cannot fetch the inventories that
# intersphinx would use to find these.
_EXTERNAL_DOCS = {
    "UserWarning": f"{_PYTHON_DOCS}/exceptions.html#UserWarning",
    "ValueError": f"{_PYTHON_DOCS}/exceptions.html#ValueError",
    "abc.ABC": f"{_PYTHON_DOCS}/abc.html#abc.ABC",
    "collections.abc.Hashable": f"{_PYTHON_DOCS}/collections.abc.html#collections.abc.Hashable",
    "collections.abc.Mapping": f"{_PYTHON_DOCS}/collections.abc.html#collections.abc.Mapping",
    "enum.Enum": f"{_PYTHON_DOCS}/enum.html#enum.Enum",
    "enum.ReprEnum": f"{_PYTHON_DOCS}/enum.html#enum.ReprEnum",
    "float": f"{_PYTHON_DOCS}/functions.html#float",
    "int": f"{_PYTHON_DOCS}/functions.html#int",
    "list": f"{_PYTHON_DOCS}/stdtypes.html#list",
    "typing.Any": f"{_PYTHON_DOCS}/typing.html#typing.Any",
    "typing.Generic": f"{_PYTHON_DOCS}/typing.html#typing.Generic",
    "typing.NamedTuple": f"{_PYTHON_DOCS}/typing.html#typing.NamedTuple",
    "typing.Protocol": f"{_PYTHON_DOCS}/typing.html#typing.Protocol",
    "ldpc.BeliefFindDecoder": "https://software.roffe.eu/ldpc/ldpc/belief_find_decoder.html",
    "ldpc.BpOsdDecoder": "https://software.roffe.eu/ldpc/ldpc/bposd_decoder.html",
    "ldpc.bplsd_decoder.BpLsdDecoder": "https://software.roffe.eu/ldpc/ldpc/bplsd_decoder.html",
    "networkx.DiGraph": "https://networkx.org/documentation/stable/reference/classes/digraph.html",
    "numpy.dtype": "https://numpy.org/doc/stable/reference/generated/numpy.dtype.html",
    "numpy.ndarray": "https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html",
    "numpy.object_": "https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.object_",
    "pymatching.Matching": (
        "https://pymatching.readthedocs.io/en/stable/api.html#pymatching.matching.Matching"
    ),
    "sympy.combinatorics.Permutation": (
        "https://docs.sympy.org/latest/modules/combinatorics/permutations.html"
        "#sympy.combinatorics.permutations.Permutation"
    ),
}


def _resolve_external_references(
    app: typing.Any, env: typing.Any, node: typing.Any, contnode: typing.Any
) -> typing.Any:
    """Link references to the objects in _EXTERNAL_DOCS to their documentation.

    Only references in code style, such as those in docstrings and lists of base classes, are
    linked.  Signatures and lists of raised exceptions also name external objects that
    _EXTERNAL_DOCS omits, such as bool and TypeError, so linking names there would link some names
    but not others.

    An unqualified name from typing, such as Protocol, resolves only in a module that imports it
    from typing.  Code imports networkx lazily, as qldpc._utils.networkx, so a reference to a name
    in qldpc._utils.networkx resolves as a reference to that name in networkx.
    """
    from docutils import nodes

    if node.get("refdomain") != "py" or not isinstance(contnode, nodes.literal):
        return None
    target = node.get("reftarget", "")
    if target.startswith("qldpc._utils.networkx."):
        target = target.removeprefix("qldpc._utils.")
    module_name = node.get("py:module")
    if (
        module_name
        and hasattr(typing, target)
        and getattr(importlib.import_module(module_name), target, None) is getattr(typing, target)
    ):
        target = f"typing.{target}"
    if target not in _EXTERNAL_DOCS:
        return None
    reference = nodes.reference(
        "", "", internal=False, refuri=_EXTERNAL_DOCS[target], reftitle=target
    )
    reference.append(contnode)
    return reference


def _warn_on_missing_tuning_reference(
    app: typing.Any, env: typing.Any, node: typing.Any, contnode: typing.Any
) -> None:
    """Warn on a reference into the private tuning module that no documented object resolves.

    exact_search_tuning.rst documents qldpc.codes.distance._tuning, and refers to its fields and
    constants by name.  Sphinx resolves Python references only leniently unless nitpicky mode is on,
    which would report thousands of unrelated references, so this handler makes a renamed or removed
    name of that module fail the strict build.  It runs after every handler that could resolve the
    reference.
    """
    if node.get("refdomain") != "py":
        return
    module = "qldpc.codes.distance._tuning"
    target = node.get("reftarget", "")
    # Within the module, a reference to a name of the module is the record, a field of it, a
    # function, or an upper-case constant; other names, such as int, belong to other modules.
    local = node.get("py:module") == module and (
        target.partition(".")[0] in ("SearchTuning", "get_tuning", "use_tuning") or target.isupper()
    )
    if target.startswith(f"{module}.") or local:
        from sphinx.util import logging

        logging.getLogger(__name__).warning(
            f"unresolved reference to {target!r} in {module}", location=node
        )


def setup(app: typing.Any) -> None:
    """Register Sphinx event handlers."""
    app.connect("autoapi-skip-member", _skip_decoder_specification_helpers)
    app.connect("missing-reference", _resolve_package_reexports)
    app.connect("missing-reference", _resolve_external_references)
    app.connect("missing-reference", _warn_on_missing_tuning_reference, priority=900)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
