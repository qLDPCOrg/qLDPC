# SPDX-License-Identifier: Apache-2.0

"""Check that the modules of this subpackage import each other in layers."""

from __future__ import annotations

from qldpc.conftest import check_import_layering

# The layers of the modules, from the lowest to the highest, consistent with the dependencies that
# the section "How the modules depend on each other" of the README of qldpc.codes.distance gives.
# Add each new module to one.
_LAYERS = (
    ("affine_instance", "affine_search"),
    ("coset_tables",),
    ("coset_routes",),
    ("coset", "distance"),
)


def test_import_layering() -> None:
    """The modules of qldpc.codes.distance._brouwer_zimmermann import each other in layers."""
    check_import_layering("qldpc.codes.distance._brouwer_zimmermann", _LAYERS)
