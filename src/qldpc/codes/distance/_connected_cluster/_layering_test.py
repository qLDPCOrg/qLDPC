# SPDX-License-Identifier: Apache-2.0

"""Check that the modules of this subpackage import each other in layers."""

from __future__ import annotations

from qldpc.conftest import check_import_layering

# The layers of the modules, from the lowest to the highest, which hold their complete order
# (``README.md#sec-layers`` of qldpc.codes.distance).  Add each new module to one.
_LAYERS = (("sparse_checks",), ("search",), ("coset",))


def test_import_layering() -> None:
    """The modules of qldpc.codes.distance._connected_cluster import each other in layers."""
    check_import_layering("qldpc.codes.distance._connected_cluster", _LAYERS)
