# SPDX-License-Identifier: Apache-2.0

"""Check that the modules and subpackages of qldpc.codes.distance import each other in layers."""

from __future__ import annotations

from qldpc.conftest import check_import_layering

# Layers from shared foundations (first) to public interfaces (last), which hold the complete order
# of the top-level modules and subpackages (``README.md#sec-layers``).  Add each new top-level
# module or subpackage to one.
_LAYERS = (
    ("_code_matrices", "_threads"),
    ("_tuning",),
    ("_information_sets", "_cosets", "min_weight_bounds", "_sector_maps"),
    ("_brouwer_zimmermann", "_connected_cluster", "_counting", "_brute_force"),
    ("_auto",),
    ("distances",),
    ("logical_counts", "coset_leaders"),
)


def test_import_layering() -> None:
    """The top-level modules and subpackages of qldpc.codes.distance import each other in layers."""
    check_import_layering("qldpc.codes.distance", _LAYERS)
