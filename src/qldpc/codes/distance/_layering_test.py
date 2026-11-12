# SPDX-License-Identifier: Apache-2.0

"""Check that the modules and subpackages of qldpc.codes.distance import each other in layers."""

from __future__ import annotations

from qldpc.conftest import check_import_layering

# The top-level modules and subpackages of qldpc.codes.distance, in layers from shared foundations
# (first) to public interfaces (last).  Add each new top-level module or subpackage to a layer.
_LAYERS = (
    ("_binary",),
    ("_tuning",),
    ("_brute_force", "_information_sets", "_cosets"),
    ("_brouwer_zimmermann", "_connected_support", "_counting"),
    ("_coset_leader_search", "core"),
    ("logical_counts",),
)


def test_import_layering() -> None:
    """The top-level modules and subpackages of qldpc.codes.distance import each other in layers."""
    check_import_layering("qldpc.codes.distance", _LAYERS)
