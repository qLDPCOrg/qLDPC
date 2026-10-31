# SPDX-License-Identifier: Apache-2.0

"""Check that the modules and subpackages of qldpc.circuits import each other in layers."""

from __future__ import annotations

from qldpc.conftest import check_import_layering

# The top-level modules and subpackages of qldpc.circuits, in layers from shared foundations
# (first) to complete workflows (last).  Add each new top-level module or subpackage to a layer.
_LAYERS = (
    ("common", "bookkeeping"),
    ("transforms", "_post_selection"),
    ("encoding", "noise_model"),
    ("benchmarking", "certification", "memory", "transversal"),
)


def test_import_layering() -> None:
    """The top-level modules and subpackages of qldpc.circuits import each other in layers."""
    check_import_layering("qldpc.circuits", _LAYERS)
