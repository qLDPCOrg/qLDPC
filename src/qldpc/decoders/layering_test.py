# SPDX-License-Identifier: Apache-2.0

"""Check that the modules and subpackages of qldpc.decoders import each other in layers."""

from __future__ import annotations

from qldpc.conftest import check_import_layering

# The top-level modules and subpackages of qldpc.decoders, in layers from shared foundations
# (first) to complete workflows (last).  Add each new top-level module or subpackage to a layer.
_LAYERS = (
    ("dems", "protocols"),
    ("adapters", "common"),
    ("construction",),
    ("_capabilities",),
    ("custom", "external"),
    ("_resolution",),
    ("sinter",),
)


def test_import_layering() -> None:
    """The top-level modules and subpackages of qldpc.decoders import each other in layers."""
    check_import_layering("qldpc.decoders", _LAYERS)
