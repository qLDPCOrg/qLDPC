"""Surgery construction package (simplified — see
docs/superpowers/specs/2026-06-07-surgery-simplification-design.md).

Public API:
    build_gadget, build_bridge,
    build_single_ppm_circuit, build_joint_ppm_circuit,
    keep_only_observable,
    boost_gadget, cheeger_constant
"""

from __future__ import annotations

from .gadget import GadgetLayout, build_gadget
from .bridge import Bridge, build_bridge
from .circuit import (
    build_single_ppm_circuit, build_joint_ppm_circuit,
    keep_only_observable, logical_state_init,
)
from .cheeger import boost_gadget, cheeger_constant

__all__ = [
    "GadgetLayout",
    "Bridge",
    "build_gadget",
    "build_bridge",
    "build_single_ppm_circuit",
    "build_joint_ppm_circuit",
    "keep_only_observable",
    "logical_state_init",
    "boost_gadget",
    "cheeger_constant",
]
