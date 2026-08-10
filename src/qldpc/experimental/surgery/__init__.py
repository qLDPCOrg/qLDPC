"""Lattice-surgery constructions for CSS / quantum LDPC codes.

EXPERIMENTAL: this subpackage is under active development; its public API is unstable and may
change without notice or deprecation.

Fault-tolerant logical Pauli-product measurement (PPM) via code surgery: single-PPM "gadgets",
joint-PPM "bridges" (universal adapters), a distance boost, and a stim circuit layer. See the
individual modules for references.

Public API:
    build_gadget, GadgetLayout        — single-PPM gadget
    build_bridge, Bridge              — joint-PPM adapter (bridge)
    boost_gadget, cheeger_constant    — distance boost + boundary Cheeger check
    build_single_ppm_circuit, build_joint_ppm_circuit,
    logical_state_init, keep_only_observable   — stim circuit assembly
"""

from __future__ import annotations

from .bridge import Bridge, build_bridge
from .cheeger import boost_gadget, cheeger_constant
from .circuit import (
    build_joint_ppm_circuit,
    build_single_ppm_circuit,
    keep_only_observable,
    logical_state_init,
)
from .gadget import GadgetLayout, build_gadget

__all__ = [
    "Bridge",
    "GadgetLayout",
    "boost_gadget",
    "build_bridge",
    "build_gadget",
    "build_joint_ppm_circuit",
    "build_single_ppm_circuit",
    "cheeger_constant",
    "keep_only_observable",
    "logical_state_init",
]
