# SPDX-License-Identifier: Apache-2.0

"""Experimental quantum operators.

A ``PhasedPauliString`` is a Pauli string followed by layers of π/2 and π/4 Pauli rotations, which
is the form that a Pauli error takes after it propagates through a Clifford+T circuit.

This subpackage mirrors ``qldpc.objects``, where its tools may move once their API settles.
"""

from .phased_pauli_string import PhasedPauliString

__all__ = [
    "PhasedPauliString",
]
