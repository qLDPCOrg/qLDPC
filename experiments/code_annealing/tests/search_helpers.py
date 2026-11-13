# SPDX-License-Identifier: Apache-2.0

"""Small codes that several tests of the search share."""

from __future__ import annotations

import numpy as np

from qldpc import codes
from qldpc.codes.distance import CSSLogicalCounter


def steane_counter() -> CSSLogicalCounter:
    """The Steane code, with a counter."""
    return CSSLogicalCounter.from_code(codes.SteaneCode())


def weak_counter(num_qubits: int = 7) -> CSSLogicalCounter:
    """A code of one logical qubit and distance 1, which is worse than the Steane code.

    Its Z checks fix every qubit but the first, which carries the logical qubit, so a single X or Z
    on the first qubit is a logical operator.
    """
    return CSSLogicalCounter(np.zeros((0, num_qubits)), np.eye(num_qubits)[1:])


def canonical_checks(counter: CSSLogicalCounter) -> tuple[bytes, bytes]:
    """The checks of a code in reduced row echelon form, which identify its stabilizer group."""
    return counter.checks_x.tobytes(), counter.checks_z.tobytes()
