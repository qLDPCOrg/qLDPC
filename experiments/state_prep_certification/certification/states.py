# SPDX-License-Identifier: Apache-2.0

"""The stabilizer states that the circuits of the certification experiment prepare."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import stim

from qldpc import math as qldpc_math


def stabilizer_matrix(stabilizers: Any) -> npt.NDArray[np.uint8]:
    """State stabilizers, as ``stim.PauliString`` objects or their text, as rows ``[x | z]``.

    Text is parsed by stim first, since ``string_to_op`` would take a leading sign for a qubit.
    """
    return np.array(
        [qldpc_math.string_to_op(stim.PauliString(s)) for s in stabilizers], dtype=np.uint8
    )


def same_state(reference: list[str], other: list[str]) -> bool:
    """Whether every stabilizer of ``other`` has expectation +1 in the state of ``reference``."""
    tableau = stim.Tableau.from_stabilizers([stim.PauliString(text) for text in reference])
    simulator = stim.TableauSimulator()
    simulator.do_tableau(tableau, list(range(len(tableau))))
    return all(simulator.peek_observable_expectation(stim.PauliString(text)) == 1 for text in other)
