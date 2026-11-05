# SPDX-License-Identifier: Apache-2.0

"""The layout of the deprecated NoiseModel.noisy_circuit(..., insert_ticks=True).

NoiseModel.noisy_circuit imports this module within its deprecated block, so that this module can
import qldpc.circuits at import time without an import cycle.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from qldpc.circuits.noise_model.scheduling import (
    TickScheduler,
    _get_instruction_operations,
    _get_nodes,
    _place_annotations,
    _Segment,
)


class InsertTicksScheduler(TickScheduler):
    """The scheduler of the deprecated insert_ticks=True, with the rules of its released version.

    A TICK ends the current moment if the moment contains a gate, so consecutive TICKs give no empty
    moments.  A gate that takes time also starts a new moment if a gate that takes time already acts
    on one of its qubits in the current moment.  Every gate takes time, except a classically
    controlled Pauli gate.
    """

    def _get_moments(self, segment: _Segment, takes_time: Sequence[bool]) -> list[int]:
        # place each gate in the current moment, or in a new one after a TICK or where it takes
        # time on a qubit that is already occupied
        nodes = _get_nodes(segment.operations)
        gate_moments: dict[int, int] = {}  # the moment of each node with a gate
        moment = 0
        occupied: set[int] = set()  # the qubits of gates that take time in the moment
        closed = False  # whether a TICK has ended the current moment
        operations = iter(zip(nodes, segment.operations, takes_time))
        for instruction in segment.instructions:
            if instruction.name == "TICK":
                closed = bool(gate_moments)
            num_operations = len(_get_instruction_operations(instruction))
            for node, operation, gate_takes_time in itertools.islice(operations, num_operations):
                if not operation.is_gate:
                    continue
                if closed or (gate_takes_time and not occupied.isdisjoint(operation.qubits)):
                    moment, occupied, closed = moment + 1, set(), False
                if gate_takes_time:
                    occupied.update(operation.qubits)
                gate_moments[node] = moment

        # annotations and noise channels join the moments of operations before them
        return _place_annotations(segment.operations, nodes, gate_moments)
