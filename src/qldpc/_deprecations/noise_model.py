# SPDX-License-Identifier: Apache-2.0

"""The layout of the deprecated NoiseModel.noisy_circuit(..., insert_ticks=True).

NoiseModel.noisy_circuit imports this module within its deprecated block, so that this module can
import qldpc.circuits at import time without an import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence

import stim

from qldpc.circuits.noise_model.scheduling import (
    Scheduler,
    TickScheduler,
    _Segment,
)


def lay_out_with_ticks(circuit: stim.Circuit) -> tuple[stim.Circuit, Scheduler]:
    """Lay a circuit out in moments as insert_ticks=True did, with a scheduler of those moments.

    A TICK ends the current moment if the moment contains a gate, so consecutive TICKs give no empty
    moments.  A gate also starts a new moment if a gate already acts on one of its qubits in the
    current moment, even if either gate is classically controlled.  Every gate takes time, except a
    classically controlled Pauli gate, so qubits idle in the moments of those gates.
    """
    scheduler = _InsertTicksScheduler()
    return scheduler.get_schedule(circuit).to_circuit(), scheduler


class _InsertTicksScheduler(TickScheduler):
    """A TickScheduler with the rules of the released insert_ticks=True (see lay_out_with_ticks)."""

    def _place_gates(
        self, segment: _Segment, nodes: Sequence[int], takes_time: Sequence[bool]
    ) -> dict[int, int]:
        # place each gate in the current moment, or in a new one after a TICK or where it acts on a
        # qubit that is already occupied
        gate_moments: dict[int, int] = {}
        moment = 0
        occupied: set[int] = set()  # the qubits of gates in the moment
        closed = False  # whether a TICK has ended the current moment
        for instruction, span in zip(segment.instructions, segment.spans):
            if instruction.name == "TICK":
                closed = bool(gate_moments)
            for node, operation in zip(nodes[span], segment.operations[span]):
                if not operation.is_gate:
                    continue
                if closed or not occupied.isdisjoint(operation.qubits):
                    moment, occupied, closed = moment + 1, set(), False
                occupied.update(operation.qubits)
                gate_moments[node] = moment
        return gate_moments
