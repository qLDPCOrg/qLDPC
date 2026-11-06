# SPDX-License-Identifier: Apache-2.0

"""The deprecated insert_ticks argument of NoiseModel.noisy_circuit.

NoiseModel.noisy_circuit imports this module within its deprecated block, so that this module can
import qldpc.circuits at import time without an import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence

import stim

from qldpc import _deprecations
from qldpc._utils import get_external_caller_stacklevel
from qldpc.circuits.noise_model.scheduling import (
    Schedule,
    Scheduler,
    TickScheduler,
    _Segment,
)


def apply_insert_ticks(
    insert_ticks: bool, circuit: stim.Circuit, scheduler: Scheduler | Schedule
) -> tuple[stim.Circuit, Scheduler | Schedule]:
    """Warn about insert_ticks, and give the circuit and scheduler that noisy_circuit then uses.

    With insert_ticks=True, these are the layout of lay_out_with_ticks and a scheduler of its
    moments.  insert_ticks=False has no effect.
    """
    _deprecations.warn_deprecated(
        (
            "noisy_circuit(..., insert_ticks=True) is deprecated and will be removed in a future"
            " release.  To lay out a noisy circuit in moments, add noise to the layout of a"
            " schedule, as in noise_model.noisy_circuit(layout) for layout ="
            " TickScheduler().get_schedule(circuit).to_circuit(), and pass"
            " scheduler=TickScheduler() to the noise model so that its idle noise uses those"
            " moments"
        )
        if insert_ticks
        else (
            "noisy_circuit(..., insert_ticks=False) is deprecated, has no effect, and will be"
            " removed in a future release, since noisy_circuit keeps the TICKs of its input and"
            " adds only noise.  Remove the argument"
        ),
        stacklevel=get_external_caller_stacklevel(),
    )
    if insert_ticks:
        return lay_out_with_ticks(circuit)
    return circuit, scheduler


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
