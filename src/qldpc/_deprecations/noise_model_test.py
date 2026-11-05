# SPDX-License-Identifier: Apache-2.0

"""Unit tests for noise_model.py."""

from __future__ import annotations

import stim

from qldpc._deprecations.noise_model import InsertTicksScheduler


def test_insert_ticks_scheduler() -> None:
    """Gates that take time on a shared qubit, and TICKs after a gate, start new moments.

    Consecutive TICKs give no empty moment, and the noise channel joins the moment of the gate
    before it on its qubit.
    """
    circuit = stim.Circuit("TICK\nCX 0 1 1 2\nH 3\nTICK\nTICK\nX_ERROR(0.1) 0\nH 0 0")
    assert InsertTicksScheduler().get_schedule(circuit).to_circuit() == stim.Circuit("""
        CX 0 1
        X_ERROR(0.1) 0
        TICK
        CX 1 2
        H 3
        TICK
        H 0
        TICK
        H 0
    """)
