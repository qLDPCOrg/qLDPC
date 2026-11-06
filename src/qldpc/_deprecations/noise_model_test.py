# SPDX-License-Identifier: Apache-2.0

"""Unit tests for noise_model.py."""

from __future__ import annotations

import pytest
import stim

from qldpc import circuits
from qldpc._deprecations.noise_model import apply_insert_ticks, lay_out_with_ticks


def test_deprecated_insert_ticks() -> None:
    """insert_ticks=True lays a circuit out in moments, and insert_ticks=False has no effect."""
    circuit = stim.Circuit("CX 0 1 1 2")
    scheduler = circuits.ASAPScheduler()
    with pytest.warns(DeprecationWarning, match="insert_ticks=True"):
        layout, _ = apply_insert_ticks(True, circuit, scheduler)
    assert layout == lay_out_with_ticks(circuit)[0]
    with pytest.warns(DeprecationWarning, match="insert_ticks=False"):
        assert apply_insert_ticks(False, circuit, scheduler) == (circuit, scheduler)


def test_deprecated_insert_ticks_layout() -> None:
    """Gates on a shared qubit, and TICKs after a gate, start new moments.

    Consecutive TICKs give no empty moment, and the noise channel joins the moment of the gate
    before it on its qubit.  Classically controlled gates take no time, but still start new moments
    where they share a qubit with another gate.
    """
    circuit = stim.Circuit("TICK\nCX 0 1 1 2\nH 3\nTICK\nTICK\nX_ERROR(0.1) 0\nH 0 0")
    layout, _ = lay_out_with_ticks(circuit)
    assert layout == stim.Circuit("""
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

    circuit = stim.Circuit("M 0\nCX rec[-1] 1\nH 1\nCZ 1 rec[-1]")
    layout, _ = lay_out_with_ticks(circuit)
    assert layout == stim.Circuit("M 0\nCX rec[-1] 1\nTICK\nH 1\nTICK\nCZ 1 rec[-1]")
