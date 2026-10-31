# SPDX-License-Identifier: Apache-2.0

"""Unit tests for scheduling.py."""

from __future__ import annotations

import itertools

import pytest
import stim

from qldpc import circuits
from qldpc.circuits.conftest import StimCircuitWrapper

# A circuit whose moments differ for each way of scheduling it.
CIRCUIT = stim.Circuit("""
    QUBIT_COORDS(0, 0) 0
    R 0 1 2
    H 0
    CX 0 1
    M 0
    CX 1 2
    OBSERVABLE_INCLUDE(0) Z2
    H 2
    M 1 2
    DETECTOR rec[-3]
    OBSERVABLE_INCLUDE(0) rec[-1]
""")


def test_schedule_as_early_as_possible() -> None:
    """Operations are as early as possible; an observable follows the moment of the gate before it.

    With ``shorten_lifetimes``, resets are as late as possible.
    """
    expected = """
        QUBIT_COORDS(0, 0) 0
        {resets}
        H 0
        TICK
        {late_reset}
        CX 0 1
        TICK
        M 0
        CX 1 2
        DETECTOR rec[-1]
        TICK
        OBSERVABLE_INCLUDE(0) Z2
        H 2
        M 1
        TICK
        M 2
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    scheduler = circuits.Scheduler(reschedule=True)
    assert scheduler.schedule(CIRCUIT) == stim.Circuit(
        expected.format(resets="R 0 1 2\nTICK", late_reset="")
    )
    scheduler = circuits.Scheduler(reschedule=True, shorten_lifetimes=True)
    assert scheduler.schedule(CIRCUIT) == stim.Circuit(
        expected.format(resets="R 0\nTICK\nR 1", late_reset="R 2")
    )


def test_schedule_as_late_as_possible() -> None:
    """Operations are as late as possible.

    With ``shorten_lifetimes``, measurements are as early as possible.
    """
    resets = """
        QUBIT_COORDS(0, 0) 0
        R 0
        TICK
        R 1
        H 0
        TICK
        R 2
        CX 0 1
        TICK
    """
    scheduler = circuits.Scheduler(reschedule=True, alap=True)
    assert scheduler.schedule(CIRCUIT) == stim.Circuit(
        resets
        + """
        CX 1 2
        TICK
        OBSERVABLE_INCLUDE(0) Z2
        H 2
        TICK
        M 0 1 2
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )
    scheduler = circuits.Scheduler(reschedule=True, alap=True, shorten_lifetimes=True)
    assert scheduler.schedule(CIRCUIT) == stim.Circuit(
        resets
        + """
        M 0
        CX 1 2
        TICK
        OBSERVABLE_INCLUDE(0) Z2
        H 2
        M 1
        TICK
        M 2
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )


def test_schedule_instantaneous_1q_gates() -> None:
    """Single-qubit gates share moments with the operations before them."""
    scheduler = circuits.Scheduler(reschedule=True, instantaneous_1q_gates=True)
    assert scheduler.schedule(CIRCUIT) == stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        R 0 1 2
        H 0
        TICK
        CX 0 1
        TICK
        M 0
        CX 1 2
        DETECTOR rec[-1]
        TICK
        OBSERVABLE_INCLUDE(0) Z2
        H 2
        M 1 2
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)


def test_schedule_preserves_noisy_circuits() -> None:
    """Every scheduler preserves a circuit's detector error model under gate noise.

    The expected model comes from inserting each gate's noise right after it.  The circuit has a
    Pauli product, classical control, measurements that change order, and detectors whose
    measurements are not in the order of the detectors.
    """
    circuit = stim.Circuit("""
        R 0 1 2 3
        H 0 2
        CX 0 1
        TICK
        MPP X0*X1
        M 3
        DETECTOR rec[-2]
        DETECTOR rec[-1]
        CX 2 3
        MX 2
        CX rec[-3] 3
        M 3
        S 0
        OBSERVABLE_INCLUDE(0) Z0
        CX 0 1
        M 0 1
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """)
    noisy_circuit = stim.Circuit()
    for op in circuit:
        noisy_circuit.append(op)
        for group in op.target_groups():
            if op.name in ("H", "S"):
                noisy_circuit.append("DEPOLARIZE1", group, 0.01)
            elif op.name == "CX" and not group[0].is_measurement_record_target:
                noisy_circuit.append("DEPOLARIZE2", group, 0.02)
    expected = noisy_circuit.detector_error_model()
    for alap, shorten_lifetimes, instantaneous_1q_gates in itertools.product(
        [False, True], repeat=3
    ):
        scheduler = circuits.Scheduler(
            reschedule=True,
            alap=alap,
            shorten_lifetimes=shorten_lifetimes,
            instantaneous_1q_gates=instantaneous_1q_gates,
        )
        noise_model = circuits.NoiseModel(
            clifford_1q_error=0.01,
            clifford_2q_error=0.02,
            scheduler=circuits.Scheduler(instantaneous_1q_gates=instantaneous_1q_gates),
        )
        noisy = noise_model.noisy_circuit(scheduler.schedule(circuit), insert_ticks=False)
        assert noisy.detector_error_model().approx_equals(expected, atol=1e-12)


def test_schedule_other_circuits() -> None:
    """Schedulers return circuits of the type they are given, and keep a circuit by default."""
    circuit = StimCircuitWrapper(stim.Circuit("H 0\nTICK\nMPP X0*Z1"))
    scheduled = circuits.Scheduler(reschedule=True).schedule(circuit)
    assert isinstance(scheduled, StimCircuitWrapper)
    assert scheduled.stim_circuit == stim.Circuit("H 0\nTICK\nMPP X0*Z1")
    assert circuits.Scheduler().schedule(circuit) is circuit


def test_schedule_errors() -> None:
    with pytest.raises(ValueError, match="require `reschedule=True`"):
        circuits.Scheduler(alap=True)
    scheduler = circuits.Scheduler(reschedule=True)
    with pytest.raises(ValueError, match="REPEAT"):
        scheduler.schedule(stim.Circuit("REPEAT 2 {\nH 0\n}"))
    with pytest.raises(ValueError, match="X_ERROR"):
        scheduler.schedule(stim.Circuit("X_ERROR(0.1) 0"))
