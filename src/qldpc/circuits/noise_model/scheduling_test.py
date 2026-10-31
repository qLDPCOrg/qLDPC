# SPDX-License-Identifier: Apache-2.0

"""Unit tests for scheduling.py."""

from __future__ import annotations

import itertools

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
    """Operations are as early as possible, and resets as late as possible by default.

    An observable follows the moment of the gate before it on its qubit.
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
    scheduler = circuits.Scheduler.asap(late_resets=False, single_qubit_gates_take_time=True)
    assert scheduler.schedule(CIRCUIT) == stim.Circuit(
        expected.format(resets="R 0 1 2\nTICK", late_reset="")
    )
    scheduler = circuits.Scheduler.asap(single_qubit_gates_take_time=True)
    assert scheduler.schedule(CIRCUIT) == stim.Circuit(
        expected.format(resets="R 0\nTICK\nR 1", late_reset="R 2")
    )


def test_schedule_as_late_as_possible() -> None:
    """Operations are as late as possible, and measurements as early as possible by default."""
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
    scheduler = circuits.Scheduler.alap(early_measurements=False, single_qubit_gates_take_time=True)
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
    scheduler = circuits.Scheduler.alap(single_qubit_gates_take_time=True)
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


def test_schedule_untimed_gates() -> None:
    """Single-qubit gates share moments with the operations before them by default.

    ``I`` takes no time, like other single-qubit gates, while ``II`` takes a moment.
    """
    scheduler = circuits.Scheduler.asap(late_resets=False)
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
    circuit = stim.Circuit("H 0\nI 0\nII 0 1\nCX 0 1")
    assert scheduler.schedule(circuit) == stim.Circuit("H 0\nI 0\nII 0 1\nTICK\nCX 0 1")

    # a classically controlled gate takes no time
    circuit = stim.Circuit("M 0\nCX rec[-1] 1\nCZ 1 2")
    assert scheduler.schedule(circuit) == circuit


def test_schedule_measurement_order() -> None:
    """Measurements may change order, unless their order is preserved.

    A detector keeps reading the same measurements, and a REPEAT block that reads only its own
    measurements does not prevent measurements from changing order.
    """
    circuit = stim.Circuit("""
        H 0
        CX 0 1
        M 1
        M 2
        DETECTOR rec[-2]
        REPEAT 2 {
            M 0
            DETECTOR rec[-1]
        }
    """)
    block = "REPEAT 2 {\nM 0\nDETECTOR rec[-1]\nTICK\n}"
    scheduler = circuits.Scheduler.asap()
    assert scheduler.schedule(circuit) == stim.Circuit(
        f"H 0\nCX 0 1\nM 2\nTICK\nM 1\nDETECTOR rec[-1]\nTICK\n{block}"
    )
    scheduler = circuits.Scheduler.asap(preserve_measurement_order=True)
    assert scheduler.schedule(circuit) == stim.Circuit(
        f"H 0\nCX 0 1\nTICK\nM 1 2\nDETECTOR rec[-2]\nTICK\n{block}"
    )

    # measurements keep their order if a REPEAT block reads measurements from outside it
    circuit = stim.Circuit("""
        H 0
        CX 0 1
        M 1
        M 2
        REPEAT 2 {
            REPEAT 2 {
                M 2
                DETECTOR rec[-1] rec[-2]
            }
        }
    """)
    assert circuits.Scheduler.asap().schedule(circuit) == stim.Circuit("""
        H 0
        CX 0 1
        TICK
        M 1 2
        TICK
        REPEAT 2 {
            REPEAT 2 {
                M 2
                DETECTOR rec[-1] rec[-2]
                TICK
            }
        }
    """)


def test_schedule_preserves_noisy_circuits() -> None:
    """Built-in rescheduling preserves a circuit's detector error model under gate noise.

    The expected model comes from inserting each gate's noise right after it.  The circuit has a
    Pauli product, classical control, a chain of correlated errors, an observable with a Pauli
    target, and a REPEAT block whose detectors read measurements from before the block and from its
    previous iteration.
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
        MX 2
        CX rec[-1] 3
        E(0.1) X3
        ELSE_CORRELATED_ERROR(0.2) X0
        M 3
        DETECTOR rec[-1]
        S 0
        OBSERVABLE_INCLUDE(0) Z0
        M 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-2]
        REPEAT 2 {
            R 2
            CX 1 2
            M 2
            DETECTOR rec[-1] rec[-2]
        }
    """)
    expected = _with_gate_noise(circuit).detector_error_model(approximate_disjoint_errors=True)
    for single_qubit_gates_take_time, preserve_measurement_order in itertools.product(
        [False, True], repeat=2
    ):
        kwargs = {
            "single_qubit_gates_take_time": single_qubit_gates_take_time,
            "preserve_measurement_order": preserve_measurement_order,
        }
        noise_model = circuits.NoiseModel(
            clifford_1q_error=0.01,
            clifford_2q_error=0.02,
            scheduler=circuits.Scheduler.preserve_ticks(
                single_qubit_gates_take_time=single_qubit_gates_take_time
            ),
        )
        for scheduler in [
            circuits.Scheduler.asap(**kwargs),
            circuits.Scheduler.asap(late_resets=False, **kwargs),
            circuits.Scheduler.alap(**kwargs),
            circuits.Scheduler.alap(early_measurements=False, **kwargs),
        ]:
            noisy = noise_model.noisy_circuit(scheduler.schedule(circuit))
            dem = noisy.detector_error_model(approximate_disjoint_errors=True)
            assert dem.approx_equals(expected, atol=1e-12)


def _with_gate_noise(circuit: stim.Circuit) -> stim.Circuit:
    """Insert depolarizing noise right after each H, S, and CX gate without classical control."""
    noisy_circuit = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            body = _with_gate_noise(op.body_copy())
            noisy_circuit.append(stim.CircuitRepeatBlock(op.repeat_count, body))
            continue
        noisy_circuit.append(op)
        for group in op.target_groups():
            if op.name in ("H", "S"):
                noisy_circuit.append("DEPOLARIZE1", group, 0.01)
            elif op.name == "CX" and not group[0].is_measurement_record_target:
                noisy_circuit.append("DEPOLARIZE2", group, 0.02)
    return noisy_circuit


def test_schedule_coordinates() -> None:
    """Rescheduling keeps the coordinates of qubits and detectors, including in REPEAT blocks."""
    circuit = stim.Circuit("""
        R 0 1
        M 1
        SHIFT_COORDS(0, 1)
        QUBIT_COORDS(2, 3) 0
        DETECTOR(0, 0) rec[-1]
        REPEAT 2 {
            CX 0 1
            SHIFT_COORDS(0, 1)
            M 1
            DETECTOR(0, 0) rec[-1] rec[-2]
        }
        M 0
        DETECTOR(1, 0) rec[-1]
    """)
    scheduled = circuits.Scheduler.asap().schedule(circuit)
    assert scheduled.get_detector_coordinates() == circuit.get_detector_coordinates()
    assert scheduled.get_final_qubit_coordinates() == circuit.get_final_qubit_coordinates()


def test_schedule_other_circuits() -> None:
    """Schedulers return circuits of the type they are given.

    Rescheduling ignores TICKs, and keeps the tags of instructions and REPEAT blocks, and Pauli
    products.
    """
    circuit = StimCircuitWrapper(stim.Circuit("H[gate] 0\nTICK\nSPP X1*Z2"))
    scheduled = circuits.Scheduler.asap().schedule(circuit)
    assert isinstance(scheduled, StimCircuitWrapper)
    assert scheduled.stim_circuit == stim.Circuit("H[gate] 0\nSPP X1*Z2")
    scheduled = circuits.Scheduler.preserve_ticks().schedule(circuit)
    assert isinstance(scheduled, StimCircuitWrapper)
    assert scheduled.stim_circuit == circuit.stim_circuit

    circuit = StimCircuitWrapper(stim.Circuit("H 0\nREPEAT[block] 2 {\nH[gate] 0\n}"))
    scheduled = circuits.Scheduler.asap().schedule(circuit)
    # stim's equality of circuits ignores the tags of REPEAT blocks, so compare their text
    assert str(scheduled.stim_circuit) == str(
        stim.Circuit("H 0\nTICK\nREPEAT[block] 2 {\nH[gate] 0\nTICK\n}")
    )


def test_custom_scheduler() -> None:
    """By default, every gate of a custom scheduler takes time, except for classical control.

    Annotations and noise channels take no time, for any scheduler.
    """
    classical_control = stim.CircuitInstruction("CX", [stim.target_rec(-1), 0])
    annotation = stim.CircuitInstruction("QUBIT_COORDS", [0], [0])
    noise = stim.CircuitInstruction("X_ERROR", [0], [0.1])
    for scheduler in [_KeepCircuit(), circuits.Scheduler.asap(single_qubit_gates_take_time=True)]:
        assert scheduler.takes_time(stim.CircuitInstruction("H", [0]))
        assert not any(scheduler.takes_time(op) for op in [classical_control, annotation, noise])


class _KeepCircuit(circuits.Scheduler):
    """A scheduler that keeps a circuit as it is."""

    def schedule(self, circuit: circuits.StimCircuitLike) -> circuits.StimCircuitLike:
        return circuit


def test_scheduler_repr() -> None:
    """A scheduler's repr reconstructs it."""
    schedulers = [
        circuits.Scheduler.preserve_ticks(single_qubit_gates_take_time=flag)
        for flag in [False, True]
    ]
    for flags in itertools.product([False, True], repeat=3):
        shorten, preserve, take_time = flags
        options = {
            "preserve_measurement_order": preserve,
            "single_qubit_gates_take_time": take_time,
        }
        schedulers.append(circuits.Scheduler.asap(late_resets=shorten, **options))
        schedulers.append(circuits.Scheduler.alap(early_measurements=shorten, **options))
    for scheduler in schedulers:
        assert eval(repr(scheduler), {"Scheduler": circuits.Scheduler}) == scheduler
