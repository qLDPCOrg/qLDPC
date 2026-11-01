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


def test_asap() -> None:
    """Operations are as early as possible, and resets as late as possible by default.

    An observable shares the moment of the gate before it on its qubit, and a detector shares the
    moment of the measurement that it reads.
    """
    expected = """
        QUBIT_COORDS(0, 0) 0
        {resets}
        TICK
        {reset_1}
        H 0
        TICK
        {reset_2}
        CX 0 1
        TICK
        M 0
        CX 1 2
        OBSERVABLE_INCLUDE(0) Z2
        DETECTOR rec[-1]
        TICK
        H 2
        M 1
        TICK
        M 2
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    scheduler = circuits.Scheduler.asap(late_resets=False, single_qubit_gates_take_time=True)
    circuit, measurements = scheduler.get_schedule(CIRCUIT).to_circuit()
    assert circuit == stim.Circuit(expected.format(resets="R 0 1 2", reset_1="", reset_2=""))
    assert measurements == (0, 1, 2)
    scheduler = circuits.Scheduler.asap(single_qubit_gates_take_time=True)
    circuit, _ = scheduler.get_schedule(CIRCUIT).to_circuit()
    assert circuit == stim.Circuit(expected.format(resets="R 0", reset_1="R 1", reset_2="R 2"))


def test_alap() -> None:
    """Operations are as late as possible, and measurements as early as possible by default."""
    expected = """
        R 0
        TICK
        R 1
        H 0
        TICK
        R 2
        CX 0 1
        TICK
        {early_measurement}
        CX 1 2
        TICK
        OBSERVABLE_INCLUDE(0) Z2
        H 2
        {measurements}
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    scheduler = circuits.Scheduler.alap(early_measurements=False, single_qubit_gates_take_time=True)
    circuit, _ = scheduler.get_schedule(CIRCUIT).to_circuit()
    assert circuit == stim.Circuit(
        expected.format(early_measurement="", measurements="TICK\nQUBIT_COORDS(0, 0) 0\nM 0 1 2")
    )
    scheduler = circuits.Scheduler.alap(single_qubit_gates_take_time=True)
    circuit, _ = scheduler.get_schedule(CIRCUIT).to_circuit()
    assert circuit == stim.Circuit(
        expected.format(
            early_measurement="M 0", measurements="M 1\nTICK\nQUBIT_COORDS(0, 0) 0\nM 2"
        )
    )


def test_preserve_ticks() -> None:
    """Moments end at TICKs, and where an operation that takes time reuses a qubit.

    Single-qubit gates take no time by default, nor do classically controlled gates, annotations,
    or noise channels.  ``I`` is a single-qubit gate, while ``II`` takes a moment.
    """
    circuit = stim.Circuit("""
        H 0
        I 0
        CX 0 1
        H 1
        X_ERROR(0.1) 1
        CX 1 2
        TICK
        M 0
        CX rec[-1] 1
        II 1 2
    """)
    schedule = circuits.Scheduler.preserve_ticks().get_schedule(circuit)
    assert schedule.get_moments() == [
        list(stim.Circuit("H 0\nI 0\nCX 0 1\nH 1\nX_ERROR(0.1) 1")),
        list(stim.Circuit("CX 1 2")),
        list(stim.Circuit("M 0\nCX rec[-1] 1\nII 1 2")),
    ]
    schedule = circuits.Scheduler.preserve_ticks(single_qubit_gates_take_time=True).get_schedule(
        circuit
    )
    assert [len(moment) for moment in schedule.get_moments() if isinstance(moment, list)] == [
        1,
        1,
        1,
        2,
        1,
        3,
    ]


def test_measurement_order() -> None:
    """Measurements may change order, while instructions keep reading the same measurements.

    The first iteration of a REPEAT block is written out if it reads a measurement before the block
    at another offset than later iterations read the measurements of earlier iterations.
    """
    circuit = stim.Circuit("""
        H 0
        CX 0 1
        M 1
        M 2
        REPEAT 3 {
            M 2
            DETECTOR rec[-1] rec[-2]
        }
    """)
    scheduled, measurements = circuits.Scheduler.asap().get_schedule(circuit).to_circuit()
    assert scheduled == stim.Circuit("""
        H 0
        CX 0 1
        M 2
        TICK
        M 1
        TICK
        M 2
        DETECTOR rec[-1] rec[-3]
        TICK
        REPEAT 2 {
            M 2
            DETECTOR rec[-1] rec[-2]
            TICK
        }
    """)
    assert measurements == (1, 0, 2, 3, 4)

    # a REPEAT block that reads only measurements of its own iterations stays whole
    circuit = stim.Circuit("REPEAT 3 {\nCX 1 2\nCX 1 2\nM 1\nM 0\nDETECTOR rec[-2]\n}")
    scheduled, measurements = circuits.Scheduler.asap().get_schedule(circuit).to_circuit()
    assert scheduled == stim.Circuit("""
        REPEAT 3 {
            CX 1 2
            M 0
            TICK
            CX 1 2
            TICK
            M 1
            DETECTOR rec[-1]
            TICK
        }
    """)
    assert measurements == (1, 0, 3, 2, 5, 4)


def test_schedules_keep_noisy_circuits() -> None:
    """The layout of every built-in schedule has the error model of the circuit under gate noise.

    The noise of each gate is inserted right after it, independently of moments.  The circuit has a
    Pauli product, classical control, a chain of correlated errors, an observable with a Pauli
    target, and a REPEAT block whose detectors read measurements from before the block and from its
    previous iteration, and a nested REPEAT block.
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
            REPEAT 2 {
                R 2
                CX 1 2
                M 2
                DETECTOR rec[-1] rec[-2]
            }
        }
    """)
    expected = _with_gate_noise(circuit).detector_error_model(approximate_disjoint_errors=True)
    for single_qubit_gates_take_time, shorten_lifetimes in itertools.product([False, True], [0, 1]):
        options = {"single_qubit_gates_take_time": single_qubit_gates_take_time}
        for scheduler in [
            circuits.Scheduler.preserve_ticks(**options),
            circuits.Scheduler.asap(late_resets=bool(shorten_lifetimes), **options),
            circuits.Scheduler.alap(early_measurements=bool(shorten_lifetimes), **options),
        ]:
            scheduled, _ = scheduler.get_schedule(circuit).to_circuit()
            dem = _with_gate_noise(scheduled).detector_error_model(approximate_disjoint_errors=True)
            assert dem.approx_equals(expected, atol=1e-12)


def _with_gate_noise(circuit: stim.Circuit) -> stim.Circuit:
    """Insert depolarizing noise right after each H, S, and CX gate without classical control."""
    noisy_circuit = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            body = _with_gate_noise(op.body_copy())
            noisy_circuit.append(stim.CircuitRepeatBlock(op.repeat_count, body))
            continue
        for group in op.target_groups() if op.name in ("H", "S", "CX") else [op.targets_copy()]:
            noisy_circuit.append(op.name, group, op.gate_args_copy())
            if op.name in ("H", "S"):
                noisy_circuit.append("DEPOLARIZE1", group, 0.01)
            elif op.name == "CX" and not group[0].is_measurement_record_target:
                noisy_circuit.append("DEPOLARIZE2", group, 0.02)
    return noisy_circuit


def test_coordinates_and_tags() -> None:
    """Layouts keep the coordinates of qubits and detectors, and the tags of instructions.

    Schedulers accept stim-wrapping circuits.
    """
    circuit = stim.Circuit("""
        R 0 1
        CX 0 1
        M 1
        DETECTOR(0, 0) rec[-1]
        SHIFT_COORDS(0, 1)
        QUBIT_COORDS(2, 3) 0
        REPEAT[block] 2 {
            CX[gate] 0 1
            SHIFT_COORDS(0, 1)
            M 1
            DETECTOR(0, 0) rec[-1] rec[-2]
        }
        M 0
        DETECTOR(1, 0) rec[-1]
    """)
    schedule = circuits.Scheduler.asap().get_schedule(StimCircuitWrapper(circuit))
    scheduled, _ = schedule.to_circuit()
    assert scheduled.get_detector_coordinates() == circuit.get_detector_coordinates()
    assert scheduled.get_final_qubit_coordinates() == circuit.get_final_qubit_coordinates()
    # stim's equality of circuits ignores the tags of REPEAT blocks, so check their text
    assert "REPEAT[block] 2" in str(scheduled) and "CX[gate] 0 1" in str(scheduled)


def test_custom_scheduler() -> None:
    """A custom scheduler needs only to make schedules, which are validated.

    By default, every gate takes time, except for classically controlled gates.  Annotations and
    noise channels take no time.
    """
    circuit = stim.Circuit("H 0 1\nM 0\nREPEAT 2 {\nCX rec[-1] 1\n}")
    scheduler = _EachInItsOwnMoment()
    schedule = scheduler.get_schedule(circuit)
    scheduler.validate(circuit, schedule)
    body_schedule = circuits.Schedule(stim.Circuit("CX rec[-1] 1"), [0])
    assert schedule.get_moments() == [
        list(stim.Circuit("H 0")),
        list(stim.Circuit("H 1")),
        list(stim.Circuit("M 0")),
        body_schedule,
    ]
    scheduled, _ = schedule.to_circuit()
    assert scheduled == stim.Circuit(
        "H 0\nTICK\nH 1\nTICK\nM 0\nTICK\nREPEAT 2 {\nCX rec[-1] 1\nTICK\n}"
    )
    assert scheduler.takes_time(stim.CircuitInstruction("H", [0]))
    for op in ["CX rec[-1] 0", "QUBIT_COORDS(0) 0", "X_ERROR(0.1) 0"]:
        assert not scheduler.takes_time(stim.Circuit(op)[0])


class _EachInItsOwnMoment(circuits.Scheduler):
    """A scheduler that puts each operation in a moment of its own."""

    def get_schedule(self, circuit: circuits.StimCircuitLike) -> circuits.Schedule:
        moments: list[int | circuits.Schedule] = []
        for index, operation in enumerate(circuits.Schedule.get_operations(circuit)):
            if isinstance(operation, stim.CircuitRepeatBlock):
                moments.append(self.get_schedule(operation.body_copy()))
            else:
                moments.append(index)
        return circuits.Schedule(circuit, moments)


@pytest.mark.parametrize(
    ("circuit", "moments", "error"),
    [
        ("H 0\nTICK\nX 0", [1, 0], "change order"),
        ("CX 0 1\nCX 1 2", [0, 0], "two operations that take time"),
        ("M 0\nDETECTOR rec[-1]", [1, 0], "precedes a measurement"),
        ("DETECTOR(0)\nSHIFT_COORDS(1)\nDETECTOR(0)", [0, 1, 0], "coordinates"),
        ("E(0.1) X0\nELSE_CORRELATED_ERROR(0.1) X1", [0, 1], "error chain"),
    ],
)
def test_invalid_schedules(circuit: str, moments: list[int], error: str) -> None:
    """Schedules are rejected if they change the meaning of a circuit."""
    scheduler = circuits.Scheduler.asap()
    with pytest.raises(ValueError, match=error):
        scheduler.validate(stim.Circuit(circuit), circuits.Schedule(stim.Circuit(circuit), moments))


def test_schedule_errors() -> None:
    """A schedule has a nonnegative moment for each operation, and a schedule for each block."""
    circuit = stim.Circuit("H 0 1\nREPEAT 2 {\nH 0\n}")
    body_schedule = circuits.Schedule(stim.Circuit("H 0"), [0])
    with pytest.raises(ValueError, match="needs 3 entries"):
        circuits.Schedule(circuit, [0, body_schedule])
    with pytest.raises(ValueError, match="nonnegative moment"):
        circuits.Schedule(circuit, [0, -1, body_schedule])
    with pytest.raises(ValueError, match="nonnegative moment"):
        circuits.Schedule(circuit, [0, body_schedule, body_schedule])
    with pytest.raises(ValueError, match="schedule of its body"):
        circuits.Schedule(circuit, [0, 0, 0])
    with pytest.raises(ValueError, match="schedule of its body"):
        circuits.Schedule(circuit, [0, 0, circuits.Schedule(stim.Circuit("H 1"), [0])])
    with pytest.raises(ValueError, match="different circuit"):
        circuits.Scheduler.asap().validate(
            stim.Circuit("H 0"), circuits.Schedule(stim.Circuit("H 1"), [0])
        )


def test_repr() -> None:
    """Schedules and schedulers have reprs that reconstruct them."""
    schedulers = [
        circuits.Scheduler.preserve_ticks(single_qubit_gates_take_time=flag)
        for flag in [False, True]
    ]
    for shorten, take_time in itertools.product([False, True], repeat=2):
        schedulers.append(
            circuits.Scheduler.asap(late_resets=shorten, single_qubit_gates_take_time=take_time)
        )
        schedulers.append(
            circuits.Scheduler.alap(
                early_measurements=shorten, single_qubit_gates_take_time=take_time
            )
        )
    for scheduler in schedulers:
        assert eval(repr(scheduler), {"Scheduler": circuits.Scheduler}) == scheduler

    schedule = circuits.Scheduler.asap().get_schedule(stim.Circuit("H 0\nREPEAT 2 {\nH 0\n}"))
    namespace = {"Schedule": circuits.Schedule, "stim": stim}
    assert eval(repr(schedule), namespace) == schedule
    assert hash(eval(repr(schedule), namespace)) == hash(schedule)
    assert schedule != circuits.Scheduler.asap().get_schedule(stim.Circuit("H 1"))
