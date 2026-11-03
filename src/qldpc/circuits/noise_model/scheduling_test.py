# SPDX-License-Identifier: Apache-2.0

"""Unit tests for scheduling.py."""

from __future__ import annotations

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
    """Gates are as early as possible, and then by default resets as late as other gates allow.

    An annotation joins the moment of the operation before it on its qubits, or, without qubits, of
    the last gate before it, or a later moment to follow a measurement that it reads or the
    coordinate annotation before it.
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
        TICK
        H 2
        M 1
        TICK
        M 2
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    scheduler = circuits.ASAPScheduler(late_resets=False, single_qubit_gates_take_time=True)
    circuit, measurements = scheduler.get_schedule(CIRCUIT).to_circuit()
    assert circuit == stim.Circuit(expected.format(resets="R 0 1 2", reset_1="", reset_2=""))
    assert measurements == (0, 1, 2)
    scheduler = circuits.ASAPScheduler(single_qubit_gates_take_time=True)
    circuit, _ = scheduler.get_schedule(CIRCUIT).to_circuit()
    assert circuit == stim.Circuit(expected.format(resets="R 0", reset_1="R 1", reset_2="R 2"))

    circuit = stim.Circuit("CX 0 1 0 1\nM 1\nH 0\nDETECTOR rec[-1]\nH 2\nSHIFT_COORDS(1)")
    ops = circuits.Schedule.get_operations(circuit)
    assert circuits.ASAPScheduler().get_schedule(circuit).get_moments() == [
        [ops[0], ops[5]],
        [ops[1], ops[3]],
        [ops[2], ops[4], ops[6]],
    ]

    # a classically controlled gate follows the measurement that it reads
    circuit = stim.Circuit("CX 0 2\nCX 0 2\nM 0\nCX rec[-1] 1")
    schedule = circuits.ASAPScheduler().get_schedule(circuit)
    assert schedule == circuits.Schedule(circuit, [0, 1, 2, 2])


def test_alap() -> None:
    """Gates are as late as possible, and then by default measurements as early as others allow.

    Annotations join the moment of the operation before them, as in ASAP schedules.
    """
    expected = """
        QUBIT_COORDS(0, 0) 0
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
        OBSERVABLE_INCLUDE(0) Z2
        TICK
        H 2
        {measurements}
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    scheduler = circuits.ALAPScheduler(early_measurements=False, single_qubit_gates_take_time=True)
    circuit, _ = scheduler.get_schedule(CIRCUIT).to_circuit()
    assert circuit == stim.Circuit(
        expected.format(early_measurement="", measurements="TICK\nM 0 1 2")
    )
    scheduler = circuits.ALAPScheduler(single_qubit_gates_take_time=True)
    circuit, _ = scheduler.get_schedule(CIRCUIT).to_circuit()
    assert circuit == stim.Circuit(
        expected.format(early_measurement="M 0", measurements="M 1\nTICK\nM 2")
    )


def test_tick_preserving_scheduler() -> None:
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
    schedule = circuits.TickPreservingScheduler().get_schedule(circuit)
    assert schedule.get_moments() == [
        list(stim.Circuit("H 0\nI 0\nCX 0 1\nH 1\nX_ERROR(0.1) 1")),
        list(stim.Circuit("CX 1 2")),
        list(stim.Circuit("M 0\nCX rec[-1] 1\nII 1 2")),
    ]
    schedule = circuits.TickPreservingScheduler(single_qubit_gates_take_time=True).get_schedule(
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


def test_operations() -> None:
    """The operations of a circuit are its gate applications, annotations, and noise channels.

    A gate application is one target of a one-qubit gate, one pair of targets of a two-qubit gate,
    or one Pauli product of an MPP or SPP gate, and noise channels split likewise.  An observable
    splits into its measurement records and each of its Pauli targets, and one without targets
    stays whole.  REPEAT blocks are listed whole.
    """
    circuit = stim.Circuit("""
        MPP X0*Z1 Z2
        X_ERROR(0.1) 0 1
        DEPOLARIZE2(0.1) 0 1 2 3
        OBSERVABLE_INCLUDE(0) rec[-1] Z0 X1 rec[-2]
        OBSERVABLE_INCLUDE(1)
        TICK
        REPEAT 2 {
            H 0
        }
    """)
    assert circuits.Schedule.get_operations(circuit) == [
        stim.CircuitInstruction(
            "MPP", [stim.target_x(0), stim.target_combiner(), stim.target_z(1)]
        ),
        stim.CircuitInstruction("MPP", [stim.target_z(2)]),
        stim.CircuitInstruction("X_ERROR", [0], [0.1]),
        stim.CircuitInstruction("X_ERROR", [1], [0.1]),
        stim.CircuitInstruction("DEPOLARIZE2", [0, 1], [0.1]),
        stim.CircuitInstruction("DEPOLARIZE2", [2, 3], [0.1]),
        stim.CircuitInstruction(
            "OBSERVABLE_INCLUDE", [stim.target_rec(-1), stim.target_rec(-2)], [0]
        ),
        stim.CircuitInstruction("OBSERVABLE_INCLUDE", [stim.target_z(0)], [0]),
        stim.CircuitInstruction("OBSERVABLE_INCLUDE", [stim.target_x(1)], [0]),
        circuit[4],
        circuit[6],
    ]


def test_noise_on_several_qubits() -> None:
    """Only correlated noise ties the schedules of its qubits together.

    In each circuit, qubit 1 could be measured in moment 1, while qubit 0 has gates until moment 3.
    """
    scheduler = circuits.ASAPScheduler(late_resets=False)
    for noise, moments in [
        ("X_ERROR(0.1) 0 1", [0, 0, 0, 1, 2, 3, 3, 0, 1]),
        ("OBSERVABLE_INCLUDE(0) Z0 Z1", [0, 0, 0, 1, 2, 3, 3, 0, 1]),
        ("DEPOLARIZE2(0.1) 0 1", [0, 0, 0, 1, 2, 3, 3, 3]),
    ]:
        circuit = stim.Circuit(f"R 0 1 2\nCX 0 2\nCX 0 2\nCX 0 2\n{noise}\nM 1")
        assert scheduler.get_schedule(circuit) == circuits.Schedule(circuit, moments)


def test_shortened_lifetimes() -> None:
    """Resets in ASAP schedules, and measurements in ALAP schedules, move without moving others.

    Qubits 0 and 1 could start in either of the first two of the four moments of the circuit.
    """
    circuit = stim.Circuit("R 0 1\nCX 0 1\nM 0 1\nCX 2 3\nCX 2 3\nCX 2 3\nCX 2 3")
    for scheduler, moments in [
        (circuits.ASAPScheduler(), [0, 0, 1, 2, 2, 0, 1, 2, 3]),
        (circuits.ALAPScheduler(), [1, 1, 2, 3, 3, 0, 1, 2, 3]),
    ]:
        assert scheduler.get_schedule(circuit) == circuits.Schedule(circuit, moments)


def test_correlated_error_chains() -> None:
    """A chain of correlated errors joins the latest moment that one of its errors would join."""
    circuit = stim.Circuit("H 0\nTICK\nH 1\nE(0.1) X0\nELSE_CORRELATED_ERROR(0.1) X1")
    scheduler = circuits.TickPreservingScheduler(single_qubit_gates_take_time=True)
    assert scheduler.get_schedule(circuit).get_moments() == [[circuit[0]], list(circuit[2:])]


def test_measurement_order() -> None:
    """Measurements may change order, while instructions keep reading the same measurements.

    The first iteration of a REPEAT block is written out if it reads a measurement before the block
    (here, in a nested block) at another offset than later iterations read the measurements of
    earlier iterations.
    """
    circuit = stim.Circuit("""
        H 0
        CX 0 1
        M 1
        M 2
        REPEAT 3 {
            M 2
            REPEAT 1 {
                H 3
                DETECTOR rec[-1] rec[-2]
            }
        }
    """)
    scheduled, measurements = circuits.ASAPScheduler().get_schedule(circuit).to_circuit()
    assert scheduled == stim.Circuit("""
        H 0
        CX 0 1
        M 2
        TICK
        M 1
        TICK
        M 2
        TICK
        REPEAT 1 {
            H 3
            DETECTOR rec[-1] rec[-3]
            TICK
        }
        REPEAT 2 {
            M 2
            TICK
            REPEAT 1 {
                H 3
                DETECTOR rec[-1] rec[-2]
                TICK
            }
        }
    """)
    assert measurements == (1, 0, 2, 3, 4)

    # a written-out iteration of a tagged REPEAT block keeps the tag
    tagged = stim.Circuit(str(circuit).replace("REPEAT 3", "REPEAT[block] 3"))
    scheduled, _ = circuits.ASAPScheduler().get_schedule(tagged).to_circuit()
    assert "REPEAT[block] 1" in str(scheduled) and "REPEAT[block] 2" in str(scheduled)

    # iterations written out with the same layout share a REPEAT block
    circuit = stim.Circuit("CX 0 4\nM 0 1 2\nREPEAT 5 {\nM 3\nDETECTOR rec[-3]\n}")
    scheduled, measurements = circuits.ASAPScheduler().get_schedule(circuit).to_circuit()
    assert scheduled == stim.Circuit("""
        CX 0 4
        M 1 2
        TICK
        M 0
        TICK
        REPEAT 2 {
            M 3
            DETECTOR rec[-4]
            TICK
        }
        REPEAT 3 {
            M 3
            DETECTOR rec[-3]
            TICK
        }
    """)
    assert measurements == (1, 2, 0, 3, 4, 5, 6, 7)


def test_repeat_blocks_are_moment_boundaries() -> None:
    """The boundaries of REPEAT blocks and of their iterations are boundaries between moments.

    Instructions between REPEAT blocks without gates have no moments, so they take no time.
    """
    circuit = stim.Circuit("H 2\nREPEAT 3 {\nH 0\nTICK\nH 1\n}\nH 3")
    scheduler: circuits.Scheduler = circuits.TickPreservingScheduler(
        single_qubit_gates_take_time=True
    )
    schedule = scheduler.get_schedule(circuit)
    scheduler.validate(circuit, schedule)
    body_schedule = circuits.Schedule(stim.Circuit("H 0\nTICK\nH 1"), [0, 1])
    assert schedule.get_moments() == [[circuit[0]], body_schedule, [circuit[2]]]
    assert schedule.num_moments == 8  # 4 for the circuit with its REPEAT block written out
    assert schedule.to_circuit()[0] == stim.Circuit(
        "H 2\nTICK\nREPEAT 3 {\nH 0\nTICK\nH 1\nTICK\n}\nH 3"
    )

    # ASAP and ALAP schedulers schedule the body of a REPEAT block by their own rules
    circuit = stim.Circuit("REPEAT 2 {\nH 0\nTICK\nH 1\n}")
    for scheduler in [
        circuits.ASAPScheduler(single_qubit_gates_take_time=True),
        circuits.ALAPScheduler(single_qubit_gates_take_time=True),
    ]:
        assert scheduler.get_schedule(circuit).num_moments == 2

    circuit = stim.Circuit("M 0\nREPEAT 3 {\nM 1\n}\nM 2")
    schedule = circuits.ASAPScheduler().get_schedule(circuit)
    assert schedule.num_moments == 5
    assert schedule.to_circuit() == (
        stim.Circuit("M 0\nTICK\nREPEAT 3 {\nM 1\nTICK\n}\nM 2"),
        (0, 1, 2, 3, 4),
    )

    circuit = stim.Circuit("M 0\nREPEAT 2 {\nTICK\n}\nREPEAT 3 {\nX_ERROR(0.1) 1\n}\nM 2")
    schedule = circuits.ASAPScheduler().get_schedule(circuit)
    assert schedule.num_moments == 2
    gate_free_body = schedule.get_moments()[2]
    assert isinstance(gate_free_body, circuits.Schedule) and gate_free_body.get_moments() == []
    assert schedule.to_circuit()[0] == stim.Circuit(
        "M 0\nTICK\nREPEAT 2 {\n}\nREPEAT 3 {\nX_ERROR(0.1) 1\n}\nM 2"
    )


def test_tick_preserving_scheduler_with_repeat_blocks() -> None:
    """A TICK ends a moment with a gate, also inside a REPEAT block.

    Annotations and noise channels join the moment of the operation before them, or the first
    moment of the REPEAT body that they start.
    """
    scheduler = circuits.TickPreservingScheduler(single_qubit_gates_take_time=True)
    for text, num_moments in [
        ("R 0\nTICK\nREPEAT 4 {\nX_ERROR(0.1) 0\nTICK\nH 0\n}", 5),
        ("M 0\nTICK\nDETECTOR rec[-1]\nREPEAT 2 {\nH 0\n}", 3),
        ("M 0\nTICK\nOBSERVABLE_INCLUDE(0) Z1", 1),
    ]:
        circuit = stim.Circuit(text)
        schedule = scheduler.get_schedule(circuit)
        scheduler.validate(circuit, schedule)
        assert schedule.num_moments == num_moments


def test_schedules_keep_noisy_circuits() -> None:
    """The layout of each built-in schedule has the error model of the circuit under gate noise.

    The noise of each gate is inserted right after it, independently of moments.  The detectors of
    the REPEAT block read measurements from before the block and from its previous iteration.
    """
    circuit = stim.Circuit("""
        R 0 1 2
        H 0
        CX 0 1 0 2 1 2
        M 2
        DETECTOR rec[-1]
        REPEAT 2 {
            R 2
            CX 0 2 1 2
            M 2
            DETECTOR rec[-1] rec[-2]
        }
        M 0 1
        DETECTOR rec[-1] rec[-2]
    """)
    expected = _with_gate_noise(circuit).detector_error_model()
    for scheduler in [
        circuits.TickPreservingScheduler(),
        circuits.ASAPScheduler(),
        circuits.ALAPScheduler(),
    ]:
        scheduled, _ = scheduler.get_schedule(circuit).to_circuit()
        assert (
            _with_gate_noise(scheduled).detector_error_model().approx_equals(expected, atol=1e-12)
        )


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
    schedule = circuits.ASAPScheduler().get_schedule(StimCircuitWrapper(circuit))
    scheduled, _ = schedule.to_circuit()
    assert scheduled.get_detector_coordinates() == circuit.get_detector_coordinates()
    assert scheduled.get_final_qubit_coordinates() == circuit.get_final_qubit_coordinates()
    # stim's equality of circuits ignores the tags of REPEAT blocks, so check their text
    assert "REPEAT[block] 2" in str(scheduled) and "CX[gate] 0 1" in str(scheduled)


def test_custom_scheduler() -> None:
    """A custom scheduler places the operations of circuits without REPEAT blocks.

    Its schedules are validated.  By default, every gate takes time, except for classically
    controlled gates.  Annotations and noise channels take no time.
    """
    circuit = stim.Circuit("H 0 1\nM 0\nREPEAT 2 {\nCX rec[-1] 1\n}")
    scheduler = _EachInItsOwnMoment()
    schedule = scheduler.get_schedule(circuit)
    scheduler.validate(circuit, schedule)
    # the iterations of the REPEAT block follow each other in moments 3 and 4
    assert schedule.num_moments == 5
    assert schedule.get_moments() == [
        list(stim.Circuit("H 0")),
        list(stim.Circuit("H 1")),
        list(stim.Circuit("M 0")),
        circuits.Schedule(stim.Circuit("CX rec[-1] 1"), [0]),
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

    def get_operation_moments(self, circuit: stim.Circuit) -> list[int]:
        assert not any(isinstance(instruction, stim.CircuitRepeatBlock) for instruction in circuit)
        return list(range(len(circuits.Schedule.get_operations(circuit))))


def test_moments_without_operations() -> None:
    """A moment without operations in a custom schedule counts, and is laid out as an extra TICK."""
    circuit = stim.Circuit("H 0 1")
    schedule = circuits.Schedule(circuit, [0, 2])
    circuits.ASAPScheduler().validate(circuit, schedule)
    assert schedule.num_moments == 3
    assert schedule.get_moments() == [list(stim.Circuit("H 0")), [], list(stim.Circuit("H 1"))]
    assert schedule.to_circuit() == (stim.Circuit("H 0\nTICK\nTICK\nH 1"), ())


@pytest.mark.parametrize(
    ("circuit", "moments", "error"),
    [
        ("H 0\nTICK\nX 0", [1, 0], "change order"),
        ("CX 0 1\nCX 1 2", [0, 0], "two operations that take time"),
        ("M 0\nDETECTOR rec[-1]", [1, 0], "precedes a measurement"),
        ("H 0\nDETECTOR(0)\nSHIFT_COORDS(1)\nDETECTOR(0)", [0, 0, 1, 0], "coordinates"),
        ("H 0\nE(0.1) X0\nELSE_CORRELATED_ERROR(0.1) X1", [0, 0, 1], "error chain"),
    ],
)
def test_invalid_schedules(circuit: str, moments: list[int], error: str) -> None:
    """Schedules are rejected if they change the meaning of a circuit."""
    scheduler = circuits.ASAPScheduler()
    with pytest.raises(ValueError, match=error):
        scheduler.validate(stim.Circuit(circuit), circuits.Schedule(stim.Circuit(circuit), moments))


def test_schedule_errors() -> None:
    """A schedule has a nonnegative integer moment for each operation, and schedules each body."""
    circuit = stim.Circuit("H 0 1\nREPEAT 2 {\nH 0\n}")
    body_schedule = circuits.Schedule(stim.Circuit("H 0"), [0])
    assert circuits.Schedule(circuit, [0, 1, body_schedule]).num_moments == 4
    with pytest.raises(ValueError, match="needs 3 entries"):
        circuits.Schedule(circuit, [0, body_schedule])
    for moment in [-1, True, body_schedule]:
        with pytest.raises(ValueError, match="nonnegative integer"):
            circuits.Schedule(circuit, [0, moment, body_schedule])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="schedule of its body"):
        circuits.Schedule(circuit, [0, 0, 0])
    with pytest.raises(ValueError, match="schedule of its body"):
        circuits.Schedule(circuit, [0, 0, circuits.Schedule(stim.Circuit("H 1"), [0])])
    with pytest.raises(ValueError, match="different circuit"):
        circuits.ASAPScheduler().validate(
            stim.Circuit("H 0"), circuits.Schedule(stim.Circuit("H 1"), [0])
        )


def test_schedules_are_immutable_values() -> None:
    """Schedules cannot change, and are equal if they schedule the same circuit alike.

    Changing the circuit or moments that made a schedule does not change it.  Unlike Stim's
    equality of circuits, the equality of schedules does not ignore the tags of REPEAT blocks.
    """
    circuit, moments = stim.Circuit("H 0"), [0]
    schedule = circuits.Schedule(circuit, moments)
    circuit.append("H", 1)
    moments.append(0)
    assert schedule == circuits.Schedule(stim.Circuit("H 0"), [0])
    assert schedule != circuits.Schedule(stim.Circuit("H 0"), [1])
    with pytest.raises(AttributeError):
        schedule.num_moments = 2  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del schedule.num_moments

    body = circuits.Schedule(stim.Circuit("H 0"), [0])
    tagged = circuits.Schedule(stim.Circuit("REPEAT[tag] 2 {\nH 0\n}"), [body])
    untagged = circuits.Schedule(stim.Circuit("REPEAT 2 {\nH 0\n}"), [body])
    assert tagged != untagged


def test_built_in_schedulers() -> None:
    """Built-in schedulers are immutable values with constructor-like reprs.

    A built-in scheduler equals only a scheduler of the same class with the same configuration,
    which is its only public data.
    """
    schedulers = {
        "ASAPScheduler()": circuits.ASAPScheduler(),
        "ALAPScheduler()": circuits.ALAPScheduler(),
        "ASAPScheduler(late_resets=False)": circuits.ASAPScheduler(late_resets=False),
        "ALAPScheduler(early_measurements=False)": circuits.ALAPScheduler(early_measurements=False),
        "TickPreservingScheduler(single_qubit_gates_take_time=True)": (
            circuits.TickPreservingScheduler(single_qubit_gates_take_time=True)
        ),
    }
    for text, scheduler in schedulers.items():
        assert repr(scheduler) == text
        assert eval(text, vars(circuits)) == scheduler
        assert hash(eval(text, vars(circuits))) == hash(scheduler)
    assert len(set(schedulers.values())) == len(schedulers)

    class SlowHadamards(circuits.ASAPScheduler):
        def takes_time(self, op: stim.CircuitInstruction) -> bool:
            return op.name == "H" or super().takes_time(op)

    assert SlowHadamards().get_schedule(stim.Circuit("H 0 0")).num_moments == 2
    assert SlowHadamards() == SlowHadamards()
    assert SlowHadamards() != circuits.ASAPScheduler()

    for scheduler, config in [
        (circuits.ASAPScheduler(), {"late_resets": True, "single_qubit_gates_take_time": False}),
        (
            circuits.ALAPScheduler(single_qubit_gates_take_time=True),
            {"early_measurements": True, "single_qubit_gates_take_time": True},
        ),
        (circuits.TickPreservingScheduler(), {"single_qubit_gates_take_time": False}),
    ]:
        names = {name for name in dir(scheduler) if not name.startswith("_")}
        names -= set(dir(circuits.Scheduler))
        assert {name: getattr(scheduler, name) for name in names} == config
        for name in names:
            with pytest.raises(AttributeError):
                setattr(scheduler, name, not config[name])
