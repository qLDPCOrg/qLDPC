# SPDX-License-Identifier: Apache-2.0

"""Unit tests for scheduling.py."""

from __future__ import annotations

import copy
import pickle

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
    circuit = scheduler.get_schedule(CIRCUIT).to_circuit()
    assert circuit == stim.Circuit(expected.format(resets="R 0 1 2", reset_1="", reset_2=""))
    scheduler = circuits.ASAPScheduler(single_qubit_gates_take_time=True)
    circuit = scheduler.get_schedule(CIRCUIT).to_circuit()
    assert circuit == stim.Circuit(expected.format(resets="R 0", reset_1="R 1", reset_2="R 2"))

    circuit = stim.Circuit("CX 0 1 0 1\nM 1\nH 0\nDETECTOR rec[-1]\nH 2\nSHIFT_COORDS(1)")
    assert circuits.ASAPScheduler().get_schedule(circuit).get_moments() == [
        list(stim.Circuit("CX 0 1\nH 2")),
        list(stim.Circuit("CX 0 1\nH 0")),
        list(stim.Circuit("M 1\nDETECTOR rec[-1]\nSHIFT_COORDS(1)")),
    ]

    # a classically controlled gate follows the measurement that it reads
    circuit = stim.Circuit("CX 0 2\nCX 0 2\nM 0\nCX rec[-1] 1")
    assert circuits.ASAPScheduler().get_schedule(circuit).to_circuit() == stim.Circuit(
        "CX 0 2\nTICK\nCX 0 2\nTICK\nM 0\nCX rec[-1] 1"
    )


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
    assert scheduler.get_schedule(CIRCUIT).to_circuit() == stim.Circuit(
        expected.format(early_measurement="", measurements="TICK\nM 0 1 2")
    )
    scheduler = circuits.ALAPScheduler(single_qubit_gates_take_time=True)
    assert scheduler.get_schedule(CIRCUIT).to_circuit() == stim.Circuit(
        expected.format(early_measurement="M 0", measurements="M 1\nTICK\nM 2")
    )


def test_tick_scheduler() -> None:
    """Each TICK after the first gate ends a moment, and gates on one qubit may share a moment.

    Consecutive TICKs give an empty moment, and TICKs before the first gate or after the last one
    give none.
    """
    circuit = stim.Circuit("""
        TICK
        H 0
        CX 0 1
        X_ERROR(0.1) 1
        CX 1 2
        TICK
        TICK
        M 0
        CX rec[-1] 1
        II 1 2
        TICK
    """)
    schedule = circuits.TickScheduler().get_schedule(circuit)
    assert schedule.num_moments == 3
    assert schedule.get_moments() == [
        list(stim.Circuit("H 0\nCX 0 1\nX_ERROR(0.1) 1\nCX 1 2")),
        [],
        list(stim.Circuit("M 0\nCX rec[-1] 1\nII 1 2")),
    ]
    assert schedule.to_circuit() == circuit[1:-1]


def test_single_qubit_gates_take_time() -> None:
    """By default, single-qubit gates in TICK layers take time, so qubits do not idle in them.

    Classically controlled Pauli gates take no time, so qubit 0 idles in the second moment.
    """
    circuit = stim.Circuit("H 0\nTICK\nM 1\nCX rec[-1] 0")
    noise_model = circuits.NoiseModel(idle_error={"Z": 0.1}, scheduler=circuits.TickScheduler())
    noisy_circuit = noise_model.noisy_circuit(circuit, system_qubits=[0])
    assert noisy_circuit == circuit + stim.Circuit("Z_ERROR(0.1) 0")
    noise_model = circuits.NoiseModel(
        idle_error={"Z": 0.1},
        scheduler=circuits.TickScheduler(single_qubit_gates_take_time=False),
    )
    two_moments = 2 * 0.1 * 0.9
    assert noise_model.noisy_circuit(circuit, system_qubits=[0]).approx_equals(
        circuit + stim.Circuit(f"Z_ERROR({two_moments}) 0"), atol=1e-12
    )


def test_operations() -> None:
    """The operations of a circuit are its gate applications, annotations, and noise channels.

    A gate application is one target of a one-qubit gate, one pair of targets of a two-qubit gate,
    or one Pauli product of an MPP or SPP gate, and noise channels split likewise.  An observable
    splits into its measurement records and each of its Pauli targets, and one without targets
    stays whole.
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
    moments = circuits.TickScheduler().get_schedule(circuit).get_moments()
    assert moments[0] == [
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
    ]
    assert len(moments) == 2


def test_noise_on_several_qubits() -> None:
    """Only correlated noise ties the schedules of its qubits together.

    In each circuit, qubit 1 could be measured in moment 1, while qubit 0 has gates until moment 3.
    """
    scheduler = circuits.ASAPScheduler(late_resets=False)
    for noise, layout in [
        ("X_ERROR(0.1) 0 1", "X_ERROR(0.1) 1\nTICK\nCX 0 2\nM 1\nTICK\n{}TICK\n{}X_ERROR(0.1) 0"),
        (
            "OBSERVABLE_INCLUDE(0) Z0 Z1",
            "OBSERVABLE_INCLUDE(0) Z1\nTICK\nCX 0 2\nM 1\nTICK\n{}TICK\n{}OBSERVABLE_INCLUDE(0) Z0",
        ),
        ("DEPOLARIZE2(0.1) 0 1", "TICK\nCX 0 2\nTICK\n{}TICK\n{}DEPOLARIZE2(0.1) 0 1\nM 1"),
    ]:
        circuit = stim.Circuit(f"R 0 1 2\nCX 0 2\nCX 0 2\nCX 0 2\n{noise}\nM 1")
        expected = "R 0 1 2\n" + layout.format("CX 0 2\n", "CX 0 2\n")
        assert scheduler.get_schedule(circuit).to_circuit() == stim.Circuit(expected)


def test_shortened_lifetimes() -> None:
    """Resets in ASAP schedules, and measurements in ALAP schedules, move without moving others.

    Qubits 0 and 1 could start in either of the first two of the four moments of the circuit.
    """
    circuit = stim.Circuit("R 0 1\nCX 0 1\nM 0 1\nCX 2 3\nCX 2 3\nCX 2 3\nCX 2 3")
    gates_2_3 = ["CX 2 3\n"] * 4
    asap = "R 0 1\n{}TICK\nCX 0 1\n{}TICK\nM 0 1\n{}TICK\n{}"
    alap = "{}TICK\nR 0 1\n{}TICK\nCX 0 1\n{}TICK\nM 0 1\n{}"
    for scheduler, layout in [(circuits.ASAPScheduler(), asap), (circuits.ALAPScheduler(), alap)]:
        expected = stim.Circuit(layout.format(*gates_2_3))
        assert scheduler.get_schedule(circuit).to_circuit() == expected


def test_correlated_error_chains() -> None:
    """A chain of correlated errors joins the latest moment that one of its errors would join."""
    circuit = stim.Circuit("H 0\nTICK\nH 1\nE(0.1) X0\nELSE_CORRELATED_ERROR(0.1) X1")
    schedule = circuits.TickScheduler().get_schedule(circuit)
    assert schedule.get_moments() == [[circuit[0]], list(circuit[2:])]


def test_layouts() -> None:
    """Layouts sort operations by moment, and keep the text of instructions and REPEAT blocks.

    So a record target may read another measurement than in the circuit.  Layouts keep the tags of
    instructions and REPEAT blocks, and the coordinates of qubits and detectors.  Schedulers accept
    stim-wrapping circuits.
    """
    circuit = stim.Circuit("""
        H 0
        CX 0 1
        M 1
        M 2
        DETECTOR(1) rec[-2]
        REPEAT[block] 2 {
            M 2
            SHIFT_COORDS(1)
            QUBIT_COORDS(2) 0
            DETECTOR(0) rec[-1]
        }
    """)
    schedule = circuits.ASAPScheduler().get_schedule(StimCircuitWrapper(circuit))
    scheduled = schedule.to_circuit()
    assert scheduled == stim.Circuit("""
        H 0
        CX 0 1
        M 2
        TICK
        M 1
        DETECTOR(1) rec[-2]
        TICK
        REPEAT 2 {
            M 2
            SHIFT_COORDS(1)
            QUBIT_COORDS(2) 0
            DETECTOR(0) rec[-1]
            TICK
        }
    """)
    assert scheduled.get_detector_coordinates() == circuit.get_detector_coordinates()
    assert scheduled.get_final_qubit_coordinates() == circuit.get_final_qubit_coordinates()
    # stim's equality of circuits ignores the tags of REPEAT blocks, so check their text
    assert "REPEAT[block] 2" in str(scheduled)


def test_tick_layouts_keep_measurement_order() -> None:
    """A heralded noise channel stays after a measurement before it on another qubit."""
    circuit = stim.Circuit("""
        R 0 1
        TICK
        X_ERROR(0.125) 1
        TICK
        M 1
        HERALDED_ERASE(0.25) 0
        DETECTOR rec[-2]
    """)
    expected = circuit.detector_error_model(approximate_disjoint_errors=True)
    scheduled = circuits.TickScheduler().get_schedule(circuit).to_circuit()
    dem = scheduled.detector_error_model(approximate_disjoint_errors=True)
    assert dem.approx_equals(expected, atol=1e-12)


def test_tick_layouts_keep_noisy_circuits() -> None:
    """The layout of a TickScheduler schedule has the error model of its circuit under gate noise.

    The noise of each gate is inserted right after it, independently of moments.  The detectors of
    the REPEAT block read measurements from before the block and from its previous iteration.
    """
    circuit = stim.Circuit("""
        R 0 1 2
        H 0
        TICK
        CX 0 1 0 2 1 2
        M 2
        DETECTOR rec[-1]
        REPEAT 2 {
            R 2
            TICK
            TICK
            CX 0 2 1 2
            M 2
            DETECTOR rec[-1] rec[-2]
        }
        M 0 1
        DETECTOR rec[-1] rec[-2]
    """)
    expected = _with_gate_noise(circuit).detector_error_model()
    scheduled = circuits.TickScheduler().get_schedule(circuit).to_circuit()
    assert _with_gate_noise(scheduled).detector_error_model().approx_equals(expected, atol=1e-12)


def _with_gate_noise(circuit: stim.Circuit) -> stim.Circuit:
    """Insert depolarizing noise right after each H and CX gate without classical control."""
    noisy_circuit = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            body = _with_gate_noise(op.body_copy())
            noisy_circuit.append(stim.CircuitRepeatBlock(op.repeat_count, body))
            continue
        for group in op.target_groups() if op.name in ("H", "CX") else [op.targets_copy()]:
            noisy_circuit.append(op.name, group, op.gate_args_copy())
            if op.name == "H":
                noisy_circuit.append("DEPOLARIZE1", group, 0.01)
            elif op.name == "CX":
                noisy_circuit.append("DEPOLARIZE2", group, 0.02)
    return noisy_circuit


def test_repeat_blocks_are_moment_boundaries() -> None:
    """The boundaries of REPEAT blocks and of their iterations are boundaries between moments.

    Instructions between REPEAT blocks without gates have no moments, so they take no time.  An
    annotation or noise channel joins the first moment of the REPEAT body that it starts.
    """
    circuit = stim.Circuit("H 2\nREPEAT 3 {\nH 0\nTICK\nH 1\n}\nH 3")
    schedule = circuits.TickScheduler().get_schedule(circuit)
    moments = schedule.get_moments()
    assert moments[0] == [circuit[0]] and moments[2] == [circuit[2]]
    assert isinstance(moments[1], tuple) and moments[1][0] == 3
    assert moments[1][1].get_moments() == [list(stim.Circuit("H 0")), list(stim.Circuit("H 1"))]
    assert schedule.num_moments == 8  # 4 for the circuit with its REPEAT block written out
    assert schedule.to_circuit() == stim.Circuit(
        "H 2\nTICK\nREPEAT 3 {\nH 0\nTICK\nH 1\nTICK\n}\nH 3"
    )

    # each scheduler schedules the body of a REPEAT block by its own rules
    circuit = stim.Circuit("REPEAT 2 {\nH 0\nTICK\nH 1\n}")
    for scheduler, num_moments in [
        (circuits.TickScheduler(), 4),
        (circuits.ASAPScheduler(single_qubit_gates_take_time=True), 2),
        (circuits.ALAPScheduler(single_qubit_gates_take_time=True), 2),
    ]:
        assert scheduler.get_schedule(circuit).num_moments == num_moments

    circuit = stim.Circuit("M 0\nREPEAT 2 {\nTICK\n}\nREPEAT 3 {\nX_ERROR(0.1) 1\n}\nM 2")
    schedule = circuits.ASAPScheduler().get_schedule(circuit)
    assert schedule.num_moments == 2
    gate_free_body = schedule.get_moments()[2]
    assert isinstance(gate_free_body, tuple) and gate_free_body[1].get_moments() == []
    assert schedule.to_circuit() == stim.Circuit(
        "M 0\nTICK\nREPEAT 2 {\n}\nREPEAT 3 {\nX_ERROR(0.1) 1\n}\nM 2"
    )

    for text, num_moments in [
        ("R 0\nTICK\nREPEAT 4 {\nX_ERROR(0.1) 0\nTICK\nH 0\n}", 5),
        ("M 0\nTICK\nDETECTOR rec[-1]\nREPEAT 2 {\nH 0\n}", 3),
    ]:
        assert circuits.TickScheduler().get_schedule(stim.Circuit(text)).num_moments == num_moments


def test_schedules_as_schedulers() -> None:
    """A noise model with a schedule takes the moments of each circuit from that schedule.

    The schedule matches a circuit if each qubit has the same operations in both, in the same order
    and apart from record targets, and REPEAT blocks have the same counts and matching bodies.  Here
    the schedule measures qubit 1 in moment 0, so it idles in the other two moments, after its
    measurement.  Annotations without qubits, such as detectors, do not need to match.
    """
    timed_circuit = stim.Circuit("R 0 1\nM 1\nTICK\nH 0\nTICK\nM 0")
    schedule = circuits.TickScheduler().get_schedule(timed_circuit)
    noise_model = circuits.NoiseModel(idle_error={"Z": 0.1}, scheduler=schedule)
    circuit = stim.Circuit("R 0 1\nTICK\nH 0\nTICK\nM 0 1")
    two_moments = 2 * 0.1 * 0.9
    assert noise_model.noisy_circuit(circuit).approx_equals(
        circuit + stim.Circuit(f"Z_ERROR({two_moments}) 1"), atol=1e-12
    )

    # annotations without qubits do not need to match, also in REPEAT blocks
    timed_circuit = stim.Circuit("H 0\nREPEAT 2 {\nH 1\nTICK\nH 0\nDETECTOR(1)\n}")
    circuit = stim.Circuit(str(timed_circuit).replace("DETECTOR(1)", "DETECTOR(2)"))
    schedule = circuits.TickScheduler().get_schedule(timed_circuit)
    noise_model = circuits.NoiseModel(idle_error={"Z": 0.1}, scheduler=schedule)
    ticks_model = circuits.NoiseModel(idle_error={"Z": 0.1}, scheduler=circuits.TickScheduler())
    assert noise_model.noisy_circuit(circuit) == ticks_model.noisy_circuit(circuit)


def test_matching_ignores_record_targets() -> None:
    """A schedule matches a circuit whose operations differ only in their record targets.

    Here the timed circuit moves ``M 2`` earlier and fixes up the record target of ``CX``.  The
    classically controlled ``CX`` takes no time, so qubit 1 idles in both moments, and qubits 0 and
    2 in the moment after their measurements.
    """
    timed_circuit = stim.Circuit("M 0\nM 2\nTICK\nCX rec[-2] 1")
    schedule = circuits.TickScheduler().get_schedule(timed_circuit)
    noise_model = circuits.NoiseModel(idle_error={"Z": 0.1}, scheduler=schedule)
    circuit = stim.Circuit("M 0\nCX rec[-1] 1\nM 2")
    expected = circuit + stim.Circuit(f"Z_ERROR(0.1) 0\nZ_ERROR({2 * 0.1 * 0.9}) 1\nZ_ERROR(0.1) 2")
    assert noise_model.noisy_circuit(circuit).approx_equals(expected, atol=1e-12)


@pytest.mark.parametrize(
    "circuit",
    [
        "H 0\nTICK\nH 0 1",  # an extra operation, on qubit 1
        "H 0\nTICK\nX 0",  # another operation on qubit 0
        "H 0",  # a missing operation on qubit 0
        "H 0\nREPEAT 2 {\nH 0\n}\nREPEAT 1 {\nH 1\n}",  # other REPEAT blocks
        "H 0\nREPEAT 3 {\nH 0\n}",  # another repeat count
    ],
)
def test_mismatched_schedules(circuit: str) -> None:
    """A noise model with a schedule rejects circuits that the schedule does not match."""
    timed_circuit = stim.Circuit("H 0\nREPEAT 2 {\nH 0\n}" if "REPEAT" in circuit else "H 0\nH 0")
    schedule = circuits.TickScheduler().get_schedule(timed_circuit)
    noise_model = circuits.NoiseModel(idle_error=0.1, scheduler=schedule)
    with pytest.raises(ValueError, match="different circuit"):
        noise_model.noisy_circuit(stim.Circuit(circuit))


def test_schedules_are_immutable_values() -> None:
    """Schedules are made by schedulers, cannot change, and are equal if they schedule alike.

    Changing the circuit that made a schedule does not change it, and schedules can be copied and
    pickled.  Unlike Stim's equality of circuits, the equality of schedules does not ignore the tags
    of REPEAT blocks.
    """
    with pytest.raises(TypeError, match="made by schedulers"):
        circuits.Schedule()

    circuit = stim.Circuit("H 0\nTICK\nH 0")
    schedule = circuits.TickScheduler().get_schedule(circuit)
    circuit.append("H", 1)
    schedule.circuit.append("H", 1)
    assert schedule.circuit == stim.Circuit("H 0\nTICK\nH 0")
    assert repr(schedule) == "<Schedule with 2 moments>"
    assert schedule == circuits.TickScheduler().get_schedule(stim.Circuit("H 0\nTICK\nH 0"))
    assert schedule != circuits.TickScheduler().get_schedule(stim.Circuit("H 0\nH 0"))
    assert schedule != circuits.TickScheduler(single_qubit_gates_take_time=False).get_schedule(
        stim.Circuit("H 0\nTICK\nH 0")
    )
    with pytest.raises(AttributeError):
        schedule.num_moments = 2  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del schedule.num_moments

    tagged = circuits.TickScheduler().get_schedule(stim.Circuit("REPEAT[tag] 2 {\nH 0\n}"))
    untagged = circuits.TickScheduler().get_schedule(stim.Circuit("REPEAT 2 {\nH 0\n}"))
    assert tagged != untagged
    assert copy.deepcopy(tagged) == tagged
    # This deserializes only the in-memory object constructed above, never untrusted external data.
    assert pickle.loads(pickle.dumps(tagged)) == tagged  # noqa: S301


def test_built_in_schedulers() -> None:
    """Built-in schedulers are immutable values with constructor-like reprs.

    A built-in scheduler equals only a scheduler of the same class with the same configuration,
    which is its only public data.  A scheduler's only method is get_schedule, and a schedule's
    public interface is its circuit and its moments.
    """
    schedulers = {
        "ASAPScheduler()": circuits.ASAPScheduler(),
        "ALAPScheduler()": circuits.ALAPScheduler(),
        "TickScheduler()": circuits.TickScheduler(),
        "ASAPScheduler(late_resets=False)": circuits.ASAPScheduler(late_resets=False),
        "ALAPScheduler(early_measurements=False)": circuits.ALAPScheduler(early_measurements=False),
        "TickScheduler(single_qubit_gates_take_time=False)": (
            circuits.TickScheduler(single_qubit_gates_take_time=False)
        ),
    }
    for text, scheduler in schedulers.items():
        assert repr(scheduler) == text
        assert eval(text, vars(circuits)) == scheduler
        assert hash(eval(text, vars(circuits))) == hash(scheduler)
    assert len(set(schedulers.values())) == len(schedulers)

    for scheduler, config in [
        (circuits.ASAPScheduler(), {"late_resets": True, "single_qubit_gates_take_time": False}),
        (
            circuits.ALAPScheduler(single_qubit_gates_take_time=True),
            {"early_measurements": True, "single_qubit_gates_take_time": True},
        ),
        (circuits.TickScheduler(), {"single_qubit_gates_take_time": True}),
    ]:
        names = {name for name in dir(scheduler) if not name.startswith("_")}
        names -= set(dir(circuits.Scheduler))
        assert {name: getattr(scheduler, name) for name in names} == config
        for name in names:
            with pytest.raises(AttributeError):
                setattr(scheduler, name, not config[name])

    assert {name for name in dir(circuits.Scheduler) if not name.startswith("_")} == {
        "get_schedule"
    }
    assert {name for name in dir(circuits.Schedule) if not name.startswith("_")} == {
        "circuit",
        "get_moments",
        "num_moments",
        "to_circuit",
    }
