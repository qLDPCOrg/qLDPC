# SPDX-License-Identifier: Apache-2.0

"""Unit tests for encoding.py."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import circuits, codes, math
from qldpc._binary import get_independent_rows
from qldpc.circuits import encoding
from qldpc.objects import Pauli

# Stim's one- and two-qubit Clifford gates, resets, and measurements, other than heralded noise
_GATES = sorted(
    name
    for name, data in stim.gate_data().items()
    if (data.is_single_qubit_gate or data.is_two_qubit_gate)
    and (data.is_unitary or data.is_reset or data.produces_measurements)
    and 0 in data.num_parens_arguments_range
)


def test_encoding_circuit(pytestconfig: pytest.Config) -> None:
    """Prepare logical Pauli states of qubit codes."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.SHPCode(codes.ClassicalCode.random(4, 2, seed=np.random.randint(2**31)))

    for only_zero in [True, False]:
        simulator = stim.TableauSimulator()

        if not only_zero:
            # choose random logical and gauge Paulis that should be +1 after encoding
            bases = np.random.choice(
                [Pauli.Z, Pauli.X, Pauli.Y],  # type:ignore[arg-type]  # NumPy expects array-likes
                size=code.dimension + code.gauge_dimension,
            )
            basis_prep = stim.Circuit()
            basis_prep.append("H", np.flatnonzero(bases != Pauli.Z))
            basis_prep.append("S", np.flatnonzero(bases == Pauli.Y))
            simulator.do(basis_prep)

        encoder = circuits.get_encoding_circuit(code, only_zero=only_zero)
        simulator.do(encoder)

        # all stabilizers have expectation value +1
        for op in code.get_stabilizer_ops():
            string = math.op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 1

        if only_zero:
            # logical Z operators have expectation value +1
            for op in code.get_logical_ops(Pauli.Z, symplectic=True):
                string = math.op_to_string(op)
                assert simulator.peek_observable_expectation(string) == 1

        else:
            # examine logical operators that should have expectation value +1
            for qq in range(code.dimension):
                basis = bases[qq]
                op_x = code.get_logical_ops(Pauli.X, symplectic=True)[qq]
                op_z = code.get_logical_ops(Pauli.Z, symplectic=True)[qq]
                string = _get_logical_pauli_string(basis, op_x, op_z)
                assert simulator.peek_observable_expectation(string) == 1

            # examine gauge operators that should have expectation value +1
            for qq in range(code.gauge_dimension):
                basis = bases[code.dimension + qq]
                op_x = code.get_gauge_ops(Pauli.X, symplectic=True)[qq]
                op_z = code.get_gauge_ops(Pauli.Z, symplectic=True)[qq]
                string = _get_logical_pauli_string(basis, op_x, op_z)
                assert simulator.peek_observable_expectation(string) == 1


def _get_logical_pauli_string(  # pragma: no cover
    basis: Pauli, op_x: npt.NDArray[np.int_], op_z: npt.NDArray[np.int_]
) -> stim.PauliString:
    """Build a logical operator Pauli string from symplectic logical Pauli X and Z vectors."""
    if basis is Pauli.X:
        return math.op_to_string(op_x)
    if basis is Pauli.Z:
        return math.op_to_string(op_z)
    string_x = math.op_to_string(op_x)
    string_z = math.op_to_string(op_z)
    return 1j * string_x * string_z


def test_encoding_overcomplete_stabilizers() -> None:
    """Build an encoding circuit for a code with an overcomplete stabilizer matrix."""
    code = codes.ToricCode(4)  # has more stabilizer generators than len(code) - dimension
    assert len(code.get_stabilizer_ops()) != len(code) - code.dimension - code.gauge_dimension

    simulator = stim.TableauSimulator()
    simulator.do(circuits.get_encoding_circuit(code))

    # all stabilizers have expectation value +1
    for op in code.get_stabilizer_ops():
        assert simulator.peek_observable_expectation(math.op_to_string(op)) == 1


def test_logical_tableau() -> None:
    """Reconstruct a logical tableau."""
    code = codes.FiveQubitCode()
    encoder, decoder = circuits.get_encoder_and_decoder(code, deformation=stim.Circuit())

    logical_circuit = stim.Circuit("H 0")
    extended_logical_circuit = logical_circuit + stim.Circuit(f"I {len(code) - 1}")
    physical_tableau = decoder.then(extended_logical_circuit.to_tableau()).then(encoder)
    physical_circuit = physical_tableau.to_circuit()

    reconstructed_logical_tableau = circuits.get_logical_tableau(code, physical_circuit)
    assert logical_circuit.to_tableau() == reconstructed_logical_tableau

    # logical Pauli operations (which flip logical operator signs) are valid logical operations
    logical_x = math.op_to_string(code.get_logical_ops(Pauli.X, symplectic=True)[0])
    reconstructed_logical_tableau = circuits.get_logical_tableau(code, logical_x.to_tableau())
    assert stim.Circuit("X 0").to_tableau() == reconstructed_logical_tableau

    # a physical circuit that does not preserve the code space is not a logical operation
    with pytest.raises(ValueError, match="does not implement a logical operation"):
        circuits.get_logical_tableau(code, stim.Circuit("X 0"))


def test_state_stabilizers(pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Identify the stabilizers of state prepared by a circuit."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.SHPCode(codes.ClassicalCode.random(4, 2, seed=np.random.randint(2**31)))
    encoder = circuits.get_encoding_circuit(code)

    # prepare a random logical stabilizer state
    circuit = stim.Tableau.random(num_qubits=code.dimension).to_circuit() + encoder

    # the circuit can include detectors and observables
    circuit.append("X", [len(code)])
    circuit.append("M", [len(code)])
    circuit.append("DETECTOR", [stim.target_rec(-1)])
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    simulator = stim.TableauSimulator()
    simulator.do(circuit)

    # all stabilizers of the code have expectation value +1
    for op in code.get_stabilizer_ops():
        string = math.op_to_string(op)
        assert simulator.peek_observable_expectation(string) == 1

    # all stabilizers of the state have expectation value +1
    state_stabs = circuits.get_state_stabilizers(circuit, len(code))
    assert len(state_stabs) == len(code)
    for stab in state_stabs:
        assert simulator.peek_observable_expectation(stab) == 1

    # all logical stabilizers of the state have expectation value +1
    logical_stabs = circuits.get_logical_state_stabilizers(code, circuit)
    assert len(logical_stabs) == code.dimension
    for stab in logical_stabs:
        assert simulator.peek_observable_expectation(stab) == 1

    # the logical stabilizers are "pure", and have no stabilizer content
    for stab in logical_stabs:
        decoded_stab = stab.before(encoder)
        xs, zs = decoded_stab.to_numpy()
        assert not np.any(xs[code.dimension :])
        assert not np.any(zs[code.dimension :])

    assert circuits.get_state_stabilizers(stim.Circuit("X 1"), [1]) == [stim.PauliString("-Z")]

    simulator_class = stim.TableauSimulator

    class ZeroExpectationSimulator:
        """A simulator that reports expectation 0 for every Pauli string."""

        def __init__(self) -> None:
            self.simulator = simulator_class()

        def __getattr__(self, name: str) -> object:
            return getattr(self.simulator, name)

        def peek_observable_expectation(self, _pauli: stim.PauliString) -> int:
            return 0

    monkeypatch.setattr(encoding.stim, "TableauSimulator", ZeroExpectationSimulator)
    with pytest.raises(ValueError, match="not an eigenstate"):
        circuits.get_state_stabilizers(stim.Circuit("H 0"), [0])


@pytest.mark.parametrize("qubits", [[0, 1, 0], [-1]])
def test_state_stabilizers_of_invalid_qubits(qubits: list[int]) -> None:
    """Requested qubits must be nonnegative and distinct.

    Position i of a returned string refers to qubits[i], so with qubits [0, 1, 0] after
    ``H 0; CX 0 1``, a returned XXX would stand for X0 X1 X0 = X1, which is not a stabilizer.
    """
    with pytest.raises(ValueError, match="nonnegative and distinct"):
        circuits.get_state_stabilizers(stim.Circuit("H 0\nCX 0 1"), qubits)


def test_state_stabilizer_failure_regressions() -> None:
    """Exercise row-space, detector-parity, noisy-sign, and zero-logical edge cases."""
    ghz = circuits.get_state_stabilizers(stim.Circuit("H 0\nCX 0 1"), [0, 1])
    assert stim.PauliString("+XX") in ghz
    assert stim.PauliString("+ZZ") in ghz

    repeated_detector = stim.Circuit("H 0\nM 0\nDETECTOR rec[-1] rec[-1]")
    assert circuits.get_state_stabilizers(repeated_detector, [0]) == []

    noisy_signs = [
        circuits.get_state_stabilizers(stim.Circuit("X_ERROR(0.5) 0"), [0]) for _ in range(20)
    ]
    assert all(signs == noisy_signs[0] for signs in noisy_signs)

    zero_logical = codes.TrivialCode(1, num_stabs_x=1)
    assert circuits.get_logical_state_stabilizers(zero_logical, stim.Circuit("R 0")) == []

    code = codes.SteaneCode()
    data = tuple(range(3, 3 + len(code)))
    prep = circuits.with_remapped_qubits(circuits.get_encoding_circuit(code, only_zero=True), data)
    qids = circuits.QubitIDs(data, tuple(range(20, 20 + code.num_checks)))
    assert len(circuits.get_logical_state_stabilizers(code, prep, qids)) == code.dimension


def test_state_stabilizer_signs_must_be_determined() -> None:
    """Raise when a stabilizer on the requested qubits needs post-selection on random detectors."""
    # Z1 is a stabilizer only after post-selecting on the random outcome of M 0.  With a third
    # qubit, Z1 Z2 is a stabilizer with a fixed sign, but Z1 and Z2 need that post-selection.
    measured_bell = stim.Circuit("H 0\nCX 0 1\nM 0\nDETECTOR rec[-1]")
    measured_ghz = stim.Circuit("H 0\nCX 0 1\nCX 0 2\nM 0\nDETECTOR rec[-1]")
    with pytest.raises(ValueError, match=r"stabilizer Z1 .*detectors \[0\]"):
        circuits.get_state_stabilizers(measured_bell, [1])
    with pytest.raises(ValueError, match=r"stabilizer Z[12] .*detectors \[0\]"):
        circuits.get_state_stabilizers(measured_ghz, [1, 2])

    # Here a deterministic detector precedes the random one.  The simulation leaves the
    # deterministic one out of its matrix, but the error still names the random one by its index in
    # the circuit.
    circuit = stim.Circuit("M 0\nDETECTOR rec[-1]\nH 1\nCX 1 2\nM 1\nDETECTOR rec[-1]")
    with pytest.raises(ValueError, match=r"stabilizer Z2 .*detectors \[1\]"):
        circuits.get_state_stabilizers(circuit, [2])

    # the error names a logical stabilizer by its logical qubit, here on physical qubit 2
    qubit_ids = circuits.QubitIDs(data=(2,))
    with pytest.raises(ValueError, match=r"logical stabilizer Z0 .*detectors \[0\]"):
        circuits.get_logical_state_stabilizers(
            codes.TrivialCode(1), stim.Circuit("H 0\nCX 0 2\nM 0\nDETECTOR rec[-1]"), qubit_ids
        )

    # a random detector away from the requested qubits, and deterministic detectors, do not raise
    assert circuits.get_state_stabilizers(measured_bell, [2]) == [stim.PauliString("+Z")]
    circuit = stim.Circuit("X 0\nCX 0 1\nM 1\nDETECTOR rec[-1]\nH 2\nM 2\nDETECTOR rec[-1]")
    stabilizers = circuits.get_state_stabilizers(circuit, [0, 1])
    assert sorted(map(str, stabilizers)) == ["-Z_", "-_Z"]
    circuit = stim.Circuit("H 0\nCX 0 1\nCX 0 2\nM 0 1\nDETECTOR rec[-2] rec[-1]")
    assert circuits.get_state_stabilizers(circuit, [2]) == []


def test_state_stabilizers_match_flows() -> None:
    """Simulation finds the fixed parities that flows find, on random circuits.

    Fixed parities are as the comment in get_state_stabilizers defines them.  The reference,
    _get_fixed_parities_by_flows, finds them from stim.Circuit.flow_generators instead.  Fixed
    parities carry no signs, which test_state_stabilizer_signs tests.
    """
    rng = np.random.default_rng(0)
    cases = [_get_random_state_prep_circuit(rng) for _ in range(400)]

    # the random circuits include every feature that the generator draws from
    noisy_names = {instruction.name for circuit, _ in cases for instruction in circuit.flattened()}
    assert {"HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"} <= noisy_names
    instructions = [
        instruction for circuit, _ in cases for instruction in circuit.flattened().without_noise()
    ]
    names = {instruction.name for instruction in instructions}
    assert names >= {*_GATES, "MPP", "SPP", "SPP_DAG", "MPAD", "DETECTOR", "OBSERVABLE_INCLUDE"}
    targets = [
        (instruction.name, target)
        for instruction in instructions
        for target in instruction.targets_copy()
    ]
    record_users = {name for name, target in targets if target.is_measurement_record_target}
    assert record_users >= {"CX", "CY", "CZ", "XCZ", "YCZ", "OBSERVABLE_INCLUDE"}
    assert any(target.is_sweep_bit_target for _, target in targets)
    assert any(
        target.is_inverted_result_target
        for name, target in targets
        if name not in ("MPP", "SPP", "SPP_DAG")
    )
    assert any(
        instruction.name in ("MR", "MRX", "MRY") and _repeats_target(instruction)
        for instruction in instructions
    )

    # the random circuits include Pauli-target observables followed by operations on their qubits,
    # and observables in REPEAT blocks
    assert any(
        _count_uses_after_pauli_observables(circuit.flattened().without_noise())
        for circuit, _ in cases
    )
    assert any(
        instruction.body_copy().num_observables
        for circuit, _ in cases
        for instruction in circuit
        if isinstance(instruction, stim.CircuitRepeatBlock)
    )

    # compare with flows
    num_stabilizers = num_undetermined = 0
    for circuit, qubits in cases:
        rows = _get_fixed_parities(circuit, qubits)
        assert np.array_equal(rows, _get_fixed_parities(circuit, qubits, by_flows=True))
        on_detectors = np.any(rows[:, : circuit.num_detectors], axis=1)
        on_state = np.any(rows[:, circuit.num_detectors :], axis=1)
        num_stabilizers += int(np.count_nonzero(on_state & ~on_detectors))
        num_undetermined += int(np.count_nonzero(on_state & on_detectors))

    # the cases include stabilizers, and stabilizers with undetermined signs
    assert num_stabilizers and num_undetermined


@pytest.mark.parametrize(
    ("circuit", "qubits", "expected"),
    [
        ("CX sweep[0] 0", [0], ["+Z"]),
        ("M !0\nCX rec[-1] 1", [1], ["-Z"]),
        ("MPAD 1\nCX rec[-1] 0", [0], ["-Z"]),
        ("HERALDED_ERASE(0.1) 0\nCX rec[-1] 1", [1], ["+Z"]),
        ("REPEAT 3 {\n    X 0\n}", [0], ["-Z"]),
    ],
)
def test_state_stabilizer_signs(circuit: str, qubits: list[int], expected: list[str]) -> None:
    """Find the signs of stabilizers that sweep bits, measurement results, and REPEAT blocks fix.

    Sweep bits are 0.  ``M !0`` reports 1 for qubit 0 in |0>, and so flips qubit 1.  ``MPAD 1``
    reports 1.  Without noise, heralded noise reports 0.
    """
    stabilizers = circuits.get_state_stabilizers(stim.Circuit(circuit), qubits)
    assert stabilizers == [stim.PauliString(string) for string in expected]


@pytest.mark.parametrize(
    ("circuit", "qubits", "expected"),
    [
        ("H 0\nCX 0 1\nMR 0 0\nDETECTOR rec[-1]", [1, 2], ["+_Z"]),
        ("H 0\nCX 0 1\nMRX 0 0\nDETECTOR rec[-1]", [1, 2], ["+_Z"]),
        ("H 0\nCX 0 1\nMRY 0 0\nDETECTOR rec[-1]", [1, 2], ["+_Z"]),
        ("H 0\nCX 0 1\nMR 0 0\nDETECTOR rec[-2]", [1], r"stabilizer Z1 .*detectors \[0\]"),
        ("H 0\nCX 0 1\nREPEAT 2 {\n    MR 0 0\n}\nDETECTOR rec[-3]", [1, 2], ["+_Z"]),
    ],
)
def test_state_stabilizers_with_repeated_reset_measurements(
    circuit: str, qubits: list[int], expected: list[str] | str
) -> None:
    """Find state stabilizers after MR, MRX, and MRY with a repeated target.

    Stim's flows of ``MR 0 0`` attribute the first result to the second
    (https://github.com/quantumlib/Stim/issues/1109), so the flows reference splits such
    instructions.  Here, after ``H 0; CX 0 1``, the first result of ``MR 0 0`` is random, and gives
    the sign of Z on qubit 1.  The second result is 0, so a detector on it is deterministic.  So
    only Z on the untouched qubit 2 has a fixed sign.  A detector on the first result instead leaves
    the sign of Z on qubit 1 undetermined.  MRX and MRY act alike in their own bases, and so does
    the first ``MR 0 0`` of a REPEAT block.
    """
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            circuits.get_state_stabilizers(stim.Circuit(circuit), qubits)
    else:
        stabilizers = circuits.get_state_stabilizers(stim.Circuit(circuit), qubits)
        assert stabilizers == [stim.PauliString(string) for string in expected]


def test_deprecated_logical_state_stabilizers_argument_order() -> None:
    """The released argument order (state_prep_circuit, code, qubit_ids) warns and still works."""
    code = codes.SteaneCode()
    data = tuple(range(3, 3 + len(code)))
    prep = circuits.with_remapped_qubits(circuits.get_encoding_circuit(code, only_zero=True), data)
    qids = circuits.QubitIDs(data, tuple(range(20, 20 + code.num_checks)))
    expected = circuits.get_logical_state_stabilizers(code, prep, qids)

    # the released keyword names still work in any order, without warning
    assert expected == circuits.get_logical_state_stabilizers(
        state_prep_circuit=prep, code=code, qubit_ids=qids
    )

    with pytest.warns(DeprecationWarning, match="DEPRECATED") as record:
        stabilizers = circuits.get_logical_state_stabilizers(prep, code, qids)
    assert stabilizers == expected
    assert len(record) == 1 and record[0].filename == __file__


def test_get_logical_tableau_from_code_data() -> None:
    """Identify a logical tableau, and reject invalid encoders and non-logical circuits."""
    code = codes.FiveQubitCode()
    encoder, decoder = circuits.get_encoder_and_decoder(code)
    logical_x = math.op_to_string(code.get_logical_ops(Pauli.X, symplectic=True)[0])
    logical_tableau = encoding._get_logical_tableau_from_code_data(
        1, 0, encoder, decoder, logical_x.to_tableau().to_circuit()
    )
    assert logical_tableau == stim.Tableau.from_named_gate("X")

    with pytest.raises(ValueError, match="does not implement a logical operation"):
        encoding._get_logical_tableau_from_code_data(1, 0, encoder, decoder, stim.Circuit("X 0"))
    with pytest.raises(ValueError, match="incompatible dimensions"):
        encoding._get_logical_tableau_from_code_data(
            1, 0, stim.Tableau(1), stim.Tableau(2), stim.Circuit()
        )


def _get_random_state_prep_circuit(rng: np.random.Generator) -> tuple[stim.Circuit, list[int]]:
    """A random circuit, and a random list of distinct qubits to request.

    The requested qubits come in a random order, and may include a qubit that the circuit leaves
    untouched.
    """
    num_qubits = int(rng.integers(2, 6))
    circuit = _get_random_circuit(rng, num_qubits, num_measurements_before=0, in_repeat_block=False)
    num_requested = int(rng.integers(0, num_qubits + 2))
    requested = rng.choice(num_qubits + 1, size=num_requested, replace=False).tolist()
    return circuit, requested


def _get_random_circuit(
    rng: np.random.Generator, num_qubits: int, num_measurements_before: int, in_repeat_block: bool
) -> stim.Circuit:
    """A random circuit, which may follow the given number of measurements.

    Its instructions come from _GATES; MPP, SPP, and SPP_DAG; detectors; observables with
    measurement-record and Pauli targets; Paulis controlled by measurement records or sweep bits;
    MPAD; annotations; noise, including heralded noise; and, outside a REPEAT block, REPEAT blocks.
    Measurement results may be inverted.  Each gate has one or two groups of targets, and each MPP,
    SPP, or SPP_DAG one or two Pauli products, each from one group.  The groups are drawn with
    replacement, so that targets repeat on purpose.
    """
    circuit = stim.Circuit()
    for _ in range(int(rng.integers(1, 6 if in_repeat_block else 25))):
        num_measurements = num_measurements_before + circuit.num_measurements
        size = int(rng.integers(1, min(3, num_measurements) + 1)) if num_measurements else 0
        records = [
            stim.target_rec(-int(lookback) - 1)
            for lookback in rng.choice(num_measurements, size=size, replace=False)
        ]
        groups = [rng.permutation(num_qubits).tolist() for _ in range(rng.integers(1, 3))]
        kind = rng.random()
        if kind < 0.45:
            name = str(rng.choice(_GATES))
            data = stim.gate_data(name)
            width = 1 if data.is_single_qubit_gate else 2
            targets = [
                stim.target_inv(qubit)
                if data.produces_measurements and rng.random() < 0.3
                else qubit
                for group in groups
                for qubit in group[:width]
            ]
            circuit.append(name, targets)
        elif kind < 0.55:
            products: list[stim.GateTarget] = []
            for group in groups:
                paulis = ["_"] * num_qubits
                for qubit in group[: rng.integers(1, 4)]:
                    paulis[qubit] = str(rng.choice(["X", "Y", "Z"]))
                pauli_string = stim.PauliString("".join(paulis))
                products += stim.target_combined_paulis(pauli_string, bool(rng.random() < 0.3))
            circuit.append(str(rng.choice(["MPP", "SPP", "SPP_DAG"])), products)
        elif kind < 0.67:
            if records:
                circuit.append("DETECTOR", records)
        elif kind < 0.72:
            paulis = [
                stim.target_pauli(qubit, rng.choice(["X", "Y", "Z"])) for qubit in groups[0][:2]
            ]
            targets = [*paulis, *records]
            order = rng.permutation(len(targets))[: rng.integers(1, 4)]
            circuit.append(
                "OBSERVABLE_INCLUDE", [targets[index] for index in order], int(rng.integers(2))
            )
        elif kind < 0.78:
            control = (
                records[0]
                if records and rng.random() < 0.7
                else stim.target_sweep_bit(int(rng.integers(2)))
            )
            if rng.random() < 0.5:
                circuit.append(str(rng.choice(["CX", "CY", "CZ"])), [control, groups[0][0]])
            else:
                circuit.append(str(rng.choice(["CZ", "XCZ", "YCZ"])), [groups[0][0], control])
        elif kind < 0.8:
            circuit.append("MPAD", [int(rng.integers(2))])
        elif kind < 0.85:
            if not in_repeat_block:
                body = _get_random_circuit(rng, num_qubits, num_measurements, in_repeat_block=True)
                if len(body):
                    circuit.append(stim.CircuitRepeatBlock(int(rng.integers(1, 4)), body))
        elif kind < 0.9:
            circuit.append("TICK")
            circuit.append("QUBIT_COORDS", groups[0][:1], [0, 1])
            circuit.append("SHIFT_COORDS", [], [1])
        else:
            noise = [
                ("X_ERROR", 1, [0.1]),
                ("DEPOLARIZE2", 2, [0.1]),
                ("HERALDED_ERASE", 1, [0.1]),
                ("HERALDED_PAULI_CHANNEL_1", 1, [0.1, 0.1, 0, 0]),
            ]
            name, width, args = noise[rng.integers(len(noise))]
            circuit.append(name, groups[0][:width], args)
    return circuit


def _repeats_target(instruction: stim.CircuitInstruction) -> bool:
    """Whether an instruction has a target more than once."""
    values = [target.value for target in instruction.targets_copy()]
    return len(set(values)) < len(values)


def _count_uses_after_pauli_observables(circuit: stim.Circuit) -> int:
    """The number of operations on a qubit that an earlier observable has a Pauli target on.

    Operations are Clifford gates, resets, and measurements, and the circuit must be flat.
    """
    observed: set[int] = set()
    count = 0
    for instruction in circuit:
        targets = instruction.targets_copy()
        qubits = {target.value for target in targets if target.is_qubit_target}
        qubits |= {target.value for target in targets if target.pauli_type != "I"}
        data = stim.gate_data(instruction.name)
        if instruction.name == "OBSERVABLE_INCLUDE":
            observed |= qubits
        elif data.is_unitary or data.is_reset or data.produces_measurements:
            count += bool(qubits & observed)
    return count


def _get_fixed_parities(
    circuit: stim.Circuit, qubits: list[int], *, by_flows: bool = False
) -> npt.NDArray[np.uint8]:
    """Find the fixed parities of a circuit and requested qubits, as get_state_stabilizers does.

    Returns the reduced row echelon form of the fixed parities, with a column for each detector of
    the circuit.  If by_flows is True, find them with the flows reference instead of by simulation.
    The simulation leaves out detectors that are deterministic on their own, so each of them gets a
    row of its own.
    """
    state_qubits = np.asarray(qubits, dtype=int)
    num_qubits = max(circuit.num_qubits, int(state_qubits.max(initial=-1)) + 1)
    circuit = stim.Circuit("R " + " ".join(map(str, range(num_qubits)))) + circuit.without_noise()
    if by_flows:
        return _get_fixed_parities_by_flows(circuit, state_qubits)
    simulator = stim.TableauSimulator()
    simulator.do(circuit)
    detectors, rows = encoding._get_fixed_parities_by_simulation(circuit, simulator, state_qubits)
    assert np.array_equal(get_independent_rows(rows), rows)
    left_out = np.setdiff1d(np.arange(circuit.num_detectors), detectors)
    matrix = np.zeros(
        (len(rows) + len(left_out), circuit.num_detectors + 2 * len(qubits)), np.uint8
    )
    matrix[: len(rows), detectors] = rows[:, : len(detectors)]
    matrix[: len(rows), circuit.num_detectors :] = rows[:, len(detectors) :]
    matrix[len(rows) + np.arange(len(left_out)), left_out] = 1
    return get_independent_rows(matrix)


def _get_fixed_parities_by_flows(
    circuit: stim.Circuit, state_qubits: npt.NDArray[np.int_]
) -> npt.NDArray[np.uint8]:
    """Find the fixed parities of a circuit from its flows, as a reference for simulation.

    Fixed parities are as the comment in get_state_stabilizers defines them.  The circuit must be
    noiseless and reset every qubit first, so each of its flows has the form ``1 -> P xor M``, for
    a Pauli string P and a set M of measurements.  Stim's flows ignore observables.  Returns the
    reduced row echelon form of the fixed parities, with a column for each detector of the circuit.
    """
    flow_generators = _split_repeated_reset_measurements(circuit).flow_generators()

    # identify some useful numbers
    num_qubits = circuit.num_qubits
    num_measurements = circuit.num_measurements
    num_columns = 2 * num_qubits + num_measurements
    num_detectors = circuit.num_detectors
    num_rows = len(flow_generators) + num_detectors

    # build a matrix of stabilizers for the entire circuit output, determined by the flows
    matrix = np.zeros((num_rows, num_columns), dtype=np.uint8)
    for gg, flow in enumerate(flow_generators):
        pauli_string = flow.output_copy()
        if pauli_string:
            xs, zs = pauli_string.to_numpy()
            matrix[gg, :num_qubits] = xs
            matrix[gg, num_qubits : 2 * num_qubits] = zs
        for measurement in flow.measurements_copy():
            matrix[gg, -num_measurements + measurement] = 1
        assert not flow.included_observables_copy()

    # add stabilizers defined by detectors in the circuit
    detector_counter = 0
    measurement_counter = 0
    for instruction in circuit.flattened():
        measurement_counter += instruction.num_measurements
        if instruction.name == "DETECTOR":
            row = len(flow_generators) + detector_counter
            for target in instruction.targets_copy():
                col = -num_measurements + measurement_counter + target.value
                matrix[row, col] ^= 1
            detector_counter += 1

    # Row reduction must eliminate unwanted support before selecting a basis supported on the
    # requested qubits.  Natural column order would instead preserve state-qubit pivots and lose
    # stabilizers that are sums of the original flow rows.  Columns q and num_qubits + q hold the X
    # and Z parts on qubit q.  Columns between the unwanted and state columns record which detector
    # rows each row is a sum of.
    other_qubits = np.array([qq for qq in range(num_qubits) if qq not in state_qubits], dtype=int)
    unwanted_columns = np.concatenate(
        [other_qubits, other_qubits + num_qubits, np.arange(2 * num_qubits, num_columns, dtype=int)]
    )
    state_columns = np.concatenate([state_qubits, state_qubits + num_qubits])
    detector_columns = np.zeros((num_rows, num_detectors), dtype=np.uint8)
    detector_columns[len(flow_generators) :] = np.eye(num_detectors, dtype=np.uint8)
    reordered_matrix = np.hstack(
        [matrix[:, unwanted_columns], detector_columns, matrix[:, state_columns]]
    )

    # Identify the fixed parities.  The unwanted columns come first, so the rows of the reduced row
    # echelon form with pivots on detectors or requested qubits vanish on these columns.  These
    # rows span every fixed parity, since a sum of rows is nonzero at the pivot of each row in the
    # sum.
    rows = get_independent_rows(reordered_matrix)
    num_unwanted_columns = len(unwanted_columns)
    on_others = np.any(rows[:, :num_unwanted_columns], axis=1)
    return rows[~on_others, num_unwanted_columns:]


def _split_repeated_reset_measurements(circuit: stim.Circuit) -> stim.Circuit:
    """Split each MR, MRX, or MRY instruction that repeats a target into single-target instructions.

    This works around https://github.com/quantumlib/Stim/issues/1109: for such an instruction, such
    as ``MR 0 0``, ``stim.Circuit.flow_generators`` attributes the outcome of one measurement to
    another.  The issue also gives MPAD wrong flows, but they differ only in sign, which
    _get_fixed_parities_by_flows ignores, so MPAD needs no split.  TICK instructions before,
    between, and after the single-target instructions stop Stim from merging them with each other
    or with their neighbors.

    Once qLDPC requires a stim release that fixes this issue, delete this function and its call in
    _get_fixed_parities_by_flows.
    """
    split_circuit = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            body = _split_repeated_reset_measurements(instruction.body_copy())
            split_circuit.append(stim.CircuitRepeatBlock(instruction.repeat_count, body))
        elif instruction.name in ("MR", "MRX", "MRY") and _repeats_target(instruction):
            split_circuit.append("TICK")
            for target in instruction.targets_copy():
                split_circuit.append(
                    instruction.name, [target], instruction.gate_args_copy(), tag=instruction.tag
                )
                split_circuit.append("TICK")
        else:
            split_circuit.append(instruction)
    return split_circuit
