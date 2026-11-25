# SPDX-License-Identifier: Apache-2.0

"""Unit tests for encoding.py."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import circuits, codes, math
from qldpc.circuits import encoding
from qldpc.objects import Pauli


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


@pytest.mark.parametrize("qubits", [[0, 1, 0], [-1], -1, [0.5, 1]])
def test_state_stabilizers_of_invalid_qubits(qubits: list[int] | int) -> None:
    """Reject negative, repeated, or non-integer requested qubits, and a negative number of qubits.

    Position i of a returned string refers to qubits[i], so with qubits [0, 1, 0] after
    ``H 0; CX 0 1``, a returned XXX would stand for X0 X1 X0 = X1, which is not a stabilizer.
    """
    with pytest.raises(ValueError, match="distinct nonnegative integers"):
        circuits.get_state_stabilizers(stim.Circuit("H 0\nCX 0 1"), qubits)


def test_state_stabilizer_failure_regressions() -> None:
    """Exercise row-space, qubit-order, detector-parity, noisy-sign, and zero-logical cases."""
    ghz = circuits.get_state_stabilizers(stim.Circuit("H 0\nCX 0 1"), [0, 1])
    assert stim.PauliString("+XX") in ghz
    assert stim.PauliString("+ZZ") in ghz

    # The only stabilizer of this rotated GHZ state on qubits 1 and 2 is X1 Z2, the product of Z0 X1
    # and Z0 Z2, so finding it requires eliminating qubit 0 from the stabilizers.  Requested as
    # [2, 1], it is returned as ZX, since position i refers to qubits[i].
    rotated_ghz = stim.Circuit("H 0\nCX 0 1 0 2\nH 1")
    assert circuits.get_state_stabilizers(rotated_ghz, [1, 2]) == [stim.PauliString("+XZ")]
    assert circuits.get_state_stabilizers(rotated_ghz, [2, 1]) == [stim.PauliString("+ZX")]

    # the integer 0 requests no qubits, and gives no stabilizers
    assert circuits.get_state_stabilizers(rotated_ghz, 0) == []

    repeated_detector = stim.Circuit("H 0\nM 0\nDETECTOR rec[-1] rec[-1]")
    assert circuits.get_state_stabilizers(repeated_detector, [0]) == []

    # noise is dropped, so every run gives +Z
    noisy_signs = [
        circuits.get_state_stabilizers(stim.Circuit("X_ERROR(0.5) 0"), [0]) for _ in range(20)
    ]
    assert noisy_signs == [[stim.PauliString("+Z")]] * 20

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

    # Each round's random outcome flips qubit 1 when it is 1, so the sign of Z1 is the parity of
    # three random detectors.  From three repetitions on, Stim keeps the REPEAT block in the
    # detector error model instead of unrolling it, so this case checks that the random detectors
    # inside the folded block are all found.
    circuit = stim.Circuit("REPEAT 3 {\n    H 0\n    CX 0 1\n    MR 0\n    DETECTOR rec[-1]\n}")
    with pytest.raises(ValueError, match=r"stabilizer Z1 .*detectors \[0, 1, 2\]"):
        circuits.get_state_stabilizers(circuit, [1])

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


@pytest.mark.parametrize(
    ("circuit", "qubits", "expected"),
    [
        ("H 0\nOBSERVABLE_INCLUDE(0) Z0\nH 0", [0], ["+Z"]),
        ("H 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", [0], []),
        ("H 0\nREPEAT 3 {\n    OBSERVABLE_INCLUDE(0) Z0\n}", [0], ["+X"]),
        (
            (
                "REPEAT 2 {\n    H 0\n    CX 0 1\n    MR 0\n    OBSERVABLE_INCLUDE(0) rec[-1]\n}\n"
                "CX rec[-1] 1"
            ),
            [1],
            [],
        ),
    ],
)
def test_state_stabilizers_ignore_observables(
    circuit: str, qubits: list[int], expected: list[str]
) -> None:
    """Ignore observables, even random ones, which Stim's detector error model rejects.

    In the last case, the final feedback leaves qubit 1 flipped by the random result of the first
    round, so Z1 has no fixed sign.  This checks that removing the observables keeps the block's
    repeat count: with one repetition, the detector error model would see no random outcome
    reaching Z1, and Z1 would be returned with a random sign.
    """
    stabilizers = circuits.get_state_stabilizers(stim.Circuit(circuit), qubits)
    assert stabilizers == [stim.PauliString(string) for string in expected]


@pytest.mark.parametrize(
    ("circuit", "qubits", "expected"),
    [
        ("CX sweep[0] 0", [0], ["+Z"]),
        ("M !0\nCX rec[-1] 1", [1], ["-Z"]),
        ("H 0\nCX 0 1\nM 0\nCX rec[-1] 1", [1], ["+Z"]),
        ("MPAD 1\nCX rec[-1] 0", [0], ["-Z"]),
        ("HERALDED_ERASE(0.1) 0\nCX rec[-1] 1", [1], ["+Z"]),
        ("REPEAT 3 {\n    X 0\n}", [0], ["-Z"]),
    ],
)
def test_state_stabilizer_signs(circuit: str, qubits: list[int], expected: list[str]) -> None:
    """Find the signs of stabilizers that sweep bits, measurement results, and REPEAT blocks fix.

    Sweep bits are 0.  ``M !0`` reports 1 for qubit 0 in |0>, and so flips qubit 1.  Feedback on
    the random result of ``M 0`` in a Bell state undoes the result's effect on qubit 1.  ``MPAD 1``
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

    After ``H 0; CX 0 1``, the first result of ``MR 0 0`` is random, and gives the sign of Z on
    qubit 1.  The second result is 0, so a detector on it is deterministic.  So only Z on the
    untouched qubit 2 has a fixed sign.  A detector on the first result instead leaves the sign of Z
    on qubit 1 undetermined.  MRX and MRY act alike in their own bases, and so does the first
    ``MR 0 0`` of a REPEAT block.  Stim's flow generators attribute the first result to the second
    (https://github.com/quantumlib/Stim/issues/1109), so a stabilizer finder built on them would
    give other answers here.
    """
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            circuits.get_state_stabilizers(stim.Circuit(circuit), qubits)
    else:
        stabilizers = circuits.get_state_stabilizers(stim.Circuit(circuit), qubits)
        assert stabilizers == [stim.PauliString(string) for string in expected]


@pytest.mark.parametrize(
    ("preparation", "reset", "expected"),
    [
        ("", "RX", "+X"),
        ("X 0", "RX", "+X"),
        ("", "RY", "+Y"),
        ("X 0", "RY", "+Y"),
        ("H 0", "R", "+Z"),
        ("X 0\nH 0", "R", "+Z"),
        ("H 0", "RY", "+Y"),
        ("X 0\nH 0", "RY", "+Y"),
        ("H_YZ 0", "R", "+Z"),
        ("X 0\nH_YZ 0", "R", "+Z"),
        ("H_YZ 0", "RX", "+X"),
        ("X 0\nH_YZ 0", "RX", "+X"),
        ("X 0", "R", "+Z"),
        ("H 0", "RX", "+X"),
        ("X 0\nH 0", "RX", "+X"),
        ("X 0\nH_YZ 0", "RY", "+Y"),
    ],
)
def test_state_stabilizers_after_replaced_resets(
    monkeypatch: pytest.MonkeyPatch, preparation: str, reset: str, expected: str
) -> None:
    """Reset a qubit in an eigenstate of a single-qubit Pauli, with the simulation stepping.

    The preparation puts the qubit in the +1 or -1 eigenstate of Z, X, or Y, since H_YZ maps Z to
    Y.  The simulation steps through every circuit when its threshold is 0.  It replaces a reset of
    a qubit in an eigenstate of another Pauli by gates, and keeps a reset of a qubit in an
    eigenstate of the reset's Pauli, which must leave the qubit in the +1 eigenstate.

    TEMPORARY until stim.TableauSimulator resets an unentangled qubit in linear time: see
    circuits/stim_workarounds.md, "Resets with random outcomes in stim.TableauSimulator".
    """
    monkeypatch.setattr(encoding, "_MIN_QUBITS_FOR_CHEAP_RESETS", 0)
    circuit = stim.Circuit(f"{preparation}\n{reset} 0")
    assert circuits.get_state_stabilizers(circuit, [0]) == [stim.PauliString(expected)]


# TEMPORARY until stim.TableauSimulator resets an unentangled qubit in linear time: see
# circuits/stim_workarounds.md, "Resets with random outcomes in stim.TableauSimulator".
@pytest.mark.parametrize("min_qubits", [0, None])
def test_state_stabilizers_after_resets(
    monkeypatch: pytest.MonkeyPatch, min_qubits: int | None
) -> None:
    """Find state stabilizers after resets that mix replaced, kept, and repeated targets.

    The comment on each expected stabilizer gives the final state of its qubit.  The circuit has
    too few qubits for the simulation to replace resets by gates, unless min_qubits is 0.  Then the
    simulation replaces the reset of qubit 0 but keeps that of qubit 3 in the same instruction,
    since qubit 3 is entangled with qubit 4.  The feedback on the result of ``M 1`` checks that
    replacing the reset of qubit 1 keeps the measurement record.  The second target of ``RY 2 2``
    and the reset in the REPEAT block run as they are.
    """
    if min_qubits is not None:
        monkeypatch.setattr(encoding, "_MIN_QUBITS_FOR_CHEAP_RESETS", min_qubits)
    circuit = stim.Circuit("""
        X 0 1
        M 1
        RY 1
        CX rec[-1] 5
        H 2
        RY 2 2
        H 3
        CX 3 4
        RX 3 0
        REPEAT 2 {
            RX 6
        }
    """)
    expected = [
        "+X______",  # |+>
        "+_Y_____",  # |+i>
        "+__Y____",  # |+i>
        "+___X___",  # |+>, while qubit 4 is in |0> or |1> at random
        "+______X",  # |+>
        "-_____Z_",  # |1>
    ]
    stabilizers = circuits.get_state_stabilizers(circuit, 7)
    assert stabilizers == [stim.PauliString(string) for string in expected]


def test_cheap_resets_on_large_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace resets by gates on a circuit with many qubits, but not on one with few qubits.

    TEMPORARY until stim.TableauSimulator resets an unentangled qubit in linear time: see
    circuits/stim_workarounds.md, "Resets with random outcomes in stim.TableauSimulator".
    """
    calls = []
    do_with_cheap_resets = encoding._do_with_cheap_resets

    def recording_do_with_cheap_resets(
        simulator: stim.TableauSimulator, circuit: stim.Circuit
    ) -> None:
        calls.append(circuit.num_qubits)
        do_with_cheap_resets(simulator, circuit)

    monkeypatch.setattr(encoding, "_do_with_cheap_resets", recording_do_with_cheap_resets)
    for num_qubits in [2, 2000]:
        simulator = encoding._simulate_noiseless_run(stim.Circuit(f"RX {num_qubits - 1}"))
        assert simulator.peek_x(num_qubits - 1) == 1
    assert calls == [2000]


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
