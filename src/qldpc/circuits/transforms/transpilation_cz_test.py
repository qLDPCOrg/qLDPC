# SPDX-License-Identifier: Apache-2.0

"""Unit tests for transpilation_cz.py."""

from __future__ import annotations

import random
import warnings
from collections.abc import Sequence

import pytest
import stim

from qldpc.circuits import DEFAULT_IMMUNE_OP_TAG, DEFAULT_IMMUNE_QUBIT_TAG
from qldpc.circuits.transforms import transpile_to_cz
from qldpc.circuits.transforms.transpilation_cz import _clean_up, _commute_through_cz

_SINGLE_QUBIT_GATES = sorted(
    name for name, data in stim.gate_data().items() if data.is_unitary and data.is_single_qubit_gate
)
_TWO_QUBIT_GATES = sorted(
    name for name, data in stim.gate_data().items() if data.is_unitary and data.is_two_qubit_gate
)


def test_two_qubit_gates() -> None:
    """Each two-qubit gate becomes at most one CZ, single-qubit gates, and SWAPs at the end."""
    for name in _TWO_QUBIT_GATES:
        circuit = stim.Circuit(f"{name} 0 1")
        transpiled = transpile_to_cz(circuit)
        assert _get_tableau(transpiled, 2) == circuit.to_tableau(), name
        assert _count(transpiled, "CZ") == int(name not in ("II", "SWAP")), name
        names = [op.name for op in transpiled]
        assert "SWAP" not in names or "CZ" not in names[names.index("SWAP") :], name


@pytest.mark.filterwarnings("ignore:REPEAT support:UserWarning")
def test_random_transpilation(pytestconfig: pytest.Config) -> None:
    """Random circuits are equivalent to their transpiled versions, without cleanup."""
    rng = random.Random(pytestconfig.getoption("randomly_seed"))
    for index in range(100):
        num_qubits = rng.randint(2, 4)
        allow_repeat = index % 2 == 0
        circuit = _get_random_circuit(rng, num_qubits, rng.randint(1, 20), allow_repeat)
        immune_qubits = rng.sample(range(num_qubits), rng.randint(0, 2))
        transpiled = transpile_to_cz(circuit, immune_qubits=immune_qubits, cleanup=False)
        _assert_equivalent(circuit, transpiled)
        _assert_in_cz_gate_set(transpiled)


def test_random_cleanup(pytestconfig: pytest.Config) -> None:
    """The cleanup keeps random circuits whose untagged two-qubit gates are CZ and SWAP equivalent.

    It also keeps them in the gate set of transpile_to_cz, and never adds CZ gates.
    """
    rng = random.Random(pytestconfig.getoption("randomly_seed"))
    for index in range(100):
        allow_repeat = index % 2 == 0
        circuit = _get_random_circuit(
            rng, rng.randint(2, 4), rng.randint(1, 20), allow_repeat, two_qubit_gates=["CZ", "SWAP"]
        )
        cleaned_circuit = _clean_up(circuit)
        _assert_equivalent(circuit, cleaned_circuit)
        _assert_in_cz_gate_set(cleaned_circuit)
        assert _count(cleaned_circuit, "CZ") <= _count(circuit, "CZ")


def test_commute_through_cz() -> None:
    """Single-qubit gates before a CZ become H, SQRT_X, or nothing, and the rest moves past it."""
    cz = stim.Tableau.from_named_gate("CZ")
    for name_0 in _SINGLE_QUBIT_GATES:
        for name_1 in _SINGLE_QUBIT_GATES:
            gate_0, gate_1 = (stim.Tableau.from_named_gate(name) for name in (name_0, name_1))
            first_0, first_1, last_0, last_1 = _commute_through_cz(gate_0, gate_1)
            assert (gate_0 + gate_1).then(cz) == (first_0 + first_1).then(cz).then(last_0 + last_1)
            allowed = [stim.Tableau.from_named_gate(name) for name in ("I", "H", "SQRT_X")]
            assert first_0 in allowed and first_1 in allowed

    # diagonal gates and Paulis move past CZ gates, so that no gate comes before the first CZ
    circuit = stim.Circuit("S 0\nX 1\nH 1\nH 1\nCZ 0 1\nCZ 1 2\nH 0")
    transpiled = transpile_to_cz(circuit)
    assert _get_tableau(transpiled, 3) == circuit.to_tableau()
    assert transpiled == stim.Circuit("CZ 0 1 1 2\nC_XYZ 0\nX 1\nZ 2")


def test_cleanup() -> None:
    """The cleanup cancels CZ pairs, merges gates into resets, and packs gates into layers."""
    # CZ gates cancel if only diagonal gates separate them
    assert transpile_to_cz(stim.Circuit("CX 0 1\nCX 0 1"), cleanup=False) == stim.Circuit(
        "H 1\nCZ 0 1 0 1\nH 1"
    )
    assert transpile_to_cz(stim.Circuit("CX 0 1\nCX 0 1")) == stim.Circuit()
    assert transpile_to_cz(stim.Circuit("CZ 0 1\nS 0\nCZ 0 1")) == stim.Circuit("S 0")
    circuit = stim.Circuit("CZ 0 1\nSQRT_X 0\nCZ 0 1")
    assert transpile_to_cz(circuit) == circuit
    # X and Y map Z to -Z, so they block the cancellation, too (checked on the cleanup directly,
    # since transpile_to_cz moves them past CZ gates)
    for pauli in "XY":
        circuit = stim.Circuit(f"CZ 0 1\n{pauli} 0\nCZ 0 1")
        cleaned_circuit = _clean_up(circuit)
        _assert_equivalent(circuit, cleaned_circuit)
        assert _count(cleaned_circuit, "CZ") == 2

    # gates right after untagged resets merge into them, leaving at most a Pauli gate
    assert transpile_to_cz(stim.Circuit("R 0\nH 0\nCZ 0 1")) == stim.Circuit("RX 0\nCZ 0 1")
    circuit = stim.Circuit("R 0\nSQRT_X 0\nCZ 0 1")
    transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    assert transpiled == stim.Circuit("RY 0\nCZ 0 1\nZ 0")
    circuit = stim.Circuit("R[t] 0\nH 0\nCZ 0 1")
    assert transpile_to_cz(circuit) == circuit

    # gates are held past instructions that do not address their qubits
    circuit = stim.Circuit("R 1\nSQRT_X 1\nCZ 1 0\nM 0\nCZ 1 0")
    transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    assert transpiled == stim.Circuit("RY 1\nCZ 1 0\nM 0\nCZ 1 0\nZ 1")

    # CZ gates come as early as possible
    circuit = stim.Circuit("CX 0 1\nCX 2 3\nCX 1 2")
    assert transpile_to_cz(circuit, cleanup=False) == stim.Circuit(
        "H 1\nCZ 0 1\nH 3\nCZ 2 3\nH 1 2\nCZ 1 2\nH 2 3"
    )
    assert transpile_to_cz(circuit) == stim.Circuit("H 1 3\nCZ 0 1 2 3\nH 1 2\nCZ 1 2\nH 2 3")


def test_measurements_and_resets() -> None:
    """Delayed gates merge into measurements and observables, and resets discard them."""
    circuit = stim.Circuit("""
        H 0
        M !0
        X 1
        MR 1
        CZ 1 2
        H 2
        MRY 2
        OBSERVABLE_INCLUDE(0) X2 rec[-1]
        S 0
        R 0
        MPP X0*Z1
    """)
    transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    assert transpiled == stim.Circuit("""
        MX !0
        MR !1
        CZ 1 2
        MRY !2
        OBSERVABLE_INCLUDE(0) !Z2 rec[-1]
        R 0
        MPP X0*Z1
        SQRT_Y_DAG 2
    """)


def test_tagged_instructions() -> None:
    """Tagged instructions are copied, after delayed gates, with relabeled qubit targets."""
    circuit = stim.Circuit("""
        SWAP 0 1
        S 0
        H[t] 0
        X 1
        R[t] 1
        CX[t] 0 1
        M[t] 0
        DETECTOR[t](1) rec[-1]
    """)
    transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    # the delayed S comes before the tagged H, and the tagged reset discards the delayed X
    assert transpiled == stim.Circuit("""
        S 1
        H[t] 1
        R[t] 0
        CX[t] 1 0
        M[t] 1
        DETECTOR[t](1) rec[-1]
        SWAP 1 0
    """)

    # classically controlled gates and SPP gates are fine if they are tagged
    circuit = stim.Circuit("M 0\nCX[t] rec[-1] 1\nSPP[t] X0*Z1")
    assert transpile_to_cz(circuit) == circuit


def test_immune_qubits() -> None:
    """States never move between qubits that are immune to noise and qubits that are not."""
    circuit = stim.Circuit("SWAP 0 1 1 2\nCZ 0 2\nM 0")
    transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    assert transpiled == stim.Circuit("CZ 1 0\nM 1\nSWAP 1 0 2 1")
    assert transpile_to_cz(circuit, immune_qubits=[0]) == stim.Circuit(
        "SWAP 0 1\nCZ 0 1\nM 0\nSWAP 2 1"
    )

    tagged_circuit = stim.Circuit(f"QUBIT_COORDS[{DEFAULT_IMMUNE_QUBIT_TAG}](0) 0") + circuit
    transpiled = transpile_to_cz(tagged_circuit)
    _assert_equivalent(tagged_circuit, transpiled)
    assert transpiled == tagged_circuit[:1] + transpile_to_cz(circuit, immune_qubits=[0])
    assert transpile_to_cz(tagged_circuit, immune_qubit_tag=None) == tagged_circuit[
        :1
    ] + transpile_to_cz(circuit)


def test_repeat_blocks() -> None:
    """REPEAT blocks start and end with settled qubits, with a warning if that adds gates."""
    circuit = stim.Circuit("REPEAT 3 {\nCX 0 1\nH 1\nM 1\n}")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    assert transpiled == stim.Circuit("REPEAT 3 {\nH 1\nCZ 0 1\nM 1\n}")

    # gates before a REPEAT block and at the end of its iterations, and tags, stay where they are
    circuit = stim.Circuit(f"""
        S 0
        REPEAT[{DEFAULT_IMMUNE_OP_TAG}] 2 {{
            SWAP 0 1
            CX 0 1
            M 1
        }}
    """)
    with pytest.warns(UserWarning, match="REPEAT support"):
        transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    assert transpiled == stim.Circuit(f"""
        S 0
        REPEAT[{DEFAULT_IMMUNE_OP_TAG}] 2 {{
            H 0
            CZ 1 0
            MX 0
            H 0
            SWAP 1 0
        }}
    """)


def test_annotations() -> None:
    """Annotations are copied as they are, and TICK instructions are dropped."""
    circuit = stim.Circuit("""
        QUBIT_COORDS(1, 2) 0
        SWAP 0 1
        TICK
        QUBIT_COORDS(3) 0
        M 0
        DETECTOR(0) rec[-1]
        SHIFT_COORDS(1)
        MPAD 1
    """)
    assert transpile_to_cz(circuit) == stim.Circuit("""
        QUBIT_COORDS(1, 2) 0
        QUBIT_COORDS(3) 0
        M 1
        DETECTOR(0) rec[-1]
        SHIFT_COORDS(1)
        MPAD 1
        SWAP 1 0
    """)


def test_unsupported_instructions() -> None:
    """Noise, untagged classically controlled gates, and untagged SPP gates are not supported."""
    for circuit in ["X_ERROR(0.1) 0", "X_ERROR[t](0.1) 0", "M(0.1) 0", "HERALDED_ERASE(0.1) 0"]:
        with pytest.raises(ValueError, match="Noise"):
            transpile_to_cz(stim.Circuit(circuit))
    with pytest.raises(ValueError, match="classically controlled"):
        transpile_to_cz(stim.Circuit("M 0\nCX rec[-1] 1"))
    with pytest.raises(ValueError, match="SPP gates are not supported"):
        transpile_to_cz(stim.Circuit("SPP X0*Z1"))
    with pytest.raises(ValueError, match="non-empty"):
        transpile_to_cz(stim.Circuit(), immune_qubit_tag="")


def _assert_equivalent(circuit: stim.Circuit, transpiled: stim.Circuit) -> None:
    """Assert that two noiseless circuits have the same stabilizer flows.

    Flows ignore observables, so observables with Pauli targets are first replaced by MPP
    measurements of the same Pauli products, which both circuits then must agree on.
    """
    circuit = _work_around_stim_issue_1109(_measure_pauli_observables(circuit))
    transpiled = _work_around_stim_issue_1109(_measure_pauli_observables(transpiled))
    assert all(transpiled.has_flow(flow) for flow in circuit.flow_generators())
    assert all(circuit.has_flow(flow) for flow in transpiled.flow_generators())


def _measure_pauli_observables(circuit: stim.Circuit) -> stim.Circuit:
    """Replace each OBSERVABLE_INCLUDE by an MPP measurement of the product of its Pauli targets."""
    new_circuit = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            body = _measure_pauli_observables(op.body_copy())
            new_circuit.append(stim.CircuitRepeatBlock(op.repeat_count, body))
        elif op.name == "OBSERVABLE_INCLUDE":
            # measure the product of the Pauli targets, with the parity of their inversions
            paulis = [target for target in op.targets_copy() if target.pauli_type != "I"]
            invert = sum(target.is_inverted_result_target for target in paulis) % 2 == 1
            targets = []
            for index, target in enumerate(paulis):
                if index:
                    targets.append(stim.target_combiner())
                targets.append(
                    stim.target_pauli(target.value, target.pauli_type, invert and not index)
                )
            if targets:
                new_circuit.append("MPP", targets)
        else:
            new_circuit.append(op)
    return new_circuit


def _work_around_stim_issue_1109(circuit: stim.Circuit) -> stim.Circuit:
    """Work around https://github.com/quantumlib/Stim/issues/1109 in stim.Circuit.flow_generators.

    Bug (open as of stim 1.16.0): for an MR, MRX, MRY, or MPAD instruction with several targets,
    such as ``MR 0 0``, ``flow_generators`` returns flows that ``has_flow`` rejects on the same
    circuit, so _assert_equivalent would fail on equivalent circuits.  Splitting each such
    instruction into single-target instructions avoids the bug.  The single-target instructions are
    separated by TICK, since stim merges adjacent instructions with the same name.

    Once qLDPC requires a stim release that fixes this issue, delete this function and its calls in
    _assert_equivalent.
    """
    new_circuit = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            body = _work_around_stim_issue_1109(op.body_copy())
            new_circuit.append(stim.CircuitRepeatBlock(op.repeat_count, body))
        elif op.name in ("MR", "MRX", "MRY", "MPAD"):
            # split the instruction, with TICKs that stop stim from merging it back together
            for target in op.targets_copy():
                new_circuit.append(op.name, [target], op.gate_args_copy(), tag=op.tag)
                new_circuit.append("TICK")
        else:
            new_circuit.append(op)
    return new_circuit


def _get_random_circuit(
    rng: random.Random,
    num_qubits: int,
    num_ops: int,
    allow_repeat: bool,
    two_qubit_gates: Sequence[str] = _TWO_QUBIT_GATES,
) -> stim.Circuit:
    """A random noiseless circuit, whose untagged two-qubit gates are among two_qubit_gates."""
    qubits = range(num_qubits)
    circuit = stim.Circuit()
    for _ in range(num_ops):
        kind = rng.random()

        # unitary gates
        if kind < 0.3:
            circuit.append(rng.choice(_SINGLE_QUBIT_GATES), rng.sample(qubits, rng.randint(1, 2)))
        elif kind < 0.65:
            circuit.append(rng.choice(two_qubit_gates), rng.sample(qubits, 2))

        # single-qubit measurements (with a second, inverted target on qubit 0) and resets
        elif kind < 0.75:
            name = rng.choice(["M", "MX", "MY", "MR", "MRX", "MRY"])
            circuit.append(name, [rng.choice(qubits), stim.target_inv(0)])
        elif kind < 0.8:
            circuit.append(rng.choice(["R", "RX", "RY"]), [rng.choice(qubits)])

        # Pauli-product measurements and observables with Pauli targets
        elif kind < 0.85:
            string = stim.PauliString.random(num_qubits)
            if string.weight:
                circuit.append(
                    "MPP", stim.target_combined_paulis(string, invert=rng.random() < 0.5)
                )
        elif kind < 0.9:
            paulis = [
                stim.target_pauli(qubit, rng.choice("XYZ")) for qubit in rng.sample(qubits, 2)
            ]
            circuit.append("OBSERVABLE_INCLUDE", [*paulis, stim.target_rec(-1)], 0)

        # tagged instructions, and REPEAT blocks (which do not nest)
        elif kind < 0.95:
            name, num_targets = rng.choice([("H", 1), ("CX", 2), ("M", 1), ("RX", 1)])
            targets = rng.sample(qubits, num_targets)
            circuit.append(stim.CircuitInstruction(name, targets, tag=DEFAULT_IMMUNE_OP_TAG))
        elif allow_repeat:
            body = _get_random_circuit(
                rng,
                num_qubits,
                rng.randint(1, 6),
                allow_repeat=False,
                two_qubit_gates=two_qubit_gates,
            )
            circuit.append(stim.CircuitRepeatBlock(rng.randint(1, 3), body))
    return circuit


def _assert_in_cz_gate_set(circuit: stim.Circuit) -> None:
    """Assert that the untagged unitary gates of a circuit are CZ, SWAP, or single-qubit gates.

    Unless the circuit has REPEAT blocks, at whose boundaries any single-qubit gate can come before
    a CZ, also assert that the untagged single-qubit gates right before CZ gates are H or SQRT_X.
    """
    names = {
        op.name for op in circuit.flattened() if not op.tag and stim.gate_data(op.name).is_unitary
    }
    assert names <= {"CZ", "SWAP", *_SINGLE_QUBIT_GATES}
    if not any(isinstance(op, stim.CircuitRepeatBlock) for op in circuit):
        assert _get_gates_before_czs(circuit) <= {"H", "SQRT_X"}


def _get_gates_before_czs(circuit: stim.Circuit) -> set[str]:
    """The names of the untagged single-qubit gates that come right before CZ gates on a qubit."""
    # the name of the last operation on each qubit, if it is an untagged single-qubit gate
    last_gates: dict[int, str | None] = {}
    gates_before_czs = set()
    for op in circuit.flattened():
        data = stim.gate_data(op.name)
        is_gate = data.is_unitary and data.is_single_qubit_gate and not op.tag
        for group in op.target_groups():
            qubits = [target.qubit_value for target in group if target.qubit_value is not None]
            if op.name == "CZ" and not op.tag:
                gates_before_czs |= {last_gates.get(qubit) for qubit in qubits}
            for qubit in qubits:
                last_gates[qubit] = op.name if is_gate else None
    return {name for name in gates_before_czs if name is not None}


def _get_tableau(circuit: stim.Circuit, num_qubits: int) -> stim.Tableau:
    """The tableau of a unitary circuit on a given number of qubits."""
    tableau = circuit.to_tableau()
    return tableau + stim.Tableau(num_qubits - len(tableau))


def _count(circuit: stim.Circuit, name: str) -> int:
    """The number of two-qubit gates with a given name in a circuit."""
    return sum(len(op.targets_copy()) // 2 for op in circuit.flattened() if op.name == name)
