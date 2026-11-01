# SPDX-License-Identifier: Apache-2.0

"""Unit tests for cz_transpilation.py."""

from __future__ import annotations

import itertools
import random

import pytest
import stim

from qldpc.circuits.transforms import transpile_to_cz

_SINGLE_QUBIT_GATES = sorted(
    name for name, data in stim.gate_data().items() if data.is_unitary and data.is_single_qubit_gate
)
_TWO_QUBIT_GATES = sorted(
    name for name, data in stim.gate_data().items() if data.is_unitary and data.is_two_qubit_gate
)
_OUTPUT_NAMES = {
    *_SINGLE_QUBIT_GATES,
    *("CZ", "SWAP", "M", "MX", "MY", "MR", "MRX", "MRY", "R", "RX", "RY", "MPP"),
    *("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"),
}


def test_two_qubit_gates() -> None:
    """Each two-qubit gate becomes at most one CZ, single-qubit gates, and a final relabeling."""
    for name in _TWO_QUBIT_GATES:
        circuit = stim.Circuit(f"{name} 0 1")
        transpiled = transpile_to_cz(circuit)
        assert _get_tableau(transpiled, 2) == circuit.to_tableau(), name
        assert _count(transpiled, "CZ") == int(name not in ("II", "SWAP")), name
        # SWAPs come only at the end, where they return the states to their qubits
        names = [op.name for op in transpiled]
        assert "SWAP" not in names or "CZ" not in names[names.index("SWAP") :], name


def test_random_circuits(pytestconfig: pytest.Config) -> None:
    """Random circuits are equivalent to their transpiled versions."""
    rng = random.Random(pytestconfig.getoption("randomly_seed"))
    for _ in range(100):
        num_qubits = rng.randint(2, 4)
        circuit = _get_random_circuit(rng, num_qubits, rng.randint(1, 20), allow_repeat=True)
        transpiled = transpile_to_cz(circuit)
        _assert_equivalent(circuit, transpiled)
        assert {op.name for op in transpiled.flattened()} <= _OUTPUT_NAMES | {"CX"}


def test_squashing() -> None:
    """Single-qubit gates squash, move past CZ gates when they can, and CZ pairs cancel."""
    # gates that map Z to +Z or -Z move past CZ gates, so that no gate comes before the first CZ
    circuit = stim.Circuit("S 0\nX 1\nH 1\nH 1\nCZ 0 1\nCZ 1 2\nH 0")
    transpiled = transpile_to_cz(circuit)
    assert _get_tableau(transpiled, 3) == circuit.to_tableau()
    assert transpiled[0].name == "CZ"

    # other gates leave H or SQRT_X before each CZ, which comes right before the CZ
    circuit = stim.Circuit("S 0\nH 0\nCZ 1 2\nCZ 2 3\nCZ 0 1\nC_XYZ 0\nCZ 0 1")
    transpiled = transpile_to_cz(circuit)
    assert _get_tableau(transpiled, 4) == circuit.to_tableau()
    gates_before_czs = [
        op.name
        for op, next_op in itertools.pairwise(transpiled)
        if next_op.name == "CZ" and op.name != "CZ"
    ]
    assert set(gates_before_czs) <= {"H", "SQRT_X"}
    assert str(transpiled).startswith("CZ 1 2\nSQRT_X 0\nCZ 2 3 0 1")

    # consecutive CZ gates on the same qubits cancel, as do the gates around them
    assert transpile_to_cz(stim.Circuit("CX 0 1\nCX 0 1")) == stim.Circuit()
    assert transpile_to_cz(stim.Circuit("H 0\nCZ 0 1\nCZ 1 0\nCZ 0 1\nH 0")) == stim.Circuit(
        "H 0\nCZ 0 1\nH 0"
    )
    assert transpile_to_cz(stim.Circuit("CZ 0 1\nCX 2 3\nCZ 0 1")) == stim.Circuit(
        "H 3\nCZ 2 3\nH 3"
    )
    # but not if something reads or acts on their qubits between them
    circuit = stim.Circuit("CZ 0 1\nOBSERVABLE_INCLUDE(0) X0\nCZ 0 1")
    assert transpile_to_cz(circuit) == circuit


def test_measurements_and_resets() -> None:
    """Pending gates merge into measurements, observables, and resets."""
    circuit = stim.Circuit("""
        H 0
        M !0
        MR 1
        H 1
        MXX 0 !1
        OBSERVABLE_INCLUDE(0) X0 rec[-1]
        R 2
        H 2
        S 2
        S 2
        CZ 2 0
    """)
    transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    assert transpiled == stim.Circuit("""
        MX !0
        MR 1
        MPP !Z0*Z1
        OBSERVABLE_INCLUDE(0) Z0 rec[-1]
        RX 2
        H 0
        CZ 2 0
        H 1
        Z 2
    """)

    # a measure-reset that changes its basis prepares the reset state of its original basis
    circuit = stim.Circuit("H 0\nMR 0\nCZ 0 1")
    transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    assert transpiled[0] == stim.CircuitInstruction("MRX", [0])


def test_tagged_instructions() -> None:
    """Tagged instructions are opaque, except that their qubit targets are relabeled."""
    circuit = stim.Circuit("""
        SWAP 0 1
        S 0
        H[t] 0
        X 1
        R[t] 1
        CX[t] 0 1
        DETECTOR[t](1) rec[-1]
    """)
    transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    # the pending S comes before the tagged H, and the tagged reset discards the pending X
    assert transpiled == stim.Circuit("""
        S 1
        H[t] 1
        R[t] 0
        CX[t] 1 0
        DETECTOR[t](1) rec[-1]
        SWAP 1 0
    """)


def test_annotations() -> None:
    """TICK instructions are dropped, and annotations are kept."""
    circuit = stim.Circuit("""
        QUBIT_COORDS(1, 2) 0
        TICK
        M 0
        DETECTOR(0) rec[-1]
        SHIFT_COORDS(1)
        MPAD 1
    """)
    assert transpile_to_cz(circuit) == stim.Circuit("""
        QUBIT_COORDS(1, 2) 0
        M 0
        DETECTOR(0) rec[-1]
        SHIFT_COORDS(1)
        MPAD 1
    """)


def test_repeat_blocks() -> None:
    """REPEAT blocks are kept, with their first iteration unrolled if that makes them periodic."""
    # the state at the start of each iteration is the same
    circuit = stim.Circuit("REPEAT 3 {\nCX 0 1\nH 1\nM 1\n}")
    transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    assert _get_repeat_counts(transpiled) == [3]

    # a REPEAT block with one iteration always stays
    circuit = stim.Circuit("REPEAT 1 {\nCX 0 1\nM 1\n}")
    transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    assert _get_repeat_counts(transpiled) == [1]

    # the state at the start of each iteration is the same after the first iteration
    circuit = stim.Circuit("REPEAT[t] 3 {\nCX 0 1\nM 1\n}")
    transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    assert _get_repeat_counts(transpiled) == [2]
    assert _count(transpiled, "CZ") == 3
    assert transpiled[-2].tag == "t"

    # pending gates and qubit labels alternate between iterations, so each iteration restores them
    circuit = stim.Circuit("REPEAT 3 {\nX 0\nCZ 0 1\nSWAP 1 2\nM 2\n}")
    transpiled = transpile_to_cz(circuit)
    _assert_equivalent(circuit, transpiled)
    assert _get_repeat_counts(transpiled) == [3]
    assert _count(transpiled[0].body_copy(), "SWAP") == 1


def test_unsupported_instructions() -> None:
    """Noise, classically controlled gates, and untagged SPP gates are rejected."""
    for circuit in [
        "X_ERROR(0.1) 0",
        "X_ERROR[t](0.1) 0",
        "M(0.1) 0",
        "HERALDED_ERASE(0.1) 0",
    ]:
        with pytest.raises(ValueError, match="Noisy"):
            transpile_to_cz(stim.Circuit(circuit))
    with pytest.raises(ValueError, match="Classically controlled"):
        transpile_to_cz(stim.Circuit("M 0\nCX rec[-1] 1"))
    with pytest.raises(ValueError, match="not supported"):
        transpile_to_cz(stim.Circuit("SPP X0*Z1"))

    # tagged instructions are opaque, whatever they are
    circuit = stim.Circuit("M 0\nCX[t] rec[-1] 1\nSPP[t] X0*Z1")
    assert transpile_to_cz(circuit) == circuit


def _assert_equivalent(circuit: stim.Circuit, transpiled: stim.Circuit) -> None:
    """Assert that two noiseless circuits have the same stabilizer flows.

    Before comparing flows, this function strips tags, replaces observables with Pauli targets by
    MPP measurements of their Pauli products, and splits measure-resets into measurements and
    resets, which stim.Circuit.flow_generators mishandles (as of stim 1.16).
    """
    circuit, transpiled = _prepare_for_flows(circuit), _prepare_for_flows(transpiled)
    assert all(transpiled.has_flow(flow) for flow in circuit.flow_generators())
    assert all(circuit.has_flow(flow) for flow in transpiled.flow_generators())


def _prepare_for_flows(circuit: stim.Circuit) -> stim.Circuit:
    """Prepare a circuit for a comparison of flows (see _assert_equivalent)."""
    prepared = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            body = _prepare_for_flows(op.body_copy())
            prepared.append(stim.CircuitRepeatBlock(op.repeat_count, body))
        elif op.name == "OBSERVABLE_INCLUDE":
            paulis = [
                target for target in op.targets_copy() if not target.is_measurement_record_target
            ]
            invert = sum(target.is_inverted_result_target for target in paulis) % 2 == 1
            targets = []
            for index, target in enumerate(paulis):
                if index:
                    targets.append(stim.target_combiner())
                targets.append(
                    stim.target_pauli(target.value, target.pauli_type, invert and not index)
                )
            if targets:
                prepared.append("MPP", targets)
        elif op.name in ("MR", "MRX", "MRY"):
            for target in op.targets_copy():
                prepared.append(op.name.replace("R", ""), [target])
                prepared.append(op.name[1:], [target.value])
        else:
            prepared.append(op.name, op.targets_copy(), op.gate_args_copy())
    return prepared


def _get_random_circuit(
    rng: random.Random, num_qubits: int, num_ops: int, allow_repeat: bool
) -> stim.Circuit:
    """A random noiseless circuit."""
    qubits = range(num_qubits)
    circuit = stim.Circuit()
    for _ in range(num_ops):
        kind = rng.random()
        if kind < 0.3:
            circuit.append(rng.choice(_SINGLE_QUBIT_GATES), rng.sample(qubits, rng.randint(1, 2)))
        elif kind < 0.65:
            circuit.append(rng.choice(_TWO_QUBIT_GATES), rng.sample(qubits, 2))
        elif kind < 0.75:
            name = rng.choice(["M", "MX", "MY", "MR", "MRX", "MRY"])
            circuit.append(name, [stim.GateTarget(rng.choice(qubits)), stim.target_inv(0)])
        elif kind < 0.8:
            circuit.append(rng.choice(["R", "RX", "RY"]), [rng.choice(qubits)])
        elif kind < 0.85:
            name = rng.choice(["MXX", "MYY", "MZZ"])
            qubit_0, qubit_1 = rng.sample(qubits, 2)
            circuit.append(name, [stim.target_inv(qubit_0), stim.GateTarget(qubit_1)])
        elif kind < 0.9:
            string = stim.PauliString.random(num_qubits)
            if string.weight:
                circuit.append(
                    "MPP", stim.target_combined_paulis(string, invert=rng.random() < 0.5)
                )
        elif kind < 0.93:
            paulis = [
                stim.target_pauli(qubit, rng.choice("XYZ")) for qubit in rng.sample(qubits, 2)
            ]
            circuit.append("OBSERVABLE_INCLUDE", [*paulis, stim.target_rec(-1)], 0)
        elif kind < 0.96:
            name, num_targets = rng.choice([("H", 1), ("CX", 2), ("M", 1), ("RX", 1)])
            targets = rng.sample(qubits, num_targets)
            circuit.append(stim.CircuitInstruction(name, targets, tag="t"))
        elif allow_repeat:
            body = _get_random_circuit(rng, num_qubits, rng.randint(1, 6), allow_repeat=False)
            circuit.append(stim.CircuitRepeatBlock(rng.randint(1, 4), body))
    return circuit


def _get_tableau(circuit: stim.Circuit, num_qubits: int) -> stim.Tableau:
    """The tableau of a unitary circuit on a given number of qubits."""
    tableau = circuit.to_tableau()
    return tableau + stim.Tableau(num_qubits - len(tableau))


def _count(circuit: stim.Circuit, name: str) -> int:
    """The number of gates with a given name in a circuit, with REPEAT blocks unrolled."""
    data = stim.gate_data(name)
    return sum(
        len(op.targets_copy()) // (2 if data.is_two_qubit_gate else 1)
        for op in circuit.flattened()
        if op.name == name
    )


def _get_repeat_counts(circuit: stim.Circuit) -> list[int]:
    """The repetition counts of the top-level REPEAT blocks of a circuit."""
    return [op.repeat_count for op in circuit if isinstance(op, stim.CircuitRepeatBlock)]
