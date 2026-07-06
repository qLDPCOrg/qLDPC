"""Unit tests for noise_model.py

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import stim
import tsim

from qldpc import circuits


def _circuits_are_equivalent(
    circuit_a: stim.Circuit, circuit_b: stim.Circuit, atol: float = 1e-10
) -> bool:
    """Test equivalence between circuits after some standardization."""
    trivial_noise_model = circuits.NoiseModel()
    circuit_a = trivial_noise_model.noisy_circuit(circuit_a)
    circuit_b = trivial_noise_model.noisy_circuit(circuit_b)
    return circuit_a.approx_equals(circuit_b, atol=atol)


def test_gate_errors() -> None:
    """Add gate errors to a circuit."""

    # ordinary gate errors
    circuit = stim.Circuit("""
        H 0
        CX 0 1 1 2
        TICK
        M 0
        RX 1
        MR 2
    """)
    noise_model = circuits.NoiseModel(
        clifford_1q_error=0.1, clifford_2q_error=0.2, readout_error=0.3, reset_error=0.4
    )
    noisy_circuit = stim.Circuit("""
        H 0
        DEPOLARIZE1(0.1) 0
        CX 0 1
        DEPOLARIZE2(0.2) 0 1
        CX 1 2
        DEPOLARIZE2(0.2) 1 2
        TICK
        MZ(0.3) 0
        RX 1
        MR(0.3) 2
        Z_ERROR(0.4) 1
        X_ERROR(0.4) 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # multiple errors after one gate
    circuit = stim.Circuit("""
        CX 0 1
    """)
    noise_rule = circuits.NoiseRule(after={"DEPOLARIZE2": 0.2, "PAULI_CHANNEL_1": [0, 0.1, 0.1]})
    noise_model = circuits.NoiseModel(rules={"CX": noise_rule})
    noisy_circuit = stim.Circuit("""
        CX 0 1
        DEPOLARIZE2(0.2) 0 1
        PAULI_CHANNEL_1(0, 0.1, 0.1) 0 1
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # compose gate errors
    p_m = 0.1
    double_p_m = 1 - (1 - p_m) ** 2
    noise_model = circuits.NoiseModel(readout_error=p_m)
    circuit = stim.Circuit("""
        H 0
        M 0
    """)
    noisy_circuit = stim.Circuit(f"""
        H 0
        MZ({p_m}) 0
    """)
    double_noisy_circuit = stim.Circuit(f"""
        H 0
        MZ({double_p_m}) 0
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
    assert _circuits_are_equivalent(double_noisy_circuit, noise_model.noisy_circuit(noisy_circuit))

    # reusing a qubit in the same moment raises an error
    circuit = stim.Circuit("""
        CX 0 1 1 2
    """)
    noise_model = circuits.SI1000NoiseModel(0.1)
    with pytest.raises(ValueError, match="multiple uses"):
        noise_model.noisy_circuit(circuit, insert_ticks=False)


def test_idle_errors() -> None:
    """Add idling errors to a circuit."""

    circuit = stim.Circuit("""
        H 0 1 2
        H 1
        M 0
        DETECTOR rec[-1]
    """)
    noise_model = circuits.NoiseModel(
        readout_error=0.1, idle_error=0.2, additional_error_waiting_for_m_or_r=0.3
    )
    noisy_circuit = stim.Circuit("""
        H 0 1 2
        H 1
        M(0.1) 0
        DETECTOR rec[-1]
        DEPOLARIZE1(0.2) 2
        DEPOLARIZE1(0.3) 1 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # user-defined idling noise via NoiseRule (overrides the default DEPOLARIZE1 channel)
    idle_rule = circuits.NoiseRule(after={"PAULI_CHANNEL_1": [0.05, 0.0, 0.1]})
    m_or_r_rule = circuits.NoiseRule(after={"X_ERROR": 0.2, "Z_ERROR": 0.3})
    noise_model = circuits.NoiseModel(
        readout_error=0.1, idle_error=idle_rule, additional_error_waiting_for_m_or_r=m_or_r_rule
    )
    noisy_circuit = stim.Circuit("""
        H 0 1 2
        H 1
        M(0.1) 0
        DETECTOR rec[-1]
        PAULI_CHANNEL_1(0.05, 0.0, 0.1) 2
        X_ERROR(0.2) 1 2
        Z_ERROR(0.3) 1 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # zero/None errors normalize to None for all NoiseRule-valued fields
    noise_model = circuits.NoiseModel(
        clifford_1q_error=0,
        clifford_2q_error=0,
        idle_error=0,
        additional_error_waiting_for_m_or_r=None,
    )
    assert noise_model.clifford_1q_error is None
    assert noise_model.clifford_2q_error is None
    assert noise_model.idle_error is None
    assert noise_model.additional_error_waiting_for_m_or_r is None
    assert not bool(noise_model)


def test_immunity() -> None:
    """Qubits and operations can be immune to errors."""

    # qubits can be immune to errors
    circuit = stim.Circuit("""
        H 0 1
        CNOT 1 2
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False)
    noisy_circuit = stim.Circuit("""
        H 0 1
        CNOT 1 2
        DEPOLARIZE1(0.1) 1
        DEPOLARIZE2(0.1) 1 2
    """)
    assert _circuits_are_equivalent(
        noisy_circuit, noise_model.noisy_circuit(circuit, immune_qubits=[0], insert_ticks=False)
    )

    with pytest.raises(ValueError, match="does not support immune qubits"):
        assert _circuits_are_equivalent(
            noisy_circuit, noise_model.noisy_circuit(circuit, immune_qubits=[0], insert_ticks=True)
        )

    # operations can be immune to errors
    immune_op_tag = "_TEST_"
    circuit = stim.Circuit(f"""
        H['{immune_op_tag}'] 0
        CX 0 1
        H['{immune_op_tag}'] 0
        X 1
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=True)
    noisy_circuit = stim.Circuit(f"""
        H['{immune_op_tag}'] 0
        CX 0 1
        DEPOLARIZE2(0.1) 0 1
        H['{immune_op_tag}'] 0
        X 1
        DEPOLARIZE1(0.1) 1
    """)
    assert _circuits_are_equivalent(
        noisy_circuit, noise_model.noisy_circuit(circuit, immune_op_tag=immune_op_tag)
    )

    # circuits can be made immune to errors
    tableau = stim.Tableau.random(5)
    noiseless_circuit = circuits.as_noiseless_circuit(tableau.to_circuit())
    assert noise_model.noisy_circuit(noiseless_circuit).to_tableau() == tableau


def test_immune_project() -> None:
    noise_model: circuits.NoiseModel
    circuit = stim.Circuit("""
        CNOT 0 1
        CNOT 1 2
        CNOT 3 4
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False)

    # error project on part of DEPOLARIZE2 on qubit 0 1
    noisy_circuit_project = stim.Circuit("""
        CNOT 0 1
        CNOT 1 2
        CNOT 3 4
        DEPOLARIZE1(0.02) 2
        DEPOLARIZE2(0.1) 3 4
    """)
    assert _circuits_are_equivalent(
        noisy_circuit_project,
        noise_model.noisy_circuit(circuit, immune_qubits=[0, 1], insert_ticks=False, project=True),
    )

    # turn off projection
    noisy_circuit_project = stim.Circuit("""
        CNOT 0 1
        CNOT 1 2
        CNOT 3 4
        DEPOLARIZE2(0.1) 3 4
    """)
    assert _circuits_are_equivalent(
        noisy_circuit_project,
        noise_model.noisy_circuit(circuit, immune_qubits=[0, 1], insert_ticks=False, project=False),
    )

    # test PAULI_CHANNEL_2
    circuit = stim.Circuit("""
        CNOT 0 1
    """)
    noise_model = circuits.NoiseModel(
        clifford_2q_error=circuits.NoiseRule(
            after={
                "PAULI_CHANNEL_2": [0.01, 0.02, 0.03, 0.04, 0, 0, 0, 0.05, 0, 0, 0, 0.06, 0, 0, 0]
            }
        )
    )

    assert _circuits_are_equivalent(
        stim.Circuit("""
            CNOT 0 1
            PAULI_CHANNEL_1(0.01, 0.02, 0.03) 1
        """),
        noise_model.noisy_circuit(circuit, immune_qubits=[0], insert_ticks=False, project=True),
    )

    assert _circuits_are_equivalent(
        stim.Circuit("""
            CNOT 0 1
            PAULI_CHANNEL_1(0.04, 0.05, 0.06) 0
        """),
        noise_model.noisy_circuit(circuit, immune_qubits=[1], insert_ticks=False, project=True),
    )

    # multi-qubit PauliChannel projection: keep only strings that are I on the immune qubit
    circuit = stim.Circuit("SPP X0*Y1*Z2")
    channel = circuits.PauliChannel({"XYZ": 0.01, "XIZ": 0.02, "IZI": 0.03})
    noise_model = circuits.NoiseModel(
        rules={"SPP": circuits.NoiseRule(after_pauli_channel=channel)}
    )

    # project=True, immune qubit 1: only "XIZ" survives; projected to positions [0, 2].
    # The projected 2-qubit channel is emitted natively as PAULI_CHANNEL_2 with XZ at index 6.
    assert _circuits_are_equivalent(
        stim.Circuit("""
            SPP X0*Y1*Z2
            PAULI_CHANNEL_2(0, 0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0, 0) 0 2
        """),
        noise_model.noisy_circuit(circuit, immune_qubits=[1], insert_ticks=False, project=True),
    )

    # project=False, immune qubit 1: drop the whole channel.
    assert _circuits_are_equivalent(
        stim.Circuit("SPP X0*Y1*Z2"),
        noise_model.noisy_circuit(circuit, immune_qubits=[1], insert_ticks=False, project=False),
    )

    # project=True but every string has a non-I on the immune qubit -> nothing emitted.
    empty_marg_channel = circuits.PauliChannel({"XYZ": 0.01, "IYY": 0.02})
    noise_model = circuits.NoiseModel(
        rules={"SPP": circuits.NoiseRule(after_pauli_channel=empty_marg_channel)}
    )
    assert _circuits_are_equivalent(
        stim.Circuit("SPP X0*Y1*Z2"),
        noise_model.noisy_circuit(circuit, immune_qubits=[1], insert_ticks=False, project=True),
    )

    # 4-qubit channel projecting one qubit -> 3-qubit surviving channel, emitted as CE chain.
    circuit = stim.Circuit("SPP X0*Y1*Z2*X3")
    channel = circuits.PauliChannel({"XIZX": 0.01, "IIYX": 0.02, "XYZX": 0.03})
    noise_model = circuits.NoiseModel(
        rules={"SPP": circuits.NoiseRule(after_pauli_channel=channel)}
    )
    # Immune qubit 1: only "XIZX" and "IIYX" have I at pos 1.  Projected onto positions
    # [0, 2, 3] -> {"IYX": 0.02, "XZX": 0.01} (lex-sorted).  Emitted as a CE chain on qubits 0, 2, 3
    # with the second entry renormalized to conditional-fire prob = 0.01 / (1 - 0.02).
    assert _circuits_are_equivalent(
        stim.Circuit(f"""
            SPP X0*Y1*Z2*X3
            CORRELATED_ERROR(0.02) Y2 X3
            ELSE_CORRELATED_ERROR({0.01 / (1 - 0.02)}) X0 Z2 X3
        """),
        noise_model.noisy_circuit(circuit, immune_qubits=[1], insert_ticks=False, project=True),
    )

    # `after` broadcast noise: 1q entries drop immune targets; 2q entries drop / project pairs.
    circuit = stim.Circuit("""
        H 0
        H 1
        TICK
        CX 0 1 2 3
    """)
    noise_model = circuits.NoiseModel(
        clifford_1q_error=circuits.NoiseRule(after={"DEPOLARIZE1": 0.1}),
        clifford_2q_error=circuits.NoiseRule(after={"DEPOLARIZE2": 0.2}),
    )
    # immune_qubits=[1] with project=False: DEPOLARIZE1 keeps only qubit 0, DEPOLARIZE2 pair
    # (0, 1) dropped, pair (2, 3) kept.  With project=True: pair (0, 1) becomes
    # DEPOLARIZE1(0.2/5) on qubit 0.
    assert _circuits_are_equivalent(
        stim.Circuit("""
            H 0
            H 1
            DEPOLARIZE1(0.1) 0
            TICK
            CX 0 1 2 3
            DEPOLARIZE2(0.2) 2 3
        """),
        noise_model.noisy_circuit(circuit, immune_qubits=[1], insert_ticks=False, project=False),
    )
    assert _circuits_are_equivalent(
        stim.Circuit("""
            H 0
            H 1
            DEPOLARIZE1(0.1) 0
            TICK
            CX 0 1 2 3
            DEPOLARIZE1(0.04) 0
            DEPOLARIZE2(0.2) 2 3
        """),
        noise_model.noisy_circuit(circuit, immune_qubits=[1], insert_ticks=False, project=True),
    )

    # reset_error: emitted as X_ERROR / Z_ERROR broadcast on all targets; immune targets are
    # dropped by the moment-level filter regardless of `project`.
    circuit = stim.Circuit("R 0 1 2")
    noise_model = circuits.NoiseModel(reset_error=0.1)
    assert _circuits_are_equivalent(
        stim.Circuit("""
            R 0 1 2
            X_ERROR(0.1) 0 2
        """),
        noise_model.noisy_circuit(circuit, immune_qubits=[1], insert_ticks=False),
    )

    # readout_error: dropped only when *every* measured qubit is immune.  A measurement whose
    # classical result depends on any non-immune qubit keeps its readout error unchanged.
    circuit = stim.Circuit("""
        M 0
        TICK
        M 1
        TICK
        MPP X0*Y1
    """)
    noise_model = circuits.NoiseModel(readout_error=0.1)
    assert _circuits_are_equivalent(
        stim.Circuit("""
            M(0.1) 0
            TICK
            M 1
            TICK
            MPP(0.1) X0*Y1
        """),
        noise_model.noisy_circuit(circuit, immune_qubits=[1], insert_ticks=False),
    )


def test_classical_controls() -> None:
    """Classically controled gates get special treatment."""
    noise_model: circuits.NoiseModel

    # classically controls are immune to noise, but the qubits still pick up idling errors
    circuit = stim.Circuit("""
        CX 0 1 rec[-1] 2
    """)
    noise_model = circuits.SI1000NoiseModel(0.1)
    noisy_circuit = stim.Circuit("""
        CX 0 1 rec[-1] 2
        DEPOLARIZE2(0.1) 0 1
        DEPOLARIZE1(0.01) 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # qubits addressed by classical controls pick up idling errors by default
    circuit = stim.Circuit("""
        H 0
        CX rec[-1] 1
        TICK
        H 0 1 2
    """)
    noise_model = circuits.NoiseModel(idle_error=0.1)
    noisy_circuit = stim.Circuit("""
        H 0
        CX rec[-1] 1
        DEPOLARIZE1(0.1) 1 2
        TICK
        H 0 1 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))


def test_pauli_product_measurements() -> None:
    """Pauli product measurements get special treatment."""

    circuit = stim.Circuit("""
        MPP X1*Y2*Z3
    """)
    noise_model = circuits.NoiseModel(readout_error=0.1, idle_error=0.2)
    noisy_circuit = stim.Circuit("""
        MPP(0.1) X1*Y2*Z3
        DEPOLARIZE1(0.2) 0
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # override the default MPP rule for specific Pauli products
    circuit = stim.Circuit("""
        MPP Z0*Z1*Z2
        MPP X0*Y1*Z2
    """)
    noise_rule = circuits.NoiseRule(readout_error=0.2)
    noise_model = circuits.NoiseModel(readout_error=0.1, rules={"MXYZ": noise_rule})
    noisy_circuit = stim.Circuit("""
        MPP(0.1) Z0*Z1*Z2
        MPP(0.2) X0*Y1*Z2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))


def test_pauli_product_cliffords() -> None:
    """SPP gates on 1 or 2 qubits get, respectively, 1q or 2q noise."""

    # SPP on one qubit -> clifford_1q_error; SPP on two qubits -> clifford_2q_error;
    # SPP on three or more qubits is ignored by default.
    circuit = stim.Circuit("""
        SPP X0
        TICK
        SPP X0*Y1
        TICK
        SPP_DAG X0*Y1*Z2
    """)
    noise_model = circuits.NoiseModel(clifford_1q_error=0.1, clifford_2q_error=0.2)
    noisy_circuit = stim.Circuit("""
        SPP X0
        DEPOLARIZE1(0.1) 0
        TICK
        SPP X0*Y1
        DEPOLARIZE2(0.2) 0 1
        TICK
        SPP_DAG X0*Y1*Z2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # multi-product SPP is split so each Pauli product is treated independently
    circuit = stim.Circuit("""
        SPP X0 Y1*Z2 X3*Y4*Z5
    """)
    noisy_circuit = stim.Circuit("""
        SPP X0 Y1*Z2 X3*Y4*Z5
        DEPOLARIZE1(0.1) 0
        DEPOLARIZE2(0.2) 1 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # explicit rules dict overrides the default weight-based dispatch, including for weight >= 3
    noise_rule = circuits.NoiseRule(after={"DEPOLARIZE1": 0.3})
    noise_model = circuits.NoiseModel(clifford_1q_error=0.1, rules={"SPP": noise_rule})
    circuit = stim.Circuit("""
        SPP X0
        TICK
        SPP X0*Y1*Z2
    """)
    noisy_circuit = stim.Circuit("""
        SPP X0
        DEPOLARIZE1(0.3) 0
        TICK
        SPP X0*Y1*Z2
        DEPOLARIZE1(0.3) 0 1 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))


def test_pauli_channel_class() -> None:
    """PauliChannel construction, validation, and helpers."""

    ch = circuits.PauliChannel({"XYZ": 0.01, "ZZZ": 0.02})
    assert ch.num_qubits == 3
    assert dict(ch.probabilities) == {"XYZ": 0.01, "ZZZ": 0.02}
    assert bool(ch)
    assert ch == circuits.PauliChannel({"XYZ": 0.01, "ZZZ": 0.02})
    assert ch != circuits.PauliChannel({"XYZ": 0.01})
    assert ch != "foo"  # __eq__ returns NotImplemented for non-PauliChannel operands
    assert repr(ch).startswith("PauliChannel(")

    # zero-prob channel is falsy but still constructible
    assert not bool(circuits.PauliChannel({"X": 0.0}))

    # depolarizing(n, p): 4**n - 1 entries, probabilities sum to p, first key in lex order is "IX"
    dp = circuits.PauliChannel.depolarizing(2, 0.15)
    assert dp.num_qubits == 2
    assert len(dp.probabilities) == 15
    assert abs(sum(dp.probabilities.values()) - 0.15) < 1e-12
    assert next(iter(dp.probabilities.keys())) == "IX"

    # empty channel is the trivial 0-qubit channel
    empty = circuits.PauliChannel({})
    assert empty.num_qubits == 0
    assert dict(empty.probabilities) == {}
    assert not bool(empty)

    # validation errors
    with pytest.raises(ValueError, match="Identity string"):
        circuits.PauliChannel({"": 0.1})
    with pytest.raises(ValueError, match="must have length"):
        circuits.PauliChannel({"XY": 0.1, "X": 0.1})
    with pytest.raises(ValueError, match="invalid characters"):
        circuits.PauliChannel({"W": 0.1})
    with pytest.raises(ValueError, match="Identity string"):
        circuits.PauliChannel({"II": 0.1})
    with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
        circuits.PauliChannel({"X": 1.5})
    with pytest.raises(ValueError, match="Sum of Pauli channel"):
        circuits.PauliChannel({"X": 0.7, "Y": 0.7})
    with pytest.raises(ValueError, match="num_qubits=0"):
        circuits.PauliChannel.depolarizing(0, 0.1)
    with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
        circuits.PauliChannel.depolarizing(2, 1.5)


def test_pauli_channel_project() -> None:
    """PauliChannel.project projects the channel onto identity-on-immune subspace."""

    channel = circuits.PauliChannel({"XYZ": 0.01, "XIZ": 0.02, "IZI": 0.03, "XII": 0.04})

    # Marginalize position 1 (middle): keep strings with I at pos 1.
    # XYZ: pos 1 is Y, drop.  XIZ: pos 1 is I, keep -> "XZ".
    # IZI: pos 1 is Z, drop.  XII: pos 1 is I, keep -> "XI".
    assert channel.project([1]) == circuits.PauliChannel({"XZ": 0.02, "XI": 0.04})

    # Marginalize positions 0 and 2: keep strings with I at pos 0 AND pos 2.
    # Only "IZI" has I at 0 and 2 -> "Z" on surviving position 1.
    assert channel.project([0, 2]) == circuits.PauliChannel({"Z": 0.03})

    # Empty index list is a no-op (returns an equal channel).
    assert channel.project([]) == channel

    # Marginalizing every position yields an empty channel (no non-identity string is all-I).
    assert channel.project([0, 1, 2]) == circuits.PauliChannel({})

    # Repeated indices are deduplicated.
    assert channel.project([1, 1]) == channel.project([1])

    # If nothing survives, the result is empty.
    all_non_id = circuits.PauliChannel({"XY": 0.1, "YX": 0.1})
    assert all_non_id.project([0]) == circuits.PauliChannel({})

    # Out-of-range indices raise.
    with pytest.raises(ValueError, match="not in"):
        channel.project([3])
    with pytest.raises(ValueError, match="not in"):
        channel.project([-1])


def test_multi_qubit_pauli_channel_after_gate() -> None:
    """NoiseRule.after_pauli_channel emits a CORRELATED_ERROR / ELSE_CORRELATED_ERROR chain."""

    # A sparse 3-qubit Pauli channel applied via rules={"SPP": ...}
    channel = circuits.PauliChannel({"XYZ": 0.01, "ZZZ": 0.02})
    rule = circuits.NoiseRule(after_pauli_channel=channel)
    noise_model = circuits.NoiseModel(rules={"SPP": rule})
    circuit = stim.Circuit("""
        SPP X0*Y1*Z2
    """)
    # Marginals: XYZ at 0.01, ZZZ at 0.02.  ELSE renormalized: 0.02 / (1 - 0.01).
    noisy_circuit = stim.Circuit("""
        SPP X0*Y1*Z2
        CORRELATED_ERROR(0.01) X0 Y1 Z2
        ELSE_CORRELATED_ERROR(0.020202020202020204) Z0 Z1 Z2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # 2-qubit channels emit a native PAULI_CHANNEL_2 (args in stim's order:
    # IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ)
    channel = circuits.PauliChannel({"XI": 0.0, "IX": 0.05, "XX": 0.05})
    rule = circuits.NoiseRule(after_pauli_channel=channel)
    noise_model = circuits.NoiseModel(rules={"SPP": rule})
    circuit = stim.Circuit("SPP X0*Y1")
    noisy_circuit = stim.Circuit("""
        SPP X0*Y1
        PAULI_CHANNEL_2(0.05, 0, 0, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) 0 1
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # A raw dict is auto-wrapped into a PauliChannel
    rule = circuits.NoiseRule(after_pauli_channel={"XY": 0.01})
    assert isinstance(rule.after_pauli_channel, circuits.PauliChannel)


def test_clifford_nq_error() -> None:
    """clifford_nq_error dispatches by qubit count k."""

    # A float value produces a uniform depolarizing channel of that weight
    noise_model = circuits.NoiseModel(clifford_nq_error={3: 0.001})
    circuit = stim.Circuit("SPP X0*Y1*Z2")
    noisy = noise_model.noisy_circuit(circuit)
    # 63 non-identity 3q Paulis: 1 CORRELATED_ERROR (canonicalized by stim to "E") plus
    # 62 ELSE_CORRELATED_ERROR
    ce_count = sum(1 for op in noisy if op.name == "E")
    else_ce_count = sum(1 for op in noisy if op.name == "ELSE_CORRELATED_ERROR")
    assert ce_count == 1 and else_ce_count == 62

    # multi-product SPP: each product is routed to its weight's rule.  Weight-1 and weight-2
    # channels emit native PAULI_CHANNEL_1 / PAULI_CHANNEL_2; weight-3 and above use a
    # CORRELATED_ERROR chain.  Values may be a raw Mapping (auto-wrapped) or a PauliChannel.
    noise_model = circuits.NoiseModel(
        clifford_nq_error={
            1: {"X": 0.01},  # raw Mapping — auto-wrapped as PauliChannel
            2: circuits.PauliChannel({"XY": 0.02}),
            3: circuits.PauliChannel({"XYZ": 0.03}),
        }
    )
    circuit = stim.Circuit("SPP X0 Y1*Z2 X3*Y4*Z5")
    noisy = noise_model.noisy_circuit(circuit)
    expected = stim.Circuit("""
        SPP X0 Y1*Z2 X3*Y4*Z5
        PAULI_CHANNEL_1(0.01, 0, 0) 0
        PAULI_CHANNEL_2(0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0) 1 2
        CORRELATED_ERROR(0.03) X3 Y4 Z5
    """)
    assert _circuits_are_equivalent(expected, noisy)

    # A NoiseRule value is used directly (allowing e.g. combined channels)
    rule = circuits.NoiseRule(
        after={"X_ERROR": 0.01},
        after_pauli_channel=circuits.PauliChannel({"XYZ": 0.02}),
    )
    noise_model = circuits.NoiseModel(clifford_nq_error={3: rule})
    circuit = stim.Circuit("SPP X0*Y1*Z2")
    noisy = noise_model.noisy_circuit(circuit)
    expected = stim.Circuit("""
        SPP X0*Y1*Z2
        X_ERROR(0.01) 0 1 2
        CORRELATED_ERROR(0.02) X0 Y1 Z2
    """)
    assert _circuits_are_equivalent(expected, noisy)

    # Falsy values (zero float, empty NoiseRule) are dropped from the normalized dict
    assert circuits.NoiseModel(clifford_nq_error={3: 0.0}).clifford_nq_error == {}
    assert circuits.NoiseModel(clifford_nq_error={3: circuits.NoiseRule()}).clifford_nq_error == {}


def test_clifford_nq_error_errors() -> None:
    """Validation errors around clifford_nq_error and related rules."""

    # Ambiguity: pp[k] and clifford_kq_error both user-specified (symmetric — either side raises,
    # even when either value is a no-op zero).
    with pytest.raises(ValueError, match="Ambiguous"):
        circuits.NoiseModel(clifford_1q_error=0.01, clifford_nq_error={1: 0.02})
    with pytest.raises(ValueError, match="Ambiguous"):
        circuits.NoiseModel(clifford_2q_error=0.01, clifford_nq_error={2: 0.02})
    with pytest.raises(ValueError, match="Ambiguous"):
        circuits.NoiseModel(clifford_1q_error=0.01, clifford_nq_error={1: 0.0})
    with pytest.raises(ValueError, match="Ambiguous"):
        circuits.NoiseModel(clifford_1q_error=0.0, clifford_nq_error={1: 0.01})

    # Invalid key / out-of-range probability / bool rejection
    with pytest.raises(ValueError, match="must be >= 1"):
        circuits.NoiseModel(clifford_nq_error={0: 0.01})
    with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
        circuits.NoiseModel(clifford_nq_error={3: 1.5})
    with pytest.raises(TypeError, match="unsupported type bool"):
        circuits.NoiseModel(clifford_nq_error={2: True})

    # num_qubits mismatch between key and PauliChannel value (checked at construction).
    with pytest.raises(ValueError, match="num_qubits=2"):
        circuits.NoiseModel(clifford_nq_error={3: circuits.PauliChannel.depolarizing(2, 0.01)})

    # num_qubits mismatch via `rules=` is not caught at construction (arity varies), but is caught
    # at emission by `emit_after`.
    bad_rule = circuits.NoiseRule(after_pauli_channel=circuits.PauliChannel({"XY": 0.01}))
    noise_model = circuits.NoiseModel(rules={"SPP": bad_rule})
    with pytest.raises(ValueError, match="cannot be applied to operation"):
        noise_model.noisy_circuit(stim.Circuit("SPP X0*Y1*Z2"))

    # CORRELATED_ERROR / ELSE_CORRELATED_ERROR / E are rejected in `after`.
    for name in ("CORRELATED_ERROR", "ELSE_CORRELATED_ERROR", "E"):
        with pytest.raises(ValueError, match="use `after_pauli_channel`"):
            circuits.NoiseRule(after={name: 0.01})

    # Pairwise `after` channels (DEPOLARIZE2, ...) demand an even number of targets.  The
    # `_validate_rule_for_arity` helper enforces this at construction for known-arity call sites;
    # `emit_after` re-checks for variable-arity ones (like the standardized MPP name "MXYZ").
    with pytest.raises(ValueError, match="requires an even number of qubit targets"):
        circuits.NoiseModel(rules={"H": circuits.NoiseRule(after={"DEPOLARIZE2": 0.01})})
    with pytest.raises(ValueError, match="requires an even number of qubit targets"):
        circuits.NoiseModel(
            rules={"MXYZ": circuits.NoiseRule(after={"DEPOLARIZE2": 0.01})}
        ).noisy_circuit(stim.Circuit("MPP X0*Y1*Z2"))

    # readout_error / reset_error are rejected on rules for gates that can't measure/reset.
    with pytest.raises(ValueError, match="readout_error.*only valid on measurement"):
        circuits.NoiseModel(rules={"H": circuits.NoiseRule(readout_error=0.1)})
    with pytest.raises(ValueError, match="reset_error.*only valid on reset"):
        circuits.NoiseModel(rules={"M": circuits.NoiseRule(reset_error=0.1)})

    # Unrecognized `after` channel name → ValueError (not a bare KeyError).
    with pytest.raises(ValueError, match="Invalid or unrecognized noise channel"):
        circuits.NoiseRule(after={"NOT_A_GATE": 0.01})


def test_pauli_channel_idle_error_rejection() -> None:
    """after_pauli_channel is not accepted on idle-error rules (idle noise is per-qubit)."""
    channel = circuits.PauliChannel({"X": 0.01})
    with pytest.raises(ValueError, match="idle_error.*after_pauli_channel"):
        circuits.NoiseModel(idle_error=circuits.NoiseRule(after_pauli_channel=channel))
    with pytest.raises(
        ValueError, match="additional_error_waiting_for_m_or_r.*after_pauli_channel"
    ):
        circuits.NoiseModel(
            additional_error_waiting_for_m_or_r=circuits.NoiseRule(after_pauli_channel=channel)
        )


def test_pauli_channel_canonicalizes_order_and_drops_zeros() -> None:
    """PauliChannel drops zero-prob entries and canonicalizes insertion order."""
    ch = circuits.PauliChannel({"XI": 0.0, "IX": 0.05, "XX": 0.05})
    # "XI":0.0 is dropped, remaining keys are lex-sorted.
    assert list(ch.probabilities.keys()) == ["IX", "XX"]
    # Two channels differing only in insertion order compare equal and emit identical circuits.
    ch_alt = circuits.PauliChannel({"XX": 0.05, "IX": 0.05, "XI": 0.0})
    assert ch == ch_alt
    m1 = circuits.NoiseModel(rules={"SPP": circuits.NoiseRule(after_pauli_channel=ch)})
    m2 = circuits.NoiseModel(rules={"SPP": circuits.NoiseRule(after_pauli_channel=ch_alt)})
    assert m1.noisy_circuit(stim.Circuit("SPP X0*Y1")) == m2.noisy_circuit(
        stim.Circuit("SPP X0*Y1")
    )
    # An all-zero PauliChannel reduces to the trivial 0-qubit channel.
    all_zero = circuits.PauliChannel({"XY": 0.0})
    assert all_zero.num_qubits == 0
    assert dict(all_zero.probabilities) == {}
    assert not bool(all_zero)
    # ...and attaching one to a NoiseRule likewise leaves after_pauli_channel = None.
    all_zero_rule = circuits.NoiseRule(after_pauli_channel=circuits.PauliChannel({"XY": 0.0}))
    assert all_zero_rule.after_pauli_channel is None
    assert not bool(all_zero_rule)


def test_pauli_channel_hashable_and_frozen() -> None:
    """PauliChannel is hashable (equal channels hash equal) and its probabilities are frozen."""
    ch1 = circuits.PauliChannel({"XI": 0.1, "IX": 0.2})
    ch2 = circuits.PauliChannel({"IX": 0.2, "XI": 0.1})  # same channel, different insertion order
    assert ch1 == ch2 and hash(ch1) == hash(ch2)
    # Usable as a dict key / set member
    assert {ch1: "a", ch2: "b"} == {ch1: "b"}
    # The `probabilities` view is read-only
    with pytest.raises(TypeError):
        ch1.probabilities["XI"] = 0.5  # type: ignore[index]


def test_pauli_channel_float_drift_clamped() -> None:
    """Chain emission clamps and short-circuits when FP drift makes remaining <= prob."""
    # 3-qubit channel with keys chosen so lex order gives probs {0.1, 0.2, 0.3, 0.4}.
    # Sequential subtraction leaves remaining = 0.39999999999999997 < 0.4 at the final step, so
    # the last ELSE_CORRELATED_ERROR must clamp to 1.0 (rather than emit a value > 1.0) and stop.
    channel = circuits.PauliChannel({"XXX": 0.1, "XXY": 0.2, "XXZ": 0.3, "YYY": 0.4})
    rule = circuits.NoiseRule(after_pauli_channel=channel)
    noise_model = circuits.NoiseModel(rules={"SPP": rule})
    noisy = noise_model.noisy_circuit(stim.Circuit("SPP X0*Y1*Z2"))
    expected = stim.Circuit("""
        SPP X0*Y1*Z2
        CORRELATED_ERROR(0.1) X0 X1 X2
        ELSE_CORRELATED_ERROR(0.2222222222222222) X0 X1 Y2
        ELSE_CORRELATED_ERROR(0.42857142857142855) X0 X1 Z2
        ELSE_CORRELATED_ERROR(1.0) Y0 Y1 Y2
    """)
    assert _circuits_are_equivalent(expected, noisy)


def test_pauli_channel_pickle_round_trip() -> None:
    """PauliChannel supports pickle round-trip (needed for cache / multiprocessing)."""
    import pickle

    channel = circuits.PauliChannel({"XYZ": 0.01, "ZZZ": 0.02})
    restored = pickle.loads(pickle.dumps(channel))
    assert restored == channel
    assert hash(restored) == hash(channel)
    assert restored.num_qubits == 3


def test_repeat_blocks() -> None:
    """Repeat blocks get special treatment."""

    circuit = stim.Circuit("""
        H 0
        REPEAT 3 {
            CX 0 1
        }
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False)
    noisy_circuit = stim.Circuit("""
        H 0
        DEPOLARIZE1(0.1) 0
        REPEAT 3 {
            CX 0 1
            DEPOLARIZE2(0.1) 0 1
        }
    """)
    assert _circuits_are_equivalent(
        noisy_circuit, noise_model.noisy_circuit(circuit, insert_ticks=False)
    )

    immune_op_tag = "_TEST_"
    circuit = stim.Circuit(f"""
        H 0
        REPEAT['{immune_op_tag}'] 3 {{
            CX 0 1 1 2
        }}
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False)
    noisy_circuit = stim.Circuit(f"""
        H 0
        DEPOLARIZE1(0.1) 0
        REPEAT['{immune_op_tag}'] 3 {{
            CX 0 1 1 2
        }}
    """)
    assert _circuits_are_equivalent(
        noisy_circuit,
        noise_model.noisy_circuit(circuit, immune_op_tag=immune_op_tag),
    )


def test_noise_rule_errors() -> None:
    """Cover various NoiseRule errors."""
    with pytest.raises(ValueError, match="not between 0 and 1"):
        circuits.NoiseRule(readout_error=1.1)
    with pytest.raises(ValueError, match="not between 0 and 1"):
        circuits.NoiseRule(reset_error=1.1)
    with pytest.raises(ValueError, match="not between 0 and 1"):
        circuits.NoiseRule(after={"X_ERROR": -0.1})
    with pytest.raises(ValueError, match="Invalid or unrecognized noise channel"):
        circuits.NoiseRule(after={"S": 0.5})


def test_trivial_noise() -> None:
    """Boolean testing for trivial noise rules/models."""
    assert not bool(circuits.NoiseRule())
    assert not bool(circuits.NoiseModel())
    assert bool(circuits.NoiseRule(readout_error=0.1))
    assert bool(circuits.NoiseModel(readout_error=0.1))


def test_tsim_circuits() -> None:
    """noisy_circuit and as_noiseless_circuit accept and return tsim.Circuit."""
    noise_model = circuits.DepolarizingNoiseModel(0.01)

    tsim_circuit = tsim.Circuit("H 0\nCX 0 1\nM 0 1")
    stim_circuit = tsim_circuit.stim_circuit

    stim_noisy = noise_model.noisy_circuit(stim_circuit)
    tsim_noisy = noise_model.noisy_circuit(tsim_circuit)
    assert isinstance(stim_noisy, stim.Circuit)
    assert isinstance(tsim_noisy, tsim.Circuit)
    assert stim_noisy == tsim_noisy.stim_circuit

    stim_noiseless = circuits.as_noiseless_circuit(stim_circuit)
    tsim_noiseless = circuits.as_noiseless_circuit(tsim_circuit)
    assert isinstance(stim_noiseless, stim.Circuit)
    assert isinstance(tsim_noiseless, tsim.Circuit)
