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


def test_immunity() -> None:
    """Qubits and operations can be immune to errors."""

    # qubits can be immune to errors
    circuit = stim.Circuit("""
        H 0 1
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False)
    noisy_circuit = stim.Circuit("""
        H 0 1
        DEPOLARIZE1(0.1) 1
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
            CX 0 1
        }}
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False)
    noisy_circuit = stim.Circuit(f"""
        H 0
        DEPOLARIZE1(0.1) 0
        REPEAT['{immune_op_tag}'] 3 {{
            CX 0 1
        }}
    """)
    assert _circuits_are_equivalent(
        noisy_circuit,
        noise_model.noisy_circuit(circuit, immune_op_tag=immune_op_tag, insert_ticks=False),
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
