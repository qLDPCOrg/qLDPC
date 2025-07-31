import random

import pytest
import stim

from qldpc.stim import NoiseModel, NoiseRule


def _are_equivalent(circuit_a: stim.Circuit, circuit_b: stim.Circuit, atol: float = 1e-10) -> bool:
    """Test equivalence between circuits after some standardization."""
    trivial_noise_model = NoiseModel()
    circuit_a = trivial_noise_model.noisy_circuit(circuit_a)
    circuit_b = trivial_noise_model.noisy_circuit(circuit_b)
    if not circuit_a.approx_equals(circuit_b, atol=atol):
        print()
        print()
        print()
        print()
        print(circuit_a)
        print()
        print()
        print()
        print(circuit_b)
        print()
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
        MRY 2
    """)
    noise_model = NoiseModel(
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
        MRY(0.3) 2
        Z_ERROR(0.4) 1
        X_ERROR(0.4) 2
    """)
    assert _are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # multiple errors after one gate
    circuit = stim.Circuit("""
        CX 0 1
    """)
    noise_rule = NoiseRule(after={"DEPOLARIZE2": 0.2, "PAULI_CHANNEL_1": [0, 0.1, 0.1]})
    noise_model = NoiseModel(rules={"CX": noise_rule})
    noisy_circuit = stim.Circuit("""
        CX 0 1
        DEPOLARIZE2(0.2) 0 1
        PAULI_CHANNEL_1(0, 0.1, 0.1) 0 1
    """)
    assert _are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # compose gate errors
    p_m = 0.1
    double_p_m = 1 - (1 - p_m) ** 2
    circuit = stim.Circuit("""
        H 0
        M 0
    """)
    noise_model = NoiseModel(readout_error=p_m)
    noisy_circuit = stim.Circuit(f"""
        H 0
        MZ({p_m}) 0
    """)
    double_noisy_circuit = stim.Circuit(f"""
        H 0
        MZ({double_p_m}) 0
    """)
    assert _are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
    assert _are_equivalent(double_noisy_circuit, noise_model.noisy_circuit(noisy_circuit))


def test_idle_errors() -> None:
    """Add idling errors to a circuit."""

    circuit = stim.Circuit("""
        H 0 1 2
        Z 1
        M 0
    """)
    noise_model = NoiseModel(
        readout_error=0.1, idle_error=0.2, additional_error_waiting_for_m_or_r=0.3
    )
    noisy_circuit = stim.Circuit("""
        H 0 1 2
        Z 1
        M(0.1) 0
        DEPOLARIZE1(0.2) 2
        DEPOLARIZE1(0.3) 1 2
    """)
    assert _are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))


def test_immunity() -> None:
    """Qubits can be immune to errors."""

    circuit = stim.Circuit("""
        H 0 1
    """)
    noise_model = NoiseModel(clifford_1q_error=0.1)
    noisy_circuit = stim.Circuit("""
        H 0 1
        DEPOLARIZE1(0.1) 1
    """)
    assert _are_equivalent(
        noisy_circuit, noise_model.noisy_circuit(circuit, immune_qubits=[0], insert_ticks=False)
    )

    with pytest.raises(ValueError, match="does not support immune qubits"):
        assert _are_equivalent(
            noisy_circuit, noise_model.noisy_circuit(circuit, immune_qubits=[0], insert_ticks=True)
        )


def test_classical_controls() -> None:
    """Classically controled gates get special treatment."""

    circuit = stim.Circuit("""
        CX 0 1 rec[-1] 2
    """)
    noise_model = NoiseModel(clifford_2q_error=0.2)
    noisy_circuit = stim.Circuit("""
        CX 0 1 rec[-1] 2
        DEPOLARIZE2(0.2) 0 1
    """)
    assert _are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))


def test_pauli_product_measurements() -> None:
    """Pauli product measurements get special treatment."""


def test_repeat_blocks() -> None:
    """Repeat blocks get special treatment."""


def test_noise_rule_errors() -> None:
    """Cover various NoiseRule errors."""
    with pytest.raises(ValueError, match="not between 0 and 1"):
        NoiseRule(readout_error=1.1)
    with pytest.raises(ValueError, match="not between 0 and 1"):
        NoiseRule(reset_error=1.1)
    with pytest.raises(ValueError, match="not between 0 and 1"):
        NoiseRule(after={"X_ERROR": -0.1})
    with pytest.raises(ValueError, match="Invalid or unrecognized noise channel"):
        NoiseRule(after={"S": 0.5})
