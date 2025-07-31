import random

import pytest
import stim

from qldpc.stim import NoiseModel, NoiseRule


def _are_equivalent(circuit_a: stim.Circuit, circuit_b: stim.Circuit, atol: float = 1e-10) -> bool:
    """Test equivalence between circuits after some standardization."""
    trivial_noise_model = NoiseModel()
    circuit_a = trivial_noise_model.noisy_circuit(circuit_a)
    circuit_b = trivial_noise_model.noisy_circuit(circuit_b)
    return circuit_a.approx_equals(circuit_b, atol=atol)


def test_noise_injection(pytestconfig: pytest.Config) -> None:
    """Inject noise into a circuit."""
    random.seed(pytestconfig.getoption("randomly_seed"))

    p_1 = random.random() / 3
    p_2 = random.random()
    p_m = random.random()
    p_r = random.random()
    p_i = random.random()
    p_imr = random.random()

    ##################################################
    # GATE ERRORS

    circuit = stim.Circuit("""
        H 0
        CX 0 1 1 2
        TICK
        M 0 1
        MRY 2
    """)

    noise_model = NoiseModel(
        clifford_1q_error=p_1, clifford_2q_error=p_2, readout_error=p_m, reset_error=p_r
    )
    noisy_circuit = stim.Circuit(f"""
        H 0
        DEPOLARIZE1({p_1}) 0
        CX 0 1
        DEPOLARIZE2({p_2}) 0 1
        CX 1 2
        DEPOLARIZE2({p_2}) 1 2
        TICK
        MZ({p_m}) 0 1
        MRY({p_m}) 2
        X_ERROR({p_r}) 2
    """)
    assert _are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    ##################################################
    # COMPOSITION OF GATE ERRORS

    double_p_m = 1 - (1 - p_m) ** 2

    circuit = stim.Circuit("""
        H 0
        M 0
    """)
    noise_model = NoiseModel(clifford_1q_error=p_1, readout_error=p_m)
    noisy_circuit = stim.Circuit(f"""
        H 0
        DEPOLARIZE1({p_1}) 0
        MZ({p_m}) 0
    """)
    double_noisy_circuit = stim.Circuit(f"""
        H 0
        DEPOLARIZE1({p_1}) 0
        TICK
        DEPOLARIZE1({p_1}) 0
        MZ({double_p_m}) 0
    """)
    assert _are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
    assert _are_equivalent(double_noisy_circuit, noise_model.noisy_circuit(noisy_circuit))

    circuit = stim.Circuit("CX 0 1")
    noise_rule = NoiseRule(after={"DEPOLARIZE2": p_2, "PAULI_CHANNEL_1": [0, p_1, p_1]})
    noise_model = NoiseModel(rules={"CX": noise_rule})
    noisy_circuit = stim.Circuit(f"""
        CX 0 1
        DEPOLARIZE2({p_2}) 0 1
        PAULI_CHANNEL_1(0, {p_1}, {p_1}) 0 1
    """)
    assert _are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    ##################################################
    # IDLING ERRORS

    circuit = stim.Circuit("""
        H 0 1 2
        Z 1
        M 0
    """)
    noise_model = NoiseModel(
        readout_error=p_m, idle_error=p_i, additional_error_waiting_for_m_or_r=p_imr
    )
    noisy_circuit = stim.Circuit(f"""
        H 0 1 2
        Z 1
        M({p_m}) 0
        DEPOLARIZE1({p_i}) 2
        DEPOLARIZE1({p_imr}) 1 2
    """)
    assert _are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
