import random

import pytest
import stim

import qldpc


def _are_equivalent(circuit_a: stim.Circuit, circuit_b: stim.Circuit, atol: float = 1e-10) -> bool:
    """Test equivalence between circuits after some standardization."""
    trivial_noise_model = qldpc.stim.NoiseModel()
    circuit_a = trivial_noise_model.noisy_circuit(circuit_a)
    circuit_b = trivial_noise_model.noisy_circuit(circuit_b)
    return circuit_a.approx_equals(circuit_b, atol=atol)


def test_noise_injection(pytestconfig: pytest.Config) -> None:
    """Inject noise into a circuit."""
    random.seed(pytestconfig.getoption("randomly_seed"))

    circuit = stim.Circuit("""
        H 0
        CX 0 1 1 2
        TICK
        M 0 1
        MRY 2
    """)

    # pick some random noise parameters
    p_1 = random.random() / 3
    p_2 = random.random()
    p_m = random.random()
    p_r = random.random()

    noise_model = qldpc.stim.NoiseModel(
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

    noise_rule = qldpc.stim.NoiseRule(after={"DEPOLARIZE2": p_2, "PAULI_CHANNEL_1": [0, p_1, p_1]})
    noise_model = qldpc.stim.NoiseModel(rules={"CX": noise_rule})
    noisy_circuit = stim.Circuit(f"""
        H 0
        CX 0 1
        DEPOLARIZE2({p_2}) 0 1
        PAULI_CHANNEL_1(0, {p_1}, {p_1}) 0 1
        CX 1 2
        DEPOLARIZE2({p_2}) 1 2
        PAULI_CHANNEL_1(0, {p_1}, {p_1}) 1 2
        TICK
        MZ 0 1
        MRY 2
    """)
    assert _are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
