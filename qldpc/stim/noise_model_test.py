import stim

import qldpc


def _are_equivalent(circuit_a: stim.Circuit, circuit_b: stim.Circuit, atol: float = 1e-10) -> bool:
    """Test equivalence between circuits after some standardization."""
    trivial_noise_model = qldpc.stim.NoiseModel()
    circuit_a = trivial_noise_model.noisy_circuit(circuit_a)
    circuit_b = trivial_noise_model.noisy_circuit(circuit_b)
    return circuit_a.approx_equals(circuit_b, atol=atol)


def test_noise_injections() -> None:
    """Inject noise into a circuit."""
    circuit = stim.Circuit("""
        H 0
        CX 0 1 1 2
        TICK
        M 0 1 2
    """)

    p_1 = 1e-3
    p_2 = 1e-2
    p_m = 5e-2
    noise_model = qldpc.stim.NoiseModel(
        clifford_1q_error=p_1, clifford_2q_error=p_2, readout_error=p_m
    )
    noisy_circuit = stim.Circuit(f"""
        H 0
        DEPOLARIZE1({p_1}) 0
        CX 0 1
        DEPOLARIZE2({p_2}) 0 1
        CX 1 2
        DEPOLARIZE2({p_2}) 1 2
        TICK
        M({p_m}) 0 1 2
    """)
    assert _are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
