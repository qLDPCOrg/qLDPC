import stim

import qldpc


def _are_equivalent(circuit_a: stim.Circuit, circuit_b: stim.Circuit, atol: float = 1e-10) -> bool:
    """Test equivalence between circuits after some standardization."""
    trivial_noise_model = qldpc.stim.NoiseModel()
    circuit_a = trivial_noise_model.noisy_circuit(circuit_a)
    circuit_b = trivial_noise_model.noisy_circuit(circuit_b)
    print()
    print()
    print()
    print(circuit_a)
    print()
    print()
    print()
    print(circuit_b)
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
        {
            "H": qldpc.stim.NoiseRule(after={"DEPOLARIZE1": p_1}),
            "CX": qldpc.stim.NoiseRule(after={"DEPOLARIZE2": p_2}),
            "MZ": qldpc.stim.NoiseRule(flip_result=p_m),
        }
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

    noise_model = qldpc.stim.NoiseModel.from_probs(p_1, p_2, measure_flip_z=p_m)
    assert _are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
