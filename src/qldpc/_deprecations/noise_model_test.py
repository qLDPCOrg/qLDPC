# SPDX-License-Identifier: Apache-2.0

"""Unit tests for noise_model.py."""

import pytest
import stim

from qldpc import circuits
from qldpc._deprecations import noise_model


def test_deprecated_noisy_circuit_with_ticks() -> None:
    """Noisy circuits laid out in moments, as qLDPC made them before the deprecation."""
    # gates, measurements, idling, waiting, classical control, and Pauli products
    model = circuits.NoiseModel(
        clifford_1q_error=0.01,
        clifford_2q_error=0.02,
        readout_error=0.1,
        idle_error=0.2,
        additional_error_waiting_for_m_or_r=0.3,
    )
    circuit = stim.Circuit(
        "H 0 1 2\nH 1\nM 0\nDETECTOR rec[-1]\nCX rec[-1] 1 2 3\nTICK\nSPP X0*Y1 Z2\nMPP X0*Z1"
    )
    assert _get_noisy_circuit(model, circuit).approx_equals(
        stim.Circuit("""
        H 0 1 2
        DEPOLARIZE1(0.01) 0 1 2
        TICK
        H 1
        M(0.1) 0
        DETECTOR rec[-1]
        DEPOLARIZE1(0.01) 1
        DEPOLARIZE1(0.3) 1 2 3
        DEPOLARIZE1(0.2) 2
        DEPOLARIZE1(0.346667) 3
        TICK
        CX rec[-1] 1 2 3
        DEPOLARIZE2(0.02) 2 3
        DEPOLARIZE1(0.2) 0 1
        TICK
        SPP X0*Y1 Z2
        DEPOLARIZE2(0.02) 0 1
        DEPOLARIZE1(0.01) 2
        TICK
        MPP(0.1) X0*Z1
        DEPOLARIZE1(0.3) 2 3
        DEPOLARIZE1(0.2) 2
        DEPOLARIZE1(0.346667) 3
    """),
        atol=1e-6,
    )

    # idle noise from Stim fragments, which is applied in each moment, and quadratic idle noise
    model = circuits.NoiseModel(
        idle_error=circuits.NoiseRule(after=stim.Circuit("X_ERROR(0.1) 0")),
        quadratic_idle_error=0.03,
    )
    circuit = stim.Circuit("H 0\nTICK\nH 1\nTICK\nH 1\nTICK\nH 0")
    assert _get_noisy_circuit(model, circuit, system_qubits=[0, 1]).approx_equals(
        stim.Circuit("""
        H 0
        X_ERROR(0.1) 1
        DEPOLARIZE1(0.03) 1
        TICK
        H 1
        X_ERROR(0.1) 0
        TICK
        H 1
        X_ERROR(0.1) 0
        DEPOLARIZE1(0.109626) 0
        TICK
        H 0
        X_ERROR(0.1) 1
        DEPOLARIZE1(0.03) 1
        """),
        atol=1e-6,
    )

    # repeat blocks that become compact, noiseless and gate-free repeat blocks, and empty ones
    model = circuits.NoiseModel(
        idle_error={"X": 0.1}, quadratic_idle_error={"Z": 0.01}, readout_error=0.05
    )
    circuit = stim.Circuit("""
        R 0 1
        TICK
        REPEAT 5 {
            X 1
            TICK
        }
        REPEAT[__IMMUNE_TO_NOISE__] 2 {
            H 0
        }
        REPEAT 2 {
            DETECTOR
        }
        REPEAT 2 {
            REPEAT 3 {
            }
        }
        M 0 1
    """)
    assert _get_noisy_circuit(model, circuit).approx_equals(
        stim.Circuit("""
        R 0 1
        REPEAT 5 {
            TICK
            X 1
        }
        PAULI_CHANNEL_1(0.26951, 0.0666496, 0.131618) 0
        TICK
        REPEAT[__IMMUNE_TO_NOISE__] 2 {
            H 0
        }
        REPEAT 2 {
            DETECTOR
        }
        M(0.05) 0 1
        """),
        atol=1e-6,
    )

    # a repeat block whose iterations get the same noise only after the first
    model = circuits.NoiseModel(clifford_1q_error=0.01)
    circuit = stim.Circuit("H 0\nREPEAT 2 {\nH 0\n}")
    expected = stim.Circuit("H 0\nDEPOLARIZE1(0.01) 0\nTICK\n" * 2 + "H 0\nDEPOLARIZE1(0.01) 0")
    assert _get_noisy_circuit(model, circuit) == expected

    # immune qubits, noiseless operations, and a rule function
    model = circuits.NoiseModel(
        clifford_1q_error=0.01,
        clifford_2q_error=0.02,
        readout_error=0.03,
        rule_func=lambda op: None,
    )
    circuit = stim.Circuit("H 0 1\nCX 0 1 2 3\nH[__IMMUNE_TO_NOISE__] 2\nM 0 1")
    assert _get_noisy_circuit(model, circuit, immune_qubits=frozenset({1})).approx_equals(
        stim.Circuit("""
        H 0 1
        DEPOLARIZE1(0.01) 0
        TICK
        CX 0 1 2 3
        DEPOLARIZE2(0.02) 2 3
        TICK
        H[__IMMUNE_TO_NOISE__] 2
        M(0.03) 0
        M 1
    """),
        atol=1e-6,
    )


def _get_noisy_circuit(
    model: circuits.NoiseModel,
    circuit: stim.Circuit,
    *,
    system_qubits: list[int] | None = None,
    immune_qubits: frozenset[int] = frozenset(),
) -> stim.Circuit:
    """Add noise to a circuit, laid out in moments."""
    return noise_model.get_noisy_circuit_with_ticks(
        model,
        circuit,
        system_qubits=frozenset(
            range(circuit.num_qubits) if system_qubits is None else system_qubits
        ),
        immune_qubits=immune_qubits,
        immune_op_tag=circuits.DEFAULT_IMMUNE_OP_TAG,
        immunize_gates=True,
    )


def test_deprecated_moment_qubits() -> None:
    """Qubits are categorized by their role in a moment, and qubit reuse is rejected."""
    moment = list(stim.Circuit("MPP X0*Z0\nR 1\nH 2\nI 3\nCX rec[-1] 4\nDETECTOR rec[-1]"))
    assert noise_model.categorize_moment_qubits(moment) == ([0, 1], [2, 3])
    with pytest.raises(ValueError, match="multiple uses"):
        noise_model.categorize_moment_qubits(list(stim.Circuit("H 0\nH 0")))

    circuit = stim.Circuit("""
        H 0
        OBSERVABLE_INCLUDE(0) X1
        X_ERROR(0.1) 2
        CX rec[-1] 3 4 5
        CZ 6 sweep[0]
        QUBIT_COORDS(0, 0) 8
        REPEAT 2 {
            M 9
        }
    """)
    assert noise_model.idle_barrier_qubits(circuit) == {0, 1, 4, 5, 9}
