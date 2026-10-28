# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _moments.py."""

import pytest
import stim

from qldpc import circuits
from qldpc.circuits.conftest import circuits_are_equivalent
from qldpc.circuits.noise_model import _moments


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
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

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
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))


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
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

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
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))


def test_repeat_blocks() -> None:
    """Repeat blocks get special treatment."""

    # moments are identified as though repeat blocks were flattened
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
        REPEAT 2 {
            TICK
            CX 0 1
            DEPOLARIZE2(0.1) 0 1
        }
        TICK
        CX 0 1
        DEPOLARIZE2(0.1) 0 1
    """)
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
    with pytest.raises(ValueError, match="multiple uses"):
        noise_model.noisy_circuit(circuit, insert_ticks=False)

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
    assert circuits_are_equivalent(
        noisy_circuit,
        noise_model.noisy_circuit(circuit, immune_op_tag=immune_op_tag),
    )


def test_moment_qubits() -> None:
    """Qubits are categorized by their role in a moment, and qubit reuse is rejected."""
    moment = list(stim.Circuit("MPP X0*Z0\nR 1\nH 2\nI 3\nCX rec[-1] 4\nDETECTOR rec[-1]"))
    assert _moments.categorize_moment_qubits(moment) == ([0, 1], [2])
    with pytest.raises(ValueError, match="multiple uses"):
        _moments.categorize_moment_qubits(list(stim.Circuit("I 0\nH 0")))

    circuit = stim.Circuit("""
        QUBIT_COORDS[immune](0) 0
        REPEAT 2 {
            QUBIT_COORDS[immune](1) 1
            I 2
            OBSERVABLE_INCLUDE(0) X3
            H 4
        }
    """)
    assert _moments.get_immune_qubits(circuit, "immune") == {0, 1}
    assert _moments.idle_barrier_qubits(circuit) == {3, 4}


def test_repeated_idle_channel() -> None:
    """Repeating a Pauli channel raises the eigenvalues of its Pauli transfer matrix to a power."""
    p_x, p_y, p_z = 0.1, 0.2, 0.3
    eigenvalues = (1 - 2 * (p_y + p_z), 1 - 2 * (p_x + p_z), 1 - 2 * (p_x + p_y))
    for repetitions in [1, 2, 3, 2**40]:
        l_x, l_y, l_z = (eigenvalue**repetitions for eigenvalue in eigenvalues)
        expected = {
            "X": (1 + l_x - l_y - l_z) / 4,
            "Y": (1 - l_x + l_y - l_z) / 4,
            "Z": (1 - l_x - l_y + l_z) / 4,
        }
        channel = circuits.PauliChannel({"X": p_x, "Y": p_y, "Z": p_z})
        assert _moments.repeated_idle_channel(channel, repetitions).probabilities == pytest.approx(
            expected
        )

        # a sequence of one independent error is a Pauli channel with one component
        sequence = _moments.repeated_idle_channel(
            circuits.PauliChannelSequence({"X": p_x}), repetitions
        )
        single = _moments.repeated_idle_channel(circuits.PauliChannel({"X": p_x}), repetitions)
        assert sequence.probabilities == pytest.approx(single.probabilities)
