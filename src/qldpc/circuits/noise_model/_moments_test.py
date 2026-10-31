# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _moments.py."""

import itertools
import math

import numpy as np
import pytest
import stim

from qldpc import circuits
from qldpc.circuits.conftest import circuits_are_equivalent
from qldpc.circuits.noise_model import _moments

# keep the moments of circuits, as written
TICKS = circuits.Scheduler.preserve_ticks(single_qubit_gates_take_time=True)


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
    noise_model = circuits.NoiseModel(scheduler=TICKS, idle_error=0.1)
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
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False, scheduler=TICKS)
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

    immune_op_tag = "_TEST_"
    circuit = stim.Circuit(f"""
        H 0
        REPEAT['{immune_op_tag}'] 3 {{
            CX 0 1 1 2
        }}
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False, scheduler=TICKS)
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
    assert _moments.categorize_moment_qubits(moment, TICKS.takes_time) == ([0, 1], [2, 3])
    with pytest.raises(ValueError, match="multiple uses"):
        _moments.categorize_moment_qubits(list(stim.Circuit("I 0\nH 0")), TICKS.takes_time)

    # gates that take no time neither reuse nor occupy their qubits
    moment = list(stim.Circuit("H 0\nCX 0 1\nH 0 2\nCX rec[-1] 1"))
    takes_time = circuits.Scheduler.asap().takes_time
    assert _moments.categorize_moment_qubits(moment, takes_time) == ([], [0, 1])

    # a classically controlled gate that takes time occupies its qubit, which nonetheless idles
    moment = list(stim.Circuit("CX rec[-1] 1"))
    assert _moments.categorize_moment_qubits(moment, lambda op: True) == ([], [])
    with pytest.raises(ValueError, match="multiple uses"):
        _moments.categorize_moment_qubits(list(stim.Circuit("CX rec[-1] 1\nH 1")), lambda op: True)

    circuit = stim.Circuit("""
        QUBIT_COORDS[immune](0) 0
        REPEAT 2 {
            QUBIT_COORDS[immune](1) 1
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


def test_idle_barrier_qubits() -> None:
    """Idle errors may be deferred past instructions that commute with Pauli errors."""
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
    assert _moments.idle_barrier_qubits(circuit) == {0, 1, 4, 5, 9}


def test_quadratic_idle_channel() -> None:
    """The twirl of a rotation by a Gaussian random angle that grows linearly over idle moments."""
    paulis = {
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    channel = circuits.PauliChannel({"X": 0.01, "Y": 0.02, "Z": 0.03})
    total = math.fsum(channel.probabilities.values())
    axis = sum(math.sqrt(p / total) * paulis[pauli] for pauli, p in channel.probabilities.items())

    # one moment rotates by an angle with standard deviation sigma, which flips a qubit with total
    # probability (1 - exp(-sigma**2 / 2)) / 2; average the twirl of k moments over the angle
    sigma = math.sqrt(-2 * math.log(1 - 2 * total))
    angles = np.linspace(-12 * sigma, 12 * sigma, 4001)
    weights = np.exp(-(angles**2) / (2 * sigma**2))
    weights /= weights.sum()
    for moments in [2, 30]:
        expected = dict.fromkeys(paulis, 0.0)
        for angle, weight in zip(moments * angles, weights, strict=True):
            unitary = math.cos(angle / 2) * np.eye(2) - 1j * math.sin(angle / 2) * axis
            for pauli, matrix in paulis.items():
                expected[pauli] += weight * abs(np.trace(matrix @ unitary) / 2) ** 2
        repeated = _moments.quadratic_idle_channel(channel, moments)
        assert repeated.probabilities == pytest.approx(expected, rel=1e-6)

    # a fully dephasing moment stays fully dephasing
    dephased = _moments.quadratic_idle_channel(circuits.PauliChannel({"Z": 0.5}), 3)
    assert dephased.probabilities == pytest.approx({"Z": 0.5})


def test_compose_idle_channels() -> None:
    """Independent idle errors compose into one channel, whether exclusive or independent events."""
    exclusive = circuits.PauliChannel({"X": 0.1, "Z": 0.2})
    independent = circuits.PauliChannelSequence({"X": 0.1, "Z": 0.2})
    other = circuits.PauliChannel({"Y": 0.05, "Z": 0.3})
    composed = _moments.compose_idle_channels(exclusive, other)
    assert composed.probabilities == pytest.approx(_compose(exclusive, other))
    events = _compose(circuits.PauliChannel({"X": 0.1}), circuits.PauliChannel({"Z": 0.2}))
    composed = _moments.compose_idle_channels(independent, other)
    assert composed.probabilities == pytest.approx(_compose(circuits.PauliChannel(events), other))


def _compose(first: circuits.PauliChannel, second: circuits.PauliChannel) -> dict[str, float]:
    """The probabilities of the product of independent one-qubit Pauli errors.

    Indexing Paulis as (I, X, Y, Z), the product of Paulis ``i`` and ``j`` is ``i ^ j`` up to phase.
    """

    def get_distribution(channel: circuits.PauliChannel) -> list[float]:
        probabilities = [channel.probabilities.get(pauli, 0.0) for pauli in "XYZ"]
        return [1 - sum(probabilities), *probabilities]

    result = [0.0] * 4
    for (i, p_i), (j, p_j) in itertools.product(
        enumerate(get_distribution(first)), enumerate(get_distribution(second))
    ):
        result[i ^ j] += p_i * p_j
    return dict(zip("XYZ", result[1:], strict=True))
