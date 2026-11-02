# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _moments.py."""

import itertools
import math

import numpy as np
import pytest

from qldpc import circuits
from qldpc.circuits.noise_model import _moments


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

        # each independent error of a sequence fires an odd number of times
        sequence = _moments.repeated_idle_channel(
            circuits.PauliChannelSequence({"X": p_x, "Z": p_z}), repetitions
        )
        assert sequence.probabilities == pytest.approx(
            {
                "X": (1 - (1 - 2 * p_x) ** repetitions) / 2,
                "Z": (1 - (1 - 2 * p_z) ** repetitions) / 2,
            }
        )


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
