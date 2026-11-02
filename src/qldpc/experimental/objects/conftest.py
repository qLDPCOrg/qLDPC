# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for tests of experimental operators."""

from __future__ import annotations

import numpy as np
import pytest
import stim


@pytest.fixture
def rng(pytestconfig: pytest.Config) -> np.random.Generator:
    """A random number generator seeded by the seed of pytest-randomly."""
    return np.random.default_rng(pytestconfig.getoption("randomly_seed"))


def get_random_pauli_string(
    rng: np.random.Generator, num_qubits: int, *, nontrivial: bool = False
) -> stim.PauliString:
    """A random Hermitian Pauli string with a random sign, not the identity if ``nontrivial``."""
    while True:
        letters = "".join(rng.choice(list("IXYZ"), size=num_qubits))
        string = (-1) ** int(rng.integers(2)) * stim.PauliString(letters)
        if string.weight or not nontrivial:
            return string
