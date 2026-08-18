"""Shared pytest fixtures for decoders tests.

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import itertools
import random

import galois
import numpy as np
import pytest

from qldpc import codes, math

ToyProblem = tuple[galois.FieldArray, galois.FieldArray, galois.FieldArray]
SurfaceCodeProblem = tuple[codes.SurfaceCode, galois.FieldArray, galois.FieldArray]


@pytest.fixture(scope="session")
def toy_problem() -> ToyProblem:
    """A toy classical decoding problem: a parity check matrix, an error, and its syndrome."""
    field = galois.GF(2)
    matrix = np.eye(3, 2, dtype=int).view(field)
    error = np.array([1, 1], dtype=int).view(field)
    syndrome = matrix @ error
    return matrix, error, syndrome


@pytest.fixture
def surface_code_problem(pytestconfig: pytest.Config) -> SurfaceCodeProblem:
    """A random weight-2 error and its syndrome in a GF(3) surface code."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))
    code = codes.SurfaceCode(4, field=3)
    local_errors = tuple(itertools.product(code.field.elements, repeat=2))[1:]
    qubit_a, qubit_b = np.random.choice(range(len(code)), size=2, replace=False)
    pauli_a, pauli_b = random.choices(local_errors, k=2)
    error = code.field.Zeros(2 * len(code))
    error[[qubit_a, qubit_a + len(code)]] = pauli_a
    error[[qubit_b, qubit_b + len(code)]] = pauli_b
    syndrome = code.matrix @ math.symplectic_conjugate(error)
    return code, error, syndrome
