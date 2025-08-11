"""Unit tests for decoder.py

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

import functools
import itertools
import random

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc import codes, decoders
from qldpc.math import symplectic_conjugate


def test_custom_decoder(pytestconfig: pytest.Config) -> None:
    """Inject custom decoders."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    matrix = np.random.randint(2, size=(2, 2))
    error = np.random.randint(2, size=matrix.shape[1])
    syndrome = (matrix @ error) % 2

    class CustomDecoder(decoders.Decoder):
        def __init__(self, matrix: npt.NDArray[np.int_]) -> None: ...
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return error

    assert decoders.decode(matrix, syndrome, decoder_constructor=CustomDecoder) is error
    assert decoders.decode(matrix, syndrome, static_decoder=CustomDecoder(matrix)) is error


def test_decoding() -> None:
    """Decode a simple problem."""
    matrix = np.eye(3, 2, dtype=int)
    error = np.array([1, 1], dtype=int)
    syndrome = np.array([1, 1, 0], dtype=int)
    default_decoder = decoders.get_decoder(matrix)

    assert np.array_equal(error, default_decoder.decode(syndrome))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_GUF=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_ILP=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_MWPM=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_BF=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_BP_LSD=True))

    # cover the trivial syndrome with the generalized Union-Find decoer
    assert np.array_equal(
        np.zeros_like(error), decoders.decode(matrix, np.zeros_like(syndrome), with_GUF=True)
    )

    # the distance of the given code is undefined, so lookup decoding to half the distance fails
    with pytest.warns(UserWarning, match="without specifying a maximum error weight"):
        assert np.array_equal([0, 0], decoders.decode(matrix, syndrome, with_lookup=True))

    # ... but it works if we manually tell it to try and decode errors of weight <= 2
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_lookup=True, max_weight=2))

    # decode two copies of the problem with a BlockDecoder
    block_error = np.concatenate([error, error])
    block_syndrome = np.concatenate([syndrome, syndrome])
    block_decoder = decoders.BlockDecoder(syndrome.size, default_decoder)
    assert np.array_equal(block_error, block_decoder.decode(block_syndrome))

    # decode directly from a corrupted code word
    code_word = np.zeros_like(error)
    corrupted_code_word = (code_word + error) % 2
    direct_decoder = decoders.DirectDecoder.from_indirect(default_decoder, matrix)
    assert np.array_equal(code_word, direct_decoder.decode(corrupted_code_word))

    # decode over trinary field
    field = galois.GF(3)
    matrix = -matrix.view(field)
    error = -error.view(field)
    assert np.array_equal(error, decoders.decode(matrix, syndrome))

    # decode directly from a corrupted code word
    code_word = field.Zeros(error.size)
    corrupted_code_word = code_word + error.view(field)
    ilp_decoder = decoders.ILPDecoder(matrix)
    direct_decoder = decoders.DirectDecoder.from_indirect(ilp_decoder, matrix.view(field))
    assert np.array_equal(code_word, direct_decoder.decode(corrupted_code_word))

    # the naive GUF decoder can fail sometimes
    base_code: codes.CSSCode = codes.C4Code()
    code = functools.reduce(codes.CSSCode.concatenate, [base_code] * 3)
    error = code.field.Zeros(len(code))
    error[[3, 4]] = 1
    matrix = code.matrix_z
    syndrome = matrix @ error
    assert np.count_nonzero(decoders.decode(matrix, syndrome, with_GUF=True)) > 2
    assert np.count_nonzero(decoders.decode(matrix, syndrome, with_GUF=True, max_weight=2)) == 2


def test_quantum_decoding(pytestconfig: pytest.Config) -> None:
    """Decode a full quantum code (as opposed to a classical subcode of a CSSCode)."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.SurfaceCode(5, field=3)
    local_errors = tuple(itertools.product(range(code.field.order), repeat=2))[1:]
    qubit_a, qubit_b = np.random.choice(range(len(code)), size=2, replace=False)
    pauli_a, pauli_b = random.choices(local_errors, k=2)
    error = code.field.Zeros(2 * len(code))
    error[[qubit_a, qubit_a + len(code)]] = pauli_a
    error[[qubit_b, qubit_b + len(code)]] = pauli_b
    syndrome = symplectic_conjugate(code.matrix) @ error

    decoder: decoders.Decoder
    decoder = decoders.GUFDecoder(code.matrix, symplectic=True)
    decoded_error = decoder.decode(syndrome).view(code.field)
    assert np.array_equal(syndrome, symplectic_conjugate(code.matrix) @ decoded_error)

    decoder = decoders.LookupDecoder(code.matrix, symplectic=True, max_weight=2)
    decoded_error = decoder.decode(syndrome).view(code.field)
    assert np.array_equal(syndrome, symplectic_conjugate(code.matrix) @ decoded_error)


def test_decoding_errors() -> None:
    """Fail to solve an invalid optimization problem."""
    matrix = np.ones((2, 2), dtype=int)
    syndrome = np.array([0, 1], dtype=int)

    with pytest.raises(ValueError, match="ILP decoding only supports prime number fields"):
        decoders.decode(galois.GF(4)(matrix), syndrome, with_ILP=True)

    with pytest.raises(ValueError, match="could not be found"):
        decoders.decode(matrix, syndrome, with_ILP=True)
