"""Unit tests for custom.py

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


@functools.cache
def get_toy_problem() -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Get a toy decoding problem."""
    matrix = np.eye(3, 2, dtype=int)
    error = np.array([1, 1], dtype=int)
    syndrome = matrix @ error
    return matrix, error, syndrome


def test_relay_bp() -> None:
    """The Relay-BP decoder needs a custom wrapper class."""
    matrix, error, syndrome = get_toy_problem()
    syndrome_batch = np.array([syndrome])

    decoder = decoders.get_decoder_RBP("RelayDecoderF32", matrix)
    assert np.array_equal(error, decoder.decode(syndrome))
    assert np.array_equal(error, decoder.decode_batch(syndrome_batch)[0])
    assert np.array_equal(error, decoder.decode_detailed(syndrome).decoding)
    assert np.array_equal(error, decoder.decode_detailed_batch(syndrome_batch)[0].decoding)


def test_lookup() -> None:
    """Decode with a lookup table."""
    matrix, error, syndrome = get_toy_problem()

    # the distance of the given code is undefined, so lookup decoding to half the distance fails
    with pytest.warns(UserWarning, match="without specifying a maximum error weight"):
        assert np.array_equal([0, 0], decoders.decode(matrix, syndrome, with_lookup=True))

    # ... but it works if we manually tell it to try and decode errors of weight <= 2
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_lookup=True, max_weight=2))


def test_generalized_union_find() -> None:
    """Generalized Union-Find."""
    base_code: codes.CSSCode = codes.C4Code()
    code = functools.reduce(codes.CSSCode.concatenate, [base_code] * 3)
    error = code.field.Zeros(len(code))
    error[[3, 4]] = 1
    matrix = code.matrix_z
    syndrome = matrix @ error
    assert np.count_nonzero(decoders.decode(matrix, syndrome, with_GUF=True)) > 2
    assert np.count_nonzero(decoders.decode(matrix, syndrome, with_GUF=True, max_weight=2)) == 2

    # cover the trivial syndrome with the generalized Union-Find decoer
    assert np.array_equal(
        np.zeros_like(error), decoders.decode(matrix, np.zeros_like(syndrome), with_GUF=True)
    )


def test_block_decoder() -> None:
    """Decode independent code blocks."""
    matrix, error, syndrome = get_toy_problem()
    decoder = decoders.get_decoder(matrix)

    block_error = np.concatenate([error, error])
    block_syndrome = np.concatenate([syndrome, syndrome])
    block_decoder = decoders.BlockDecoder(syndrome.size, decoder)
    assert np.array_equal(block_error, block_decoder.decode(block_syndrome))


def test_direct_ilp_decoding() -> None:
    """Decode directly from a corrupted code word using integer linear programming."""
    matrix, error, syndrome = get_toy_problem()

    code_word = np.zeros_like(error)
    corrupted_code_word = (code_word + error) % 2
    decoder = decoders.ILPDecoder(matrix)
    direct_decoder = decoders.DirectDecoder.from_indirect(decoder, matrix)
    assert np.array_equal(code_word, direct_decoder.decode(corrupted_code_word))

    # try again over the trinary field
    field = galois.GF(3)
    matrix = -matrix.view(field)
    error = -error.view(field)
    code_word = code_word.view(field)

    corrupted_code_word = code_word + error.view(field)
    decoder = decoders.ILPDecoder(matrix)
    direct_decoder = decoders.DirectDecoder.from_indirect(decoder, matrix.view(field))
    assert np.array_equal(code_word, direct_decoder.decode(corrupted_code_word))


def test_invalid_ilp() -> None:
    """Fail to solve an invalid integer linear programming problem."""
    matrix = np.ones((2, 2), dtype=int)
    syndrome = np.array([0, 1], dtype=int)

    with pytest.raises(ValueError, match="could not be found"):
        decoders.decode(matrix, syndrome, with_ILP=True)

    with pytest.raises(ValueError, match="ILP decoding only supports prime number fields"):
        decoders.decode(galois.GF(4)(matrix), syndrome, with_ILP=True)


def test_quantum_decoding(pytestconfig: pytest.Config) -> None:
    """Decode an actual quantum code with random errors."""
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
