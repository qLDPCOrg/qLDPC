# SPDX-License-Identifier: Apache-2.0

"""Tests for the generalized Union-Find decoder."""

from __future__ import annotations

import functools

import numpy as np

from qldpc import codes, decoders, math
from qldpc.decoders.conftest import SurfaceCodeProblem


def test_generalized_union_find() -> None:
    """Generalized Union-Find finds a bounded-weight solution when requested."""
    base_code: codes.CSSCode = codes.C4Code()
    code = functools.reduce(codes.CSSCode.concatenate, [base_code] * 3)
    error = code.field.Zeros(len(code))
    error[[3, 4]] = 1
    matrix = code.matrix_z
    syndrome = matrix @ error
    assert (
        np.count_nonzero(
            decoders.get_error_decoder(matrix, decoder=decoders.guf()).decode(syndrome)
        )
        > 2
    )
    assert (
        np.count_nonzero(
            decoders.get_error_decoder(matrix, decoder=decoders.guf(max_weight=2)).decode(syndrome)
        )
        == 2
    )

    assert np.array_equal(
        np.zeros_like(error),
        decoders.get_error_decoder(matrix, decoder=decoders.guf()).decode(np.zeros_like(syndrome)),
    )
    decoded = decoders.GUFDecoder(matrix, add_erasure_bit=True).decode(syndrome)
    assert decoded[-1] == 0
    assert np.array_equal(matrix @ code.field(decoded[:-1]), syndrome)


def test_guf_builder() -> None:
    """The GUF builder decodes matrices and detector error models with optional erasure."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 0.1).to_dem()
    error = np.array([1, 0, 0])
    syndrome = matrix @ error % 2

    for pcm_or_dem in [matrix, dem]:
        decoder = decoders.get_decoder_guf(pcm_or_dem, add_erasure_bit=True)
        decoded = decoder.decode(syndrome)
        assert decoded.shape == (error.size + 1,)
        assert decoded[-1] == 0
        assert np.array_equal(matrix @ decoded[:-1] % 2, syndrome)


def test_symplectic_erasure() -> None:
    """A qudit syndrome that no error can induce is erased rather than answered."""
    code = codes.FiveQubitCode()
    matrix = np.vstack([np.asarray(code.matrix, dtype=int), np.zeros(2 * len(code), dtype=int)])
    decoder = decoders.GUFDecoder(matrix, symplectic=True, add_erasure_bit=True)

    syndrome = np.zeros(matrix.shape[0], dtype=int)
    syndrome[-1] = 1
    decoded = decoder.decode(syndrome)
    assert len(decoded) == 2 * len(code) + 1
    assert decoded[-1] == 1
    assert not np.any(decoded[:-1])


def test_quantum_decoding_from_plain_matrix() -> None:
    """A parity check matrix that is not a FieldArray is interpreted over GF(2)."""
    code = codes.FiveQubitCode()
    error = code.field.Zeros(2 * len(code))
    error[2] = 1
    syndrome = np.asarray(code.matrix @ math.symplectic_conjugate(error), dtype=int)

    decoder = decoders.GUFDecoder(np.asarray(code.matrix, dtype=int), symplectic=True)
    decoded_error = code.field(decoder.decode(syndrome))
    assert np.array_equal(syndrome, code.matrix @ math.symplectic_conjugate(decoded_error))


def test_quantum_decoding(surface_code_problem: SurfaceCodeProblem) -> None:
    """Decode random weight-2 errors in a GF(3) surface code."""
    code, _error, syndrome = surface_code_problem
    decoder = decoders.GUFDecoder(code.matrix, symplectic=True)
    decoded_error = decoder.decode(syndrome).view(code.field)
    assert np.array_equal(syndrome, code.matrix @ math.symplectic_conjugate(decoded_error))
