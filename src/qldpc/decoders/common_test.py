# SPDX-License-Identifier: Apache-2.0

"""Unit tests for common.py."""

from __future__ import annotations

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc import decoders
from qldpc.decoders import common


def test_with_erasure_bits() -> None:
    """Erasure flags are appended to individual and batched inferred errors."""
    error = np.array([1, 0], dtype=int)
    assert np.array_equal(decoders.with_erasure_bits(error, True), [1, 0, 1])

    errors = np.array([[1, 0], [0, 1]], dtype=int)
    erased = np.array([False, True])
    assert np.array_equal(decoders.with_erasure_bits(errors, erased), [[1, 0, 0], [0, 1, 1]])


def test_get_error_and_erasure() -> None:
    """Decode a syndrome, with and without an erasure bit."""
    field = galois.GF2
    syndrome = field([1, 0, 1])

    class _Decoder:
        def __init__(self, output: np.ndarray, has_erasure_bit: bool = False) -> None:
            self.output = output
            if has_erasure_bit:
                self.has_erasure_bit = True

        def decode(self, syndrome: np.ndarray) -> np.ndarray:
            return self.output

    # a plain decoder returns the inferred error and no erasure
    decoder = _Decoder(np.array([1, 1, 0, 0], dtype=np.uint8))
    error, erasure = decoders.get_error_and_erasure(decoder, syndrome)
    assert not erasure and isinstance(error, field) and np.array_equal(error, field([1, 1, 0, 0]))

    # an erasure-enabled decoder strips the last (erasure) bit and reports it.  The first and last
    # entries differ, so reading the wrong end of the vector fails here
    decoder = _Decoder(np.array([0, 1, 1, 0, 1], dtype=np.uint8), has_erasure_bit=True)
    error, erasure = decoders.get_error_and_erasure(decoder, syndrome)
    assert erasure and np.array_equal(error, field([0, 1, 1, 0]))

    # the same decoder reports no erasure when the syndrome was recognized
    decoder = _Decoder(np.array([1, 1, 0, 0, 0], dtype=np.uint8), has_erasure_bit=True)
    error, erasure = decoders.get_error_and_erasure(decoder, syndrome)
    assert not erasure and np.array_equal(error, field([1, 1, 0, 0]))


def test_erasure_bit_support_decorator() -> None:
    """Erasure support declarations enforce capabilities and use an explicit display name."""

    @common._erasure_bit_support("Friendly Name", supported=True)
    def unusually_named_builder(
        matrix: npt.NDArray[np.int_], *, add_erasure_bit: bool = False
    ) -> decoders.ErrorDecoder:
        del add_erasure_bit
        return decoders.get_decoder_bp_osd(matrix)

    with pytest.raises(ValueError, match="The Friendly Name decoder cannot signal erasure"):
        unusually_named_builder(np.eye(1, dtype=int), add_erasure_bit=True)

    @common._erasure_bit_support("unsupported", supported=False)
    def unsupported_builder(
        matrix: npt.NDArray[np.int_], *, add_erasure_bit: bool = False
    ) -> decoders.ErrorDecoder:
        del add_erasure_bit
        return decoders.get_decoder_bp_osd(matrix)

    assert unsupported_builder(np.eye(1, dtype=int), add_erasure_bit=False)
    with pytest.raises(ValueError, match="The unsupported decoder cannot signal erasure"):
        unsupported_builder(np.eye(1, dtype=int), add_erasure_bit=True)


def test_to_pcm() -> None:
    """Matrices pass through unchanged and DEM detector matrices are densified."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
    assert common._to_pcm(matrix) is matrix

    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 0.1).to_dem()
    assert np.array_equal(common._to_pcm(dem), matrix)
