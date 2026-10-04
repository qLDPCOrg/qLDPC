# SPDX-License-Identifier: Apache-2.0

"""Tests for custom decoder composition."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from qldpc import decoders
from qldpc.decoders.conftest import ToyProblem
from qldpc.decoders.construction.resolution import _get_error_decoder


def test_batch_decoding_by_alias() -> None:
    """An error decoder that only implements decode and decode_batch decodes batches."""
    matrix = np.eye(2, dtype=int)
    syndromes = np.eye(2, dtype=int)

    class OldDecoder(decoders.ErrorDecoder):
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return syndrome

        def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return syndromes

    old_decoder = OldDecoder()
    assert np.array_equal(old_decoder.decode_errors(syndromes[0]), syndromes[0])
    assert np.array_equal(decoders.batch_decode_errors(old_decoder, syndromes), syndromes)

    composite_decoder = decoders.custom.CompositeDecoder((old_decoder, 1), (old_decoder, 1))
    assert composite_decoder.decoders == (old_decoder, old_decoder)
    assert np.array_equal(composite_decoder.decode_batch(syndromes), syndromes)

    direct_decoder = decoders.custom.DirectDecoder.from_indirect(old_decoder, matrix)
    assert np.array_equal(direct_decoder.decode_batch(syndromes), np.zeros_like(syndromes))


def test_composite_erasure() -> None:
    """A CompositeDecoder is erased when any of its code blocks is erased."""
    matrix = np.array([[1, 1, 0], [0, 0, 0]])
    block_decoder = decoders.external.RelayBPDecoder(matrix, add_erasure_bit=True)
    composite_decoder = decoders.custom.CompositeDecoder.from_copies(block_decoder, 2, 2)

    syndromes = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
    expected_erasures = [0, 1, 1, 1]
    for syndrome, erased in zip(syndromes, expected_erasures):
        decoded = composite_decoder.decode(syndrome)
        assert len(decoded) == 2 * matrix.shape[1] + 1
        assert decoded[-1] == erased

    decoded_batch = composite_decoder.decode_batch(syndromes)
    assert np.array_equal(decoded_batch[:, -1], expected_erasures)

    plain_decoder = decoders.external.RelayBPDecoder(matrix)
    for blocks, expected in [
        (((plain_decoder, 2), (block_decoder, 2)), syndromes[:, 3]),
        (((block_decoder, 2), (plain_decoder, 2)), syndromes[:, 1]),
    ]:
        mixed_decoder = decoders.custom.CompositeDecoder(*blocks)
        assert mixed_decoder.has_erasure_bit
        for syndrome, erased in zip(syndromes, expected):
            decoded = mixed_decoder.decode(syndrome)
            assert len(decoded) == 2 * matrix.shape[1] + 1
            assert decoded[-1] == erased


def test_augmented_decoders(toy_problem: ToyProblem) -> None:
    """Composite and direct decoders can be built from other decoders."""
    matrix, error, syndrome = toy_problem
    decoder = _get_error_decoder(matrix, decoder=decoders.mwpm())

    direct_decoder = decoders.custom.DirectDecoder.from_indirect(decoder, matrix)
    assert np.array_equal(np.zeros_like(error), direct_decoder.decode(error))

    errors = np.array([error] * 3)
    assert np.array_equal(np.zeros_like(errors), direct_decoder.decode_batch(errors))

    composite_decoder = decoders.custom.CompositeDecoder.from_copies(decoder, syndrome.size, 2)
    composite_error = np.concatenate([error] * 2)
    composite_syndrome = np.concatenate([syndrome] * 2)
    assert np.array_equal(composite_error, composite_decoder.decode(composite_syndrome))

    composite_errors = np.array([composite_error] * 3)
    composite_syndromes = np.array([composite_syndrome] * 3)
    assert np.array_equal(composite_errors, composite_decoder.decode_batch(composite_syndromes))

    class WideDecoder:
        """A decoder that appends an extra entry to every error it infers."""

        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.zeros(error.size + 1, dtype=int)

        def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.zeros((len(syndromes), error.size + 1), dtype=int)

    direct_decoder = decoders.custom.DirectDecoder.from_indirect(WideDecoder(), matrix)
    with pytest.raises(ValueError, match="cannot be subtracted"):
        direct_decoder.decode(error)
    with pytest.raises(ValueError, match="cannot be subtracted"):
        direct_decoder.decode_batch(errors)
