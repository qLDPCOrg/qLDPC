# SPDX-License-Identifier: Apache-2.0

"""Unit tests for adapters/backends.py."""

from __future__ import annotations

import ldpc
import ldpc.bplsd_decoder
import numpy as np
import pymatching

from qldpc.decoders import adapters


def test_adapters() -> None:
    """Adapted decoders are instances of their original classes, and also error decoders."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
    syndromes = np.array([[1, 0], [1, 1]], dtype=int)
    errors = np.array([[1, 0, 0], [0, 1, 0]], dtype=int)

    ldpc_decoders = [
        (adapters.BpOsdDecoder, ldpc.BpOsdDecoder),
        (adapters.BpLsdDecoder, ldpc.bplsd_decoder.BpLsdDecoder),
        (adapters.BeliefFindDecoder, ldpc.BeliefFindDecoder),
    ]
    for adapted_class, original_class in ldpc_decoders:
        decoder = adapted_class(matrix, error_channel=[0.1] * 3)
        assert isinstance(decoder, original_class)
        for syndrome, error in zip(syndromes, errors):
            assert np.array_equal(decoder.decode_errors(syndrome), error)
            assert np.array_equal(decoder.decode(syndrome), error)

    matching = adapters.Matching()
    matching.load_from_check_matrix(matrix)
    assert isinstance(matching, pymatching.Matching)
    assert np.array_equal(matching.decode_errors(syndromes[0]), errors[0])
    assert np.array_equal(matching.decode_errors_batch(syndromes), errors)
    assert np.array_equal(matching.decode_batch(syndromes), errors)
