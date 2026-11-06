# SPDX-License-Identifier: Apache-2.0

"""Unit tests for observable-decoder adapters."""

from __future__ import annotations

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc.decoders.adapters import _observable_decoders


def test_bit_packed_observable_decoder() -> None:
    """A raw compiled Sinter decoder is adapted to unpacked observable predictions."""
    syndrome = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1])
    compiled_decoder = _BitPackedCompiledDecoder(np.array([[1, 1]], dtype=np.uint8))
    decoder = _observable_decoders.BitPackedObservableDecoder(compiled_decoder, num_observables=9)
    assert np.array_equal(decoder.decode_observables(syndrome), syndrome.tolist() + [0])
    assert np.array_equal(compiled_decoder.packed_syndromes[-1], [[1, 1]])

    compiled_decoder = _BitPackedCompiledDecoder(np.array([[1, 1, 1]], dtype=np.uint8))
    decoder = _observable_decoders.BitPackedObservableDecoder(compiled_decoder, num_observables=9)
    assert decoder.decode_observables(syndrome)[-1] == 1

    compiled_decoder = _BitPackedCompiledDecoder(np.array([[1]], dtype=np.uint8))
    decoder = _observable_decoders.BitPackedObservableDecoder(compiled_decoder, num_observables=9)
    with pytest.raises(ValueError, match=r"of shape \(1, 1\)"):
        decoder.decode_observables(syndrome)


def test_validate_observable_decoder() -> None:
    """Observable-decoder validation narrows valid objects and rejects invalid ones."""
    decoder = _observable_decoders.BitPackedObservableDecoder(
        _BitPackedCompiledDecoder(np.zeros((1, 1), dtype=np.uint8)), num_observables=1
    )
    assert _observable_decoders.validate_observable_decoder(decoder, "The decoder") is decoder
    with pytest.raises(TypeError, match="must provide a decode_observables method"):
        _observable_decoders.validate_observable_decoder(object(), "The decoder")


def test_validate_decoder_output() -> None:
    """Decoder values belong to the field, followed only by binary erasure flags."""
    field = galois.GF(3)
    _observable_decoders.validate_decoder_output(
        np.array([0, 2, 1]), 2, 1, field, "A decoder returned"
    )

    with pytest.raises(ValueError, match=r"expected shape \(3,\)"):
        _observable_decoders.validate_decoder_output(
            np.array([0, 1]), 2, 1, field, "A decoder returned"
        )
    with pytest.raises(ValueError, match="expected integers"):
        _observable_decoders.validate_decoder_output(
            np.array([0.0, 1.0]), 2, 0, field, "A decoder returned"
        )
    with pytest.raises(ValueError, match="not elements of GF"):
        _observable_decoders.validate_decoder_output(
            np.array([0, 3]), 2, 0, field, "A decoder returned"
        )
    with pytest.raises(ValueError, match="erasure flags that are not 0 or 1"):
        _observable_decoders.validate_decoder_output(
            np.array([0, 2]), 1, 1, field, "A decoder returned"
        )


class _BitPackedCompiledDecoder:
    def __init__(self, packed_prediction: npt.NDArray[np.uint8]) -> None:
        self.packed_prediction = packed_prediction
        self.packed_syndromes: list[npt.NDArray[np.uint8]] = []

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        self.packed_syndromes.append(bit_packed_detection_event_data)
        return self.packed_prediction
