# SPDX-License-Identifier: Apache-2.0

"""Unit tests for observable-decoder adapters."""

from __future__ import annotations

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc import decoders
from qldpc.decoders.adapters import observable_decoders


class _FixedErrorDecoder(decoders.ErrorDecoder):
    def __init__(self, output: npt.ArrayLike, has_erasure_bit: bool = False) -> None:
        self.output = np.asarray(output)
        self.has_erasure_bit = has_erasure_bit

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return self.output


class _BitPackedCompiledDecoder:
    def __init__(self, packed_prediction: npt.NDArray[np.uint8]) -> None:
        self.packed_prediction = packed_prediction
        self.packed_syndromes: list[npt.NDArray[np.uint8]] = []

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        self.packed_syndromes.append(bit_packed_detection_event_data)
        return self.packed_prediction


def test_validate_decoder_output() -> None:
    """Decoder values belong to the field, followed only by binary erasure flags."""
    field = galois.GF(3)
    observable_decoders.validate_decoder_output(
        np.array([0, 2, 1]), 2, 1, field, "A decoder returned"
    )

    with pytest.raises(ValueError, match=r"shape \(2,\).+2 value\(s\) and 1 erasure flag"):
        observable_decoders.validate_decoder_output(
            np.array([0, 1]), 2, 1, field, "A decoder returned"
        )
    with pytest.raises(ValueError, match="expected integers"):
        observable_decoders.validate_decoder_output(
            np.array([0.0, 1.0]), 2, 0, field, "A decoder returned"
        )
    with pytest.raises(ValueError, match="not elements of GF"):
        observable_decoders.validate_decoder_output(
            np.array([0, 3]), 2, 0, field, "A decoder returned"
        )
    with pytest.raises(ValueError, match="erasure flags that are not 0 or 1"):
        observable_decoders.validate_decoder_output(
            np.array([0, 2]), 1, 1, field, "A decoder returned"
        )


def test_validate_observable_decoder() -> None:
    """Observable-decoder validation narrows valid objects and rejects invalid ones."""
    decoder = observable_decoders.ErrorsToFieldObservablesDecoder(
        _FixedErrorDecoder([0]), None, galois.GF(2), 1
    )
    assert observable_decoders.validate_observable_decoder(decoder, "The decoder") is decoder
    with pytest.raises(TypeError, match="must provide a decode_observables method"):
        observable_decoders.validate_observable_decoder(object(), "The decoder")


def test_errors_to_field_observables_decoder() -> None:
    """An inferred field-valued error is projected onto the requested observables."""
    field = galois.GF(3)
    observable_matrix = field([[1, 2, 0]])
    decoder = observable_decoders.ErrorsToFieldObservablesDecoder(
        _FixedErrorDecoder([2, 1, 0]), observable_matrix, field, 3
    )
    assert np.array_equal(decoder.decode_observables(np.array([0])), [1])

    identity_decoder = observable_decoders.ErrorsToFieldObservablesDecoder(
        _FixedErrorDecoder([2, 1, 0, 1], has_erasure_bit=True), None, field, 3
    )
    assert np.array_equal(identity_decoder.decode_observables(np.array([0])), [2, 1, 0, 1])


def test_bit_packed_observable_decoder() -> None:
    """A raw compiled Sinter decoder is adapted to unpacked observable predictions."""
    syndrome = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1])
    compiled_decoder = _BitPackedCompiledDecoder(np.array([[1, 1]], dtype=np.uint8))
    decoder = observable_decoders.BitPackedObservableDecoder(compiled_decoder, num_observables=9)
    assert np.array_equal(decoder.decode_observables(syndrome), syndrome.tolist() + [0])
    assert np.array_equal(compiled_decoder.packed_syndromes[-1], [[1, 1]])

    compiled_decoder = _BitPackedCompiledDecoder(np.array([[1, 1, 1]], dtype=np.uint8))
    decoder = observable_decoders.BitPackedObservableDecoder(compiled_decoder, num_observables=9)
    assert decoder.decode_observables(syndrome)[-1] == 1

    compiled_decoder = _BitPackedCompiledDecoder(np.array([[1]], dtype=np.uint8))
    decoder = observable_decoders.BitPackedObservableDecoder(compiled_decoder, num_observables=9)
    with pytest.raises(ValueError, match=r"shape \(1, 1\) for one shot"):
        decoder.decode_observables(syndrome)
