# SPDX-License-Identifier: Apache-2.0

"""Unit tests for protocols.py."""

from __future__ import annotations

import copy
import types
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from qldpc import decoders
from qldpc.decoders import protocols


def test_error_decoder_protocols() -> None:
    """Error decoder protocols provide decode as an alias for decode_errors, and vice versa."""
    syndromes = np.eye(2, dtype=int)

    # a subclass implements either name of a decoding method, and inherits the other
    class NewDecoder(decoders.BatchErrorDecoder):
        def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return 2 * syndrome

        def decode_errors_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return 3 * syndromes

    class OldDecoder(decoders.BatchErrorDecoder):
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return 2 * syndrome

        def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return 3 * syndromes

    for decoder in [NewDecoder(), OldDecoder()]:
        assert np.array_equal(decoder.decode_errors(syndromes[0]), [2, 0])
        assert np.array_equal(decoder.decode(syndromes[0]), [2, 0])
        assert np.array_equal(decoder.decode_errors_batch(syndromes), 3 * syndromes)
        assert np.array_equal(decoder.decode_batch(syndromes), 3 * syndromes)

    # a subclass must implement one of them
    class IncompleteDecoder(decoders.BatchErrorDecoder): ...

    with pytest.raises(NotImplementedError, match="must implement decode_errors"):
        IncompleteDecoder().decode(syndromes[0])
    with pytest.raises(NotImplementedError, match="must implement decode_errors_batch"):
        IncompleteDecoder().decode_batch(syndromes)


def test_error_decoder_coercion() -> None:
    """Objects with only one of the decode and decode_errors methods are wrapped to provide both."""
    syndromes = np.eye(2, dtype=int)

    class BareDecoder:
        scale = 2

        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return self.scale * syndrome

    class BareBatchDecoder(BareDecoder):
        def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return 3 * syndromes

    # a wrapped decoder decodes with, and reads attributes of, the object that it wraps
    bare_decoder = BareDecoder()
    decoder: Any = protocols._as_error_decoder(bare_decoder)
    assert decoder.decoder is bare_decoder
    assert repr(decoder) == f"_WrappedErrorDecoder({bare_decoder!r})"
    assert np.array_equal(decoder.decode_errors(syndromes[0]), [2, 0])
    assert np.array_equal(decoder.decode(syndromes[0]), [2, 0])
    bare_decoder.scale = 4
    assert decoder.scale == 4
    assert not hasattr(decoder, "decode_errors_batch")
    assert not hasattr(decoder, "__missing_attribute__")
    assert copy.copy(decoder).decoder is bare_decoder

    # batch decoding methods are provided if the wrapped object has them
    decoder = protocols._as_error_decoder(BareBatchDecoder())
    assert np.array_equal(decoder.decode_errors_batch(syndromes), 3 * syndromes)
    assert protocols._supports_batch_decoding(decoder)
    assert np.array_equal(protocols._batch_decode_errors(decoder, syndromes), 3 * syndromes)

    # an object that implements only decode_errors is wrapped to provide decode
    decoder = protocols._as_error_decoder(
        types.SimpleNamespace(
            decode_errors=lambda syndrome: 2 * syndrome,
            decode_errors_batch=lambda syndromes: 3 * syndromes,
        )
    )
    assert np.array_equal(decoder.decode(syndromes[0]), [2, 0])
    assert np.array_equal(decoder.decode_batch(syndromes), 3 * syndromes)

    # an error decoder is returned as is, and its native batch method is used
    lookup_decoder = decoders.custom.ErrorLookupDecoder(np.eye(2, dtype=int), max_weight=1)
    assert protocols._as_error_decoder(lookup_decoder) is lookup_decoder
    assert protocols._supports_batch_decoding(lookup_decoder)
    assert np.array_equal(protocols._batch_decode_errors(lookup_decoder, syndromes), syndromes)

    # an error decoder without a batch method decodes batches one syndrome at a time
    class SingleSyndromeDecoder(decoders.ErrorDecoder):
        def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.concatenate([syndrome, syndrome])

    error_decoder = SingleSyndromeDecoder()
    assert not protocols._supports_batch_decoding(error_decoder)
    batch = protocols._batch_decode_errors(error_decoder, syndromes)
    assert np.array_equal(batch, np.hstack([syndromes, syndromes]))
    assert protocols._batch_decode_errors(error_decoder, syndromes[:0]).shape == (0, 4)

    # objects that predict observable flips, or that do not decode, are rejected
    with pytest.raises(TypeError, match="observable flips rather than errors"):
        protocols._as_error_decoder(
            types.SimpleNamespace(decode_observables=lambda syndrome: syndrome)
        )
    with pytest.raises(TypeError, match="cannot decode until it is compiled"):
        protocols._as_error_decoder(decoders.SinterDecoder())
    with pytest.raises(TypeError, match="must be an ErrorDecoder, or have a decode method"):
        protocols._as_error_decoder(object())


def test_deprecated_decode_returns_observables() -> None:
    """A decode method that declares deprecated decode_returns_observables=True predicts flips."""
    legacy_decoder = types.SimpleNamespace(
        decode=lambda syndrome: syndrome, decode_returns_observables=True
    )
    with (
        pytest.warns(DeprecationWarning, match="decode_returns_observables=True is deprecated"),
        pytest.raises(TypeError, match="observable flips rather than errors"),
    ):
        protocols._as_error_decoder(legacy_decoder)
