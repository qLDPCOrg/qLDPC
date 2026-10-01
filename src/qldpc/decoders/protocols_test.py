# SPDX-License-Identifier: Apache-2.0

"""Unit tests for protocols.py."""

from __future__ import annotations

import copy
import types
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders


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
    """Objects with a decode method are wrapped to provide decode_errors."""
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
    decoder: Any = decoders.as_error_decoder(bare_decoder)
    assert decoder.decoder is bare_decoder
    assert repr(decoder) == f"WrappedErrorDecoder({bare_decoder!r})"
    assert np.array_equal(decoder.decode_errors(syndromes[0]), [2, 0])
    assert np.array_equal(decoder.decode(syndromes[0]), [2, 0])
    bare_decoder.scale = 4
    assert decoder.scale == 4
    assert not hasattr(decoder, "decode_errors_batch")
    assert not hasattr(decoder, "__missing_attribute__")
    assert copy.copy(decoder).decoder is bare_decoder

    # batch decoding methods are provided if the wrapped object has them
    decoder = decoders.as_error_decoder(BareBatchDecoder())
    assert np.array_equal(decoder.decode_errors_batch(syndromes), 3 * syndromes)
    assert decoders.supports_batch_decoding(decoder)
    assert np.array_equal(decoders.batch_decode_errors(decoder, syndromes), 3 * syndromes)

    # an error decoder is returned as is, and its native batch method is used
    lookup_decoder = decoders.LookupDecoder(np.eye(2, dtype=int), max_weight=1)
    assert decoders.as_error_decoder(lookup_decoder) is lookup_decoder
    assert decoders.supports_batch_decoding(lookup_decoder)
    assert np.array_equal(decoders.batch_decode_errors(lookup_decoder, syndromes), syndromes)

    # an error decoder without a batch method decodes batches one syndrome at a time
    class SingleSyndromeDecoder(decoders.ErrorDecoder):
        def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.concatenate([syndrome, syndrome])

    error_decoder = SingleSyndromeDecoder()
    assert not decoders.supports_batch_decoding(error_decoder)
    batch = decoders.batch_decode_errors(error_decoder, syndromes)
    assert np.array_equal(batch, np.hstack([syndromes, syndromes]))
    assert decoders.batch_decode_errors(error_decoder, syndromes[:0]).shape == (0, 4)

    # objects that predict observable flips, or that do not decode, are rejected
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    with pytest.warns(DeprecationWarning):
        legacy_decoder = decoders.LookupDecoder(dem, max_weight=1, predict_observable_flips=True)

    observable_decoders = [
        decoders.ObservableLookupDecoder(dem, max_weight=1),
        legacy_decoder,
        decoders.TrivialDecoder().compile_decoder_for_dem(dem),
        types.SimpleNamespace(decode_observables=lambda syndrome: syndrome),
    ]
    for observable_decoder in observable_decoders:
        with pytest.raises(TypeError, match="observable flips rather than errors"):
            decoders.as_error_decoder(observable_decoder)
    with pytest.raises(TypeError, match="cannot decode until it is compiled"):
        decoders.as_error_decoder(decoders.SinterDecoder())
    with pytest.raises(TypeError, match="must be an ErrorDecoder, or have a decode method"):
        decoders.as_error_decoder(object())
