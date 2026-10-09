# SPDX-License-Identifier: Apache-2.0

"""Unit tests for capabilities.py."""

from __future__ import annotations

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc import decoders


class _FixedObservableDecoder(decoders.ObservableDecoder):
    def __init__(self, output: npt.ArrayLike) -> None:
        self.output = np.asarray(output)

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return self.output


class _FixedErrorDecoder(decoders.ErrorDecoder):
    def __init__(self, output: npt.ArrayLike) -> None:
        self.output = np.asarray(output)

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return self.output


class _BitPackedCompiledDecoder:
    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        return np.zeros((len(bit_packed_detection_event_data), 1), dtype=np.uint8)


def test_is_prebuilt_decoder() -> None:
    """Prebuilt decoders decode, and are not specifications, constructors, or compilers."""
    matrix = np.eye(2, dtype=int)
    prebuilt_decoders: list[object] = [
        _FixedErrorDecoder([0, 0]),
        _FixedObservableDecoder([0]),
        _BitPackedCompiledDecoder(),
        decoders.TrivialDecoder().compile_decoder_for_dem(stim.DetectorErrorModel()),
        decoders.bp_osd().build(matrix),
    ]
    for decoder in prebuilt_decoders:
        assert decoders.is_prebuilt_decoder(decoder)
    deferred_decoders: list[object] = [
        None,
        decoders.bp_osd(),
        decoders.custom.LookupDecoder,
        lambda matrix: _FixedErrorDecoder([0, 0]),
        decoders.from_matrix(lambda matrix: _FixedErrorDecoder([0, 0])),
        decoders.from_dem(lambda dem: _FixedObservableDecoder([0])),
        decoders.SinterDecoder(),
        decoders.TrivialDecoder(),
    ]
    for decoder in deferred_decoders:
        assert not decoders.is_prebuilt_decoder(decoder)


def test_is_prebuilt_observable_decoder() -> None:
    """Classify decoder inputs by explicit capabilities."""
    matrix = galois.GF2([[1, 1]])
    observable_decoder = _FixedObservableDecoder([0])
    assert np.array_equal(observable_decoder.decode_observables(np.array([0])), [0])
    assert decoders.is_prebuilt_observable_decoder(observable_decoder)

    bit_packed_decoder = _BitPackedCompiledDecoder()
    assert np.array_equal(
        bit_packed_decoder.decode_shots_bit_packed(
            bit_packed_detection_event_data=np.zeros((1, 1), dtype=np.uint8)
        ),
        [[0]],
    )
    assert decoders.is_prebuilt_observable_decoder(bit_packed_decoder)

    error_decoder = _FixedErrorDecoder([0, 0])
    assert np.array_equal(error_decoder.decode_errors(np.array([0])), [0, 0])
    assert not decoders.is_prebuilt_observable_decoder(error_decoder)
    assert not decoders.is_prebuilt_observable_decoder(decoders.lookup(max_weight=1))
    assert not decoders.is_prebuilt_observable_decoder(decoders.custom.ObservableLookupDecoder)
    sinter_decoder = decoders.TrivialDecoder()
    assert not isinstance(sinter_decoder, decoders.ObservableDecoder)
    assert not decoders.is_prebuilt_observable_decoder(sinter_decoder)
    compiled_sinter_decoder = sinter_decoder.compile_decoder_for_dem(stim.DetectorErrorModel())
    assert decoders.is_prebuilt_observable_decoder(compiled_sinter_decoder)
    assert not decoders.is_prebuilt_observable_decoder(decoders.relay_bp().build(matrix))
    assert decoders.compiles_for_dem(sinter_decoder)
    assert not decoders.compiles_for_dem(decoders.TrivialDecoder)
