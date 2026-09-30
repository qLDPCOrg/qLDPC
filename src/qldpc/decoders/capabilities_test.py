# SPDX-License-Identifier: Apache-2.0

"""Unit tests for capabilities.py."""

from __future__ import annotations

from typing import Any

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc import decoders
from qldpc.decoders import capabilities


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
    assert not decoders.is_prebuilt_observable_decoder(decoders.lookup_table(max_weight=1))
    assert not decoders.is_prebuilt_observable_decoder(decoders.ObservableLookupDecoder)
    assert not decoders.is_prebuilt_observable_decoder(decoders.TrivialDecoder())
    assert not decoders.is_prebuilt_observable_decoder(decoders.relay_bp().build(matrix))
    assert decoders.compiles_for_dem(decoders.TrivialDecoder())
    assert not decoders.compiles_for_dem(decoders.TrivialDecoder)

    class _UnannotatedConstructor:
        def __call__(self, dem: stim.DetectorErrorModel) -> Any:
            return None  # pragma: no cover

    def unresolved_constructor(dem: stim.DetectorErrorModel) -> Any:
        return None  # pragma: no cover

    unresolved_constructor.__annotations__["return"] = "MissingDecoder"

    def observable_constructor(dem: stim.DetectorErrorModel) -> decoders.ObservableDecoder:
        return _FixedObservableDecoder([0])

    assert capabilities.constructs_observable_decoder(_FixedObservableDecoder)
    assert capabilities.constructs_observable_decoder(observable_constructor)
    assert isinstance(observable_constructor(stim.DetectorErrorModel()), decoders.ObservableDecoder)
    assert not capabilities.constructs_observable_decoder(None)
    assert not capabilities.constructs_observable_decoder(lambda matrix: None)
    assert not capabilities.constructs_observable_decoder(_UnannotatedConstructor())
    assert not capabilities.constructs_observable_decoder(unresolved_constructor)
