# SPDX-License-Identifier: Apache-2.0

"""Tests for sequential and sliding-window Sinter decoders."""

from __future__ import annotations

import typing
from collections.abc import Sequence

import numpy as np
import pytest
import stim

from qldpc import decoders


def test_sequential_decoding() -> None:
    """Decode segments sequentially."""
    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(1) D1
        detector(2) D2
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L1
        error(0.1) D2 L2
    """)
    sampler = dem.compile_sampler()
    det_data, obs_data, _err_data = sampler.sample(100)

    decoder_1 = decoders.SinterDecoder(decoder=decoders.lookup(max_weight=3))
    compiled_decoder_1 = decoder_1.compile_decoder_for_dem(dem)
    predicted_flips_1 = compiled_decoder_1.decode_shots_bit_packed(
        compiled_decoder_1.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, compiled_decoder_1.packbits(obs_data))

    decoder_2 = decoders.SequentialWindowDecoder(
        [[0], [1], [2]], decoder=decoders.lookup(max_weight=1)
    )
    compiled_decoder_2 = decoder_2.compile_decoder_for_dem(dem)
    predicted_flips_2 = compiled_decoder_2.decode_shots_bit_packed(
        compiled_decoder_2.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, predicted_flips_2)

    decoder_2 = decoders.SlidingWindowDecoder(1, 1, decoder=decoders.lookup(max_weight=1))
    compiled_decoder_2 = decoder_2.compile_decoder_for_dem(dem)
    predicted_flips_2 = compiled_decoder_2.decode_shots_bit_packed(
        compiled_decoder_2.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, predicted_flips_2)

    no_flips = compiled_decoder_2.decode_shots(det_data[:0])
    assert no_flips.shape == (0, dem.num_observables)


def test_sequential_decoding_with_merged_window_errors() -> None:
    """SequentialWindowDecoder expands errors that a window decoder merges."""
    dem = stim.DetectorErrorModel("""
        error(0.3) D0 D1 L0
        error(0.2) D0 D2 L0
    """)

    sinter_decoder = decoders.SequentialWindowDecoder([[0], [1, 2]], decoder=decoders.bp_osd())
    compiled_sinter_decoder = sinter_decoder.compile_decoder_for_dem(dem)
    assert isinstance(
        compiled_sinter_decoder.window_decoders[0],
        decoders.ExpandedErrorDecoder,
    )

    shots = np.array(
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(compiled_sinter_decoder.decode_shots(shots), [[0], [1], [1]])
    assert np.array_equal(compiled_sinter_decoder.window_decoders[0].decode(np.array([0])), [0, 0])
    assert np.array_equal(compiled_sinter_decoder.window_decoders[0].decode(np.array([1])), [0, 1])

    dem = stim.DetectorErrorModel("""
        error(0.3) D0 D1 L0
        error(0.2) D0 D2 L0
        error(0) D0 D3 L1
    """)
    compiled_sinter_decoder = decoders.SequentialWindowDecoder(
        [[0], [1, 2, 3]], simplify=False, decoder=decoders.bp_osd()
    ).compile_decoder_for_dem(dem)
    assert np.array_equal(
        compiled_sinter_decoder.window_decoders[0].decode(np.array([1])), [0, 1, 0]
    )


def test_sequential_window_decoder_with_erasure() -> None:
    """An erased window erases the whole shot through the shared erasure bit."""
    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(1) D1
        error(0.1) D0 D1 L0
    """)
    decoder = decoders.SequentialWindowDecoder(
        [[0], [1]], decoder=decoders.guf(add_erasure_bit=True)
    )
    compiled = decoder.compile_decoder_for_dem(dem)
    assert compiled.num_observables == dem.num_observables
    assert compiled.num_erasure_bits == 1

    shots = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.uint8)
    result = compiled.decode_shots(shots)
    assert result.shape == (4, dem.num_observables + 1)
    assert np.array_equal(result[:, 0], [0, 1, 0, 1])
    assert np.array_equal(result[:, -1], [0, 0, 1, 1])

    net_error, erased = compiled.decode_shots_to_error_and_erasure(shots)
    assert net_error.shape == (4, compiled.dem_arrays.num_errors)
    assert np.array_equal(net_error, compiled.decode_shots_to_error(shots))
    assert np.array_equal(erased, [False, False, True, True])


def test_sequential_window_decoder_erasure_with_merged_window_errors() -> None:
    """A window decoder that merges errors and erases is still expanded correctly."""
    dem = stim.DetectorErrorModel("""
        error(0.3) D0 D2 L0
        error(0.2) D0 D3 L0
        error(0.1) D1 L1
        detector D4
    """)
    compiled = decoders.SequentialWindowDecoder(
        [[0, 1, 4], [2, 3]], decoder=decoders.guf(add_erasure_bit=True)
    ).compile_decoder_for_dem(dem)

    window_decoder = compiled.window_decoders[0]
    assert isinstance(window_decoder, decoders.ExpandedErrorDecoder)
    assert window_decoder.has_erasure_bit

    explained = window_decoder.decode(np.array([1, 1, 0]))
    assert explained[-1] == 0

    erased = window_decoder.decode(np.array([0, 0, 1]))
    assert np.array_equal(erased, [0] * dem.num_errors + [1])
    assert np.array_equal(
        window_decoder.decode_batch(np.array([[1, 1, 0], [0, 0, 1]])), [explained, erased]
    )

    predicted_flips = compiled.decode_shots(np.array([[0, 0, 0, 0, 1]], dtype=np.uint8))
    assert np.array_equal(predicted_flips, [[0, 0, 1]])


def test_window_region_validation() -> None:
    """A SequentialWindowDecoder rejects regions that it cannot decode."""
    with pytest.raises(ValueError, match="inconsistent"):
        decoders.SequentialWindowDecoder([[0], [1]], [[0]])

    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L1
        error(0.1) D2 L2
    """)
    decoder = decoders.SequentialWindowDecoder(
        [[0], [1, 2]], [[0, 1], [2]], decoder=decoders.lookup(max_weight=1)
    )
    with pytest.raises(ValueError, match="cannot be decoded before"):
        decoder.compile_decoder_for_dem(dem)


def test_compiled_window_input_validation() -> None:
    """Compiled window decoders reject inconsistent regions and detector counts."""
    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(1) D1
        error(0.1) D0 L0
        error(0.1) D1 L1
    """)
    wide_shots = np.zeros((1, dem.num_detectors + 1), dtype=np.uint8)
    window_decoder = decoders.SequentialWindowDecoder(
        [[0], [1]], decoder=decoders.lookup(max_weight=1)
    ).compile_decoder_for_dem(dem)
    with pytest.raises(ValueError, match="per window"):
        decoders.sinter.CompiledSequentialWindowDecoder(window_decoder.dem_arrays, [[0]], [], [])
    with pytest.raises(ValueError, match="detectors per shot"):
        window_decoder.decode_shots_to_error(wide_shots)


def test_sliding_window_time_coordinate() -> None:
    """SlidingWindowDecoder reads time from a coordinate that can index time."""

    def dem_with_coords(coords: Sequence[Sequence[float]]) -> stim.DetectorErrorModel:
        dem = stim.DetectorErrorModel()
        for detector, detector_coords in enumerate(coords):
            dem.append(
                "detector", list(detector_coords), [stim.DemTarget.relative_detector_id(detector)]
            )
        for detector in range(len(coords) - 1):
            targets = [stim.DemTarget.relative_detector_id(dd) for dd in [detector, detector + 1]]
            dem.append("error", 0.1, targets)
        return dem

    decoder = decoders.SlidingWindowDecoder(1, 1, decoder=decoders.lookup(max_weight=1))

    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(0, 0), (0, 1), (1, 2), (1, 3)]))
    assert list(compiled.window_detectors) == [[0, 1], [2, 3]]

    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(0, 0), (1, 0), (0, 1), (1, 1)]))
    assert list(compiled.window_detectors) == [[0, 1], [2, 3]]

    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(0, 0, 0), (0, 0, 1), (0, 0, 2)]))
    assert list(compiled.window_detectors) == [[0, 1, 2]]

    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(1, 0, 0), (0, 1, 1), (1, 2, 2)]))
    assert list(compiled.window_detectors) == [[1], [0, 2]]

    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(1, 5), (0, 5), (1, 5)]))
    assert list(compiled.window_detectors) == [[1], [0, 2]]


def test_sliding_window_time_gaps() -> None:
    """A gap between time indices contributes no window of its own."""
    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(2) D1
        detector(4) D2
        error(0.1) D0 D1
        error(0.1) D1 D2
    """)
    decoder = decoders.SlidingWindowDecoder(1, 1, decoder=decoders.lookup(max_weight=1))
    compiled = decoder.compile_decoder_for_dem(dem)
    assert list(compiled.window_detectors) == [[0], [1], [2]]

    decoder = decoders.SlidingWindowDecoder(2, 1, decoder=decoders.lookup(max_weight=1))
    compiled = decoder.compile_decoder_for_dem(dem)
    assert list(compiled.window_detectors) == [[0], [1], [2]]


def test_sliding_window_ignores_undecoded_detectors() -> None:
    """Only detectors that get windowed need a time index."""
    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(1) D1
        detector D2
        detector(0, 5) D3
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D3
    """)
    decoder = decoders.SlidingWindowDecoder(1, 1, [[0, 1]], decoder=decoders.lookup(max_weight=1))
    compiled = decoder.compile_decoder_for_dem(dem)
    assert list(compiled.window_detectors) == [[0], [1]]

    decoder = decoders.SlidingWindowDecoder(1, 1, [[1, 2]], decoder=decoders.lookup(max_weight=1))
    with pytest.raises(ValueError, match="no coordinates"):
        decoder.compile_decoder_for_dem(dem)


def test_sliding_window_recompilation() -> None:
    """One SlidingWindowDecoder derives time indices afresh for every model."""
    one_round = stim.DetectorErrorModel("detector(0) D0\ndetector(0) D1\nerror(0.1) D0 D1")
    two_rounds = stim.DetectorErrorModel("detector(0) D0\ndetector(1) D1\nerror(0.1) D0 D1")

    decoder = decoders.SlidingWindowDecoder(1, 1, decoder=decoders.lookup(max_weight=1))
    decoder.compile_decoder_for_dem(two_rounds)
    compiled = decoder.compile_decoder_for_dem(one_round)
    assert list(compiled.window_detectors) == [[0, 1]]


def test_sliding_window_validation() -> None:
    """A SlidingWindowDecoder rejects window shapes and unusable time indices."""
    with pytest.raises(ValueError, match="window_size >= stride"):
        decoders.SlidingWindowDecoder(1, 2)

    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(1) D1
        error(0.1) D0 D1
    """)

    def detector_to_time(detector: int) -> typing.Any:
        return detector / 2

    decoder = decoders.SlidingWindowDecoder(
        1, 1, detector_to_time=detector_to_time, decoder=decoders.lookup(max_weight=1)
    )
    with pytest.raises(TypeError, match="non-integer"):
        decoder.compile_decoder_for_dem(dem)

    times = np.array([0, 1])

    def array_lookup(detector: int) -> typing.Any:
        return times[detector]

    decoder = decoders.SlidingWindowDecoder(
        1, 1, detector_to_time=array_lookup, decoder=decoders.lookup(max_weight=1)
    )
    assert list(decoder.compile_decoder_for_dem(dem).window_detectors) == [[0], [1]]
