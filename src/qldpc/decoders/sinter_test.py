"""Unit tests for sinter.py.

Copyright 2025 The qLDPC Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import typing
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pytest
import sinter
import stim

from qldpc import decoders


def test_sinter_decoder() -> None:
    """Try out a simple decoding problem."""
    dem = stim.DetectorErrorModel("""
        error(0.0001) D0
        error(0.0002) D0 D1
        error(0.0003) D2 L1
    """)

    # mock some circuit errors associated and observable flips
    # each row of circuit_errors indicates which error mechanisms fired in a shot
    circuit_errors = [[1, 0, 0], [1, 1, 0], [1, 0, 1]]
    observable_flips = [[0, 0], [0, 0], [0, 1]]

    bit_packed_shots = np.packbits(circuit_errors, bitorder="little", axis=1)
    expected_flips = np.packbits(observable_flips, bitorder="little", axis=1)

    # try decoders with and without a decode_batch method
    for decoder in [
        decoders.SinterDecoder(with_BP_OSD=True),
        decoders.SinterDecoder(with_RBP="MinSumBPDecoderF32"),
        decoders.SinterDecoder(with_MWPM=True),
    ]:
        compiled_decoder = decoder.compile_decoder_for_dem(dem)
        predicted_flips = compiled_decoder.decode_shots_bit_packed(bit_packed_shots)
        assert np.array_equal(predicted_flips, expected_flips)

        # decode one shot at a time
        with pytest.raises(decoders.sinter.DecoderNotCompiledError, match="needs to be compiled"):
            decoder.decode(np.array([], dtype=int))
        assert np.array_equal(
            [compiled_decoder.decode(np.asarray(error)) for error in circuit_errors],
            observable_flips,
        )

    # the trivial decoder always returns a trivial result
    decoder = decoders.TrivialDecoder()
    compiled_decoder = decoder.compile_decoder_for_dem(dem)
    assert np.array_equal(
        compiled_decoder.decode_shots(np.array(circuit_errors)),
        np.zeros_like(observable_flips),
    )
    assert np.array_equal(
        compiled_decoder.decode_shots_bit_packed(bit_packed_shots),
        np.zeros_like(expected_flips),
    )


def test_subgraph_decoding() -> None:
    """Decode by parts."""
    # construct a simple detector error model and sample from it
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D1 L1
        error(0.1) D2 L2
    """)
    sampler = dem.compile_sampler()
    det_data, obs_data, _err_data = sampler.sample(100)

    # build a monolithic lookup-table decoder, compile, and predict observable flips
    decoder_1 = decoders.SinterDecoder(with_lookup=True, max_weight=3)
    compiled_decoder_1 = decoder_1.compile_decoder_for_dem(dem)
    predicted_flips_1 = compiled_decoder_1.decode_shots_bit_packed(
        compiled_decoder_1.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, compiled_decoder_1.packbits(obs_data))

    # build a subgraph decoder, compile, and predict observable flips
    decoder_2 = decoders.SubgraphDecoder([[0], [1], [2]], with_lookup=True, max_weight=1)
    compiled_decoder_2 = decoder_2.compile_decoder_for_dem(dem)
    predicted_flips_2 = compiled_decoder_2.decode_shots_bit_packed(
        compiled_decoder_2.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, predicted_flips_2)

    # if passing a sequence of sets of observables, it needs to be equal to the number of segments
    with pytest.raises(ValueError, match="inconsistent"):
        decoders.SubgraphDecoder([[0], [1], [2]], [[0]])


def test_sequential_decoding() -> None:
    """Decode segments sequentially."""
    # construct a simple detector error model and sample from it
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

    # build a monolithic lookup-table decoder, compile, and predict observable flips
    decoder_1 = decoders.SinterDecoder(with_lookup=True, max_weight=3)
    compiled_decoder_1 = decoder_1.compile_decoder_for_dem(dem)
    predicted_flips_1 = compiled_decoder_1.decode_shots_bit_packed(
        compiled_decoder_1.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, compiled_decoder_1.packbits(obs_data))

    # build a sequential decoder, compile, and predict observable flips
    decoder_2 = decoders.SequentialWindowDecoder([[0], [1], [2]], with_lookup=True, max_weight=1)
    compiled_decoder_2 = decoder_2.compile_decoder_for_dem(dem)
    predicted_flips_2 = compiled_decoder_2.decode_shots_bit_packed(
        compiled_decoder_2.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, predicted_flips_2)

    # build an equivalent sliding window decoder, compile, and predict observable flips
    decoder_2 = decoders.SlidingWindowDecoder(1, 1, with_lookup=True, max_weight=1)
    compiled_decoder_2 = decoder_2.compile_decoder_for_dem(dem)
    predicted_flips_2 = compiled_decoder_2.decode_shots_bit_packed(
        compiled_decoder_2.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, predicted_flips_2)


def test_sliding_window_recompilation() -> None:
    """One SlidingWindowDecoder builds windows from the coordinates of each model it compiles for.

    Sinter reuses a single decoder object across the tasks of one collect job, so
    compile_decoder_for_dem must derive its time indices afresh every time.
    """

    def dem_with_times(times: Sequence[int]) -> stim.DetectorErrorModel:
        """A chain of two-detector errors whose detectors carry the given time coordinates."""
        dem = stim.DetectorErrorModel()
        for detector, time in enumerate(times):
            dem.append("detector", [time], [stim.DemTarget.relative_detector_id(detector)])
        for detector in range(len(times) - 1):
            targets = [stim.DemTarget.relative_detector_id(dd) for dd in [detector, detector + 1]]
            dem.append("error", 0.1, targets)
        return dem

    decoder = decoders.SlidingWindowDecoder(1, 1, with_lookup=True, max_weight=1)

    compiled = decoder.compile_decoder_for_dem(dem_with_times([0, 1, 2, 3]))
    assert list(compiled.window_detectors) == [[0], [1], [2], [3]]

    # two detectors per time index, so each window holds both detectors of its round
    compiled = decoder.compile_decoder_for_dem(dem_with_times([0, 0, 1, 1]))
    assert list(compiled.window_detectors) == [[0, 1], [2, 3]]

    # a model with more detectors than the first one is also compiled from its own coordinates
    compiled = decoder.compile_decoder_for_dem(dem_with_times([0, 1, 2, 3, 4]))
    assert list(compiled.window_detectors) == [[0], [1], [2], [3], [4]]


def test_sliding_window_time_coordinate() -> None:
    """SlidingWindowDecoder reads time from a detector coordinate that can be indexing time.

    Coordinates are assigned as a circuit is built, so a coordinate that indexes time never
    decreases from one detector to the next.  The first coordinate is read whenever it has that
    property; a first coordinate that decreases somewhere is spatial, and a later coordinate is
    read instead.  The circuits that stim generates place time last.
    """

    def dem_with_coords(coords: Sequence[Sequence[float]]) -> stim.DetectorErrorModel:
        """A chain of two-detector errors whose detectors carry the given coordinates."""
        dem = stim.DetectorErrorModel()
        for detector, detector_coords in enumerate(coords):
            dem.append(
                "detector", list(detector_coords), [stim.DemTarget.relative_detector_id(detector)]
            )
        for detector in range(len(coords) - 1):
            targets = [stim.DemTarget.relative_detector_id(dd) for dd in [detector, detector + 1]]
            dem.append("error", 0.1, targets)
        return dem

    decoder = decoders.SlidingWindowDecoder(1, 1, with_lookup=True, max_weight=1)

    # the first coordinate counts rounds, so it is read
    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(0, 7), (0, 7), (1, 7), (1, 7)]))
    assert list(compiled.window_detectors) == [[0, 1], [2, 3]]

    # the first coordinate is a position that repeats each round, so the second is read
    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(0, 0), (1, 0), (0, 1), (1, 1)]))
    assert list(compiled.window_detectors) == [[0, 1], [2, 3]]

    # two later coordinates could each be indexing time, so the first is read after all
    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(1, 0, 0), (0, 1, 1), (1, 2, 2)]))
    assert list(compiled.window_detectors) == [[1], [0, 2]]

    # the only later coordinate never varies, so the first is read after all
    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(1, 5), (0, 5), (1, 5)]))
    assert list(compiled.window_detectors) == [[1], [0, 2]]

    # a detector with no coordinates has nothing to read a time index from
    dem = stim.DetectorErrorModel("error(0.1) D0 D1")
    with pytest.raises(ValueError, match="no coordinates"):
        decoder.compile_decoder_for_dem(dem)

    # an explicit mapping is used regardless of what the coordinates say
    decoder = decoders.SlidingWindowDecoder(
        1, 1, detector_to_time=lambda det: det // 2, with_lookup=True, max_weight=1
    )
    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(0, 0), (1, 0), (0, 1), (1, 1)]))
    assert list(compiled.window_detectors) == [[0, 1], [2, 3]]


def test_sequential_decoding_with_merged_window_errors() -> None:
    """SequentialWindowDecoder wraps with _ExpandedWindowDecoder when window errors merge.

    Consider two globally distinct errors:
        E0: flips D0, D1, L0,
        E1: flips D0, D2, L0.
    If a window decoder is restricted to detector D0, both errors look identical:
        E0: flips D0, L0,
        E1: flips D0, L0.
    A window decoder may therefore merge these errors into one:
        E0': flips D0, L0.
     In this case, after decoding the window decoder has to map the error E0' back to E0 or E1
     after decoding.  Note that E0 and E1 flip the same observable (L0), so the choice of E0 or E1
     does not affect observable predictions.
    """

    dem = stim.DetectorErrorModel("""
        error(0.3) D0 D1 L0
        error(0.2) D0 D2 L0
    """)

    sinter_decoder = decoders.SequentialWindowDecoder([[0], [1, 2]], with_BP_OSD=True)
    compiled_sinter_decoder = sinter_decoder.compile_decoder_for_dem(dem)
    assert isinstance(
        compiled_sinter_decoder.window_decoders[0],
        decoders.sinter._ExpandedWindowDecoder,
    )

    # Check correctness on explicit shots: no error, E0, and E1 individually.
    # Both E0 and E1 flip L0, so either firing → L0 = 1.
    shots = np.array(
        [
            [0, 0, 0],  # no error      → L0 = 0
            [1, 1, 0],  # E0 fires      → L0 = 1
            [1, 0, 1],  # E1 fires      → L0 = 1
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(compiled_sinter_decoder.decode_shots(shots), [[0], [1], [1]])

    # decode the first detector: 0 syndrome -> no errors, 1 syndrome -> E1
    assert np.array_equal(compiled_sinter_decoder.window_decoders[0].decode(np.array([0])), [0, 0])
    assert np.array_equal(compiled_sinter_decoder.window_decoders[0].decode(np.array([1])), [0, 1])

    # an error that cannot occur is absent from the merged errors of a window
    dem = stim.DetectorErrorModel("""
        error(0.3) D0 D1 L0
        error(0.2) D0 D2 L0
        error(0) D0 D3 L1
    """)
    compiled_sinter_decoder = decoders.SequentialWindowDecoder(
        [[0], [1, 2, 3]], simplify=False, with_BP_OSD=True
    ).compile_decoder_for_dem(dem)
    # the decoded error is one that can occur, not the zero-probability error
    assert np.array_equal(
        compiled_sinter_decoder.window_decoders[0].decode(np.array([1])), [0, 1, 0]
    )


def test_rejected_decoder_arguments() -> None:
    """Decoder arguments that a SinterDecoder cannot honor are rejected."""
    with pytest.raises(ValueError, match="DEFUNCT"):
        decoders.SinterDecoder(priors_arg="error_channel")

    # a SinterDecoder converts decoded errors into observable flips itself
    dem = stim.DetectorErrorModel("""
        error(0.3) D0 L0 L1
        error(0.1) D1 L1
    """)
    decoder = decoders.SinterDecoder(with_lookup=True, max_weight=2, predict_observable_flips=True)
    with pytest.raises(ValueError, match="must predict errors"):
        decoder.compile_decoder_for_dem(dem)

    # window decoders are built the same way, so they reject it too
    window_decoder = decoders.SequentialWindowDecoder(
        [[0], [1]], with_lookup=True, max_weight=1, predict_observable_flips=True
    )
    with pytest.raises(ValueError, match="must predict errors"):
        window_decoder.compile_decoder_for_dem(dem)


def test_window_region_validation() -> None:
    """A SequentialWindowDecoder rejects window regions that it cannot decode."""
    with pytest.raises(ValueError, match="inconsistent"):
        decoders.SequentialWindowDecoder([[0], [1]], [[0]])

    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L1
        error(0.1) D2 L2
    """)
    decoder = decoders.SequentialWindowDecoder(
        [[0], [1, 2]], [[0, 1], [2]], with_lookup=True, max_weight=1
    )
    with pytest.raises(ValueError, match="cannot be decoded before"):
        decoder.compile_decoder_for_dem(dem)


def test_compiled_decoder_input_validation() -> None:
    """Compiled decoders reject inconsistent numbers of regions and detectors."""
    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(1) D1
        error(0.1) D0 L0
        error(0.1) D1 L1
    """)
    wide_shots = np.zeros((1, dem.num_detectors + 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="per subgraph"):
        decoders.CompiledSubgraphDecoder([[0]], [[0], [1]], [], 2, 2)
    subgraph_decoder = decoders.SubgraphDecoder(
        [[0], [1]], with_lookup=True, max_weight=1
    ).compile_decoder_for_dem(dem)
    with pytest.raises(ValueError, match="detectors per shot"):
        subgraph_decoder.decode_shots(wide_shots)

    window_decoder = decoders.SequentialWindowDecoder(
        [[0], [1]], with_lookup=True, max_weight=1
    ).compile_decoder_for_dem(dem)
    with pytest.raises(ValueError, match="per window"):
        decoders.CompiledSequentialWindowDecoder(window_decoder.dem_arrays, [[0]], [], [])
    with pytest.raises(ValueError, match="detectors per shot"):
        window_decoder.decode_shots_to_error(wide_shots)


def test_sliding_window_time_gaps() -> None:
    """A gap between time indices contributes no window of its own."""
    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(2) D1
        detector(4) D2
        error(0.1) D0 D1
        error(0.1) D1 D2
    """)
    decoder = decoders.SlidingWindowDecoder(1, 1, with_lookup=True, max_weight=1)
    compiled = decoder.compile_decoder_for_dem(dem)
    assert list(compiled.window_detectors) == [[0], [1], [2]]


def test_sliding_window_ignores_undecoded_detectors() -> None:
    """Only the detectors that get windowed need a time index."""
    # D2 has no coordinates, and D3's coordinates would disqualify the first as a time index
    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(1) D1
        detector D2
        detector(0, 5) D3
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D3
    """)
    decoder = decoders.SlidingWindowDecoder(1, 1, [[0, 1]], with_lookup=True, max_weight=1)
    compiled = decoder.compile_decoder_for_dem(dem)
    assert list(compiled.window_detectors) == [[0], [1]]

    # a coordinate-less detector that does get windowed is still rejected
    decoder = decoders.SlidingWindowDecoder(1, 1, [[1, 2]], with_lookup=True, max_weight=1)
    with pytest.raises(ValueError, match="no coordinates"):
        decoder.compile_decoder_for_dem(dem)


def test_sliding_window_validation() -> None:
    """A SlidingWindowDecoder rejects window shapes and time indices that it cannot use."""
    with pytest.raises(ValueError, match="window_size >= stride"):
        decoders.SlidingWindowDecoder(1, 2)

    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(1) D1
        error(0.1) D0 D1
    """)
    # a mapping that violates its annotation by handing back a non-integer time index
    detector_to_time = typing.cast("Callable[[int], int]", lambda detector: detector / 2)
    decoder = decoders.SlidingWindowDecoder(
        1, 1, detector_to_time=detector_to_time, with_lookup=True, max_weight=1
    )
    with pytest.raises(TypeError, match="non-integer"):
        decoder.compile_decoder_for_dem(dem)


def test_deprecated_aliases() -> None:
    """The deprecated aliases of the sinter decoders warn when they are used."""
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        assert decoders.SubgraphSinterDecoder([[0]]).simplify
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        assert decoders.SequentialSinterDecoder([[0]]).simplify


def test_sinter_decoder_with_erasure() -> None:
    """compile_decoder_for_dem expands the DEM with an erasure observable when has_erasure_bit."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.1) D1 L0
    """)
    decoder = decoders.SinterDecoder(with_lookup=True, max_weight=1, add_erasure_bit=True)
    compiled = decoder.compile_decoder_for_dem(dem)

    # one extra observable for the erasure bit
    assert compiled.dem_arrays.num_observables == dem.num_observables + 1

    # known syndromes: correct observables, erasure bit = 0
    shots = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    result = compiled.decode_shots(shots)
    assert result.shape == (2, dem.num_observables + 1)
    assert np.array_equal(result[:, :-1], [[0], [1]])  # L0: not flipped by D0, flipped by D1
    assert np.all(result[:, -1] == 0)

    # unknown syndrome (no weight-1 error explains both D0 and D1): erasure bit = 1
    assert compiled.decode_shots(np.array([[1, 1]], dtype=np.uint8))[0, -1] == 1


@pytest.mark.parametrize("num_observables", [2, 7, 8, 9, 16])
def test_erasure_signalled_in_an_added_byte(num_observables: int) -> None:
    """An erasure bit is bit-packed into one whole byte added past the observable flips."""
    dem = stim.DetectorErrorModel(
        "\n".join(f"error(0.1) D{oo} L{oo}" for oo in range(num_observables))
    )
    decoder = decoders.SinterDecoder(with_lookup=True, max_weight=1, add_erasure_bit=True)
    compiled = decoder.compile_decoder_for_dem(dem)

    # no weight-one error explains a syndrome of weight two, so that shot is erased
    erased_syndrome = np.zeros(num_observables, dtype=np.uint8)
    erased_syndrome[:2] = 1
    shots = np.array([np.zeros(num_observables, dtype=np.uint8), erased_syndrome])
    packed_flips = compiled.decode_shots_bit_packed(compiled.packbits(shots))

    # sinter discards a shot whose prediction is one byte wider than the observables require
    assert packed_flips.shape == (2, (num_observables + 7) // 8 + 1)
    assert packed_flips[0, -1] == 0
    assert packed_flips[1, -1] == 1


@pytest.mark.parametrize("num_observables", [7, 8])
def test_predict_observables_with_erasure(num_observables: int) -> None:
    """Predictions written to a file report observable flips alone, without the discard byte."""
    dem = stim.DetectorErrorModel(
        "\n".join(f"error(0.2) D{oo} L{oo}" for oo in range(num_observables))
    )
    detection_events = np.zeros((3, num_observables), dtype=bool)
    detection_events[1, 0] = True
    detection_events[2, :2] = True  # no weight-one error explains this syndrome, so it is erased

    decoder = decoders.SinterDecoder(with_lookup=True, max_weight=1, add_erasure_bit=True)
    predictions = sinter.predict_observables(
        dem=dem, dets=detection_events, decoder="qldpc", custom_decoders={"qldpc": decoder}
    )

    # an erased shot is reported with the flips its decoder predicts for it, which are trivial
    expected_flips = np.zeros((3, num_observables), dtype=int)
    expected_flips[1, 0] = 1
    assert np.array_equal(np.asarray(predictions, dtype=int), expected_flips)

    # predictions that fit neither the observables nor one added byte cannot be written
    class WideCompiledDecoder(decoders.CompiledSinterDecoder):
        """A compiled decoder whose bit-packed predictions are two bytes too wide."""

        def pack_observable_flips(
            self, observable_flips: npt.NDArray[np.uint8]
        ) -> npt.NDArray[np.uint8]:
            packed_flips = super().pack_observable_flips(observable_flips)
            padding = np.zeros((len(packed_flips), 2), dtype=np.uint8)
            return np.hstack([packed_flips, padding])

    class WideDecoder(decoders.SinterDecoder):
        """A decoder whose bit-packed predictions are two bytes too wide."""

        def compile_decoder_for_dem(
            self, dem: stim.DetectorErrorModel
        ) -> decoders.CompiledSinterDecoder:
            compiled = super().compile_decoder_for_dem(dem)
            return WideCompiledDecoder(compiled.dem_arrays, compiled.decoder)

    with pytest.raises(ValueError, match="bytes of observable flips per shot"):
        sinter.predict_observables(
            dem=dem,
            dets=detection_events,
            decoder="qldpc",
            custom_decoders={"qldpc": WideDecoder(with_lookup=True, max_weight=1)},
        )


def test_subgraph_partition_warnings() -> None:
    """Compiling a SubgraphDecoder warns about a partition whose predictions do not add up."""
    # both subgraphs witness error 0, and by default both own the observable that it flips
    contested_dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0")
    with pytest.warns(UserWarning, match="can be predicted by more than one subgraph"):
        decoders.SubgraphDecoder(
            [[0], [1]], with_lookup=True, max_weight=1
        ).compile_decoder_for_dem(contested_dem)

    # detector 1 belongs to no subgraph, so error 1 is never witnessed
    uncovered_dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D1 L0")
    with pytest.warns(UserWarning, match="belong to no subgraph"):
        decoders.SubgraphDecoder(
            [[0]], [[0]], with_lookup=True, max_weight=1
        ).compile_decoder_for_dem(uncovered_dem)

    # a partition that gives each subgraph only the observables its own detectors witness is silent
    sound_dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D1 L1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decoders.SubgraphDecoder(
            [[0], [1]], [[0], [1]], with_lookup=True, max_weight=1
        ).compile_decoder_for_dem(sound_dem)


def test_subgraph_decoder_with_erasure() -> None:
    """SubgraphDecoder collects one erasure bit per subgraph past the observables of the model."""
    # error 0 flips both D0 and D1 (so D0-alone is an unknown syndrome for subgraph 0)
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(0.1) D2 L1
    """)
    decoder = decoders.SubgraphDecoder(
        [[0, 1], [2]], with_lookup=True, max_weight=1, add_erasure_bit=True
    )
    compiled = decoder.compile_decoder_for_dem(dem)

    # the erasure bits sit past the observables of the model, which they do not add to
    assert compiled.num_observables == dem.num_observables
    assert compiled.num_erasure_bits == 2

    # known syndromes: correct logical observables, both erasure bits = 0
    shots = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8)
    result = compiled.decode_shots(shots)
    assert result.shape == (3, dem.num_observables + 2)
    assert np.array_equal(result[:, :2], [[1, 0], [0, 1], [0, 0]])  # L0, L1
    assert np.all(result[:, 2:] == 0)

    # D0 alone is not explained by any weight-1 error in subgraph 0
    # erasure_0 fires, erasure_1 does not
    unknown_result = compiled.decode_shots(np.array([[1, 0, 0]], dtype=np.uint8))
    assert unknown_result[0, 2] == 1  # erasure for subgraph 0
    assert unknown_result[0, 3] == 0  # no erasure for subgraph 1

    # every subgraph signals erasure in one shared added byte
    packed_flips = compiled.decode_shots_bit_packed(compiled.packbits(shots))
    assert packed_flips.shape == (3, 1 + 1)
    assert np.all(packed_flips[:, -1] == 0)
    packed_unknown = compiled.decode_shots_bit_packed(
        compiled.packbits(np.array([[1, 0, 0]], dtype=np.uint8))
    )
    assert packed_unknown[0, -1] == 1


def test_sequential_window_decoder_with_erasure() -> None:
    """An erased window erases the whole shot, through the one erasure bit the windows share."""
    # the only error is committed by the first window, so the second window explains nothing
    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(1) D1
        error(0.1) D0 D1 L0
    """)
    decoder = decoders.SequentialWindowDecoder([[0], [1]], with_GUF=True, add_erasure_bit=True)
    compiled = decoder.compile_decoder_for_dem(dem)
    assert compiled.num_observables == dem.num_observables
    assert compiled.num_erasure_bits == 1

    # the two syndromes the model can produce decode unerased; the two it cannot are erased
    shots = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.uint8)
    result = compiled.decode_shots(shots)
    assert result.shape == (4, dem.num_observables + 1)
    assert np.array_equal(result[:, 0], [0, 1, 0, 1])
    assert np.array_equal(result[:, -1], [0, 0, 1, 1])

    # the shared erasure bit reaches the byte in which sinter reads a discard
    packed_flips = compiled.decode_shots_bit_packed(compiled.packbits(shots))
    assert packed_flips.shape == (4, 1 + 1)
    assert np.array_equal(packed_flips[:, -1], [0, 0, 1, 1])

    # the net circuit error is reported without the erasure bit
    net_error, erased = compiled.decode_shots_to_error_and_erasure(shots)
    assert net_error.shape == (4, compiled.dem_arrays.num_errors)
    assert np.array_equal(net_error, compiled.decode_shots_to_error(shots))
    assert np.array_equal(erased, [False, False, True, True])


def test_sequential_window_decoder_erasure_with_merged_window_errors() -> None:
    """A window decoder that both merges window errors and erases is still expanded.

    An erasure bit is not an error mechanism, so it cannot count toward the width that decides
    whether a window decoder merged equivalent errors and needs its output expanded.
    """
    # restricted to detectors 0, 1 and 4, the first two errors both flip D0 and L0, so they merge,
    # and no error at all flips detector 4, so a syndrome there is explained by nothing
    dem = stim.DetectorErrorModel("""
        error(0.3) D0 D2 L0
        error(0.2) D0 D3 L0
        error(0.1) D1 L1
        detector D4
    """)
    compiled = decoders.SequentialWindowDecoder(
        [[0, 1, 4], [2, 3]], with_GUF=True, add_erasure_bit=True
    ).compile_decoder_for_dem(dem)

    window_decoder = compiled.window_decoders[0]
    assert isinstance(window_decoder, decoders.sinter._ExpandedWindowDecoder)
    assert window_decoder.has_erasure_bit

    # the expanded error spans every error of the window, followed by the erasure bit
    explained = window_decoder.decode(np.array([1, 1, 0]))
    assert len(explained) == dem.num_errors + 1
    assert explained[-1] == 0

    # the erasure bit survives the expansion, rather than being scattered over the errors
    erased = window_decoder.decode(np.array([0, 0, 1]))
    assert np.array_equal(erased, [0] * dem.num_errors + [1])
    assert np.array_equal(
        window_decoder.decode_batch(np.array([[1, 1, 0], [0, 0, 1]])), [explained, erased]
    )

    # and it reaches the erasure bit that the whole decoder reports
    predicted_flips = compiled.decode_shots(np.array([[0, 0, 0, 0, 1]], dtype=np.uint8))
    assert predicted_flips.shape == (1, dem.num_observables + 1)
    assert np.array_equal(predicted_flips, [[0, 0, 1]])
