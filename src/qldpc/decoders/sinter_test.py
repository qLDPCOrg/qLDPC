# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sinter.py."""

import typing
import warnings
from collections.abc import Sequence

import ldpc
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
        decoders.SinterDecoder(decoder=decoders.bp_osd()),
        decoders.SinterDecoder(decoder=decoders.min_sum_bp()),
        decoders.SinterDecoder(decoder=decoders.mwpm()),
    ]:
        compiled_decoder = decoder.compile_decoder_for_dem(dem)
        predicted_flips = compiled_decoder.decode_shots_bit_packed(bit_packed_shots)
        assert np.array_equal(predicted_flips, expected_flips)

        # decode no shots
        no_flips = compiled_decoder.decode_shots_bit_packed(bit_packed_shots[:0])
        assert no_flips.shape == (0, expected_flips.shape[1])

        # decode one shot at a time
        with pytest.raises(decoders.DecoderNotCompiledError, match="needs to be compiled"):
            decoder.decode_observables(np.array([], dtype=int))
        assert np.array_equal(
            [compiled_decoder.decode_observables(np.asarray(error)) for error in circuit_errors],
            observable_flips,
        )

    # a compiled decoder exposes the decoder that its settings build
    compiled_decoder = decoders.SinterDecoder().compile_decoder_for_dem(dem)
    assert isinstance(compiled_decoder.decoder, ldpc.BpOsdDecoder)
    assert isinstance(compiled_decoder.observable_decoder, decoders.ErrorsToObservablesDecoder)
    compiled_decoder = decoders.SinterDecoder(decoder=decoders.mwpm()).compile_decoder_for_dem(dem)
    assert compiled_decoder.decoder is compiled_decoder.observable_decoder

    # a compiled decoder can also be constructed directly from an error decoder
    error_decoder = decoders.get_decoder_lookup(dem, max_weight=3)
    compiled_decoder = decoders.CompiledSinterDecoder(
        decoders.DetectorErrorModelArrays(dem), error_decoder
    )
    assert compiled_decoder.decoder is error_decoder
    assert np.array_equal(
        compiled_decoder.decode_shots_bit_packed(bit_packed_shots), expected_flips
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


def test_sinter_decoder_correlated_matching() -> None:
    """A SinterDecoder can decode with correlated matching."""
    import pymatching

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.02,
        before_measure_flip_probability=0.02,
        after_reset_flip_probability=0.02,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    syndromes = circuit.compile_detector_sampler(seed=0).sample(1000)

    # compiled decoders keep the error decompositions that correlated matching relies on
    matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    expected_flips = matching.decode_batch(syndromes, enable_correlations=True)
    spec = decoders.mwpm(enable_correlations=True)
    for simplify in [True, False]:
        decoder = decoders.SinterDecoder(simplify=simplify, decoder=spec)
        compiled_decoder = decoder.compile_decoder_for_dem(dem)
        assert np.array_equal(compiled_decoder.decode_shots(syndromes), expected_flips)

    # correlated matching needs the decompositions that decompose_errors=True discards
    with pytest.raises(ValueError, match="decompose_errors=True discards"):
        decoders.SinterDecoder(decompose_errors=True, decoder=spec)


def test_sinter_decoder_classes_and_aliases() -> None:
    """Sinter decoders are observable decoders, and deprecated decoder names remain aliases."""
    for sinter_class, decoder_class in [
        (sinter.Decoder, decoders.SinterDecoder),
        (sinter.CompiledDecoder, decoders.CompiledSinterDecoder),
    ]:
        assert sinter_class in decoder_class.__mro__
        assert decoders.ObservableDecoder in decoder_class.__mro__

    with pytest.warns(DeprecationWarning, match="BatchDecoder is deprecated"):
        assert decoders.BatchDecoder is decoders.BatchErrorDecoder
    with pytest.warns(DeprecationWarning, match="Decoder is deprecated"):
        assert decoders.custom.Decoder is decoders.ErrorDecoder
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = decoders.NotADecoder  # type: ignore[attr-defined]

    # star imports, which retrieve every name in __all__, define the deprecated names
    with pytest.warns(DeprecationWarning, match="is deprecated"):
        star_imports = {name: getattr(decoders, name) for name in decoders.__all__}
    assert star_imports["SubgraphSinterDecoder"] is decoders.SubgraphDecoder
    assert star_imports["Decoder"] is decoders.ErrorDecoder

    # an uncompiled observable decoder cannot decode
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    # the deprecated decode methods of these classes are hidden from type checkers
    decoder: typing.Any = decoders.SinterDecoder(decoder=decoders.lookup_table(max_weight=1))
    with pytest.raises(decoders.DecoderNotCompiledError, match="compile_decoder_for_dem"):
        decoder.decode(np.array([1], dtype=int))

    compiled: typing.Any = decoder.compile_decoder_for_dem(dem)
    with pytest.warns(DeprecationWarning, match="decode is deprecated"):
        assert np.array_equal(compiled.decode(np.array([1], dtype=int)), [1])


def test_unsimplified_dense_decoder() -> None:
    """A dense decoder is built for the error mechanisms that its model declares.

    Merging equivalent mechanisms would leave the decoder inferring fewer errors than the compiled
    decoder maps onto observables.
    """
    # the first two mechanisms are equivalent, so merging them would drop a column
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D0 L0
        error(0.1) D1
    """)
    compiled = decoders.SinterDecoder(
        simplify=False, decoder=decoders.guf()
    ).compile_decoder_for_dem(dem)
    assert compiled.decode_shots(np.array([[1, 0]], dtype=np.uint8)).tolist() == [[1]]


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
    decoder_1 = decoders.SinterDecoder(decoder=decoders.lookup_table(max_weight=3))
    compiled_decoder_1 = decoder_1.compile_decoder_for_dem(dem)
    predicted_flips_1 = compiled_decoder_1.decode_shots_bit_packed(
        compiled_decoder_1.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, compiled_decoder_1.packbits(obs_data))

    # build a subgraph decoder, compile, and predict observable flips
    decoder_2 = decoders.SubgraphDecoder(
        [[0], [1], [2]], decoder=decoders.lookup_table(max_weight=1)
    )
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
    decoder_1 = decoders.SinterDecoder(decoder=decoders.lookup_table(max_weight=3))
    compiled_decoder_1 = decoder_1.compile_decoder_for_dem(dem)
    predicted_flips_1 = compiled_decoder_1.decode_shots_bit_packed(
        compiled_decoder_1.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, compiled_decoder_1.packbits(obs_data))

    # build a sequential decoder, compile, and predict observable flips
    decoder_2 = decoders.SequentialWindowDecoder(
        [[0], [1], [2]], decoder=decoders.lookup_table(max_weight=1)
    )
    compiled_decoder_2 = decoder_2.compile_decoder_for_dem(dem)
    predicted_flips_2 = compiled_decoder_2.decode_shots_bit_packed(
        compiled_decoder_2.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, predicted_flips_2)

    # build an equivalent sliding window decoder, compile, and predict observable flips
    decoder_2 = decoders.SlidingWindowDecoder(1, 1, decoder=decoders.lookup_table(max_weight=1))
    compiled_decoder_2 = decoder_2.compile_decoder_for_dem(dem)
    predicted_flips_2 = compiled_decoder_2.decode_shots_bit_packed(
        compiled_decoder_2.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, predicted_flips_2)

    # decode no shots with window decoders that decode one syndrome at a time
    no_flips = compiled_decoder_2.decode_shots(det_data[:0])
    assert no_flips.shape == (0, dem.num_observables)


def test_sliding_window_recompilation() -> None:
    """One SlidingWindowDecoder builds windows from the coordinates of each model it compiles for.

    Sinter reuses a single decoder object across the tasks of one collect job, so
    compile_decoder_for_dem must derive its time indices afresh every time.
    """

    one_round = stim.DetectorErrorModel("detector(0) D0\ndetector(0) D1\nerror(0.1) D0 D1")
    two_rounds = stim.DetectorErrorModel("detector(0) D0\ndetector(1) D1\nerror(0.1) D0 D1")

    decoder = decoders.SlidingWindowDecoder(1, 1, decoder=decoders.lookup_table(max_weight=1))
    decoder.compile_decoder_for_dem(two_rounds)

    # the second model puts both detectors in one window, read from its own coordinates
    compiled = decoder.compile_decoder_for_dem(one_round)
    assert list(compiled.window_detectors) == [[0, 1]]


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

    decoder = decoders.SlidingWindowDecoder(1, 1, decoder=decoders.lookup_table(max_weight=1))

    # the first coordinate counts rounds, so it is read in preference to any later one
    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(0, 0), (0, 1), (1, 2), (1, 3)]))
    assert list(compiled.window_detectors) == [[0, 1], [2, 3]]

    # the first coordinate is a position that repeats each round, so the second is read
    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(0, 0), (1, 0), (0, 1), (1, 1)]))
    assert list(compiled.window_detectors) == [[0, 1], [2, 3]]

    # qLDPC records (time, 0, check_index), so in a single round the constant first coordinate must
    # win over the monotone check index and keep the entire round in one window
    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(0, 0, 0), (0, 0, 1), (0, 0, 2)]))
    assert list(compiled.window_detectors) == [[0, 1, 2]]

    # two later coordinates could each be indexing time, so the first is read after all
    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(1, 0, 0), (0, 1, 1), (1, 2, 2)]))
    assert list(compiled.window_detectors) == [[1], [0, 2]]

    # the only later coordinate never varies, so the first is read after all
    compiled = decoder.compile_decoder_for_dem(dem_with_coords([(1, 5), (0, 5), (1, 5)]))
    assert list(compiled.window_detectors) == [[1], [0, 2]]


def test_sequential_decoding_with_merged_window_errors() -> None:
    """SequentialWindowDecoder wraps with ExpandedErrorDecoder when window errors merge.

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

    sinter_decoder = decoders.SequentialWindowDecoder([[0], [1, 2]], decoder=decoders.bp_osd())
    compiled_sinter_decoder = sinter_decoder.compile_decoder_for_dem(dem)
    assert isinstance(
        compiled_sinter_decoder.window_decoders[0],
        decoders.ExpandedErrorDecoder,
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
        [[0], [1, 2, 3]], simplify=False, decoder=decoders.bp_osd()
    ).compile_decoder_for_dem(dem)
    # the decoded error is one that can occur, not the zero-probability error
    assert np.array_equal(
        compiled_sinter_decoder.window_decoders[0].decode(np.array([1])), [0, 1, 0]
    )


def test_rejected_decoder_arguments() -> None:
    """Decoder arguments that an ObservableDecoder cannot honor are rejected."""
    with (
        pytest.warns(DeprecationWarning, match="free-form decoder options"),
        pytest.raises(ValueError, match="DEFUNCT"),
    ):
        decoders.SinterDecoder(priors_arg="error_channel")

    # a constructor of an observable decoder is built natively
    dem = stim.DetectorErrorModel("""
        error(0.3) D0 L0 L1
        error(0.1) D1 L1
    """)

    def build_observable_decoder(dem: stim.DetectorErrorModel) -> decoders.ObservableDecoder:
        return decoders.ObservableLookupDecoder(dem, 2)

    decoder = decoders.SinterDecoder(decoder=build_observable_decoder)
    compiled = decoder.compile_decoder_for_dem(dem)
    assert isinstance(compiled.observable_decoder, decoders.ObservableLookupDecoder)
    assert np.array_equal(compiled.decode_observables(np.array([1, 0])), [1, 1])

    # ... but window decoders need error decoders
    window_decoder = decoders.SequentialWindowDecoder(
        [[0], [1]],
        decoder=build_observable_decoder,  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="predicts observable flips rather than errors"):
        window_decoder.compile_decoder_for_dem(dem)

    # an observable decoder without a batch method decodes one shot at a time
    class SingleShotDecoder:
        def __init__(self, dem: stim.DetectorErrorModel) -> None:
            self.lookup = decoders.ObservableLookupDecoder(dem, 2)

        def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return self.lookup.decode_observables(syndrome)

    compiled = decoders.SinterDecoder(decoder=SingleShotDecoder).compile_decoder_for_dem(dem)
    assert np.array_equal(compiled.decode_shots(np.array([[1, 0], [0, 1]])), [[1, 1], [0, 1]])

    # a prebuilt observable decoder is rejected, like a prebuilt error decoder, because a
    # SinterDecoder builds a new decoder for each (simplified) detector error model
    with pytest.raises(ValueError, match="prebuilt decoder cannot be passed as decoder="):
        decoders.SinterDecoder(
            decoder=decoders.ObservableLookupDecoder(dem, 2)  # type: ignore[arg-type]
        )

    # the deprecated keyword API is rejected in the same way
    with pytest.warns(DeprecationWarning):
        decoder = decoders.SinterDecoder(
            with_lookup=True, max_weight=2, predict_observable_flips=True
        )
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(TypeError, match="observable flips rather than errors"),
    ):
        decoder.compile_decoder_for_dem(dem)

    # window decoders are built the same way, so they reject it too
    with pytest.warns(DeprecationWarning):
        window_decoder = decoders.SequentialWindowDecoder(
            [[0], [1]], with_lookup=True, max_weight=1, predict_observable_flips=True
        )
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(TypeError, match="observable flips rather than errors"),
    ):
        window_decoder.compile_decoder_for_dem(dem)


def test_observable_decoders_reject_prebuilt_decoders() -> None:
    """A prebuilt error decoder cannot be rebuilt for each model that an observable decoder sees."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    prebuilt = decoders.LookupDecoder(dem, max_weight=1)
    for build_decoder, reason in [
        (
            lambda: decoders.SinterDecoder(decoder=prebuilt),  # type: ignore[arg-type]
            "detector error model",
        ),
        (
            lambda: decoders.SubgraphDecoder([[0]], decoder=prebuilt),  # type: ignore[arg-type]
            "each subgraph",
        ),
        (
            lambda: decoders.SequentialWindowDecoder([[0]], decoder=prebuilt),  # type: ignore[arg-type]
            "each window",
        ),
        (
            lambda: decoders.SlidingWindowDecoder(1, 1, decoder=prebuilt),  # type: ignore[arg-type]
            "each window",
        ),
    ]:
        with pytest.raises(
            ValueError, match=f"prebuilt decoder cannot be passed as decoder=.*{reason}"
        ):
            build_decoder()
    with pytest.raises(TypeError, match="static_decoder argument has been removed"):
        decoders.SinterDecoder(static_decoder=prebuilt)

    # a decoder constructor is built for each model
    decoder = decoders.SequentialWindowDecoder(
        [[0]], decoder=lambda dem: decoders.LookupDecoder(dem, max_weight=1)
    )
    assert np.array_equal(
        decoder.compile_decoder_for_dem(dem).decode_observables(np.array([1])), [1]
    )


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
        [[0], [1, 2]], [[0, 1], [2]], decoder=decoders.lookup_table(max_weight=1)
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
        [[0], [1]], decoder=decoders.lookup_table(max_weight=1)
    ).compile_decoder_for_dem(dem)
    with pytest.raises(ValueError, match="detectors per shot"):
        subgraph_decoder.decode_shots(wide_shots)

    window_decoder = decoders.SequentialWindowDecoder(
        [[0], [1]], decoder=decoders.lookup_table(max_weight=1)
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
    decoder = decoders.SlidingWindowDecoder(1, 1, decoder=decoders.lookup_table(max_weight=1))
    compiled = decoder.compile_decoder_for_dem(dem)
    assert list(compiled.window_detectors) == [[0], [1], [2]]

    # a window is dropped for committing nothing, even when it detects something
    decoder = decoders.SlidingWindowDecoder(2, 1, decoder=decoders.lookup_table(max_weight=1))
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
    decoder = decoders.SlidingWindowDecoder(
        1, 1, [[0, 1]], decoder=decoders.lookup_table(max_weight=1)
    )
    compiled = decoder.compile_decoder_for_dem(dem)
    assert list(compiled.window_detectors) == [[0], [1]]

    # a coordinate-less detector that does get windowed is still rejected
    decoder = decoders.SlidingWindowDecoder(
        1, 1, [[1, 2]], decoder=decoders.lookup_table(max_weight=1)
    )
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

    # a mapping that hands back a non-integer time index
    def detector_to_time(detector: int) -> typing.Any:
        return detector / 2

    decoder = decoders.SlidingWindowDecoder(
        1, 1, detector_to_time=detector_to_time, decoder=decoders.lookup_table(max_weight=1)
    )
    with pytest.raises(TypeError, match="non-integer"):
        decoder.compile_decoder_for_dem(dem)

    # an integral time index is a time index whatever its type, as an array lookup returns
    times = np.array([0, 1])

    def array_lookup(detector: int) -> typing.Any:
        return times[detector]

    decoder = decoders.SlidingWindowDecoder(
        1, 1, detector_to_time=array_lookup, decoder=decoders.lookup_table(max_weight=1)
    )
    assert list(decoder.compile_decoder_for_dem(dem).window_detectors) == [[0], [1]]


def test_deprecated_aliases() -> None:
    """Deprecated aliases of sinter decoders warn when accessed, and refer to their replacements."""
    with pytest.warns(DeprecationWarning, match="SubgraphSinterDecoder is deprecated"):
        assert decoders.SubgraphSinterDecoder is decoders.SubgraphDecoder
    with pytest.warns(DeprecationWarning, match="SequentialSinterDecoder is deprecated"):
        assert decoders.sinter.SequentialSinterDecoder is decoders.SequentialWindowDecoder
    with pytest.warns(DeprecationWarning, match="Decoder is deprecated; use ErrorDecoder"):
        assert decoders.sinter.Decoder is decoders.ErrorDecoder


def test_native_observable_decoders_on_subgraphs() -> None:
    """Native observable decoders agree with converted error decoders, even without observables.

    A subgraph that owns no observables, such as one CSS sector of a memory experiment, decodes a
    detector error model without observables.
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=2, after_clifford_depolarization=0.01
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    detection_events = circuit.compile_detector_sampler(seed=0).sample(200).astype(np.uint8)
    num_detectors = dem.num_detectors
    subgraph_detectors = [range(num_detectors // 2), range(num_detectors // 2, num_detectors)]

    for spec in [
        decoders.mwpm(),
        decoders.relay_bp(),
        decoders.min_sum_bp(gamma0=0.5),
        decoders.lookup_table(max_weight=1),
    ]:
        assert spec.predicts_observables_natively
        predicted_flips = []
        decoder_inputs: list[decoders.DeferredObservableDecoderInput] = [spec, spec.build]
        for decoder in decoder_inputs:
            sinter_decoder = decoders.SubgraphDecoder(
                subgraph_detectors, [[0], []], decompose_errors=True, decoder=decoder
            )
            compiled = sinter_decoder.compile_decoder_for_dem(dem)
            predicted_flips.append(compiled.decode_shots(detection_events))
        assert predicted_flips[0].shape == (len(detection_events), dem.num_observables)
        assert np.array_equal(predicted_flips[0], predicted_flips[1]), spec


def test_sinter_decoder_with_erasure() -> None:
    """A compiled decoder appends an erasure bit to its predictions if its inner decoder erases."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.1) D1 L0
    """)
    decoder = decoders.SinterDecoder(
        decoder=decoders.lookup_table(max_weight=1, add_erasure_bit=True)
    )
    compiled = decoder.compile_decoder_for_dem(dem)

    # one erasure bit, predicted natively by an ObservableLookupDecoder
    assert isinstance(compiled.observable_decoder, decoders.ObservableLookupDecoder)
    assert compiled.num_observables == dem.num_observables
    assert compiled.num_erasure_bits == 1

    # known syndromes: correct observables, erasure bit = 0
    shots = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    result = compiled.decode_shots(shots)
    assert result.shape == (2, dem.num_observables + 1)
    assert np.array_equal(result[:, :-1], [[0], [1]])  # L0: not flipped by D0, flipped by D1
    assert np.all(result[:, -1] == 0)

    # unknown syndrome (no weight-1 error explains both D0 and D1): erasure bit = 1
    assert compiled.decode_shots(np.array([[1, 1]], dtype=np.uint8))[0, -1] == 1


@pytest.mark.parametrize("num_observables", [7, 8])
def test_erasure_signalled_in_an_added_byte(num_observables: int) -> None:
    """An erasure bit is bit-packed into one whole byte added past the observable flips."""
    dem = stim.DetectorErrorModel(
        "\n".join(f"error(0.1) D{oo} L{oo}" for oo in range(num_observables))
    )
    decoder = decoders.SinterDecoder(
        decoder=decoders.lookup_table(max_weight=1, add_erasure_bit=True)
    )
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

    decoder = decoders.SinterDecoder(
        decoder=decoders.lookup_table(max_weight=1, add_erasure_bit=True)
    )
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
            custom_decoders={"qldpc": WideDecoder(decoder=decoders.lookup_table(max_weight=1))},
        )


def test_subgraph_partition_warnings() -> None:
    """Compiling a SubgraphDecoder warns about a partition whose predictions do not add up."""
    # both subgraphs witness error 0, and by default both own the observable that it flips
    contested_dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0")
    with pytest.warns(UserWarning, match="can be predicted by more than one subgraph") as contested:
        decoders.SubgraphDecoder(
            [[0], [1]], decoder=decoders.lookup_table(max_weight=1)
        ).compile_decoder_for_dem(contested_dem)

    # the warning names the code that compiled the decoder, not the library that raised it
    assert all(warning.filename == __file__ for warning in contested)

    # detector 1 belongs to no subgraph, so error 1 is never witnessed
    uncovered_dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D1 L0")
    with pytest.warns(UserWarning, match="belong to no subgraph"):
        decoders.SubgraphDecoder(
            [[0]], [[0]], decoder=decoders.lookup_table(max_weight=1)
        ).compile_decoder_for_dem(uncovered_dem)

    # a partition that gives each subgraph only the observables its own detectors witness is silent
    sound_dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D1 L1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decoders.SubgraphDecoder(
            [[0], [1]],
            [[0], [1]],
            decoder=decoders.lookup_table(max_weight=1),
        ).compile_decoder_for_dem(sound_dem)


def test_subgraph_decoder_with_erasure() -> None:
    """SubgraphDecoder collects one erasure bit per subgraph past the observables of the model."""
    # error 0 flips both D0 and D1 (so D0-alone is an unknown syndrome for subgraph 0)
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(0.1) D2 L1
    """)
    decoder = decoders.SubgraphDecoder(
        [[0, 1], [2]], decoder=decoders.lookup_table(max_weight=1, add_erasure_bit=True)
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
    decoder = decoders.SequentialWindowDecoder(
        [[0], [1]], decoder=decoders.guf(add_erasure_bit=True)
    )
    compiled = decoder.compile_decoder_for_dem(dem)
    assert compiled.num_observables == dem.num_observables
    assert compiled.num_erasure_bits == 1

    # the two syndromes the model can produce decode unerased; the two it cannot are erased
    shots = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.uint8)
    result = compiled.decode_shots(shots)
    assert result.shape == (4, dem.num_observables + 1)
    assert np.array_equal(result[:, 0], [0, 1, 0, 1])
    assert np.array_equal(result[:, -1], [0, 0, 1, 1])

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
        [[0, 1, 4], [2, 3]], decoder=decoders.guf(add_erasure_bit=True)
    ).compile_decoder_for_dem(dem)

    window_decoder = compiled.window_decoders[0]
    assert isinstance(window_decoder, decoders.ExpandedErrorDecoder)
    assert window_decoder.has_erasure_bit

    # the expanded error spans every error of the window, followed by the erasure bit
    explained = window_decoder.decode(np.array([1, 1, 0]))
    assert explained[-1] == 0

    # the erasure bit survives the expansion, rather than being scattered over the errors
    erased = window_decoder.decode(np.array([0, 0, 1]))
    assert np.array_equal(erased, [0] * dem.num_errors + [1])
    assert np.array_equal(
        window_decoder.decode_batch(np.array([[1, 1, 0], [0, 0, 1]])), [explained, erased]
    )

    # and it reaches the erasure bit that the whole decoder reports
    predicted_flips = compiled.decode_shots(np.array([[0, 0, 0, 0, 1]], dtype=np.uint8))
    assert np.array_equal(predicted_flips, [[0, 0, 1]])
