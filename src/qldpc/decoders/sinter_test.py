"""Unit tests for sinter.py

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

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

import numpy as np
import pytest
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
    det_data, obs_data, err_data = sampler.sample(100)

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
    det_data, obs_data, err_data = sampler.sample(100)

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
     In this case, after decoding the window decoder has to map the error E0' back to E0 or E1 after
     decoding.  Note that E0 and E1 flip the same observable (L0), so the choice of E0 or E1 does not
     affect observable predictions.
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


def test_subgraph_decoder_with_erasure() -> None:
    """SubgraphDecoder appends one erasure observable per subgraph that has_erasure_bit."""
    # error 0 flips both D0 and D1 (so D0-alone is an unknown syndrome for subgraph 0)
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(0.1) D2 L1
    """)
    decoder = decoders.SubgraphDecoder(
        [[0, 1], [2]], with_lookup=True, max_weight=1, add_erasure_bit=True
    )
    compiled = decoder.compile_decoder_for_dem(dem)

    # two original observables plus one erasure observable per subgraph
    assert compiled.num_observables == dem.num_observables + 2

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


def test_sequential_window_decoder_erasure_not_implemented() -> None:
    """SequentialWindowDecoder raises NotImplementedError when has_erasure_bit is set."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D1 L1
    """)
    decoder = decoders.SequentialWindowDecoder(
        [[0], [1]], with_lookup=True, max_weight=1, add_erasure_bit=True
    )
    with pytest.raises(NotImplementedError, match="erasure"):
        decoder.compile_decoder_for_dem(dem)
