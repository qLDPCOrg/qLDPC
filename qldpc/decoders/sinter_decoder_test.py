"""Unit tests for sinter_decoders.py

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
import stim

from qldpc import decoders


def test_sinter_decoder() -> None:
    """Try out a simple decoding problem."""
    dem = stim.DetectorErrorModel("""
        error(0.0001) D0
        error(0.0002) D0 D1
        error(0.0003) D2 L1
    """)
    circuit_errors = [[1, 0, 0], [1, 1, 0], [1, 0, 1]]
    observable_flips = [[0, 0], [0, 0], [0, 1]]
    bit_packed_shots = np.packbits(circuit_errors, bitorder="little", axis=1)
    expected_flips = np.packbits(observable_flips, bitorder="little", axis=1)

    # try decoders with and without a decode_batch method
    for decoder, priors_arg in [
        (decoders.SinterDecoder(with_BP_OSD=True), "error_channel"),
        (decoders.SinterDecoder(with_RBP="MinSumBPDecoderF32"), "error_priors"),
        (decoders.SinterDecoder(with_MWPM=True), "weights"),
    ]:
        assert decoder.priors_arg == priors_arg

        compiled_decoder = decoder.compile_decoder_for_dem(dem)
        predicted_flips = compiled_decoder.decode_shots_bit_packed(bit_packed_shots)
        assert np.array_equal(predicted_flips, expected_flips)


def test_segment_decoding() -> None:
    """Decode in segments."""
    # construct a simple detector error model and sample from it
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D1 L1
    """)
    sampler = dem.compile_sampler()
    det_data, obs_data, err_data = sampler.sample(100)

    # build a monolithic vs. segmented decoder
    decoder_1 = decoders.SinterDecoder(with_lookup=True, max_weight=2)
    decoder_2 = decoders.CompositeSinterDecoder(
        ([0], [0]), ([1], [1]), with_lookup=True, max_weight=2
    )

    # compile the decoders and predict observable flips
    compiled_decoder_1 = decoder_1.compile_decoder_for_dem(dem)
    compiled_decoder_2 = decoder_2.compile_decoder_for_dem(dem)
    predicted_flips_1 = compiled_decoder_1.decode_shots_bit_packed(
        compiled_decoder_1.packbits(det_data)
    )
    predicted_flips_2 = compiled_decoder_2.decode_shots_bit_packed(
        compiled_decoder_2.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, predicted_flips_2)
