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

from qldpc import circuits


def test_dem_arrays() -> None:
    """Basic functionality of DetectorErrorModelArrays."""

    # convert from/to a stim.DetectorErrorModel
    dem = stim.DetectorErrorModel("""
        error(0.001) D0
        error(0.002) D0 D1
        error(0.003) D2 L1
    """)
    dem_arrays = circuits.DetectorErrorModelArrays(dem)
    assert dem.approx_equals(dem_arrays.to_detector_error_model(), atol=1e-10)
    assert dem_arrays.num_errors == 3
    assert dem_arrays.num_detectors == 3
    assert dem_arrays.num_observables == 2

    # simplify and merge errors
    dem = stim.DetectorErrorModel("""
        error(0.001) D0 D0 D0
        error(0.002) D0 D3
        error(0.003) D0
        error(0.004) D0 D3
        error(0.005) L1
        error(0.5) D2 D2
        error(0) D1
    """)
    simplified_dem = stim.DetectorErrorModel("""
        error(0.004) D0
        error(0.006) D0 D3
        error(0.005) L1
    """)
    dem_arrays = circuits.DetectorErrorModelArrays(dem)
    assert simplified_dem.approx_equals(dem_arrays.to_detector_error_model(), atol=1e-4)
    assert dem_arrays.num_errors == 3
    assert dem_arrays.num_detectors == 4
    assert dem_arrays.num_observables == 2


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
    for sinter_decoder, priors_arg in [
        (circuits.SinterDecoder(with_BP_OSD=True), "error_channel"),
        (circuits.SinterDecoder(with_RBP="MinSumBPDecoderF32"), "error_priors"),
        (circuits.SinterDecoder(with_MWPM=True), "weights"),
    ]:
        assert sinter_decoder.priors_arg == priors_arg

        compiled_sinter_decoder = sinter_decoder.compile_decoder_for_dem(dem)
        predicted_flips = compiled_sinter_decoder.decode_shots_bit_packed(bit_packed_shots)
        assert np.array_equal(predicted_flips, expected_flips)
