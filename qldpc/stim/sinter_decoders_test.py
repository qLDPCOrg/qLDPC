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

import qldpc


def test_dem_arrays() -> None:
    """Conversion to/from DetectorErrorModelArrays."""
    dem = stim.DetectorErrorModel("""
        error(0.0001) D0
        error(0.0002) D0 D1
        error(0.0003) D2 L1
    """)
    dem_arrays = qldpc.stim.DetectorErrorModelArrays(dem)
    assert dem.approx_equals(dem_arrays.to_detector_error_model(), atol=1e-4)
    assert dem_arrays.num_errors == 3
    assert dem_arrays.num_detectors == 3
    assert dem_arrays.num_observables == 2

    # merge equivalent errors
    dem = stim.DetectorErrorModel("""
        error(0.0001) D0
        error(0.0002) D0 D3
        error(0.0003) D0
        error(0.0004) D0 D3
        error(0.0005) L1
    """)
    simplified_dem = stim.DetectorErrorModel("""
        error(0.0004) D0
        error(0.0006) D0 D3
        error(0.0005) L1
    """)
    dem_arrays = qldpc.stim.DetectorErrorModelArrays(dem)
    assert simplified_dem.approx_equals(dem_arrays.to_detector_error_model(), atol=1e-4)
    assert dem_arrays.num_errors == 3
    assert dem_arrays.num_detectors == 4
    assert dem_arrays.num_observables == 2


def test_sinter_decoder() -> None:
    """Default parameter setting for a SinterDecoder."""
    sinter_decoder = qldpc.stim.SinterDecoder(with_MWPM=True)
    assert sinter_decoder.error_probs_arg == "weights"

    sinter_decoder = qldpc.stim.SinterDecoder(with_BP_OSD=True)
    assert sinter_decoder.error_probs_arg == "error_channel"

    dem = stim.DetectorErrorModel("""
        error(0.0001) D0
        error(0.0002) D0 D1
        error(0.0003) D2 L1
    """)
    compiled_sinter_decoder = sinter_decoder.compile_decoder_for_dem(dem)
    circuit_errors = [[1, 0, 0], [1, 1, 0], [1, 0, 1]]
    observable_flips = [[0, 0], [0, 0], [0, 1]]

    bit_packed_shots = np.packbits(circuit_errors, bitorder="little", axis=1)
    expected_flips = np.packbits(observable_flips, bitorder="little", axis=1)
    predicted_observable_flips = compiled_sinter_decoder.decode_shots_bit_packed(bit_packed_shots)
    assert np.array_equal(predicted_observable_flips, expected_flips)
