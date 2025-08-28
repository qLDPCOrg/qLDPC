"""Unit tests for dem_arrays.py

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


def test_initialization() -> None:
    """Initialize DetectorErrorModelArray objects."""

    dem = stim.DetectorErrorModel("""
        error(0.001) D0
        error(0.002) D0 D1
        error(0.003) D2 L1
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem)
    assert dem.approx_equals(dem_arrays.to_detector_error_model(), atol=1e-10)
    assert dem_arrays.num_errors == 3
    assert dem_arrays.num_detectors == 3
    assert dem_arrays.num_observables == 2

    other_dem_arrays = decoders.DetectorErrorModelArrays.from_arrays(
        dem_arrays.detector_flip_matrix,
        dem_arrays.observable_flip_matrix,
        dem_arrays.error_probs,
    )
    assert np.allclose(
        other_dem_arrays.detector_flip_matrix.todense(), dem_arrays.detector_flip_matrix.todense()
    )
    assert np.allclose(
        other_dem_arrays.observable_flip_matrix.todense(),
        dem_arrays.observable_flip_matrix.todense(),
    )
    assert np.allclose(other_dem_arrays.error_probs, dem_arrays.error_probs)


def test_simplify() -> None:
    """Simplify and merge errors."""

    # merge errors
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
    dem_arrays = decoders.DetectorErrorModelArrays(dem)
    assert simplified_dem.approx_equals(dem_arrays.to_detector_error_model(), atol=1e-4)
    assert dem_arrays.num_errors == 3
    assert dem_arrays.num_detectors == 4
    assert dem_arrays.num_observables == 2

    dem_arrays = decoders.DetectorErrorModelArrays.from_arrays(
        np.array([[1, 0, 1], [1, 1, 1]]), np.array([[1, 0, 1]]), np.ones(3) * 0.3
    )
    dem = stim.DetectorErrorModel("""
        error(0.3) D0 D1 L0
        error(0.3) D1
        error(0.3) D0 D1 L0
    """)
    simplified_dem = stim.DetectorErrorModel("""
        error(0.42) D0 D1 L0
        error(0.3) D1
    """)
    assert dem == dem_arrays.to_detector_error_model()
    assert simplified_dem == dem_arrays.simplify().to_detector_error_model()
