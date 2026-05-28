"""Unit tests for dems.py

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
        detector D0
        detector D1
        detector D2
        logical_observable L0
        logical_observable L1
        error(0.001) D0
        error(0.002) D0 D1
        error(0.003) D2 L1
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem)
    assert dem.approx_equals(dem_arrays.to_dem(), atol=1e-10)
    assert dem_arrays.num_errors == 3
    assert dem_arrays.num_detectors == 3
    assert dem_arrays.num_observables == 2

    other_dem_arrays = decoders.DetectorErrorModelArrays.from_arrays(*dem_arrays.get_arrays())
    assert np.allclose(
        other_dem_arrays.detector_flip_matrix.todense(), dem_arrays.detector_flip_matrix.todense()
    )
    assert np.allclose(
        other_dem_arrays.observable_flip_matrix.todense(),
        dem_arrays.observable_flip_matrix.todense(),
    )
    assert np.allclose(other_dem_arrays.error_probs, dem_arrays.error_probs)

    # initialize from detector_flip_matrix only
    error_prob = 1e-3
    other_dem_arrays = decoders.DetectorErrorModelArrays.from_arrays(
        dem_arrays.get_arrays()[0], None, error_prob
    )
    assert np.allclose(
        other_dem_arrays.detector_flip_matrix.todense(), dem_arrays.detector_flip_matrix.todense()
    )
    assert other_dem_arrays.num_observables == 0
    assert np.allclose(other_dem_arrays.error_probs, [error_prob] * dem_arrays.num_detectors)


def test_simplify() -> None:
    """Simplify and merge errors."""

    # simplification rules exercised below:
    # - D0 D0 D0 → D0 (odd number of like targets collapses to one)
    # - error(0) and error(p) with even-count targets (D2 D2) are dropped
    # - two errors with identical syndromes merge via the XOR formula:
    #   p_combined = p1 + p2 - 2*p1*p2  (probability of an odd number of occurrences)
    #   e.g. 0.001 D0 and 0.003 D0  →  0.001 + 0.003 - 2*0.001*0.003 ≈ 0.004
    #        0.002 D0D3 and 0.004 D0D3  →  0.002 + 0.004 - 2*0.002*0.004 ≈ 0.006
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
        detector D0
        detector D1
        detector D2
        detector D3
        logical_observable L0
        logical_observable L1
        error(0.004) D0
        error(0.006) D0 D3
        error(0.005) L1
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem, simplify=True)
    assert simplified_dem.approx_equals(dem_arrays.to_detector_error_model(), atol=1e-4)
    assert dem_arrays.num_errors == 3
    assert dem_arrays.num_detectors == 4
    assert dem_arrays.num_observables == 2

    dem_arrays = decoders.DetectorErrorModelArrays.from_arrays(
        np.array([[1, 0, 1], [1, 1, 1]]), np.array([[1, 0, 1]]), np.ones(3) * 0.3
    )
    dem = stim.DetectorErrorModel("""
        detector D0
        detector D1
        logical_observable L0
        error(0.3) D0 D1 L0
        error(0.3) D1
        error(0.3) D0 D1 L0
    """)
    simplified_dem = stim.DetectorErrorModel("""
        detector D0
        detector D1
        logical_observable L0
        error(0.42) D0 D1 L0
        error(0.3) D1
    """)
    assert dem == dem_arrays.to_detector_error_model()
    assert simplified_dem == dem_arrays.simplified().to_detector_error_model()


def test_with_erasure() -> None:
    """Add erasure bits to a DetectorErrorModelArrays."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.2) D1 L0
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem)
    erasure_arrays = dem_arrays.with_erasure()

    # one new error mechanism and one new observable; detectors unchanged
    assert erasure_arrays.num_errors == dem_arrays.num_errors + 1
    assert erasure_arrays.num_observables == dem_arrays.num_observables + 1
    assert erasure_arrays.num_detectors == dem_arrays.num_detectors

    # erasure mechanism flips no detectors and has zero probability
    assert erasure_arrays.detector_flip_matrix[:, -1].nnz == 0
    assert erasure_arrays.error_probs[-1] == 0

    # erasure mechanism flips only the new erasure observable, not the original ones
    assert erasure_arrays.observable_flip_matrix[:-1, -1].nnz == 0
    assert erasure_arrays.observable_flip_matrix[-1, -1] == 1

    # original errors do not flip the new erasure observable
    assert erasure_arrays.observable_flip_matrix[-1, :-1].nnz == 0

    # original detector/observable matrices and error probs are preserved
    assert np.array_equal(
        erasure_arrays.detector_flip_matrix[:, :-1].todense(),
        dem_arrays.detector_flip_matrix.todense(),
    )
    assert np.array_equal(
        erasure_arrays.observable_flip_matrix[:-1, :-1].todense(),
        dem_arrays.observable_flip_matrix.todense(),
    )
    assert np.array_equal(erasure_arrays.error_probs[:-1], dem_arrays.error_probs)

    # with bits=2, the bottom-right block of observable_flip_matrix is a 2×2 identity
    erasure_arrays_2 = dem_arrays.with_erasure(bits=2)
    assert erasure_arrays_2.num_errors == dem_arrays.num_errors + 2
    assert erasure_arrays_2.num_observables == dem_arrays.num_observables + 2
    assert np.array_equal(
        erasure_arrays_2.observable_flip_matrix[-2:, -2:].todense(), np.eye(2, dtype=int)
    )


def test_post_selection() -> None:
    """Post select on some detectors."""
    dem = stim.DetectorErrorModel("""
        detector D0
        detector D1
        logical_observable L0
        logical_observable L1
        error(0.3) D0 D1 L0
        error(0.3) D1 L1
    """)
    post_selected_dem = stim.DetectorErrorModel("""
        detector D0
        logical_observable L0
        logical_observable L1
        error(0.3) D0 L1
    """)
    assert (
        post_selected_dem == decoders.DetectorErrorModelArrays(dem).post_selected_on([0]).to_dem()
    )
