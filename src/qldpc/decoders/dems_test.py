"""Unit tests for dems.py.

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

import numpy as np
import pytest
import scipy.sparse
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
        error(0.002) D0 ^ D1
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

    dem_arrays = decoders.DetectorErrorModelArrays.from_arrays(
        np.array([[1, 0, 1], [1, 1, 1]]), np.array([[1, 0, 1]]), 0.3
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

    # a decomposition whose components cancel leaves nothing to flip, so the error is dropped
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 ^ D0 D2 ^ D1 D2
        error(0.2) D0
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem, simplify=True)
    assert error_instructions(dem_arrays.to_dem()) == ["error(0.2) D0"]


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
    # post-selecting removes detectors, the errors that trigger them, and remaps detector IDs
    dem = stim.DetectorErrorModel("""
        detector D0
        detector D1
        detector D2
        error(0.1) D1 ^ D2
        error(0.2) D0 D1
    """)
    post_selected_dem = stim.DetectorErrorModel("""
        detector D0
        detector D1
        error(0.1) D0 ^ D1
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem)
    assert dem_arrays.post_selected_on([0]).to_dem() == post_selected_dem

    # if passed an integer order=2, pairs error mechanisms that whose post-selected detector flips
    # cancel out are added back to the detector error model
    prob = 0.1
    dem = stim.DetectorErrorModel(f"""
        detector D0
        detector D1
        logical_observable L0
        error({prob}) D0 D2
        error({prob}) D0 L0
        error({prob}) D1 D2
        error({prob}) D1 L0
    """)
    post_selected_dem = stim.DetectorErrorModel(f"""
        detector D0
        logical_observable L0
        error({2 * prob**2 - 2 * prob**4}) D0 L0
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem)
    assert dem_arrays.post_selected_on([0, 1], order=2).to_dem() == post_selected_dem

    # setting order=3 will recover combinations of four error mechanisms
    prob = 0.1
    dem = stim.DetectorErrorModel(f"""
        detector D0
        detector D1
        detector D2
        logical_observable L0
        error({prob}) D0 D1 L0
        error({prob}) D1 D2 L0
        error({prob}) D2 D0 L0
    """)
    post_selected_dem = stim.DetectorErrorModel(f"""
        logical_observable L0
        error({prob**3}) L0
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem)
    assert dem_arrays.post_selected_on([0, 1, 2], order=3).to_dem() == post_selected_dem

    # setting order=4 will recover combinations of four error mechanisms
    prob = 0.1
    dem = stim.DetectorErrorModel(f"""
        detector D0
        detector D1
        detector D2
        logical_observable L0
        error({prob}) D0 L0
        error({prob}) D1 L0
        error({prob}) D2 L0
        error({prob}) D0 D1 D2
    """)
    post_selected_dem = stim.DetectorErrorModel(f"""
        logical_observable L0
        error({prob**4}) L0
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem)
    assert (
        dem_arrays.post_selected_on([0, 1, 2], order=4)
        .to_dem()
        .approx_equals(post_selected_dem, atol=1e-10)
    )

    with pytest.raises(ValueError, match="order"):
        decoders.DetectorErrorModelArrays(dem).post_selected_on([0], order=0)

    # two identical errors that trigger a post-selected detector cancel completely, so their
    # co-occurrence flips nothing and is not added back
    dem_arrays = decoders.DetectorErrorModelArrays.from_arrays(
        np.array([[1, 1, 0], [0, 0, 1]]), None, 0.1
    )
    assert error_instructions(dem_arrays.post_selected_on([0], order=2).to_dem()) == [
        "error(0.1) D0"
    ]

    # an error whose components cancel on a post-selected detector survives, but loses a whole
    # decomposition component along with that detector, leaving no decomposition to suggest
    dem = stim.DetectorErrorModel("""
        detector D0
        detector D1
        error(0.1) D0 ^ D0 D1
        error(0.2) D1
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem, simplify=False)
    post_selected = dem_arrays.post_selected_on([0])
    assert not post_selected.suggested_decompositions
    assert error_instructions(post_selected.to_dem()) == ["error(0.1) D0", "error(0.2) D0"]

    # keep_detectors retains the post-selected detectors, leaving them untriggered
    dem = stim.DetectorErrorModel("""
        detector D0
        detector D1
        detector D2
        error(0.1) D1 ^ D2
        error(0.2) D0 D1
    """)
    kept_dem = stim.DetectorErrorModel("""
        detector D0
        detector D1
        detector D2
        error(0.1) D1 ^ D2
    """)
    assert (
        decoders.DetectorErrorModelArrays(dem).post_selected_on([0], keep_detectors=True).to_dem()
        == kept_dem
    )


def test_without_untriggered_detectors() -> None:
    """Drop detectors that no error mechanism triggers."""
    # with no dead detectors, an equivalent (independent) copy is returned
    dem_arrays = decoders.DetectorErrorModelArrays(stim.DetectorErrorModel("error(0.1) D0"))
    result = dem_arrays.without_untriggered_detectors()
    assert result is not dem_arrays
    assert result.to_dem() == dem_arrays.to_dem()

    # D1 is dead but appears in the decomposition D0 D1 ^ D2 D1, where it cancels; it must be
    # filtered out rather than remapped to a stale index (live detectors: D0 -> 0, D2 -> 1)
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 ^ D2 D1
        error(0.2) D0
    """)
    pruned = decoders.DetectorErrorModelArrays(dem).without_untriggered_detectors()
    assert pruned.num_detectors == 2
    assert pruned.suggested_decompositions[0] == frozenset(
        [decoders.FlipPattern([0]), decoders.FlipPattern([1])]
    )


def error_instructions(dem: stim.DetectorErrorModel) -> list[str]:
    """The error instructions of a detector error model, without its target declarations."""
    return [str(instruction) for instruction in dem if instruction.type == "error"]


def test_dropping_decomposed_detectors() -> None:
    """Dropping detectors leaves each suggested decomposition consistent with its error.

    The components of a decomposition are alternative manifestations of one error, so together they
    must flip exactly what that error flips.  Dropping detectors can leave two components flipping
    the same targets, or leave a component with nothing to flip, and either way the surviving
    components have to keep agreeing with the error's column of the flip matrices.
    """
    # both components collapse onto D2 and cancel, leaving D3 as the only flip
    dem = stim.DetectorErrorModel("error(0.1) D0 D2 ^ D1 D2 ^ D3")
    dropped = decoders.DetectorErrorModelArrays(dem).without_detectors([0, 1])
    assert error_instructions(dropped.to_dem()) == ["error(0.1) D1"]
    assert not dropped.suggested_decompositions

    # one component is left flipping only an observable, which no matching graph can represent
    dem = stim.DetectorErrorModel("error(0.1) D0 L0 ^ D1 ^ D2")
    dropped = decoders.DetectorErrorModelArrays(dem).without_detectors([0])
    assert error_instructions(dropped.to_dem()) == ["error(0.1) D0 D1 L0"]
    assert not dropped.suggested_decompositions

    # a decomposition whose components share an observable can cancel down to nothing at all
    dem = stim.DetectorErrorModel("error(0.1) D0 L0 ^ D1 L0")
    dropped = decoders.DetectorErrorModelArrays(dem).without_detectors([0, 1])
    assert dropped.num_errors == 0

    # a decomposition that survives intact keeps both components and stays emittable
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 ^ D2 D3")
    dropped = decoders.DetectorErrorModelArrays(dem).without_detectors([3])
    assert error_instructions(dropped.to_dem()) == ["error(0.1) D0 D1 ^ D2"]

    # post-selection remaps decompositions through the same helper
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D2 ^ D1 D2 ^ D3
        error(0.2) D4
    """)
    post_selected = decoders.DetectorErrorModelArrays(dem).post_selected_on([0, 1])
    assert error_instructions(post_selected.to_dem()) == ["error(0.2) D2"]


def test_dropping_untriggered_decomposed_detectors() -> None:
    """Dropping untriggered detectors does not change what the model samples.

    A detector that no error mechanism flips is deterministically 0, so removing it can only shrink
    the model.  Such a detector can still appear inside a decomposition, where it cancels.
    """
    dem = stim.DetectorErrorModel("""
        error(1) D0 D1 ^ D1 ^ D0 D2
        error(0.2) D1 D3
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem)
    assert dem_arrays.detector_flip_matrix[0].getnnz() == 0  # D0 is untriggered
    pruned = dem_arrays.without_untriggered_detectors()

    # sampling the pruned model matches sampling the original, minus the dropped detector
    shots = 20_000
    original = dem.compile_sampler(seed=1).sample(shots)[0].mean(axis=0)
    reduced = pruned.to_dem().compile_sampler(seed=1).sample(shots)[0].mean(axis=0)
    assert np.allclose(original[1:], reduced, atol=0.02)


def test_validating_arrays() -> None:
    """from_arrays checks that its arrays describe the same error mechanisms."""
    matrix = np.eye(2, dtype=np.uint8)

    # an integer probability is broadcast to all error mechanisms
    assert np.array_equal(
        decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 1).error_probs, [1.0, 1.0]
    )

    with pytest.raises(ValueError, match="observable flip matrix addresses"):
        decoders.DetectorErrorModelArrays.from_arrays(matrix, np.ones((1, 3), dtype=np.uint8), 0.1)

    with pytest.raises(ValueError, match="error probabilities of shape"):
        decoders.DetectorErrorModelArrays.from_arrays(matrix, None, np.array([0.1, 0.2, 0.3]))


def test_from_arrays_copies_its_inputs() -> None:
    """A DetectorErrorModelArrays built from arrays shares no state with them."""
    matrix = scipy.sparse.csc_matrix(np.eye(2, dtype=np.uint8))
    error_probs = np.array([0.1, 0.2])
    dem_arrays = decoders.DetectorErrorModelArrays.from_arrays(matrix, matrix, error_probs)
    matrix.data[:] = 0
    error_probs[:] = 0.5
    expected_dem = stim.DetectorErrorModel("""
        detector D0
        detector D1
        logical_observable L0
        logical_observable L1
        error(0.1) D0 L0
        error(0.2) D1 L1
    """)
    assert expected_dem.approx_equals(dem_arrays.to_dem(), atol=1e-10)

    # the dictionary of suggested decompositions is copied as well
    decompositions = {0: frozenset([decoders.FlipPattern([0]), decoders.FlipPattern([1])])}
    dem_arrays = decoders.DetectorErrorModelArrays.from_arrays(
        np.array([[1], [1]], dtype=np.uint8), None, 0.1, decompositions
    )
    assert dem_arrays.suggested_decompositions is not decompositions


def test_validating_suggested_decompositions() -> None:
    """A suggested decomposition passed to from_arrays must agree with the flip matrices."""
    matrix = np.array([[1], [1], [0]], dtype=np.uint8)

    # the components below flip D0 and D2, while the error itself flips D0 and D1
    components = frozenset([decoders.FlipPattern([0]), decoders.FlipPattern([2])])
    with pytest.raises(ValueError, match="flips detectors"):
        decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 0.1, {0: components})

    # the components do agree with the error, so this model is accepted
    components = frozenset([decoders.FlipPattern([0]), decoders.FlipPattern([1])])
    dem_arrays = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 0.1, {0: components})
    assert dem_arrays.suggested_decompositions == {0: components}

    # there is no error mechanism to decompose at index 1
    with pytest.raises(ValueError, match="with 1 error mechanisms"):
        decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 0.1, {1: components})


def test_to_circuit() -> None:
    """Round-trip a DEM through DetectorErrorModelArrays and to_circuit."""
    dem = stim.DetectorErrorModel("""
        detector D0
        detector D1
        logical_observable L0
        error(0.1) D0 D1
        error(0.2) D1 L0
    """)
    circuit = decoders.DetectorErrorModelArrays(dem).to_circuit()
    assert dem.approx_equals(decoders.DetectorErrorModelArrays(circuit).to_dem(), atol=1e-10)


def test_decomposing_errors() -> None:
    """Apply suggested decompositions to split errors into their components."""
    dem = stim.DetectorErrorModel("""
        error(0.001) D0
        error(0.002) D1 ^ D2
        error(0.001) D2
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem)

    # error(0.002) D1 ^ D2 splits into two; all four resulting components are distinct so
    # simplify=True (the default) does not merge any of them
    split_dem = stim.DetectorErrorModel("""
        detector D0
        detector D1
        detector D2
        error(0.001) D0
        error(0.002) D1
        error(0.002) D2
        error(0.001) D2
    """)
    assert dem_arrays.with_decomposed_errors(simplify=False).to_dem() == split_dem


def test_error_targets_dem_targets() -> None:
    """FlipPattern.dem_targets returns sorted DemTarget lists for detectors and observables."""
    targets = decoders.FlipPattern([2, 0, 1], [3, 1, 2, 2])
    det_targets, obs_targets = targets.dem_targets()
    assert det_targets == [stim.DemTarget.relative_detector_id(dd) for dd in (0, 1, 2)]
    assert obs_targets == [stim.DemTarget.logical_observable_id(oo) for oo in (1, 3)]

    empty_det, empty_obs = decoders.FlipPattern().dem_targets()
    assert empty_det == [] and empty_obs == []
