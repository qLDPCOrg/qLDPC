"""Alternative representations of a Stim detector error model

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

from __future__ import annotations

import collections
from collections.abc import Collection

import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim


class DetectorErrorModelArrays:
    """Representation of a stim.DetectorErrorModel by a collection of arrays.

    A DetectorErrorModelArrays object organizes the data in a stim.DetectorErrorModel into:
    1. detector_flip_matrix: a binary matrix that maps circuit errors to detector flips,
    2. observable_flip_matrix: a binary matrix that maps circuit errors to observable flips, and
    3. error_probs: an array of probabilities of occurrence for each circuit error.

    A DetectorErrorModelArrays is almost one-to-one with a stim.DetectorErrorModel instance.  The
    only differences are that a DetectorErrorModelArrays (a) "merges" circuit errors that flip the
    same set of detectors and observables, and (b) does not preserve detector coordinate data.
    """

    detector_flip_matrix: scipy.sparse.csc_matrix  # maps errors to detector flips
    observable_flip_matrix: scipy.sparse.csc_matrix  # maps errors to observable flips
    error_probs: npt.NDArray[np.floating]  # probability of occurrence for each error

    def __init__(
        self, circuit_or_dem: stim.Circuit | stim.DetectorErrorModel, *, simplify: bool = True
    ) -> None:
        """Initialize from a stim.DetectorErrorModel."""
        dem = (
            circuit_or_dem.detector_error_model()
            if isinstance(circuit_or_dem, stim.Circuit)
            else circuit_or_dem
        )
        errors = DetectorErrorModelArrays.get_circuit_errors(dem)
        if simplify:
            errors = DetectorErrorModelArrays.get_merged_circuit_errors(errors)
        self.detector_flip_matrix, self.observable_flip_matrix, self.error_probs = (
            DetectorErrorModelArrays.get_arrays_from_errors(
                errors, dem.num_detectors, dem.num_observables
            )
        )

    def get_arrays(
        self,
    ) -> tuple[scipy.sparse.csc_matrix, scipy.sparse.csc_matrix, npt.NDArray[np.floating]]:
        """The arrays of this DetectorErrorModelArrays.

        Returns:
            detector_flip_matrix: a binary matrix that maps circuit errors to detector flips.
            observable_flip_matrix: a binary matrix that maps circuit errors to observable flips.
            error_probs: an array of probabilities of occurrence for each circuit error.
        """
        return self.detector_flip_matrix, self.observable_flip_matrix, self.error_probs

    @staticmethod
    def from_arrays(
        detector_flip_matrix: scipy.sparse.csc_matrix | npt.NDArray[np.int_],
        observable_flip_matrix: scipy.sparse.csc_matrix | npt.NDArray[np.int_] | None,
        error_probs: npt.NDArray[np.floating] | float,
    ) -> DetectorErrorModelArrays:
        """Initialize from arrays directly."""
        dem_arrays = object.__new__(DetectorErrorModelArrays)
        dem_arrays.detector_flip_matrix = scipy.sparse.csc_matrix(detector_flip_matrix)

        num_error_mechanisms = dem_arrays.detector_flip_matrix.shape[1]
        if observable_flip_matrix is None:
            shape = (0, num_error_mechanisms)
            dem_arrays.observable_flip_matrix = scipy.sparse.csc_matrix(shape, dtype=int)
        else:
            dem_arrays.observable_flip_matrix = scipy.sparse.csc_matrix(observable_flip_matrix)

        if isinstance(error_probs, float):
            dem_arrays.error_probs = np.array([error_probs] * num_error_mechanisms)
        else:
            dem_arrays.error_probs = np.asarray(error_probs)

        return dem_arrays

    @property
    def num_errors(self) -> int:
        """The number of distinct circuit errors."""
        return self.detector_flip_matrix.shape[1]

    @property
    def num_detectors(self) -> int:
        """The number of detectors that witness circuit errors."""
        return self.detector_flip_matrix.shape[0]

    @property
    def num_observables(self) -> int:
        """The number of tracked logical observables."""
        return self.observable_flip_matrix.shape[0]

    @staticmethod
    def get_circuit_errors(
        dem: stim.DetectorErrorModel,
    ) -> list[tuple[frozenset[int], frozenset[int], float]]:
        """Collect all circuit errors in a stim.DetectorErrorModel.

        Each circuit error is identified by:
        - a set of detectors that are flipped,
        - a set of observables that are flipped, and
        - a probability of occurrence.

        If a detector or observable appears multiple times in an error, its occurrences are reduced
        to the original value mod 2.
        """
        errors = []
        for instruction in dem.flattened():
            if instruction.type == "error":
                probability = instruction.args_copy()[0]
                targets = instruction.targets_copy()
                detectors = _values_that_occur_an_odd_number_of_times(
                    [target.val for target in targets if target.is_relative_detector_id()]
                )
                observables = _values_that_occur_an_odd_number_of_times(
                    [target.val for target in targets if target.is_logical_observable_id()]
                )
                errors.append((detectors, observables, probability))
        return errors

    @staticmethod
    def get_merged_circuit_errors(
        errors: list[tuple[frozenset[int], frozenset[int], float]],
    ) -> list[tuple[frozenset[int], frozenset[int], float]]:
        """Merge circuit errors that flip the same detectors and observables."""
        merged_errors: dict[tuple[frozenset[int], frozenset[int]], float] = {}
        for detector_ids, observable_ids, prob in errors:
            key = (detector_ids, observable_ids)
            previous_prob = merged_errors.get(key, 0.0)
            merged_errors[key] = previous_prob + prob - 2 * previous_prob * prob
        return [
            (detectors, observables, prob)
            for (detectors, observables), prob in merged_errors.items()
            if (detectors or observables) and prob  # drop inconsequential error mechanisms
        ]

    @staticmethod
    def get_arrays_from_errors(
        errors: list[tuple[frozenset[int], frozenset[int], float]],
        num_detectors: int,
        num_observables: int,
    ) -> tuple[scipy.sparse.csc_matrix, scipy.sparse.csc_matrix, npt.NDArray[np.floating]]:
        """Convert circuit errors into DetectorErrorModelArrays data."""
        # initialize empty arrays
        detector_flip_matrix = scipy.sparse.dok_matrix((num_detectors, len(errors)), dtype=np.uint8)
        observable_flip_matrix = scipy.sparse.dok_matrix(
            (num_observables, len(errors)), dtype=np.uint8
        )
        error_probs = np.zeros(len(errors), dtype=float)

        # iterate over and account for all circuit errors
        for error_index, (detector_ids, observable_ids, probability) in enumerate(errors):
            detector_flip_matrix[list(detector_ids), error_index] = 1
            observable_flip_matrix[list(observable_ids), error_index] = 1
            error_probs[error_index] = probability

        return detector_flip_matrix.tocsc(), observable_flip_matrix.tocsc(), error_probs

    def to_dem(self) -> stim.DetectorErrorModel:
        """Alias for self.to_detector_error_model()."""
        return self.to_detector_error_model()

    def to_detector_error_model(self) -> stim.DetectorErrorModel:
        """Convert this object into a stim.DetectorErrorModel."""
        dem = stim.DetectorErrorModel()

        # add detectors and observables
        for dd in range(self.num_detectors):
            dem += stim.DetectorErrorModel(f"detector D{dd}")
        for dd in range(self.num_observables):
            dem += stim.DetectorErrorModel(f"logical_observable L{dd}")

        # add errors
        for detector_vec, observable_vec, prob in zip(
            self.detector_flip_matrix.T, self.observable_flip_matrix.T, self.error_probs
        ):
            detectors = " ".join([f"D{dd}" for dd in sorted(detector_vec.nonzero()[1])])
            observables = " ".join([f"L{dd}" for dd in sorted(observable_vec.nonzero()[1])])
            dem += stim.DetectorErrorModel(f"error({prob}) {detectors} {observables}")

        return dem

    def simplified(self) -> DetectorErrorModelArrays:
        """Simplify this DetectorErrorModelArrays object by merging errors."""
        return DetectorErrorModelArrays(self.to_detector_error_model(), simplify=True)

    def post_selected_on(self, detectors: Collection[int]) -> DetectorErrorModelArrays:
        """Condition this detector error model on the given detectors being in 0 (untriggered).

        In effect, remove the given detectors and the error mechanisms that trigger them.
        """
        detectors = list(detectors)
        detectors_to_keep = np.ones(self.num_detectors, dtype=bool)
        detectors_to_keep[detectors] = False
        errors_to_keep = self.detector_flip_matrix[detectors].getnnz(axis=0) == 0
        return DetectorErrorModelArrays.from_arrays(
            self.detector_flip_matrix[detectors_to_keep][:, errors_to_keep],
            self.observable_flip_matrix[:, errors_to_keep],
            self.error_probs[errors_to_keep],
        )

    def with_erasure(self, bits: int = 1) -> DetectorErrorModelArrays:
        """Construct the DetectorErrorModelArrays obtained by adding erasure bits to the DEM.

        Each erasure bit is essentially a zero-probability error mechanism that flips no detectors,
        but flips one newly added observable.  The erasure bit thereby allows decoders to indicate
        erasure by flipping the erasure bit.
        """
        detector_flip_stack = [
            self.detector_flip_matrix,
            scipy.sparse.csc_matrix((self.num_detectors, bits)),
        ]
        detector_flip_matrix = scipy.sparse.hstack(detector_flip_stack, format="csc")

        observable_flip_blocks = [
            [self.observable_flip_matrix, None],
            [None, scipy.sparse.eye(bits, dtype=int, format="csc")],
        ]
        observable_flip_matrix = scipy.sparse.bmat(observable_flip_blocks, format="csc")

        return DetectorErrorModelArrays.from_arrays(
            detector_flip_matrix,
            observable_flip_matrix,
            np.hstack([self.error_probs, [0] * bits]),
        )


def _values_that_occur_an_odd_number_of_times(items: Collection[int]) -> frozenset[int]:
    """Subset of items that occur an odd number of times."""
    return frozenset([item for item, count in collections.Counter(items).items() if count % 2])
