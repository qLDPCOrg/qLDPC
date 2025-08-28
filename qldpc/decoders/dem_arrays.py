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
import itertools
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
    error_probs: npt.NDArray[np.float64]  # probability of occurrence for each error

    def __init__(self, dem: stim.DetectorErrorModel) -> None:
        """Initialize from a stim.DetectorErrorModel."""
        errors = DetectorErrorModelArrays.get_merged_circuit_errors(dem)

        # initialize empty arrays
        detector_flip_matrix = scipy.sparse.dok_matrix(
            (dem.num_detectors, len(errors)), dtype=np.uint8
        )
        observable_flip_matrix = scipy.sparse.dok_matrix(
            (dem.num_observables, len(errors)), dtype=np.uint8
        )
        self.error_probs = np.zeros(len(errors), dtype=float)

        # iterate over and account for all circuit errors
        for error_index, ((detector_ids, observable_ids), probability) in enumerate(errors.items()):
            detector_flip_matrix[list(detector_ids), error_index] = 1
            observable_flip_matrix[list(observable_ids), error_index] = 1
            self.error_probs[error_index] = probability

        self.detector_flip_matrix = detector_flip_matrix.tocsc()
        self.observable_flip_matrix = observable_flip_matrix.tocsc()

    @staticmethod
    def from_arrays(
        detector_flip_matrix: scipy.sparse.csc_matrix | npt.NDArray[np.float64],
        observable_flip_matrix: scipy.sparse.csc_matrix | npt.NDArray[np.float64],
        error_probs: npt.NDArray[np.float64],
    ) -> DetectorErrorModelArrays:
        """Initialize from arrays directly."""
        dem_arrays = object.__new__(DetectorErrorModelArrays)
        dem_arrays.detector_flip_matrix = scipy.sparse.csc_matrix(detector_flip_matrix)
        dem_arrays.observable_flip_matrix = scipy.sparse.csc_matrix(observable_flip_matrix)
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
    def get_merged_circuit_errors(
        dem: stim.DetectorErrorModel,
    ) -> dict[tuple[frozenset[int], frozenset[int]], float]:
        """Organize and merge circuit errors in a stim.DetectorErrorModel.

        Each circuit error is identified by:
        - a set of detectors that are flipped,
        - a set of observables that are flipped, and
        - a probability of occurrence.

        This method organizes circuit errors into a dictionary that looks like
            {(detector_ids, observable_ids): probability}}.
        Circuit errors that flip the same set of detectors and observables are merged.
        """
        # Collect all circuit errors in the stim.DetectorErrorModel, accounting for the possibility
        # of indistinguishable errors that flip the same sets of detectors and observables.
        errors = collections.defaultdict(list)
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
                if (detectors or observables) and probability:
                    errors[detectors, observables].append(probability)

        # Combine circuit errors to obtain a single probability of occurrence for each set of flipped
        # detectors and observables.
        return {
            detectors_observables: _probability_of_an_odd_number_of_events(probabilities)
            for detectors_observables, probabilities in errors.items()
        }

    def to_detector_error_model(self) -> stim.DetectorErrorModel:
        """Convert this object into a stim.DetectorErrorModel."""
        dem = stim.DetectorErrorModel()
        for prob, detector_vec, observable_vec in zip(
            self.error_probs, self.detector_flip_matrix.T, self.observable_flip_matrix.T
        ):
            detectors = " ".join([f"D{dd}" for dd in sorted(detector_vec.nonzero()[1])])
            observables = " ".join([f"L{dd}" for dd in sorted(observable_vec.nonzero()[1])])
            dem += stim.DetectorErrorModel(f"error({prob}) {detectors} {observables}")
        return dem

    def simplify(self) -> DetectorErrorModelArrays:
        """Simplify this DetectorErrorModelArrays object by merging errors."""
        return DetectorErrorModelArrays(self.to_detector_error_model())


def _values_that_occur_an_odd_number_of_times(items: Collection[int]) -> frozenset[int]:
    """Subset of items that occur an odd number of times."""
    return frozenset([item for item, count in collections.Counter(items).items() if count % 2])


def _probability_of_an_odd_number_of_events(event_probabilities: Collection[float]) -> float:
    """Identify the probability that an odd number of (otherwise independent) events occurs."""
    net_probability = 0.0
    num_events = len(event_probabilities)
    for num_events_that_occur in range(1, num_events + 1, 2):
        for events_that_occur in itertools.combinations(range(num_events), num_events_that_occur):
            probability_that_these_events_occur = np.prod(
                [
                    prob if event in events_that_occur else 1 - prob
                    for event, prob in enumerate(event_probabilities)
                ]
            )
            net_probability += float(probability_that_these_events_occur)
    return net_probability
