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
from collections.abc import Collection, Hashable
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

HashableType = TypeVar("HashableType", bound=Hashable)

ErrorTargets = tuple[frozenset[int], frozenset[int]]  # flipped detectors and observables
CircuitError = tuple[float, frozenset[ErrorTargets]]


class DetectorErrorModelArrays:
    """Representation of a stim.DetectorErrorModel by a collection of arrays.

    A DetectorErrorModelArrays object organizes the data in a stim.DetectorErrorModel into:
        1. detector_flip_matrix: a binary matrix that maps circuit errors to detector flips,
        2. observable_flip_matrix: a binary matrix that maps circuit errors to observable flips, and
        3. error_probs: an array of probabilities of occurrence for each circuit error.

    In addition, DetectorErrorModelArrays keeps track of any suggestions that a
    stim.DetectorErrorModel provides for how to decompose errors.

    A DetectorErrorModelArrays is _almost_ one-to-one with a stim.DetectorErrorModel instance.  The
    primary differences are that a DetectorErrorModelArrays object
        (a) merges equivalent circuit errors (which can be disabled with simplify=False), and
        (b) does not preserve detector coordinate data.
    """

    detector_flip_matrix: scipy.sparse.csc_matrix  # maps errors to detector flips
    observable_flip_matrix: scipy.sparse.csc_matrix  # maps errors to observable flips
    error_probs: npt.NDArray[np.floating]  # probability of occurrence for each error
    suggested_decompositions: dict[int, frozenset[ErrorTargets]]

    def __init__(
        self,
        circuit_or_dem: stim.Circuit | stim.DetectorErrorModel,
        *,
        simplify: bool = True,
        decompose_errors: bool = False,
    ) -> None:
        """Initialize from a stim.DetectorErrorModel."""
        dem = (
            circuit_or_dem.detector_error_model()
            if isinstance(circuit_or_dem, stim.Circuit)
            else circuit_or_dem
        )
        errors = DetectorErrorModelArrays.get_circuit_errors(dem, decompose_errors=decompose_errors)
        if simplify:
            errors = DetectorErrorModelArrays.get_merged_circuit_errors(errors)
        self.detector_flip_matrix, self.observable_flip_matrix, self.error_probs = (
            DetectorErrorModelArrays.get_arrays_from_errors(
                errors, dem.num_detectors, dem.num_observables
            )
        )
        self.suggested_decompositions = {
            error_index: components
            for error_index, (_, components) in enumerate(errors)
            if len(components) > 1
        }

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
        suggested_decompositions: dict[int, frozenset[ErrorTargets]] | None = None,
        *,
        simplify: bool = False,
    ) -> DetectorErrorModelArrays:
        """Initialize from arrays directly.

        Args:
            detector_flip_matrix: binary matrix mapping errors (columns) to detector flips (rows).
            observable_flip_matrix: binary matrix mapping errors to observable flips, or None for
                zero observables.
            error_probs: per-error probabilities, or a single float broadcast to all errors.
            suggested_decompositions (optional): dictionary that maps an error (by index) into
                suggested decomposition components, reperesented by a frozenset of
                (detector_frozenset, observable_frozenset) tuples.
        """
        dem_arrays = object.__new__(DetectorErrorModelArrays)
        dem_arrays.detector_flip_matrix = scipy.sparse.csc_matrix(
            detector_flip_matrix, dtype=np.uint8
        )

        num_error_mechanisms = dem_arrays.detector_flip_matrix.shape[1]
        if observable_flip_matrix is None:
            shape = (0, num_error_mechanisms)
            dem_arrays.observable_flip_matrix = scipy.sparse.csc_matrix(shape, dtype=np.uint8)
        else:
            dem_arrays.observable_flip_matrix = scipy.sparse.csc_matrix(
                observable_flip_matrix, dtype=np.uint8
            )

        if isinstance(error_probs, float):
            dem_arrays.error_probs = np.array([error_probs] * num_error_mechanisms)
        else:
            dem_arrays.error_probs = np.asarray(error_probs)

        dem_arrays.suggested_decompositions = suggested_decompositions or {}
        return dem_arrays.simplified() if simplify else dem_arrays

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
        dem: stim.DetectorErrorModel, *, decompose_errors: bool = False
    ) -> list[CircuitError]:
        """Collect all circuit errors in a stim.DetectorErrorModel into a list.

        Each circuit error is nominally identified by:
            - a probability of occurrence,
            - a set of detectors that are flipped,
            - a set of observables that are flipped.
        In addition, a stim.DetectorErrorModel can come equipped with suggested decompositions of
        errors, which splits the detector/observable targets of an error into groups.  To accomodate
        decomposition suggestions, a circuit error is identified by
            - a probability of occurrence,
            - a set of (detector_set, observable_set) tuples, one per suggested component.
        Errors with no suggested decompositions have a single component.

        If decompose_errors is True, all errors are decomposed into single-component errors.

        If a detector or observable appears multiple times within one component, its occurrences
        are reduced to the original value mod 2.
        """
        errors: list[CircuitError] = []
        for instruction in dem.flattened():
            if instruction.type != "error":
                continue
            probability = instruction.args_copy()[0]

            # identify components that are split by target separators
            target_components: list[list[stim.DemTarget]] = [[]]
            for target in instruction.targets_copy():
                if target.is_separator():
                    target_components.append([])
                else:
                    target_components[-1].append(target)

            components: list[ErrorTargets] = []
            for targets in target_components:
                detectors = _values_that_occur_an_odd_number_of_times(
                    [target.val for target in targets if target.is_relative_detector_id()]
                )
                observables = _values_that_occur_an_odd_number_of_times(
                    [target.val for target in targets if target.is_logical_observable_id()]
                )
                if decompose_errors:
                    errors.append((probability, frozenset([(detectors, observables)])))
                else:
                    components.append((detectors, observables))

            if not decompose_errors:
                errors.append((probability, _values_that_occur_an_odd_number_of_times(components)))

        return errors

    @staticmethod
    def get_merged_circuit_errors(errors: list[CircuitError]) -> list[CircuitError]:
        """Merge circuit errors that have the same targets."""
        merged: dict[frozenset[ErrorTargets], float] = {}
        for prob, targets in errors:
            previous_prob = merged.get(targets, 0.0)
            merged[targets] = previous_prob + prob - 2 * previous_prob * prob
        return [
            (prob, targets)
            for targets, prob in merged.items()
            if any(det or obs for det, obs in targets) and prob  # drop inconsequential errors
        ]

    @staticmethod
    def get_arrays_from_errors(
        errors: list[CircuitError], num_detectors: int, num_observables: int
    ) -> tuple[scipy.sparse.csc_matrix, scipy.sparse.csc_matrix, npt.NDArray[np.floating]]:
        """Convert circuit errors into DetectorErrorModelArrays data."""
        # initialize empty arrays
        detector_flip_matrix = scipy.sparse.dok_matrix((num_detectors, len(errors)), dtype=np.uint8)
        observable_flip_matrix = scipy.sparse.dok_matrix(
            (num_observables, len(errors)), dtype=np.uint8
        )
        error_probs = np.zeros(len(errors), dtype=float)

        # iterate over and account for all circuit errors
        for error_index, (probability, components) in enumerate(errors):
            detector_ids = _values_that_occur_an_odd_number_of_times(
                [det for det_set, _ in components for det in det_set]
            )
            observable_ids = _values_that_occur_an_odd_number_of_times(
                [obs for _, obs_set in components for obs in obs_set]
            )
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
            dem.append("detector", [], [stim.DemTarget.relative_detector_id(dd)])
        for oo in range(self.num_observables):
            dem.append("logical_observable", [], [stim.DemTarget.logical_observable_id(oo)])

        # add errors
        for error_index, prob in enumerate(self.error_probs):
            if error_index in self.suggested_decompositions:
                targets = []
                target_groups = sorted(
                    [
                        (sorted(detectors), sorted(observables))
                        for detectors, observables in self.suggested_decompositions[error_index]
                    ],
                )
                for gg, (detectors, observables) in enumerate(target_groups):
                    if gg > 0:
                        targets.append(stim.DemTarget.separator())
                    det_targets = [stim.DemTarget.relative_detector_id(dd) for dd in detectors]
                    obs_targets = [stim.DemTarget.logical_observable_id(oo) for oo in observables]
                    targets.extend(det_targets)
                    targets.extend(obs_targets)
            else:
                detectors = self.detector_flip_matrix[:, error_index].nonzero()[0]
                observables = self.observable_flip_matrix[:, error_index].nonzero()[0]
                det_targets = [stim.DemTarget.relative_detector_id(dd) for dd in detectors]
                obs_targets = [stim.DemTarget.logical_observable_id(oo) for oo in observables]
                targets = det_targets + obs_targets

            dem.append("error", prob, targets)

        return dem

    def to_circuit(self) -> stim.Circuit:
        """Convert this DEM to a synthetic stim.Circuit with the same detector error model.

        Each error mechanism becomes a noisy measurement M(p) on a dedicated qubit.
        DETECTOR and OBSERVABLE_INCLUDE instructions then reference those measurements.
        """
        circuit = stim.Circuit()

        for error, prob in enumerate(self.error_probs):
            circuit.append("M", [error], float(prob))

        for det in range(self.num_detectors):
            triggers = self.detector_flip_matrix[det].nonzero()[1].tolist()
            targets = [stim.target_rec(trigger - self.num_errors) for trigger in triggers]
            circuit.append("DETECTOR", targets)

        for obs in range(self.num_observables):
            triggers = self.observable_flip_matrix[obs].nonzero()[1].tolist()
            targets = [stim.target_rec(trigger - self.num_errors) for trigger in triggers]
            circuit.append("OBSERVABLE_INCLUDE", targets, obs)

        return circuit

    def simplified(self) -> DetectorErrorModelArrays:
        """Simplify this DetectorErrorModelArrays object by merging errors."""
        return DetectorErrorModelArrays(self.to_detector_error_model(), simplify=True)

    def with_decomposed_errors(self, *, simplify: bool = True) -> DetectorErrorModelArrays:
        """Split error mechanisms according to their suggested decompositions.

        Each error with a suggested decomposition is replaced by its individual components, each
        inheriting the same probability.  Errors without a decomposition are kept as-is.
        """
        return DetectorErrorModelArrays(
            self.to_detector_error_model(), simplify=simplify, decompose_errors=True
        )

    def post_selected_on(
        self, detectors: Collection[int], *, keep_detectors: bool = False, order: int = 0
    ) -> DetectorErrorModelArrays:
        """Condition this detector error model on the given detectors being in 0 (untriggered).

        The errors that trigger the post-selected detectors are removed from the DEM.
        The post-selected detectors are similarly removed unless keep_detectors is True.
        If order > 0, combinations of up to 2*order removed error mechanisms are added back to the
        DEM as synthetic error mechanisms.
        """
        if not 0 <= order <= 2:
            raise ValueError(f"The 'order' parameter must be 0, 1, or 2, not {order}")

        # identify detectors to discard and errors to keep
        detectors = list(detectors)
        detectors_to_keep = np.ones(self.num_detectors, dtype=bool)
        if not keep_detectors:
            detectors_to_keep[detectors] = False
        errors_to_keep = self.detector_flip_matrix[detectors].getnnz(axis=0) == 0

        suggested_decompositions = {}
        if self.suggested_decompositions:
            old_to_new_det = np.cumsum(detectors_to_keep) - 1
            old_to_new_err = np.cumsum(errors_to_keep) - 1
            for old_err_idx, components in self.suggested_decompositions.items():
                if errors_to_keep[old_err_idx]:
                    new_err_idx = int(old_to_new_err[old_err_idx])
                    new_components = set()
                    for dets, obs in components:
                        new_dets = frozenset(
                            int(old_to_new_det[dd]) for dd in dets if detectors_to_keep[dd]
                        )
                        new_components.add((new_dets, obs))
                    suggested_decompositions[new_err_idx] = frozenset(new_components)

        # build the post-selected arrays
        detector_flip_matrix = self.detector_flip_matrix[detectors_to_keep][:, errors_to_keep]
        observable_flip_matrix = self.observable_flip_matrix[:, errors_to_keep]
        error_probs = self.error_probs[errors_to_keep]

        if order > 0:
            detector_flip_matrix, observable_flip_matrix, error_probs = (
                _get_post_selection_additions(
                    self,
                    detectors,
                    detectors_to_keep,
                    errors_to_keep,
                    order,
                    detector_flip_matrix,
                    observable_flip_matrix,
                    error_probs,
                )
            )

        return DetectorErrorModelArrays.from_arrays(
            detector_flip_matrix,
            observable_flip_matrix,
            error_probs,
            suggested_decompositions,
            simplify=order > 0,
        )

    def with_erasure(self, bits: int = 1) -> DetectorErrorModelArrays:
        """Construct the DetectorErrorModelArrays obtained by adding erasure bits to the DEM.

        Each erasure bit is essentially a zero-probability error mechanism that flips no detectors,
        but flips one newly added observable.  The erasure bit thereby allows decoders to indicate
        erasure by flipping the erasure bit.
        """
        detector_flip_stack = [
            self.detector_flip_matrix,
            scipy.sparse.csc_matrix((self.num_detectors, bits), dtype=np.uint8),
        ]
        detector_flip_matrix = scipy.sparse.hstack(detector_flip_stack, format="csc")

        observable_flip_blocks = [
            [self.observable_flip_matrix, None],
            [None, scipy.sparse.eye(bits, dtype=np.uint8, format="csc")],
        ]
        observable_flip_matrix = scipy.sparse.bmat(observable_flip_blocks, format="csc")

        return DetectorErrorModelArrays.from_arrays(
            detector_flip_matrix,
            observable_flip_matrix,
            np.hstack([self.error_probs, [0] * bits]),
            self.suggested_decompositions,
        )


def _values_that_occur_an_odd_number_of_times(
    items: Collection[HashableType],
) -> frozenset[HashableType]:
    """Subset of items that occur an odd number of times."""
    return frozenset([item for item, count in collections.Counter(items).items() if count % 2])


def _get_post_selection_additions(
    dem_arrays: DetectorErrorModelArrays,
    detectors_to_remove: list[int],
    detectors_to_keep: npt.NDArray[np.bool_],
    errors_to_keep: npt.NDArray[np.bool_],
    order: int,
    detector_flip_matrix: scipy.sparse.csc_matrix,
    observable_flip_matrix: scipy.sparse.csc_matrix,
    error_probs: npt.NDArray[np.floating],
) -> tuple[scipy.sparse.csc_matrix, scipy.sparse.csc_matrix, npt.NDArray[np.floating]]:
    """Extend post-selected arrays by recovering combinations of individually removed errors.

    Finds all combinations of up to 2*order removed errors whose net flip on the post-selected
    detectors cancels, then appends them as new error mechanisms.
    """
    assert 0 <= order <= 2

    removed_error_indices = np.where(~errors_to_keep)[0]
    removed_det_flip_submatrix = dem_arrays.detector_flip_matrix[
        np.ix_(detectors_to_remove, removed_error_indices.tolist())
    ]
    removed_det_to_removed_errors = _get_removed_det_to_removed_errors(removed_det_flip_submatrix)

    # identify pairs of removed errors to add back to the DEM
    combinations_to_add: set[frozenset[int]] = set()
    for triggering_errors in removed_det_to_removed_errors:
        for pair in itertools.combinations(triggering_errors, 2):
            if not np.any(removed_det_flip_submatrix[:, pair].sum(axis=1) % 2):
                combinations_to_add.add(frozenset(removed_error_indices[list(pair)]))

    if order >= 2:
        # identify quadruples of removed errors to add back to the DEM
        for pairs in _get_pairs_grouped_by_pattern(removed_det_flip_submatrix).values():
            for (e1, e2), (e3, e4) in itertools.combinations(pairs, 2):
                indices = frozenset(removed_error_indices[[e1, e2, e3, e4]])
                if len(indices) == 4:
                    combinations_to_add.add(indices)

    new_errors: dict[bytes, tuple[scipy.sparse.csc_matrix, scipy.sparse.csc_matrix, float]] = {}
    for comb_to_add in combinations_to_add:
        comb = sorted(comb_to_add)
        # identify detectors and observables that are flipped by this combination of errors
        det_flips = scipy.sparse.csc_matrix(
            dem_arrays.detector_flip_matrix[detectors_to_keep][:, comb].sum(axis=1) % 2
        )
        obs_flips = scipy.sparse.csc_matrix(
            dem_arrays.observable_flip_matrix[:, comb].sum(axis=1) % 2
        )
        if det_flips.nnz == 0 and obs_flips.nnz == 0:  # pragma: no cover
            continue

        # add this combination as a new error mechanism
        flip_pattern = det_flips.toarray().tobytes() + obs_flips.toarray().tobytes()
        prob = float(np.prod(dem_arrays.error_probs[comb]))
        if flip_pattern in new_errors:
            previous_prob = new_errors[flip_pattern][2]
            prob = previous_prob + prob - 2 * previous_prob * prob
        new_errors[flip_pattern] = (det_flips, obs_flips, prob)

    if new_errors:
        new_det_flips, new_obs_flips, new_probs = zip(*new_errors.values())
        detector_flip_matrix = scipy.sparse.hstack(
            [detector_flip_matrix, *new_det_flips], format="csc"
        )
        observable_flip_matrix = scipy.sparse.hstack(
            [observable_flip_matrix, *new_obs_flips], format="csc"
        )
        error_probs = np.hstack([error_probs, new_probs])

    return detector_flip_matrix, observable_flip_matrix, error_probs


def _get_removed_det_to_removed_errors(
    removed_det_flip_submatrix: scipy.sparse.csc_matrix,
) -> list[list[int]]:
    """Map each post-selected detector to removed errors that trigger it.

    More specifically, for each detector, identify errors that:
    (1) trigger that detector, and
    (2) do not trigger any preceding detectors.
    """
    seen_errors: set[int] = set()
    removed_det_to_removed_errors = []
    for row in removed_det_flip_submatrix:
        triggering_errors = scipy.sparse.find(row)[1]
        removed_det_to_removed_errors.append(
            [err for err in triggering_errors if err not in seen_errors]
        )
        seen_errors.update(triggering_errors.tolist())
    return removed_det_to_removed_errors


def _get_pairs_grouped_by_pattern(
    removed_det_flip_submatrix: scipy.sparse.csc_matrix,
) -> dict[bytes, list[tuple[int, int]]]:
    """Group pairs of removed-error column indices by their combined post-selected detector flips."""
    num_errors = removed_det_flip_submatrix.shape[1]
    flips_to_pairs: dict[bytes, list[tuple[int, int]]] = collections.defaultdict(list)
    for pair in itertools.combinations(range(num_errors), 2):
        flips = removed_det_flip_submatrix[:, list(pair)].sum(axis=1).A1 % 2
        flips_to_pairs[flips.tobytes()].append(pair)
    return flips_to_pairs
