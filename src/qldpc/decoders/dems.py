"""Alternative representations of a Stim detector error model.

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

from __future__ import annotations

import collections
import dataclasses
import itertools
import numbers
from collections.abc import Collection, Hashable, Iterable
from collections.abc import Set as AbstractSet
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

HashableType = TypeVar("HashableType", bound=Hashable)


@dataclasses.dataclass(frozen=True, init=False)
class FlipPattern:
    """A set of flipped detectors and observables."""

    detectors: frozenset[int]
    observables: frozenset[int]

    def __init__(self, detectors: Iterable[int] = (), observables: Iterable[int] = ()) -> None:
        object.__setattr__(self, "detectors", _xor_reduce(detectors))
        object.__setattr__(self, "observables", _xor_reduce(observables))

    @classmethod
    def from_data(cls, detectors: AbstractSet[int], observables: AbstractSet[int]) -> FlipPattern:
        """Construct from sets, skipping the mod-2 pass at normal initialization."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "detectors", frozenset(detectors))
        object.__setattr__(instance, "observables", frozenset(observables))
        return instance

    def __xor__(self, other: FlipPattern) -> FlipPattern:
        return FlipPattern.from_data(
            self.detectors ^ other.detectors, self.observables ^ other.observables
        )

    def __bool__(self) -> bool:
        return bool(self.detectors) or bool(self.observables)

    def dem_targets(self) -> tuple[list[stim.DemTarget], list[stim.DemTarget]]:
        """Lists of stim.DemTarget objects for the flipped detectors and observables."""
        det_targets = [stim.DemTarget.relative_detector_id(dd) for dd in sorted(self.detectors)]
        obs_targets = [stim.DemTarget.logical_observable_id(oo) for oo in sorted(self.observables)]
        return det_targets, obs_targets


CircuitError = tuple[float, frozenset[FlipPattern]]


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

        (a) merges circuit errors with identical targets, where an error's suggested decomposition
            is part of its targets (which can be disabled with simplify=False), and
        (b) does not preserve detector coordinate data.
    """

    detector_flip_matrix: scipy.sparse.csc_matrix  # maps errors to detector flips
    observable_flip_matrix: scipy.sparse.csc_matrix  # maps errors to observable flips
    error_probs: npt.NDArray[np.floating]  # probability of occurrence for each error
    suggested_decompositions: dict[int, frozenset[FlipPattern]]

    def __init__(
        self,
        circuit_or_dem: stim.Circuit | stim.DetectorErrorModel,
        *,
        simplify: bool = True,
        decompose_errors: bool = False,
    ) -> None:
        """Initialize from a stim.Circuit or a stim.DetectorErrorModel.

        Args:
            circuit_or_dem: an error model, or a circuit whose error model is extracted with
                stim.Circuit.detector_error_model(approximate_disjoint_errors=True).  qLDPC noise
                channels may contain correlated ``ELSE_CORRELATED_ERROR`` chains, so the
                disjoint-error approximation is enabled for this convenience path.  A model
                extracted here carries no decomposition suggestions; to obtain those, extract it
                yourself by calling circuit.detector_error_model(decompose_errors=True) and pass
                the result.
            simplify: If True, merge equivalent error mechanisms (see
                DetectorErrorModelArrays.simplified).  Defaults to True.
            decompose_errors: If True, split every error into the components that the error model
                suggests for it, leaving errors with no suggestion alone.  Each component inherits
                the probability of the error it came from, and the correlation between components is
                discarded, so a split model addresses fewer detectors per error -- as a matching
                decoder requires -- at the cost of no longer sampling like the model it came from.
                Simplifying afterwards then merges components that coincide, combining their
                probabilities.  Defaults to False.
        """
        dem = (
            circuit_or_dem.detector_error_model(approximate_disjoint_errors=True)
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
        suggested_decompositions: dict[int, frozenset[FlipPattern]] | None = None,
        *,
        simplify: bool = False,
    ) -> DetectorErrorModelArrays:
        """Initialize from arrays directly.

        Args:
            detector_flip_matrix: binary matrix mapping errors (columns) to detector flips (rows).
            observable_flip_matrix: binary matrix mapping errors to observable flips, or None for
                zero observables.
            error_probs: per-error probabilities, or a single number broadcast to all errors.
            suggested_decompositions (optional): dictionary that maps an error (by index) into
                a frozenset of FlipPattern, one per suggested decomposition component.
            simplify: If True, return a simplified model with equivalent error mechanisms merged
                (see DetectorErrorModelArrays.simplified).  Defaults to False.

        The returned object shares no memory with the given arrays.
        """
        dem_arrays = object.__new__(DetectorErrorModelArrays)
        dem_arrays.detector_flip_matrix = _canonicalize_mod2(
            scipy.sparse.csc_matrix(detector_flip_matrix, dtype=np.uint8)
        )

        num_error_mechanisms = dem_arrays.detector_flip_matrix.shape[1]
        if observable_flip_matrix is None:
            shape = (0, num_error_mechanisms)
            dem_arrays.observable_flip_matrix = scipy.sparse.csc_matrix(shape, dtype=np.uint8)
        else:
            dem_arrays.observable_flip_matrix = _canonicalize_mod2(
                scipy.sparse.csc_matrix(observable_flip_matrix, dtype=np.uint8)
            )

        if isinstance(error_probs, numbers.Real):
            dem_arrays.error_probs = np.full(num_error_mechanisms, float(error_probs))
        else:
            dem_arrays.error_probs = np.array(error_probs)

        num_observable_columns = dem_arrays.observable_flip_matrix.shape[1]
        if num_observable_columns != num_error_mechanisms:
            raise ValueError(
                f"The observable flip matrix addresses {num_observable_columns} error mechanisms,"
                f" but the detector flip matrix addresses {num_error_mechanisms}"
            )
        if dem_arrays.error_probs.shape != (num_error_mechanisms,):
            raise ValueError(
                f"Got error probabilities of shape {dem_arrays.error_probs.shape} for a detector"
                f" error model with {num_error_mechanisms} error mechanisms"
            )

        dem_arrays.suggested_decompositions = dict(suggested_decompositions or {})
        if dem_arrays.suggested_decompositions:
            _validate_decompositions(
                dem_arrays.suggested_decompositions,
                dem_arrays.detector_flip_matrix,
                dem_arrays.observable_flip_matrix,
            )
        return dem_arrays.simplified() if simplify else dem_arrays

    def copy(self) -> DetectorErrorModelArrays:
        """Return an independent copy of this DetectorErrorModelArrays."""
        dem_arrays = object.__new__(DetectorErrorModelArrays)
        dem_arrays.detector_flip_matrix = self.detector_flip_matrix.copy()
        dem_arrays.observable_flip_matrix = self.observable_flip_matrix.copy()
        dem_arrays.error_probs = self.error_probs.copy()
        dem_arrays.suggested_decompositions = dict(self.suggested_decompositions)
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
        dem: stim.DetectorErrorModel, *, decompose_errors: bool = False
    ) -> list[CircuitError]:
        """Collect all circuit errors in a stim.DetectorErrorModel into a list.

        Each circuit error is nominally identified by:

            - a probability of occurrence,
            - a set of detectors that are flipped,
            - a set of observables that are flipped.

        In addition, a stim.DetectorErrorModel can come equipped with suggested decompositions of
        errors, which splits the detector/observable targets of an error into groups.  To
        accommodate decomposition suggestions, a circuit error is identified by

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

            components: list[FlipPattern] = []
            for targets in target_components:
                error_targets = FlipPattern(
                    (target.val for target in targets if target.is_relative_detector_id()),
                    (target.val for target in targets if target.is_logical_observable_id()),
                )
                if decompose_errors:
                    errors.append((probability, frozenset([error_targets])))
                else:
                    components.append(error_targets)

            if not decompose_errors:
                errors.append((probability, _xor_reduce(components)))

        return errors

    @staticmethod
    def get_merged_circuit_errors(errors: list[CircuitError]) -> list[CircuitError]:
        """Merge circuit errors that have the same targets.

        Targets include suggested decompositions, so two errors that flip the same detectors and
        observables stay distinct if they suggest different decompositions.
        """
        merged: dict[frozenset[FlipPattern], float] = {}
        for prob, targets in errors:
            previous_prob = merged.get(targets, 0.0)
            merged[targets] = previous_prob + prob - 2 * previous_prob * prob
        return [
            (prob, targets)
            for targets, prob in merged.items()
            if _combined_flips(targets) and prob  # drop inconsequential errors
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
            combined = _combined_flips(components)
            detector_flip_matrix[list(combined.detectors), error_index] = 1
            observable_flip_matrix[list(combined.observables), error_index] = 1
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
                components = sorted(
                    self.suggested_decompositions[error_index],
                    key=lambda component: (
                        sorted(component.detectors),
                        sorted(component.observables),
                    ),
                )
                for gg, component in enumerate(components):
                    if gg > 0:
                        targets.append(stim.DemTarget.separator())
                    det_targets, obs_targets = component.dem_targets()
                    targets.extend(det_targets + obs_targets)
            else:
                detectors = self.detector_flip_matrix[:, error_index].nonzero()[0]
                observables = self.observable_flip_matrix[:, error_index].nonzero()[0]
                det_targets = [stim.DemTarget.relative_detector_id(dd) for dd in detectors]
                obs_targets = [stim.DemTarget.logical_observable_id(oo) for oo in observables]
                targets = det_targets + obs_targets

            # use DemInstruction (rather than dem.append("error", prob, targets)) so that error
            # mechanisms with no targets -- valid, deterministically silent errors -- can be emitted
            dem.append(stim.DemInstruction("error", [prob], targets))

        return dem

    def to_circuit(self) -> stim.Circuit:
        """Convert this DEM to a synthetic stim.Circuit.

        Each error mechanism becomes a noisy measurement ``M(p)`` on a dedicated qubit. DETECTOR and
        OBSERVABLE_INCLUDE instructions then reference those measurements.

        The detector error model of that circuit reproduces this DEM up to reordering of error
        mechanisms, merging of mechanisms with identical flips, and omission of mechanisms that
        cannot be observed -- those with zero probability, whose ``M(0)`` is deterministic, and
        those that flip nothing.  Error indices and suggested decompositions are not preserved.
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

    def without_detectors(self, detectors: Collection[int]) -> DetectorErrorModelArrays:
        """Drop the given detectors.

        Also drop or merge error mechanisms that are empty or redundant.  Remaining detectors get
        re-indexed to range(num_remaining_detectors).
        """
        keep = np.ones(self.num_detectors, dtype=bool)
        keep[list(detectors)] = False
        if keep.all():
            return self.copy()
        old_to_new = np.cumsum(keep) - 1
        suggested_decompositions = {
            error_index: remapped
            for error_index, components in self.suggested_decompositions.items()
            if (remapped := _remap_decomposition_detectors(components, keep, old_to_new))
        }
        return DetectorErrorModelArrays.from_arrays(
            self.detector_flip_matrix[keep],
            self.observable_flip_matrix,
            self.error_probs,
            suggested_decompositions,
            simplify=True,
        )

    def without_untriggered_detectors(self) -> DetectorErrorModelArrays:
        """Drop all detectors that are not triggered by any error mechanism.

        Such detectors are deterministically 0, so removing them changes no sampling outcome -- it
        only shrinks the DEM.  Error mechanisms and observables are left untouched.
        """
        return self.without_detectors(np.flatnonzero(self.detector_flip_matrix.getnnz(axis=1) == 0))

    def with_decomposed_errors(self, *, simplify: bool = True) -> DetectorErrorModelArrays:
        """Split error mechanisms according to their suggested decompositions.

        Each error with a suggested decomposition is replaced by its individual components, each
        inheriting the same probability.  Errors without a decomposition are kept as-is.
        """
        return DetectorErrorModelArrays(
            self.to_detector_error_model(), simplify=simplify, decompose_errors=True
        )

    def post_selected_on(
        self, detectors: Collection[int], *, keep_detectors: bool = False, order: int = 1
    ) -> DetectorErrorModelArrays:
        """Condition this detector error model on the given detectors being in 0 (untriggered).

        The errors that trigger the post-selected detectors are removed from the DEM.
        The post-selected detectors are similarly removed unless keep_detectors is True.

        If order > 1, combinations of up to 'order' removed error mechanisms whose co-occurrence
        does not trigger any of the post-selected detectors are added back to the DEM as synthetic
        error mechanisms.
        """
        if not order >= 1:
            raise ValueError(f"The 'order' parameter must >= 1, not {order}")

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
                    remapped = _remap_decomposition_detectors(
                        components, detectors_to_keep, old_to_new_det
                    )
                    if remapped:
                        suggested_decompositions[int(old_to_new_err[old_err_idx])] = remapped

        # build the post-selected arrays
        detector_flip_matrix = self.detector_flip_matrix[detectors_to_keep][:, errors_to_keep]
        observable_flip_matrix = self.observable_flip_matrix[:, errors_to_keep]
        error_probs = self.error_probs[errors_to_keep]

        if order > 1:
            detector_flip_matrix, observable_flip_matrix, error_probs = (
                _with_higher_order_corrections(
                    self,
                    detector_flip_matrix,
                    observable_flip_matrix,
                    error_probs,
                    detectors,
                    detectors_to_keep,
                    errors_to_keep,
                    order,
                )
            )

        return DetectorErrorModelArrays.from_arrays(
            detector_flip_matrix,
            observable_flip_matrix,
            error_probs,
            suggested_decompositions,
            simplify=order > 1,
        )

    def with_erasure(self, bits: int = 1) -> DetectorErrorModelArrays:
        """Construct the DetectorErrorModelArrays obtained by adding erasure bits to the DEM.

        Each erasure bit is essentially a zero-probability error mechanism that flips no detectors,
        but flips one newly added observable.  The erasure bit thereby allows decoders to indicate
        erasure by flipping the erasure bit.

        Zero probability makes an erasure mechanism inconsequential to sampling, so simplification
        drops it.  Add erasure bits after simplified, without_detectors,
        without_untriggered_detectors, with_decomposed_errors, post_selected_on with order > 1, and
        to_circuit.
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


def _xor_reduce(items: Iterable[HashableType]) -> frozenset[HashableType]:
    """Subset of items that occur an odd number of times."""
    return frozenset([item for item, count in collections.Counter(items).items() if count % 2])


def _combined_flips(components: Iterable[FlipPattern]) -> FlipPattern:
    """Net flips of a collection of decomposition components."""
    combined = FlipPattern()
    for component in components:
        combined ^= component
    return combined


def _remap_decomposition_detectors(
    components: frozenset[FlipPattern],
    detectors_to_keep: npt.NDArray[np.bool_],
    old_to_new_det: npt.NDArray[np.int_],
) -> frozenset[FlipPattern]:
    """Remap detector indices within decomposition components, dropping removed detectors.

    Detectors not in detectors_to_keep are omitted rather than remapped, since old_to_new_det has no
    valid new index for them.  Components that coincide once their detectors are dropped cancel in
    pairs, and components left with nothing to flip are discarded, so the surviving components still
    flip exactly what the error they decompose flips.

    A decomposition is only informative if at least two components survive, and a component that
    flips no detectors cannot be an edge of a matching graph, so an empty frozenset is returned in
    both of those cases to indicate that the decomposition should be dropped.
    """
    remapped = _xor_reduce(
        FlipPattern.from_data(
            frozenset(int(old_to_new_det[dd]) for dd in targets.detectors if detectors_to_keep[dd]),
            targets.observables,
        )
        for targets in components
    )
    surviving = frozenset(filter(None, remapped))
    if len(surviving) < 2 or not all(component.detectors for component in surviving):
        return frozenset()
    return surviving


def _validate_decompositions(
    suggested_decompositions: dict[int, frozenset[FlipPattern]],
    detector_flip_matrix: scipy.sparse.csc_matrix,
    observable_flip_matrix: scipy.sparse.csc_matrix,
) -> None:
    """Check that each suggested decomposition flips exactly what its error mechanism flips.

    The components of a decomposition are alternative ways for one error to manifest, so their
    combined flips must agree with the error's column of the flip matrices.
    """
    num_errors = detector_flip_matrix.shape[1]
    for error_index, components in suggested_decompositions.items():
        if not 0 <= error_index < num_errors:
            raise ValueError(
                f"Suggested decomposition given for error {error_index} of a detector error model"
                f" with {num_errors} error mechanisms"
            )
        combined = _combined_flips(components)
        combined_detectors = combined.detectors
        combined_observables = combined.observables
        detectors = _column_support(detector_flip_matrix, error_index)
        observables = _column_support(observable_flip_matrix, error_index)
        if combined_detectors != detectors or combined_observables != observables:
            raise ValueError(
                f"The suggested decomposition of error {error_index} flips detectors"
                f" {sorted(combined_detectors)} and observables {sorted(combined_observables)},"
                f" but that error flips detectors {sorted(detectors)}"
                f" and observables {sorted(observables)}"
            )


def _column_support(matrix: scipy.sparse.csc_matrix, column: int) -> frozenset[int]:
    """Rows in which one column of a compressed-column matrix is nonzero.

    Read from the compressed arrays directly, which is much cheaper than slicing out the column.
    """
    start, stop = matrix.indptr[column], matrix.indptr[column + 1]
    return frozenset(int(row) for row in matrix.indices[start:stop])


def _canonicalize_mod2(matrix: scipy.sparse.csc_matrix) -> scipy.sparse.csc_matrix:
    """Collapse duplicate stored entries mod 2 and drop resulting zeros.

    The given matrix is left alone, so the result shares no memory with it.
    """
    matrix = matrix.copy()
    matrix.sum_duplicates()
    matrix.data %= 2
    matrix.eliminate_zeros()
    return matrix


def _with_higher_order_corrections(
    dem_arrays: DetectorErrorModelArrays,
    detector_flip_matrix: scipy.sparse.csc_matrix,
    observable_flip_matrix: scipy.sparse.csc_matrix,
    error_probs: npt.NDArray[np.floating],
    detectors_to_remove: list[int],
    detectors_to_keep: npt.NDArray[np.bool_],
    errors_to_keep: npt.NDArray[np.bool_],
    order: int,
) -> tuple[scipy.sparse.csc_matrix, scipy.sparse.csc_matrix, npt.NDArray[np.floating]]:
    """Extend post-selected arrays by recovering combinations of individually removed errors.

    Finds all combinations of up to order removed errors whose net flip on the post-selected
    detectors cancels, then appends them as new error mechanisms.
    """
    assert order > 1

    removed_error_indices = np.flatnonzero(~errors_to_keep)
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

    if order >= 3:
        all_errors = set(range(len(removed_error_indices)))
        seen_errors = set()
        for triggering_errors in removed_det_to_removed_errors:
            seen_errors.update(triggering_errors)
            other_errors = all_errors - seen_errors
            for pair in itertools.combinations(triggering_errors, 2):
                for num_rest in range(1, order - 1):
                    for rest in itertools.combinations(other_errors, num_rest):
                        comb = [*pair, *rest]
                        if not np.any(removed_det_flip_submatrix[:, comb].sum(axis=1) % 2):
                            combinations_to_add.add(frozenset(removed_error_indices[comb]))

            # A combination cancels on this detector only if an even number of its errors flip it,
            # and the errors of a later group flip no earlier detector, so the errors this detector
            # contributes come in even numbers: four, six, and so on, as well as the pairs above.
            for num_head in range(4, min(len(triggering_errors), order) + 1, 2):
                for head in itertools.combinations(triggering_errors, num_head):
                    for num_rest in range(order - num_head + 1):
                        for rest in itertools.combinations(other_errors, num_rest):
                            comb = [*head, *rest]
                            if not np.any(removed_det_flip_submatrix[:, comb].sum(axis=1) % 2):
                                combinations_to_add.add(frozenset(removed_error_indices[comb]))

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

    More specifically, for each detector, identify errors that: (1) trigger that detector, and (2)
    do not trigger any preceding detectors.
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
