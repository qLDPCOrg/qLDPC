from __future__ import annotations

import collections
import itertools
from collections.abc import Collection, Hashable
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import scipy
import scipy.sparse
import sinter
import stim

from qldpc import decoders


class CompiledSinterDecoder(sinter.CompiledDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit."""

    def __init__(self, dem_arrays: DemArrays, decoder: decoders.Decoder) -> None:
        self.num_detectors = dem_arrays.detector_flip_matrix.shape[0]
        self.dem_arrays = dem_arrays
        self.decoder = decoder

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        See help(sinter.CompiledDecoder) for additional information.
        """
        observable_flips = []
        for bit_packed_syndrome in bit_packed_detection_event_data:
            syndrome = np.unpackbits(
                bit_packed_syndrome, count=self.num_detectors, bitorder="little"
            )
            predicted_errors = self.decoder.decode(syndrome)
            observable_flips.append(self.dem_arrays.observable_flip_matrix @ predicted_errors % 2)
        return np.packbits(np.array(observable_flips, dtype=np.uint8), bitorder="little", axis=1)


class SinterDecoder(sinter.Decoder):
    """Decoder usable by Sinter for decoding circuit errors."""

    def __init__(self, *, priors_arg: str | None = None, **decoder_kwargs: object) -> None:
        """Initialize a SinterDecoder.

        Args:
            prior_args: The keyword argument to which to pass priors about circuit-level error
                likelihoods when constructing a decoder with qldpc.decoders.get_decoder.
            decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        self.priors_arg = priors_arg
        self.decoder_kwargs = decoder_kwargs

        if self.priors_arg is None:
            # address some known cases
            if (
                decoder_kwargs.get("with_BP_OSD")
                or decoder_kwargs.get("with_BP_LSD")
                or decoder_kwargs.get("with_BF")
            ):
                self.priors_arg = "error_channel"

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DemArrays(dem)
        priors_kwargs = {self.priors_arg: list(dem_arrays.error_probs)} if self.priors_arg else {}
        decoder = decoders.get_decoder(
            dem_arrays.detector_flip_matrix, **priors_kwargs, **self.decoder_kwargs
        )
        return CompiledSinterDecoder(dem_arrays, decoder)


class DemArrays:
    """Representation of a stim.DetectorErrorModel by a collection of arrays."""

    detector_flip_matrix: scipy.sparse.csc_matrix  # maps errors to detector flips
    observable_flip_matrix: scipy.sparse.csc_matrix  # maps errors to observable flips
    error_probs: npt.NDArray[np.float64]  # probability of occurrence for each error

    def __init__(self, dem: stim.DetectorErrorModel) -> None:
        """Initialize from a stim.DetectorErrorModel."""
        errors = DemArrays._collect_and_organize_circuit_errors(dem)

        # initialize empty arrays
        detector_flip_matrix = scipy.sparse.dok_matrix(
            (dem.num_detectors, len(errors)), dtype=np.uint8
        )
        observable_flip_matrix = scipy.sparse.dok_matrix(
            (dem.num_observables, len(errors)), dtype=np.uint8
        )
        self.error_probs = np.zeros(len(errors), dtype=float)

        # iterate over all circuit errors
        for error_index, (detectors, observables_probs) in enumerate(errors.items()):
            detector_flip_matrix[[target.val for target in detectors], error_index] = 1

            """
            If a decoder decides that this error occurs and len(observables_probs) > 1, then strictly
            speaking we can only make probabilistic statements about which set of observables was
            flipped.  This is rather messy.  To make our lives easier, we give up on sorting out this
            mess and conservatively declare that all observables that may have been flipped were in
            fact flipped.
            """
            observables = frozenset.union(*observables_probs.keys())
            observable_flip_matrix[[target.val for target in observables], error_index] = 1

            probability = _probability_of_an_odd_number_of_events(observables_probs.values())
            self.error_probs[error_index] = probability

        self.detector_flip_matrix = detector_flip_matrix.tocsr()
        self.observable_flip_matrix = observable_flip_matrix.tocsr()

    @staticmethod
    def _collect_and_organize_circuit_errors(
        dem: stim.DetectorErrorModel,
    ) -> dict[frozenset[stim.DemTarget], dict[frozenset[stim.DemTarget], float]]:
        """Identify and organize circuit errors in a stim.DetectorErrorModel.

        Each circuit error is associated with:
        - a set of detectors that are flipped,
        - a set of observables that are flipped, and
        - a probability of occurrence.

        This method organizes circuit errors into a dictionary of dictionaries that looks like
            {frozenset_of_detectors: {frozenset_of_observables: probability}},
        where "probability" is the probability of occurrence for a circuit error that flips the
        corresponding detectors and observables.

        The motivation for organizing cirucit errors in this way is that a real experiment cannot
        distinguish error mechanisms that flip the same set of detectors.  We therefore have to
        combine such circuit errors when making inferences from detector data.  We defer the
        consideration of how these circuit errors are combined to downstream methods.
        """
        # First, collect all circuit errors in the stim.DetectorErrorModel, accounting for the
        # possibility of redundant errors that flip the same set of detectors and observables.
        errors: dict[frozenset[stim.DemTarget], dict[frozenset[stim.DemTarget], list[float]]] = (
            collections.defaultdict(lambda: collections.defaultdict(list))
        )
        for instruction in dem.flattened():
            if instruction.type == "error":
                probability = instruction.args_copy()[0]
                targets = instruction.targets_copy()
                detectors = _frozenset_of_items_that_occur_an_odd_number_of_times(
                    [target for target in targets if target.is_relative_detector_id()]
                )
                observables = _frozenset_of_items_that_occur_an_odd_number_of_times(
                    [target for target in targets if target.is_logical_observable_id()]
                )
                errors[detectors][observables].append(probability)

        # Combine circuit errors to obtain a single independent probability of occurrence for each
        # set of flipped detectors and observables.
        return {
            detectors: {
                observables: _probability_of_an_odd_number_of_events(probabilities)
                for observables, probabilities in observables_and_probabilities.items()
            }
            for detectors, observables_and_probabilities in errors.items()
        }


HashableType = TypeVar("HashableType", bound=Hashable)


def _frozenset_of_items_that_occur_an_odd_number_of_times(
    items: Collection[HashableType],
) -> frozenset[HashableType]:
    """Frozen subset of items that occur an odd number of times."""
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
            net_probability += probability_that_these_events_occur
    return net_probability
