from __future__ import annotations

import collections
import itertools
from collections.abc import Collection, Hashable
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import scipy.sparse
import sinter
import stim

from qldpc import decoders


class SinterDecoder(sinter.Decoder):
    """Decoder usable by Sinter for decoding circuit errors."""

    def __init__(self, *, error_probs_arg: str | None = None, **decoder_kwargs: object) -> None:
        """Initialize a SinterDecoder.

        Args:
            error_probs_arg: The keyword argument to which to pass the probabilities of occurrence
                for each circuit error (determined by a detector error model) to a decoder in
                qldpc.decoders.get_decoder.  This argument is only necessary for custom decoders.
            decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        self.error_probs_arg = error_probs_arg
        self.decoder_kwargs = decoder_kwargs

        if self.error_probs_arg is None:
            # address some known cases
            if (
                decoder_kwargs.get("with_BP_OSD")
                or decoder_kwargs.get("with_BP_LSD")
                or decoder_kwargs.get("with_BF")
            ):
                self.error_probs_arg = "error_channel"
            if decoder_kwargs.get("with_MWPM"):
                self.error_probs_arg = "weights"

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(dem)
        probs_kwarg = (
            {self.error_probs_arg: list(dem_arrays.error_probs)} if self.error_probs_arg else {}
        )
        decoder = decoders.get_decoder(
            dem_arrays.detector_flip_matrix, **self.decoder_kwargs, **probs_kwarg
        )
        return CompiledSinterDecoder(dem_arrays, decoder)


class CompiledSinterDecoder(sinter.CompiledDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit."""

    def __init__(self, dem_arrays: DetectorErrorModelArrays, decoder: decoders.Decoder) -> None:
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
            predicted_errors = self.decoder.decode(syndrome.astype(int))
            observable_flips.append(self.dem_arrays.observable_flip_matrix @ predicted_errors % 2)
        return np.packbits(np.array(observable_flips, dtype=np.uint8), bitorder="little", axis=1)


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
        for error_index, ((detectors, observables), probability) in enumerate(errors.items()):
            detector_flip_matrix[[target.val for target in detectors], error_index] = 1
            observable_flip_matrix[[target.val for target in observables], error_index] = 1
            self.error_probs[error_index] = probability

        self.detector_flip_matrix = detector_flip_matrix.tocsc()
        self.observable_flip_matrix = observable_flip_matrix.tocsc()

    @staticmethod
    def get_merged_circuit_errors(
        dem: stim.DetectorErrorModel,
    ) -> dict[tuple[frozenset[stim.DemTarget], frozenset[stim.DemTarget]], float]:
        """Organize and merge circuit errors in a stim.DetectorErrorModel.

        Each circuit error is identified by:
        - a set of detectors that are flipped,
        - a set of observables that are flipped, and
        - a probability of occurrence.

        This method organizes circuit errors into a dictionary of dictionaries that looks like
            {(frozenset_of_detectors, frozenset_of_observables): probability}},
        where "probability" is the probability of occurrence for a circuit error that flips the
        corresponding detectors and observables.  Circuit errors

        Circuit errors that flip the same set of detectors and observables are merged.
        """
        # Collect all circuit errors in the stim.DetectorErrorModel, accounting for the possibility
        # of indistinguishable errors that flip the same sets of detectors and observables.
        errors: dict[tuple[frozenset[stim.DemTarget], frozenset[stim.DemTarget]], list[float]] = (
            collections.defaultdict(list)
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
                errors[detectors, observables].append(probability)

        # Combine circuit errors to obtain a single probability of occurrence for each set of flipped
        # detectors and observables.
        return {
            detectors_observables: _probability_of_an_odd_number_of_events(probabilities)
            for detectors_observables, probabilities in errors.items()
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
            net_probability += float(probability_that_these_events_occur)
    return net_probability
