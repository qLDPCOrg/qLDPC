# SPDX-License-Identifier: Apache-2.0

"""Sinter decoders that partition a detector error model into subgraphs."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Sequence

import numpy as np
import numpy.typing as npt
import stim

from ..construction.specs import DeferredDecoderInput
from ..dems import DetectorErrorModelArrays
from .core import CompiledSinterDecoder, SinterDecoder


class SubgraphDecoder(SinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors.

    A SubgraphDecoder splits the Tanner graph of a detector error model into subgraphs, and decodes
    these subgraphs independently.  Each subgraph is defined by a subset of detectors, S.  When
    compiling a SubgraphDecoder for a specific detector error model D, this decoder constructs, for
    each subgraph S, a smaller detector error model ``D_S`` that restricts D to the detectors in S
    and the error mechanisms that flip the detectors in S.

    A SubgraphDecoder may optionally assign each subgraph S a set of observables, ``O_S``, in which
    case the subgraph detector error model ``D_S`` only considers (and predicts corrections for) the
    observables in ``O_S``.

    The subgraphs predict observable flips independently, and their predictions are combined by
    exclusive or.  Every observable therefore has to be assigned to the subgraphs in a way that lets
    exactly one of them predict each of its flips: if two subgraphs both witness an error mechanism
    and both own an observable that the mechanism flips, then both predict that flip and the two
    predictions cancel.  Compiling a SubgraphDecoder warns when a detector error model and a
    partition permit that, and when a detector belongs to no subgraph at all.

    As an example, a SubgraphDecoder is useful for independently decoding the X and Z sectors of a
    CSS code, where each sector owns the observables of the opposite type.
    """

    _prebuilt_decoder_rejection_reason = "a SubgraphDecoder builds a new decoder for each subgraph"

    def __init__(
        self,
        subgraph_detectors: Sequence[Collection[int]],
        subgraph_observables: Sequence[Collection[int]] | None = None,
        *,
        simplify: bool = True,
        decompose_errors: bool = False,
        decoder: DeferredDecoderInput = None,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize an observable decoder that splits a model into disjoint subgraphs.

        A SubgraphDecoder is used by Sinter to decode detection events from a detector error model
        to predict observable flips.

        See help(sinter.Decoder) for additional information.

        Args:
            subgraph_detectors: A sequence containing one set of detectors per subgraph.
            subgraph_observables: A sequence containing one set of observables per subgraph; or None
                to indicate that every subgraph should decode every observable.  Default: None.
            simplify: Whether to merge equivalent errors in a DEM when compiling a decoder for
                that DEM.
            decompose_errors: Whether to decompose errors according to their suggested decomposition
                when compiling a decoder for a DEM.
            decoder: A specification for the inner decoder, such as ``decoders.mwpm(...)``, or a
                constructor that builds an error or observable decoder from a detector error model,
                an observable-decoder compiler such as a SinterDecoder, or None to select the
                default decoder.  A prebuilt decoder is rejected, because an inner decoder is built
                for each subgraph.
            **decoder_kwargs: Deprecated arguments to pass to qldpc.decoders.get_decoder.
        """
        SinterDecoder.__init__(
            self,
            simplify=simplify,
            decompose_errors=decompose_errors,
            decoder=decoder,
            **decoder_kwargs,
        )

        # consistency checks
        self.num_subgraphs = len(subgraph_detectors)
        num_observable_sets = None if subgraph_observables is None else len(subgraph_observables)
        if not (num_observable_sets is None or num_observable_sets == self.num_subgraphs):
            raise ValueError(
                f"The number of detector sets ({self.num_subgraphs}) is inconsistent with the"
                f" number of observable sets ({num_observable_sets})"
            )

        self.subgraph_detectors = [sorted(dets) for dets in subgraph_detectors]
        self.subgraph_observables = (
            None if subgraph_observables is None else [sorted(obs) for obs in subgraph_observables]
        )

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> CompiledSubgraphDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(
            dem, simplify=self.simplify, decompose_errors=self.decompose_errors
        )
        subgraph_observables = (
            [list(range(dem.num_observables)) for _ in range(self.num_subgraphs)]
            if self.subgraph_observables is None
            else [list(obs) for obs in self.subgraph_observables]
        )
        num_erasure_bits = 0

        # count, for every observable flip, the subgraphs that can predict it
        flip_observables, flip_errors = dem_arrays.observable_flip_matrix.nonzero()
        flip_predictors = np.zeros(len(flip_errors), dtype=int)
        covered_detectors = np.zeros(dem.num_detectors, dtype=bool)

        # build a decoder for each subgraph
        subgraph_decoders = []
        for ss, (detectors, observables) in enumerate(
            zip(self.subgraph_detectors, subgraph_observables)
        ):
            # identify the error mechanisms that flip these detectors
            errors = dem_arrays.detector_flip_matrix[detectors].getnnz(axis=0) != 0

            # this subgraph can predict a flip if it owns the observable and witnesses the error
            owned = np.zeros(dem.num_observables, dtype=bool)
            owned[observables] = True
            flip_predictors += owned[flip_observables] & errors[flip_errors]
            covered_detectors[detectors] = True

            # build the detector error model for this subgraph
            subgraph_dem = DetectorErrorModelArrays.from_arrays(
                dem_arrays.detector_flip_matrix[detectors][:, errors],
                dem_arrays.observable_flip_matrix[observables][:, errors],
                dem_arrays.error_probs[errors],
            ).to_detector_error_model()

            # compile the decoder for this subgraph
            subgraph_decoder = SinterDecoder.compile_decoder_for_dem(self, subgraph_dem)
            subgraph_decoders.append(subgraph_decoder)

            # collect the erasure bit of this subgraph past the observables of the whole model
            if getattr(subgraph_decoder.decoder, "has_erasure_bit", False):
                subgraph_observables[ss].append(dem.num_observables + num_erasure_bits)
                num_erasure_bits += 1

        _warn_about_subgraph_partition(
            flip_errors, flip_observables, flip_predictors, np.flatnonzero(~covered_detectors)
        )

        return CompiledSubgraphDecoder(
            self.subgraph_detectors,
            subgraph_observables,
            subgraph_decoders,
            dem.num_detectors,
            dem.num_observables,
            num_erasure_bits,
        )


class CompiledSubgraphDecoder(CompiledSinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit.

    This decoder splits a decoding problem into subgraphs that are decoded independently.

    Instances of this class are meant to be constructed by a SubgraphDecoder, whose
    .compile_decoder_for_dem method returns a CompiledSubgraphDecoder.
    See help(SubgraphDecoder).
    """

    def __init__(
        self,
        subgraph_detectors: Sequence[Sequence[int] | slice],
        subgraph_observables: Sequence[Sequence[int] | slice],
        subgraph_decoders: Sequence[CompiledSinterDecoder],
        num_detectors: int,
        num_observables: int,
        num_erasure_bits: int = 0,
    ) -> None:
        if not len(subgraph_detectors) == len(subgraph_observables) == len(subgraph_decoders):
            raise ValueError(
                "A CompiledSubgraphDecoder needs one detector set, one observable set, and one"
                f" decoder per subgraph (provided: {len(subgraph_detectors)},"
                f" {len(subgraph_observables)}, {len(subgraph_decoders)})"
            )
        self.subgraph_detectors = subgraph_detectors
        self.subgraph_observables = subgraph_observables
        self.subgraph_decoders = subgraph_decoders
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.num_erasure_bits = num_erasure_bits

    def decode_shots(self, detection_event_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns boolean data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        if detection_event_data.shape[1] != self.num_detectors:
            raise ValueError(
                f"Detection event data has {detection_event_data.shape[1]} detectors per shot, but"
                f" this decoder was compiled for {self.num_detectors} detectors"
            )

        # initialize predicted observable flips, followed by one erasure bit per erasing subgraph
        observable_flips = np.zeros(
            (len(detection_event_data), self.num_observables + self.num_erasure_bits),
            dtype=np.uint8,
        )

        # decode segments independently
        for detectors, observables, decoder in zip(
            self.subgraph_detectors, self.subgraph_observables, self.subgraph_decoders
        ):
            syndromes = detection_event_data[:, detectors]
            observable_flips[:, observables] ^= decoder.decode_shots(syndromes)

        return observable_flips


def _warn_about_subgraph_partition(
    flip_errors: npt.NDArray[np.int_],
    flip_observables: npt.NDArray[np.int_],
    flip_predictors: npt.NDArray[np.int_],
    uncovered_detectors: npt.NDArray[np.int_],
) -> None:
    """Warn about a partition into subgraphs whose predictions do not add up.

    Args:
        flip_errors: The error mechanism of each observable flip in a detector error model.
        flip_observables: The observable of each of those flips.
        flip_predictors: The number of subgraphs that can predict each of those flips.
        uncovered_detectors: The detectors that belong to no subgraph.
    """
    contested = np.flatnonzero(flip_predictors > 1)
    if contested.size:
        first = contested[0]
        warnings.warn(
            f"{contested.size} observable flips of this detector error model can be predicted by"
            " more than one subgraph, and predictions are combined by exclusive or, so two"
            " subgraphs predicting the same flip cancel each other.  Assign each observable only to"
            " subgraphs whose detectors witness its flips.  For example, error mechanism"
            f" {flip_errors[first]} flips observable {flip_observables[first]}, which"
            f" {flip_predictors[first]} subgraphs can predict",
            stacklevel=3,
        )
    if uncovered_detectors.size:
        warnings.warn(
            f"{uncovered_detectors.size} detectors of this detector error model belong to no"
            " subgraph, so no decoder ever sees their detection events:"
            f" {uncovered_detectors[:10].tolist()}",
            stacklevel=3,
        )
