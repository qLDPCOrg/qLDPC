"""Decoders for sinter to sample quantum error correction circuits

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

import functools
import operator
from collections.abc import Collection, Sequence

import numpy as np
import numpy.typing as npt
import sinter
import stim

from .decoders import Decoder, get_decoder
from .dem_arrays import DetectorErrorModelArrays


class SinterDecoder(sinter.Decoder):
    """Decoder usable by Sinter for decoding circuit errors."""

    def __init__(
        self,
        *,
        priors_arg: str | None = None,
        log_likelihood_priors: bool = False,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize a SinterDecoder.

        A SinterDecoder is used by Sinter to decode detection events from circuit (or, more
        generally, detector error model) simulations to predict observable flips.

        See help(sinter.Decoder) for additional information.

        Args:
            priors_arg: The keyword argument to which to pass the probabilities of circuit error
                likelihoods.  This argument is only necessary for custom decoders.
            log_likelihood_priors: If True, instead of error probabilities p, pass log-likelihoods
                np.log((1 - p) / p) to the priors_arg.  This argument is only necessary for custom
                decoders.  Default: False (unless decoding with MWPM).
            **decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        self.priors_arg = priors_arg
        self.log_likelihood_priors = log_likelihood_priors
        self.decoder_kwargs = decoder_kwargs

        if self.priors_arg is None:
            # address some known cases
            if (
                decoder_kwargs.get("with_BP_OSD")
                or decoder_kwargs.get("with_BP_LSD")
                or decoder_kwargs.get("with_BF")
            ):
                self.priors_arg = "error_channel"
            if decoder_kwargs.get("with_RBP"):
                self.priors_arg = "error_priors"
            if decoder_kwargs.get("with_MWPM"):
                self.priors_arg = "weights"
                self.log_likelihood_priors = True

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> CompiledSinterDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(dem)
        decoder = self.get_configured_decoder(dem_arrays)
        return CompiledSinterDecoder(dem_arrays, decoder)

    def get_configured_decoder(self, dem_arrays: DetectorErrorModelArrays) -> Decoder:
        """Configure a Decoder from the given DetectorErrorModelArrays."""
        priors = dem_arrays.error_probs
        if self.log_likelihood_priors:
            priors = np.log((1 - priors) / priors)
        priors_kwarg = {self.priors_arg: list(priors)} if self.priors_arg else {}
        decoder = get_decoder(
            dem_arrays.detector_flip_matrix, **self.decoder_kwargs, **priors_kwarg
        )
        return decoder


class CompiledSinterDecoder(sinter.CompiledDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit.

    Instances of this class are meant to be constructed by a SinterDecoder, whose
    .compile_decoder_for_dem method returns a CompiledSinterDecoder.
    """

    def __init__(self, dem_arrays: DetectorErrorModelArrays, decoder: Decoder) -> None:
        self.dem_arrays = dem_arrays
        self.decoder = decoder
        self.num_detectors = self.dem_arrays.num_detectors

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns bit-packed data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        detection_event_data = self.unpack_detection_event_data(bit_packed_detection_event_data)
        observable_flips = self.decode_shots(detection_event_data)
        return self.packbits(observable_flips)

    def decode_shots(self, detection_event_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns boolean data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        if hasattr(self.decoder, "decode_batch"):
            predicted_errors_T = self.decoder.decode_batch(detection_event_data)
            observable_flips = predicted_errors_T @ self.dem_arrays.observable_flip_matrix.T % 2
        else:
            observable_flips = []
            for syndrome in detection_event_data:
                predicted_errors = self.decoder.decode(syndrome)
                observable_flips.append(
                    self.dem_arrays.observable_flip_matrix @ predicted_errors % 2
                )
        return np.asarray(observable_flips, dtype=np.uint8)

    def packbits(self, data: npt.NDArray[np.uint8], axis: int = 1) -> npt.NDArray[np.uint8]:
        """Bit-pack the data along an axis.

        Working with bit-packed data is more memory and compute-efficient, which is why Sinter
        generally passes around bit-packed data.
        """
        return np.packbits(np.asarray(data, dtype=np.uint8), bitorder="little", axis=axis)

    def unpack_detection_event_data(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8], axis: int = 1
    ) -> npt.NDArray[np.uint8]:
        """Unpack the bit-packed data along an axis.

        By default, bit_packed_detection_event_data is assumed to be a two-dimensional array in
        which each row contains bit-packed detection events from one sample of a detector error
        model (DEM).  In this case, the unpacked data is a boolean matrix whose entry in row ss and
        column kk specify whether detector kk was flipped in sample ss of a DEM.
        """
        return np.unpackbits(
            np.asarray(bit_packed_detection_event_data, dtype=np.uint8),
            count=self.num_detectors,
            bitorder="little",
            axis=axis,
        )


class CompositeSinterDecoder(SinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors.

    This decoder splits a detector error model into independent decoding problems, or segments,
    defined by subsets of detectors and observables in a detector error model.  This is useful for
    independently decoding the X and Z sectors of a CSS code.
    """

    def __init__(
        self,
        *detectors_and_observables: tuple[Collection[int], Collection[int]],
        priors_arg: str | None = None,
        log_likelihood_priors: bool = False,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize a SinterDecoder to independently decode subsets of detectors and observables.

        A CompositeSinterDecoder is used by Sinter to decode detection events from circuit (or, more
        generally, detector error model) simulations to predict observable flips.

        See help(sinter.Decoder) for additional information.

        Args:
            *detectors_and_observables: Tuples of detector indices and associated observable indices
                that define the segments to decode independently.
            priors_arg: The keyword argument to which to pass the probabilities of circuit error
                likelihoods.  This argument is only necessary for custom decoders.
            log_likelihood_priors: If True, instead of error probabilities p, pass log-likelihoods
                np.log((1 - p) / p) to the priors_arg.  This argument is only necessary for custom
                decoders.  Default: False (unless decoding with MWPM).
            **decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        self.segment_detectors, self.segment_observables = zip(
            *[
                (list(detectors), list(observables))
                for detectors, observables in detectors_and_observables
            ]
        )
        SinterDecoder.__init__(
            self,
            priors_arg=priors_arg,
            log_likelihood_priors=log_likelihood_priors,
            **decoder_kwargs,
        )

    def compile_decoder_for_dem(
        self, dem: stim.DetectorErrorModel
    ) -> CompiledCompositeSinterDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(dem)
        segment_dems = [
            DetectorErrorModelArrays.from_arrays(
                dem_arrays.detector_flip_matrix[detectors, :],
                dem_arrays.observable_flip_matrix[observables, :],
                dem_arrays.error_probs,
            )
            .simplify()
            .to_detector_error_model()
            for detectors, observables in zip(self.segment_detectors, self.segment_observables)
        ]
        compiled_decoders = [
            SinterDecoder.compile_decoder_for_dem(self, segment_dem) for segment_dem in segment_dems
        ]
        return CompiledCompositeSinterDecoder(
            self.segment_detectors, self.segment_observables, compiled_decoders
        )


class CompiledCompositeSinterDecoder(CompiledSinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit.

    This decoder splits a decoding problem into segments and solves each segment independently.
    Here a segment is defined by its own subset of detectors, subset of observables, and decoder.

    Instances of this class are meant to be constructed by a CompositeSinterDecoder, whose
    .compile_decoder_for_dem method returns a CompiledCompositeSinterDecoder.
    """

    def __init__(
        self,
        segment_detectors: Sequence[list[int]],
        segment_observables: Sequence[list[int]],
        segment_decoders: Sequence[CompiledSinterDecoder],
    ) -> None:
        assert len(segment_detectors) == len(segment_observables) == len(segment_decoders)
        self.segment_detectors = segment_detectors
        self.segment_observables = segment_observables
        self.segment_decoders = segment_decoders

        self.num_detectors = sum(
            decoder.dem_arrays.num_detectors for decoder in self.segment_decoders
        )
        self.permutation_to_sort_observables = np.argsort(
            functools.reduce(operator.add, self.segment_observables)
        )

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns bit-packed data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        detection_event_data = self.unpack_detection_event_data(bit_packed_detection_event_data)
        observable_flips = self.decode_shots(detection_event_data)
        return self.packbits(observable_flips)

    def decode_shots(self, detection_event_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns boolean data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        # split detection event data into syndromes in each segment
        syndromes = [
            detection_event_data.T[detectors].T
            for detectors, decoder in zip(self.segment_detectors, self.segment_decoders)
        ]

        # decode segments independently
        observable_flips = [
            decoder.decode_shots(segment_syndromes)
            for segment_syndromes, decoder in zip(syndromes, self.segment_decoders)
        ]

        # stack observable flips and permute observables appropriately
        return np.hstack(observable_flips).T[self.permutation_to_sort_observables].T
