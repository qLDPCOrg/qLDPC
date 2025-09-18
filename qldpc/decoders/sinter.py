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

from collections.abc import Collection, Sequence

import numpy as np
import numpy.typing as npt
import sinter
import stim

from .dems import DetectorErrorModelArrays
from .retrieval import Decoder, get_decoder


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
                decoder_kwargs.get("with_lookup")
                or decoder_kwargs.get("with_BP_OSD")
                or decoder_kwargs.get("with_BP_LSD")
                or decoder_kwargs.get("with_BF")
            ):
                self.priors_arg = "error_channel"
            if decoder_kwargs.get("with_RBP"):
                self.priors_arg = "error_priors"
            if decoder_kwargs.get("with_MWPM"):
                self.priors_arg = "weights"
                self.log_likelihood_priors = True

    def compile_decoder_for_dem(
        self, dem: stim.DetectorErrorModel, *, simplify: bool = True
    ) -> CompiledSinterDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(dem, simplify=simplify)
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

    def packbits(self, data: npt.NDArray[np.uint8], axis: int = -1) -> npt.NDArray[np.uint8]:
        """Bit-pack the data along an axis.

        Working with bit-packed data is more memory and compute-efficient, which is why Sinter
        generally passes around bit-packed data.
        """
        return np.packbits(np.asarray(data, dtype=np.uint8), bitorder="little", axis=axis)

    def unpack_detection_event_data(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8], axis: int = -1
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

    This decoder splits a detector error model into independent decoding problems, or segments.
    Each segment S is defined by a subset of detectors d_S.  When compiling a CompositeSinterDecoder
    for a specific detector error model D, this decoder constructs, for each segment S, a smaller
    detector error model D_S that restricts D to the error mechanisms that flip detectors in d_S,
    and ignores detectors not in d_S.

    A segment S may optionally be assigned a set of observables, O_S, in which case the segment
    detector error model D_S only considers the observables in O_S.

    As an example, the capability to split detector error model into segments is useful for
    independently decoding the X and Z sectors of a CSS code.

    Finally, a segment S may also be assigned an "exclusion set" of detectors, e_S, in which case
    the segment detector error model D_S excludes error mechanisms that trigger detectors in e_S.
    This capability can be useful when post-selecting on the detectors in e_S.
    """

    def __init__(
        self,
        segment_detectors: Sequence[Collection[int]],
        segment_observables: Sequence[Collection[int]] | None = None,
        segment_exclusions: Sequence[Collection[int]] | None = None,
        priors_arg: str | None = None,
        log_likelihood_priors: bool = False,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize a SinterDecoder to independently decode subsets of detectors and observables.

        A CompositeSinterDecoder is used by Sinter to decode detection events from circuit (or, more
        generally, detector error model) simulations to predict observable flips.

        See help(sinter.Decoder) for additional information.

        Args:
            segment_detectors: A sequence containing one set of detectors per segment.
            segment_observables: A sequence containing one set of observables per segment; or None
                to indicate that every segment should decode every observable.  Default: None.
            segment_exclusions: A sequence containing one detector exclusion set per segment; or
                None to indicate no exclusions for all segments.  Default: None.
            priors_arg: The keyword argument to which to pass the probabilities of circuit error
                likelihoods.  This argument is only necessary for custom decoders.
            log_likelihood_priors: If True, instead of error probabilities p, pass log-likelihoods
                np.log((1 - p) / p) to the priors_arg.  This argument is only necessary for custom
                decoders.  Default: False (unless decoding with MWPM).
            **decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        # consistency check
        self.num_segments = len(segment_detectors)
        num_observables = None if segment_observables is None else len(segment_observables)
        num_exclusions = None if segment_exclusions is None else len(segment_exclusions)
        if not (
            (num_observables is None or num_observables == self.num_segments)
            and (num_exclusions is None or num_exclusions == self.num_segments)
        ):
            raise ValueError(
                f"The number of detector sets ({self.num_segments}) is inconsistent with the number"
                f" of observable sets ({num_observables}) or exclusion sets ({num_exclusions})"
            )

        self.segment_detectors = list(map(list, segment_detectors))
        self.segment_observables = (
            None if segment_observables is None else list(map(list, segment_observables))
        )
        self.segment_exclusions = (
            None if segment_exclusions is None else list(map(list, segment_exclusions))
        )

        SinterDecoder.__init__(
            self,
            priors_arg=priors_arg,
            log_likelihood_priors=log_likelihood_priors,
            **decoder_kwargs,
        )

    def compile_decoder_for_dem(
        self, dem: stim.DetectorErrorModel, *, simplify: bool = True
    ) -> CompiledCompositeSinterDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(dem, simplify=simplify)
        segment_observables = (
            [range(dem.num_observables)] * self.num_segments
            if self.segment_observables is None
            else self.segment_observables
        )

        # build a restricted detector error model for each segment
        segment_dems = []
        for ss in range(self.num_segments):
            detectors = self.segment_detectors[ss]
            observables = segment_observables[ss]

            detector_flip_matrix = dem_arrays.detector_flip_matrix[detectors, :]
            observable_flip_matrix = dem_arrays.observable_flip_matrix[observables, :]
            error_probs = dem_arrays.error_probs

            # restrict to error mechanisms that flip the specified detectors
            mask = detector_flip_matrix.getnnz(axis=0) > 0

            # if applicable, restrict to error mechanisms that DO NOT trigger exclusions
            if self.segment_exclusions is not None:
                exclusions = self.segment_exclusions[ss]
                mask &= dem_arrays.detector_flip_matrix[exclusions, :].getnnz(axis=0) == 0

            detector_flip_matrix = detector_flip_matrix[:, mask]
            observable_flip_matrix = observable_flip_matrix[:, mask]
            error_probs = error_probs[mask]

            segment_dem = DetectorErrorModelArrays.from_arrays(
                detector_flip_matrix,
                observable_flip_matrix,
                error_probs,
                simplify=simplify,  # TODO: is simplifying here redundant with simplifying above?
            ).to_detector_error_model()
            segment_dems.append(segment_dem)

        compiled_decoders = [
            SinterDecoder.compile_decoder_for_dem(self, segment_dem) for segment_dem in segment_dems
        ]
        return CompiledCompositeSinterDecoder(
            self.segment_detectors,
            segment_observables,
            compiled_decoders,
            dem.num_detectors,
            dem.num_observables,
        )


class CompiledCompositeSinterDecoder(CompiledSinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit.

    This decoder splits a decoding problem into segments and solves each segment independently.
    Here a segment is defined by a set of detectors, a set of observables, and decoder.

    Instances of this class are meant to be constructed by a CompositeSinterDecoder, whose
    .compile_decoder_for_dem method returns a CompiledCompositeSinterDecoder.
    See help(CompositeSinterDecoder).
    """

    def __init__(
        self,
        segment_detectors: Sequence[Sequence[int]],
        segment_observables: Sequence[Sequence[int]],
        segment_decoders: Sequence[CompiledSinterDecoder],
        num_detectors: int,
        num_observables: int,
    ) -> None:
        assert len(segment_detectors) == len(segment_observables) == len(segment_decoders)
        self.segment_detectors = segment_detectors
        self.segment_observables = segment_observables
        self.segment_decoders = segment_decoders
        self.num_detectors = num_detectors
        self.num_observables = num_observables

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
        # initialize predicted observable flips
        observable_flips = np.zeros(
            (len(detection_event_data), self.num_observables), dtype=np.uint8
        )

        # decode segments independently
        for detectors, observables, decoder in zip(
            self.segment_detectors, self.segment_observables, self.segment_decoders
        ):
            syndromes = detection_event_data.T[detectors].T
            observable_flips[:, observables] ^= decoder.decode_shots(syndromes)

        return observable_flips
