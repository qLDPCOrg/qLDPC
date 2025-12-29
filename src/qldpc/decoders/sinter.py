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

        A SinterDecoder is used by Sinter to decode detection events from a detector error model to
        predict observable flips.

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
            predicted_errors = self.decoder.decode_batch(detection_event_data)
            return predicted_errors @ self.dem_arrays.observable_flip_matrix.T % 2
        observable_flips = []
        for syndrome in detection_event_data:
            predicted_errors = self.decoder.decode(syndrome)
            observable_flips.append(self.dem_arrays.observable_flip_matrix @ predicted_errors)
        return np.asarray(observable_flips, dtype=np.uint8) % 2

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


class SubgraphSinterDecoder(SinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors.

    A SubgraphSinterDecoder splits the Tanner graph of a detector error model into subgraphs, and
    decodes these subgraphs independently.  Each subgraph is defined by a subset of detectors, S.
    When compiling a SubgraphSinterDecoder for a specific detector error model D, this decoder
    constructs, for each subgraph S, a smaller detector error model D_S that restricts D to the
    detectors in S and the error mechanisms that flip the detectors in S.

    A SubgraphSinterDecoder may optionally assign each subgraph S a set of observables, O_S, in
    which case the subgraph detector error model D_S only considers (and predicts corrections for)
    the observables in O_S.

    As an example, a SubgraphSinterDecoder is useful for independently decoding the X and Z sectors of a CSS code.
    """

    def __init__(
        self,
        subgraph_detectors: Sequence[Collection[int]],
        subgraph_observables: Sequence[Collection[int]] | None = None,
        priors_arg: str | None = None,
        log_likelihood_priors: bool = False,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize a SinterDecoder that splits a detector error model into disjoint subgraphs.

        A SubgraphSinterDecoder is used by Sinter to decode detection events from a detector error
        model to predict observable flips.

        See help(sinter.Decoder) for additional information.

        Args:
            subgraph_detectors: A sequence containing one set of detectors per subgraph.
            subgraph_observables: A sequence containing one set of observables per subgraph; or None
                to indicate that every subgraph should decode every observable.  Default: None.
            priors_arg: The keyword argument to which to pass the probabilities of circuit error
                likelihoods.  This argument is only necessary for custom decoders.
            log_likelihood_priors: If True, instead of error probabilities p, pass log-likelihoods
                np.log((1 - p) / p) to the priors_arg.  This argument is only necessary for custom
                decoders.  Default: False (unless decoding with MWPM).
            **decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        # consistency checks
        self.num_subgraphs = len(subgraph_detectors)
        num_observable_sets = None if subgraph_observables is None else len(subgraph_observables)
        if not (num_observable_sets is None or num_observable_sets == self.num_subgraphs):
            raise ValueError(
                f"The number of detector sets ({self.num_subgraphs}) is inconsistent with the"
                f" number of observable sets ({num_observable_sets})"
            )

        self.subgraph_detectors = list(map(list, subgraph_detectors))
        self.subgraph_observables = (
            None if subgraph_observables is None else list(map(list, subgraph_observables))
        )

        SinterDecoder.__init__(
            self,
            priors_arg=priors_arg,
            log_likelihood_priors=log_likelihood_priors,
            **decoder_kwargs,
        )

    def compile_decoder_for_dem(
        self, dem: stim.DetectorErrorModel, *, simplify: bool = True
    ) -> CompiledSubgraphSinterDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(dem, simplify=simplify)
        subgraph_observables = (
            [slice(None)] * self.num_subgraphs
            if self.subgraph_observables is None
            else self.subgraph_observables
        )

        # build a decoder for each subgraph
        subgraph_decoders = []
        for detectors, observables in zip(self.subgraph_detectors, subgraph_observables):
            # identify the error mechanisms that flip these detectors
            errors = dem_arrays.detector_flip_matrix.getnnz(axis=0) != 0

            # build the detector error model for this subgraph
            subgraph_dem = DetectorErrorModelArrays.from_arrays(
                dem_arrays.detector_flip_matrix[detectors][:, errors],
                dem_arrays.observable_flip_matrix[observables][:, errors],
                dem_arrays.error_probs[errors],
            ).to_detector_error_model()

            # compile the decoder for this subgraph
            subgraph_decoder = SinterDecoder.compile_decoder_for_dem(self, subgraph_dem)
            subgraph_decoders.append(subgraph_decoder)

        return CompiledSubgraphSinterDecoder(
            self.subgraph_detectors,
            subgraph_observables,
            subgraph_decoders,
            dem.num_detectors,
            dem.num_observables,
        )


class CompiledSubgraphSinterDecoder(CompiledSinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit.

    This decoder splits a decoding problem into subgraphs that are decoded independently.

    Instances of this class are meant to be constructed by a SubgraphSinterDecoder, whose
    .compile_decoder_for_dem method returns a CompiledSubgraphSinterDecoder.
    See help(SubgraphSinterDecoder).
    """

    def __init__(
        self,
        subgraph_detectors: Sequence[Sequence[int] | slice],
        subgraph_observables: Sequence[Sequence[int] | slice],
        subgraph_decoders: Sequence[CompiledSinterDecoder],
        num_detectors: int,
        num_observables: int,
    ) -> None:
        assert len(subgraph_detectors) == len(subgraph_observables) == len(subgraph_decoders)
        self.subgraph_detectors = subgraph_detectors
        self.subgraph_observables = subgraph_observables
        self.subgraph_decoders = subgraph_decoders
        self.num_detectors = num_detectors
        self.num_observables = num_observables

    def decode_shots(self, detection_event_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns boolean data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        assert detection_event_data.shape[1] == self.num_detectors

        # initialize predicted observable flips
        observable_flips = np.zeros(
            (len(detection_event_data), self.num_observables), dtype=np.uint8
        )

        # decode segments independently
        for detectors, observables, decoder in zip(
            self.subgraph_detectors, self.subgraph_observables, self.subgraph_decoders
        ):
            syndromes = detection_event_data[:, detectors]
            observable_flips[:, observables] ^= decoder.decode_shots(syndromes)

        return observable_flips


class SequentialSinterDecoder(SinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors.

    A SequentialSinterDecoder splits a detector error model into time-ordered segments, and emulates
    applying active mid-circuit corrections after every segment in a quantum circuit.  Specifically,
    a SequentialSinterDecoder decodes segments sequentially, one by one.  After decoding the
    syndrome for segment j to infer a circuit error, this decoder emulates applying a corresponding
    correction by appropriately updating the syndrome for segment j+1.

    Formally, we denote the full parity check matrix of a detector error model by H, denote the
    segments to be decoded by S_1, S_2, ..., S_n, and denode the full syndrome to be decoded by s_1.
    The result of decoding segment S_1 is a decoded circuit error e_1.  This error is used to
    construct the syndrome for segment S_2, namely s_2 = s_1 + H @ e_1.  More generally, the
    syndrome for segment S_(j+1) is s_(j+1) = s_j + H @ e_j = s_1 + H @ sum_(k=1)^j e_k.  After
    decoding all segments, the net error sum_(j=1)^n e_j is used to predict observable flips.
    """

    def __init__(
        self,
        segment_detectors: Sequence[Collection[int]],
        priors_arg: str | None = None,
        log_likelihood_priors: bool = False,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize a SinterDecoder that splits a detector error model into disjoint subgraphs.

        A SequentialSinterDecoder is used by Sinter to decode detection events from a detector error
        model to predict observable flips.

        See help(sinter.Decoder) for additional information.

        Args:
            segment_detectors: A sequence containing one set of detectors per segment.
            priors_arg: The keyword argument to which to pass the probabilities of circuit error
                likelihoods.  This argument is only necessary for custom decoders.
            log_likelihood_priors: If True, instead of error probabilities p, pass log-likelihoods
                np.log((1 - p) / p) to the priors_arg.  This argument is only necessary for custom
                decoders.  Default: False (unless decoding with MWPM).
            **decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        self.segment_detectors = [list(detectors) for detectors in segment_detectors if detectors]
        SinterDecoder.__init__(
            self,
            priors_arg=priors_arg,
            log_likelihood_priors=log_likelihood_priors,
            **decoder_kwargs,
        )

    def compile_decoder_for_dem(
        self, dem: stim.DetectorErrorModel, *, simplify: bool = True
    ) -> CompiledSubgraphSinterDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(dem, simplify=simplify)

        # identify addressed circuit errors and compile a decoder for each segment
        segment_errors = []
        segment_decoders = []
        addressed_errors = np.zeros(dem_arrays.num_errors, dtype=bool)
        for detectors in self.segment_detectors:
            # identify errors that
            # (a) trigger the detectors for this segment, and
            # (b) have not been addressed by preceding segments
            errors = dem_arrays.detector_flip_matrix[detectors].getnnz(axis=0) != 0
            errors[addressed_errors] = False

            # build the detector error model for this segment, and compile a detector for it
            segment_dem_arrays = DetectorErrorModelArrays.from_arrays(
                dem_arrays.detector_flip_matrix[detectors][:, errors],
                dem_arrays.observable_flip_matrix[:, errors],
                dem_arrays.error_probs[errors],
            )
            segment_decoder = self.get_configured_decoder(segment_dem_arrays)

            # update the history of errors that were dealt with by preceding segments
            addressed_errors |= errors

            # save the errors that this segment addresses, and the decoder for the segment
            segment_errors.append(errors)
            segment_decoders.append(segment_decoder)

        return CompiledSequentialSinterDecoder(
            dem_arrays,
            self.segment_detectors,
            segment_errors,
            segment_decoders,
        )


class CompiledSequentialSinterDecoder(CompiledSinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit.

    This decoder splits a decoding problem into segments that are decoded sequentially.

    Instances of this class are meant to be constructed by a SequentialSinterDecoder, whose
    .compile_decoder_for_dem method returns a CompiledSequentialSinterDecoder.
    See help(SequentialSinterDecoder).
    """

    def __init__(
        self,
        dem_arrays: DetectorErrorModelArrays,
        segment_detectors: Sequence[Sequence[int] | slice],
        segment_errors: Sequence[Sequence[int] | slice],
        segment_decoders: Sequence[Decoder],
    ) -> None:
        assert len(segment_detectors) == len(segment_errors) == len(segment_decoders)
        self.dem_arrays = dem_arrays
        self.segment_detectors = segment_detectors
        self.segment_errors = segment_errors
        self.segment_decoders = segment_decoders

        self.num_detectors = dem_arrays.num_detectors

    def decode_shots(self, detection_event_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns boolean data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        return (
            self.decode_shots_to_error(detection_event_data)
            @ self.dem_arrays.observable_flip_matrix.T
            % 2
        )

    def decode_shots_to_error(
        self, detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predicts a net circuit error from the given detection events.

        This method accepts and returns boolean data.
        """
        num_samples, num_detectors = detection_event_data.shape
        assert num_detectors == self.dem_arrays.num_detectors

        # identify the net circuit error predicted by decoding one segment at a time
        net_error = np.zeros((num_samples, self.dem_arrays.num_errors), dtype=int)
        detector_flip_matrix_T = self.dem_arrays.detector_flip_matrix.T
        for detectors, errors, decoder in zip(
            self.segment_detectors, self.segment_errors, self.segment_decoders
        ):
            # the bare syndrome plus any corrections we have inferred so far
            syndromes = (
                detection_event_data[:, detectors]
                + net_error @ detector_flip_matrix_T[:, detectors]
            ) % 2

            # decode this syndrome and update the net error appropriately
            net_error[:, errors] = (
                decoder.decode_batch(syndromes)
                if hasattr(decoder, "decode_batch")
                else np.array([decoder.decode(syndrome) for syndrome in syndromes])
            )

        return net_error
