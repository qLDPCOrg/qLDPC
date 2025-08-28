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

from collections.abc import Collection

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

        See help(sinter.Decoder) for additional information.

        Args:
            priors_arg: The keyword argument to which to pass the probabilities of circuit error
                likelihoods.  This argument is only necessary for custom decoders.
            log_likelihood_priors: If True, instead of error probabilities p, pass log-likelihoods
                np.log((1 - p) / p) to the priors_arg.  This argument is only necessary for custom
                decoders.  Default: False.
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

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(dem)
        decoder = self.get_configured_decoder(dem_arrays)
        return CompiledSinterDecoder(dem_arrays, decoder)


class CompiledSinterDecoder(sinter.CompiledDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit."""

    def __init__(self, dem_arrays: DetectorErrorModelArrays, decoder: Decoder) -> None:
        self.dem_arrays = dem_arrays
        self.decoder = decoder
        self.num_detectors = self.dem_arrays.num_detectors

    def packbits(self, data: npt.NDArray[np.uint8], axis: int = 1) -> npt.NDArray[np.uint8]:
        """Pack the data along an axis."""
        return np.packbits(np.asarray(data, dtype=np.uint8), bitorder="little", axis=axis)

    def unpackbits(self, data: npt.NDArray[np.uint8], axis: int = 1) -> npt.NDArray[np.uint8]:
        """Unpack the data along an axis."""
        return np.unpackbits(
            np.asarray(data, dtype=np.uint8), count=self.num_detectors, bitorder="little", axis=axis
        )

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        See help(sinter.CompiledDecoder) for additional information.
        """
        syndromes = self.unpackbits(bit_packed_detection_event_data)
        if hasattr(self.decoder, "decode_batch"):
            predicted_errors_T = self.decoder.decode_batch(syndromes)
            observable_flips = predicted_errors_T @ self.dem_arrays.observable_flip_matrix.T % 2
        else:
            observable_flips = []
            for syndrome in syndromes:
                predicted_errors = self.decoder.decode(syndrome)
                observable_flips.append(
                    self.dem_arrays.observable_flip_matrix @ predicted_errors % 2
                )
        return self.packbits(observable_flips)


class SinterDecoderXZ(SinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors that split into X and Z sectors.

    This decoder splits the detector error model of a circuit into two subsets of detectors whose
    syndromes are decoded independently to infer the errors of the first and second half of the
    annotated observables in a circuit, which are suggestively referred to as X and Z observables.
    """

    def __init__(
        self,
        detectors_x: Collection[int],
        detectors_z: Collection[int],
        *,
        priors_arg: str | None = None,
        log_likelihood_priors: bool = False,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize a SinterDecoder to independently decode two sectors of a detector error model.

        See help(sinter.Decoder) for additional information.

        Args:
            detectors_x: Detectors used to correct the X observables.
            detectors_z: Detectors used to correct the Z observables.
            priors_arg: The keyword argument to which to pass the probabilities of circuit error
                likelihoods.  This argument is only necessary for custom decoders.
            log_likelihood_priors: If True, instead of error probabilities p, pass log-likelihoods
                np.log((1 - p) / p) to the priors_arg.  This argument is only necessary for custom
                decoders.  Default: False.
            **decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        self.detectors_x = sorted(detectors_x)
        self.detectors_z = sorted(detectors_z)
        SinterDecoder.__init__(
            self,
            priors_arg=priors_arg,
            log_likelihood_priors=log_likelihood_priors,
            **decoder_kwargs,
        )

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(dem)
        num_observables = dem_arrays.observable_flip_matrix.shape[0]
        if num_observables % 2:
            raise ValueError(
                "SinterDecoderXZ only handles detector error models with an even number of"
                f" observables (provided: {num_observables})"
            )

        detector_flip_matrix = dem_arrays.detector_flip_matrix
        observable_flip_matrix = dem_arrays.observable_flip_matrix
        error_probs = dem_arrays.error_probs

        # build separate detector error models for the X and Z sectors
        detector_flip_matrix_x = detector_flip_matrix[self.detectors_x, :]
        detector_flip_matrix_z = detector_flip_matrix[self.detectors_z, :]
        observable_flip_matrix_x = observable_flip_matrix[: num_observables // 2, :]
        observable_flip_matrix_z = observable_flip_matrix[num_observables // 2 :, :]
        dem_x = DetectorErrorModelArrays.from_arrays(
            detector_flip_matrix_x, observable_flip_matrix_x, error_probs
        ).to_detector_error_model()
        dem_z = DetectorErrorModelArrays.from_arrays(
            detector_flip_matrix_z, observable_flip_matrix_z, error_probs
        ).to_detector_error_model()

        # construct and combine compiled sinter decoders
        compiled_decoder_x = SinterDecoder.compile_decoder_for_dem(self, dem_x)
        compiled_decoder_z = SinterDecoder.compile_decoder_for_dem(self, dem_z)
        return CompiledSinterDecoderXZ(
            self.detectors_x, self.detectors_z, compiled_decoder_x, compiled_decoder_z
        )


class CompiledSinterDecoderXZ(CompiledSinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit.

    This decoder splits a syndrome into "X" and "Z" sectors for the first and second half of the
    observables in a circuit.  See help(SinterDecoderXZ) for additional information.
    """

    def __init__(
        self,
        detectors_x: list[int],
        detectors_z: list[int],
        compiled_decoder_x: sinter.CompiledDecoder,
        compiled_decoder_z: sinter.CompiledDecoder,
    ) -> None:
        self.detectors_x = detectors_x
        self.detectors_z = detectors_z
        self.compiled_decoder_x = compiled_decoder_x
        self.compiled_decoder_z = compiled_decoder_z

        self.num_detectors_x = self.compiled_decoder_x.dem_arrays.num_detectors
        self.num_detectors_z = self.compiled_decoder_z.dem_arrays.num_detectors
        self.num_detectors = self.num_detectors_x + self.num_detectors_z

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        See help(sinter.CompiledDecoder) for additional information.
        """
        # unpack the syndrome, split into X and Z sectors, and repack
        syndromes = self.unpackbits(bit_packed_detection_event_data)
        syndromes_x = self.compiled_decoder_x.packbits(syndromes.T[self.detectors_x].T)
        syndromes_z = self.compiled_decoder_z.packbits(syndromes.T[self.detectors_z].T)

        # decode X and Z syndromes independently
        observable_flips_x = self.compiled_decoder_x.decode_shots_bit_packed(syndromes_x)
        observable_flips_z = self.compiled_decoder_z.decode_shots_bit_packed(syndromes_z)

        # unpack the predicted observable flips, combine them, and repack
        observable_flips = [
            self.compiled_decoder_x.unpackbits(observable_flips_x),
            self.compiled_decoder_x.unpackbits(observable_flips_z),
        ]
        return self.packbits(np.hstack(observable_flips))
