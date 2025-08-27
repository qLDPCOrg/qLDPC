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

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(dem)
        priors = dem_arrays.error_probs
        if self.log_likelihood_priors:
            priors = np.log((1 - priors) / priors)
        priors_kwarg = {self.priors_arg: list(priors)} if self.priors_arg else {}
        decoder = get_decoder(
            dem_arrays.detector_flip_matrix, **self.decoder_kwargs, **priors_kwarg
        )
        return CompiledSinterDecoder(dem_arrays, decoder)


class CompiledSinterDecoder(sinter.CompiledDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit."""

    def __init__(self, dem_arrays: DetectorErrorModelArrays, decoder: Decoder) -> None:
        self.dem_arrays = dem_arrays
        self.decoder = decoder

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        See help(sinter.CompiledDecoder) for additional information.
        """
        syndromes = np.unpackbits(
            bit_packed_detection_event_data,
            count=self.dem_arrays.num_detectors,
            bitorder="little",
            axis=1,
        )
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
        return np.packbits(np.array(observable_flips, dtype=np.uint8), bitorder="little", axis=1)
