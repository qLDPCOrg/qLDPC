# SPDX-License-Identifier: Apache-2.0

"""Adapters that align or convert errors inferred by error decoders."""

from __future__ import annotations

from typing import Self, cast

import numpy as np
import numpy.typing as npt
import stim

from ..dems import DetectorErrorModelArrays
from ..protocols import (
    BatchErrorDecoder,
    DetailedErrorDecoder,
    ErrorDecoder,
    ErrorDecodeResult,
    ObservableDecoder,
    ObservableDecodeResult,
    batch_decode_errors,
)

# Observable prediction from inferred errors


class ErrorsToObservablesDecoder(ObservableDecoder):
    """Convert errors inferred by an error decoder into binary DEM observable flips.

    If the error decoder is a DetailedErrorDecoder, this decoder is a DetailedObservableDecoder that
    forwards the erasure flag and diagnostics of each detailed error result.
    """

    def __new__(cls, error_decoder: object = None, *args: object, **kwargs: object) -> Self:
        if cls is ErrorsToObservablesDecoder and isinstance(error_decoder, DetailedErrorDecoder):
            return super().__new__(cast(type[Self], _DetailedErrorsToObservablesDecoder))
        return super().__new__(cls)

    def __init__(self, error_decoder: ErrorDecoder, dem: stim.DetectorErrorModel) -> None:
        self.error_decoder = error_decoder
        self._aligned_error_decoder = match_error_decoder_to_dem(error_decoder, dem)
        self.has_erasure_bit = bool(getattr(error_decoder, "has_erasure_bit", False))
        self.observable_flip_matrix = DetectorErrorModelArrays(
            dem, simplify=False
        ).observable_flip_matrix

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode one syndrome to predicted observable flips."""
        return self.decode_observables_batch(np.asarray(syndrome)[None, :])[0]

    def decode_observables_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes to predicted observable flips."""
        errors = batch_decode_errors(self._aligned_error_decoder, syndromes)
        erasure_bits = errors[:, -1:] if self.has_erasure_bit else errors[:, :0]
        errors = errors[:, : errors.shape[1] - erasure_bits.shape[1]]
        flips = np.asarray(errors @ self.observable_flip_matrix.T) % 2
        return np.hstack([flips, erasure_bits]).astype(np.uint8)


class _DetailedErrorsToObservablesDecoder(ErrorsToObservablesDecoder):
    """An ErrorsToObservablesDecoder that forwards the diagnostics of its error decoder."""

    def decode_observables_detailed(self, syndrome: npt.NDArray[np.int_]) -> ObservableDecodeResult:
        """Decode one syndrome to observable flips, with the error decoder's diagnostics."""
        error_decoder = cast(DetailedErrorDecoder, self._aligned_error_decoder)
        result = error_decoder.decode_errors_detailed(syndrome)
        flips = np.asarray(result.error @ self.observable_flip_matrix.T) % 2
        return ObservableDecodeResult(flips.astype(np.uint8), result.erasure, result.diagnostics)


# Error-mechanism alignment


class ExpandedErrorDecoder(BatchErrorDecoder):
    """Map errors inferred for a simplified DEM back to the full DEM.

    If the wrapped decoder is a DetailedErrorDecoder, this decoder is also a DetailedErrorDecoder.
    """

    def __new__(cls, decoder: object = None, *args: object, **kwargs: object) -> Self:
        if cls is ExpandedErrorDecoder and isinstance(decoder, DetailedErrorDecoder):
            return super().__new__(cast(type[Self], _DetailedExpandedErrorDecoder))
        return super().__new__(cls)

    def __init__(self, decoder: ErrorDecoder, dem: stim.DetectorErrorModel) -> None:
        self._decoder = decoder
        self.has_erasure_bit = bool(getattr(decoder, "has_erasure_bit", False))

        original_errors = DetectorErrorModelArrays.get_circuit_errors(dem)
        simplified_errors = DetectorErrorModelArrays.get_merged_circuit_errors(original_errors)
        self._num_original_errors = len(original_errors)
        signature_to_original_error_index = {
            signature: original_error_index
            for original_error_index, (_, signature) in enumerate(original_errors)
        }
        self._simplified_to_original_index = np.array(
            [signature_to_original_error_index[signature] for _, signature in simplified_errors],
            dtype=np.intp,
        )

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode one syndrome and expand the inferred error."""
        simplified_error = self._decoder.decode_errors(syndrome)
        original_error = np.zeros(
            self._num_original_errors + self.has_erasure_bit, dtype=syndrome.dtype
        )
        if self.has_erasure_bit:
            original_error[-1] = simplified_error[-1]
            simplified_error = simplified_error[:-1]
        original_error[self._simplified_to_original_index] = simplified_error
        return np.asarray(original_error, dtype=syndrome.dtype)

    def decode_errors_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes and expand the inferred errors."""
        simplified_errors = batch_decode_errors(self._decoder, syndromes)
        original_errors = np.zeros(
            (len(syndromes), self._num_original_errors + self.has_erasure_bit),
            dtype=syndromes.dtype,
        )
        if self.has_erasure_bit:
            original_errors[:, -1] = simplified_errors[:, -1]
            simplified_errors = simplified_errors[:, :-1]
        original_errors[:, self._simplified_to_original_index] = simplified_errors
        return original_errors


def match_error_decoder_to_dem(decoder: ErrorDecoder, dem: stim.DetectorErrorModel) -> ErrorDecoder:
    """Validate an error decoder's output width and align merged errors with a DEM."""
    if getattr(decoder, "_infers_decomposed_errors", False):
        raise ValueError(
            "The error decoder infers errors in the components of decomposed error mechanisms,"
            " which cannot be read as errors of the detector error model.  To predict observable"
            " flips with decomposed errors, pass decoder settings such as"
            " decoders.mwpm(decompose_errors=True), which build a matching decoder that predicts"
            " observable flips natively"
        )
    num_erasure_bits = int(getattr(decoder, "has_erasure_bit", False))
    test_error = decoder.decode_errors(np.zeros(dem.num_detectors, dtype=int))
    num_inferred_errors = len(test_error) - num_erasure_bits
    circuit_errors = DetectorErrorModelArrays.get_circuit_errors(dem)
    num_merged_errors = len(DetectorErrorModelArrays.get_merged_circuit_errors(circuit_errors))
    if num_inferred_errors == len(circuit_errors):
        return decoder
    if num_inferred_errors == num_merged_errors:
        return ExpandedErrorDecoder(decoder, dem)
    raise ValueError(
        f"An error decoder inferred an error of length {num_inferred_errors} for a detector error"
        f" model with {len(circuit_errors)} error mechanisms ({num_merged_errors} after merging"
        " equivalent mechanisms).  If the decoder predicts observable flips rather than errors,"
        " give it a decode_observables method, and pass it where an observable decoder is accepted"
    )


class _DetailedExpandedErrorDecoder(ExpandedErrorDecoder):
    """An ExpandedErrorDecoder that forwards the diagnostics of its wrapped decoder."""

    def decode_errors_detailed(self, syndrome: npt.NDArray[np.int_]) -> ErrorDecodeResult:
        """Decode one syndrome, expand the inferred error, and keep the decoder's diagnostics."""
        result = cast(DetailedErrorDecoder, self._decoder).decode_errors_detailed(syndrome)
        error = np.zeros(self._num_original_errors, dtype=result.error.dtype)
        error[self._simplified_to_original_index] = result.error
        return ErrorDecodeResult(error, result.erasure, result.diagnostics)
