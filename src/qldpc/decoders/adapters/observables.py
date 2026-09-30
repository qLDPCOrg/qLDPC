# SPDX-License-Identifier: Apache-2.0

"""Adapters that normalize observable-decoder outputs."""

from __future__ import annotations

from typing import Any

import galois
import numpy as np
import numpy.typing as npt

from ..protocols import ErrorDecoder, ObservableDecoder


def validate_decoder_output(
    output: npt.NDArray[Any],
    num_values: int,
    num_erasure_flags: int,
    field: type[galois.FieldArray],
    source: str,
) -> None:
    """Check that a decoder output holds num_values field elements followed by erasure flags."""
    expected_shape = (num_values + num_erasure_flags,)
    if output.shape != expected_shape:
        flags = f" and {num_erasure_flags} erasure flag(s)" if num_erasure_flags else ""
        raise ValueError(
            f"{source} of shape {output.shape}, but expected shape {expected_shape}:"
            f" {num_values} value(s){flags}"
        )
    if not (np.issubdtype(output.dtype, np.integer) or np.issubdtype(output.dtype, np.bool_)):
        raise ValueError(f"{source} of dtype {output.dtype}, but expected integers")
    values, erasure_flags = output[:num_values].astype(int), output[num_values:].astype(int)
    if np.any(values < 0) or np.any(values >= field.order):
        raise ValueError(f"{source} with entries that are not elements of {field.name}")
    if np.any((erasure_flags != 0) & (erasure_flags != 1)):
        raise ValueError(f"{source} with erasure flags that are not 0 or 1")


class ErrorsToFieldObservablesDecoder(ObservableDecoder):
    """Observable decoder that converts the errors that an error decoder infers into observables.

    The observable values of an inferred error are ``observable_matrix @ error``, over the field of
    observable_matrix.  If the error decoder signals erasure, its erasure bit is appended to each
    prediction.
    """

    def __init__(
        self,
        error_decoder: ErrorDecoder,
        observable_matrix: galois.FieldArray | None,
        field: type[galois.FieldArray],
        num_error_locations: int,
    ) -> None:
        self.error_decoder = error_decoder
        self.observable_matrix = observable_matrix
        self.field = field
        self.num_error_locations = num_error_locations
        self.has_erasure_bit = bool(getattr(error_decoder, "has_erasure_bit", False))

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return predicted observable values."""
        num_error_locations = self.num_error_locations
        error = np.asarray(self.error_decoder.decode_errors(syndrome))
        validate_decoder_output(
            error,
            num_error_locations,
            int(self.has_erasure_bit),
            self.field,
            "An error decoder inferred an error",
        )
        inferred_error = self.field(error[:num_error_locations].astype(int))
        observables = (
            inferred_error
            if self.observable_matrix is None
            else self.observable_matrix @ inferred_error
        )
        return np.concatenate([observables.view(np.ndarray), error[num_error_locations:]])


class BitPackedObservableDecoder(ObservableDecoder):
    """Observable decoder that wraps a compiled Sinter decoder with bit-packed inputs and outputs.

    The compiled decoder predicts one bit-packed byte per eight observables, and may add one byte,
    which asks for the shot to be discarded if it is nonzero.  Each prediction of this decoder ends
    with an erasure flag that is set if the shot is to be discarded.
    """

    has_erasure_bit = True

    def __init__(self, compiled_decoder: Any, num_observables: int) -> None:
        self.compiled_decoder = compiled_decoder
        self.num_observables = num_observables

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return predicted observable flips and an erasure flag."""
        packed_syndrome = np.packbits(
            np.asarray(syndrome, dtype=np.uint8).reshape(1, -1), bitorder="little", axis=1
        )
        packed_prediction = np.asarray(
            self.compiled_decoder.decode_shots_bit_packed(
                bit_packed_detection_event_data=packed_syndrome
            ),
            dtype=np.uint8,
        )
        num_bytes = -(-self.num_observables // 8)
        if packed_prediction.shape not in [(1, num_bytes), (1, num_bytes + 1)]:
            raise ValueError(
                f"A compiled Sinter decoder predicted bit-packed observable flips of shape"
                f" {packed_prediction.shape} for one shot, but {self.num_observables} observables"
                f" take shape (1, {num_bytes}), or (1, {num_bytes + 1}) with a byte added to signal"
                " discards"
            )
        flips = np.unpackbits(
            packed_prediction[0, :num_bytes], count=self.num_observables, bitorder="little"
        )
        erased = bool(np.any(packed_prediction[0, num_bytes:]))
        return np.append(flips, np.uint8(erased))
