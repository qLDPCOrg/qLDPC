# SPDX-License-Identifier: Apache-2.0

"""Adapters that normalize or wrap observable decoders."""

from __future__ import annotations

from typing import Any

import galois
import numpy as np
import numpy.typing as npt

from ..protocols import ObservableDecoder

####################################################################################################
# observable decoder adapters


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
        packed_prediction = self.compiled_decoder.decode_shots_bit_packed(
            bit_packed_detection_event_data=packed_syndrome
        )
        packed_flips, discarded = split_discard_byte(packed_prediction, 1, self.num_observables)
        flips = np.unpackbits(packed_flips[0], count=self.num_observables, bitorder="little")
        return np.append(flips, np.uint8(discarded[0]))


def split_discard_byte(
    packed_predictions: npt.ArrayLike, num_shots: int, num_observables: int
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
    """Split bit-packed predictions into packed observable flips and a mask of discarded shots.

    A compiled Sinter decoder predicts one bit-packed byte per eight observables for each shot, and
    may add one byte that asks for the shot to be discarded if it is nonzero, as in
    qldpc.decoders.sinter.CompiledSinterDecoder.pack_observable_flips.

    Args:
        packed_predictions: Bit-packed predictions of a compiled Sinter decoder, one row per shot.
        num_shots: The number of shots that were decoded.
        num_observables: The number of observables of the decoded detector error model.

    Returns:
        The bit-packed observable flips without the added byte, and a boolean array that marks the
        shots to discard.

    Raises:
        ValueError: If the predictions do not have one row per shot of the expected width.
    """
    packed_predictions = np.asarray(packed_predictions, dtype=np.uint8)
    num_bytes = -(-num_observables // 8)
    if packed_predictions.shape not in [(num_shots, num_bytes), (num_shots, num_bytes + 1)]:
        raise ValueError(
            f"A decoder predicted bit-packed observable flips of shape {packed_predictions.shape},"
            f" but {num_shots} shot(s) of {num_observables} observable(s) take shape"
            f" {(num_shots, num_bytes)}, or {(num_shots, num_bytes + 1)} with a byte added to"
            " signal discards"
        )
    return packed_predictions[:, :num_bytes], np.any(packed_predictions[:, num_bytes:], axis=1)


####################################################################################################
# validation helpers


def validate_observable_decoder(decoder: object, source: str) -> ObservableDecoder:
    """Validate and type-narrow an object expected to decode to observable flips."""
    if not isinstance(decoder, ObservableDecoder):
        raise TypeError(f"{source} must provide a decode_observables method")
    return decoder


def validate_decoder_output(
    output: npt.NDArray[np.generic],
    num_values: int,
    num_erasure_bits: int,
    field: type[galois.FieldArray],
    source: str,
) -> None:
    """Check that a decoder output holds num_values field elements followed by erasure flags."""
    expected_shape = (num_values + num_erasure_bits,)
    if output.shape != expected_shape:
        flags = f" and {num_erasure_bits} erasure flag(s)" if num_erasure_bits else ""
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
