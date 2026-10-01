# SPDX-License-Identifier: Apache-2.0

"""Decoder composition helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import galois
import numpy as np
import numpy.typing as npt

from qldpc.math import IntegerArray

from ..protocols import (
    ErrorDecoder,
    SupportsDecode,
    as_error_decoder,
    batch_decode_errors,
    supports_batch_decoding,
)


class CompositeDecoder(ErrorDecoder):
    """Decoder for a composite syndrome from multiple independent code blocks.

    A CompositeDecoder is instantiated from a sequence of tuples, where each tuple contains

    (a) the decoder for a one code block
    (b) the length of a syndrome vector for that code block.

    When asked to decode a syndrome, a CompositeDecoder splits the syndrome into segments of
    appropriate lengths, and decodes these segments independently with their corresponding decoders.

    Decoded segments are concatenated.  Any number of the decoders may have an erasure bit; a
    CompositeDecoder collects them into the single erasure bit that it advertises as its own, which
    is set whenever any code block is erased.
    """

    def __init__(
        self, *decoders_and_syndrome_lengths: tuple[ErrorDecoder | SupportsDecode, int]
    ) -> None:
        self.decoders, syndrome_lengths = zip(*decoders_and_syndrome_lengths)
        self._error_decoders = tuple(as_error_decoder(decoder) for decoder in self.decoders)
        self.erasing_decoders = tuple(
            bool(getattr(decoder, "has_erasure_bit", False)) for decoder in self.decoders
        )
        self.has_erasure_bit = any(self.erasing_decoders)
        self.slices = tuple(
            slice(sum(syndrome_lengths[:ss]), sum(syndrome_lengths[: ss + 1]))
            for ss in range(len(syndrome_lengths))
        )

        self.decode_batch_implemented = all(
            supports_batch_decoding(decoder) for decoder in self._error_decoders
        )
        if self.decode_batch_implemented:
            self.decode_errors_batch = self.decode_batch = self._decode_batch

    @staticmethod
    def from_copies(
        decoder: ErrorDecoder | SupportsDecode, syndrome_length: int, num_copies: int
    ) -> CompositeDecoder:
        """Initialize a CompositeDecoder from copies of a given decoder and syndrome_length."""
        return CompositeDecoder(*[(decoder, syndrome_length)] * num_copies)

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome by parts."""
        return self._join_segments(
            [
                decoder.decode_errors(syndrome[slice])
                for decoder, slice in zip(self._error_decoders, self.slices)
            ]
        )

    def _decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes by parts."""
        return self._join_segments(
            [
                batch_decode_errors(decoder, syndromes[:, slice])
                for decoder, slice in zip(self._error_decoders, self.slices)
            ]
        )

    def _join_segments(self, segments: Sequence[npt.NDArray[np.int_]]) -> npt.NDArray[np.int_]:
        """Concatenate decoded segments, collecting their erasure bits into one trailing bit."""
        if not self.has_erasure_bit:
            return np.concatenate(segments, axis=-1)

        errors = []
        erased = np.zeros(segments[0].shape[:-1], dtype=bool)
        for segment, erasing in zip(segments, self.erasing_decoders):
            if erasing:
                erased = erased | (segment[..., -1] != 0)
                segment = segment[..., :-1]
            errors.append(segment)
        errors.append(erased[..., None].astype(segments[0].dtype))
        return np.concatenate(errors, axis=-1)


class DirectDecoder:
    """Decoder that maps corrupted code words to corrected code words.

    In contrast, an "indirect" decoder maps a syndrome to an error.

    A DirectDecoder can be instantiated from:

    - an indirect decoder, and
    - a parity check matrix.

    When asked to decode a candidate code word, a DirectDecoder first computes a syndrome, decodes
    the syndrome with an indirect decoder to infer an error, and then subtracts the error from the
    candidate word.
    """

    def __init__(
        self,
        decode_func: Callable[[npt.NDArray[np.int_]], npt.NDArray[np.int_]],
        decode_batch_func: Callable[[npt.NDArray[np.int_]], npt.NDArray[np.int_]] | None = None,
    ) -> None:
        self.decode_func = decode_func
        self.decode_batch_func = decode_batch_func
        if decode_batch_func is not None:
            self.decode_batch = decode_batch_func

    def decode(self, word: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a corrupted code word and return a corrected code word."""
        return self.decode_func(word)

    @staticmethod
    def from_indirect(
        decoder: ErrorDecoder | SupportsDecode, matrix: IntegerArray
    ) -> DirectDecoder:
        """Instantiate a DirectDecoder from an indirect decoder and a parity check matrix."""
        field = type(matrix) if isinstance(matrix, galois.FieldArray) else galois.GF2
        field_matrix = matrix.view(field)
        error_decoder = as_error_decoder(decoder)

        def check_subtractable(errors: npt.NDArray[np.int_], words: npt.NDArray[np.int_]) -> None:
            """Reject inferred errors that cannot be subtracted from candidate code words."""
            if errors.shape != words.shape:
                raise ValueError(
                    f"The given decoder inferred errors of shape {errors.shape}, which cannot be"
                    f" subtracted from candidate code words of shape {words.shape}.  A decoder that"
                    " appends an erasure bit, or that predicts observable flips rather than an"
                    " error, cannot be used to decode code words directly."
                )

        def decode_func(candidate_word: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            candidate_word = candidate_word.view(field)
            syndrome = field_matrix @ candidate_word
            error = error_decoder.decode_errors(syndrome.view(np.ndarray)).view(field)
            check_subtractable(error, candidate_word)
            return (candidate_word - error).view(np.ndarray)

        decode_batch_func: Callable[[npt.NDArray[np.int_]], npt.NDArray[np.int_]] | None = None

        if supports_batch_decoding(error_decoder):

            def decode_batch_func(candidate_words: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
                candidate_words = candidate_words.view(field)
                syndromes = candidate_words @ field_matrix.T
                errors = batch_decode_errors(error_decoder, syndromes.view(np.ndarray)).view(field)
                check_subtractable(errors, candidate_words)
                return (candidate_words - errors).view(np.ndarray)

        return DirectDecoder(decode_func, decode_batch_func)
