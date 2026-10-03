# SPDX-License-Identifier: Apache-2.0

"""Protocols for error and observable decoders, and helpers that coerce objects into them."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

# Decoder protocols


@runtime_checkable
class ErrorDecoder(Protocol):
    """Protocol for a decoder that maps a syndrome to an inferred error.

    An error decoder has a ``decode_errors`` method, and a ``decode`` method that is an alias for
    it.  A subclass of ErrorDecoder may implement either one, and inherits the other.

    Methods of qLDPC that accept an error decoder also accept any object whose ``decode`` method
    returns an inferred error, such as a decoder from the ldpc package, and wrap it to provide
    ``decode_errors``.
    """

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        if type(self).decode is not ErrorDecoder.decode:
            return self.decode(syndrome)
        raise NotImplementedError(f"{type(self).__name__} must implement decode_errors or decode")

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error (alias for decode_errors)."""
        return self.decode_errors(syndrome)


@runtime_checkable
class BatchErrorDecoder(ErrorDecoder, Protocol):
    """Protocol for an error decoder that can decode in batches.

    A batch error decoder has a ``decode_errors_batch`` method, and a ``decode_batch`` method that
    is an alias for it.  A subclass of BatchErrorDecoder may implement either one, and inherits the
    other.
    """

    def decode_errors_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes, one per row, and return inferred errors."""
        if type(self).decode_batch is not BatchErrorDecoder.decode_batch:
            return self.decode_batch(syndromes)
        raise NotImplementedError(
            f"{type(self).__name__} must implement decode_errors_batch or decode_batch"
        )

    def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes, one per row (alias for decode_errors_batch)."""
        return self.decode_errors_batch(syndromes)


@runtime_checkable
class ObservableDecoder(Protocol):
    """Protocol for a decoder that maps a syndrome to predicted observable flips.

    An observable decoder is built for a specific set of observables, such as those of a detector
    error model, and predicts which of them an error with the given syndrome flips.  If an
    observable decoder signals erasure (``has_erasure_bit = True``), it appends an erasure flag to
    each prediction.
    """

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return predicted observable flips."""


@runtime_checkable
class BatchObservableDecoder(ObservableDecoder, Protocol):
    """Protocol for an observable decoder that can decode in batches."""

    def decode_observables_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes, one per row, and return predicted observable flips."""


@runtime_checkable
class SupportsDecode(Protocol):
    """Protocol for an object whose decode method returns an inferred error.

    Such an object, for example a decoder from the ldpc package, is accepted wherever an error
    decoder is, and is wrapped to provide the decode_errors method of an ErrorDecoder.
    """

    def decode(self, syndrome: Any, /) -> Any:
        """Decode an error syndrome and return an inferred error."""


# Coercion into error decoders


def as_error_decoder(decoder: object, source: str = "A decoder") -> ErrorDecoder:
    """Coerce an object into an error decoder, or raise an error if it is not one.

    An ErrorDecoder is returned as is.  Another object with a decode method is wrapped in a
    WrappedErrorDecoder.  An object that predicts observable flips is rejected; this
    includes an object whose decode_returns_observables attribute is True, which declares that its
    decode method returns observable flips.

    Args:
        decoder: The object to coerce.
        source: A description of the object, which begins any error message.
    """
    predicts_observables = TypeError(
        f"{source} predicts observable flips rather than errors.  " + _OBSERVABLE_DECODER_ADVICE
    )
    if getattr(decoder, "decode_returns_observables", False):
        raise predicts_observables
    if isinstance(decoder, ErrorDecoder):
        return decoder
    if getattr(decoder, "decode_is_defunct", False):
        if isinstance(decoder, ObservableDecoder):
            raise predicts_observables
        raise TypeError(f"{source} cannot decode until it is compiled for a detector error model")
    if isinstance(decoder, SupportsDecode):
        return WrappedErrorDecoder(decoder)
    if isinstance(decoder, ObservableDecoder):
        raise predicts_observables
    raise TypeError(f"{source} must be an ErrorDecoder, or have a decode method")


_OBSERVABLE_DECODER_ADVICE = (
    "Pass error-decoder settings such as decoders.bp_osd(...), or pass the observable decoder where"
    " one is accepted, such as to decoders.SinterDecoder"
)


class WrappedErrorDecoder(ErrorDecoder):
    """Error decoder that wraps an object whose decode method returns an inferred error.

    The wrapped object is the .decoder attribute.  Its decode method provides decode_errors, its
    decode_batch method (if any) provides decode_errors_batch, and its other attributes are
    readable from the wrapper.
    """

    def __init__(self, decoder: SupportsDecode) -> None:
        self.decoder = decoder

    def __getattr__(self, name: str) -> Any:
        """Read an attribute of the wrapped object, reading decode_errors_batch as decode_batch.

        Special (dunder) attributes are not read from the wrapped object, which keeps copying and
        unpickling from recursing before the wrapped object is set.
        """
        if name == "decoder" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.decoder, "decode_batch" if name == "decode_errors_batch" else name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.decoder!r})"

    def decode_errors(self, syndrome: npt.NDArray[np.int_], *args: Any, **kwargs: Any) -> Any:
        """Decode an error syndrome and return an inferred error."""
        return self.decoder.decode(syndrome, *args, **kwargs)

    decode = decode_errors


# Batch decoding


def batch_decode_errors(
    decoder: ErrorDecoder, syndromes: npt.NDArray[np.int_]
) -> npt.NDArray[np.int_]:
    """Decode a batch of syndromes, one per row, and return inferred errors, one per row.

    The inferred errors form a two-dimensional array even if the batch is empty.
    """
    syndromes = np.asarray(syndromes)
    if (decode_errors_batch := _get_batch_decoding_method(decoder)) is not None:
        return np.asarray(decode_errors_batch(syndromes))
    if len(syndromes) == 0:
        # decode a trivial syndrome to identify the length of an inferred error
        test_error = decoder.decode_errors(np.zeros(syndromes.shape[1], dtype=syndromes.dtype))
        return np.zeros((0, len(test_error)), dtype=np.asarray(test_error).dtype)
    return np.array([decoder.decode_errors(syndrome) for syndrome in syndromes])


def supports_batch_decoding(decoder: ErrorDecoder) -> bool:
    """Whether an error decoder has a decode_errors_batch method, or its alias decode_batch."""
    return _get_batch_decoding_method(decoder) is not None


def _get_batch_decoding_method(decoder: ErrorDecoder) -> Any:
    """The decode_errors_batch method of an error decoder, its alias decode_batch, or None."""
    return getattr(decoder, "decode_errors_batch", None) or getattr(decoder, "decode_batch", None)
