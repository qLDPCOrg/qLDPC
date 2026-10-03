# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for decoders."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Sequence
from typing import ParamSpec, TypeAlias, TypeVar

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc._util import get_external_caller_stacklevel
from qldpc.math import IntegerArray

from .dems import DetectorErrorModelArrays
from .protocols import (
    ErrorDecoder,
    ErrorDecodeResult,
    ObservableDecoder,
    ObservableDecodeResult,
    SupportsDecode,
    as_error_decoder,
)

PLACEHOLDER_ERROR_RATE = 1e-3  # required for some decoding methods


_PcmOrDem: TypeAlias = IntegerArray | stim.DetectorErrorModel
_Parameters = ParamSpec("_Parameters")
_Decoder = TypeVar("_Decoder", bound=ErrorDecoder)
_ErrorChannel: TypeAlias = float | npt.NDArray[np.floating] | Sequence[float] | None


# Detailed decode results


def decode_errors_detailed(
    decoder: ErrorDecoder | SupportsDecode, syndrome: npt.NDArray[np.int_]
) -> ErrorDecodeResult:
    """Decode one syndrome and return its inferred error, erasure flag, and diagnostics.

    A decoder that implements ``decode_errors_detailed`` supplies its own diagnostics. Otherwise,
    this helper wraps its existing hard prediction and separates any qLDPC-appended erasure bit.
    """
    error_decoder = as_error_decoder(decoder)
    detailed_decoder = getattr(error_decoder, "decode_errors_detailed", None)
    if detailed_decoder is not None:
        result = detailed_decoder(syndrome)
        if not isinstance(result, ErrorDecodeResult):
            raise TypeError("decode_errors_detailed must return an ErrorDecodeResult")
        return result

    error = np.asarray(error_decoder.decode_errors(syndrome))
    erasure = False
    if getattr(error_decoder, "has_erasure_bit", False):
        erasure = bool(error[-1])
        error = error[:-1]
    return ErrorDecodeResult(error, erasure)


def decode_errors_detailed_batch(
    decoder: ErrorDecoder | SupportsDecode, syndromes: npt.NDArray[np.int_]
) -> tuple[ErrorDecodeResult, ...]:
    """Decode a batch and return one detailed error result per syndrome, in input order."""
    error_decoder = as_error_decoder(decoder)
    detailed_batch_decoder = getattr(error_decoder, "decode_errors_detailed_batch", None)
    if detailed_batch_decoder is not None:
        results = tuple(detailed_batch_decoder(np.asarray(syndromes)))
        if len(results) != len(syndromes) or any(
            not isinstance(result, ErrorDecodeResult) for result in results
        ):
            raise TypeError(
                "decode_errors_detailed_batch must return one ErrorDecodeResult per syndrome"
            )
        return results
    return tuple(decode_errors_detailed(error_decoder, syndrome) for syndrome in syndromes)


def decode_observables_detailed(
    decoder: ObservableDecoder, syndrome: npt.NDArray[np.int_]
) -> ObservableDecodeResult:
    """Decode one syndrome and return observable flips, erasure, and diagnostics.

    A decoder that implements ``decode_observables_detailed`` supplies its own diagnostics.
    Otherwise, this helper wraps its existing hard prediction and separates any qLDPC-appended
    erasure bit.
    """
    detailed_decoder = getattr(decoder, "decode_observables_detailed", None)
    if detailed_decoder is not None:
        result = detailed_decoder(syndrome)
        if not isinstance(result, ObservableDecodeResult):
            raise TypeError("decode_observables_detailed must return an ObservableDecodeResult")
        return result

    flips = np.asarray(decoder.decode_observables(syndrome))
    erasure = False
    if getattr(decoder, "has_erasure_bit", False):
        erasure = bool(flips[-1])
        flips = flips[:-1]
    return ObservableDecodeResult(flips, erasure)


def decode_observables_detailed_batch(
    decoder: ObservableDecoder, syndromes: npt.NDArray[np.int_]
) -> tuple[ObservableDecodeResult, ...]:
    """Decode a batch and return one detailed observable result per syndrome, in input order."""
    detailed_batch_decoder = getattr(decoder, "decode_observables_detailed_batch", None)
    if detailed_batch_decoder is not None:
        results = tuple(detailed_batch_decoder(np.asarray(syndromes)))
        if len(results) != len(syndromes) or any(
            not isinstance(result, ObservableDecodeResult) for result in results
        ):
            raise TypeError(
                "decode_observables_detailed_batch must return one ObservableDecodeResult per"
                " syndrome"
            )
        return results
    return tuple(decode_observables_detailed(decoder, syndrome) for syndrome in syndromes)


# Erasure signaling


def get_error_and_erasure(
    decoder: ErrorDecoder | SupportsDecode,
    syndrome: galois.FieldArray,
) -> tuple[galois.FieldArray, bool]:
    """Decode a syndrome and return the inferred error together with an erasure flag.

    If the decoder has a has_erasure_bit attribute set to True (e.g., a LookupDecoder constructed
    with ``add_erasure_bit=True``), the last element of the decoded vector is treated as the erasure
    bit: 1 means the syndrome was not recognized and the sample should be discarded, 0 means a
    correction was found normally.  The erasure bit is stripped before returning the error.
    """
    error = as_error_decoder(decoder).decode_errors(syndrome.view(np.ndarray))
    if getattr(decoder, "has_erasure_bit", False):
        return error[:-1].view(type(syndrome)), bool(error[-1])
    return error.view(type(syndrome)), False


def with_erasure_bits(
    errors: npt.NDArray[np.int_], erased: npt.NDArray[np.bool_] | bool
) -> npt.NDArray[np.int_]:
    """Append an erasure bit to each inferred error, in the dtype of that error.

    A decoder that signals erasure reports it in the last entry of every error it infers, set to 1
    for a syndrome that the error does not reproduce.  Accepts one error with one flag, or a batch
    of errors with one flag per error.
    """
    flags = np.asarray(erased, dtype=errors.dtype).reshape(errors.shape[:-1] + (1,))
    return np.hstack([errors, flags])


# Private builder helpers


def _erasure_bit_support(
    display_name: str,
    *,
    supported: bool,
) -> Callable[[Callable[_Parameters, _Decoder]], Callable[_Parameters, _Decoder]]:
    """Declare and enforce whether a decoder builder supports an erasure bit."""

    def decorator(
        decoder_builder: Callable[_Parameters, _Decoder],
    ) -> Callable[_Parameters, _Decoder]:
        message = (
            f"The {display_name} decoder cannot signal erasure, so it does not accept the"
            " add_erasure_bit argument"
        )

        @functools.wraps(decoder_builder)
        def checked_builder(*args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Decoder:
            add_erasure_bit = bool(kwargs.get("add_erasure_bit"))
            if not supported:
                if add_erasure_bit:
                    raise ValueError(message)
                kwargs.pop("add_erasure_bit", None)

            decoder = decoder_builder(*args, **kwargs)
            if add_erasure_bit and not getattr(decoder, "has_erasure_bit", False):
                raise ValueError(message)
            return decoder

        return checked_builder

    return decorator


def _to_pcm(pcm_or_dem: _PcmOrDem) -> IntegerArray:
    """Return a parity-check matrix, densifying a detector error model."""
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        return DetectorErrorModelArrays(pcm_or_dem).detector_flip_matrix.toarray()
    return pcm_or_dem


def _get_matrix_error_channel(
    pcm_or_dem: _PcmOrDem,
    error_channel: _ErrorChannel,
    error_rate: float | None,
) -> npt.NDArray[np.floating] | None:
    """Normalize matrix error probabilities and reject explicit probabilities for a DEM."""
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        _reject_dem_error_probabilities(error_channel, error_rate)
        return None

    if error_rate is not None:
        if error_channel is not None:
            raise ValueError("error_rate and error_channel cannot both be specified")
        warnings.warn(
            f"error_rate={error_rate!r} is deprecated; use error_channel={error_rate!r} instead",
            DeprecationWarning,
            stacklevel=get_external_caller_stacklevel(),
        )
        error_channel = error_rate
    if error_channel is None:
        error_channel = PLACEHOLDER_ERROR_RATE

    if np.isscalar(error_channel):
        probabilities = np.full(pcm_or_dem.shape[1], error_channel, dtype=float)
    else:
        probabilities = np.asarray(error_channel, dtype=float)
    expected_shape = (pcm_or_dem.shape[1],)
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"error probabilities of shape {expected_shape} are required in error_channel, but got"
            f" {probabilities.shape}"
        )
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("error_channel probabilities must be finite and between 0 and 1")
    return probabilities


def _reject_dem_error_probabilities(error_channel: object, error_rate: object) -> None:
    """Reject explicit error probabilities for a detector error model, which supplies its own.

    Earlier qLDPC releases ignored error_rate for a detector error model, and let error_channel
    override its probabilities, so the error explains how to migrate.
    """
    options = {"error_channel": error_channel, "error_rate": error_rate}
    if specified := [f"{name}={value!r}" for name, value in options.items() if value is not None]:
        raise ValueError(
            "A detector error model supplies its own error probabilities, so"
            f" {' and '.join(specified)} cannot be specified with one.  (In qLDPC 0.4.0, the ldpc"
            " decoders BP+OSD, BP+LSD, and BF ignored error_rate for a detector error model, and"
            " let error_channel override its probabilities.)  Remove the option, as for a"
            " SinterDecoder, which always decodes detector error models.  Alternatively, pass"
            " error_channel with the detector-flip matrix of the model,"
            " decoders.DetectorErrorModelArrays(dem).detector_flip_matrix"
        )


# Deprecated compatibility helpers


def _deprecate_error_rate_option(
    options: dict[str, object], explicitly_provided: frozenset[str]
) -> dict[str, object]:
    """Replace an explicitly supplied error_rate option with error_channel.

    An explicit error_rate=None is the default value, so it is dropped as if it were omitted.
    """
    if "error_rate" not in explicitly_provided or options.get("error_rate") is None:
        options.pop("error_rate", None)
        return options
    if "error_channel" in explicitly_provided:
        raise ValueError("error_rate and error_channel cannot both be specified")
    error_rate = options["error_rate"]
    warnings.warn(
        f"error_rate={error_rate!r} is deprecated; use error_channel={error_rate!r} instead",
        DeprecationWarning,
        stacklevel=get_external_caller_stacklevel(),
    )
    options.pop("error_rate")
    options["error_channel"] = error_rate
    return options
