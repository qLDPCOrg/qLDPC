# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for decoders."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeAlias, TypeVar

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc.math import IntegerArray

from .dems import DetectorErrorModelArrays
from .protocols import ErrorDecoder, SupportsDecode, as_error_decoder

PLACEHOLDER_ERROR_RATE = 1e-3  # required for some decoding methods

_PcmOrDem: TypeAlias = IntegerArray | stim.DetectorErrorModel
_Parameters = ParamSpec("_Parameters")
_Decoder = TypeVar("_Decoder", bound=ErrorDecoder)


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
