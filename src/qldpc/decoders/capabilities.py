# SPDX-License-Identifier: Apache-2.0

"""Capability checks for decoder inputs."""

from __future__ import annotations

from typing import TypeGuard

from .construction.specs import DecoderSpec, ObservableDecoderCompiler
from .protocols import ErrorDecoder, ObservableDecoder, SupportsDecode


def is_prebuilt_decoder(decoder: object) -> bool:
    """Whether a decoder input is already built for a matrix or detector error model.

    A prebuilt decoder has a decoding method.  A decoder specification, a constructor, and a
    Sinter-style decoder that is compiled for a detector error model (see compiles_for_dem) are not
    prebuilt.
    """
    return (
        decoder is not None
        and not isinstance(decoder, (DecoderSpec, type))
        and not compiles_for_dem(decoder)
        and any(
            hasattr(decoder, method)
            for method in (
                "decode_errors",
                "decode",
                "decode_observables",
                "decode_shots_bit_packed",
            )
        )
    )


def compiles_for_dem(decoder: object) -> TypeGuard[ObservableDecoderCompiler]:
    """Whether a decoder input is a Sinter-style decoder, compiled for a detector error model.

    Such a decoder, like a decoders.SinterDecoder, has a compile_decoder_for_dem method.
    """
    return not isinstance(decoder, type) and callable(
        getattr(decoder, "compile_decoder_for_dem", None)
    )


def is_prebuilt_observable_decoder(decoder: object) -> bool:
    """Whether a decoder input is a prebuilt decoder that predicts observables, but not errors.

    Such a decoder has a decode_observables method, or is a compiled Sinter decoder with a
    decode_shots_bit_packed method.  It is not a Sinter-style decoder that still has to be compiled
    for a detector error model (see compiles_for_dem), and it is not an error decoder: a decoder
    that can do both, such as a RelayBPDecoder, is used as an error decoder if passed prebuilt to
    a code-capacity estimator, since it cannot be rebuilt for that sector's observables.
    """
    returns_observables = bool(getattr(decoder, "decode_returns_observables", False))
    is_error_decoder = isinstance(decoder, ErrorDecoder) or (
        isinstance(decoder, SupportsDecode)
        and not returns_observables
        and not getattr(decoder, "decode_is_defunct", False)
    )
    return (
        not isinstance(decoder, (type, DecoderSpec))
        and not is_error_decoder
        and not compiles_for_dem(decoder)
        and (
            isinstance(decoder, ObservableDecoder)
            or callable(getattr(decoder, "decode_shots_bit_packed", None))
        )
    )
