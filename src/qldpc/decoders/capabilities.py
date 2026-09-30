# SPDX-License-Identifier: Apache-2.0

"""Capability checks for decoder inputs."""

from __future__ import annotations

from typing import get_type_hints

from .protocols import ErrorDecoder, ObservableDecoder, SupportsDecode
from .retrieval import DecoderSpec, is_prebuilt_decoder


def compiles_for_dem(decoder: object) -> bool:
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
    that can do both, such as a RelayBPDecoder, is used as an error decoder.
    """
    returns_observables = bool(getattr(decoder, "decode_returns_observables", False))
    is_error_decoder = isinstance(decoder, ErrorDecoder) or (
        isinstance(decoder, SupportsDecode) and not returns_observables
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


def constructs_observable_decoder(decoder: object) -> bool:
    """Whether a callable explicitly declares that it constructs an observable decoder."""
    if not callable(decoder) or is_prebuilt_decoder(decoder):
        return False
    if isinstance(decoder, type):
        return issubclass(decoder, ObservableDecoder) and not issubclass(decoder, ErrorDecoder)
    try:
        return_annotation = get_type_hints(decoder).get("return")
    except (NameError, TypeError):
        return False
    return _annotation_is_observable_decoder(return_annotation)


def _annotation_is_observable_decoder(annotation: object) -> bool:
    """Whether a return annotation identifies an observable decoder, but not an error decoder."""
    if not isinstance(annotation, type):
        return False
    return issubclass(annotation, ObservableDecoder) and not issubclass(annotation, ErrorDecoder)
