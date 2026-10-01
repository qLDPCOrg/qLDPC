# SPDX-License-Identifier: Apache-2.0

"""Resolution of typed decoder inputs into built decoders."""

from __future__ import annotations

from collections.abc import Mapping

import galois
import numpy as np
import numpy.typing as npt
import stim

from ..adapters.error_decoders import ErrorsToObservablesDecoder as _ErrorsToObservablesDecoder
from ..custom.guf import get_decoder_guf as _get_decoder_guf
from ..external.ldpc import get_decoder_bp_osd as _get_decoder_bp_osd
from ..protocols import ErrorDecoder, ObservableDecoder, as_error_decoder
from .specs import (
    DecoderSpec,
    ErrorDecoderInput,
    ObservableDecoderInput,
    PcmOrDem,
)

# Modern decoder resolution APIs


def is_prebuilt_decoder(decoder: object) -> bool:
    """Whether a decoder input is already built rather than settings or a constructor."""
    return (
        decoder is not None
        and not isinstance(decoder, (DecoderSpec, type))
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


def reject_prebuilt_decoder(decoder: object, reason: str) -> None:
    """Raise if a decoder input is prebuilt and cannot be rebuilt for a new matrix."""
    if is_prebuilt_decoder(decoder):
        raise ValueError(
            f"A prebuilt decoder cannot be passed as decoder= here because {reason}.  Pass decoder"
            " settings such as decoder=decoders.bp_osd(...), or a decoder constructor, instead"
        )


def reject_removed_decoder_args(decoder_args: Mapping[str, object]) -> None:
    """Reject the removed static_decoder argument outside the legacy APIs."""
    if "static_decoder" in decoder_args:
        raise TypeError(
            "The static_decoder argument has been removed; pass a prebuilt decoder as decoder="
            " instead"
        )


def get_error_decoder(pcm_or_dem: PcmOrDem, *, decoder: ErrorDecoderInput = None) -> ErrorDecoder:
    """Build or retrieve a decoder that maps a syndrome to an inferred error."""
    return as_error_decoder(*_build_decoder(pcm_or_dem, decoder))


def resolve_decoder(
    pcm_or_dem: PcmOrDem,
    decoder: ErrorDecoderInput,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> ErrorDecoder:
    """Resolve an error decoder input and optional deprecated construction arguments."""
    from .legacy import _merge_legacy_decoder_args

    decoder_input = _merge_legacy_decoder_args(
        pcm_or_dem, decoder, decoder_args, warn_deprecated=warn_deprecated
    )
    return as_error_decoder(*_build_decoder(pcm_or_dem, decoder_input))


def decode_observables(
    dem: stim.DetectorErrorModel,
    syndrome: npt.NDArray[np.int_],
    *,
    decoder: ObservableDecoderInput = None,
) -> npt.NDArray[np.int_]:
    """Construct a decoder and predict the observable flips of one syndrome."""
    return get_observable_decoder(dem, decoder=decoder).decode_observables(syndrome)


def get_observable_decoder(
    dem: stim.DetectorErrorModel, *, decoder: ObservableDecoderInput = None
) -> ObservableDecoder:
    """Build or retrieve an observable decoder for a detector error model."""
    return resolve_observable_decoder(dem, decoder, {})


def resolve_observable_decoder(
    dem: stim.DetectorErrorModel,
    decoder: ObservableDecoderInput,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> ObservableDecoder:
    """Resolve an observable decoder input and optional deprecated construction arguments."""
    if decoder_args:
        from .legacy import _merge_legacy_decoder_args

        decoder = _merge_legacy_decoder_args(
            dem, decoder, decoder_args, warn_deprecated=warn_deprecated
        )
    elif isinstance(decoder, DecoderSpec):
        return decoder.build_observable_decoder(dem)

    built_decoder, source = _build_decoder(dem, decoder)
    if isinstance(built_decoder, ObservableDecoder):
        return built_decoder
    return _ErrorsToObservablesDecoder(as_error_decoder(built_decoder, source), dem)


# Resolution internals


def _build_decoder(pcm_or_dem: PcmOrDem, decoder: ObservableDecoderInput) -> tuple[object, str]:
    """Build or retrieve a decoder without constraining its output kind."""
    built_decoder: object
    if decoder is None:
        is_nonbinary = isinstance(pcm_or_dem, galois.FieldArray) and type(pcm_or_dem).order != 2
        default_builder = _get_decoder_guf if is_nonbinary else _get_decoder_bp_osd
        built_decoder, source = default_builder(pcm_or_dem), "The default decoder"
    elif isinstance(decoder, DecoderSpec):
        built_decoder = decoder.build(pcm_or_dem)
        source = "A decoder spec"
    elif is_prebuilt_decoder(decoder):
        built_decoder, source = decoder, "A prebuilt decoder"
    elif callable(decoder):
        built_decoder, source = decoder(pcm_or_dem), "A decoder constructor"
    else:
        raise TypeError(
            "decoder must be decoder settings such as decoders.bp_osd(...), a decoder constructor,"
            " a prebuilt error decoder, or None"
        )
    return built_decoder, source
