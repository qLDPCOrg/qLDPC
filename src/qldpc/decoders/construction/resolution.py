# SPDX-License-Identifier: Apache-2.0

"""Resolution of typed decoder inputs into built decoders."""

from __future__ import annotations

import galois
import numpy as np
import numpy.typing as npt
import stim

from ..adapters.error_decoders import ErrorsToObservablesDecoder as _ErrorsToObservablesDecoder
from ..adapters.observable_decoders import BitPackedObservableDecoder as _BitPackedObservableDecoder
from ..capabilities import compiles_for_dem, is_prebuilt_decoder
from ..custom.guf import get_decoder_guf as _get_decoder_guf
from ..external.ldpc import get_decoder_bp_osd as _get_decoder_bp_osd
from ..protocols import ErrorDecoder, ObservableDecoder, as_error_decoder
from .specs import (
    DecoderInput,
    DecoderSpec,
    ErrorDecoderInput,
    ObservableDecoderCompiler,
    PcmOrDem,
)

# Modern decoder resolution APIs


def get_error_decoder(pcm_or_dem: PcmOrDem, *, decoder: ErrorDecoderInput = None) -> ErrorDecoder:
    """Build or retrieve a decoder that maps a syndrome to an inferred error.

    Args:
        pcm_or_dem: The parity-check matrix or detector error model to decode.
        decoder: Decoder settings such as ``decoders.bp_osd(...)``, a constructor that builds an
            error decoder from pcm_or_dem, a prebuilt error decoder, or None to select the default
            decoder: GUF for a nonbinary FieldArray, and BP+OSD otherwise.

    Returns:
        An ErrorDecoder.
    """
    return as_error_decoder(*_build_decoder(pcm_or_dem, decoder))


def get_observable_decoder(
    dem: stim.DetectorErrorModel, *, decoder: DecoderInput = None
) -> ObservableDecoder:
    """Build or retrieve a decoder that maps a syndrome to predicted observable flips.

    Args:
        dem: The detector error model to decode.
        decoder: Decoder settings such as ``decoders.mwpm(...)``, which build a native observable
            decoder where the settings support one, and otherwise an error decoder; an
            observable-decoder compiler such as ``decoders.frontier(...)`` or a
            ``decoders.SinterDecoder``, which is compiled for dem; a constructor that builds an
            error decoder or an observable decoder from dem; a prebuilt error decoder or observable
            decoder; or None to select the default decoder.  An error decoder is wrapped so that the
            observable flips of the errors that it infers become its predictions.  A decoder that is
            both an error decoder and an observable decoder is used as an observable decoder.

    Returns:
        An ObservableDecoder.
    """
    if isinstance(decoder, DecoderSpec):
        return decoder.build_observable_decoder(dem)
    if compiles_for_dem(decoder):
        return _compile_observable_decoder(decoder, dem)
    built_decoder, source = _build_decoder(dem, decoder)
    if isinstance(built_decoder, ObservableDecoder):
        return built_decoder
    return _ErrorsToObservablesDecoder(as_error_decoder(built_decoder, source), dem)


def decode_observables(
    dem: stim.DetectorErrorModel,
    syndrome: npt.NDArray[np.int_],
    *,
    decoder: DecoderInput = None,
) -> npt.NDArray[np.int_]:
    """Construct a decoder and predict the observable flips of one syndrome.

    See help(qldpc.decoders.get_observable_decoder) for the accepted decoder inputs.
    """
    return get_observable_decoder(dem, decoder=decoder).decode_observables(syndrome)


def reject_prebuilt_decoder(decoder: object, reason: str) -> None:
    """Raise if a decoder input is prebuilt and cannot be rebuilt for a new matrix."""
    if is_prebuilt_decoder(decoder):
        raise ValueError(
            f"A prebuilt decoder cannot be passed as decoder= here because {reason}.  Pass decoder"
            " settings such as decoder=decoders.bp_osd(...), or a decoder constructor, instead"
        )


# Resolution internals


def _build_decoder(pcm_or_dem: PcmOrDem, decoder: DecoderInput) -> tuple[object, str]:
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


def _compile_observable_decoder(
    compiler: ObservableDecoderCompiler, dem: stim.DetectorErrorModel
) -> ObservableDecoder:
    """Compile an observable decoder for a detector error model.

    Besides an ObservableDecoder, this accepts a compiled Sinter decoder that only decodes
    bit-packed shots, such as one compiled by a decoder in sinter.BUILT_IN_DECODERS.
    """
    compiled_decoder: object = compiler.compile_decoder_for_dem(dem=dem)
    if isinstance(compiled_decoder, ObservableDecoder):
        return compiled_decoder
    if callable(getattr(compiled_decoder, "decode_shots_bit_packed", None)):
        return _BitPackedObservableDecoder(compiled_decoder, dem.num_observables)
    raise TypeError(
        "A decoder compiled by compile_decoder_for_dem must provide a decode_observables or"
        " decode_shots_bit_packed method"
    )
