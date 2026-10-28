# SPDX-License-Identifier: Apache-2.0

"""Resolution of typed decoder inputs into built decoders."""

from __future__ import annotations

import galois
import stim

from .._capabilities import compiles_for_dem, is_prebuilt_decoder
from ..adapters._observable_decoders import BitPackedObservableDecoder, validate_observable_decoder
from ..adapters.error_decoders import ErrorsToObservablesDecoder
from ..common import _PLACEHOLDER_ERROR_RATE
from ..custom.guf import guf
from ..external.ldpc import bp_osd
from ..protocols import ErrorDecoder, ObservableDecoder, _as_error_decoder
from .factories import DEMDecoderFactory, MatrixDecoderFactory
from .specs import (
    DecoderInput,
    DecoderSpec,
    ErrorDecoderInput,
    ObservableDecoderCompiler,
    PcmOrDem,
)

# The error rate that decoders assume when none is given, which qldpc.codes also uses to build
# detector error models for code-capacity estimates.
PLACEHOLDER_ERROR_RATE = _PLACEHOLDER_ERROR_RATE

####################################################################################################
# decoder input resolution


def build_error_decoder(pcm_or_dem: PcmOrDem, *, decoder: ErrorDecoderInput = None) -> ErrorDecoder:
    """Build or retrieve a decoder that maps a syndrome to an inferred error.

    Args:
        pcm_or_dem: The parity-check matrix or detector error model to decode.
        decoder: A decoder specification such as ``decoders.bp_osd(...)``, a constructor or
            ``decoders.from_matrix(...)`` factory that builds an error decoder, a prebuilt error
            decoder, or None to select GUF for a nonbinary FieldArray and BP+OSD otherwise.

    Returns:
        An ErrorDecoder.
    """
    return _as_error_decoder(*_build_decoder(pcm_or_dem, decoder))


def build_observable_decoder(
    dem: stim.DetectorErrorModel, *, decoder: DecoderInput = None
) -> ObservableDecoder:
    """Build or retrieve a decoder that maps a syndrome to predicted observable flips.

    Args:
        dem: The detector error model to decode.
        decoder: A decoder specification such as ``decoders.mwpm(...)``, which builds a native
            observable decoder when supported; a ``decoders.from_dem(...)`` factory; an object with
            a compile_decoder_for_dem method, such as a ``decoders.SinterDecoder``; a constructor
            that builds an error or observable decoder from dem; a prebuilt error or observable
            decoder; or None to select BP+OSD.

    Returns:
        An ObservableDecoder.  An error decoder is wrapped so that the observable flips of the
        errors that it infers become its predictions.
    """
    if isinstance(decoder, DecoderSpec):
        return decoder.build_observable_decoder(dem)
    if isinstance(decoder, DEMDecoderFactory):
        return validate_observable_decoder(decoder.build(dem), "A from_dem factory")
    if compiles_for_dem(decoder):
        return _compile_observable_decoder(decoder, dem)
    built_decoder, source = _build_decoder(dem, decoder)
    if isinstance(built_decoder, ObservableDecoder):
        return built_decoder
    return ErrorsToObservablesDecoder(_as_error_decoder(built_decoder, source), dem)


def reject_prebuilt_decoder(decoder: object, reason: str) -> None:
    """Raise if a decoder input is prebuilt and cannot be rebuilt for a new matrix."""
    if is_prebuilt_decoder(decoder):
        raise ValueError(
            f"A prebuilt decoder cannot be passed as decoder= here because {reason}.  Pass a"
            " decoder specification such as decoder=decoders.bp_osd(...), or a decoder"
            " constructor, instead"
        )


def _build_decoder(pcm_or_dem: PcmOrDem, decoder: DecoderInput) -> tuple[object, str]:
    """Build or retrieve a decoder without constraining its output kind."""
    built_decoder: object
    if decoder is None:
        is_nonbinary = isinstance(pcm_or_dem, galois.FieldArray) and type(pcm_or_dem).order != 2
        default_spec = guf() if is_nonbinary else bp_osd()
        built_decoder, source = default_spec.build(pcm_or_dem), "The default decoder"
    elif isinstance(decoder, DecoderSpec):
        built_decoder = decoder.build(pcm_or_dem)
        source = "A decoder specification"
    elif isinstance(decoder, MatrixDecoderFactory):
        if isinstance(pcm_or_dem, stim.DetectorErrorModel):
            raise ValueError(
                "A from_matrix factory needs a parity-check matrix; this workflow provides a"
                " detector error model.  Pass a decoder factory that accepts a DEM directly, or"
                " use a decoder specification instead"
            )
        built_decoder, source = decoder.build(pcm_or_dem), "A from_matrix factory"
    elif isinstance(decoder, DEMDecoderFactory):
        raise TypeError(
            "A from_dem factory predicts observables, but this workflow needs inferred errors"
        )
    elif is_prebuilt_decoder(decoder):
        built_decoder, source = decoder, "A prebuilt decoder"
    elif callable(decoder):
        built_decoder, source = decoder(pcm_or_dem), "A decoder constructor"
    else:
        raise TypeError(
            "decoder must be a decoder specification such as decoders.bp_osd(...), a decoder"
            " constructor, a prebuilt error decoder, or None"
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
        return BitPackedObservableDecoder(compiled_decoder, dem.num_observables)
    raise TypeError(
        "A decoder compiled by compile_decoder_for_dem must provide a decode_observables or"
        " decode_shots_bit_packed method"
    )
