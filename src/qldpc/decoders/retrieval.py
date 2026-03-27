"""Methods to decode, or retrieve various decoders

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import Any

import galois
import ldpc
import numpy as np
import numpy.typing as npt
import pymatching
import stim

from qldpc.math import IntegerArray

from .custom import (
    PLACEHOLDER_ERROR_RATE,
    BatchDecoder,
    Decoder,
    GUFDecoder,
    ILPDecoder,
    LookupDecoder,
    RelayBPDecoder,
)
from .dems import DetectorErrorModelArrays


def decode(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    syndrome: npt.NDArray[np.int_],
    **decoder_args: object,
) -> npt.NDArray[np.int_]:
    """Find a `vector` that solves `matrix @ vector == syndrome mod 2`."""
    decoder = get_decoder(pcm_or_dem, **decoder_args)
    return decoder.decode(syndrome)


def get_decoder(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> Decoder:
    """Retrieve a decoder."""
    if constructor := decoder_args.pop("decoder_constructor", None):
        assert callable(constructor)
        return constructor(pcm_or_dem, **decoder_args)

    if decoder := decoder_args.pop("static_decoder", None):
        assert hasattr(decoder, "decode") and callable(getattr(decoder, "decode"))
        assert not decoder_args, "if passed a static decoder, we cannot process decoding arguments"
        return decoder

    if decoder_args.pop("with_BP_LSD", False):
        return get_decoder_BP_LSD(pcm_or_dem, **decoder_args)

    if decoder_args.pop("with_BF", False):
        return get_decoder_BF(pcm_or_dem, **decoder_args)

    if name := decoder_args.pop("with_RBP", None):
        return get_decoder_RBP(str(name), pcm_or_dem, **decoder_args)

    if decoder_args.pop("with_MWPM", False):
        return get_decoder_MWPM(pcm_or_dem, **decoder_args)

    if decoder_args.pop("with_lookup", False):
        return get_decoder_lookup(pcm_or_dem, **decoder_args)

    if decoder_args.pop("with_ILP", False):
        return get_decoder_ILP(pcm_or_dem, **decoder_args)

    # use GUF if requested, or by default for non-binary fields
    with_GUF = decoder_args.pop("with_GUF", False) or (
        isinstance(pcm_or_dem, galois.FieldArray) and type(pcm_or_dem).order != 2
    )
    if with_GUF:
        return get_decoder_GUF(pcm_or_dem, **decoder_args)

    # use BP+OSD by default
    decoder_args.pop("with_BP_OSD", None)
    return get_decoder_BP_OSD(pcm_or_dem, **decoder_args)


def get_decoder_BP_OSD(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> Decoder:
    """Decoder based on belief propagation with ordered statistics (BP+OSD).

    For details about the BD-OSD decoder and its arguments, see:
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2005.07016
    """
    pcm, error_channel, error_rate = _to_ldpc_data(pcm_or_dem, decoder_args)
    return ldpc.BpOsdDecoder(
        pcm, error_channel=error_channel, error_rate=error_rate, **decoder_args
    )


def get_decoder_BP_LSD(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> Decoder:
    """Decoder based on belief propagation with localized statistics (BP+LSD).

    For details about the BD-LSD decoder and its arguments, see:
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2406.18655
    """
    pcm, error_channel, error_rate = _to_ldpc_data(pcm_or_dem, decoder_args)
    return ldpc.bplsd_decoder.BpLsdDecoder(
        pcm, error_channel=error_channel, error_rate=error_rate, **decoder_args
    )


def get_decoder_BF(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> Decoder:
    """Decoder based on belief finding (BF).

    For details about the BF decoder and its arguments, see:
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - References:
      - https://arxiv.org/abs/1709.06218
      - https://arxiv.org/abs/2103.08049
      - https://arxiv.org/abs/2209.01180
    """
    pcm, error_channel, error_rate = _to_ldpc_data(pcm_or_dem, decoder_args)
    return ldpc.BeliefFindDecoder(
        pcm, error_channel=error_channel, error_rate=error_rate, **decoder_args
    )


def _to_ldpc_data(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, decoder_args: dict[str, Any]
) -> tuple[IntegerArray, npt.NDArray[np.float64] | None, float | None]:
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        dem_arrays = DetectorErrorModelArrays(pcm_or_dem)
        pcm = dem_arrays.detector_flip_matrix
        error_channel = dem_arrays.error_probs
        error_rate = None
        return pcm, error_channel, error_rate
    else:
        pcm = pcm_or_dem
        error_channel = decoder_args.pop("error_channel", None)
        error_rate = decoder_args.pop("error_rate", None)
        if error_channel is None and error_rate is None:
            error_rate = PLACEHOLDER_ERROR_RATE
    return pcm, error_channel, error_rate


def get_decoder_MWPM(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> BatchDecoder:
    """Decoder based on minimum weight perfect matching (MWPM).

    A point of potential confusion: even if passed a detector error model, we DO NOT USE the
    pymatching.Matching.from_check_matrix method here because this returns a decoder that maps a
    syndrome to observable flips, whereas we want a decoder that maps a syndrome to an error.
    If you want a decoder that maps syndromes to observable flips, see qldpc.decoders.sinter.
    """
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        dem_arrays = DetectorErrorModelArrays(pcm_or_dem)
        return pymatching.Matching.from_check_matrix(
            dem_arrays.detector_flip_matrix,
            error_probabilities=dem_arrays.error_probs,
            **decoder_args,
        )
    return pymatching.Matching.from_check_matrix(pcm_or_dem, **decoder_args)


def get_decoder_RBP(
    name: str, pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> RelayBPDecoder:
    """Relay-BP decoders.

    For details about Relay-BP decoders, see:
    - Documentation: https://pypi.org/project/relay-bp
    - Reference: https://arxiv.org/abs/2506.01779
    """
    error_priors = decoder_args.pop("error_priors", None)
    observable_error_matrix = decoder_args.pop("observable_error_matrix", None)
    include_decode_result = bool(decoder_args.pop("include_decode_result", False))
    if decoder_args:
        raise ValueError(  # pragma: no cover
            f"Unrecognized arguments for a Relay-BP decoder: {list(decoder_args.keys())}"
        )
    return RelayBPDecoder(
        name,
        pcm_or_dem,
        error_priors,  # type:ignore[arg-type]
        observable_error_matrix=observable_error_matrix,
        include_decode_result=include_decode_result,
    )


def get_decoder_lookup(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> LookupDecoder:
    """Decoder based on a lookup table from errors to syndromes."""
    return LookupDecoder(pcm_or_dem, **decoder_args)  # type:ignore[arg-type]


def get_decoder_ILP(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> ILPDecoder:
    """Decoder based on solving an integer linear program (ILP).

    All remaining keyword arguments are passed to `cvxpy.Problem.solve`.
    """
    return ILPDecoder(_to_pcm(pcm_or_dem), **decoder_args)


def get_decoder_GUF(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> GUFDecoder:
    """Decoder based on a generalization of Union-Find, described in arXiv:2103.08049."""
    return GUFDecoder(_to_pcm(pcm_or_dem), **decoder_args)  # type:ignore[arg-type]


def _to_pcm(pcm_or_dem: IntegerArray | stim.DetectorErrorModel) -> IntegerArray:
    """Convert the input to a parity check matrix."""
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        return DetectorErrorModelArrays(pcm_or_dem).detector_flip_matrix
    return pcm_or_dem
