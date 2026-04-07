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

import inspect
import sys
import warnings
from collections.abc import Sequence

import galois
import ldpc
import numpy as np
import numpy.typing as npt
import pymatching
import scipy.sparse
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
    """Construct a decoder and decode a syndrome.  Return the result of decoding."""
    decoder = get_decoder(pcm_or_dem, **decoder_args)
    return decoder.decode(syndrome)


def get_decoder(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> Decoder:
    """Retrieve a decoder.

    This method looks for a keyword "with_<DECODER_NAME>: bool" argument, and returns
    "get_decoder_<DECODER_NAME>(pcm_or_dem, **decoder_args)".

    This method also recognizes the following keyword arguments for injecting a custom decoder:
    - decoder_constructor: return decoder_constructor(pcm_or_dem, **decoder_args).
    - static_decoder: ignore all other arguments and return static_decoder.

    If no decoder is specified, this method defaults to generalized union-find (GUF) for non-binary
    parity check matrices, and BP+OSD otherwise.
    """
    # optionally inject a decoder constructor
    if (decoder_constructor := decoder_args.pop("decoder_constructor", None)) is not None:
        assert callable(decoder_constructor)
        return decoder_constructor(pcm_or_dem, **decoder_args)

    # optionally inject a static decoder, ignoring all other arguments
    if (static_decoder := decoder_args.pop("static_decoder", None)) is not None:
        assert hasattr(static_decoder, "decode") and callable(getattr(static_decoder, "decode"))
        assert not decoder_args, "If passed a static decoder, we cannot process decoding arguments"
        return static_decoder

    # look for and construct a recognized decoder
    for name in DECODER_CONSTRUCTORS.keys():
        if decoder_args.pop(f"with_{name}", False):
            decoder_constructor = getattr(sys.modules[__name__], f"get_decoder_{name}")
            return decoder_constructor(pcm_or_dem, **decoder_args)

    # use GUF by default for codes over non-binary fields
    if isinstance(pcm_or_dem, galois.FieldArray) and type(pcm_or_dem).order != 2:
        return get_decoder_GUF(pcm_or_dem, **decoder_args)

    # use BP+OSD by default otherwise
    decoder_args.pop("with_BP_OSD", None)
    return get_decoder_BP_OSD(pcm_or_dem, **decoder_args)  # type:ignore[arg-type]


def get_decoder_BP_OSD(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.float64] | Sequence[float] | None = None,
    **decoder_args: object,
) -> Decoder:
    f"""Decoder based on belief propagation with ordered statistics (BP+OSD).

    Args:
        pcm_or_dem: A parity check matrix or detector error model (DEM) to decode.
        error_rate: The i.i.d. probability of each error in pcm_or_dem.  This argument is ignored if
            pcm_or_dem is a DEM.  Default: {PLACEHOLDER_ERROR_RATE}.
        error_channel: A vector declaring the probability of each errer mechanism in pcm_or_dem.
            If pcm_or_dem is a matrix, the error_channel defaults to [error_rate] * num_errors.
            If pcm_or_dem is a DEM, its error probabilities are used as the default error_channel.
            If an explicit error_channel is provided, it overrides all defaults.
        **decoder_args: Additional keyword arguments passed to ldpc.BpOsdDecoder.

    Returns:
        A decoder constructed by the ldpc package.

    For details about the BD-OSD decoder and its arguments, see:
    - help(ldpc.BpOsdDecoder)
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2005.07016
    """
    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return ldpc.BpOsdDecoder(pcm, error_channel=error_channel, **decoder_args)


def get_decoder_BP_LSD(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.float64] | Sequence[float] | None = None,
    **decoder_args: object,
) -> Decoder:
    f"""Decoder based on belief propagation with localized statistics (BP+LSD).

    Args:
        pcm_or_dem: A parity check matrix or detector error model (DEM) to decode.
        error_rate: The i.i.d. probability of each error in pcm_or_dem.  This argument is ignored if
            pcm_or_dem is a DEM.  Default: {PLACEHOLDER_ERROR_RATE}.
        error_channel: A vector declaring the probability of each errer mechanism in pcm_or_dem.
            If pcm_or_dem is a matrix, the error_channel defaults to [error_rate] * num_errors.
            If pcm_or_dem is a DEM, its error probabilities are used as the default error_channel.
            If an explicit error_channel is provided, it overrides all defaults.
        **decoder_args: Additional keyword arguments passed to ldpc.bplsd_decoder.BpLsdDecoder.

    Returns:
        A decoder constructed by the ldpc package.

    For details about the BD-LSD decoder and its arguments, see:
    - help(ldpc.bplsd_decoder.BpLsdDecoder)
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2406.18655
    """
    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return ldpc.bplsd_decoder.BpLsdDecoder(pcm, error_channel=error_channel, **decoder_args)


def get_decoder_BF(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.float64] | Sequence[float] | None = None,
    **decoder_args: object,
) -> Decoder:
    f"""Decoder based on belief finding (BF).

    Args:
        pcm_or_dem: A parity check matrix or detector error model (DEM) to decode.
        error_rate: The i.i.d. probability of each error in pcm_or_dem.  This argument is ignored if
            pcm_or_dem is a DEM.  Default: {PLACEHOLDER_ERROR_RATE}.
        error_channel: A vector declaring the probability of each errer mechanism in pcm_or_dem.
            If pcm_or_dem is a matrix, the error_channel defaults to [error_rate] * num_errors.
            If pcm_or_dem is a DEM, its error probabilities are used as the default error_channel.
            If an explicit error_channel is provided, it overrides all defaults.
        **decoder_args: Additional keyword arguments passed to ldpc.BeliefFindDecoder.

    Returns:
        A decoder constructed by the ldpc package.

    For details about the BF decoder and its arguments, see:
    - help(ldpc.BeliefFindDecoder)
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - References:
      - https://arxiv.org/abs/1709.06218
      - https://arxiv.org/abs/2103.08049
      - https://arxiv.org/abs/2209.01180
    """
    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return ldpc.BeliefFindDecoder(pcm, error_channel=error_channel, **decoder_args)


def _to_ldpc_inputs(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    error_rate: float,
    error_channel: npt.NDArray[np.float64] | Sequence[float] | None,
) -> tuple[IntegerArray, npt.NDArray[np.float64] | Sequence[float]]:
    """Post-process the arguments to ldpc decoders."""
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        dem_arrays = DetectorErrorModelArrays(pcm_or_dem)
        pcm = dem_arrays.detector_flip_matrix
        error_channel = dem_arrays.error_probs if error_channel is None else error_channel
    else:
        pcm = pcm_or_dem
        error_channel = [error_rate] * pcm.shape[1] if error_channel is None else error_channel
    return pcm, error_channel


def get_decoder_MWPM(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    ignore_non_graphlike_errors: bool = False,
    **decoder_args: object,
) -> BatchDecoder:
    """Decoder based on minimum weight perfect matching (MWPM).

    Args:
        pcm_or_dem: A parity check matrix or detector error model (DEM) to decode.
        ignore_graphlike_errors: Whether to ignore errors that trigger > 2 detectors.
        **decoder_args: Additional keyword arguments passed to ldpc.BeliefFindDecoder.

    Returns:
        A decoder constructed by pymatching.Matching.from_check_matrix.

    If called with the keyword argument ignore_non_graphlike_errors=True, columns of the parity
    check matrix with more than two ones (which correspond to error mechanisms that trigger more
    than two detectors in a detector error model) are ignored.  Otherwise, such columns cause
    pymatching to throw an error.

    All other keyword arguments are passed to pymatching.Matching.from_check_matrix.

    A point of potential confusion: even if passed a detector error model, we DO NOT USE the
    pymatching.Matching.from_check_matrix method here because this returns a decoder that maps a
    syndrome to observable flips, whereas we want a decoder that maps a syndrome to an error.
    If you want a decoder that maps syndromes to observable flips, see qldpc.decoders.sinter.
    """
    # identify parity check matrix and error probabilities
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        dem_arrays = DetectorErrorModelArrays(pcm_or_dem)
        pcm = dem_arrays.detector_flip_matrix
        if decoder_args.get("error_probabilities") is not None:  # pragma: no cover
            warnings.warn(
                "Explicitly provided error_probabilities will override the error probabilities of"
                " the provided detector error model",
                stacklevel=2,
            )
        else:
            decoder_args["error_probabilities"] = dem_arrays.error_probs
    else:
        pcm = pcm_or_dem

    # possibly ignore non-graphlike errors
    if ignore_non_graphlike_errors:
        detectors_per_error = np.asarray(np.sum(pcm, axis=0)).ravel()
        error_is_not_graphlike = detectors_per_error > 2
        if np.any(error_is_not_graphlike):
            mask = np.ones(pcm.shape[1])
            mask[error_is_not_graphlike] = 0
            pcm = pcm @ scipy.sparse.diags(mask)

    # retrieve a matching decoder from pymatching
    return pymatching.Matching.from_check_matrix(pcm, **decoder_args)


def get_decoder_RBP(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    error_priors: npt.NDArray[np.float64] | Sequence[float] | None = None,
    **decoder_args: object,
) -> RelayBPDecoder:
    """Relay-BP decoders.

    For details about Relay-BP decoders, see:
    - Documentation: https://pypi.org/project/relay-bp
    - Reference: https://arxiv.org/abs/2506.01779
    """
    return RelayBPDecoder(pcm_or_dem, error_priors, **decoder_args)  # type:ignore[arg-type]


def get_decoder_lookup(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> LookupDecoder:
    """Decoder based on a lookup table that maps errors to syndromes."""
    return LookupDecoder(pcm_or_dem, **decoder_args)  # type:ignore[arg-type]


def get_decoder_ILP(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> ILPDecoder:
    """Decoder based on solving an integer linear program (ILP)."""
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


# collect all decoder constructors in this file into a dictionary
DECODER_CONSTRUCTORS = {
    name.removeprefix("get_decoder_"): func
    for name, func in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    if name.startswith("get_decoder_")
}
