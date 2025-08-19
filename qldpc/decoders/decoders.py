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

import galois
import ldpc
import numpy as np
import numpy.typing as npt
import pymatching

from .custom import BatchDecoder, Decoder, GUFDecoder, ILPDecoder, LookupDecoder, RelayBPDecoder

PLACEHOLDER_ERROR_RATE = 1e-3  # required for some decoding methods


def decode(
    matrix: npt.NDArray[np.int_], syndrome: npt.NDArray[np.int_], **decoder_args: object
) -> npt.NDArray[np.int_]:
    """Find a `vector` that solves `matrix @ vector == syndrome mod 2`."""
    decoder = get_decoder(matrix, **decoder_args)
    return decoder.decode(syndrome)


def get_decoder(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Retrieve a decoder."""
    if constructor := decoder_args.pop("decoder_constructor", None):
        assert callable(constructor)
        return constructor(matrix, **decoder_args)

    if decoder := decoder_args.pop("static_decoder", None):
        assert hasattr(decoder, "decode") and callable(getattr(decoder, "decode"))
        assert not decoder_args, "if passed a static decoder, we cannot process decoding arguments"
        return decoder

    if decoder_args.pop("with_BP_LSD", False):
        return get_decoder_BP_LSD(matrix, **decoder_args)

    if decoder_args.pop("with_BF", False):
        return get_decoder_BF(matrix, **decoder_args)

    if name := decoder_args.pop("with_RBP", None):
        return get_decoder_RBP(name, matrix, **decoder_args)

    if decoder_args.pop("with_MWPM", False):
        return get_decoder_MWPM(matrix, **decoder_args)

    if decoder_args.pop("with_lookup", False):
        return get_decoder_lookup(matrix, **decoder_args)

    if decoder_args.pop("with_ILP", False):
        return get_decoder_ILP(matrix, **decoder_args)

    # use GUF if requested, or by default for non-binary fields
    with_GUF = decoder_args.pop("with_GUF", False) or (
        isinstance(matrix, galois.FieldArray) and type(matrix).order != 2
    )
    if with_GUF:
        return get_decoder_GUF(matrix, **decoder_args)

    # use BP+OSD by default
    decoder_args.pop("with_BP_OSD", None)
    return get_decoder_BP_OSD(matrix, **decoder_args)


def get_decoder_BP_OSD(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Decoder based on belief propagation with ordered statistics (BP+OSD).

    For details about the BD-OSD decoder and its arguments, see:
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2005.07016
    """
    if "error_channel" not in decoder_args and "error_rate" not in decoder_args:
        decoder_args["error_rate"] = PLACEHOLDER_ERROR_RATE
    return ldpc.BpOsdDecoder(matrix, **decoder_args)


def get_decoder_BP_LSD(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Decoder based on belief propagation with localized statistics (BP+LSD).

    For details about the BD-LSD decoder and its arguments, see:
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2406.18655
    """
    if "error_channel" not in decoder_args and "error_rate" not in decoder_args:
        decoder_args["error_rate"] = PLACEHOLDER_ERROR_RATE
    return ldpc.bplsd_decoder.BpLsdDecoder(matrix, **decoder_args)


def get_decoder_BF(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Decoder based on belief finding (BF).

    For details about the BF decoder and its arguments, see:
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - References:
      - https://arxiv.org/abs/1709.06218
      - https://arxiv.org/abs/2103.08049
      - https://arxiv.org/abs/2209.01180
    """
    if "error_channel" not in decoder_args and "error_rate" not in decoder_args:
        decoder_args["error_rate"] = PLACEHOLDER_ERROR_RATE
    return ldpc.BeliefFindDecoder(matrix, **decoder_args)


def get_decoder_MWPM(matrix: npt.NDArray[np.int_], **decoder_args: object) -> BatchDecoder:
    """Decoder based on minimum weight perfect matching (MWPM)."""
    return pymatching.Matching.from_check_matrix(matrix, **decoder_args)


def get_decoder_RBP(
    name: object, matrix: npt.NDArray[np.int_], **decoder_args: object
) -> RelayBPDecoder:
    """Relay-BP decoders.

    For details about Relay-BP decoders, see:
    - Documentation: https://pypi.org/project/relay-bp
    - Reference: https://arxiv.org/abs/2506.01779
    """
    try:
        import relay_bp
    except ImportError:
        raise ImportError("Failed to import relay-bp.  Try installing 'qldpc[relay-bp]'")
    if not isinstance(name, str) or not hasattr(relay_bp, name):
        raise ValueError(
            f"Relay BP decoder name not recognized: {name}\n"
            "See 'import relay_bp; help(relay_bp.bp)' for available decoders"
        )
    check_matrix = matrix.view(np.ndarray) if isinstance(matrix, galois.FieldArray) else matrix
    error_priors = decoder_args.pop("error_priors", [PLACEHOLDER_ERROR_RATE] * matrix.shape[1])
    decoder = getattr(relay_bp, name)(check_matrix, np.asarray(error_priors), **decoder_args)
    return RelayBPDecoder(decoder)


def get_decoder_lookup(matrix: npt.NDArray[np.int_], **decoder_args: object) -> LookupDecoder:
    """Decoder based on a lookup table from errors to syndromes."""
    return LookupDecoder(matrix, **decoder_args)  # type:ignore[arg-type]


def get_decoder_ILP(matrix: npt.NDArray[np.int_], **decoder_args: object) -> ILPDecoder:
    """Decoder based on solving an integer linear program (ILP).

    All remaining keyword arguments are passed to `cvxpy.Problem.solve`.
    """
    return ILPDecoder(matrix, **decoder_args)


def get_decoder_GUF(matrix: npt.NDArray[np.int_], **decoder_args: object) -> GUFDecoder:
    """Decoder based on a generalization of Union-Find, described in arXiv:2103.08049."""
    return GUFDecoder(matrix, **decoder_args)  # type:ignore[arg-type]
