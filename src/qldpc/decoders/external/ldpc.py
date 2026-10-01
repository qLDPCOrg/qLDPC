# SPDX-License-Identifier: Apache-2.0

"""Builders for decoders provided by the ldpc package."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import stim

from qldpc._util import format_docstring
from qldpc.math import IntegerArray

from ..common import PLACEHOLDER_ERROR_RATE, _erasure_bit_support
from ..dems import DetectorErrorModelArrays
from ..protocols import ErrorDecoder

_PcmOrDem: TypeAlias = IntegerArray | stim.DetectorErrorModel


# Public builders


@_erasure_bit_support("BP_OSD", supported=False)
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
def get_decoder_bp_osd(
    pcm_or_dem: _PcmOrDem,
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: object,
) -> ErrorDecoder:
    """Build a belief-propagation with ordered-statistics (BP+OSD) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.
        error_rate: The i.i.d. probability of each error in a matrix.  Ignored for a DEM.
            Default: {PLACEHOLDER_ERROR_RATE}.
        error_channel: The probability of each error mechanism.  For a matrix, defaults to
            ``[error_rate] * num_errors``.  For a DEM, defaults to its error probabilities.  An
            explicit value overrides either default.
        **decoder_args: Additional keyword arguments passed to ``ldpc.BpOsdDecoder``.

    Returns:
        An ``ldpc.BpOsdDecoder`` subclass that is also an
        :class:`~qldpc.decoders.protocols.ErrorDecoder`.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.

    See ``help(ldpc.BpOsdDecoder)``, the
    `ldpc decoder documentation <https://software.roffe.eu/ldpc/quantum_decoder.html>`_, and
    `arXiv:2005.07016 <https://arxiv.org/abs/2005.07016>`_.
    """
    from ..adapters.backends import BpOsdDecoder

    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return BpOsdDecoder(pcm, error_channel=error_channel, **decoder_args)


@_erasure_bit_support("BP_LSD", supported=False)
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
def get_decoder_bp_lsd(
    pcm_or_dem: _PcmOrDem,
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: object,
) -> ErrorDecoder:
    """Build a belief-propagation with localized-statistics (BP+LSD) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.
        error_rate: The i.i.d. probability of each error in a matrix.  Ignored for a DEM.
            Default: {PLACEHOLDER_ERROR_RATE}.
        error_channel: The probability of each error mechanism.  For a matrix, defaults to
            ``[error_rate] * num_errors``.  For a DEM, defaults to its error probabilities.  An
            explicit value overrides either default.
        **decoder_args: Additional keyword arguments passed to
            ``ldpc.bplsd_decoder.BpLsdDecoder``.

    Returns:
        An ``ldpc.bplsd_decoder.BpLsdDecoder`` subclass that is also an
        :class:`~qldpc.decoders.protocols.ErrorDecoder`.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.

    See ``help(ldpc.bplsd_decoder.BpLsdDecoder)``, the
    `ldpc decoder documentation <https://software.roffe.eu/ldpc/quantum_decoder.html>`_, and
    `arXiv:2406.18655 <https://arxiv.org/abs/2406.18655>`_.
    """
    from ..adapters.backends import BpLsdDecoder

    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return BpLsdDecoder(pcm, error_channel=error_channel, **decoder_args)


@_erasure_bit_support("BF", supported=False)
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
def get_decoder_bf(
    pcm_or_dem: _PcmOrDem,
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: object,
) -> ErrorDecoder:
    """Build a belief-find (BF) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.
        error_rate: The i.i.d. probability of each error in a matrix.  Ignored for a DEM.
            Default: {PLACEHOLDER_ERROR_RATE}.
        error_channel: The probability of each error mechanism.  For a matrix, defaults to
            ``[error_rate] * num_errors``.  For a DEM, defaults to its error probabilities.  An
            explicit value overrides either default.
        **decoder_args: Additional keyword arguments passed to ``ldpc.BeliefFindDecoder``.

    Returns:
        An ``ldpc.BeliefFindDecoder`` subclass that is also an
        :class:`~qldpc.decoders.protocols.ErrorDecoder`.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.

    See ``help(ldpc.BeliefFindDecoder)``, the
    `ldpc decoder documentation <https://software.roffe.eu/ldpc/quantum_decoder.html>`_,
    `arXiv:1709.06218 <https://arxiv.org/abs/1709.06218>`_,
    `arXiv:2103.08049 <https://arxiv.org/abs/2103.08049>`_, and
    `arXiv:2209.01180 <https://arxiv.org/abs/2209.01180>`_.
    """
    from ..adapters.backends import BeliefFindDecoder

    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return BeliefFindDecoder(pcm, error_channel=error_channel, **decoder_args)


# Private input helpers


def _to_ldpc_inputs(
    pcm_or_dem: _PcmOrDem,
    error_rate: float,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None,
) -> tuple[IntegerArray, list[float]]:
    """Convert backend input to the matrix and probabilities expected by ldpc."""
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        dem_arrays = DetectorErrorModelArrays(pcm_or_dem)
        pcm = dem_arrays.detector_flip_matrix
        error_channel = dem_arrays.error_probs if error_channel is None else error_channel
    else:
        pcm = pcm_or_dem
        error_channel = [error_rate] * pcm.shape[1] if error_channel is None else error_channel
    if pcm.dtype.kind in "biu":
        pcm = pcm.astype(np.uint8, copy=False)
    return pcm, list(error_channel)
