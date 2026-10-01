# SPDX-License-Identifier: Apache-2.0

"""Builders for decoders provided by the ldpc package.

qLDPC imports this integration module while initializing its public decoder API.  Importing ``ldpc``
and PyMatching eagerly here adds roughly 0.18 seconds (about 25 percent) to ``import qldpc`` in
fresh-process development benchmarks.  The protocol-compatible subclasses are therefore created on
first use in the private lazy-backend section at the bottom of this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import stim

from qldpc._util import format_docstring
from qldpc.math import IntegerArray

from ..common import PLACEHOLDER_ERROR_RATE, _erasure_bit_support
from ..dems import DetectorErrorModelArrays
from ..protocols import ErrorDecoder

_PcmOrDem: TypeAlias = IntegerArray | stim.DetectorErrorModel

if TYPE_CHECKING:
    import ldpc
    import ldpc.bplsd_decoder

    class BpOsdDecoder(ldpc.BpOsdDecoder, ErrorDecoder): ...

    class BpLsdDecoder(ldpc.bplsd_decoder.BpLsdDecoder, ErrorDecoder): ...

    class BeliefFindDecoder(ldpc.BeliefFindDecoder, ErrorDecoder): ...


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
    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return _build_ldpc_decoder("BpOsdDecoder", pcm, error_channel, decoder_args)


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
    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return _build_ldpc_decoder("BpLsdDecoder", pcm, error_channel, decoder_args)


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
    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return _build_ldpc_decoder("BeliefFindDecoder", pcm, error_channel, decoder_args)


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


# Lazy backend classes
#
# These classes cannot be declared at module scope without importing ldpc during every qldpc import.
# Create them together on first use, then cache them as ordinary module globals so introspection,
# isinstance checks, copying, and pickling behave like normal module-level classes.

_BACKEND_CLASS_NAMES = frozenset({"BeliefFindDecoder", "BpLsdDecoder", "BpOsdDecoder"})
_BACKEND_CLASSES: dict[str, type[Any]] | None = None


def _build_ldpc_decoder(
    name: str,
    pcm: IntegerArray,
    error_channel: list[float],
    decoder_args: dict[str, object],
) -> ErrorDecoder:
    """Build one of the lazily declared protocol-compatible ldpc decoders."""
    decoder_type = _get_backend_class(name)
    return cast(ErrorDecoder, decoder_type(pcm, error_channel=error_channel, **decoder_args))


def _get_backend_class(name: str) -> type[Any]:
    """Return a protocol-compatible ldpc subclass, creating all three on first use."""
    global _BACKEND_CLASSES
    if _BACKEND_CLASSES is None:
        import ldpc
        import ldpc.bplsd_decoder

        class BpOsdDecoder(ldpc.BpOsdDecoder, ErrorDecoder):
            """An ldpc.BpOsdDecoder that is also an ErrorDecoder."""

        class BpLsdDecoder(ldpc.bplsd_decoder.BpLsdDecoder, ErrorDecoder):
            """An ldpc.bplsd_decoder.BpLsdDecoder that is also an ErrorDecoder."""

        class BeliefFindDecoder(ldpc.BeliefFindDecoder, ErrorDecoder):
            """An ldpc.BeliefFindDecoder that is also an ErrorDecoder."""

        classes = (BpOsdDecoder, BpLsdDecoder, BeliefFindDecoder)
        for decoder_type in classes:
            decoder_type.__module__ = __name__
            decoder_type.__qualname__ = decoder_type.__name__
            globals()[decoder_type.__name__] = decoder_type
        _BACKEND_CLASSES = {decoder_type.__name__: decoder_type for decoder_type in classes}
    return _BACKEND_CLASSES[name]


def __getattr__(name: str) -> Any:
    """Load a protocol-compatible ldpc subclass only when requested."""
    if name not in _BACKEND_CLASS_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _get_backend_class(name)
