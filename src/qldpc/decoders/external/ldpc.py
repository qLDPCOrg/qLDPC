# SPDX-License-Identifier: Apache-2.0

"""Decoder specifications for decoders provided by the ldpc package.

qLDPC imports this integration module while initializing its public decoder API.  Importing ``ldpc``
and PyMatching eagerly here adds roughly 0.18 seconds (about 25 percent) to ``import qldpc`` in
fresh-process development benchmarks.  The protocol-compatible subclasses are therefore created on
first use in the private lazy-backend section at the bottom of this module.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import stim

from qldpc._deprecations.decoders import deprecate_error_rate_option
from qldpc._utils import get_external_caller_stacklevel
from qldpc.math import IntegerArray

from ..common import _PLACEHOLDER_ERROR_RATE, _erasure_bit_support, _get_matrix_error_channel
from ..construction.specs import decoder_spec
from ..dems import DetectorErrorModelArrays
from ..protocols import ErrorDecoder, ErrorDecodeResult

_PcmOrDem: TypeAlias = IntegerArray | stim.DetectorErrorModel

if TYPE_CHECKING:
    import ldpc
    import ldpc.bplsd_decoder

    class BpOsdDecoder(ldpc.BpOsdDecoder, ErrorDecoder): ...

    class BpLsdDecoder(ldpc.bplsd_decoder.BpLsdDecoder, ErrorDecoder): ...

    class BeliefFindDecoder(ldpc.BeliefFindDecoder, ErrorDecoder): ...


####################################################################################################
# decoder specifications


@_erasure_bit_support("BP_OSD", supported=False)
def _get_decoder_bp_osd(
    pcm_or_dem: _PcmOrDem,
    *,
    error_channel: float | npt.NDArray[np.floating] | Sequence[float] | None = None,
    max_iter: int = 0,
    bp_method: Literal["product_sum", "minimum_sum", "ps", "ms"] = "product_sum",
    ms_scaling_factor: float = 1.0,
    schedule: Literal["parallel", "serial"] = "parallel",
    omp_thread_count: int = 1,
    random_schedule_seed: int = 0,
    serial_schedule_order: Sequence[int] | None = None,
    osd_method: Literal["OSD_0", "OSD_E", "OSD_CS"] = "OSD_0",
    osd_order: int = 0,
    backend_options: Mapping[str, object] | None = None,
    error_rate: float | None = None,  # DEPRECATED: see qldpc._deprecations
) -> ErrorDecoder:
    """Build a BP+OSD decoder; see ``decoders.bp_osd`` for its options."""
    # DEPRECATED: see qldpc._deprecations
    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_channel, error_rate)
    return _build_ldpc_decoder(
        "BpOsdDecoder",
        pcm,
        error_channel,
        {
            "max_iter": max_iter,
            "bp_method": bp_method,
            "ms_scaling_factor": ms_scaling_factor,
            "schedule": schedule,
            "omp_thread_count": omp_thread_count,
            "random_schedule_seed": random_schedule_seed,
            "serial_schedule_order": serial_schedule_order,
            "osd_method": osd_method,
            "osd_order": osd_order,
            **(backend_options or {}),
        },
    )


@_erasure_bit_support("BP_LSD", supported=False)
def _get_decoder_bp_lsd(
    pcm_or_dem: _PcmOrDem,
    *,
    error_channel: float | npt.NDArray[np.floating] | Sequence[float] | None = None,
    max_iter: int = 0,
    bp_method: Literal["product_sum", "minimum_sum", "ps", "ms"] = "product_sum",
    ms_scaling_factor: float = 1.0,
    schedule: Literal["parallel", "serial"] = "parallel",
    omp_thread_count: int = 1,
    random_schedule_seed: int = 0,
    random_serial_schedule: bool = False,
    serial_schedule_order: Sequence[int] | None = None,
    bits_per_step: int = 1,
    lsd_method: Literal["LSD_0", "LSD_E", "LSD_CS"] = "LSD_0",
    lsd_order: int = 0,
    always_run_lsd: bool = False,
    backend_options: Mapping[str, object] | None = None,
    error_rate: float | None = None,  # DEPRECATED: see qldpc._deprecations
) -> ErrorDecoder:
    """Build a BP+LSD decoder; see ``decoders.bp_lsd`` for its options."""
    # DEPRECATED: see qldpc._deprecations
    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_channel, error_rate)
    _warn_unknown_bp_lsd_options(backend_options or {})
    return _build_ldpc_decoder(
        "BpLsdDecoder",
        pcm,
        error_channel,
        {
            "max_iter": max_iter,
            "bp_method": bp_method,
            "ms_scaling_factor": ms_scaling_factor,
            "schedule": schedule,
            "omp_thread_count": omp_thread_count,
            "random_schedule_seed": random_schedule_seed,
            "random_serial_schedule": random_serial_schedule,
            "serial_schedule_order": serial_schedule_order,
            "bits_per_step": bits_per_step,
            "lsd_method": lsd_method,
            "lsd_order": lsd_order,
            "always_run_lsd": always_run_lsd,
            **(backend_options or {}),
        },
    )


@_erasure_bit_support("BF", supported=False)
def _get_decoder_bf(
    pcm_or_dem: _PcmOrDem,
    *,
    error_channel: float | npt.NDArray[np.floating] | Sequence[float] | None = None,
    max_iter: int = 0,
    bp_method: Literal["product_sum", "minimum_sum", "ps", "ms"] = "product_sum",
    ms_scaling_factor: float = 1.0,
    schedule: Literal["parallel", "serial"] = "parallel",
    omp_thread_count: int = 1,
    random_schedule_seed: int = 0,
    serial_schedule_order: Sequence[int] | None = None,
    uf_method: Literal["inversion", "peeling"] = "peeling",
    bits_per_step: int = 0,
    backend_options: Mapping[str, object] | None = None,
    error_rate: float | None = None,  # DEPRECATED: see qldpc._deprecations
) -> ErrorDecoder:
    """Build a belief-find decoder; see ``decoders.bf`` for its options."""
    # DEPRECATED: see qldpc._deprecations
    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_channel, error_rate)
    return _build_ldpc_decoder(
        "BeliefFindDecoder",
        pcm,
        error_channel,
        {
            "max_iter": max_iter,
            "bp_method": bp_method,
            "ms_scaling_factor": ms_scaling_factor,
            "schedule": schedule,
            "omp_thread_count": omp_thread_count,
            "random_schedule_seed": random_schedule_seed,
            "serial_schedule_order": serial_schedule_order,
            "uf_method": uf_method,
            "bits_per_step": bits_per_step,
            **(backend_options or {}),
        },
    )


bp_osd = decoder_spec(
    "bp_osd",
    _get_decoder_bp_osd,
    option_transform=deprecate_error_rate_option,
    doc=f"""Configure belief propagation with ordered-statistics decoding.

See the `ldpc BP+OSD documentation
<https://software.roffe.eu/ldpc/ldpc/bposd_decoder.html>`_ for a discussion of its options, as
well as ``help(ldpc.BpOsdDecoder)`` and `arXiv:2005.07016 <https://arxiv.org/abs/2005.07016>`_.

Args:
    error_channel: Matrix-column error probabilities, as one value or a vector.  Defaults to
        {_PLACEHOLDER_ERROR_RATE}.  A DEM supplies its own probabilities and does not accept
        an override.
    max_iter: Maximum number of belief-propagation iterations, or 0 for the number of columns of
        the parity-check matrix (error mechanisms of a DEM).
    bp_method: Belief-propagation method.
    ms_scaling_factor: Scaling factor for minimum-sum belief propagation.
    schedule: Parallel or serial update schedule.
    omp_thread_count: Number of OpenMP threads.
    random_schedule_seed: Seed for a randomized serial schedule.
    serial_schedule_order: Explicit order for serial updates.
    osd_method: Ordered-statistics decoding method.
    osd_order: Ordered-statistics decoding order.
    backend_options: Additional options for ``ldpc.BpOsdDecoder``.  The backend rejects
        unsupported names when the decoder is built.

Returns:
    A decoder specification.  ``build(pcm_or_dem)`` returns an ``ldpc.BpOsdDecoder`` subclass
    that implements :class:`~qldpc.decoders.protocols.ErrorDecoder`.
""",
)

bp_lsd = decoder_spec(
    "bp_lsd",
    _get_decoder_bp_lsd,
    option_transform=deprecate_error_rate_option,
    doc=f"""Configure belief propagation with localized-statistics decoding.

See the `ldpc BP+LSD documentation
<https://software.roffe.eu/ldpc/ldpc/bplsd_decoder.html>`_ for a discussion of its options, as
well as ``help(ldpc.bplsd_decoder.BpLsdDecoder)`` and
`arXiv:2406.18655 <https://arxiv.org/abs/2406.18655>`_.

Args:
    error_channel: Matrix-column error probabilities, as one value or a vector.  Defaults to
        {_PLACEHOLDER_ERROR_RATE}.  A DEM supplies its own probabilities and does not accept
        an override.
    max_iter: Maximum number of belief-propagation iterations, or 0 for the number of columns of
        the parity-check matrix (error mechanisms of a DEM).
    bp_method: Belief-propagation method.
    ms_scaling_factor: Scaling factor for minimum-sum belief propagation.
    schedule: Parallel or serial update schedule.
    omp_thread_count: Number of OpenMP threads.
    random_schedule_seed: Seed for a randomized serial schedule.
    random_serial_schedule: Whether to randomize the serial update order.
    serial_schedule_order: Explicit order for serial updates.
    bits_per_step: Number of bits added to each localized-statistics cluster step.
    lsd_method: Localized-statistics decoding method.
    lsd_order: Localized-statistics decoding order.
    always_run_lsd: Whether to run LSD even after belief propagation converges.
    backend_options: Additional ``ldpc.BpLsdDecoder`` options.  The backend silently ignores
        unknown names; qLDPC warns about names it does not recognize, but still forwards them.

Returns:
    A decoder specification.  ``build(pcm_or_dem)`` returns an
    ``ldpc.bplsd_decoder.BpLsdDecoder`` subclass implementing
    :class:`~qldpc.decoders.protocols.ErrorDecoder`.
""",
)

bf = decoder_spec(
    "bf",
    _get_decoder_bf,
    option_transform=deprecate_error_rate_option,
    doc=f"""Configure belief-find decoding.

See the `ldpc belief-find documentation
<https://software.roffe.eu/ldpc/ldpc/belief_find_decoder.html>`_ for a discussion of its options,
as well as ``help(ldpc.BeliefFindDecoder)``,
`arXiv:1709.06218 <https://arxiv.org/abs/1709.06218>`_,
`arXiv:2103.08049 <https://arxiv.org/abs/2103.08049>`_, and
`arXiv:2209.01180 <https://arxiv.org/abs/2209.01180>`_.

Args:
    error_channel: Matrix-column error probabilities, as one value or a vector.  Defaults to
        {_PLACEHOLDER_ERROR_RATE}.  A DEM supplies its own probabilities and does not accept
        an override.
    max_iter: Maximum number of belief-propagation iterations, or 0 for the number of columns of
        the parity-check matrix (error mechanisms of a DEM).
    bp_method: Belief-propagation method.
    ms_scaling_factor: Scaling factor for minimum-sum belief propagation.
    schedule: Parallel or serial update schedule.
    omp_thread_count: Number of OpenMP threads.
    random_schedule_seed: Seed for a randomized serial schedule.
    serial_schedule_order: Explicit order for serial updates.
    uf_method: Union-find cluster-solving method.
    bits_per_step: Number of bits added to each cluster step.
    backend_options: Additional options for ``ldpc.BeliefFindDecoder``.  The backend rejects
        unsupported names when the decoder is built.

Returns:
    A decoder specification.  ``build(pcm_or_dem)`` returns an ``ldpc.BeliefFindDecoder``
    subclass implementing :class:`~qldpc.decoders.protocols.ErrorDecoder`.
""",
)


# Options that ldpc.BpLsdDecoder reads, other than those listed by _get_decoder_bp_lsd, including
# osd_method and osd_order, which it accepts as aliases of lsd_method and lsd_order.  Unlike the
# other ldpc decoders, BpLsdDecoder silently ignores unrecognized options.  This list cannot be read
# from its compiled signature, so names outside it are flagged with a warning rather than rejected.
_BP_LSD_UNLISTED_BACKEND_OPTIONS = frozenset(
    {"input_vector_type", "channel_probs", "osd_method", "osd_order"}
)


def _warn_unknown_bp_lsd_options(backend_options: Mapping[str, object]) -> None:
    """Warn about backend options that ldpc.BpLsdDecoder may silently ignore."""
    if unknown := sorted(set(backend_options) - _BP_LSD_UNLISTED_BACKEND_OPTIONS):
        warnings.warn(
            f"ldpc.BpLsdDecoder silently ignores options that it does not recognize, and"
            f" {unknown} are not BP+LSD options known to qLDPC; check their spelling",
            UserWarning,
            stacklevel=get_external_caller_stacklevel(),
        )


def _to_ldpc_inputs(
    pcm_or_dem: _PcmOrDem,
    error_channel: float | npt.NDArray[np.floating] | Sequence[float] | None,
    error_rate: float | None,  # DEPRECATED: see qldpc._deprecations
) -> tuple[IntegerArray, list[float]]:
    """Convert backend input to the matrix and probabilities expected by ldpc."""
    # DEPRECATED: see qldpc._deprecations
    matrix_error_channel = _get_matrix_error_channel(pcm_or_dem, error_channel, error_rate)
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        dem_arrays = DetectorErrorModelArrays(pcm_or_dem)
        pcm = dem_arrays.detector_flip_matrix
        normalized_error_channel = dem_arrays.error_probs
    else:
        pcm = pcm_or_dem
        normalized_error_channel = cast(npt.NDArray[np.floating], matrix_error_channel)
    if pcm.dtype.kind in "biu":
        pcm = pcm.astype(np.uint8, copy=False)
    return pcm, list(normalized_error_channel)


####################################################################################################
# lazy backend classes
#
# These classes cannot be declared at module scope without importing ldpc during every qldpc import.
# Create them together on first use, then cache them as ordinary module globals so introspection,
# isinstance checks, copying, and pickling behave like normal module-level classes.  They are
# created under a lock so that concurrent first uses share one set of classes, which pickling needs.

_BACKEND_CLASS_NAMES = frozenset({"BeliefFindDecoder", "BpLsdDecoder", "BpOsdDecoder"})
_BACKEND_CLASSES: dict[str, type[Any]] | None = None
_BACKEND_CLASSES_LOCK = threading.Lock()


class _DetailedLdpcDecoderMixin(ErrorDecoder):
    """Snapshot stable BP diagnostics immediately after an ldpc decode."""

    def decode_errors_detailed(self, syndrome: npt.NDArray[np.int_]) -> ErrorDecodeResult:
        error = np.asarray(self.decode_errors(syndrome))
        diagnostics: dict[str, object] = {}
        for name in ("converge", "iterations", "log_prob_ratios"):
            if hasattr(self, name):
                value = getattr(self, name)
                if isinstance(value, np.ndarray):
                    value = value.copy()
                elif isinstance(value, np.generic):
                    value = value.item()
                diagnostics[f"ldpc.{name}"] = value
        return ErrorDecodeResult(error, diagnostics=diagnostics)


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
    if _BACKEND_CLASSES is not None:
        return _BACKEND_CLASSES[name]
    with _BACKEND_CLASSES_LOCK:
        if _BACKEND_CLASSES is None:
            import ldpc
            import ldpc.bplsd_decoder

            class BpOsdDecoder(_DetailedLdpcDecoderMixin, ldpc.BpOsdDecoder, ErrorDecoder):
                """An ldpc.BpOsdDecoder with qLDPC error-decoding protocols."""

            class BpLsdDecoder(
                _DetailedLdpcDecoderMixin, ldpc.bplsd_decoder.BpLsdDecoder, ErrorDecoder
            ):
                """An ldpc.bplsd_decoder.BpLsdDecoder with qLDPC error-decoding protocols."""

            class BeliefFindDecoder(
                _DetailedLdpcDecoderMixin, ldpc.BeliefFindDecoder, ErrorDecoder
            ):
                """An ldpc.BeliefFindDecoder with qLDPC error-decoding protocols."""

            classes = (BpOsdDecoder, BpLsdDecoder, BeliefFindDecoder)
            for decoder_type in classes:
                decoder_type.__module__ = __name__
                decoder_type.__qualname__ = decoder_type.__name__
                globals()[decoder_type.__name__] = decoder_type
            _BACKEND_CLASSES = {decoder_type.__name__: decoder_type for decoder_type in classes}
        return _BACKEND_CLASSES[name]


if not TYPE_CHECKING:
    # Static type checkers see the backend classes declared above, rather than an Any fallback.

    def __getattr__(name: str) -> Any:
        """Load a protocol-compatible ldpc subclass only when requested."""
        if name not in _BACKEND_CLASS_NAMES:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        return _get_backend_class(name)
