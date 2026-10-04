# SPDX-License-Identifier: Apache-2.0

"""Decoder specifications for decoders provided by the ldpc package.

qLDPC imports this integration module while initializing its public decoder API.  Importing ``ldpc``
and PyMatching eagerly here adds roughly 0.18 seconds (about 25 percent) to ``import qldpc`` in
fresh-process development benchmarks.  The protocol-compatible subclasses are therefore created on
first use in the private lazy-backend section at the bottom of this module.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import stim

from qldpc._util import format_docstring, get_external_caller_stacklevel
from qldpc.math import IntegerArray

from ..common import (
    PLACEHOLDER_ERROR_RATE,
    _erasure_bit_support,
    _get_matrix_error_channel,
)
from ..construction.legacy_options import _deprecate_error_rate_option
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


# Decoder specifications


@_erasure_bit_support("BP_OSD", supported=False)
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
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
    error_rate: float | None = None,
) -> ErrorDecoder:
    """Build a belief-propagation with ordered-statistics (BP+OSD) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.
        error_channel: One probability for every matrix-column error, or one probability per column.
            Defaults to {PLACEHOLDER_ERROR_RATE}. A detector error model supplies its own
            probabilities, so neither probability argument can be specified with one.
        max_iter: Maximum number of belief-propagation iterations.
        bp_method: Belief-propagation method.
        ms_scaling_factor: Scaling factor for minimum-sum belief propagation.
        schedule: Belief-propagation update schedule.
        omp_thread_count: Number of OpenMP threads.
        random_schedule_seed: Seed for a randomized serial schedule.
        serial_schedule_order: Explicit update order for a serial schedule.
        osd_method: Ordered-statistics decoding method.
        osd_order: Ordered-statistics decoding order.
        backend_options: Additional options for ``ldpc.BpOsdDecoder`` that are not listed
            above.  The backend rejects unsupported names when the decoder is built.
        error_rate: Deprecated i.i.d. matrix error probability; use ``error_channel`` instead.

    Returns:
        An ``ldpc.BpOsdDecoder`` subclass that is also an
        :class:`~qldpc.decoders.protocols.ErrorDecoder`.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.

    See ``help(ldpc.BpOsdDecoder)``, the
    `ldpc decoder documentation <https://software.roffe.eu/ldpc/quantum_decoder.html>`_, and
    `arXiv:2005.07016 <https://arxiv.org/abs/2005.07016>`_.
    """
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
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
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
    error_rate: float | None = None,
) -> ErrorDecoder:
    """Build a belief-propagation with localized-statistics (BP+LSD) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.
        error_channel: One probability for every matrix-column error, or one probability per column.
            Defaults to {PLACEHOLDER_ERROR_RATE}. A detector error model supplies its own
            probabilities, so neither probability argument can be specified with one.
        max_iter: Maximum number of belief-propagation iterations.
        bp_method: Belief-propagation method.
        ms_scaling_factor: Scaling factor for minimum-sum belief propagation.
        schedule: Belief-propagation update schedule.
        omp_thread_count: Number of OpenMP threads.
        random_schedule_seed: Seed for a randomized serial schedule.
        random_serial_schedule: Whether to randomize the serial schedule.
        serial_schedule_order: Explicit update order for a serial schedule.
        bits_per_step: Number of bits added to each localized-statistics cluster step.
        lsd_method: Localized-statistics decoding method.
        lsd_order: Localized-statistics decoding order.
        always_run_lsd: Whether to run LSD after belief propagation converges.
        backend_options: Additional options for ``ldpc.BpLsdDecoder`` that are not listed
            above.  ``ldpc.BpLsdDecoder`` silently ignores names that it does not recognize, so
            names other than ``input_vector_type``, ``channel_probs``, and the aliases
            ``osd_method`` and ``osd_order`` emit a warning when the decoder is built, but are
            still forwarded.
        error_rate: Deprecated i.i.d. matrix error probability; use ``error_channel`` instead.

    Returns:
        An ``ldpc.bplsd_decoder.BpLsdDecoder`` subclass that is also an
        :class:`~qldpc.decoders.protocols.ErrorDecoder`.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.

    See ``help(ldpc.bplsd_decoder.BpLsdDecoder)``, the
    `ldpc decoder documentation <https://software.roffe.eu/ldpc/quantum_decoder.html>`_, and
    `arXiv:2406.18655 <https://arxiv.org/abs/2406.18655>`_.
    """
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
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
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
    error_rate: float | None = None,
) -> ErrorDecoder:
    """Build a belief-find (BF) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.
        error_channel: One probability for every matrix-column error, or one probability per column.
            Defaults to {PLACEHOLDER_ERROR_RATE}. A detector error model supplies its own
            probabilities, so neither probability argument can be specified with one.
        max_iter: Maximum number of belief-propagation iterations.
        bp_method: Belief-propagation method.
        ms_scaling_factor: Scaling factor for minimum-sum belief propagation.
        schedule: Belief-propagation update schedule.
        omp_thread_count: Number of OpenMP threads.
        random_schedule_seed: Seed for a randomized serial schedule.
        serial_schedule_order: Explicit update order for a serial schedule.
        uf_method: Union-find cluster-solving method.
        bits_per_step: Number of bits added to each cluster step.
        backend_options: Additional options for ``ldpc.BeliefFindDecoder`` that are not listed
            above.  The backend rejects unsupported names when the decoder is built.
        error_rate: Deprecated i.i.d. matrix error probability; use ``error_channel`` instead.

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


_BP_OSD_SPEC_DOC = """Configure belief propagation with ordered-statistics decoding.

Args:
    error_channel: Matrix-column error probabilities, as one value or a vector.  A DEM supplies
        its own probabilities and does not accept an override.
    max_iter: Maximum number of belief-propagation iterations.
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

This decoder cannot signal erasure.  See ``help(ldpc.BpOsdDecoder)`` and
`arXiv:2005.07016 <https://arxiv.org/abs/2005.07016>`_.
"""

bp_osd = decoder_spec(
    "bp_osd",
    _get_decoder_bp_osd,
    option_transform=_deprecate_error_rate_option,
    doc=_BP_OSD_SPEC_DOC,
)

_BP_LSD_SPEC_DOC = """Configure belief propagation with localized-statistics decoding.

Args:
    error_channel: Matrix-column error probabilities, as one value or a vector.  A DEM supplies
        its own probabilities and does not accept an override.
    max_iter: Maximum number of belief-propagation iterations.
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
        unknown names; qLDPC warns about names it does not recognize.

Returns:
    A decoder specification.  ``build(pcm_or_dem)`` returns an
    ``ldpc.bplsd_decoder.BpLsdDecoder`` subclass implementing
    :class:`~qldpc.decoders.protocols.ErrorDecoder`.

This decoder cannot signal erasure.  See ``help(ldpc.bplsd_decoder.BpLsdDecoder)`` and
`arXiv:2406.18655 <https://arxiv.org/abs/2406.18655>`_.
"""

bp_lsd = decoder_spec(
    "bp_lsd",
    _get_decoder_bp_lsd,
    option_transform=_deprecate_error_rate_option,
    doc=_BP_LSD_SPEC_DOC,
)

_BF_SPEC_DOC = """Configure belief-find decoding.

Args:
    error_channel: Matrix-column error probabilities, as one value or a vector.  A DEM supplies
        its own probabilities and does not accept an override.
    max_iter: Maximum number of belief-propagation iterations.
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

This decoder cannot signal erasure.  See ``help(ldpc.BeliefFindDecoder)`` and
`arXiv:1709.06218 <https://arxiv.org/abs/1709.06218>`_.
"""

bf = decoder_spec(
    "bf",
    _get_decoder_bf,
    option_transform=_deprecate_error_rate_option,
    doc=_BF_SPEC_DOC,
)


# Private input helpers

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
    error_rate: float | None,
) -> tuple[IntegerArray, list[float]]:
    """Convert backend input to the matrix and probabilities expected by ldpc."""
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


# Lazy backend classes
#
# These classes cannot be declared at module scope without importing ldpc during every qldpc import.
# Create them together on first use, then cache them as ordinary module globals so introspection,
# isinstance checks, copying, and pickling behave like normal module-level classes.

_BACKEND_CLASS_NAMES = frozenset({"BeliefFindDecoder", "BpLsdDecoder", "BpOsdDecoder"})
_BACKEND_CLASSES: dict[str, type[Any]] | None = None


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
    if _BACKEND_CLASSES is None:
        import ldpc
        import ldpc.bplsd_decoder

        class BpOsdDecoder(_DetailedLdpcDecoderMixin, ldpc.BpOsdDecoder, ErrorDecoder):
            """An ldpc.BpOsdDecoder with qLDPC error-decoding protocols."""

        class BpLsdDecoder(
            _DetailedLdpcDecoderMixin, ldpc.bplsd_decoder.BpLsdDecoder, ErrorDecoder
        ):
            """An ldpc.bplsd_decoder.BpLsdDecoder with qLDPC error-decoding protocols."""

        class BeliefFindDecoder(_DetailedLdpcDecoderMixin, ldpc.BeliefFindDecoder, ErrorDecoder):
            """An ldpc.BeliefFindDecoder with qLDPC error-decoding protocols."""

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
