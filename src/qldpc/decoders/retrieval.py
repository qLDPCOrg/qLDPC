# SPDX-License-Identifier: Apache-2.0

"""Methods to decode, or retrieve various decoders."""

from __future__ import annotations

import dataclasses
import functools
import warnings
from collections.abc import Callable, Collection, Sequence
from typing import Generic, Literal, ParamSpec, Protocol, TypeAlias, TypeVar, cast

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc._util import format_docstring
from qldpc.math import IntegerArray

from .common import _get_external_caller_stacklevel
from .custom import (
    PLACEHOLDER_ERROR_RATE,
    BatchErrorDecoder,
    ErrorDecoder,
    GUFDecoder,
    ILPDecoder,
    RelayBPDecoder,
)
from .dems import DetectorErrorModelArrays
from .lookup import LookupDecoder

_Parameters = ParamSpec("_Parameters")
_Decoder = TypeVar("_Decoder", bound=ErrorDecoder)
_DecoderT_co = TypeVar("_DecoderT_co", bound=ErrorDecoder, covariant=True)

PcmOrDem: TypeAlias = IntegerArray | stim.DetectorErrorModel


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class DecoderSpec(Generic[_DecoderT_co]):
    """Deferred, typed construction settings for an error decoder."""

    _builder: Callable[..., _DecoderT_co]
    _options: tuple[tuple[str, object], ...]

    def build(self, pcm_or_dem: PcmOrDem) -> _DecoderT_co:
        """Build an error decoder for a parity-check matrix or detector error model."""
        return self._builder(pcm_or_dem, **dict(self._options))


class ErrorDecoderConstructor(Protocol):
    """Callable that builds an error decoder from a matrix or detector error model."""

    def __call__(self, pcm_or_dem: PcmOrDem, /) -> ErrorDecoder:
        """Build an error decoder."""


ErrorDecoderInput: TypeAlias = (
    DecoderSpec[ErrorDecoder] | ErrorDecoder | ErrorDecoderConstructor | None
)


def _decoder_spec(builder: Callable[..., _Decoder], **options: object) -> DecoderSpec[_Decoder]:
    """Store deferred decoder construction options."""
    return DecoderSpec(builder, tuple(options.items()))


def decode(
    pcm_or_dem: PcmOrDem,
    syndrome: npt.NDArray[np.int_],
    *,
    decoder: ErrorDecoderInput = None,
    **decoder_args: object,
) -> npt.NDArray[np.int_]:
    """Construct a decoder and decode a syndrome.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model.
        syndrome: The syndrome to decode.
        decoder: Deferred decoder settings, a prebuilt error decoder, a custom constructor, or None
            to select the default decoder.
        **decoder_args: Deprecated decoder-selection and construction arguments.

    Returns:
        The inferred error.
    """
    error_decoder = _resolve_decoder(pcm_or_dem, decoder, decoder_args)
    return error_decoder.decode(syndrome)


def get_decoder(
    pcm_or_dem: PcmOrDem,
    *,
    decoder: ErrorDecoderInput = None,
    **decoder_args: object,
) -> ErrorDecoder:
    """Build or retrieve an error decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model.
        decoder: Deferred decoder settings, a prebuilt error decoder, a custom constructor, or None
            to select the default decoder.
        **decoder_args: Deprecated decoder-selection and construction arguments.

    Returns:
        An error decoder configured for pcm_or_dem.

    If decoder is None, this method defaults to generalized union-find (GUF) for non-binary parity
    check matrices, and BP+OSD otherwise.

    The legacy ``with_<NAME>``, ``decoder_constructor``, ``static_decoder``, and free-form decoder
    arguments remain available during a deprecation period.
    """
    return _resolve_decoder(pcm_or_dem, decoder, decoder_args)


def _resolve_decoder(
    pcm_or_dem: PcmOrDem,
    decoder: ErrorDecoderInput,
    decoder_args: dict[str, object],
    *,
    warn_deprecated: bool = True,
) -> ErrorDecoder:
    """Resolve an error decoder, optionally suppressing a warning emitted by an outer API."""
    if decoder is not None:
        if decoder_args:
            raise ValueError(
                "Cannot combine decoder= with deprecated decoder-selection or construction arguments"
            )
        if getattr(decoder, "decodes_observables", False):
            raise TypeError("decoder must predict errors rather than observables")
        if isinstance(decoder, DecoderSpec):
            return _validate_error_decoder(decoder.build(pcm_or_dem), "A decoder spec")
        if isinstance(decoder, type):
            return _validate_error_decoder(decoder(pcm_or_dem), "A decoder constructor")
        if hasattr(decoder, "decode"):
            return _validate_error_decoder(decoder, "A static decoder")
        if callable(decoder):
            return _validate_error_decoder(decoder(pcm_or_dem), "A decoder constructor")
        raise TypeError(
            "decoder must be a DecoderSpec, an error decoder, a decoder constructor, or None"
        )

    if decoder_args:
        if warn_deprecated:
            warnings.warn(
                _get_legacy_decoder_migration_message(pcm_or_dem, decoder_args),
                DeprecationWarning,
                stacklevel=_get_external_caller_stacklevel(),
            )
        return _get_legacy_decoder(pcm_or_dem, decoder_args)

    if isinstance(pcm_or_dem, galois.FieldArray) and type(pcm_or_dem).order != 2:
        return get_decoder_guf(pcm_or_dem)
    return get_decoder_bp_osd(pcm_or_dem)


def _get_legacy_decoder_migration_message(
    pcm_or_dem: PcmOrDem | None, decoder_args: dict[str, object]
) -> str:
    """Describe the typed replacement for one legacy decoder request."""
    if decoder_args.get("decoder_constructor") is not None:
        return (
            "The decoder_constructor keyword is deprecated; pass the constructor as decoder="
            " instead, for example decoder=MyDecoder"
        )
    if decoder_args.get("static_decoder") is not None:
        return (
            "The static_decoder keyword is deprecated; pass the decoder instance as decoder="
            " instead"
        )
    if decoder_args.get("predict_observable_flips"):
        return (
            "predict_observable_flips=True is deprecated; construct an ObservableLookupDecoder"
            " directly and call decode_observables(...) instead"
        )

    helper_names = {
        "BF": "bf",
        "BP_LSD": "bp_lsd",
        "BP_OSD": "bp_osd",
        "GUF": "guf",
        "ILP": "ilp",
        "MWPM": "mwpm",
        "RBP": "relay_bp",
        "lookup": "lookup_table",
    }
    selected = [name for name in DECODER_CONSTRUCTORS if decoder_args.get(f"with_{name}", False)]
    if len(selected) == 1:
        old_name = f"with_{selected[0]}"
        helper_name = helper_names[selected[0]]
        return (
            f"The {old_name} keyword and free-form decoder options are deprecated; use"
            f" decoder=decoders.{helper_name}(...) instead"
        )
    if len(selected) > 1:
        return (
            "The with_<NAME> decoder-selection keywords are deprecated; pass exactly one typed"
            " decoder specification such as decoder=decoders.bp_osd(...) instead"
        )

    helper_name = (
        "guf"
        if isinstance(pcm_or_dem, galois.FieldArray) and type(pcm_or_dem).order != 2
        else "bp_osd"
    )
    return (
        "Passing decoder options directly to get_decoder is deprecated; move them into"
        f" decoder=decoders.{helper_name}(...) instead"
    )


def _validate_error_decoder(decoder: object, source: str) -> ErrorDecoder:
    """Validate and type-narrow an object expected to decode syndromes to errors."""
    if getattr(decoder, "decodes_observables", False):
        raise TypeError(f"{source} predicts observables rather than errors")
    if not hasattr(decoder, "decode") or not callable(decoder.decode):
        raise TypeError(f"{source} must provide a callable decode method")
    return cast(ErrorDecoder, decoder)


def _get_legacy_decoder(pcm_or_dem: PcmOrDem, decoder_args: dict[str, object]) -> ErrorDecoder:
    """Support the deprecated keyword-based decoder API."""
    # optionally inject a decoder constructor
    if (decoder_constructor := decoder_args.pop("decoder_constructor", None)) is not None:
        if not callable(decoder_constructor):
            raise TypeError("The decoder_constructor argument must be callable")
        return _validate_error_decoder(
            decoder_constructor(pcm_or_dem, **decoder_args), "A decoder constructor"
        )

    # optionally inject a static decoder, ignoring all other arguments
    if (static_decoder := decoder_args.pop("static_decoder", None)) is not None:
        if decoder_args:
            raise ValueError("If passed a static decoder, we cannot process decoding arguments")
        return _validate_error_decoder(static_decoder, "A static decoder")

    # look for and construct a recognized decoder, consuming every request
    decoder_names = [
        name for name in DECODER_CONSTRUCTORS if decoder_args.pop(f"with_{name}", False)
    ]
    if len(decoder_names) > 1:
        raise ValueError(
            "Only one decoder can be requested at a time, but received requests for: "
            + ", ".join(decoder_names)
        )
    if decoder_names:
        return DECODER_CONSTRUCTORS[decoder_names[0]](pcm_or_dem, **decoder_args)

    # use GUF by default for codes over non-binary fields
    if isinstance(pcm_or_dem, galois.FieldArray) and type(pcm_or_dem).order != 2:
        return DECODER_CONSTRUCTORS["GUF"](pcm_or_dem, **decoder_args)

    # use BP+OSD by default otherwise
    return DECODER_CONSTRUCTORS["BP_OSD"](pcm_or_dem, **decoder_args)


_DECODER_DISPLAY_NAMES = {
    "bf": "BF",
    "bp_lsd": "BP_LSD",
    "bp_osd": "BP_OSD",
    "guf": "GUF",
    "ilp": "ILP",
    "mwpm": "MWPM",
    "rbp": "RBP",
}


def _erasure_bit_support(
    supported: bool,
) -> Callable[[Callable[_Parameters, _Decoder]], Callable[_Parameters, _Decoder]]:
    """Declare and enforce whether a decoder getter supports an erasure bit."""

    def decorator(
        decoder_getter: Callable[_Parameters, _Decoder],
    ) -> Callable[_Parameters, _Decoder]:
        canonical_name = decoder_getter.__name__.removeprefix("get_decoder_")
        decoder_name = _DECODER_DISPLAY_NAMES.get(canonical_name, canonical_name)
        message = (
            f"The {decoder_name} decoder cannot signal erasure, so it does not accept the"
            " add_erasure_bit argument"
        )

        @functools.wraps(decoder_getter)
        def checked_getter(*args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Decoder:
            add_erasure_bit = bool(kwargs.get("add_erasure_bit"))
            if not supported:
                if add_erasure_bit:
                    raise ValueError(message)
                kwargs.pop("add_erasure_bit", None)

            decoder = decoder_getter(*args, **kwargs)
            if add_erasure_bit and not getattr(decoder, "has_erasure_bit", False):
                raise ValueError(message)
            return decoder

        return checked_getter

    return decorator


@_erasure_bit_support(False)
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
def get_decoder_bp_osd(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: object,
) -> ErrorDecoder:
    """Decoder based on belief propagation with ordered statistics (BP+OSD).

    Args:
        pcm_or_dem: A parity check matrix or detector error model (DEM) to decode.
        error_rate: The i.i.d. probability of each error in pcm_or_dem.  This argument is ignored if
            pcm_or_dem is a DEM.  Default: {PLACEHOLDER_ERROR_RATE}.
        error_channel: A vector declaring the probability of each error mechanism in pcm_or_dem.
            If pcm_or_dem is a matrix, the error_channel defaults to ``[error_rate] * num_errors``.
            If pcm_or_dem is a DEM, its error probabilities are used as the default error_channel.
            If an explicit error_channel is provided, it overrides all defaults.
        **decoder_args: Additional keyword arguments passed to ldpc.BpOsdDecoder.

    Returns:
        A decoder constructed by the ldpc package.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.

    For details about the BD-OSD decoder and its arguments, see:

    - help(ldpc.BpOsdDecoder)
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2005.07016
    """
    import ldpc

    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return ldpc.BpOsdDecoder(pcm, error_channel=error_channel, **decoder_args)


@_erasure_bit_support(False)
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
def get_decoder_bp_lsd(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: object,
) -> ErrorDecoder:
    """Decoder based on belief propagation with localized statistics (BP+LSD).

    Args:
        pcm_or_dem: A parity check matrix or detector error model (DEM) to decode.
        error_rate: The i.i.d. probability of each error in pcm_or_dem.  This argument is ignored if
            pcm_or_dem is a DEM.  Default: {PLACEHOLDER_ERROR_RATE}.
        error_channel: A vector declaring the probability of each error mechanism in pcm_or_dem.
            If pcm_or_dem is a matrix, the error_channel defaults to ``[error_rate] * num_errors``.
            If pcm_or_dem is a DEM, its error probabilities are used as the default error_channel.
            If an explicit error_channel is provided, it overrides all defaults.
        **decoder_args: Additional keyword arguments passed to ldpc.bplsd_decoder.BpLsdDecoder.

    Returns:
        A decoder constructed by the ldpc package.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.

    For details about the BD-LSD decoder and its arguments, see:

    - help(ldpc.bplsd_decoder.BpLsdDecoder)
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2406.18655
    """
    import ldpc

    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return ldpc.bplsd_decoder.BpLsdDecoder(pcm, error_channel=error_channel, **decoder_args)


@_erasure_bit_support(False)
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
def get_decoder_bf(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: object,
) -> ErrorDecoder:
    """Decoder based on belief finding (BF).

    Args:
        pcm_or_dem: A parity check matrix or detector error model (DEM) to decode.
        error_rate: The i.i.d. probability of each error in pcm_or_dem.  This argument is ignored if
            pcm_or_dem is a DEM.  Default: {PLACEHOLDER_ERROR_RATE}.
        error_channel: A vector declaring the probability of each error mechanism in pcm_or_dem.
            If pcm_or_dem is a matrix, the error_channel defaults to ``[error_rate] * num_errors``.
            If pcm_or_dem is a DEM, its error probabilities are used as the default error_channel.
            If an explicit error_channel is provided, it overrides all defaults.
        **decoder_args: Additional keyword arguments passed to ldpc.BeliefFindDecoder.

    Returns:
        A decoder constructed by the ldpc package.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.

    For details about the BF decoder and its arguments, see:

    - help(ldpc.BeliefFindDecoder)
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - References:
      - https://arxiv.org/abs/1709.06218
      - https://arxiv.org/abs/2103.08049
      - https://arxiv.org/abs/2209.01180
    """
    import ldpc

    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return ldpc.BeliefFindDecoder(pcm, error_channel=error_channel, **decoder_args)


def _to_ldpc_inputs(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    error_rate: float,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None,
) -> tuple[IntegerArray, list[float]]:
    """Post-process the arguments to ldpc decoders."""
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        dem_arrays = DetectorErrorModelArrays(pcm_or_dem)
        pcm = dem_arrays.detector_flip_matrix
        error_channel = dem_arrays.error_probs if error_channel is None else error_channel
    else:
        pcm = pcm_or_dem
        error_channel = [error_rate] * pcm.shape[1] if error_channel is None else error_channel
    return pcm, list(error_channel)


@_erasure_bit_support(False)
def get_decoder_mwpm(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    decompose_errors: bool = False,
    ignore_non_graphlike_errors: bool = False,
    **decoder_args: object,
) -> BatchErrorDecoder:
    """Decoder based on minimum weight perfect matching (MWPM).

    Args:
        pcm_or_dem: A parity check matrix or detector error model (DEM) to decode.
        decompose_errors: Whether to apply suggested decompositions of error mechanisms.
        ignore_non_graphlike_errors: Whether to ignore errors that trigger > 2 detectors (after
            decomposition, if applicable).
        **decoder_args: Additional keyword arguments passed to ldpc.BeliefFindDecoder.

    Returns:
        A decoder constructed by pymatching.Matching.from_check_matrix.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.

    All other keyword arguments are passed to pymatching.Matching.from_check_matrix.

    A point of potential confusion: even if passed a detector error model, we DO NOT USE the
    pymatching.Matching.from_check_matrix method here because this returns a decoder that maps a
    syndrome to observable flips, whereas we want a decoder that maps a syndrome to an error.
    If you want a decoder that maps syndromes to observable flips, see qldpc.decoders.sinter.
    """
    # identify parity check matrix and error probabilities
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        dem_arrays = DetectorErrorModelArrays(pcm_or_dem, decompose_errors=decompose_errors)
        pcm = dem_arrays.detector_flip_matrix
        if decoder_args.get("weights") is not None:
            raise ValueError("Cannot set error weights when initializing a MWPM decoder from a DEM")
        decoder_args["weights"] = np.log((1 - dem_arrays.error_probs) / dem_arrays.error_probs)
    else:
        pcm = pcm_or_dem

    # possibly ignore non-graphlike errors, counting the detectors that each error addresses
    detectors_per_error = np.asarray((pcm != 0).sum(axis=0)).ravel()
    error_is_not_graphlike = detectors_per_error > 2
    if ignore_non_graphlike_errors:
        if np.any(error_is_not_graphlike):
            mask = np.ones(pcm.shape[1])
            mask[error_is_not_graphlike] = 0
            pcm = pcm @ scipy.sparse.diags(mask)
    elif np.any(error_is_not_graphlike):
        column = int(np.argmax(error_is_not_graphlike))
        raise ValueError(
            "The provided parity check matrix or detector error model contains a non-graphlike"
            f" error: column {column} of the parity check matrix addresses"
            f" {detectors_per_error[column]} detectors, which may occur (for example) due to the"
            " presence of a Pauli-Y error that flips both X and Z detectors.  Try decomposing"
            " non-graphlike errors by passing 'decompose_errors=True' to the decoder, which splits"
            " errors along the decompositions that the detector error model suggests; stim provides"
            " those suggestions for a circuit via"
            " circuit.detector_error_model(decompose_errors=True).  If that does not work either,"
            " you can try 'ignore_non_graphlike_errors=True'"
        )

    # retrieve a matching decoder from pymatching
    import pymatching

    return pymatching.Matching.from_check_matrix(pcm, **decoder_args)


@_erasure_bit_support(True)
def get_decoder_rbp(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    error_priors: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: object,
) -> RelayBPDecoder:
    """Relay-BP decoders.

    For details about Relay-BP decoders, see:

    - Documentation: https://pypi.org/project/relay-bp
    - Reference: https://arxiv.org/abs/2506.01779
    """
    return RelayBPDecoder(pcm_or_dem, error_priors, **decoder_args)  # type:ignore[arg-type]


@_erasure_bit_support(True)
def get_decoder_lookup(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> LookupDecoder:
    """Decoder based on a lookup table that maps errors to syndromes."""
    return LookupDecoder(pcm_or_dem, **decoder_args)  # type:ignore[arg-type]


@_erasure_bit_support(True)
def get_decoder_ilp(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    add_erasure_bit: bool = False,
    **decoder_args: object,
) -> ILPDecoder:
    """Decoder based on solving an integer linear program (ILP)."""
    return ILPDecoder(_to_pcm(pcm_or_dem), add_erasure_bit=add_erasure_bit, **decoder_args)


@_erasure_bit_support(True)
def get_decoder_guf(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> GUFDecoder:
    """Decoder based on a generalization of Union-Find, described in arXiv:2103.08049."""
    return GUFDecoder(_to_pcm(pcm_or_dem), **decoder_args)  # type:ignore[arg-type]


def _to_pcm(pcm_or_dem: IntegerArray | stim.DetectorErrorModel) -> IntegerArray:
    """Convert the input to a parity check matrix.

    The consumers of this method build dense decoders, so a detector error model is densified here.
    """
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        return DetectorErrorModelArrays(pcm_or_dem).detector_flip_matrix.toarray()
    return pcm_or_dem


def bp_osd(
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    max_iter: int = 0,
    bp_method: Literal["product_sum", "minimum_sum", "ps", "ms"] = "minimum_sum",
    ms_scaling_factor: float = 1.0,
    schedule: Literal["parallel", "serial"] = "parallel",
    omp_thread_count: int = 1,
    random_schedule_seed: int = 0,
    serial_schedule_order: Sequence[int] | None = None,
    osd_method: str | float = 0,
    osd_order: int = 0,
    **decoder_args: object,
) -> DecoderSpec[ErrorDecoder]:
    """Configure belief-propagation with ordered-statistics decoding.

    Args:
        error_rate: The i.i.d. error probability used when no channel is provided.
        error_channel: Per-error probabilities, overriding error_rate.
        max_iter: Maximum belief-propagation iterations.
        bp_method: Belief-propagation update method.
        ms_scaling_factor: Minimum-sum scaling factor.
        schedule: Parallel or serial update schedule.
        omp_thread_count: Number of OpenMP threads.
        random_schedule_seed: Seed for a randomized serial schedule.
        serial_schedule_order: Explicit serial update order.
        osd_method: Ordered-statistics decoding method.
        osd_order: Ordered-statistics decoding order.
        **decoder_args: Additional options accepted by ldpc.BpOsdDecoder.
    """
    return _decoder_spec(
        get_decoder_bp_osd,
        error_rate=error_rate,
        error_channel=error_channel,
        max_iter=max_iter,
        bp_method=bp_method,
        ms_scaling_factor=ms_scaling_factor,
        schedule=schedule,
        omp_thread_count=omp_thread_count,
        random_schedule_seed=random_schedule_seed,
        serial_schedule_order=serial_schedule_order,
        osd_method=osd_method,
        osd_order=osd_order,
        **decoder_args,
    )


def bp_lsd(
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    max_iter: int = 0,
    bp_method: Literal["product_sum", "minimum_sum", "ps", "ms"] = "minimum_sum",
    ms_scaling_factor: float = 1.0,
    schedule: Literal["parallel", "serial"] = "parallel",
    omp_thread_count: int = 1,
    random_schedule_seed: int = 0,
    serial_schedule_order: Sequence[int] | None = None,
    bits_per_step: int = 1,
    lsd_order: int = 0,
    lsd_method: str | int = 0,
    always_run_lsd: bool = False,
    **decoder_args: object,
) -> DecoderSpec[ErrorDecoder]:
    """Configure belief-propagation with localized-statistics decoding.

    Args:
        error_rate: The i.i.d. error probability used when no channel is provided.
        error_channel: Per-error probabilities, overriding error_rate.
        max_iter: Maximum belief-propagation iterations.
        bp_method: Belief-propagation update method.
        ms_scaling_factor: Minimum-sum scaling factor.
        schedule: Parallel or serial update schedule.
        omp_thread_count: Number of OpenMP threads.
        random_schedule_seed: Seed for a randomized serial schedule.
        serial_schedule_order: Explicit serial update order.
        bits_per_step: Bits added to each LSD cluster per growth step.
        lsd_order: Localized-statistics decoding order.
        lsd_method: Localized-statistics decoding method.
        always_run_lsd: Whether to run LSD even when belief propagation converges.
        **decoder_args: Additional options accepted by ldpc.bplsd_decoder.BpLsdDecoder.
    """
    return _decoder_spec(
        get_decoder_bp_lsd,
        error_rate=error_rate,
        error_channel=error_channel,
        max_iter=max_iter,
        bp_method=bp_method,
        ms_scaling_factor=ms_scaling_factor,
        schedule=schedule,
        omp_thread_count=omp_thread_count,
        random_schedule_seed=random_schedule_seed,
        serial_schedule_order=serial_schedule_order,
        bits_per_step=bits_per_step,
        lsd_order=lsd_order,
        lsd_method=lsd_method,
        always_run_lsd=always_run_lsd,
        **decoder_args,
    )


def bf(
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    max_iter: int = 0,
    bp_method: Literal["product_sum", "minimum_sum", "ps", "ms"] = "minimum_sum",
    ms_scaling_factor: float = 1.0,
    schedule: Literal["parallel", "serial"] = "parallel",
    omp_thread_count: int = 1,
    random_schedule_seed: int = 0,
    serial_schedule_order: Sequence[int] | None = None,
    uf_method: Literal["inversion", "peeling"] = "peeling",
    bits_per_step: int = 0,
) -> DecoderSpec[ErrorDecoder]:
    """Configure belief-find decoding.

    Args:
        error_rate: The i.i.d. error probability used when no channel is provided.
        error_channel: Per-error probabilities, overriding error_rate.
        max_iter: Maximum belief-propagation iterations.
        bp_method: Belief-propagation update method.
        ms_scaling_factor: Minimum-sum scaling factor.
        schedule: Parallel or serial update schedule.
        omp_thread_count: Number of OpenMP threads.
        random_schedule_seed: Seed for a randomized serial schedule.
        serial_schedule_order: Explicit serial update order.
        uf_method: Union-find local decoding method.
        bits_per_step: Bits added to each union-find cluster per growth step.
    """
    return _decoder_spec(
        get_decoder_bf,
        error_rate=error_rate,
        error_channel=error_channel,
        max_iter=max_iter,
        bp_method=bp_method,
        ms_scaling_factor=ms_scaling_factor,
        schedule=schedule,
        omp_thread_count=omp_thread_count,
        random_schedule_seed=random_schedule_seed,
        serial_schedule_order=serial_schedule_order,
        uf_method=uf_method,
        bits_per_step=bits_per_step,
    )


def mwpm(
    *,
    decompose_errors: bool = False,
    ignore_non_graphlike_errors: bool = False,
    weights: float | npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: object,
) -> DecoderSpec[BatchErrorDecoder]:
    """Configure minimum-weight perfect-matching decoding.

    Args:
        decompose_errors: Whether to apply DEM-suggested error decompositions.
        ignore_non_graphlike_errors: Whether to drop errors that address more than two detectors.
        weights: Scalar or per-error matching weights. A DEM supplies these automatically.
        **decoder_args: Additional options accepted by pymatching.Matching.from_check_matrix.
    """
    return _decoder_spec(
        get_decoder_mwpm,
        decompose_errors=decompose_errors,
        ignore_non_graphlike_errors=ignore_non_graphlike_errors,
        weights=weights,
        **decoder_args,
    )


def relay_bp(
    *,
    error_priors: npt.NDArray[np.floating] | Sequence[float] | None = None,
    name: str = "RelayDecoderF32",
    observable_error_matrix: IntegerArray | None = None,
    include_decode_result: bool = False,
    add_erasure_bit: bool = False,
    **decoder_args: object,
) -> DecoderSpec[RelayBPDecoder]:
    """Configure Relay-BP decoding.

    Args:
        error_priors: Prior probability of each error mechanism.
        name: Relay-BP decoder class name.
        observable_error_matrix: Matrix mapping errors to observable flips.
        include_decode_result: Whether Relay-BP retains detailed decode results.
        add_erasure_bit: Whether to append an erasure flag.
        **decoder_args: Additional options accepted by the selected Relay-BP decoder.
    """
    return _decoder_spec(
        get_decoder_rbp,
        error_priors=error_priors,
        name=name,
        observable_error_matrix=observable_error_matrix,
        include_decode_result=include_decode_result,
        add_erasure_bit=add_erasure_bit,
        **decoder_args,
    )


def lookup_table(
    max_weight: int,
    *,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None = None,
    observable_flip_matrix: IntegerArray | None = None,
    post_select: Collection[int] = (),
    add_erasure_bit: bool | None = None,
    confidence_ratio: float | None = None,
    symplectic: bool = False,
) -> DecoderSpec[LookupDecoder]:
    """Configure lookup-table error decoding.

    Args:
        max_weight: Maximum enumerated error weight.
        error_channel: Per-error probabilities.
        penalty_func: Function assigning a penalty to an error.
        observable_flip_matrix: Matrix mapping errors to observable flips for probability grouping.
        post_select: Syndrome indices required to be trivial.
        add_erasure_bit: Whether to append an erasure flag.
        confidence_ratio: Required likelihood ratio before making a prediction.
        symplectic: Whether errors use symplectic `[X|Z]` form.
    """
    return _decoder_spec(
        get_decoder_lookup,
        max_weight=max_weight,
        error_channel=error_channel,
        penalty_func=penalty_func,
        observable_flip_matrix=observable_flip_matrix,
        post_select=post_select,
        add_erasure_bit=add_erasure_bit,
        confidence_ratio=confidence_ratio,
        symplectic=symplectic,
    )


def ilp(
    *,
    add_erasure_bit: bool = False,
    **solver_args: object,
) -> DecoderSpec[ILPDecoder]:
    """Configure integer-linear-program error decoding.

    Args:
        add_erasure_bit: Whether to append an erasure flag.
        **solver_args: Options passed to cvxpy.Problem.solve.
    """
    return _decoder_spec(get_decoder_ilp, add_erasure_bit=add_erasure_bit, **solver_args)


def guf(
    *,
    max_weight: int | None = None,
    symplectic: bool = False,
    add_erasure_bit: bool = False,
) -> DecoderSpec[GUFDecoder]:
    """Configure generalized union-find error decoding.

    Args:
        max_weight: Weight at which to stop the exhaustive local search.
        symplectic: Whether errors use symplectic `[X|Z]` form.
        add_erasure_bit: Whether to append an erasure flag.
    """
    return _decoder_spec(
        get_decoder_guf,
        max_weight=max_weight,
        symplectic=symplectic,
        add_erasure_bit=add_erasure_bit,
    )


DECODER_CONSTRUCTORS: dict[str, Callable[..., ErrorDecoder]] = {
    "BF": get_decoder_bf,
    "BP_LSD": get_decoder_bp_lsd,
    "BP_OSD": get_decoder_bp_osd,
    "GUF": get_decoder_guf,
    "ILP": get_decoder_ilp,
    "MWPM": get_decoder_mwpm,
    "RBP": get_decoder_rbp,
    "lookup": get_decoder_lookup,
}


# Upper-case compatibility aliases.
get_decoder_BF = get_decoder_bf
get_decoder_BP_LSD = get_decoder_bp_lsd
get_decoder_BP_OSD = get_decoder_bp_osd
get_decoder_GUF = get_decoder_guf
get_decoder_ILP = get_decoder_ilp
get_decoder_MWPM = get_decoder_mwpm
get_decoder_RBP = get_decoder_rbp
