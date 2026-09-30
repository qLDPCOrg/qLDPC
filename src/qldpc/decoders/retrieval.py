# SPDX-License-Identifier: Apache-2.0

"""Typed decoder settings and resolution, with legacy retrieval compatibility."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeAlias, TypeVar

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc._util import get_deprecated_alias, get_external_caller_stacklevel
from qldpc.math import IntegerArray

from .builders import (
    get_decoder_bf as _get_decoder_bf,
)
from .builders import (
    get_decoder_bp_lsd as _get_decoder_bp_lsd,
)
from .builders import (
    get_decoder_bp_osd as _get_decoder_bp_osd,
)
from .builders import (
    get_decoder_guf as _get_decoder_guf,
)
from .builders import (
    get_decoder_ilp as _get_decoder_ilp,
)
from .builders import (
    get_decoder_lookup,
    get_error_decoder_mwpm,
    get_min_sum_bp_decoder,
    get_observable_decoder_lookup,
    get_observable_decoder_mwpm,
    get_relay_bp_decoder,
)
from .builders import (
    get_decoder_mwpm as _get_decoder_mwpm,
)
from .builders import (
    get_decoder_rbp as _get_decoder_rbp,
)
from .common import PLACEHOLDER_ERROR_RATE
from .conversion import (
    ErrorsToObservablesDecoder as _ErrorsToObservablesDecoder,
)
from .conversion import (
    ExpandedErrorDecoder as _ExpandedErrorDecoder,
)
from .conversion import (
    match_error_decoder_to_dem as _match_error_decoder_to_dem,
)
from .conversion import (
    validate_observable_decoder as _validate_observable_decoder,
)
from .custom import GUFDecoder, ILPDecoder, RelayBPDecoder
from .lookup import LookupDecoder
from .protocols import (
    BatchErrorDecoder,
    ErrorDecoder,
    ObservableDecoder,
    SupportsDecode,
    as_error_decoder,
)

__all__ = [
    "DECODER_CONSTRUCTORS",
    "BatchDecoder",
    "BatchErrorDecoder",
    "Decoder",
    "DecoderSpec",
    "DeferredErrorDecoderInput",
    "DeferredObservableDecoderInput",
    "ErrorDecoder",
    "ErrorDecoderConstructor",
    "ErrorDecoderInput",
    "ErrorsToObservablesDecoder",
    "ExpandedErrorDecoder",
    "GUFDecoder",
    "ILPDecoder",
    "LookupDecoder",
    "ObservableDecoder",
    "ObservableDecoderConstructor",
    "ObservableDecoderInput",
    "PcmOrDem",
    "RelayBPDecoder",
    "SupportsDecode",
    "bf",
    "bp_lsd",
    "bp_osd",
    "decode",
    "decode_observables",
    "get_decoder",
    "get_decoder_BF",
    "get_decoder_BP_LSD",
    "get_decoder_BP_OSD",
    "get_decoder_GUF",
    "get_decoder_ILP",
    "get_decoder_MWPM",
    "get_decoder_RBP",
    "get_decoder_lookup",
    "get_error_decoder",
    "get_legacy_decoder_migration_message",
    "get_observable_decoder",
    "guf",
    "ilp",
    "is_prebuilt_decoder",
    "lookup_table",
    "match_error_decoder_to_dem",
    "min_sum_bp",
    "mwpm",
    "reject_prebuilt_decoder",
    "reject_removed_decoder_args",
    "relay_bp",
    "resolve_decoder",
    "resolve_observable_decoder",
]

_Decoder = TypeVar("_Decoder", bound=ErrorDecoder)
_DecoderT_co = TypeVar("_DecoderT_co", bound=ErrorDecoder, covariant=True)

PcmOrDem: TypeAlias = IntegerArray | stim.DetectorErrorModel
"""A parity-check matrix or detector error model from which to build an error decoder."""


@dataclasses.dataclass(frozen=True, slots=True, eq=False, repr=False)
class DecoderSpec(Generic[_DecoderT_co]):
    """Deferred, typed construction settings for an error decoder."""

    _helper_name: str
    _builder: Callable[..., _DecoderT_co]
    _options: tuple[tuple[str, object], ...]
    _observable_builder: Callable[..., ObservableDecoder] | None = None

    @property
    def options(self) -> dict[str, object]:
        """Return a copy of the construction options, including defaults."""
        return dict(self._options)

    def build(self, pcm_or_dem: PcmOrDem) -> _DecoderT_co:
        """Build an error decoder for a parity-check matrix or detector error model."""
        return self._builder(pcm_or_dem, **self.options)

    @property
    def predicts_observables_natively(self) -> bool:
        """Whether this specification has a native observable-decoding mode."""
        return self._observable_builder is not None

    def build_observable_decoder(self, dem: stim.DetectorErrorModel) -> ObservableDecoder:
        """Build a decoder that predicts the observable flips of a detector error model."""
        if self._observable_builder is not None:
            return _validate_observable_decoder(
                self._observable_builder(dem, **self.options), "A decoder spec"
            )
        return _ErrorsToObservablesDecoder(self.build(dem), dem)

    def __repr__(self) -> str:
        """Show the helper call that reproduces this specification."""
        helper = globals().get(self._helper_name)
        if helper is None:
            return f"DecoderSpec({self._helper_name!r}, {self._builder!r}, {self._options!r})"
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(helper).parameters.items()
        }
        options = ", ".join(
            f"{name}={value!r}"
            for name, value in self._options
            if not _is_default_value(value, defaults.get(name, inspect.Parameter.empty))
        )
        return f"decoders.{self._helper_name}({options})"


def _is_default_value(value: object, default: object) -> bool:
    """Whether an option is a plain copy of its default value."""
    if value is default:
        return True
    plain_types = (bool, int, float, str)
    return type(value) is type(default) and isinstance(value, plain_types) and value == default


class ErrorDecoderConstructor(Protocol):
    """Callable that builds an error decoder from a matrix or detector error model."""

    def __call__(self, pcm_or_dem: PcmOrDem, /) -> ErrorDecoder | SupportsDecode:
        """Build an error decoder."""


class ObservableDecoderConstructor(Protocol):
    """Callable that builds an observable decoder from a detector error model."""

    def __call__(self, dem: stim.DetectorErrorModel, /) -> ObservableDecoder:
        """Build an observable decoder."""


DeferredErrorDecoderInput: TypeAlias = DecoderSpec[ErrorDecoder] | ErrorDecoderConstructor | None
ErrorDecoderInput: TypeAlias = (
    DecoderSpec[ErrorDecoder] | ErrorDecoderConstructor | ErrorDecoder | SupportsDecode | None
)
DeferredObservableDecoderInput: TypeAlias = (
    DecoderSpec[Any] | ErrorDecoderConstructor | ObservableDecoderConstructor | None
)
ObservableDecoderInput: TypeAlias = (
    DecoderSpec[Any]
    | ErrorDecoderConstructor
    | ObservableDecoderConstructor
    | ErrorDecoder
    | SupportsDecode
    | ObservableDecoder
    | None
)


# Typed decoder-specification helpers


def _decoder_spec(
    helper_name: str,
    builder: Callable[..., _Decoder],
    observable_builder: Callable[..., ObservableDecoder] | None = None,
    /,
    **options: object,
) -> DecoderSpec[_Decoder]:
    """Store deferred decoder construction options."""
    return DecoderSpec(helper_name, builder, tuple(options.items()), observable_builder)


def bp_osd(
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    max_iter: int = 0,
    bp_method: Literal["product_sum", "minimum_sum", "ps", "ms"] = "product_sum",
    ms_scaling_factor: float = 1.0,
    schedule: Literal["parallel", "serial"] = "parallel",
    omp_thread_count: int = 1,
    random_schedule_seed: int = 0,
    serial_schedule_order: Sequence[int] | None = None,
    osd_method: Literal["OSD_0", "OSD_E", "OSD_CS"] = "OSD_0",
    osd_order: int = 0,
) -> DecoderSpec[ErrorDecoder]:
    """Configure a belief-propagation with ordered-statistics (BP+OSD) decoder."""
    return _decoder_spec(
        "bp_osd",
        _get_decoder_bp_osd,
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
    )


def bp_lsd(
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    max_iter: int = 0,
    bp_method: Literal["product_sum", "minimum_sum", "ps", "ms"] = "product_sum",
    ms_scaling_factor: float = 1.0,
    schedule: Literal["parallel", "serial"] = "parallel",
    omp_thread_count: int = 1,
    random_schedule_seed: int = 0,
    serial_schedule_order: Sequence[int] | None = None,
    bits_per_step: int = 1,
    lsd_method: Literal["LSD_0", "LSD_E", "LSD_CS"] = "LSD_0",
    lsd_order: int = 0,
    always_run_lsd: bool = False,
) -> DecoderSpec[ErrorDecoder]:
    """Configure a belief-propagation with localized-statistics (BP+LSD) decoder."""
    return _decoder_spec(
        "bp_lsd",
        _get_decoder_bp_lsd,
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
        lsd_method=lsd_method,
        lsd_order=lsd_order,
        always_run_lsd=always_run_lsd,
    )


def bf(
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    max_iter: int = 0,
    bp_method: Literal["product_sum", "minimum_sum", "ps", "ms"] = "product_sum",
    ms_scaling_factor: float = 1.0,
    schedule: Literal["parallel", "serial"] = "parallel",
    omp_thread_count: int = 1,
    random_schedule_seed: int = 0,
    serial_schedule_order: Sequence[int] | None = None,
    uf_method: Literal["inversion", "peeling"] = "peeling",
    bits_per_step: int = 0,
) -> DecoderSpec[ErrorDecoder]:
    """Configure a belief-find decoder."""
    return _decoder_spec(
        "bf",
        _get_decoder_bf,
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
    enable_correlations: bool = False,
    weights: float | npt.NDArray[np.floating] | Sequence[float] | None = None,
    error_probabilities: float | npt.NDArray[np.floating] | Sequence[float] | None = None,
    repetitions: int | None = None,
    timelike_weights: float | npt.NDArray[np.floating] | Sequence[float] | None = None,
    measurement_error_probabilities: float
    | npt.NDArray[np.floating]
    | Sequence[float]
    | None = None,
    merge_strategy: Literal[
        "disallow", "independent", "smallest-weight", "keep-original", "replace"
    ] = "smallest-weight",
    use_virtual_boundary_node: bool = False,
) -> DecoderSpec[BatchErrorDecoder]:
    """Configure a minimum-weight perfect matching decoder."""
    spec = _decoder_spec(
        "mwpm",
        get_error_decoder_mwpm,
        get_observable_decoder_mwpm,
        decompose_errors=decompose_errors,
        ignore_non_graphlike_errors=ignore_non_graphlike_errors,
        enable_correlations=enable_correlations,
        weights=weights,
        error_probabilities=error_probabilities,
        repetitions=repetitions,
        timelike_weights=timelike_weights,
        measurement_error_probabilities=measurement_error_probabilities,
        merge_strategy=merge_strategy,
        use_virtual_boundary_node=use_virtual_boundary_node,
    )
    if enable_correlations:
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(mwpm).parameters.items()
        }
        for name, value in spec.options.items():
            if name != "enable_correlations" and not _is_default_value(value, defaults[name]):
                raise ValueError(
                    f"The MWPM option {name}={value!r} is not supported with"
                    " enable_correlations=True"
                )
    return spec


def relay_bp(
    *,
    precision: Literal["F32", "F64", "I32", "I64"] = "F32",
    error_priors: npt.NDArray[np.floating] | Sequence[float] | None = None,
    observable_error_matrix: IntegerArray | None = None,
    include_decode_result: bool = False,
    add_erasure_bit: bool = False,
    alpha: float | None = None,
    alpha_iteration_scaling_factor: float = 1.0,
    gamma0: float = 0.1,
    data_scale_value: float | None = None,
    max_data_value: float | None = None,
    pre_iter: int = 80,
    num_sets: int = 300,
    set_max_iter: int = 60,
    gamma_dist_interval: tuple[float, float] | None = None,
    explicit_gammas: npt.NDArray[np.floating] | None = None,
    stop_nconv: int = 1,
    stopping_criterion: str | None = None,
    logging: bool = False,
    seed: int = 0,
) -> DecoderSpec[RelayBPDecoder]:
    """Configure a Relay-BP decoder."""
    optional_args = {
        "gamma_dist_interval": gamma_dist_interval,
        "stopping_criterion": stopping_criterion,
    }
    return _decoder_spec(
        "relay_bp",
        get_relay_bp_decoder,
        get_relay_bp_decoder,
        precision=precision,
        error_priors=error_priors,
        observable_error_matrix=observable_error_matrix,
        include_decode_result=include_decode_result,
        add_erasure_bit=add_erasure_bit,
        alpha=alpha,
        alpha_iteration_scaling_factor=alpha_iteration_scaling_factor,
        gamma0=gamma0,
        data_scale_value=data_scale_value,
        max_data_value=max_data_value,
        pre_iter=pre_iter,
        num_sets=num_sets,
        set_max_iter=set_max_iter,
        explicit_gammas=explicit_gammas,
        stop_nconv=stop_nconv,
        logging=logging,
        seed=seed,
        **{name: value for name, value in optional_args.items() if value is not None},
    )


def min_sum_bp(
    *,
    precision: Literal["F32", "F64", "I8", "I16", "I32", "I64", "Fixed"] = "F32",
    error_priors: npt.NDArray[np.floating] | Sequence[float] | None = None,
    observable_error_matrix: IntegerArray | None = None,
    include_decode_result: bool = False,
    add_erasure_bit: bool = False,
    max_iter: int = 200,
    alpha: float | None = None,
    alpha_iteration_scaling_factor: float = 1.0,
    gamma0: float | None = None,
    data_scale_value: float | None = None,
    max_data_value: float | None = None,
    int_bits: int | None = None,
    frac_bits: int | None = None,
) -> DecoderSpec[RelayBPDecoder]:
    """Configure a min-sum belief-propagation decoder from relay-bp."""
    return _decoder_spec(
        "min_sum_bp",
        get_min_sum_bp_decoder,
        get_min_sum_bp_decoder,
        precision=precision,
        error_priors=error_priors,
        observable_error_matrix=observable_error_matrix,
        include_decode_result=include_decode_result,
        add_erasure_bit=add_erasure_bit,
        max_iter=max_iter,
        alpha=alpha,
        alpha_iteration_scaling_factor=alpha_iteration_scaling_factor,
        gamma0=gamma0,
        data_scale_value=data_scale_value,
        max_data_value=max_data_value,
        int_bits=int_bits,
        frac_bits=frac_bits,
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
    """Configure a lookup-table decoder."""
    return _decoder_spec(
        "lookup_table",
        get_decoder_lookup,
        get_observable_decoder_lookup,
        max_weight=max_weight,
        error_channel=error_channel,
        penalty_func=penalty_func,
        observable_flip_matrix=observable_flip_matrix,
        post_select=post_select,
        add_erasure_bit=add_erasure_bit,
        confidence_ratio=confidence_ratio,
        symplectic=symplectic,
    )


def ilp(*, add_erasure_bit: bool = False, **solver_args: object) -> DecoderSpec[ILPDecoder]:
    """Configure an integer-linear-program decoder."""
    return _decoder_spec("ilp", _get_decoder_ilp, add_erasure_bit=add_erasure_bit, **solver_args)


def guf(
    *,
    max_weight: int | None = None,
    symplectic: bool = False,
    add_erasure_bit: bool = False,
) -> DecoderSpec[GUFDecoder]:
    """Configure a generalized union-find decoder."""
    return _decoder_spec(
        "guf",
        _get_decoder_guf,
        max_weight=max_weight,
        symplectic=symplectic,
        add_erasure_bit=add_erasure_bit,
    )


# Modern decoder resolution APIs


def is_prebuilt_decoder(decoder: object) -> bool:
    """Whether a decoder input is already built rather than settings or a constructor."""
    return (
        decoder is not None
        and not isinstance(decoder, (DecoderSpec, type))
        and any(
            hasattr(decoder, method)
            for method in (
                "decode_errors",
                "decode",
                "decode_observables",
                "decode_shots_bit_packed",
            )
        )
    )


def reject_prebuilt_decoder(decoder: object, reason: str) -> None:
    """Raise if a decoder input is prebuilt and cannot be rebuilt for a new matrix."""
    if is_prebuilt_decoder(decoder):
        raise ValueError(
            f"A prebuilt decoder cannot be passed as decoder= here because {reason}.  Pass decoder"
            " settings such as decoder=decoders.bp_osd(...), or a decoder constructor, instead"
        )


def reject_removed_decoder_args(decoder_args: Mapping[str, object]) -> None:
    """Reject the removed static_decoder argument outside the legacy APIs."""
    if "static_decoder" in decoder_args:
        raise TypeError(
            "The static_decoder argument has been removed; pass a prebuilt decoder as decoder="
            " instead"
        )


def get_error_decoder(pcm_or_dem: PcmOrDem, *, decoder: ErrorDecoderInput = None) -> ErrorDecoder:
    """Build or retrieve a decoder that maps a syndrome to an inferred error."""
    return as_error_decoder(*_build_decoder(pcm_or_dem, decoder))


def _build_decoder(pcm_or_dem: PcmOrDem, decoder: ObservableDecoderInput) -> tuple[object, str]:
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


def resolve_decoder(
    pcm_or_dem: PcmOrDem,
    decoder: ErrorDecoderInput,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> ErrorDecoder:
    """Resolve an error decoder input and optional deprecated construction arguments."""
    decoder_input = _merge_legacy_decoder_args(
        pcm_or_dem, decoder, decoder_args, warn_deprecated=warn_deprecated
    )
    return as_error_decoder(*_build_decoder(pcm_or_dem, decoder_input))


def decode_observables(
    dem: stim.DetectorErrorModel,
    syndrome: npt.NDArray[np.int_],
    *,
    decoder: ObservableDecoderInput = None,
) -> npt.NDArray[np.int_]:
    """Construct a decoder and predict the observable flips of one syndrome."""
    return get_observable_decoder(dem, decoder=decoder).decode_observables(syndrome)


def get_observable_decoder(
    dem: stim.DetectorErrorModel, *, decoder: ObservableDecoderInput = None
) -> ObservableDecoder:
    """Build or retrieve an observable decoder for a detector error model."""
    return resolve_observable_decoder(dem, decoder, {})


def resolve_observable_decoder(
    dem: stim.DetectorErrorModel,
    decoder: ObservableDecoderInput,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> ObservableDecoder:
    """Resolve an observable decoder input and optional deprecated construction arguments."""
    if decoder_args:
        decoder = _merge_legacy_decoder_args(
            dem, decoder, decoder_args, warn_deprecated=warn_deprecated
        )
    elif isinstance(decoder, DecoderSpec):
        return decoder.build_observable_decoder(dem)

    built_decoder, source = _build_decoder(dem, decoder)
    if isinstance(built_decoder, ObservableDecoder):
        return built_decoder
    return _ErrorsToObservablesDecoder(as_error_decoder(built_decoder, source), dem)


# Legacy keyword-based compatibility

_LEGACY_HELPER_NAMES = {
    "BF": "bf",
    "BP_LSD": "bp_lsd",
    "BP_OSD": "bp_osd",
    "GUF": "guf",
    "ILP": "ilp",
    "MWPM": "mwpm",
    "RBP": "relay_bp",
    "lookup": "lookup_table",
}

DECODER_CONSTRUCTORS: dict[str, Callable[..., ErrorDecoder]] = {
    "BF": _get_decoder_bf,
    "BP_LSD": _get_decoder_bp_lsd,
    "BP_OSD": _get_decoder_bp_osd,
    "GUF": _get_decoder_guf,
    "ILP": _get_decoder_ilp,
    "MWPM": _get_decoder_mwpm,
    "RBP": _get_decoder_rbp,
    "lookup": get_decoder_lookup,
}


def get_decoder(pcm_or_dem: PcmOrDem, **decoder_args: object) -> Any:
    """Retrieve a decoder through the deprecated keyword-based API."""
    warnings.warn(
        _get_deprecated_function_message("decoders.get_decoder", pcm_or_dem, decoder_args),
        DeprecationWarning,
        stacklevel=get_external_caller_stacklevel(),
    )
    return _get_legacy_decoder(pcm_or_dem, decoder_args)


def decode(
    pcm_or_dem: PcmOrDem, syndrome: npt.NDArray[np.int_], **decoder_args: object
) -> npt.NDArray[np.int_]:
    """Construct a decoder and decode one syndrome through the deprecated API."""
    warnings.warn(
        _get_deprecated_function_message(
            "decoders.decode", pcm_or_dem, decoder_args, decodes_syndrome=True
        ),
        DeprecationWarning,
        stacklevel=get_external_caller_stacklevel(),
    )
    return _get_legacy_decoder(pcm_or_dem, decoder_args).decode(syndrome)


def _get_legacy_decoder(pcm_or_dem: PcmOrDem, decoder_args: Mapping[str, object]) -> Any:
    """Build a decoder with the deprecated keyword-based API."""
    static_decoder = decoder_args.get("static_decoder")
    if decoder_args.get("decoder_constructor") is None and static_decoder is not None:
        if len(decoder_args) > 1:
            raise ValueError("If passed a static decoder, we cannot process decoding arguments")
        return static_decoder
    built_decoder, _ = _build_decoder(
        pcm_or_dem, _get_legacy_decoder_input(pcm_or_dem, decoder_args)
    )
    return built_decoder


def _get_legacy_decoder_input(
    pcm_or_dem: PcmOrDem, decoder_args: Mapping[str, object]
) -> ErrorDecoderInput:
    """Translate deprecated decoder arguments into a decoder input."""
    decoder_args = dict(decoder_args)
    if (decoder_constructor := decoder_args.pop("decoder_constructor", None)) is not None:
        if not callable(decoder_constructor):
            raise TypeError("The decoder_constructor argument must be callable")
        return functools.partial(decoder_constructor, **decoder_args)

    decoder_names = [
        name for name in DECODER_CONSTRUCTORS if decoder_args.pop(f"with_{name}", False)
    ]
    if len(decoder_names) > 1:
        raise ValueError(
            "Only one decoder can be requested at a time, but received requests for: "
            + ", ".join(decoder_names)
        )
    if decoder_names:
        (decoder_name,) = decoder_names
    elif isinstance(pcm_or_dem, galois.FieldArray) and type(pcm_or_dem).order != 2:
        decoder_name = "GUF"
    else:
        decoder_name = "BP_OSD"
    return _decoder_spec(
        _LEGACY_HELPER_NAMES[decoder_name], DECODER_CONSTRUCTORS[decoder_name], **decoder_args
    )


def _merge_legacy_decoder_args(
    pcm_or_dem: PcmOrDem,
    decoder: ObservableDecoderInput,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> ObservableDecoderInput:
    """Translate deprecated keyword arguments into a decoder input."""
    reject_removed_decoder_args(decoder_args)
    if not decoder_args:
        return decoder
    if decoder is not None:
        raise ValueError(
            "Cannot combine decoder= with deprecated decoder-selection or construction arguments"
        )
    if warn_deprecated:
        warnings.warn(
            get_legacy_decoder_migration_message(pcm_or_dem, decoder_args),
            DeprecationWarning,
            stacklevel=get_external_caller_stacklevel(),
        )
    return _get_legacy_decoder_input(pcm_or_dem, decoder_args)


def _get_legacy_decoder_replacement(
    pcm_or_dem: PcmOrDem | None,
    decoder_args: Mapping[str, object],
    *,
    argument_name: str = "decoder",
) -> str:
    """Return the decoder= expression replacing deprecated arguments."""
    if (decoder_constructor := decoder_args.get("decoder_constructor")) is not None:
        return f"{argument_name}={_get_constructor_name(decoder_constructor)}"
    return f"{argument_name}=decoders.{_get_legacy_helper_name(pcm_or_dem, decoder_args)}(...)"


def _get_constructor_name(decoder_constructor: object) -> str:
    """Return a constructor name suitable for a migration message."""
    return getattr(decoder_constructor, "__name__", "MyDecoder")


def _get_legacy_helper_name(pcm_or_dem: PcmOrDem | None, decoder_args: Mapping[str, object]) -> str:
    """Return the typed helper replacing deprecated decoder selection."""
    selected = [name for name in DECODER_CONSTRUCTORS if decoder_args.get(f"with_{name}", False)]
    if len(selected) == 1:
        return _LEGACY_HELPER_NAMES[selected[0]]
    if isinstance(pcm_or_dem, galois.FieldArray) and type(pcm_or_dem).order != 2:
        return "guf"
    return "bp_osd"


def get_legacy_decoder_migration_message(
    pcm_or_dem: PcmOrDem | None,
    decoder_args: Mapping[str, object],
    *,
    argument_name: str = "decoder",
) -> str:
    """Describe the typed replacement for deprecated decoder arguments."""
    replacement = _get_legacy_decoder_replacement(
        pcm_or_dem, decoder_args, argument_name=argument_name
    )
    if decoder_args.get("decoder_constructor") is not None:
        return (
            "The decoder_constructor keyword is deprecated; pass the constructor as"
            f" {argument_name}= instead, for example {replacement}"
        )
    if decoder_args.get("predict_observable_flips"):
        return (
            "predict_observable_flips=True is deprecated; construct an ObservableLookupDecoder"
            " directly and call decode_observables(...) instead"
        )
    selected = [name for name in DECODER_CONSTRUCTORS if decoder_args.get(f"with_{name}", False)]
    if len(selected) == 1:
        return (
            f"The with_{selected[0]} keyword and free-form decoder options are deprecated; use"
            f" {replacement} instead"
        )
    if len(selected) > 1:
        return (
            "The with_<NAME> decoder-selection keywords are deprecated; pass exactly one typed"
            f" decoder specification such as {argument_name}=decoders.bp_osd(...) instead"
        )
    return f"Passing free-form decoder options is deprecated; move them into {replacement} instead"


def _get_deprecated_function_message(
    function_name: str,
    pcm_or_dem: PcmOrDem,
    decoder_args: Mapping[str, object],
    *,
    decodes_syndrome: bool = False,
) -> str:
    """Describe the replacement for deprecated get_decoder or decode."""
    if (decoder_constructor := decoder_args.get("decoder_constructor")) is not None:
        other_args = ", ..." if len(decoder_args) > 1 else ""
        replacement = f"{_get_constructor_name(decoder_constructor)}(pcm_or_dem{other_args})"
    elif decoder_args.get("static_decoder") is not None:
        replacement = "static_decoder"
    elif decoder_args:
        helper_name = _get_legacy_helper_name(pcm_or_dem, decoder_args)
        replacement = f"decoders.{helper_name}(...).build(pcm_or_dem)"
    else:
        replacement = "decoders.get_error_decoder(pcm_or_dem)"
    if decodes_syndrome:
        replacement += ".decode(syndrome)"
    message = f"{function_name} is deprecated; use {replacement} instead"
    if decoder_args.get("predict_observable_flips"):
        message += (
            ".  To predict observable flips, construct an ObservableLookupDecoder directly and call"
            " decode_observables(...)"
        )
    return message


DEPRECATED_ALIASES: dict[str, Any] = {
    "BatchDecoder": BatchErrorDecoder,
    "Decoder": ErrorDecoder,
    "ErrorsToObservablesDecoder": _ErrorsToObservablesDecoder,
    "ExpandedErrorDecoder": _ExpandedErrorDecoder,
    "_get_error_decoder_MWPM": get_error_decoder_mwpm,
    "_get_min_sum_bp_decoder": get_min_sum_bp_decoder,
    "_get_observable_decoder_MWPM": get_observable_decoder_mwpm,
    "_get_observable_lookup_decoder": get_observable_decoder_lookup,
    "_get_relay_bp_decoder": get_relay_bp_decoder,
    "get_decoder_BF": _get_decoder_bf,
    "get_decoder_BP_LSD": _get_decoder_bp_lsd,
    "get_decoder_BP_OSD": _get_decoder_bp_osd,
    "get_decoder_GUF": _get_decoder_guf,
    "get_decoder_ILP": _get_decoder_ilp,
    "get_decoder_MWPM": _get_decoder_mwpm,
    "get_decoder_RBP": _get_decoder_rbp,
    "match_error_decoder_to_dem": _match_error_decoder_to_dem,
}

if TYPE_CHECKING:
    BatchDecoder = BatchErrorDecoder
    Decoder = ErrorDecoder
    ErrorsToObservablesDecoder = _ErrorsToObservablesDecoder
    ExpandedErrorDecoder = _ExpandedErrorDecoder
    get_decoder_BF = _get_decoder_bf
    get_decoder_BP_LSD = _get_decoder_bp_lsd
    get_decoder_BP_OSD = _get_decoder_bp_osd
    get_decoder_GUF = _get_decoder_guf
    get_decoder_ILP = _get_decoder_ilp
    get_decoder_MWPM = _get_decoder_mwpm
    get_decoder_RBP = _get_decoder_rbp
    match_error_decoder_to_dem = _match_error_decoder_to_dem
else:

    def __getattr__(name: str) -> Any:
        """Resolve deprecated retrieval names with a DeprecationWarning."""
        return get_deprecated_alias(__name__, name, DEPRECATED_ALIASES)
