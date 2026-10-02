# SPDX-License-Identifier: Apache-2.0

"""Typed settings for deferred decoder construction."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Collection, Sequence
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import stim

from qldpc.math import IntegerArray

from ..adapters.error_decoders import ErrorsToObservablesDecoder as _ErrorsToObservablesDecoder
from ..adapters.observable_decoders import (
    validate_observable_decoder as _validate_observable_decoder,
)
from ..common import PLACEHOLDER_ERROR_RATE
from ..custom.guf import GUFDecoder
from ..custom.guf import get_decoder_guf as _get_decoder_guf
from ..custom.ilp import ILPDecoder
from ..custom.ilp import get_decoder_ilp as _get_decoder_ilp
from ..custom.lookup import (
    LookupDecoder,
    get_decoder_lookup,
    get_observable_decoder_lookup,
)
from ..external.ldpc import get_decoder_bf as _get_decoder_bf
from ..external.ldpc import get_decoder_bp_lsd as _get_decoder_bp_lsd
from ..external.ldpc import get_decoder_bp_osd as _get_decoder_bp_osd
from ..external.pymatching import get_error_decoder_mwpm, get_observable_decoder_mwpm
from ..external.relay_bp import (
    RelayBPDecoder,
    get_min_sum_bp_decoder,
    get_relay_bp_decoder,
)
from ..external.tesseract import (
    TesseractDecoder,
    TesseractDetectorOrderMethod,
    get_decoder_tesseract,
)
from ..protocols import (
    BatchErrorDecoder,
    ErrorDecoder,
    ObservableDecoder,
    SupportsDecode,
)

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


class ErrorDecoderConstructor(Protocol):
    """Callable that builds an error decoder from a matrix or detector error model."""

    def __call__(self, pcm_or_dem: PcmOrDem, /) -> ErrorDecoder | SupportsDecode:
        """Build an error decoder."""


class ObservableDecoderConstructor(Protocol):
    """Callable that builds an observable decoder from a detector error model."""

    def __call__(self, dem: stim.DetectorErrorModel, /) -> ObservableDecoder:
        """Build an observable decoder."""


class ObservableDecoderCompiler(Protocol):
    """Object that compiles an observable decoder for a detector error model."""

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> ObservableDecoder:
        """Build an observable decoder specialized to one detector error model."""


DeferredErrorDecoderInput: TypeAlias = DecoderSpec[ErrorDecoder] | ErrorDecoderConstructor | None
"""A decoder= input that builds an error decoder later, for a matrix or detector error model that
the receiving method constructs: decoder settings, an error-decoder constructor, or None to select
the default decoder.  Prebuilt decoders are excluded, because they are tied to one matrix."""

ErrorDecoderInput: TypeAlias = DeferredErrorDecoderInput | ErrorDecoder | SupportsDecode
"""A decoder= input that yields an error decoder: a DeferredErrorDecoderInput, or a prebuilt error
decoder (an ErrorDecoder, or any object whose decode method returns an inferred error)."""

DeferredDecoderInput: TypeAlias = (
    DeferredErrorDecoderInput | ObservableDecoderConstructor | ObservableDecoderCompiler
)
"""A decoder= input that builds an error or observable decoder later, for a matrix or detector
error model that the receiving method constructs: a DeferredErrorDecoderInput, an
observable-decoder constructor, or an observable-decoder compiler such as a SinterDecoder.
Prebuilt decoders are excluded, because they are tied to one matrix or detector error model."""

DecoderInput: TypeAlias = ErrorDecoderInput | DeferredDecoderInput | ObservableDecoder
"""Any decoder= input: an ErrorDecoderInput, a DeferredDecoderInput, or a prebuilt observable
decoder.  The receiving method adapts the decoder that the input yields to what it needs."""


# Typed decoder-specification helpers


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


def tesseract(
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    add_erasure_bit: bool = False,
    det_beam: int = 5,
    beam_climbing: bool = False,
    no_revisit_dets: bool = True,
    verbose: bool = False,
    merge_errors: bool | None = None,
    pqlimit: int = 200_000,
    det_orders: Sequence[Sequence[int]] | None = None,
    det_penalty: float = 0.0,
    create_visualization: bool = False,
    sparsify_errors: bool = False,
    sparsify_base_degree: int = -1,
    sparsify_max_degree: int = -1,
    sparsify_reactivate_limit: int = -1,
    num_det_orders: int | None = None,
    det_order_method: TesseractDetectorOrderMethod | None = None,
    seed: int | None = None,
) -> DecoderSpec[TesseractDecoder]:
    """Configure an optional Tesseract search-based decoder."""
    return _decoder_spec(
        "tesseract",
        get_decoder_tesseract,
        get_decoder_tesseract,
        error_rate=error_rate,
        error_channel=error_channel,
        add_erasure_bit=add_erasure_bit,
        det_beam=det_beam,
        beam_climbing=beam_climbing,
        no_revisit_dets=no_revisit_dets,
        verbose=verbose,
        merge_errors=merge_errors,
        pqlimit=pqlimit,
        det_orders=det_orders,
        det_penalty=det_penalty,
        create_visualization=create_visualization,
        sparsify_errors=sparsify_errors,
        sparsify_base_degree=sparsify_base_degree,
        sparsify_max_degree=sparsify_max_degree,
        sparsify_reactivate_limit=sparsify_reactivate_limit,
        num_det_orders=num_det_orders,
        det_order_method=det_order_method,
        seed=seed,
    )


def lookup_table(
    max_weight: int,
    *,
    error_channel: (
        npt.NDArray[np.floating]
        | Sequence[float]
        | Callable[[npt.NDArray[np.int_] | Sequence[int]], float]
        | None
    ) = None,
    observable_flip_matrix: IntegerArray | None = None,
    post_select: Collection[int] = (),
    add_erasure_bit: bool | None = None,
    confidence_ratio: float | None = None,
    probability_cutoff: float = 0,
    symplectic: bool = False,
    penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None = None,
) -> DecoderSpec[LookupDecoder]:
    """Configure a lookup-table decoder."""
    return _decoder_spec(
        "lookup_table",
        get_decoder_lookup,
        get_observable_decoder_lookup,
        max_weight=max_weight,
        error_channel=error_channel,
        observable_flip_matrix=observable_flip_matrix,
        post_select=post_select,
        add_erasure_bit=add_erasure_bit,
        confidence_ratio=confidence_ratio,
        probability_cutoff=probability_cutoff,
        symplectic=symplectic,
        penalty_func=penalty_func,
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


# Specification-helper internals


def _is_default_value(value: object, default: object) -> bool:
    """Whether an option is a plain copy of its default value."""
    if value is default:
        return True
    plain_types = (bool, int, float, str)
    return type(value) is type(default) and isinstance(value, plain_types) and value == default


def _decoder_spec(
    helper_name: str,
    builder: Callable[..., _Decoder],
    observable_builder: Callable[..., ObservableDecoder] | None = None,
    /,
    **options: object,
) -> DecoderSpec[_Decoder]:
    """Store deferred decoder construction options."""
    return DecoderSpec(helper_name, builder, tuple(options.items()), observable_builder)
