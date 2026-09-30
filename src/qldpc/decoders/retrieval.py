# SPDX-License-Identifier: Apache-2.0

"""Methods to decode, or retrieve various decoders."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Generic, Literal, ParamSpec, Protocol, TypeAlias, TypeVar, cast

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc._util import format_docstring, get_external_caller_stacklevel
from qldpc.math import IntegerArray

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
"""A parity check matrix or detector error model, from which to build an error decoder."""


@dataclasses.dataclass(frozen=True, slots=True, eq=False, repr=False)
class DecoderSpec(Generic[_DecoderT_co]):
    """Deferred, typed construction settings for an error decoder.

    Build a DecoderSpec with a helper function in qldpc.decoders, such as ``decoders.bp_osd(...)``,
    rather than by calling this class directly.  A DecoderSpec only stores settings: the method that
    consumes it builds a decoder once the parity-check matrix or detector error model to decode is
    known, so one DecoderSpec can configure decoders for many matrices.
    """

    _helper_name: str
    _builder: Callable[..., _DecoderT_co]
    _options: tuple[tuple[str, object], ...]

    def build(self, pcm_or_dem: PcmOrDem) -> _DecoderT_co:
        """Build an error decoder for a parity-check matrix or detector error model."""
        return self._builder(pcm_or_dem, **dict(self._options))

    def __repr__(self) -> str:
        """Show the helper call that reproduces this spec, omitting default options."""
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
    """Whether an option value is (a plain copy of) its default value."""
    if value is default:
        return True
    plain_types = (bool, int, float, str)
    return type(value) is type(default) and isinstance(value, plain_types) and value == default


class ErrorDecoderConstructor(Protocol):
    """Callable that builds an error decoder from a matrix or detector error model."""

    def __call__(self, pcm_or_dem: PcmOrDem, /) -> ErrorDecoder:
        """Build an error decoder."""


DeferredErrorDecoderInput: TypeAlias = DecoderSpec[ErrorDecoder] | ErrorDecoderConstructor | None
"""Decoder settings, a decoder constructor, or None to select the default decoder.

This is ErrorDecoderInput without a prebuilt decoder.  It is accepted by methods that decode a
matrix or detector error model that they construct themselves, and so must build the decoder.
"""

ErrorDecoderInput: TypeAlias = (
    DecoderSpec[ErrorDecoder] | ErrorDecoderConstructor | ErrorDecoder | None
)
"""Decoder settings, a decoder constructor, a prebuilt error decoder, or None for the default.

This is accepted by methods that decode a matrix or detector error model that the caller knows, so
that the caller can prebuild a decoder for it.
"""

_OBSERVABLE_DECODER_ADVICE = (
    "Pass error-decoder settings such as decoders.bp_osd(...), or call the decode_observables method"
    " of an observable decoder directly"
)


def _decoder_spec(
    helper_name: str, builder: Callable[..., _Decoder], **options: object
) -> DecoderSpec[_Decoder]:
    """Store deferred decoder construction options."""
    return DecoderSpec(helper_name, builder, tuple(options.items()))


def _is_prebuilt_decoder(decoder: object) -> bool:
    """Whether a decoder input is an already-built decoder, rather than settings or a constructor.

    This classification matches the order in which _resolve_decoder interprets a decoder input.
    """
    return (
        decoder is not None
        and not isinstance(decoder, (DecoderSpec, type))
        and hasattr(decoder, "decode")
    )


def _reject_prebuilt_decoder(decoder: object, reason: str) -> None:
    """Raise an error if given a prebuilt decoder, which cannot be rebuilt for a new matrix.

    Args:
        decoder: A decoder input, as passed to get_decoder.
        reason: A clause that completes the error message "A prebuilt decoder cannot be passed as
            decoder= here because {reason}.", such as "windows are decoded independently".
    """
    if _is_prebuilt_decoder(decoder):
        raise ValueError(
            f"A prebuilt decoder cannot be passed as decoder= here because {reason}.  Pass decoder"
            " settings such as decoder=decoders.bp_osd(...), or a decoder constructor, instead"
        )


def _reject_removed_decoder_args(decoder_args: Mapping[str, object]) -> None:
    """Raise an error if given a decoder argument that has been removed."""
    if "static_decoder" in decoder_args:
        raise TypeError(
            "The static_decoder argument has been removed; pass a prebuilt decoder as decoder="
            " instead"
        )


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
        decoder: Decoder settings from a helper such as ``decoders.bp_osd(...)``, a prebuilt error
            decoder, a constructor that builds an error decoder from pcm_or_dem, or None to select
            the default decoder.  See help(qldpc.decoders.get_decoder).
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
        decoder: Decoder settings from a helper such as ``decoders.bp_osd(...)``, a prebuilt error
            decoder, a constructor that builds an error decoder from pcm_or_dem, or None to select
            the default decoder.  A prebuilt decoder is returned as is, so it must have been built
            for pcm_or_dem.
        **decoder_args: Deprecated decoder-selection and construction arguments.

    Returns:
        An error decoder configured for pcm_or_dem.

    If decoder is None, this method defaults to generalized union-find (GUF) for non-binary parity
    check matrices, and BP+OSD otherwise.

    The legacy ``with_<NAME>``, ``decoder_constructor``, and free-form decoder arguments remain
    available during a deprecation period.  The ``static_decoder`` argument has been removed; pass a
    prebuilt decoder as ``decoder=`` instead.
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
    _reject_removed_decoder_args(decoder_args)
    if decoder is not None:
        if decoder_args:
            raise ValueError(
                "Cannot combine decoder= with deprecated decoder-selection or construction arguments"
            )
        if getattr(decoder, "decodes_observables", False):
            raise TypeError(
                "decoder predicts observable flips rather than errors.  "
                + _OBSERVABLE_DECODER_ADVICE
            )
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
                stacklevel=get_external_caller_stacklevel(),
            )
        return _get_legacy_decoder(pcm_or_dem, decoder_args)

    if isinstance(pcm_or_dem, galois.FieldArray) and type(pcm_or_dem).order != 2:
        return get_decoder_GUF(pcm_or_dem)
    return get_decoder_BP_OSD(pcm_or_dem)


def _get_legacy_decoder_migration_message(
    pcm_or_dem: PcmOrDem | None,
    decoder_args: Mapping[str, object],
    *,
    argument_name: str = "decoder",
) -> str:
    """Describe the typed replacement for one legacy decoder request.

    Args:
        pcm_or_dem: The matrix or detector error model to decode, if known, which determines the
            default decoder.
        decoder_args: The deprecated decoder-selection and construction arguments.
        argument_name: The name of the argument that replaces decoder_args, such as "decoder" or
            "decoder_x".
    """
    if decoder_args.get("decoder_constructor") is not None:
        return (
            f"The decoder_constructor keyword is deprecated; pass the constructor as {argument_name}="
            f" instead, for example {argument_name}=MyDecoder"
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
            f" {argument_name}=decoders.{helper_name}(...) instead"
        )
    if len(selected) > 1:
        return (
            "The with_<NAME> decoder-selection keywords are deprecated; pass exactly one typed"
            f" decoder specification such as {argument_name}=decoders.bp_osd(...) instead"
        )

    helper_name = (
        "guf"
        if isinstance(pcm_or_dem, galois.FieldArray) and type(pcm_or_dem).order != 2
        else "bp_osd"
    )
    return (
        "Passing free-form decoder options is deprecated; move them into"
        f" {argument_name}=decoders.{helper_name}(...) instead"
    )


def _validate_error_decoder(decoder: object, source: str) -> ErrorDecoder:
    """Validate and type-narrow an object expected to decode syndromes to errors."""
    if getattr(decoder, "decodes_observables", False):
        raise TypeError(
            f"{source} predicts observable flips rather than errors.  " + _OBSERVABLE_DECODER_ADVICE
        )
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


def _erasure_bit_support(
    supported: bool,
) -> Callable[[Callable[_Parameters, _Decoder]], Callable[_Parameters, _Decoder]]:
    """Declare and enforce whether a decoder getter supports an erasure bit."""

    def decorator(
        decoder_getter: Callable[_Parameters, _Decoder],
    ) -> Callable[_Parameters, _Decoder]:
        decoder_name = decoder_getter.__name__.removeprefix("get_decoder_")
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
def get_decoder_BP_OSD(
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

    For details about the BP+OSD decoder and its arguments, see:

    - help(ldpc.BpOsdDecoder)
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2005.07016
    """
    import ldpc

    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return ldpc.BpOsdDecoder(pcm, error_channel=error_channel, **decoder_args)


@_erasure_bit_support(False)
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
def get_decoder_BP_LSD(
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

    For details about the BP+LSD decoder and its arguments, see:

    - help(ldpc.bplsd_decoder.BpLsdDecoder)
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2406.18655
    """
    import ldpc

    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return ldpc.bplsd_decoder.BpLsdDecoder(pcm, error_channel=error_channel, **decoder_args)


@_erasure_bit_support(False)
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
def get_decoder_BF(
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
def get_decoder_MWPM(
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
        **decoder_args: Additional keyword arguments passed to
            pymatching.Matching.from_check_matrix.

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
def get_decoder_RBP(
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
def get_decoder_ILP(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    add_erasure_bit: bool = False,
    **decoder_args: object,
) -> ILPDecoder:
    """Decoder based on solving an integer linear program (ILP)."""
    return ILPDecoder(_to_pcm(pcm_or_dem), add_erasure_bit=add_erasure_bit, **decoder_args)


@_erasure_bit_support(True)
def get_decoder_GUF(
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
    bp_method: Literal["product_sum", "minimum_sum", "ps", "ms"] = "product_sum",
    ms_scaling_factor: float = 1.0,
    schedule: Literal["parallel", "serial"] = "parallel",
    omp_thread_count: int = 1,
    random_schedule_seed: int = 0,
    serial_schedule_order: Sequence[int] | None = None,
    osd_method: Literal["OSD_0", "OSD_E", "OSD_CS"] = "OSD_0",
    osd_order: int = 0,
) -> DecoderSpec[ErrorDecoder]:
    """Configure a decoder based on belief propagation with ordered statistics (BP+OSD).

    Args:
        error_rate: The i.i.d. probability of each error, used to build a default error_channel
            when decoding a parity check matrix.  Ignored when decoding a detector error model,
            which supplies its own error probabilities.
        error_channel: The probability of each error mechanism.  If provided, this overrides both
            error_rate and the error probabilities of a detector error model.
        max_iter: Maximum number of belief-propagation iterations.  If 0 (the default), ldpc uses
            the number of error mechanisms.
        bp_method: Belief-propagation update rule: "product_sum" ("ps") or "minimum_sum" ("ms").
        ms_scaling_factor: Scaling factor for minimum-sum updates.
        schedule: Belief-propagation message-passing schedule.
        omp_thread_count: Number of OpenMP threads used by ldpc.
        random_schedule_seed: Seed that ldpc uses to randomize a serial schedule.
        serial_schedule_order: An explicit order in which to update variable nodes in a serial
            schedule.
        osd_method: Ordered-statistics post-processing method: order-0 ("OSD_0"), exhaustive
            ("OSD_E"), or combination sweep ("OSD_CS").
        osd_order: Ordered-statistics search depth.  Must be 0 if osd_method is "OSD_0".

    Returns:
        Decoder settings to pass as ``decoder=`` to a method that decodes syndromes.  These
        settings build an ldpc.BpOsdDecoder, which cannot signal erasure.

    For details about the BP+OSD decoder, see:

    - help(ldpc.BpOsdDecoder)
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2005.07016
    """
    return _decoder_spec(
        "bp_osd",
        get_decoder_BP_OSD,
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
    """Configure a decoder based on belief propagation with localized statistics (BP+LSD).

    Args:
        error_rate: The i.i.d. probability of each error, used to build a default error_channel
            when decoding a parity check matrix.  Ignored when decoding a detector error model,
            which supplies its own error probabilities.
        error_channel: The probability of each error mechanism.  If provided, this overrides both
            error_rate and the error probabilities of a detector error model.
        max_iter: Maximum number of belief-propagation iterations.  If 0 (the default), ldpc uses
            the number of error mechanisms.
        bp_method: Belief-propagation update rule: "product_sum" ("ps") or "minimum_sum" ("ms").
        ms_scaling_factor: Scaling factor for minimum-sum updates.
        schedule: Belief-propagation message-passing schedule.
        omp_thread_count: Number of OpenMP threads used by ldpc.
        random_schedule_seed: Seed that ldpc uses to randomize a serial schedule.
        serial_schedule_order: An explicit order in which to update variable nodes in a serial
            schedule.
        bits_per_step: Number of bits added to a cluster in each growth step.  If 0, ldpc uses the
            number of error mechanisms.
        lsd_method: Ordered-statistics method applied within each cluster: order-0 ("LSD_0"),
            exhaustive ("LSD_E"), or combination sweep ("LSD_CS").
        lsd_order: Ordered-statistics search depth within each cluster.
        always_run_lsd: Whether to run LSD even when belief propagation converges.

    Returns:
        Decoder settings to pass as ``decoder=`` to a method that decodes syndromes.  These
        settings build an ldpc.bplsd_decoder.BpLsdDecoder, which cannot signal erasure.

    For details about the BP+LSD decoder, see:

    - help(ldpc.bplsd_decoder.BpLsdDecoder)
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2406.18655
    """
    return _decoder_spec(
        "bp_lsd",
        get_decoder_BP_LSD,
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
    """Configure a decoder based on belief propagation with union-find (belief-find, or BF).

    Args:
        error_rate: The i.i.d. probability of each error, used to build a default error_channel
            when decoding a parity check matrix.  Ignored when decoding a detector error model,
            which supplies its own error probabilities.
        error_channel: The probability of each error mechanism.  If provided, this overrides both
            error_rate and the error probabilities of a detector error model.
        max_iter: Maximum number of belief-propagation iterations.  If 0 (the default), ldpc uses
            the number of error mechanisms.
        bp_method: Belief-propagation update rule: "product_sum" ("ps") or "minimum_sum" ("ms").
        ms_scaling_factor: Scaling factor for minimum-sum updates.
        schedule: Belief-propagation message-passing schedule.
        omp_thread_count: Number of OpenMP threads used by ldpc.
        random_schedule_seed: Seed that ldpc uses to randomize a serial schedule.
        serial_schedule_order: An explicit order in which to update variable nodes in a serial
            schedule.
        uf_method: Union-find cluster decoding method.  The "peeling" method requires every error
            mechanism to flip at most two checks; "inversion" applies to general LDPC codes.
        bits_per_step: Number of bits added to a cluster in each growth step.  If 0, ldpc uses the
            number of error mechanisms.

    Returns:
        Decoder settings to pass as ``decoder=`` to a method that decodes syndromes.  These
        settings build an ldpc.BeliefFindDecoder, which cannot signal erasure.

    For details about the BF decoder, see:

    - help(ldpc.BeliefFindDecoder)
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - References:

      - https://arxiv.org/abs/1709.06218
      - https://arxiv.org/abs/2103.08049
      - https://arxiv.org/abs/2209.01180
    """
    return _decoder_spec(
        "bf",
        get_decoder_BF,
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
    """Configure a decoder based on minimum-weight perfect matching (MWPM).

    Every error mechanism must flip at most two detectors (checks).  A detector error model can
    satisfy this condition by decomposing its errors, or by ignoring errors that it cannot match.

    This helper exposes the options of pymatching.Matching.from_check_matrix that are compatible
    with decoding syndromes to errors.  In particular, it omits the faults_matrix option, which
    would make the decoder return observable flips rather than errors.

    Args:
        decompose_errors: Whether to split the errors of a detector error model along the
            decompositions that it suggests, as provided by
            ``circuit.detector_error_model(decompose_errors=True)``.
        ignore_non_graphlike_errors: Whether to drop errors that flip more than two detectors
            (after any decomposition), rather than raising an error.
        weights: Scalar or per-error matching weights for a parity check matrix.  A detector
            error model supplies its own weights, so this must be None when decoding one.
        error_probabilities: Scalar or per-error probabilities for a parity check matrix, from
            which pymatching derives weights.
        repetitions: Number of rounds for phenomenological noise; see
            help(pymatching.Matching.from_check_matrix).
        timelike_weights: Weights of measurement errors between repetitions.
        measurement_error_probabilities: Probabilities of measurement errors between
            repetitions.
        merge_strategy: How pymatching merges parallel edges between the same pair of detectors.
        use_virtual_boundary_node: Whether pymatching models the boundary as a single virtual
            node.

    Returns:
        Decoder settings to pass as ``decoder=`` to a method that decodes syndromes.  These
        settings build a pymatching.Matching that decodes syndromes to errors (not to observable
        flips), and that cannot signal erasure.

    For details about the MWPM decoder, see:

    - help(pymatching.Matching.from_check_matrix)
    - Documentation: https://pymatching.readthedocs.io
    - Reference: https://arxiv.org/abs/2303.15933
    """
    return _decoder_spec(
        "mwpm",
        get_decoder_MWPM,
        decompose_errors=decompose_errors,
        ignore_non_graphlike_errors=ignore_non_graphlike_errors,
        weights=weights,
        error_probabilities=error_probabilities,
        repetitions=repetitions,
        timelike_weights=timelike_weights,
        measurement_error_probabilities=measurement_error_probabilities,
        merge_strategy=merge_strategy,
        use_virtual_boundary_node=use_virtual_boundary_node,
    )


def relay_bp(
    *,
    error_priors: npt.NDArray[np.floating] | Sequence[float] | None = None,
    name: str = "RelayDecoderF32",
    include_decode_result: bool = False,
    add_erasure_bit: bool = False,
    **relay_bp_args: object,
) -> DecoderSpec[RelayBPDecoder]:
    """Configure a Relay-BP decoder.

    Args:
        error_priors: The prior probability of each error mechanism.  Defaults to the error
            probabilities of a detector error model, or to a placeholder error rate for each error
            mechanism of a parity check matrix.
        name: The Relay-BP decoder class to use.  See help(relay_bp.bp) for the options.
        include_decode_result: Argument passed to relay_bp.ObservableDecoderRunner.
        add_erasure_bit: Whether to append an erasure flag to each inferred error, set to 1 when
            the inferred error does not reproduce the syndrome.
        **relay_bp_args: Options for the chosen Relay-BP decoder class, such as gamma0, pre_iter,
            num_sets, or stop_nconv.  The available options depend on that class, which rejects
            options that it does not recognize.  See, for example, help(relay_bp.RelayDecoderF32).

    Returns:
        Decoder settings to pass as ``decoder=`` to a method that decodes syndromes.  These
        settings build a qldpc.decoders.RelayBPDecoder.

    For details about Relay-BP decoders, see:

    - Documentation: https://pypi.org/project/relay-bp
    - Reference: https://arxiv.org/abs/2506.01779
    """
    return _decoder_spec(
        "relay_bp",
        get_decoder_RBP,
        error_priors=error_priors,
        name=name,
        include_decode_result=include_decode_result,
        add_erasure_bit=add_erasure_bit,
        **relay_bp_args,
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
    """Configure a decoder based on a lookup table that maps syndromes to errors.

    Args:
        max_weight: The maximum weight of the errors that the lookup table enumerates.
        error_channel: The probability of each error mechanism, used to prefer likely errors.
        penalty_func: A function that assigns a penalty to an error, overriding error_channel.
        observable_flip_matrix: A matrix mapping errors to observable flips.  If provided, each
            syndrome maps to an error that induces the most likely observable flip.
        post_select: Syndrome bits (detectors) to post-select on being trivial.
        add_erasure_bit: Whether to append an erasure flag to each inferred error, set to 1 for
            syndromes that are missing from the lookup table.
        confidence_ratio: How much more likely the most likely observable flip must be than all
            others combined for the decoder to predict it, rather than signal erasure.
        symplectic: Whether errors are symplectic ``[X|Z]`` vectors of a QuditCode.

    Returns:
        Decoder settings to pass as ``decoder=`` to a method that decodes syndromes.  These
        settings build a qldpc.decoders.LookupDecoder; see help(qldpc.decoders.LookupDecoder) for
        details about each option.
    """
    return _decoder_spec(
        "lookup_table",
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
    """Configure a decoder based on solving an integer linear program (ILP).

    Args:
        add_erasure_bit: Whether to append an erasure flag to each inferred error, set to 1 when
            the integer linear program has no solution.
        **solver_args: Options passed to cvxpy.Problem.solve, such as solver or verbose.  See
            https://www.cvxpy.org/tutorial/solvers for the options of each solver.

    Returns:
        Decoder settings to pass as ``decoder=`` to a method that decodes syndromes.  These
        settings build a qldpc.decoders.ILPDecoder, which finds a minimum-weight error exactly but
        may take exponential time.
    """
    return _decoder_spec("ilp", get_decoder_ILP, add_erasure_bit=add_erasure_bit, **solver_args)


def guf(
    *,
    max_weight: int | None = None,
    symplectic: bool = False,
    add_erasure_bit: bool = False,
) -> DecoderSpec[GUFDecoder]:
    """Configure a decoder based on a generalization of union-find (GUF).

    Args:
        max_weight: The maximum weight of the errors that the decoder considers within a cluster,
            or None to set no limit.
        symplectic: Whether errors are symplectic ``[X|Z]`` vectors of a QuditCode.
        add_erasure_bit: Whether to append an erasure flag to each inferred error, set to 1 when
            the decoder fails to find an error that reproduces the syndrome.

    Returns:
        Decoder settings to pass as ``decoder=`` to a method that decodes syndromes.  These
        settings build a qldpc.decoders.GUFDecoder, which also decodes codes over non-binary
        fields.

    Reference: https://arxiv.org/abs/2103.08049
    """
    return _decoder_spec(
        "guf",
        get_decoder_GUF,
        max_weight=max_weight,
        symplectic=symplectic,
        add_erasure_bit=add_erasure_bit,
    )


DECODER_CONSTRUCTORS: dict[str, Callable[..., ErrorDecoder]] = {
    "BF": get_decoder_BF,
    "BP_LSD": get_decoder_BP_LSD,
    "BP_OSD": get_decoder_BP_OSD,
    "GUF": get_decoder_GUF,
    "ILP": get_decoder_ILP,
    "MWPM": get_decoder_MWPM,
    "RBP": get_decoder_RBP,
    "lookup": get_decoder_lookup,
}
