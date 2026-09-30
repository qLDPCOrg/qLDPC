# SPDX-License-Identifier: Apache-2.0

"""Methods to decode, or retrieve various decoders."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Generic, Literal, ParamSpec, Protocol, TypeAlias, TypeVar, cast

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
    ObservableDecoder,
    RelayBPDecoder,
)
from .dems import DetectorErrorModelArrays
from .lookup import LookupDecoder, ObservableLookupDecoder

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
    _observable_builder: Callable[..., ObservableDecoder] | None = None

    def build(self, pcm_or_dem: PcmOrDem) -> _DecoderT_co:
        """Build an error decoder for a parity-check matrix or detector error model."""
        return self._builder(pcm_or_dem, **dict(self._options))

    @property
    def predicts_observables_natively(self) -> bool:
        """Whether the configured decoder can predict observable flips without inferring errors."""
        return self._observable_builder is not None

    def build_observable_decoder(self, dem: stim.DetectorErrorModel) -> ObservableDecoder:
        """Build a decoder that predicts the observable flips of a detector error model.

        If the configured decoder can predict observable flips natively (see
        predicts_observables_natively), build it in that mode.  Otherwise, build an error decoder,
        and convert the errors that it infers into observable flips.
        """
        if self._observable_builder is not None:
            return _validate_observable_decoder(
                self._observable_builder(dem, **dict(self._options)), "A decoder spec"
            )
        return _ErrorsToObservables(_validate_error_decoder(self.build(dem), "A decoder spec"), dem)

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


class ObservableDecoderConstructor(Protocol):
    """Callable that builds an observable decoder from a detector error model."""

    def __call__(self, dem: stim.DetectorErrorModel, /) -> ObservableDecoder:
        """Build an observable decoder."""


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

DeferredObservableDecoderInput: TypeAlias = (
    DecoderSpec[Any] | ErrorDecoderConstructor | ObservableDecoderConstructor | None
)
"""Decoder settings, a constructor of an error or observable decoder, or None for the default.

This is ObservableDecoderInput without a prebuilt decoder.  It is accepted by methods that predict
the observable flips of a detector error model that they construct themselves, such as a
SinterDecoder, and so must build the decoder.
"""

ObservableDecoderInput: TypeAlias = (
    DecoderSpec[Any]
    | ErrorDecoderConstructor
    | ObservableDecoderConstructor
    | ErrorDecoder
    | ObservableDecoder
    | None
)
"""Decoder settings, a constructor or prebuilt instance of an error or observable decoder, or None.

This is accepted by methods that predict the observable flips of a detector error model that the
caller knows.  An error decoder is used by converting the errors that it infers into observable
flips.
"""

_OBSERVABLE_DECODER_ADVICE = (
    "Pass error-decoder settings such as decoders.bp_osd(...), or pass the observable decoder where"
    " one is accepted, such as to decoders.get_observable_decoder or decoders.SinterDecoder"
)


def _decoder_spec(
    helper_name: str,
    builder: Callable[..., _Decoder],
    observable_builder: Callable[..., ObservableDecoder] | None = None,
    /,
    **options: object,
) -> DecoderSpec[_Decoder]:
    """Store deferred decoder construction options."""
    return DecoderSpec(helper_name, builder, tuple(options.items()), observable_builder)


def _is_prebuilt_decoder(decoder: object) -> bool:
    """Whether a decoder input is an already-built decoder, rather than settings or a constructor.

    This classification matches the order in which _resolve_decoder interprets a decoder input.
    """
    return (
        decoder is not None
        and not isinstance(decoder, (DecoderSpec, type))
        and (hasattr(decoder, "decode") or hasattr(decoder, "decode_observables"))
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
        if _predicts_only_observables(decoder):
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


def _predicts_only_observables(decoder: object) -> bool:
    """Whether a prebuilt decoder predicts observable flips, and cannot infer errors.

    An object can infer errors if it has a decode method, unless it declares (with the private
    _decode_returns_observables attribute) that its decode method returns observable flips, as the
    deprecated LookupDecoder(..., predict_observable_flips=True) does.
    """
    if getattr(decoder, "_decode_returns_observables", False):
        return True
    return (
        not isinstance(decoder, (DecoderSpec, type))
        and hasattr(decoder, "decode_observables")
        and not hasattr(decoder, "decode")
    )


def _validate_error_decoder(decoder: object, source: str) -> ErrorDecoder:
    """Validate and type-narrow an object expected to decode syndromes to errors."""
    if _predicts_only_observables(decoder):
        raise TypeError(
            f"{source} predicts observable flips rather than errors.  " + _OBSERVABLE_DECODER_ADVICE
        )
    if not hasattr(decoder, "decode") or not callable(decoder.decode):
        raise TypeError(f"{source} must provide a callable decode method")
    return cast(ErrorDecoder, decoder)


def _validate_observable_decoder(decoder: object, source: str) -> ObservableDecoder:
    """Validate and type-narrow an object expected to decode syndromes to observable flips."""
    if not hasattr(decoder, "decode_observables") or not callable(decoder.decode_observables):
        raise TypeError(f"{source} must provide a callable decode_observables method")
    return cast(ObservableDecoder, decoder)


class _ExpandedDecoder(ErrorDecoder):
    """Wrapper for a decoder, to map decoded errors in a simplified DEM to errors in the full DEM.

    A decoder that merges equivalent error mechanisms infers fewer errors than the DEM it was built
    for declares.  That happens when a DEM restricted to a window leaves equivalent mechanisms
    behind, and it happens when a caller asks for a model whose equivalent mechanisms are left
    unmerged.  This wrapper expands decoded errors in the simplified DEM to equivalent errors in the
    original DEM, and passes any erasure bit through as the last entry.
    """

    def __init__(self, decoder: ErrorDecoder, dem: stim.DetectorErrorModel) -> None:
        self._decoder = decoder
        self.has_erasure_bit = bool(getattr(decoder, "has_erasure_bit", False))

        original_errors = DetectorErrorModelArrays.get_circuit_errors(dem)
        simplified_errors = DetectorErrorModelArrays.get_merged_circuit_errors(original_errors)
        self._num_original_errors = len(original_errors)

        # map each detector/observable signature to an original error index
        signature_to_original_error_index = {
            signature: original_error_index
            for original_error_index, (_, signature) in enumerate(original_errors)
        }

        # locate each simplified error among the original errors
        self._simplified_to_original_index = np.array(
            [signature_to_original_error_index[signature] for _, signature in simplified_errors],
            dtype=np.intp,
        )

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        simplified_error = self._decoder.decode(syndrome)
        original_error = np.zeros(
            self._num_original_errors + self.has_erasure_bit, dtype=syndrome.dtype
        )
        if self.has_erasure_bit:
            original_error[-1] = simplified_error[-1]
            simplified_error = simplified_error[:-1]
        original_error[self._simplified_to_original_index] = simplified_error
        return np.asarray(original_error, dtype=syndrome.dtype)

    def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        simplified_errors = (
            self._decoder.decode_batch(syndromes)
            if hasattr(self._decoder, "decode_batch")
            else np.array([self._decoder.decode(syndrome) for syndrome in syndromes])
        )
        original_errors = np.zeros(
            (len(syndromes), self._num_original_errors + self.has_erasure_bit),
            dtype=syndromes.dtype,
        )
        if self.has_erasure_bit:
            original_errors[:, -1] = simplified_errors[:, -1]
            simplified_errors = simplified_errors[:, :-1]
        original_errors[:, self._simplified_to_original_index] = simplified_errors
        return original_errors


def _reject_observable_output(decoder: object) -> None:
    """Reject a decoder whose .decode method returns observable flips rather than errors."""
    if getattr(decoder, "_decode_returns_observables", False):
        raise ValueError(
            "The decoder returns observable flips rather than errors from its decode method, so it"
            " must predict errors rather than observable flips.  " + _OBSERVABLE_DECODER_ADVICE
        )


def _match_error_decoder_to_dem(
    decoder: ErrorDecoder, dem: stim.DetectorErrorModel
) -> ErrorDecoder:
    """Check that an error decoder infers errors of a detector error model, and align them with it.

    A decoder that merges equivalent error mechanisms infers fewer errors than the detector error
    model declares, so wrap it to map its errors back onto the model.  An erasure bit is not an
    error mechanism, so it does not count toward the widths compared here.  An inferred "error" of
    any other width is rejected, since it cannot be read as an error of the model.  This happens,
    for example, if the decoder predicts observable flips rather than errors.
    """
    _reject_observable_output(decoder)
    num_erasure_bits = int(getattr(decoder, "has_erasure_bit", False))
    test_error = decoder.decode(np.zeros(dem.num_detectors, dtype=int))
    num_inferred_errors = len(test_error) - num_erasure_bits
    circuit_errors = DetectorErrorModelArrays.get_circuit_errors(dem)
    num_merged_errors = len(DetectorErrorModelArrays.get_merged_circuit_errors(circuit_errors))
    if num_inferred_errors == len(circuit_errors):
        return decoder
    if num_inferred_errors == num_merged_errors:
        return _ExpandedDecoder(decoder, dem)
    raise ValueError(
        f"An error decoder inferred an error of length {num_inferred_errors} for a detector error"
        f" model with {len(circuit_errors)} error mechanisms ({num_merged_errors} after merging"
        " equivalent mechanisms).  If the decoder predicts observable flips rather than errors,"
        " give it a decode_observables method, and pass it where an observable decoder is accepted"
    )


class _ErrorsToObservables(ObservableDecoder):
    """Observable decoder that converts errors that an error decoder infers into observable flips.

    If the error decoder signals erasure, its erasure bit is appended to each prediction.
    """

    def __init__(self, error_decoder: ErrorDecoder, dem: stim.DetectorErrorModel) -> None:
        self.error_decoder = _match_error_decoder_to_dem(error_decoder, dem)
        self.has_erasure_bit = bool(getattr(error_decoder, "has_erasure_bit", False))
        self.observable_flip_matrix = DetectorErrorModelArrays(
            dem, simplify=False
        ).observable_flip_matrix

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return predicted observable flips."""
        return self.decode_observables_batch(np.asarray(syndrome)[None, :])[0]

    def decode_observables_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes, one per row, and return predicted observable flips."""
        errors = np.asarray(
            self.error_decoder.decode_batch(syndromes)
            if hasattr(self.error_decoder, "decode_batch")
            else [self.error_decoder.decode(syndrome) for syndrome in syndromes]
        )
        erasure_bits = errors[:, -1:] if self.has_erasure_bit else errors[:, :0]
        errors = errors[:, : errors.shape[1] - erasure_bits.shape[1]]
        flips = np.asarray(errors @ self.observable_flip_matrix.T) % 2
        return np.hstack([flips, erasure_bits]).astype(np.uint8)


def decode_observables(
    dem: stim.DetectorErrorModel,
    syndrome: npt.NDArray[np.int_],
    *,
    decoder: ObservableDecoderInput = None,
) -> npt.NDArray[np.int_]:
    """Construct a decoder and predict the observable flips of one syndrome.

    Args:
        dem: A detector error model.
        syndrome: The syndrome (detection events) to decode.
        decoder: Decoder settings, a decoder constructor, a prebuilt decoder, or None to select the
            default decoder.  See help(qldpc.decoders.get_observable_decoder).

    Returns:
        The predicted observable flips.
    """
    return get_observable_decoder(dem, decoder=decoder).decode_observables(syndrome)


def get_observable_decoder(
    dem: stim.DetectorErrorModel, *, decoder: ObservableDecoderInput = None
) -> ObservableDecoder:
    """Build or retrieve a decoder that predicts the observable flips of a detector error model.

    Args:
        dem: A detector error model.
        decoder: One of the following, or None (the default) to select the default decoder:

            - Decoder settings from a helper such as ``decoders.mwpm(...)``.  If the configured
              decoder can predict observable flips natively (see
              ``DecoderSpec.predicts_observables_natively``), it is built in that mode.
            - A constructor that builds an error decoder or an observable decoder from dem.
            - A prebuilt error decoder or observable decoder for dem.

            An error decoder is used by converting the errors that it infers into observable flips.

    Returns:
        An observable decoder for dem.

    If decoder is None, this method defaults to BP+OSD, whose inferred errors are converted into
    observable flips.
    """
    return _get_observable_decoder(dem, decoder, {})


def _get_observable_decoder(
    dem: stim.DetectorErrorModel,
    decoder: ObservableDecoderInput,
    decoder_args: dict[str, object],
    *,
    warn_deprecated: bool = True,
) -> ObservableDecoder:
    """Build an observable decoder, optionally suppressing a warning emitted by an outer API."""
    if isinstance(decoder, DecoderSpec) and not decoder_args:
        return decoder.build_observable_decoder(dem)

    # build or retrieve a decoder, which may predict observable flips or infer errors
    built_decoder: object = decoder
    if decoder_args or decoder is None:
        built_decoder = _resolve_decoder(
            dem, cast(ErrorDecoderInput, decoder), decoder_args, warn_deprecated=warn_deprecated
        )
    elif isinstance(decoder, type) or not (
        hasattr(decoder, "decode") or hasattr(decoder, "decode_observables")
    ):
        if not callable(decoder):
            raise TypeError(
                "decoder must be a DecoderSpec, a decoder constructor, a prebuilt decoder, or None"
            )
        built_decoder = decoder(dem)

    if hasattr(built_decoder, "decode_observables"):
        return _validate_observable_decoder(built_decoder, "A decoder")
    _reject_observable_output(built_decoder)
    return _ErrorsToObservables(_validate_error_decoder(built_decoder, "A decoder"), dem)


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

    The decoder returned by this method maps a syndrome to an error, even if built from a detector
    error model.  To predict the observable flips of a detector error model instead, pass
    ``decoders.mwpm(...)`` to qldpc.decoders.get_observable_decoder, which builds a matching decoder
    that predicts observable flips natively.
    """
    return _build_matching(
        pcm_or_dem,
        decompose_errors=decompose_errors,
        ignore_non_graphlike_errors=ignore_non_graphlike_errors,
        predict_observables=False,
        **decoder_args,
    )


def _build_matching(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    decompose_errors: bool,
    ignore_non_graphlike_errors: bool,
    predict_observables: bool,
    **decoder_args: object,
) -> Any:
    """Build a pymatching.Matching, which predicts errors or (from a DEM) observable flips."""
    # identify parity check matrix and error probabilities
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        dem_arrays = DetectorErrorModelArrays(pcm_or_dem, decompose_errors=decompose_errors)
        pcm = dem_arrays.detector_flip_matrix
        if decoder_args.get("weights") is not None:
            raise ValueError("Cannot set error weights when initializing a MWPM decoder from a DEM")
        decoder_args["weights"] = np.log((1 - dem_arrays.error_probs) / dem_arrays.error_probs)
        if predict_observables:
            # the "faults" of a matching decoder are the observables that each error flips
            decoder_args["faults_matrix"] = dem_arrays.observable_flip_matrix
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


class _MatchingObservableDecoder(ObservableDecoder):
    """Observable decoder based on a pymatching.Matching that predicts observable flips."""

    def __init__(self, matching: Any) -> None:
        self.matching = matching

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return predicted observable flips."""
        return np.asarray(self.matching.decode(syndrome), dtype=np.uint8)

    def decode_observables_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes, one per row, and return predicted observable flips."""
        return np.asarray(self.matching.decode_batch(syndromes), dtype=np.uint8)


def _get_observable_decoder_MWPM(
    dem: stim.DetectorErrorModel,
    *,
    decompose_errors: bool = False,
    ignore_non_graphlike_errors: bool = False,
    **decoder_args: object,
) -> _MatchingObservableDecoder:
    """Build a matching decoder that predicts the observable flips of a detector error model."""
    return _MatchingObservableDecoder(
        _build_matching(
            dem,
            decompose_errors=decompose_errors,
            ignore_non_graphlike_errors=ignore_non_graphlike_errors,
            predict_observables=True,
            **decoder_args,
        )
    )


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


def _get_relay_decoder(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    decoder_class_prefix: str,
    precision: str,
    **decoder_args: Any,
) -> RelayBPDecoder:
    """Build a RelayBPDecoder from the relay_bp class with the given name prefix and precision."""
    return get_decoder_RBP(pcm_or_dem, name=f"{decoder_class_prefix}{precision}", **decoder_args)


def _get_relay_bp_decoder(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: Any
) -> RelayBPDecoder:
    """Build a RelayBPDecoder that runs Relay-BP (a relay_bp.RelayDecoder* class)."""
    return _get_relay_decoder(pcm_or_dem, decoder_class_prefix="RelayDecoder", **decoder_args)


def _get_min_sum_bp_decoder(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: Any
) -> RelayBPDecoder:
    """Build a RelayBPDecoder that runs min-sum BP (a relay_bp.MinSumBPDecoder* class)."""
    return _get_relay_decoder(pcm_or_dem, decoder_class_prefix="MinSumBPDecoder", **decoder_args)


@_erasure_bit_support(True)
def get_decoder_lookup(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> LookupDecoder:
    """Decoder based on a lookup table that maps errors to syndromes."""
    return LookupDecoder(pcm_or_dem, **decoder_args)  # type:ignore[arg-type]


def _get_observable_lookup_decoder(
    dem: stim.DetectorErrorModel, **decoder_args: object
) -> ObservableDecoder:
    """Build a lookup table that maps syndromes of a detector error model to observable flips.

    An ObservableLookupDecoder groups errors by the observables that they flip, which requires
    observables.  Without them, every error flips no observables, so fall back to an error lookup
    table, whose inferred errors are converted into (empty) observable flips.
    """
    if dem.num_observables == 0:
        return _ErrorsToObservables(get_decoder_lookup(dem, **decoder_args), dem)
    return ObservableLookupDecoder(dem, **decoder_args)  # type:ignore[arg-type]


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
        Decoder settings to pass as ``decoder=``, which build a pymatching.Matching that cannot
        signal erasure.  Methods that decode syndromes to errors build a decoder that infers
        errors.  Methods that predict the observable flips of a detector error model, such as a
        SinterDecoder, build a decoder that predicts observable flips natively.

    For details about the MWPM decoder, see:

    - help(pymatching.Matching.from_check_matrix)
    - Documentation: https://pymatching.readthedocs.io
    - Reference: https://arxiv.org/abs/2303.15933
    """
    return _decoder_spec(
        "mwpm",
        get_decoder_MWPM,
        _get_observable_decoder_MWPM,
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
    """Configure a Relay-BP decoder.

    Relay-BP runs a sequence ("relay") of ensembles of belief propagation with memory, and keeps the
    best of the solutions that they find.  See the relay-bp documentation for guidance on tuning
    these options, several of which have a large effect on decoding performance and runtime.

    Args:
        precision: The numerical type of messages, which selects the relay_bp.RelayDecoderF32,
            RelayDecoderF64, RelayDecoderI32, or RelayDecoderI64 class.
        error_priors: The prior probability of each error mechanism.  Defaults to the error
            probabilities of a detector error model, or to a placeholder error rate for each error
            mechanism of a parity check matrix.
        observable_error_matrix: A binary matrix whose rows specify which error mechanisms flip
            which observables, to predict the observable flips of errors decoded from a parity check
            matrix.  A detector error model provides this matrix itself.
        include_decode_result: Argument passed to relay_bp.ObservableDecoderRunner.
        add_erasure_bit: Whether to append an erasure flag to each prediction, set to 1 when the
            inferred error does not reproduce the syndrome.
        alpha: The check-to-variable message scaling factor, or None for relay-bp's default.
        alpha_iteration_scaling_factor: The rate at which the scaling factor ramps up with
            iterations.
        gamma0: The memory weight of the first ensemble.
        data_scale_value: A scale factor for messages, to emulate a lower message precision.
        max_data_value: A cap on the magnitude of messages, to emulate a lower message precision.
        pre_iter: The maximum number of belief-propagation iterations of the first ensemble.
        num_sets: The number of ensembles to run after the first.
        set_max_iter: The maximum number of belief-propagation iterations per ensemble.
        gamma_dist_interval: The interval from which to draw random memory weights for ensembles
            after the first, or None for relay-bp's default.
        explicit_gammas: Memory weights to use instead of random ones, as an array of shape
            ``(num_sets, num_errors)``.
        stop_nconv: The number of solutions to find before stopping.
        stopping_criterion: The criterion for stopping, or None for relay-bp's default.
        logging: Whether relay-bp logs its progress.
        seed: The seed for the random memory weights.

    Returns:
        Decoder settings to pass as ``decoder=``, which build a qldpc.decoders.RelayBPDecoder.  A
        RelayBPDecoder can both infer errors and predict the observable flips of a detector error
        model, so it serves methods that need either.

    For details about Relay-BP decoders, see:

    - Documentation: https://pypi.org/project/relay-bp
    - Reference: https://arxiv.org/abs/2506.01779
    """
    # options that relay-bp does not expose defaults for are only passed on if they are provided
    optional_args = {
        "gamma_dist_interval": gamma_dist_interval,
        "stopping_criterion": stopping_criterion,
    }
    return _decoder_spec(
        "relay_bp",
        _get_relay_bp_decoder,
        _get_relay_bp_decoder,
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
    """Configure a min-sum belief-propagation decoder from the relay-bp package.

    This is the belief-propagation decoder that Relay-BP builds on, optionally with memory.

    Args:
        precision: The numerical type of messages, which selects the relay_bp.MinSumBPDecoderF32,
            MinSumBPDecoderF64, MinSumBPDecoderI8, MinSumBPDecoderI16, MinSumBPDecoderI32,
            MinSumBPDecoderI64, or (fixed-point) MinSumBPDecoderFixed class.
        error_priors: The prior probability of each error mechanism.  Defaults to the error
            probabilities of a detector error model, or to a placeholder error rate for each error
            mechanism of a parity check matrix.
        observable_error_matrix: A binary matrix whose rows specify which error mechanisms flip
            which observables, to predict the observable flips of errors decoded from a parity check
            matrix.  A detector error model provides this matrix itself.
        include_decode_result: Argument passed to relay_bp.ObservableDecoderRunner.
        add_erasure_bit: Whether to append an erasure flag to each prediction, set to 1 when the
            inferred error does not reproduce the syndrome.
        max_iter: The maximum number of belief-propagation iterations.
        alpha: The check-to-variable message scaling factor, or None for relay-bp's default.
        alpha_iteration_scaling_factor: The rate at which the scaling factor ramps up with
            iterations.
        gamma0: The memory weight, or None for belief propagation without memory.
        data_scale_value: A scale factor for messages, to emulate a lower message precision.
        max_data_value: A cap on the magnitude of messages, to emulate a lower message precision.
        int_bits: The number of integer bits of fixed-point messages (precision="Fixed").
        frac_bits: The number of fractional bits of fixed-point messages (precision="Fixed").

    Returns:
        Decoder settings to pass as ``decoder=``, which build a qldpc.decoders.RelayBPDecoder.  A
        RelayBPDecoder can both infer errors and predict the observable flips of a detector error
        model, so it serves methods that need either.

    For details, see the relay-bp documentation: https://pypi.org/project/relay-bp
    """
    return _decoder_spec(
        "min_sum_bp",
        _get_min_sum_bp_decoder,
        _get_min_sum_bp_decoder,
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
        Decoder settings to pass as ``decoder=``.  Methods that decode syndromes to errors build a
        qldpc.decoders.LookupDecoder.  Methods that predict the observable flips of a detector error
        model, such as a SinterDecoder, build a qldpc.decoders.ObservableLookupDecoder, which maps
        each syndrome directly to its most likely observable flip.  See
        help(qldpc.decoders.LookupDecoder) for details about each option.
    """
    return _decoder_spec(
        "lookup_table",
        get_decoder_lookup,
        _get_observable_lookup_decoder,
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
