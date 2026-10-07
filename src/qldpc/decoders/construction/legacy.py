# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry points for decoder construction and keyword translation."""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc._util import get_external_caller_stacklevel

from ..capabilities import (
    compiles_for_dem,
    is_prebuilt_decoder,
    is_prebuilt_observable_decoder,
)
from ..custom.guf import GUFDecoder, _get_decoder_guf
from ..custom.ilp import ILPDecoder, _get_decoder_ilp
from ..custom.lookup import LookupDecoder, _get_decoder_lookup
from ..external.ldpc import _get_decoder_bf, _get_decoder_bp_lsd, _get_decoder_bp_osd
from ..external.pymatching import _get_decoder_mwpm
from ..external.relay_bp import RelayBPDecoder, _get_decoder_rbp
from ..protocols import BatchErrorDecoder, ErrorDecoder, ObservableDecoder
from .factories import DEMDecoderFactory, MatrixDecoderFactory
from .resolution import (
    _get_error_decoder,
    _get_observable_decoder,
)
from .specs import (
    DecoderInput,
    DecoderSpec,
    ErrorDecoderConstructor,
    ErrorDecoderInput,
    PcmOrDem,
)

_InputT = TypeVar("_InputT", ErrorDecoderInput, DecoderInput)
_DecoderT = TypeVar("_DecoderT")


# Legacy keyword-based compatibility
def _call_with_flat_backend_options(
    builder: Callable[..., _DecoderT], pcm_or_dem: PcmOrDem, /, **decoder_args: Any
) -> _DecoderT:
    """Call a builder, moving deprecated flat backend options into its backend_options mapping.

    Deprecated APIs accept backend options as ordinary keyword arguments, whereas builders that
    forward backend options take them as a ``backend_options`` mapping, so that the names that they
    list are checked.  The add_erasure_bit argument is handled by the builder itself.
    """
    parameters = inspect.signature(builder).parameters
    backend_options = dict(decoder_args.pop("backend_options", None) or {})
    for name in [name for name in decoder_args if name not in parameters]:
        if name != "add_erasure_bit":
            backend_options[name] = decoder_args.pop(name)
    if backend_options:
        decoder_args["backend_options"] = backend_options
    return builder(pcm_or_dem, **decoder_args)


DECODER_CONSTRUCTORS: dict[str, Callable[..., ErrorDecoder]] = {
    "BF": functools.partial(_call_with_flat_backend_options, _get_decoder_bf),
    "BP_LSD": functools.partial(_call_with_flat_backend_options, _get_decoder_bp_lsd),
    "BP_OSD": functools.partial(_call_with_flat_backend_options, _get_decoder_bp_osd),
    "GUF": _get_decoder_guf,
    "ILP": _get_decoder_ilp,
    "MWPM": functools.partial(_call_with_flat_backend_options, _get_decoder_mwpm),
    "RBP": _get_decoder_rbp,
    "lookup": _get_decoder_lookup,
}


def get_decoder(pcm_or_dem: PcmOrDem, **decoder_args: object) -> Any:
    """Retrieve a decoder from keyword-based construction arguments."""
    warnings.warn(
        _get_deprecated_function_message("decoders.get_decoder", pcm_or_dem, decoder_args),
        DeprecationWarning,
        stacklevel=get_external_caller_stacklevel(),
    )
    return _get_legacy_decoder(pcm_or_dem, decoder_args)


def decode(
    pcm_or_dem: PcmOrDem, syndrome: npt.NDArray[np.int_], **decoder_args: object
) -> npt.NDArray[np.int_]:
    """Construct a decoder and decode one syndrome."""
    warnings.warn(
        _get_deprecated_function_message(
            "decoders.decode", pcm_or_dem, decoder_args, decodes_syndrome=True
        ),
        DeprecationWarning,
        stacklevel=get_external_caller_stacklevel(),
    )
    return _get_legacy_decoder(pcm_or_dem, decoder_args).decode(syndrome)


def get_error_decoder(pcm_or_dem: PcmOrDem, *, decoder: ErrorDecoderInput = None) -> ErrorDecoder:
    """Build or retrieve an error decoder."""
    if isinstance(decoder, DecoderSpec):
        replacement = f"{decoder!r}.build(pcm_or_dem)"
    elif decoder is None:
        replacement = f"decoders.{_get_legacy_helper_name(pcm_or_dem, {})}().build(pcm_or_dem)"
    elif is_prebuilt_decoder(decoder):
        replacement = "the prebuilt decoder directly"
    elif callable(decoder):
        replacement = f"{_get_callable_name(decoder)}(pcm_or_dem)"
    else:
        replacement = "a decoder specification such as decoders.bp_osd(...).build(pcm_or_dem)"
    _warn_deprecated("decoders.get_error_decoder", replacement)
    return _get_error_decoder(pcm_or_dem, decoder=decoder)


def get_observable_decoder(
    dem: stim.DetectorErrorModel, *, decoder: DecoderInput = None
) -> ObservableDecoder:
    """Build or retrieve an observable decoder."""
    expression, note = _get_observable_decoder_expression(decoder)
    replacement = "the prebuilt observable decoder directly" if expression is None else expression
    _warn_deprecated("decoders.get_observable_decoder", replacement + note)
    return _get_observable_decoder(dem, decoder=decoder)


def decode_observables(
    dem: stim.DetectorErrorModel,
    syndrome: npt.NDArray[np.int_],
    *,
    decoder: DecoderInput = None,
) -> npt.NDArray[np.int_]:
    """Predict the observable flips of one syndrome."""
    expression, note = _get_observable_decoder_expression(decoder)
    replacement = (
        "the decode_observables method of the prebuilt observable decoder"
        if expression is None
        else f"{expression}.decode_observables(syndrome)"
    )
    _warn_deprecated("decoders.decode_observables", replacement + note)
    return _get_observable_decoder(dem, decoder=decoder).decode_observables(syndrome)


# Deprecated per-decoder builders


def get_decoder_bp_osd(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> ErrorDecoder:
    """Build a BP+OSD decoder."""
    _warn_deprecated_builder("get_decoder_bp_osd", "bp_osd")
    return _call_with_flat_backend_options(_get_decoder_bp_osd, pcm_or_dem, **decoder_args)


def get_decoder_bp_lsd(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> ErrorDecoder:
    """Build a BP+LSD decoder."""
    _warn_deprecated_builder("get_decoder_bp_lsd", "bp_lsd")
    return _call_with_flat_backend_options(_get_decoder_bp_lsd, pcm_or_dem, **decoder_args)


def get_decoder_bf(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> ErrorDecoder:
    """Build a belief-find decoder."""
    _warn_deprecated_builder("get_decoder_bf", "bf")
    return _call_with_flat_backend_options(_get_decoder_bf, pcm_or_dem, **decoder_args)


def get_decoder_guf(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> GUFDecoder:
    """Build a GUF decoder."""
    _warn_deprecated_builder("get_decoder_guf", "guf")
    return _get_decoder_guf(pcm_or_dem, **decoder_args)


def get_decoder_ilp(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> ILPDecoder:
    """Build an ILP decoder."""
    _warn_deprecated_builder("get_decoder_ilp", "ilp")
    return _get_decoder_ilp(pcm_or_dem, **decoder_args)


def get_decoder_lookup(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> LookupDecoder:
    """Build a lookup decoder."""
    _warn_deprecated_builder("get_decoder_lookup", "lookup")
    return _get_decoder_lookup(pcm_or_dem, **decoder_args)


def get_decoder_mwpm(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> BatchErrorDecoder:
    """Build an MWPM decoder."""
    _warn_deprecated_builder("get_decoder_mwpm", "mwpm")
    return _call_with_flat_backend_options(_get_decoder_mwpm, pcm_or_dem, **decoder_args)


def get_decoder_rbp(
    pcm_or_dem: PcmOrDem,
    error_priors: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: Any,
) -> RelayBPDecoder:
    """Build a Relay-BP decoder."""
    _warn_deprecated(
        "decoders.get_decoder_rbp",
        "decoders.relay_bp(...).build(pcm_or_dem) or decoders.min_sum_bp(...).build(pcm_or_dem)",
    )
    return _get_decoder_rbp(pcm_or_dem, error_priors, **decoder_args)


# Deprecated resolution functions


def resolve_decoder(
    pcm_or_dem: PcmOrDem,
    decoder: ErrorDecoderInput,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> ErrorDecoder:
    """Resolve an error decoder input with keyword-based construction arguments.

    This serves methods that accept keyword arguments next to decoder=.  The decoder input may be a
    decoder specification such as ``decoders.bp_osd(...)``, a constructor that builds an error
    decoder from pcm_or_dem, a prebuilt error decoder, or None to select the default decoder: GUF
    for a nonbinary FieldArray, and BP+OSD otherwise.
    """
    decoder_input = _merge_legacy_decoder_args(
        pcm_or_dem, decoder, decoder_args, warn_deprecated=warn_deprecated
    )
    return _get_error_decoder(pcm_or_dem, decoder=decoder_input)


def resolve_observable_decoder(
    dem: stim.DetectorErrorModel,
    decoder: DecoderInput,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> ObservableDecoder:
    """Resolve an observable decoder input with keyword-based construction arguments.

    This serves methods that accept keyword arguments next to decoder=.  Decoder specifications
    build a native observable decoder where they support one, and otherwise an error decoder; an
    observable-decoder compiler such as a ``decoders.SinterDecoder`` is compiled for dem; a
    constructor may build an error decoder or an observable decoder from dem; and a prebuilt error
    decoder or observable decoder is used as is.  An error decoder is wrapped so that the observable
    flips of the errors that it infers become its predictions.
    """
    decoder_input = _merge_legacy_decoder_args(
        dem, decoder, decoder_args, warn_deprecated=warn_deprecated
    )
    return _get_observable_decoder(dem, decoder=decoder_input)


def reject_removed_decoder_args(decoder_args: Mapping[str, object]) -> None:
    """Reject the removed static_decoder argument outside of get_decoder and decode."""
    if "static_decoder" in decoder_args:
        raise TypeError(
            "The static_decoder argument has been removed; pass a prebuilt decoder as decoder="
            " instead"
        )


_MATRIX_OBSERVABLE_LOOKUP_ADVICE = (
    "use qldpc.decoders.custom.lookup.ObservableLookupDecoder(matrix, max_weight=...,"
    " observable_flip_matrix=..., error_channel=...).decode_observables(syndrome) instead"
)


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
        advice = (
            "use decoders.lookup(max_weight=...).build_observable_decoder(dem) instead"
            if isinstance(pcm_or_dem, stim.DetectorErrorModel)
            else (
                "pass decoder=decoders.lookup(max_weight=...) instead, which predicts observable"
                " flips natively"
                if pcm_or_dem is None
                else _MATRIX_OBSERVABLE_LOOKUP_ADVICE
            )
        )
        return f"predict_observable_flips=True is deprecated; {advice}"
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


# Legacy translation and message helpers


_LEGACY_HELPER_NAMES = {
    "BF": "bf",
    "BP_LSD": "bp_lsd",
    "BP_OSD": "bp_osd",
    "GUF": "guf",
    "ILP": "ilp",
    "MWPM": "mwpm",
    "RBP": "relay_bp",
    "lookup": "lookup",
}


def _get_legacy_decoder(pcm_or_dem: PcmOrDem, decoder_args: Mapping[str, object]) -> Any:
    """Build a decoder with the deprecated keyword-based API."""
    static_decoder = decoder_args.get("static_decoder")
    if decoder_args.get("decoder_constructor") is None and static_decoder is not None:
        if len(decoder_args) > 1:
            raise ValueError("If passed a static decoder, we cannot process decoding arguments")
        return static_decoder
    return _get_legacy_decoder_input(pcm_or_dem, decoder_args)(pcm_or_dem)


def _get_legacy_decoder_input(
    pcm_or_dem: PcmOrDem, decoder_args: Mapping[str, object]
) -> ErrorDecoderConstructor:
    """Translate deprecated decoder arguments into a decoder constructor.

    Free-form decoder options are passed through to the selected builder unchecked, as they were in
    qLDPC 0.3.3, so they become a constructor rather than a typed decoder specification.
    """
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
    return functools.partial(DECODER_CONSTRUCTORS[decoder_name], **decoder_args)


def _merge_legacy_decoder_args(
    pcm_or_dem: PcmOrDem,
    decoder: _InputT,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> _InputT | ErrorDecoderConstructor:
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


def _get_deprecated_function_message(
    function_name: str,
    pcm_or_dem: PcmOrDem,
    decoder_args: Mapping[str, object],
    *,
    decodes_syndrome: bool = False,
) -> str:
    """Describe the replacement for deprecated get_decoder or decode."""
    if (
        decoder_args.get("predict_observable_flips")
        and decoder_args.get("decoder_constructor") is None
        and decoder_args.get("static_decoder") is None
    ):
        if isinstance(pcm_or_dem, stim.DetectorErrorModel):
            replacement = "decoders.lookup(max_weight=...).build_observable_decoder(dem)"
            if decodes_syndrome:
                replacement += ".decode_observables(syndrome)"
            return f"{function_name} is deprecated; use {replacement} instead"
        return f"{function_name} is deprecated; {_MATRIX_OBSERVABLE_LOOKUP_ADVICE}"
    if (decoder_constructor := decoder_args.get("decoder_constructor")) is not None:
        other_args = ", ..." if len(decoder_args) > 1 else ""
        replacement = f"{_get_constructor_name(decoder_constructor)}(pcm_or_dem{other_args})"
    elif decoder_args.get("static_decoder") is not None:
        replacement = "static_decoder"
    elif decoder_args:
        helper_name = _get_legacy_helper_name(pcm_or_dem, decoder_args)
        replacement = f"decoders.{helper_name}(...).build(pcm_or_dem)"
    else:
        replacement = f"decoders.{_get_legacy_helper_name(pcm_or_dem, {})}().build(pcm_or_dem)"
    if decodes_syndrome:
        replacement += ".decode(syndrome)"
    return f"{function_name} is deprecated; use {replacement} instead"


def _get_observable_decoder_expression(decoder: DecoderInput) -> tuple[str | None, str]:
    """Return an expression that builds the observable decoder for an input to a deprecated API.

    The expression refers to the detector error model as ``dem``.  None indicates a prebuilt decoder
    that predicts observable flips, and is therefore used as is.  The accompanying note, which may
    be empty, qualifies the expression when it is unknown what kind of decoder a constructor builds.
    """
    if isinstance(decoder, DecoderSpec):
        return f"{decoder!r}.build_observable_decoder(dem)", ""
    if decoder is None:
        return "decoders.bp_osd().build_observable_decoder(dem)", ""
    if compiles_for_dem(decoder):
        return "decoder.compile_decoder_for_dem(dem)", ""
    if is_prebuilt_decoder(decoder):
        if isinstance(decoder, ObservableDecoder) or is_prebuilt_observable_decoder(decoder):
            return None, ""
        return "decoders.ErrorsToObservablesDecoder(decoder, dem)", ""
    if isinstance(decoder, DEMDecoderFactory):
        return "decoder.build(dem)", ""
    if isinstance(decoder, MatrixDecoderFactory):
        return "a factory that accepts a detector error model", ""
    if callable(decoder):
        name = _get_callable_name(decoder)
        error_decoder_expression = f"decoders.ErrorsToObservablesDecoder({name}(dem), dem)"
        return f"{name}(dem)", f" (if it builds an error decoder, use {error_decoder_expression})"
    return "a decoder specification such as decoders.mwpm(...).build_observable_decoder(dem)", ""


def _get_callable_name(decoder: Callable[..., object]) -> str:
    """Return the name of a decoder constructor for a migration message."""
    return getattr(decoder, "__name__", "decoder_constructor")


def _warn_deprecated_builder(function_name: str, helper_name: str) -> None:
    """Warn that a deprecated builder is replaced by a decoder specification."""
    _warn_deprecated(f"decoders.{function_name}", f"decoders.{helper_name}(...).build(pcm_or_dem)")


def _warn_deprecated(function_name: str, replacement: str) -> None:
    """Warn that a deprecated decoder function has a replacement."""
    warnings.warn(
        f"{function_name} is deprecated; use {replacement} instead",
        DeprecationWarning,
        stacklevel=get_external_caller_stacklevel(),
    )
