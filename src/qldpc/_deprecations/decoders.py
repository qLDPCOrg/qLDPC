# SPDX-License-Identifier: Apache-2.0

"""Deprecated decoder construction: package-root functions, keyword translation, and options.

The registry in qldpc._deprecations serves the deprecated functions of this module at their
package-root paths in qldpc.decoders.  Decoder modules import this module when they are imported,
so this module imports qldpc.decoders only within the functions that use it, never at import time.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc import _deprecations
from qldpc._utils import get_external_caller_stacklevel

if TYPE_CHECKING:
    from qldpc.decoders.construction.specs import (
        DecoderInput,
        ErrorDecoderConstructor,
        ErrorDecoderInput,
        PcmOrDem,
    )
    from qldpc.decoders.custom.guf import GUFDecoder
    from qldpc.decoders.custom.ilp import ILPDecoder
    from qldpc.decoders.custom.lookup import ErrorLookupDecoder
    from qldpc.decoders.external.relay_bp import RelayBPDecoder
    from qldpc.decoders.protocols import BatchErrorDecoder, ErrorDecoder, ObservableDecoder

_DecoderT = TypeVar("_DecoderT")
_ErrorRateT = TypeVar("_ErrorRateT")

# the NAME of each with_<NAME> keyword, mapped to the specification helper that replaces it
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

_MATRIX_OBSERVABLE_LOOKUP_ADVICE = (
    "use qldpc.decoders.custom.lookup.ObservableLookupDecoder(matrix, max_weight=...,"
    " observable_flip_matrix=..., error_channel=...).decode_observables(syndrome) instead"
)


####################################################################################################
# deprecated package-root functions


def get_decoder(pcm_or_dem: PcmOrDem, **decoder_args: object) -> Any:
    """Retrieve a decoder from keyword-based construction arguments."""
    _deprecations.warn_deprecated(
        _get_deprecated_function_message("decoders.get_decoder", pcm_or_dem, decoder_args),
        stacklevel=get_external_caller_stacklevel(),
    )
    return _get_legacy_decoder(pcm_or_dem, decoder_args)


def decode(
    pcm_or_dem: PcmOrDem, syndrome: npt.NDArray[np.int_], **decoder_args: object
) -> npt.NDArray[np.int_]:
    """Construct a decoder and decode one syndrome."""
    _deprecations.warn_deprecated(
        _get_deprecated_function_message(
            "decoders.decode", pcm_or_dem, decoder_args, decodes_syndrome=True
        ),
        stacklevel=get_external_caller_stacklevel(),
    )
    return _get_legacy_decoder(pcm_or_dem, decoder_args).decode(syndrome)


def get_error_decoder(pcm_or_dem: PcmOrDem, *, decoder: ErrorDecoderInput = None) -> ErrorDecoder:
    """Build or retrieve an error decoder."""
    from qldpc.decoders._capabilities import is_prebuilt_decoder
    from qldpc.decoders.construction._resolution import build_error_decoder
    from qldpc.decoders.construction.specs import DecoderSpec

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
    _warn_deprecated_function("decoders.get_error_decoder", replacement)
    return build_error_decoder(pcm_or_dem, decoder=decoder)


def get_observable_decoder(
    dem: stim.DetectorErrorModel, *, decoder: DecoderInput = None
) -> ObservableDecoder:
    """Build or retrieve an observable decoder."""
    from qldpc.decoders.construction._resolution import build_observable_decoder

    expression, note = _get_observable_decoder_expression(decoder)
    replacement = "the prebuilt observable decoder directly" if expression is None else expression
    _warn_deprecated_function("decoders.get_observable_decoder", replacement + note)
    return build_observable_decoder(dem, decoder=decoder)


def decode_observables(
    dem: stim.DetectorErrorModel,
    syndrome: npt.NDArray[np.int_],
    *,
    decoder: DecoderInput = None,
) -> npt.NDArray[np.int_]:
    """Predict the observable flips of one syndrome."""
    from qldpc.decoders.construction._resolution import build_observable_decoder

    expression, note = _get_observable_decoder_expression(decoder)
    replacement = (
        "the decode_observables method of the prebuilt observable decoder"
        if expression is None
        else f"{expression}.decode_observables(syndrome)"
    )
    _warn_deprecated_function("decoders.decode_observables", replacement + note)
    return build_observable_decoder(dem, decoder=decoder).decode_observables(syndrome)


def resolve_decoder(
    pcm_or_dem: PcmOrDem,
    decoder: ErrorDecoderInput,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> ErrorDecoder:
    """Resolve an error decoder input, together with deprecated keyword-based decoder arguments."""
    from qldpc.decoders.construction._resolution import build_error_decoder

    if decoder_args:
        decoder = merge_legacy_decoder_args(
            pcm_or_dem, decoder, decoder_args, warn_deprecated=warn_deprecated
        )
    return build_error_decoder(pcm_or_dem, decoder=decoder)


def resolve_observable_decoder(
    dem: stim.DetectorErrorModel,
    decoder: DecoderInput,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> ObservableDecoder:
    """Resolve an observable decoder input, together with deprecated keyword-based arguments."""
    from qldpc.decoders.construction._resolution import build_observable_decoder

    if decoder_args:
        decoder = merge_legacy_decoder_args(
            dem, decoder, decoder_args, warn_deprecated=warn_deprecated
        )
    return build_observable_decoder(dem, decoder=decoder)


def get_decoder_bp_osd(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> ErrorDecoder:
    """Build a BP+OSD decoder."""
    _warn_deprecated_builder("get_decoder_bp_osd", "bp_osd")
    return _get_decoder_constructors()["BP_OSD"](pcm_or_dem, **decoder_args)


def get_decoder_bp_lsd(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> ErrorDecoder:
    """Build a BP+LSD decoder."""
    _warn_deprecated_builder("get_decoder_bp_lsd", "bp_lsd")
    return _get_decoder_constructors()["BP_LSD"](pcm_or_dem, **decoder_args)


def get_decoder_bf(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> ErrorDecoder:
    """Build a belief-find decoder."""
    _warn_deprecated_builder("get_decoder_bf", "bf")
    return _get_decoder_constructors()["BF"](pcm_or_dem, **decoder_args)


def get_decoder_guf(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> GUFDecoder:
    """Build a GUF decoder."""
    _warn_deprecated_builder("get_decoder_guf", "guf")
    return cast("GUFDecoder", _get_decoder_constructors()["GUF"](pcm_or_dem, **decoder_args))


def get_decoder_ilp(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> ILPDecoder:
    """Build an ILP decoder."""
    _warn_deprecated_builder("get_decoder_ilp", "ilp")
    return cast("ILPDecoder", _get_decoder_constructors()["ILP"](pcm_or_dem, **decoder_args))


def get_decoder_lookup(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> ErrorLookupDecoder:
    """Build a lookup decoder."""
    _warn_deprecated_builder("get_decoder_lookup", "lookup")
    return cast(
        "ErrorLookupDecoder", _get_decoder_constructors()["lookup"](pcm_or_dem, **decoder_args)
    )


def get_decoder_mwpm(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> BatchErrorDecoder:
    """Build an MWPM decoder."""
    _warn_deprecated_builder("get_decoder_mwpm", "mwpm")
    return cast(
        "BatchErrorDecoder", _get_decoder_constructors()["MWPM"](pcm_or_dem, **decoder_args)
    )


def get_decoder_rbp(
    pcm_or_dem: PcmOrDem,
    error_priors: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: Any,
) -> RelayBPDecoder:
    """Build a Relay-BP decoder."""
    _warn_deprecated_function(
        "decoders.get_decoder_rbp",
        "decoders.relay_bp(...).build(pcm_or_dem) or decoders.min_sum_bp(...).build(pcm_or_dem)",
    )
    builder = _get_decoder_constructors()["RBP"]
    return cast("RelayBPDecoder", builder(pcm_or_dem, error_priors, **decoder_args))


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
    from qldpc.decoders._capabilities import (
        compiles_for_dem,
        is_prebuilt_decoder,
        is_prebuilt_observable_decoder,
    )
    from qldpc.decoders.construction.factories import DEMDecoderFactory, MatrixDecoderFactory
    from qldpc.decoders.construction.specs import DecoderSpec
    from qldpc.decoders.protocols import ObservableDecoder

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
    _warn_deprecated_function(
        f"decoders.{function_name}", f"decoders.{helper_name}(...).build(pcm_or_dem)"
    )


def _warn_deprecated_function(function_name: str, replacement: str) -> None:
    """Warn that a deprecated decoder function has a replacement."""
    _deprecations.warn_deprecated(
        f"{function_name} is deprecated; use {replacement} instead",
        stacklevel=get_external_caller_stacklevel(),
    )


####################################################################################################
# keyword-based decoder arguments


def merge_legacy_decoder_args(
    pcm_or_dem: PcmOrDem | None,
    decoder: DecoderInput,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> ErrorDecoderConstructor:
    """Translate nonempty deprecated keyword arguments into a decoder input.

    Args:
        pcm_or_dem: The parity-check matrix or detector error model to decode, which the warning
            uses to name a replacement, or None if it is not known yet.
        decoder: The decoder input passed alongside the deprecated arguments, which must be None.
        decoder_args: Nonempty deprecated keyword arguments.
        warn_deprecated: Whether to warn that the arguments are deprecated.

    Returns:
        A constructor that builds the selected decoder for a matrix or detector error model.
    """
    reject_removed_decoder_args(decoder_args)
    _reject_decoder_with_legacy_args(decoder)
    if warn_deprecated:
        _deprecations.warn_deprecated(
            get_legacy_decoder_migration_message(pcm_or_dem, decoder_args),
            stacklevel=get_external_caller_stacklevel(),
        )
    return _get_legacy_decoder_input(decoder_args)


def merge_legacy_css_decoder_args(
    matrix_x: galois.FieldArray,
    matrix_z: galois.FieldArray,
    decoder_x: DecoderInput,
    decoder_z: DecoderInput,
    decoder_args: Mapping[str, object],
    decoder_x_args: Mapping[str, object] | None,
    decoder_z_args: Mapping[str, object] | None,
) -> tuple[DecoderInput, DecoderInput]:
    """Translate deprecated keyword arguments for the two sectors of a CSS code into decoder inputs.

    Sector-specific arguments override the shared arguments.  Each nonempty argument mapping warns
    separately, naming the decoder= argument that replaces it.  If the two sectors have the same
    matrix, decoder input, and arguments, they also get the same translated decoder input.

    Args:
        matrix_x: The syndrome matrix of the sector that decodes X-type errors.
        matrix_z: The syndrome matrix of the sector that decodes Z-type errors.
        decoder_x: The decoder input for X-type errors, before translation.
        decoder_z: The decoder input for Z-type errors, before translation.
        decoder_args: Deprecated keyword arguments for both sectors.
        decoder_x_args: Deprecated keyword arguments for X-type errors, or None.
        decoder_z_args: Deprecated keyword arguments for Z-type errors, or None.

    Returns:
        The decoder inputs for X-type and Z-type errors.
    """
    for removed_args in [decoder_args, decoder_x_args or {}, decoder_z_args or {}]:
        reject_removed_decoder_args(removed_args)
    for argument_name, matrix, legacy_args in [
        ("decoder", matrix_x, decoder_args),
        ("decoder_x", matrix_x, decoder_x_args),
        ("decoder_z", matrix_z, decoder_z_args),
    ]:
        if legacy_args:
            _deprecations.warn_deprecated(
                get_legacy_decoder_migration_message(
                    matrix, legacy_args, argument_name=argument_name
                ),
                stacklevel=get_external_caller_stacklevel(),
            )
    sector_args_x = {**decoder_args, **(decoder_x_args or {})}
    sector_args_z = {**decoder_args, **(decoder_z_args or {})}
    same_x_and_z = (
        decoder_x is decoder_z
        and np.array_equal(matrix_x, matrix_z)
        and _have_equal_options(sector_args_x, sector_args_z)
    )
    if sector_args_x:
        _reject_decoder_with_legacy_args(decoder_x)
        decoder_x = _get_legacy_decoder_input(sector_args_x)
    if same_x_and_z:
        return decoder_x, decoder_x
    if sector_args_z:
        _reject_decoder_with_legacy_args(decoder_z)
        decoder_z = _get_legacy_decoder_input(sector_args_z)
    return decoder_x, decoder_z


def reject_removed_decoder_args(decoder_args: Mapping[str, object]) -> None:
    """Reject the removed static_decoder argument outside of get_decoder and decode."""
    if "static_decoder" in decoder_args:
        raise TypeError(
            "The static_decoder argument has been removed; pass a prebuilt decoder as decoder="
            " instead"
        )


def selects_legacy_decoder(decoder_args: Mapping[str, object]) -> bool:
    """Whether deprecated keyword arguments select a decoder with decoder_constructor or with_*."""
    return decoder_args.get("decoder_constructor") is not None or any(
        decoder_args.get(f"with_{name}", False) for name in _LEGACY_HELPER_NAMES
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
    selected = [name for name in _LEGACY_HELPER_NAMES if decoder_args.get(f"with_{name}", False)]
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


@functools.cache
def _get_decoder_constructors() -> dict[str, Callable[..., ErrorDecoder]]:
    """Map the NAME of each with_<NAME> keyword to the builder that it selects.

    Deprecated APIs pass free-form options to these builders unchecked, as qLDPC 0.3.3 did, rather
    than to the specification helpers that replace them, which check their options.
    """
    from qldpc.decoders.custom.guf import _get_decoder_guf
    from qldpc.decoders.custom.ilp import _get_decoder_ilp
    from qldpc.decoders.external.ldpc import (
        _get_decoder_bf,
        _get_decoder_bp_lsd,
        _get_decoder_bp_osd,
    )
    from qldpc.decoders.external.pymatching import _get_decoder_mwpm
    from qldpc.decoders.external.relay_bp import RelayBPDecoder

    return {
        "BF": functools.partial(_call_with_flat_backend_options, _get_decoder_bf),
        "BP_LSD": functools.partial(_call_with_flat_backend_options, _get_decoder_bp_lsd),
        "BP_OSD": functools.partial(_call_with_flat_backend_options, _get_decoder_bp_osd),
        "GUF": _get_decoder_guf,
        "ILP": _get_decoder_ilp,
        "MWPM": functools.partial(_call_with_flat_backend_options, _get_decoder_mwpm),
        "RBP": RelayBPDecoder,
        "lookup": _get_legacy_decoder_lookup,
    }


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


def _get_legacy_decoder_lookup(pcm_or_dem: PcmOrDem, /, **decoder_args: Any) -> ErrorDecoder:
    """Build a lookup decoder, accepting the deprecated predict_observable_flips option.

    A truthy predict_observable_flips builds the decoder that LookupDecoder(...,
    predict_observable_flips=True) builds, which warns, and a falsy one is dropped.
    """
    from qldpc.decoders.custom.lookup import LookupDecoder, _get_decoder_lookup

    if decoder_args.get("predict_observable_flips"):
        return cast("ErrorDecoder", LookupDecoder(pcm_or_dem, **decoder_args))
    decoder_args.pop("predict_observable_flips", None)
    return _get_decoder_lookup(pcm_or_dem, **decoder_args)


def _get_legacy_decoder(pcm_or_dem: PcmOrDem, decoder_args: Mapping[str, object]) -> Any:
    """Build a decoder with the deprecated keyword-based API."""
    static_decoder = decoder_args.get("static_decoder")
    if decoder_args.get("decoder_constructor") is None and static_decoder is not None:
        if len(decoder_args) > 1:
            raise ValueError("If passed a static decoder, we cannot process decoding arguments")
        return static_decoder
    return _get_legacy_decoder_input(decoder_args)(pcm_or_dem)


def _get_legacy_decoder_input(decoder_args: Mapping[str, object]) -> ErrorDecoderConstructor:
    """Translate deprecated decoder arguments into a decoder constructor.

    Free-form decoder options are passed through to the selected builder unchecked, as they were in
    qLDPC 0.3.3, so they become a constructor rather than a typed decoder specification.  If no
    decoder is selected, the constructor selects the default decoder for the matrix or detector
    error model that it is called with.
    """
    decoder_args = dict(decoder_args)
    if (decoder_constructor := decoder_args.pop("decoder_constructor", None)) is not None:
        if not callable(decoder_constructor):
            raise TypeError("The decoder_constructor argument must be callable")
        return functools.partial(decoder_constructor, **decoder_args)

    decoder_names = [
        name for name in _LEGACY_HELPER_NAMES if decoder_args.pop(f"with_{name}", False)
    ]
    if len(decoder_names) > 1:
        raise ValueError(
            "Only one decoder can be requested at a time, but received requests for: "
            + ", ".join(decoder_names)
        )
    if not decoder_names:
        return functools.partial(_get_default_legacy_decoder, **decoder_args)
    (decoder_name,) = decoder_names
    return functools.partial(_get_decoder_constructors()[decoder_name], **decoder_args)


def _get_default_legacy_decoder(pcm_or_dem: PcmOrDem, /, **decoder_args: object) -> ErrorDecoder:
    """Build the default decoder, GUF for a nonbinary FieldArray and BP+OSD otherwise."""
    is_nonbinary = isinstance(pcm_or_dem, galois.FieldArray) and type(pcm_or_dem).order != 2
    decoder_name = "GUF" if is_nonbinary else "BP_OSD"
    return _get_decoder_constructors()[decoder_name](pcm_or_dem, **decoder_args)


def _reject_decoder_with_legacy_args(decoder: DecoderInput) -> None:
    """Reject a decoder input that is passed alongside deprecated keyword arguments."""
    if decoder is not None:
        raise ValueError(
            "Cannot combine decoder= with deprecated decoder-selection or construction arguments"
        )


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


def _have_equal_options(options_a: Mapping[str, object], options_b: Mapping[str, object]) -> bool:
    """Whether two option mappings are equal, comparing array values elementwise."""
    return options_a.keys() == options_b.keys() and all(
        np.array_equal(np.asarray(value), np.asarray(options_b[key]))
        if isinstance(value, np.ndarray) or isinstance(options_b[key], np.ndarray)
        else value == options_b[key]
        for key, value in options_a.items()
    )


def _get_constructor_name(decoder_constructor: object) -> str:
    """Return a constructor name suitable for a migration message."""
    return getattr(decoder_constructor, "__name__", "MyDecoder")


def _get_legacy_helper_name(pcm_or_dem: PcmOrDem | None, decoder_args: Mapping[str, object]) -> str:
    """Return the typed helper replacing deprecated decoder selection."""
    selected = [name for name in _LEGACY_HELPER_NAMES if decoder_args.get(f"with_{name}", False)]
    if len(selected) == 1:
        return _LEGACY_HELPER_NAMES[selected[0]]
    if isinstance(pcm_or_dem, galois.FieldArray) and type(pcm_or_dem).order != 2:
        return "guf"
    return "bp_osd"


####################################################################################################
# deprecated error_rate option


def deprecate_error_rate_option(
    options: dict[str, object], explicitly_provided: frozenset[str]
) -> dict[str, object]:
    """Replace an explicitly supplied error_rate option with error_channel.

    An explicit error_rate=None is the default value, so it is dropped as if it were omitted.
    """
    error_rate = options.pop("error_rate", None)
    if "error_rate" in explicitly_provided and error_rate is not None:
        error_channel_given = "error_channel" in explicitly_provided
        options["error_channel"] = translate_error_rate(error_rate, error_channel_given)
    return options


def translate_error_rate(error_rate: _ErrorRateT, error_channel_given: bool) -> _ErrorRateT:
    """Warn that an error_rate value is deprecated, and return the error_channel that replaces it.

    Args:
        error_rate: A value that was supplied as error_rate, other than the default value None.
        error_channel_given: Whether error_channel was also supplied, which is an error.

    Returns:
        The value to use as error_channel.

    Raises:
        ValueError: If error_channel was also supplied.
    """
    if error_channel_given:
        raise ValueError("error_rate and error_channel cannot both be specified")
    _deprecations.warn_deprecated(
        f"error_rate={error_rate!r} is deprecated; use error_channel={error_rate!r} instead",
        stacklevel=get_external_caller_stacklevel(),
    )
    return error_rate


####################################################################################################
# deprecated observable flips from decode methods


def decode_returns_observables(decoder: object, *, warn: bool = True) -> bool:
    """Whether the decode methods of a decoder return observable flips, by a deprecated convention.

    A lookup decoder built with the deprecated predict_observable_flips=True option returns
    observable flips, as does another decoder object that declares decode_returns_observables=True,
    which is deprecated.  A decoder that predicts observable flips should implement
    decode_observables.

    Args:
        decoder: The decoder to check.
        warn: Whether to warn if the decoder declares decode_returns_observables=True.  A caller
            passes False if a later check of the same decoder warns, so that each use of the decoder
            warns once.
    """
    from .lookup import LegacyObservableLookupDecoder, LegacyWeightedObservableLookupDecoder

    # reading the deprecated attributes of a legacy lookup decoder would warn
    if isinstance(decoder, (LegacyObservableLookupDecoder, LegacyWeightedObservableLookupDecoder)):
        return True
    if isinstance(decoder, type) or not getattr(decoder, "decode_returns_observables", False):
        return False
    if warn:
        _deprecations.warn_deprecated(
            "Declaring decode_returns_observables=True is deprecated; a decoder that predicts"
            " observable flips should implement decode_observables instead",
            stacklevel=get_external_caller_stacklevel(),
        )
    return True
