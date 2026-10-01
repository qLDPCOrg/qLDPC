# SPDX-License-Identifier: Apache-2.0

"""Deprecated keyword-based decoder construction compatibility.

This module is an attachment on top of the modern decoder-construction API: it translates the
keyword arguments of qLDPC 0.3.3 into modern decoder inputs, and resolves them with the modern
resolution functions.  The modern modules never depend on it.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc._util import get_external_caller_stacklevel

from ..custom.guf import get_decoder_guf as _get_decoder_guf
from ..custom.ilp import get_decoder_ilp as _get_decoder_ilp
from ..custom.lookup import get_decoder_lookup
from ..external.ldpc import get_decoder_bf as _get_decoder_bf
from ..external.ldpc import get_decoder_bp_lsd as _get_decoder_bp_lsd
from ..external.ldpc import get_decoder_bp_osd as _get_decoder_bp_osd
from ..external.pymatching import get_decoder_mwpm as _get_decoder_mwpm
from ..external.relay_bp import get_decoder_rbp as _get_decoder_rbp
from ..protocols import ErrorDecoder, ObservableDecoder
from .resolution import get_error_decoder, get_observable_decoder
from .specs import (
    ErrorDecoderConstructor,
    ErrorDecoderInput,
    ObservableDecoderInput,
    PcmOrDem,
)

_DecoderInput = TypeVar("_DecoderInput", ErrorDecoderInput, ObservableDecoderInput)

# Legacy keyword-based compatibility
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


def resolve_decoder(
    pcm_or_dem: PcmOrDem,
    decoder: ErrorDecoderInput,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> ErrorDecoder:
    """Resolve an error decoder input, together with deprecated keyword-based decoder arguments.

    This serves methods that still accept deprecated keyword arguments next to decoder=.  Without
    such arguments, it is equivalent to qldpc.decoders.get_error_decoder.
    """
    decoder_input = _merge_legacy_decoder_args(
        pcm_or_dem, decoder, decoder_args, warn_deprecated=warn_deprecated
    )
    return get_error_decoder(pcm_or_dem, decoder=decoder_input)


def resolve_observable_decoder(
    dem: stim.DetectorErrorModel,
    decoder: ObservableDecoderInput,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> ObservableDecoder:
    """Resolve an observable decoder input, together with deprecated keyword-based arguments.

    This serves methods that still accept deprecated keyword arguments next to decoder=.  Without
    such arguments, it is equivalent to qldpc.decoders.get_observable_decoder.
    """
    decoder_input = _merge_legacy_decoder_args(
        dem, decoder, decoder_args, warn_deprecated=warn_deprecated
    )
    return get_observable_decoder(dem, decoder=decoder_input)


def reject_removed_decoder_args(decoder_args: Mapping[str, object]) -> None:
    """Reject the removed static_decoder argument outside of get_decoder and decode."""
    if "static_decoder" in decoder_args:
        raise TypeError(
            "The static_decoder argument has been removed; pass a prebuilt decoder as decoder="
            " instead"
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


# Legacy translation and message helpers

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

    Free-form decoder options are passed through to the selected builder unchecked, as they were
    in qLDPC 0.3.3, so they become a constructor rather than typed decoder settings.
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
    decoder: _DecoderInput,
    decoder_args: Mapping[str, object],
    *,
    warn_deprecated: bool = True,
) -> _DecoderInput | ErrorDecoderConstructor:
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
