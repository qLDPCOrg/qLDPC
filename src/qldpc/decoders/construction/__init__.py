# SPDX-License-Identifier: Apache-2.0

"""Typed decoder construction and resolution."""

from ..custom.guf import get_decoder_guf
from ..custom.ilp import get_decoder_ilp
from ..custom.lookup import get_decoder_lookup, get_observable_decoder_lookup
from ..external.ldpc import get_decoder_bf, get_decoder_bp_lsd, get_decoder_bp_osd
from ..external.pymatching import (
    get_decoder_mwpm,
    get_error_decoder_mwpm,
    get_observable_decoder_mwpm,
)
from ..external.relay_bp import (
    get_decoder_rbp,
    get_min_sum_bp_decoder,
    get_relay_bp_decoder,
)
from .legacy import (
    get_legacy_decoder_migration_message,
    reject_removed_decoder_args,
    resolve_decoder,
    resolve_observable_decoder,
)
from .resolution import (
    decode_observables,
    get_error_decoder,
    get_observable_decoder,
    reject_prebuilt_decoder,
)
from .specs import (
    DecoderSpec,
    DeferredErrorDecoderInput,
    DeferredObservableDecoderInput,
    ErrorDecoderConstructor,
    ErrorDecoderInput,
    ObservableDecoderCompiler,
    ObservableDecoderConstructor,
    ObservableDecoderInput,
    PcmOrDem,
    bf,
    bp_lsd,
    bp_osd,
    guf,
    ilp,
    lookup_table,
    min_sum_bp,
    mwpm,
    relay_bp,
)

__all__ = [
    "DecoderSpec",
    "DeferredErrorDecoderInput",
    "DeferredObservableDecoderInput",
    "ErrorDecoderConstructor",
    "ErrorDecoderInput",
    "ObservableDecoderCompiler",
    "ObservableDecoderConstructor",
    "ObservableDecoderInput",
    "PcmOrDem",
    "bf",
    "bp_lsd",
    "bp_osd",
    "decode_observables",
    "get_decoder_bf",
    "get_decoder_bp_lsd",
    "get_decoder_bp_osd",
    "get_decoder_guf",
    "get_decoder_ilp",
    "get_decoder_lookup",
    "get_decoder_mwpm",
    "get_decoder_rbp",
    "get_error_decoder",
    "get_error_decoder_mwpm",
    "get_legacy_decoder_migration_message",
    "get_min_sum_bp_decoder",
    "get_observable_decoder",
    "get_observable_decoder_lookup",
    "get_observable_decoder_mwpm",
    "get_relay_bp_decoder",
    "guf",
    "ilp",
    "lookup_table",
    "min_sum_bp",
    "mwpm",
    "reject_prebuilt_decoder",
    "reject_removed_decoder_args",
    "relay_bp",
    "resolve_decoder",
    "resolve_observable_decoder",
]
