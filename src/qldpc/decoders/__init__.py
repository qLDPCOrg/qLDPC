# SPDX-License-Identifier: Apache-2.0

"""Decoder protocols, builders, implementations, conversion helpers, and Sinter adapters."""

from typing import TYPE_CHECKING, Any

from qldpc._util import get_deprecated_alias

from . import builders, custom, sinter
from .adapters.dem import (
    ErrorsToObservablesDecoder,
    ExpandedErrorDecoder,
    match_error_decoder_to_dem,
)
from .builders import (
    get_decoder_bf,
    get_decoder_bp_lsd,
    get_decoder_bp_osd,
    get_decoder_guf,
    get_decoder_ilp,
    get_decoder_lookup,
    get_decoder_mwpm,
    get_decoder_rbp,
)
from .capabilities import (
    compiles_for_dem,
    is_prebuilt_observable_decoder,
)
from .common import (
    get_error_and_erasure,
    with_erasure_bits,
)
from .custom import (
    CompositeDecoder,
    DirectDecoder,
    GUFDecoder,
    ILPDecoder,
    RelayBPDecoder,
)
from .dems import (
    DetectorErrorModelArrays,
    FlipPattern,
)
from .lookup import (
    LookupDecoder,
    ObservableLookupDecoder,
    WeightedLookupDecoder,
    WeightedObservableLookupDecoder,
)
from .protocols import (
    BatchErrorDecoder,
    BatchObservableDecoder,
    ErrorDecoder,
    ObservableDecoder,
    SupportsDecode,
    WrappedErrorDecoder,
    as_error_decoder,
    batch_decode_errors,
    supports_batch_decoding,
)
from .retrieval import (
    DecoderSpec,
    DeferredErrorDecoderInput,
    DeferredObservableDecoderInput,
    ErrorDecoderConstructor,
    ErrorDecoderInput,
    ObservableDecoderConstructor,
    ObservableDecoderInput,
    PcmOrDem,
    bf,
    bp_lsd,
    bp_osd,
    decode,
    decode_observables,
    get_decoder,
    get_error_decoder,
    get_legacy_decoder_migration_message,
    get_observable_decoder,
    guf,
    ilp,
    is_prebuilt_decoder,
    lookup_table,
    min_sum_bp,
    mwpm,
    reject_prebuilt_decoder,
    reject_removed_decoder_args,
    relay_bp,
    resolve_decoder,
    resolve_observable_decoder,
)
from .sinter import (
    CompiledSequentialWindowDecoder,
    CompiledSinterDecoder,
    CompiledSubgraphDecoder,
    CompiledTrivialDecoder,
    DecoderNotCompiledError,
    SequentialWindowDecoder,
    SinterDecoder,
    SlidingWindowDecoder,
    SubgraphDecoder,
    TrivialDecoder,
)

__all__ = [
    "BatchDecoder",
    "BatchErrorDecoder",
    "BatchObservableDecoder",
    "CompiledSequentialWindowDecoder",
    "CompiledSinterDecoder",
    "CompiledSubgraphDecoder",
    "CompiledTrivialDecoder",
    "CompositeDecoder",
    "Decoder",
    "DecoderNotCompiledError",
    "DecoderSpec",
    "DeferredErrorDecoderInput",
    "DeferredObservableDecoderInput",
    "DetectorErrorModelArrays",
    "DirectDecoder",
    "ErrorDecoder",
    "ErrorDecoderConstructor",
    "ErrorDecoderInput",
    "ErrorsToObservablesDecoder",
    "ExpandedErrorDecoder",
    "FlipPattern",
    "GUFDecoder",
    "ILPDecoder",
    "LookupDecoder",
    "ObservableDecoder",
    "ObservableDecoderConstructor",
    "ObservableDecoderInput",
    "ObservableLookupDecoder",
    "PcmOrDem",
    "RelayBPDecoder",
    "SequentialSinterDecoder",
    "SequentialWindowDecoder",
    "SinterDecoder",
    "SlidingWindowDecoder",
    "SubgraphDecoder",
    "SubgraphSinterDecoder",
    "SupportsDecode",
    "TrivialDecoder",
    "WeightedLookupDecoder",
    "WeightedObservableLookupDecoder",
    "WrappedErrorDecoder",
    "as_error_decoder",
    "batch_decode_errors",
    "bf",
    "bp_lsd",
    "bp_osd",
    "compiles_for_dem",
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
    "get_decoder_bf",
    "get_decoder_bp_lsd",
    "get_decoder_bp_osd",
    "get_decoder_guf",
    "get_decoder_ilp",
    "get_decoder_lookup",
    "get_decoder_mwpm",
    "get_decoder_rbp",
    "get_error_and_erasure",
    "get_error_decoder",
    "get_legacy_decoder_migration_message",
    "get_observable_decoder",
    "guf",
    "ilp",
    "is_prebuilt_decoder",
    "is_prebuilt_observable_decoder",
    "lookup_table",
    "match_error_decoder_to_dem",
    "min_sum_bp",
    "mwpm",
    "reject_prebuilt_decoder",
    "reject_removed_decoder_args",
    "relay_bp",
    "resolve_decoder",
    "resolve_observable_decoder",
    "supports_batch_decoding",
    "with_erasure_bits",
]

# Deprecated names remain importable (including by star imports, since they are listed in __all__),
# and resolve at runtime through a module-level __getattr__ (PEP 562) that warns when accessed.
# Type checkers instead see plain aliases, so that they still flag misspelled attributes.
DEPRECATED_ALIASES = (
    custom.DEPRECATED_ALIASES
    | sinter.DEPRECATED_ALIASES
    | {
        "get_decoder_BF": builders.get_decoder_bf,
        "get_decoder_BP_LSD": builders.get_decoder_bp_lsd,
        "get_decoder_BP_OSD": builders.get_decoder_bp_osd,
        "get_decoder_GUF": builders.get_decoder_guf,
        "get_decoder_ILP": builders.get_decoder_ilp,
        "get_decoder_MWPM": builders.get_decoder_mwpm,
        "get_decoder_RBP": builders.get_decoder_rbp,
    }
)

if TYPE_CHECKING:
    from .builders import get_decoder_bf as get_decoder_BF
    from .builders import get_decoder_bp_lsd as get_decoder_BP_LSD
    from .builders import get_decoder_bp_osd as get_decoder_BP_OSD
    from .builders import get_decoder_guf as get_decoder_GUF
    from .builders import get_decoder_ilp as get_decoder_ILP
    from .builders import get_decoder_mwpm as get_decoder_MWPM
    from .builders import get_decoder_rbp as get_decoder_RBP
    from .custom import BatchDecoder as BatchDecoder
    from .custom import Decoder as Decoder
    from .sinter import SequentialSinterDecoder as SequentialSinterDecoder
    from .sinter import SubgraphSinterDecoder as SubgraphSinterDecoder
else:

    def __getattr__(name: str) -> Any:
        """Resolve deprecated names of decoder classes and protocols, with a DeprecationWarning."""
        return get_deprecated_alias(__name__, name, DEPRECATED_ALIASES)
