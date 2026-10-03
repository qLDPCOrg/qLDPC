# SPDX-License-Identifier: Apache-2.0

"""Decoder protocols, builders, implementations, conversion helpers, and Sinter adapters."""

from typing import TYPE_CHECKING, Any

from qldpc._util import get_deprecated_alias

from . import custom, sinter
from .adapters.error_decoders import (
    ErrorsToObservablesDecoder,
    ExpandedErrorDecoder,
    match_error_decoder_to_dem,
)
from .capabilities import (
    compiles_for_dem,
    is_prebuilt_decoder,
    is_prebuilt_observable_decoder,
)
from .common import (
    decode_errors_detailed,
    decode_errors_detailed_batch,
    decode_observables_detailed,
    decode_observables_detailed_batch,
    get_error_and_erasure,
    with_erasure_bits,
)
from .construction.legacy import (
    decode_observables,
    get_decoder_bf,
    get_decoder_bp_lsd,
    get_decoder_bp_osd,
    get_decoder_guf,
    get_decoder_ilp,
    get_decoder_lookup,
    get_decoder_mwpm,
    get_decoder_rbp,
    get_error_decoder,
    get_legacy_decoder_migration_message,
    get_observable_decoder,
    reject_removed_decoder_args,
    resolve_decoder,
    resolve_observable_decoder,
)
from .construction.resolution import reject_prebuilt_decoder
from .construction.specs import (
    DecoderInput,
    DecoderSpec,
    DeferredDecoderInput,
    DeferredErrorDecoderInput,
    ErrorDecoderConstructor,
    ErrorDecoderInput,
    ObservableDecoderCompiler,
    ObservableDecoderConstructor,
    PcmOrDem,
)
from .custom import (
    CompositeDecoder,
    DirectDecoder,
    GUFDecoder,
    ILPDecoder,
    LookupDecoder,
    ObservableLookupDecoder,
    WeightedLookupDecoder,
    WeightedObservableLookupDecoder,
)
from .custom.guf import guf
from .custom.ilp import ilp
from .custom.lookup import lookup
from .dems import (
    DetectorErrorModelArrays,
    FlipPattern,
)
from .external.frontier import FrontierObservableDecoder, frontier
from .external.ldpc import bf, bp_lsd, bp_osd
from .external.pymatching import mwpm
from .external.relay_bp import RelayBPDecoder, min_sum_bp, relay_bp
from .external.tesseract import TesseractDecoder, tesseract, tesseract_preset
from .protocols import (
    BatchDetailedErrorDecoder,
    BatchDetailedObservableDecoder,
    BatchErrorDecoder,
    BatchObservableDecoder,
    DetailedErrorDecoder,
    DetailedObservableDecoder,
    ErrorDecoder,
    ErrorDecodeResult,
    ObservableDecoder,
    ObservableDecodeResult,
    SupportsDecode,
    WrappedErrorDecoder,
    as_error_decoder,
    batch_decode_errors,
    supports_batch_decoding,
)
from .sinter import (
    CompiledSequentialWindowDecoder,
    CompiledSinterDecoder,
    CompiledSubgraphDecoder,
    CompiledTrivialDecoder,
    SequentialWindowDecoder,
    SinterDecoder,
    SlidingWindowDecoder,
    SubgraphDecoder,
    TrivialDecoder,
)

__all__ = [
    "BatchDecoder",
    "BatchDetailedErrorDecoder",
    "BatchDetailedObservableDecoder",
    "BatchErrorDecoder",
    "BatchObservableDecoder",
    "CompiledSequentialWindowDecoder",
    "CompiledSinterDecoder",
    "CompiledSubgraphDecoder",
    "CompiledTrivialDecoder",
    "CompositeDecoder",
    "Decoder",
    "DecoderInput",
    "DecoderSpec",
    "DeferredDecoderInput",
    "DeferredErrorDecoderInput",
    "DetailedErrorDecoder",
    "DetailedObservableDecoder",
    "DetectorErrorModelArrays",
    "DirectDecoder",
    "ErrorDecodeResult",
    "ErrorDecoder",
    "ErrorDecoderConstructor",
    "ErrorDecoderInput",
    "ErrorsToObservablesDecoder",
    "ExpandedErrorDecoder",
    "FlipPattern",
    "FrontierObservableDecoder",
    "GUFDecoder",
    "ILPDecoder",
    "LookupDecoder",
    "ObservableDecodeResult",
    "ObservableDecoder",
    "ObservableDecoderCompiler",
    "ObservableDecoderConstructor",
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
    "TesseractDecoder",
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
    "decode_errors_detailed",
    "decode_errors_detailed_batch",
    "decode_observables",
    "decode_observables_detailed",
    "decode_observables_detailed_batch",
    "frontier",
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
    "lookup",
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
    "tesseract",
    "tesseract_preset",
    "with_erasure_bits",
]

# Deprecated compatibility aliases

# Deprecated names remain importable (including by star imports, since they are listed in __all__),
# and resolve at runtime through a module-level __getattr__ (PEP 562) that warns when accessed.
# Type checkers instead see plain aliases, so that they still flag misspelled attributes.
from .construction.legacy import decode, get_decoder

DEPRECATED_ALIASES = (
    custom.DEPRECATED_ALIASES
    | sinter.DEPRECATED_ALIASES
    | {
        "get_decoder_BF": get_decoder_bf,
        "get_decoder_BP_LSD": get_decoder_bp_lsd,
        "get_decoder_BP_OSD": get_decoder_bp_osd,
        "get_decoder_GUF": get_decoder_guf,
        "get_decoder_ILP": get_decoder_ilp,
        "get_decoder_MWPM": get_decoder_mwpm,
        "get_decoder_RBP": get_decoder_rbp,
        "lookup_table": lookup,
    }
)

if TYPE_CHECKING:
    from .construction.legacy import get_decoder_bf as get_decoder_BF
    from .construction.legacy import get_decoder_bp_lsd as get_decoder_BP_LSD
    from .construction.legacy import get_decoder_bp_osd as get_decoder_BP_OSD
    from .construction.legacy import get_decoder_guf as get_decoder_GUF
    from .construction.legacy import get_decoder_ilp as get_decoder_ILP
    from .construction.legacy import get_decoder_mwpm as get_decoder_MWPM
    from .construction.legacy import get_decoder_rbp as get_decoder_RBP
    from .custom import BatchDecoder as BatchDecoder
    from .custom import Decoder as Decoder
    from .custom.lookup import lookup as lookup_table
    from .sinter import SequentialSinterDecoder as SequentialSinterDecoder
    from .sinter import SubgraphSinterDecoder as SubgraphSinterDecoder
else:

    def __getattr__(name: str) -> Any:
        """Resolve deprecated names of decoder classes and protocols, with a DeprecationWarning."""
        return get_deprecated_alias(__name__, name, DEPRECATED_ALIASES)
