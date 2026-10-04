# SPDX-License-Identifier: Apache-2.0

"""Decoder specifications, protocols, and workflow entry points."""

from typing import TYPE_CHECKING, Any

from qldpc._util import get_deprecated_alias

from . import custom, dems, external, sinter
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
    ObservableDecodingFallbackWarning,
    get_error_and_erasure,
    with_erasure_bits,
)
from .construction.factories import (
    DEMDecoderFactory,
    MatrixDecoderFactory,
    from_dem,
    from_matrix,
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
from .custom.guf import guf
from .custom.ilp import ilp
from .custom.lookup import lookup
from .dems import DetectorErrorModelArrays
from .external.frontier import frontier
from .external.ldpc import bf, bp_lsd, bp_osd
from .external.pymatching import mwpm
from .external.relay_bp import min_sum_bp, relay_bp
from .external.tesseract import tesseract, tesseract_preset
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
    CompiledSinterDecoder,
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
    "DEMDecoderFactory",
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
    "MatrixDecoderFactory",
    "ObservableDecodeResult",
    "ObservableDecoder",
    "ObservableDecoderCompiler",
    "ObservableDecoderConstructor",
    "ObservableDecodingFallbackWarning",
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
    "decode_observables",
    "from_dem",
    "from_matrix",
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

_HISTORICAL_ROOT_CLASSES = {
    "CompiledSequentialWindowDecoder": sinter.CompiledSequentialWindowDecoder,
    "CompiledSubgraphDecoder": sinter.CompiledSubgraphDecoder,
    "CompiledTrivialDecoder": sinter.CompiledTrivialDecoder,
    "CompositeDecoder": custom.CompositeDecoder,
    "DirectDecoder": custom.DirectDecoder,
    "FlipPattern": dems.FlipPattern,
    "FrontierObservableDecoder": external.FrontierObservableDecoder,
    "GUFDecoder": custom.GUFDecoder,
    "ILPDecoder": custom.ILPDecoder,
    "LookupDecoder": custom.LookupDecoder,
    "ObservableLookupDecoder": custom.ObservableLookupDecoder,
    "RelayBPDecoder": external.RelayBPDecoder,
    "TesseractDecoder": external.TesseractDecoder,
    "WeightedLookupDecoder": custom.WeightedLookupDecoder,
    "WeightedObservableLookupDecoder": custom.WeightedObservableLookupDecoder,
}
_HISTORICAL_ROOT_NAMES = {
    name: f"{decoder.__module__}.{decoder.__name__}"
    for name, decoder in _HISTORICAL_ROOT_CLASSES.items()
}

DEPRECATED_ALIASES = (
    custom.DEPRECATED_ALIASES
    | sinter.DEPRECATED_ALIASES
    | _HISTORICAL_ROOT_CLASSES
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
    from .custom import (
        CompositeDecoder as CompositeDecoder,
    )
    from .custom import Decoder as Decoder
    from .custom import (
        DirectDecoder as DirectDecoder,
    )
    from .custom import (
        GUFDecoder as GUFDecoder,
    )
    from .custom import (
        ILPDecoder as ILPDecoder,
    )
    from .custom import (
        LookupDecoder as LookupDecoder,
    )
    from .custom import (
        ObservableLookupDecoder as ObservableLookupDecoder,
    )
    from .custom import (
        WeightedLookupDecoder as WeightedLookupDecoder,
    )
    from .custom import (
        WeightedObservableLookupDecoder as WeightedObservableLookupDecoder,
    )
    from .custom.lookup import lookup as lookup_table
    from .dems import FlipPattern as FlipPattern
    from .external import (
        FrontierObservableDecoder as FrontierObservableDecoder,
    )
    from .external import (
        RelayBPDecoder as RelayBPDecoder,
    )
    from .external import (
        TesseractDecoder as TesseractDecoder,
    )
    from .sinter import (
        CompiledSequentialWindowDecoder as CompiledSequentialWindowDecoder,
    )
    from .sinter import (
        CompiledSubgraphDecoder as CompiledSubgraphDecoder,
    )
    from .sinter import (
        CompiledTrivialDecoder as CompiledTrivialDecoder,
    )
    from .sinter import SequentialSinterDecoder as SequentialSinterDecoder
    from .sinter import SubgraphSinterDecoder as SubgraphSinterDecoder
else:

    def __getattr__(name: str) -> Any:
        """Resolve deprecated names of decoder classes and protocols, with a DeprecationWarning."""
        return get_deprecated_alias(__name__, name, DEPRECATED_ALIASES, _HISTORICAL_ROOT_NAMES)
