# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from qldpc._util import get_deprecated_alias

from . import custom, sinter
from .common import (
    with_erasure_bits,
)
from .custom import (
    BatchErrorDecoder,
    BatchObservableDecoder,
    CompositeDecoder,
    DirectDecoder,
    ErrorDecoder,
    GUFDecoder,
    ILPDecoder,
    ObservableDecoder,
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
    get_decoder_BF,
    get_decoder_BP_LSD,
    get_decoder_BP_OSD,
    get_decoder_GUF,
    get_decoder_ILP,
    get_decoder_lookup,
    get_decoder_MWPM,
    get_decoder_RBP,
    get_observable_decoder,
    guf,
    ilp,
    lookup_table,
    min_sum_bp,
    mwpm,
    relay_bp,
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
    "BatchErrorDecoder",
    "BatchObservableDecoder",
    "CompiledSequentialWindowDecoder",
    "CompiledSinterDecoder",
    "CompiledSubgraphDecoder",
    "CompiledTrivialDecoder",
    "CompositeDecoder",
    "Decoder",
    "DecoderSpec",
    "DeferredErrorDecoderInput",
    "DeferredObservableDecoderInput",
    "DetectorErrorModelArrays",
    "DirectDecoder",
    "ErrorDecoder",
    "ErrorDecoderConstructor",
    "ErrorDecoderInput",
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
    "TrivialDecoder",
    "WeightedLookupDecoder",
    "WeightedObservableLookupDecoder",
    "bf",
    "bp_lsd",
    "bp_osd",
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
    "get_decoder_lookup",
    "get_observable_decoder",
    "guf",
    "ilp",
    "lookup_table",
    "min_sum_bp",
    "mwpm",
    "relay_bp",
    "with_erasure_bits",
]

# Deprecated names remain importable (including by star imports, since they are listed in __all__),
# and resolve at runtime through a module-level __getattr__ (PEP 562) that warns when accessed.
# Type checkers instead see plain aliases, so that they still flag misspelled attributes.
_DEPRECATED_ALIASES = custom._DEPRECATED_ALIASES | sinter._DEPRECATED_ALIASES

if TYPE_CHECKING:
    from .custom import BatchDecoder as BatchDecoder
    from .custom import Decoder as Decoder
    from .sinter import SequentialSinterDecoder as SequentialSinterDecoder
    from .sinter import SubgraphSinterDecoder as SubgraphSinterDecoder
else:

    def __getattr__(name: str) -> Any:
        """Resolve deprecated names of decoder classes and protocols, with a DeprecationWarning."""
        return get_deprecated_alias(__name__, name, _DEPRECATED_ALIASES)
