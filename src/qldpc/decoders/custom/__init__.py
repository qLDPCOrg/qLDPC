# SPDX-License-Identifier: Apache-2.0

"""Custom decoder classes."""

from typing import TYPE_CHECKING, Any

from qldpc._util import get_deprecated_alias

from ..common import PLACEHOLDER_ERROR_RATE
from ..external.relay_bp import RelayBPDecoder as _RelayBPDecoder
from ..protocols import BatchErrorDecoder, ErrorDecoder
from .composition import CompositeDecoder, DirectDecoder
from .guf import GUFDecoder
from .ilp import ILPDecoder
from .lookup import (
    LookupDecoder,
    ObservableLookupDecoder,
    WeightedLookupDecoder,
    WeightedObservableLookupDecoder,
)

__all__ = [
    "PLACEHOLDER_ERROR_RATE",
    "BatchDecoder",
    "CompositeDecoder",
    "Decoder",
    "DirectDecoder",
    "GUFDecoder",
    "ILPDecoder",
    "LookupDecoder",
    "ObservableLookupDecoder",
    "RelayBPDecoder",
    "WeightedLookupDecoder",
    "WeightedObservableLookupDecoder",
]

# Deprecated compatibility aliases

DEPRECATED_ALIASES = {
    "Decoder": ErrorDecoder,
    "BatchDecoder": BatchErrorDecoder,
    "RelayBPDecoder": _RelayBPDecoder,
}

if TYPE_CHECKING:
    Decoder = ErrorDecoder
    BatchDecoder = BatchErrorDecoder
    RelayBPDecoder = _RelayBPDecoder
else:

    def __getattr__(name: str) -> Any:
        """Resolve deprecated custom-package names with a DeprecationWarning."""
        return get_deprecated_alias(
            __name__,
            name,
            DEPRECATED_ALIASES,
            {"RelayBPDecoder": "qldpc.decoders.external.relay_bp.RelayBPDecoder"},
        )
