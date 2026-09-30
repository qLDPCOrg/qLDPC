# SPDX-License-Identifier: Apache-2.0

"""Custom decoder classes."""

from typing import TYPE_CHECKING, Any

from qldpc._util import get_deprecated_alias

from ..common import PLACEHOLDER_ERROR_RATE
from ..protocols import BatchErrorDecoder, ErrorDecoder
from .composition import CompositeDecoder, DirectDecoder
from .guf import GUFDecoder
from .ilp import ILPDecoder
from .relay_bp import RelayBPDecoder

__all__ = [
    "PLACEHOLDER_ERROR_RATE",
    "BatchDecoder",
    "CompositeDecoder",
    "Decoder",
    "DirectDecoder",
    "GUFDecoder",
    "ILPDecoder",
    "RelayBPDecoder",
]

DEPRECATED_ALIASES = {"Decoder": ErrorDecoder, "BatchDecoder": BatchErrorDecoder}

if TYPE_CHECKING:
    Decoder = ErrorDecoder
    BatchDecoder = BatchErrorDecoder
else:

    def __getattr__(name: str) -> Any:
        """Resolve deprecated names of decoder protocols, with a DeprecationWarning."""
        return get_deprecated_alias(__name__, name, DEPRECATED_ALIASES)
