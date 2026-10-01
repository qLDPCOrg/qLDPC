# SPDX-License-Identifier: Apache-2.0

"""Decoder adapters and conversion helpers.

This module is a compatibility/public facade for adapted external decoders:
``qldpc.decoders.adapters.BpOsdDecoder``, ``BpLsdDecoder``, ``BeliefFindDecoder``, and
``Matching`` remain importable from this module path.  Their external dependencies are imported
only when one of these classes is first accessed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ..protocols import BatchErrorDecoder, ErrorDecoder
from .error_decoders import (
    ErrorsToObservablesDecoder,
    ExpandedErrorDecoder,
    match_error_decoder_to_dem,
)
from .observable_decoders import (
    BitPackedObservableDecoder,
    ErrorsToFieldObservablesDecoder,
    validate_decoder_output,
    validate_observable_decoder,
)

__all__ = [
    "BatchErrorDecoder",
    "BeliefFindDecoder",
    "BitPackedObservableDecoder",
    "BpLsdDecoder",
    "BpOsdDecoder",
    "ErrorDecoder",
    "ErrorsToFieldObservablesDecoder",
    "ErrorsToObservablesDecoder",
    "ExpandedErrorDecoder",
    "Matching",
    "match_error_decoder_to_dem",
    "validate_decoder_output",
    "validate_observable_decoder",
]

_BACKEND_NAMES = frozenset({"BeliefFindDecoder", "BpLsdDecoder", "BpOsdDecoder", "Matching"})

if TYPE_CHECKING:
    from .backends import BeliefFindDecoder, BpLsdDecoder, BpOsdDecoder, Matching
else:

    def __getattr__(name: str) -> Any:
        """Load adapted external decoder classes only when they are first accessed."""
        if name not in _BACKEND_NAMES:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        backends = importlib.import_module(f"{__name__}.backends")
        value = getattr(backends, name)
        globals()[name] = value
        return value
