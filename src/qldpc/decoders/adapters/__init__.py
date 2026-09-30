# SPDX-License-Identifier: Apache-2.0

"""Decoder adapters and conversion helpers.

This module is a compatibility/public facade for adapted external decoders:
``qldpc.decoders.adapters.BpOsdDecoder``, ``BpLsdDecoder``, ``BeliefFindDecoder``, and
``Matching`` remain importable from this module path.
"""

from .backends import BeliefFindDecoder, BpLsdDecoder, BpOsdDecoder, Matching
from .dem import (
    ErrorsToObservablesDecoder,
    ExpandedErrorDecoder,
    match_error_decoder_to_dem,
    validate_observable_decoder,
)
from .observables import (
    BitPackedObservableDecoder,
    ErrorsToFieldObservablesDecoder,
    validate_decoder_output,
)

__all__ = [
    "BeliefFindDecoder",
    "BitPackedObservableDecoder",
    "BpLsdDecoder",
    "BpOsdDecoder",
    "ErrorsToFieldObservablesDecoder",
    "ErrorsToObservablesDecoder",
    "ExpandedErrorDecoder",
    "Matching",
    "match_error_decoder_to_dem",
    "validate_decoder_output",
    "validate_observable_decoder",
]
