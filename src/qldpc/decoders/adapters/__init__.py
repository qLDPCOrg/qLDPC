# SPDX-License-Identifier: Apache-2.0

"""Generic error- and observable-decoder adapters."""

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
    "BitPackedObservableDecoder",
    "ErrorDecoder",
    "ErrorsToFieldObservablesDecoder",
    "ErrorsToObservablesDecoder",
    "ExpandedErrorDecoder",
    "match_error_decoder_to_dem",
    "validate_decoder_output",
    "validate_observable_decoder",
]
