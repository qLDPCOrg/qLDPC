# SPDX-License-Identifier: Apache-2.0

"""Custom decoder classes."""

from typing import TYPE_CHECKING

from qldpc import _deprecations

from .composition import CompositeDecoder, DirectDecoder
from .guf import GUFDecoder
from .ilp import ILPDecoder
from .lookup import (
    ErrorLookupDecoder,
    LookupDecoder,
    ObservableLookupDecoder,
    WeightedErrorLookupDecoder,
    WeightedLookupDecoder,
    WeightedObservableLookupDecoder,
)

__all__ = [
    "CompositeDecoder",
    "DirectDecoder",
    "ErrorLookupDecoder",
    "GUFDecoder",
    "ILPDecoder",
    "LookupDecoder",
    "ObservableLookupDecoder",
    "WeightedErrorLookupDecoder",
    "WeightedLookupDecoder",
    "WeightedObservableLookupDecoder",
]

if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
