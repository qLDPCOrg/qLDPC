# SPDX-License-Identifier: Apache-2.0

"""Generic error- and observable-decoder adapters."""

from typing import TYPE_CHECKING

from qldpc import _deprecations

from .error_decoders import ErrorsToObservablesDecoder

__all__ = [
    "ErrorsToObservablesDecoder",
]

if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
