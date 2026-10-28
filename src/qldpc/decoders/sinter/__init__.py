# SPDX-License-Identifier: Apache-2.0

"""Decoders for Sinter to sample quantum error correction circuits."""

from typing import TYPE_CHECKING

from qldpc import _deprecations

from .core import (
    CompiledSinterDecoder,
    CompiledTrivialDecoder,
    SinterDecoder,
    TrivialDecoder,
)
from .subgraph import CompiledSubgraphDecoder, SubgraphDecoder
from .window import (
    CompiledSequentialWindowDecoder,
    SequentialWindowDecoder,
    SlidingWindowDecoder,
)

__all__ = [
    "CompiledSequentialWindowDecoder",
    "CompiledSinterDecoder",
    "CompiledSubgraphDecoder",
    "CompiledTrivialDecoder",
    "SequentialWindowDecoder",
    "SinterDecoder",
    "SlidingWindowDecoder",
    "SubgraphDecoder",
    "TrivialDecoder",
]

if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
