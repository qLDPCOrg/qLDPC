# SPDX-License-Identifier: Apache-2.0

"""Explicit input contracts for custom decoder factories."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import stim

from qldpc.math import IntegerArray

from ..protocols import ErrorDecoder, ObservableDecoder, SupportsDecode


@dataclasses.dataclass(frozen=True, slots=True)
class _MatrixDecoderFactory:
    factory: Callable[..., ErrorDecoder | SupportsDecode]

    def build(self, matrix: IntegerArray) -> ErrorDecoder | SupportsDecode:
        """Build an error decoder for a parity-check matrix."""
        return self.factory(matrix)


@dataclasses.dataclass(frozen=True, slots=True)
class _DEMDecoderFactory:
    factory: Callable[..., ObservableDecoder]

    def build(self, dem: stim.DetectorErrorModel) -> ObservableDecoder:
        """Build an observable decoder for a detector error model."""
        return self.factory(dem)


def from_matrix(factory: Callable[..., ErrorDecoder | SupportsDecode]) -> _MatrixDecoderFactory:
    """Declare a parity-check matrix input for a custom error-decoder factory."""
    if not callable(factory):
        raise TypeError("from_matrix requires a callable factory")
    return _MatrixDecoderFactory(factory)


def from_dem(factory: Callable[..., ObservableDecoder]) -> _DEMDecoderFactory:
    """Declare a binary detector error model input for a custom observable-decoder factory."""
    if not callable(factory):
        raise TypeError("from_dem requires a callable factory")
    return _DEMDecoderFactory(factory)
